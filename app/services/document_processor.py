"""
Main document processing orchestration service.

This module coordinates all document classification components
and manages the complete processing pipeline.
"""
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import traceback

from app.models.document import (
    Document,
    Section,
    Metadata,
    QAPair,
    ContentElement,
    ProcessingResult,
    ProcessingStatus,
    DocumentType,
    ContentType,
)
from app.models.text_element import TextElement
from app.services.pdf_extractor import PDFExtractor, PDFExtractorError
from app.ml.pattern_learner import PatternLearner
from app.ml.section_detector import SectionDetector
from app.ml.metadata_extractor import MetadataExtractor
from app.ml.qa_detector import QADetector
from app.ml.content_classifier import ContentClassifier


logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DocumentProcessingError(Exception):
    """Base exception for document processing."""
    pass


class PDFValidationError(DocumentProcessingError):
    """PDF file validation failed."""
    pass


class ProcessingCancelledException(DocumentProcessingError):
    """Processing was cancelled."""
    pass


class ResourceLimitExceeded(DocumentProcessingError):
    """Resource limit exceeded (memory, time, etc.)."""
    pass


# ============================================================================
# STORAGE AND CACHE SERVICES (Interfaces)
# ============================================================================

class StorageService:
    """
    Storage service interface for database operations.
    
    In production, this would connect to PostgreSQL, MongoDB, etc.
    """
    
    async def save_document(self, document: Document) -> None:
        """Save document to database."""
        logger.info(f"Saving document: {document.id}")
        # Implementation would save to database
        pass
    
    async def save_processing_result(self, result: ProcessingResult) -> None:
        """Save complete processing result."""
        logger.info(f"Saving processing result for: {result.document.id}")
        # Implementation would save all components
        pass
    
    async def update_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update document processing status."""
        logger.info(f"Updating status for {document_id}: {status}")
        # Implementation would update database
        pass
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        # Implementation would query database
        return None


class CacheService:
    """
    Cache service interface for Redis or similar.
    
    Caches intermediate and final results for performance.
    """
    
    def __init__(self):
        self.cache = {}  # In-memory cache for demo
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set value in cache with TTL."""
        self.cache[key] = value
        logger.debug(f"Cached: {key} (TTL: {ttl_seconds}s)")
    
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.cache


# ============================================================================
# MAIN DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """
    Central document processing orchestrator.
    
    Coordinates all classification components and manages
    the complete processing pipeline from PDF to structured data.
    """
    
    # Processing configuration
    MAX_CONCURRENT_BATCH = 5
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2
    CACHE_EXTRACTION_TTL = 3600  # 1 hour
    CACHE_RESULT_TTL = 86400  # 24 hours
    MAX_MEMORY_MB = 2048  # 2GB per document
    
    def __init__(
        self,
        pdf_extractor: Optional[PDFExtractor] = None,
        pattern_learner: Optional[PatternLearner] = None,
        section_detector: Optional[SectionDetector] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
        qa_detector: Optional[QADetector] = None,
        content_classifier: Optional[ContentClassifier] = None,
        storage_service: Optional[StorageService] = None,
        cache_service: Optional[CacheService] = None,
    ):
        """
        Initialize document processor with dependencies.
        
        Args:
            pdf_extractor: PDF extraction service
            pattern_learner: Pattern learning service
            section_detector: Section detection service
            metadata_extractor: Metadata extraction service
            qa_detector: Q&A detection service
            content_classifier: Content classification service
            storage_service: Database storage service
            cache_service: Caching service
        """
        # Initialize services (lazy loading supported)
        self.pdf_extractor = pdf_extractor
        self.pattern_learner = pattern_learner
        self.section_detector = section_detector
        self.metadata_extractor = metadata_extractor
        self.qa_detector = qa_detector
        self.content_classifier = content_classifier
        
        self.storage_service = storage_service or StorageService()
        self.cache_service = cache_service or CacheService()
        
        # Processing state
        self.active_processing = {}  # document_id -> task
        self.cancelled_documents = set()
        
        logger.info("DocumentProcessor initialized")
    
    async def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a document through complete pipeline.
        
        Args:
            file_path: Path to PDF file
            document_id: Optional document ID (generated if not provided)
            user_id: Optional user ID
            
        Returns:
            ProcessingResult with all extracted information
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        file_path = Path(file_path)
        
        # Generate document ID if not provided
        if not document_id:
            document_id = f"doc_{int(time.time())}_{file_path.stem}"
        
        logger.info(f"Starting processing: {document_id} ({file_path})")
        
        # Initialize document
        document = Document(
            id=document_id,
            filename=file_path.name,
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            upload_date=datetime.now(),
            processing_status=ProcessingStatus.PENDING,
            created_by=user_id
        )
        
        start_time = time.time()
        
        try:
            # ========== PHASE 1: Initialization ==========
            logger.info(f"[{document_id}] Phase 1: Initialization")
            await self._update_status(document_id, ProcessingStatus.PROCESSING)
            
            # Check cache
            cache_key = f"processed:{document_id}"
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                logger.info(f"[{document_id}] Found cached result")
                return cached_result
            
            # Validate PDF
            self.validate_pdf(str(file_path))
            
            # Load models if needed
            await self._ensure_models_loaded()
            
            # ========== PHASE 2: PDF Extraction ==========
            logger.info(f"[{document_id}] Phase 2: PDF Extraction")
            extraction_result = await self._extract_with_retry(file_path, document_id)
            
            elements = extraction_result.elements
            tables = extraction_result.tables
            document.total_pages = extraction_result.total_pages
            
            logger.info(
                f"[{document_id}] Extracted {len(elements)} elements, "
                f"{len(tables)} tables from {document.total_pages} pages"
            )
            
            # Cache extraction
            extraction_cache_key = f"extraction:{document_id}"
            await self.cache_service.set(
                extraction_cache_key,
                extraction_result,
                self.CACHE_EXTRACTION_TTL
            )
            
            # ========== PHASE 3: Preprocessing ==========
            logger.info(f"[{document_id}] Phase 3: Preprocessing")
            elements = await self._preprocess_elements(elements)
            doc_stats = self._calculate_document_stats(elements)
            
            # ========== PHASE 4: Parallel Classification ==========
            logger.info(f"[{document_id}] Phase 4: Parallel Classification")
            
            # Check for cancellation
            if document_id in self.cancelled_documents:
                raise ProcessingCancelledException(f"Processing cancelled: {document_id}")
            
            # Run all classifiers in parallel
            (
                pattern_results,
                sections,
                metadata_list,
                qa_pairs,
                content_classifications
            ) = await asyncio.gather(
                self._classify_patterns(elements),
                self._detect_sections(elements),
                self._extract_metadata(elements, tables),
                self._detect_qa_pairs(elements),
                self._classify_content(elements),
                return_exceptions=True
            )
            
            # Handle exceptions from parallel tasks
            results = [
                pattern_results, sections, metadata_list, qa_pairs, content_classifications
            ]
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Classification task {i} failed: {result}")
                    # Use empty results for failed tasks
                    results[i] = [] if i != 0 else {}
            
            pattern_results, sections, metadata_list, qa_pairs, content_classifications = results
            
            logger.info(
                f"[{document_id}] Classification complete: "
                f"{len(sections)} sections, {len(metadata_list)} metadata, "
                f"{len(qa_pairs)} Q&A pairs"
            )
            
            # ========== PHASE 5: Integration ==========
            logger.info(f"[{document_id}] Phase 5: Integration")
            
            # Build section hierarchy
            sections = self._build_section_hierarchy(sections)
            
            # Assign elements to sections
            self._assign_elements_to_sections(sections, elements)
            
            # Link Q&A to sections
            self._link_qa_to_sections(qa_pairs, sections)
            
            # Associate metadata with sections
            self._associate_metadata_with_sections(metadata_list, sections)
            
            # ========== PHASE 6: Validation ==========
            logger.info(f"[{document_id}] Phase 6: Validation")
            warnings = await self._validate_results(
                elements, sections, metadata_list, qa_pairs
            )
            
            # ========== PHASE 7: Aggregation ==========
            logger.info(f"[{document_id}] Phase 7: Aggregation")
            
            # Update document with final stats
            document.total_sections = len(sections)
            document.total_qa_pairs = len(qa_pairs)
            document.processing_status = ProcessingStatus.COMPLETED
            document.processed_date = datetime.now()
            document.processing_time_seconds = time.time() - start_time
            document.confidence_score = self._calculate_overall_confidence(
                sections, metadata_list, qa_pairs
            )
            
            # Build content elements
            content_elements = self._build_content_elements(
                elements, content_classifications, document_id
            )
            
            # Create processing result
            result = ProcessingResult(
                document=document,
                sections=[self._convert_to_section_model(s, document_id) for s in sections],
                metadata=[self._convert_to_metadata_model(m, document_id) for m in metadata_list],
                qa_pairs=[self._convert_to_qa_model(q, document_id) for q in qa_pairs],
                content_elements=content_elements,
                warnings=warnings,
                processing_stats={
                    'total_time_seconds': document.processing_time_seconds,
                    'extraction_time': doc_stats.get('extraction_time', 0),
                    'classification_time': doc_stats.get('classification_time', 0),
                    'total_elements': len(elements),
                    'total_characters': doc_stats.get('total_chars', 0),
                    'avg_confidence': document.confidence_score,
                }
            )
            
            # ========== PHASE 8: Storage ==========
            logger.info(f"[{document_id}] Phase 8: Storage")
            
            await self.storage_service.save_processing_result(result)
            
            # Cache result
            await self.cache_service.set(cache_key, result, self.CACHE_RESULT_TTL)
            
            # Update status
            await self._update_status(document_id, ProcessingStatus.COMPLETED)
            
            logger.info(
                f"[{document_id}] Processing complete in "
                f"{document.processing_time_seconds:.2f}s"
            )
            
            return result
            
        except ProcessingCancelledException:
            logger.warning(f"[{document_id}] Processing cancelled")
            await self._update_status(document_id, ProcessingStatus.CANCELLED)
            raise
            
        except Exception as e:
            # ========== PHASE 9: Error Handling ==========
            logger.error(f"[{document_id}] Processing failed: {e}")
            logger.error(traceback.format_exc())
            
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            document.processing_time_seconds = time.time() - start_time
            
            await self._update_status(document_id, ProcessingStatus.FAILED, str(e))
            
            # Try to save partial results
            try:
                await self.storage_service.save_document(document)
            except Exception as save_error:
                logger.error(f"Failed to save error state: {save_error}")
            
            raise DocumentProcessingError(f"Processing failed: {e}") from e
        
        finally:
            # Cleanup
            self.active_processing.pop(document_id, None)
            self.cancelled_documents.discard(document_id)
    
    async def process_batch(
        self,
        file_paths: List[str],
        user_id: Optional[str] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in parallel.
        
        Args:
            file_paths: List of PDF file paths
            user_id: Optional user ID
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Starting batch processing: {len(file_paths)} documents")
        
        # Process in chunks to limit concurrency
        results = []
        
        for i in range(0, len(file_paths), self.MAX_CONCURRENT_BATCH):
            batch = file_paths[i:i + self.MAX_CONCURRENT_BATCH]
            
            logger.info(
                f"Processing batch {i // self.MAX_CONCURRENT_BATCH + 1}: "
                f"{len(batch)} documents"
            )
            
            # Process batch
            batch_results = await asyncio.gather(
                *[
                    self.process_document(path, user_id=user_id)
                    for path in batch
                ],
                return_exceptions=True
            )
            
            # Handle exceptions
            for path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {path}: {result}")
                else:
                    results.append(result)
        
        logger.info(
            f"Batch processing complete: {len(results)}/{len(file_paths)} successful"
        )
        
        return results
    
    def validate_pdf(self, file_path: str) -> None:
        """
        Validate PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Raises:
            PDFValidationError: If validation fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise PDFValidationError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise PDFValidationError(f"Not a file: {file_path}")
        
        if path.suffix.lower() != '.pdf':
            raise PDFValidationError(f"Not a PDF file: {file_path}")
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 100:  # 100 MB limit
            raise PDFValidationError(f"File too large: {size_mb:.1f} MB")
        
        # Try to open with PDF library
        try:
            import PyPDF2
            with open(path, 'rb') as f:
                PyPDF2.PdfReader(f)
        except Exception as e:
            raise PDFValidationError(f"Invalid PDF file: {e}")
    
    def estimate_processing_time(
        self,
        file_size: int,
        page_count: Optional[int] = None
    ) -> float:
        """
        Estimate processing time in seconds.
        
        Args:
            file_size: File size in bytes
            page_count: Number of pages (if known)
            
        Returns:
            Estimated time in seconds
        """
        # Rough estimation: 2 seconds per page or 1 second per MB
        if page_count:
            return page_count * 2.0
        else:
            size_mb = file_size / (1024 * 1024)
            return size_mb * 1.0
    
    async def cancel_processing(self, document_id: str) -> bool:
        """
        Cancel ongoing processing.
        
        Args:
            document_id: Document ID to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        if document_id in self.active_processing:
            self.cancelled_documents.add(document_id)
            logger.info(f"Cancelling processing: {document_id}")
            return True
        return False
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    async def _ensure_models_loaded(self) -> None:
        """Ensure all models are loaded (lazy loading)."""
        if self.pdf_extractor is None:
            self.pdf_extractor = PDFExtractor()
        
        if self.section_detector is None:
            self.section_detector = SectionDetector()
        
        if self.metadata_extractor is None:
            self.metadata_extractor = MetadataExtractor()
        
        if self.qa_detector is None:
            self.qa_detector = QADetector()
        
        if self.content_classifier is None:
            self.content_classifier = ContentClassifier()
        
        # Pattern learner needs training data, so we leave it optional
    
    async def _extract_with_retry(
        self,
        file_path: Path,
        document_id: str
    ) -> Any:
        """Extract PDF with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.pdf_extractor.extract_from_file(str(file_path))
                return result
            except PDFExtractorError as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                logger.warning(
                    f"[{document_id}] Extraction attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {self.RETRY_DELAY_SECONDS}s..."
                )
                await asyncio.sleep(self.RETRY_DELAY_SECONDS * (attempt + 1))
    
    async def _preprocess_elements(
        self,
        elements: List[TextElement]
    ) -> List[TextElement]:
        """Preprocess extracted elements."""
        # Clean text
        for elem in elements:
            # Normalize whitespace
            elem.text = ' '.join(elem.text.split())
            # Remove artifacts (would be more sophisticated in production)
            elem.text = elem.text.replace('\x00', '')
        
        return elements
    
    def _calculate_document_stats(
        self,
        elements: List[TextElement]
    ) -> Dict[str, Any]:
        """Calculate document-level statistics."""
        total_chars = sum(len(elem.text) for elem in elements)
        total_words = sum(len(elem.text.split()) for elem in elements)
        
        return {
            'total_chars': total_chars,
            'total_words': total_words,
            'total_elements': len(elements),
        }
    
    async def _classify_patterns(self, elements: List[TextElement]) -> Dict[str, Any]:
        """Classify patterns (if pattern learner available)."""
        if self.pattern_learner and hasattr(self.pattern_learner, 'predict_batch'):
            try:
                return self.pattern_learner.predict_batch(elements)
            except Exception as e:
                logger.warning(f"Pattern classification failed: {e}")
        return {}
    
    async def _detect_sections(self, elements: List[TextElement]) -> List[Any]:
        """Detect document sections."""
        if self.section_detector:
            return self.section_detector.detect_sections(elements)
        return []
    
    async def _extract_metadata(
        self,
        elements: List[TextElement],
        tables: List[Dict]
    ) -> List[Any]:
        """Extract metadata."""
        if self.metadata_extractor:
            return self.metadata_extractor.extract_metadata(elements, tables)
        return []
    
    async def _detect_qa_pairs(self, elements: List[TextElement]) -> List[Any]:
        """Detect Q&A pairs."""
        if self.qa_detector:
            return self.qa_detector.detect_qa_pairs(elements)
        return []
    
    async def _classify_content(self, elements: List[TextElement]) -> List[Any]:
        """Classify content types."""
        if self.content_classifier:
            return self.content_classifier.classify_batch(elements)
        return []
    
    def _build_section_hierarchy(self, sections: List[Any]) -> List[Any]:
        """Build hierarchical section structure."""
        # Already handled by section detector
        return sections
    
    def _assign_elements_to_sections(
        self,
        sections: List[Any],
        elements: List[TextElement]
    ) -> None:
        """Assign content elements to sections."""
        # Implementation would map elements to sections by position
        pass
    
    def _link_qa_to_sections(self, qa_pairs: List[Any], sections: List[Any]) -> None:
        """Link Q&A pairs to their parent sections."""
        # Implementation would match by page/position
        pass
    
    def _associate_metadata_with_sections(
        self,
        metadata_list: List[Any],
        sections: List[Any]
    ) -> None:
        """Associate metadata with relevant sections."""
        # Implementation would match by page/section
        pass
    
    async def _validate_results(
        self,
        elements: List[TextElement],
        sections: List[Any],
        metadata_list: List[Any],
        qa_pairs: List[Any]
    ) -> List[str]:
        """Validate processing results."""
        warnings = []
        
        # Check for empty results
        if not elements:
            warnings.append("No text elements extracted")
        
        if not sections:
            warnings.append("No sections detected")
        
        # Check confidence thresholds
        # (would be more sophisticated in production)
        
        return warnings
    
    def _calculate_overall_confidence(
        self,
        sections: List[Any],
        metadata_list: List[Any],
        qa_pairs: List[Any]
    ) -> float:
        """Calculate overall processing confidence."""
        confidences = []
        
        # Collect all confidence scores
        for section in sections:
            if hasattr(section, 'confidence'):
                confidences.append(section.confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _build_content_elements(
        self,
        elements: List[TextElement],
        classifications: List[Any],
        document_id: str
    ) -> List[ContentElement]:
        """Build ContentElement models from text elements."""
        content_elements = []
        
        for i, elem in enumerate(elements):
            # Get classification if available
            content_type = ContentType.UNKNOWN
            confidence = 0.5
            
            if i < len(classifications) and classifications[i]:
                classification = classifications[i]
                if hasattr(classification, 'primary_type'):
                    content_type = classification.primary_type
                    confidence = classification.primary_confidence
            
            content_elem = ContentElement(
                document_id=document_id,
                page_number=elem.page_number,
                content_type=content_type,
                text=elem.text,
                bbox=(elem.bbox.x0, elem.bbox.y0, elem.bbox.x1, elem.bbox.y1),
                font_size=elem.font_size,
                font_name=elem.font_name,
                is_bold=elem.is_bold,
                is_italic=elem.is_italic,
                confidence=confidence,
                line_number=elem.line_number,
                paragraph_id=elem.paragraph_id
            )
            
            content_elements.append(content_elem)
        
        return content_elements
    
    def _convert_to_section_model(self, section: Any, document_id: str) -> Section:
        """Convert internal section to Section model."""
        return Section(
            document_id=document_id,
            title=section.title if hasattr(section, 'title') else "Untitled",
            level=section.level if hasattr(section, 'level') else 1,
            content="",
            section_number=section.number if hasattr(section, 'number') else None,
            confidence_score=section.confidence if hasattr(section, 'confidence') else 0.5,
            page_numbers=[section.page_number] if hasattr(section, 'page_number') else [],
        )
    
    def _convert_to_metadata_model(self, metadata: Any, document_id: str) -> Metadata:
        """Convert internal metadata to Metadata model."""
        # Implementation would convert from metadata extractor format
        return Metadata(
            document_id=document_id,
            key="example",
            value="example_value",
            data_type="text"
        )
    
    def _convert_to_qa_model(self, qa: Any, document_id: str) -> QAPair:
        """Convert internal Q&A to QAPair model."""
        return QAPair(
            document_id=document_id,
            question_text=qa.question_text if hasattr(qa, 'question_text') else "",
            answer_text=qa.answer_text if hasattr(qa, 'answer_text') else None,
            is_answered=qa.is_answered if hasattr(qa, 'is_answered') else False,
            question_number=qa.question_number if hasattr(qa, 'question_number') else None,
            confidence_score=qa.confidence_score if hasattr(qa, 'confidence_score') else 0.5,
        )
    
    async def _update_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update processing status."""
        await self.storage_service.update_status(document_id, status, error_message)
    
    @asynccontextmanager
    async def _resource_manager(self, document_id: str):
        """Context manager for resource cleanup."""
        try:
            yield
        finally:
            # Cleanup resources
            logger.debug(f"Cleaning up resources for: {document_id}")


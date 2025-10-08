"""
Comprehensive data persistence service using SQLAlchemy and PostgreSQL.

This module provides database operations for the document classification system
with transaction management, connection pooling, and full-text search support.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, select, update, delete, func, or_, and_, text
from sqlalchemy.orm import sessionmaker, selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import (
    SQLAlchemyError,
    IntegrityError,
    OperationalError,
    TimeoutError as SQLTimeoutError
)
from sqlalchemy.pool import NullPool, QueuePool

from app.models.document import (
    Document,
    Section,
    Metadata,
    QAPair,
    ContentElement,
    ProcessingResult,
    FeedbackEntry,
    ProcessingStatus,
    DocumentType,
)
from app.models.db_models import (
    Base,
    DocumentModel,
    SectionModel,
    MetadataModel,
    QAPairModel,
    ContentElementModel,
    PatternModel,
    FeedbackModel,
    ProcessingLogModel,
)


logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class ConnectionError(StorageError):
    """Database connection error."""
    pass


class DocumentNotFoundError(StorageError):
    """Document not found in database."""
    pass


class TransactionError(StorageError):
    """Transaction commit/rollback error."""
    pass


# ============================================================================
# STORAGE SERVICE
# ============================================================================

class StorageService:
    """
    Comprehensive data persistence service.
    
    Manages all database operations with PostgreSQL using SQLAlchemy.
    Supports async operations, connection pooling, transactions, and full-text search.
    """
    
    # Connection pool configuration
    DEFAULT_POOL_SIZE = 20
    DEFAULT_MAX_OVERFLOW = 10
    DEFAULT_POOL_TIMEOUT = 30
    DEFAULT_POOL_RECYCLE = 3600  # 1 hour
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = DEFAULT_POOL_SIZE,
        max_overflow: int = DEFAULT_MAX_OVERFLOW,
        echo: bool = False,
        **engine_kwargs
    ):
        """
        Initialize storage service.
        
        Args:
            database_url: Database connection string
            pool_size: Connection pool size
            max_overflow: Max overflow connections
            echo: Echo SQL statements (debug mode)
            **engine_kwargs: Additional engine configuration
        """
        self.database_url = database_url
        
        # Configure engine
        engine_config = {
            'poolclass': QueuePool,
            'pool_size': pool_size,
            'max_overflow': max_overflow,
            'pool_timeout': self.DEFAULT_POOL_TIMEOUT,
            'pool_recycle': self.DEFAULT_POOL_RECYCLE,
            'pool_pre_ping': True,  # Test connections before using
            'echo': echo,
        }
        engine_config.update(engine_kwargs)
        
        # Create engines
        if database_url.startswith('postgresql+asyncpg'):
            # Async engine
            self.engine = create_async_engine(database_url, **engine_config)
            self.SessionLocal = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            self.is_async = True
        else:
            # Sync engine (fallback)
            self.engine = create_engine(database_url, **engine_config)
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
            self.is_async = False
        
        logger.info(f"StorageService initialized (async={self.is_async})")
    
    @asynccontextmanager
    async def get_session(self):
        """
        Get database session with automatic cleanup.
        
        Yields:
            AsyncSession or Session
        """
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            await session.rollback() if self.is_async else session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            await session.close() if self.is_async else session.close()
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    async def save_document(self, document: Document) -> str:
        """
        Save document to database.
        
        Args:
            document: Document object to save
            
        Returns:
            Document ID
            
        Raises:
            StorageError: If save operation fails
        """
        try:
            async with self.get_session() as session:
                # Create database model
                db_document = DocumentModel(
                    id=document.id,
                    filename=document.filename,
                    file_size=document.file_size,
                    upload_date=document.upload_date,
                    processed_date=document.processed_date,
                    document_type=document.document_type,
                    processing_status=document.processing_status,
                    confidence_score=document.confidence_score,
                    total_pages=document.total_pages,
                    total_sections=document.total_sections,
                    total_qa_pairs=document.total_qa_pairs,
                    raw_text=document.raw_text,
                    metadata_json=document.metadata,
                    processing_time_seconds=document.processing_time_seconds,
                    error_message=document.error_message,
                    created_by=document.created_by,
                )
                
                session.add(db_document)
                await session.commit()
                
                logger.info(f"Saved document: {document.id}")
                return document.id
                
        except IntegrityError as e:
            logger.error(f"Integrity error saving document: {e}")
            raise StorageError(f"Document already exists or constraint violation: {e}")
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise StorageError(f"Failed to save document: {e}")
    
    async def save_processing_result(self, result: ProcessingResult) -> None:
        """
        Save complete processing result in a transaction.
        
        Args:
            result: ProcessingResult with all components
            
        Raises:
            TransactionError: If transaction fails
        """
        try:
            async with self.get_session() as session:
                # Start transaction
                async with session.begin():
                    # 1. Save or update document
                    doc = result.document
                    db_document = await session.get(DocumentModel, doc.id)
                    
                    if db_document:
                        # Update existing
                        db_document.processed_date = doc.processed_date
                        db_document.processing_status = doc.processing_status
                        db_document.confidence_score = doc.confidence_score
                        db_document.total_pages = doc.total_pages
                        db_document.total_sections = len(result.sections)
                        db_document.total_qa_pairs = len(result.qa_pairs)
                        db_document.raw_text = doc.raw_text
                        db_document.metadata_json = doc.metadata
                        db_document.processing_time_seconds = doc.processing_time_seconds
                        db_document.updated_at = datetime.now()
                    else:
                        # Create new
                        db_document = DocumentModel(
                            id=doc.id,
                            filename=doc.filename,
                            file_size=doc.file_size,
                            upload_date=doc.upload_date,
                            processed_date=doc.processed_date,
                            document_type=doc.document_type,
                            processing_status=doc.processing_status,
                            confidence_score=doc.confidence_score,
                            total_pages=doc.total_pages,
                            total_sections=len(result.sections),
                            total_qa_pairs=len(result.qa_pairs),
                            raw_text=doc.raw_text,
                            metadata_json=doc.metadata,
                            processing_time_seconds=doc.processing_time_seconds,
                            error_message=doc.error_message,
                            created_by=doc.created_by,
                        )
                        session.add(db_document)
                    
                    # 2. Save sections
                    for section in result.sections:
                        db_section = SectionModel(
                            id=section.id,
                            document_id=section.document_id,
                            parent_id=section.parent_id,
                            title=section.title,
                            level=section.level,
                            content=section.content,
                            section_number=section.section_number,
                            order_index=section.order_index,
                            page_numbers=section.page_numbers,
                            start_position=section.start_position,
                            end_position=section.end_position,
                            confidence_score=section.confidence_score,
                            formatting_json=section.formatting,
                        )
                        session.add(db_section)
                    
                    # 3. Save metadata
                    for metadata in result.metadata:
                        db_metadata = MetadataModel(
                            id=metadata.id,
                            document_id=metadata.document_id,
                            section_id=metadata.section_id,
                            key=metadata.key,
                            value=metadata.value,
                            data_type=metadata.data_type,
                            normalized_value=str(metadata.normalized_value) if metadata.normalized_value else None,
                            extraction_method=metadata.extraction_method,
                            confidence=metadata.confidence,
                            page_number=metadata.page_number,
                        )
                        session.add(db_metadata)
                    
                    # 4. Save Q&A pairs
                    for qa in result.qa_pairs:
                        db_qa = QAPairModel(
                            id=qa.id,
                            document_id=qa.document_id,
                            section_id=qa.section_id,
                            parent_question_id=qa.parent_question_id,
                            question_number=qa.question_number,
                            question_text=qa.question_text,
                            answer_text=qa.answer_text,
                            question_type=qa.question_type,
                            is_answered=qa.is_answered,
                            is_required=qa.is_required,
                            confidence_score=qa.confidence_score,
                            answer_quality_score=qa.answer_quality_score,
                            page_numbers=qa.page_numbers,
                            indentation_level=qa.indentation_level,
                        )
                        session.add(db_qa)
                    
                    # 5. Save content elements (batch insert for performance)
                    if result.content_elements:
                        for element in result.content_elements:
                            db_element = ContentElementModel(
                                id=element.id,
                                document_id=element.document_id,
                                section_id=element.section_id,
                                page_number=element.page_number,
                                content_type=element.content_type,
                                text=element.text,
                                bbox=list(element.bbox),
                                font_size=element.font_size,
                                font_name=element.font_name,
                                is_bold=element.is_bold,
                                is_italic=element.is_italic,
                                confidence=element.confidence,
                                line_number=element.line_number,
                                paragraph_id=element.paragraph_id,
                            )
                            session.add(db_element)
                    
                    # 6. Update search vector for full-text search
                    if doc.raw_text:
                        await session.execute(
                            text("""
                                UPDATE documents 
                                SET search_vector = to_tsvector('english', :text)
                                WHERE id = :doc_id
                            """),
                            {"text": doc.raw_text[:100000], "doc_id": doc.id}  # Limit to 100k chars
                        )
                    
                    # Commit transaction
                    await session.commit()
                    
                logger.info(
                    f"Saved processing result for {doc.id}: "
                    f"{len(result.sections)} sections, {len(result.metadata)} metadata, "
                    f"{len(result.qa_pairs)} Q&A pairs, {len(result.content_elements)} elements"
                )
                
        except Exception as e:
            logger.error(f"Transaction error saving processing result: {e}")
            raise TransactionError(f"Failed to save processing result: {e}")
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document object or None if not found
        """
        try:
            async with self.get_session() as session:
                stmt = select(DocumentModel).where(
                    DocumentModel.id == document_id,
                    DocumentModel.is_deleted == False
                )
                result = await session.execute(stmt)
                db_document = result.scalar_one_or_none()
                
                if not db_document:
                    return None
                
                # Convert to Pydantic model
                document = Document(
                    id=db_document.id,
                    filename=db_document.filename,
                    file_size=db_document.file_size,
                    upload_date=db_document.upload_date,
                    processed_date=db_document.processed_date,
                    document_type=db_document.document_type,
                    processing_status=db_document.processing_status,
                    confidence_score=db_document.confidence_score,
                    total_pages=db_document.total_pages,
                    total_sections=db_document.total_sections,
                    total_qa_pairs=db_document.total_qa_pairs,
                    raw_text=db_document.raw_text,
                    metadata=db_document.metadata_json or {},
                    processing_time_seconds=db_document.processing_time_seconds,
                    error_message=db_document.error_message,
                    created_by=db_document.created_by,
                )
                
                return document
                
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            raise StorageError(f"Failed to retrieve document: {e}")
    
    async def get_document_with_content(self, document_id: str) -> Optional[ProcessingResult]:
        """
        Retrieve document with all related content.
        
        Args:
            document_id: Document ID
            
        Returns:
            ProcessingResult with all components or None
        """
        try:
            async with self.get_session() as session:
                # Load document with eager loading
                stmt = select(DocumentModel).where(
                    DocumentModel.id == document_id,
                    DocumentModel.is_deleted == False
                ).options(
                    selectinload(DocumentModel.sections),
                    selectinload(DocumentModel.metadata_entries),
                    selectinload(DocumentModel.qa_pairs),
                    selectinload(DocumentModel.content_elements),
                )
                
                result = await session.execute(stmt)
                db_document = result.scalar_one_or_none()
                
                if not db_document:
                    return None
                
                # Convert to Pydantic models
                document = self._convert_document_model(db_document)
                
                sections = [
                    self._convert_section_model(s)
                    for s in await session.execute(
                        select(SectionModel).where(SectionModel.document_id == document_id)
                    ).scalars()
                ]
                
                metadata_list = [
                    self._convert_metadata_model(m)
                    for m in await session.execute(
                        select(MetadataModel).where(MetadataModel.document_id == document_id)
                    ).scalars()
                ]
                
                qa_pairs = [
                    self._convert_qa_model(q)
                    for q in await session.execute(
                        select(QAPairModel).where(QAPairModel.document_id == document_id)
                    ).scalars()
                ]
                
                content_elements = [
                    self._convert_content_element_model(e)
                    for e in await session.execute(
                        select(ContentElementModel).where(ContentElementModel.document_id == document_id)
                    ).scalars()
                ]
                
                processing_result = ProcessingResult(
                    document=document,
                    sections=sections,
                    metadata=metadata_list,
                    qa_pairs=qa_pairs,
                    content_elements=content_elements,
                    warnings=[],
                    processing_stats={}
                )
                
                return processing_result
                
        except Exception as e:
            logger.error(f"Error retrieving document with content {document_id}: {e}")
            raise StorageError(f"Failed to retrieve document with content: {e}")
    
    async def get_sections(
        self,
        document_id: str,
        parent_id: Optional[str] = None
    ) -> List[Section]:
        """
        Get sections for document.
        
        Args:
            document_id: Document ID
            parent_id: Optional parent section ID for filtering
            
        Returns:
            List of Section objects
        """
        try:
            async with self.get_session() as session:
                stmt = select(SectionModel).where(
                    SectionModel.document_id == document_id
                )
                
                if parent_id is not None:
                    stmt = stmt.where(SectionModel.parent_id == parent_id)
                
                stmt = stmt.order_by(SectionModel.order_index)
                
                result = await session.execute(stmt)
                db_sections = result.scalars().all()
                
                sections = [self._convert_section_model(s) for s in db_sections]
                return sections
                
        except Exception as e:
            logger.error(f"Error retrieving sections for {document_id}: {e}")
            raise StorageError(f"Failed to retrieve sections: {e}")
    
    async def get_metadata(self, document_id: str) -> List[Metadata]:
        """
        Get all metadata for document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of Metadata objects
        """
        try:
            async with self.get_session() as session:
                stmt = select(MetadataModel).where(
                    MetadataModel.document_id == document_id
                ).order_by(MetadataModel.key)
                
                result = await session.execute(stmt)
                db_metadata = result.scalars().all()
                
                metadata_list = [self._convert_metadata_model(m) for m in db_metadata]
                return metadata_list
                
        except Exception as e:
            logger.error(f"Error retrieving metadata for {document_id}: {e}")
            raise StorageError(f"Failed to retrieve metadata: {e}")
    
    async def get_qa_pairs(
        self,
        document_id: str,
        include_unanswered: bool = True
    ) -> List[QAPair]:
        """
        Get Q&A pairs for document.
        
        Args:
            document_id: Document ID
            include_unanswered: Include unanswered questions
            
        Returns:
            List of QAPair objects
        """
        try:
            async with self.get_session() as session:
                stmt = select(QAPairModel).where(
                    QAPairModel.document_id == document_id
                )
                
                if not include_unanswered:
                    stmt = stmt.where(QAPairModel.is_answered == True)
                
                result = await session.execute(stmt)
                db_qa_pairs = result.scalars().all()
                
                qa_pairs = [self._convert_qa_model(q) for q in db_qa_pairs]
                return qa_pairs
                
        except Exception as e:
            logger.error(f"Error retrieving Q&A pairs for {document_id}: {e}")
            raise StorageError(f"Failed to retrieve Q&A pairs: {e}")
    
    async def update_document_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update document processing status.
        
        Args:
            document_id: Document ID
            status: New processing status
            error_message: Optional error message
        """
        try:
            async with self.get_session() as session:
                stmt = update(DocumentModel).where(
                    DocumentModel.id == document_id
                ).values(
                    processing_status=status,
                    error_message=error_message,
                    processed_date=datetime.now() if status == ProcessingStatus.COMPLETED else None,
                    updated_at=datetime.now()
                )
                
                await session.execute(stmt)
                await session.commit()
                
                logger.info(f"Updated status for {document_id}: {status}")
                
        except Exception as e:
            logger.error(f"Error updating status for {document_id}: {e}")
            raise StorageError(f"Failed to update status: {e}")
    
    # ========================================================================
    # FEEDBACK OPERATIONS
    # ========================================================================
    
    async def save_feedback(self, feedback: FeedbackEntry) -> None:
        """
        Save user feedback/corrections.
        
        Args:
            feedback: FeedbackEntry object
        """
        try:
            async with self.get_session() as session:
                db_feedback = FeedbackModel(
                    id=feedback.id,
                    document_id=feedback.document_id,
                    element_id=feedback.element_id,
                    element_type=feedback.element_type,
                    original_value_json={"value": feedback.original_value},
                    corrected_value_json={"value": feedback.corrected_value},
                    feedback_type=feedback.feedback_type,
                    user_id=feedback.user_id,
                    notes=feedback.notes,
                    is_processed=False,
                    timestamp=feedback.timestamp,
                )
                
                session.add(db_feedback)
                await session.commit()
                
                logger.info(f"Saved feedback: {feedback.id}")
                
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            raise StorageError(f"Failed to save feedback: {e}")
    
    async def get_pending_feedback(self, limit: int = 100) -> List[FeedbackEntry]:
        """
        Get unprocessed feedback for retraining.
        
        Args:
            limit: Maximum number of entries to retrieve
            
        Returns:
            List of FeedbackEntry objects
        """
        try:
            async with self.get_session() as session:
                stmt = select(FeedbackModel).where(
                    FeedbackModel.is_processed == False
                ).order_by(
                    FeedbackModel.timestamp
                ).limit(limit)
                
                result = await session.execute(stmt)
                db_feedback = result.scalars().all()
                
                feedback_list = []
                for fb in db_feedback:
                    feedback_entry = FeedbackEntry(
                        id=fb.id,
                        document_id=fb.document_id,
                        element_id=fb.element_id,
                        element_type=fb.element_type,
                        original_value=fb.original_value_json.get("value"),
                        corrected_value=fb.corrected_value_json.get("value"),
                        feedback_type=fb.feedback_type,
                        user_id=fb.user_id,
                        timestamp=fb.timestamp,
                        notes=fb.notes,
                    )
                    feedback_list.append(feedback_entry)
                
                return feedback_list
                
        except Exception as e:
            logger.error(f"Error retrieving pending feedback: {e}")
            raise StorageError(f"Failed to retrieve feedback: {e}")
    
    # ========================================================================
    # SEARCH AND LIST OPERATIONS
    # ========================================================================
    
    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[Document]:
        """
        Full-text search using PostgreSQL ts_vector.
        
        Args:
            query: Search query
            filters: Optional filters (document_type, status, etc.)
            limit: Maximum results
            
        Returns:
            List of Document objects ranked by relevance
        """
        try:
            async with self.get_session() as session:
                # Build search query
                stmt = select(DocumentModel).where(
                    DocumentModel.is_deleted == False
                )
                
                # Full-text search
                if query:
                    stmt = stmt.where(
                        DocumentModel.search_vector.op('@@')(
                            func.plainto_tsquery('english', query)
                        )
                    ).order_by(
                        func.ts_rank(
                            DocumentModel.search_vector,
                            func.plainto_tsquery('english', query)
                        ).desc()
                    )
                
                # Apply filters
                if filters:
                    if 'document_type' in filters:
                        stmt = stmt.where(DocumentModel.document_type == filters['document_type'])
                    if 'status' in filters:
                        stmt = stmt.where(DocumentModel.processing_status == filters['status'])
                    if 'date_from' in filters:
                        stmt = stmt.where(DocumentModel.upload_date >= filters['date_from'])
                    if 'date_to' in filters:
                        stmt = stmt.where(DocumentModel.upload_date <= filters['date_to'])
                
                stmt = stmt.limit(limit)
                
                result = await session.execute(stmt)
                db_documents = result.scalars().all()
                
                documents = [self._convert_document_model(d) for d in db_documents]
                return documents
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise StorageError(f"Failed to search documents: {e}")
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], int]:
        """
        Paginated document listing.
        
        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            filters: Optional filters
            
        Returns:
            Tuple of (documents list, total count)
        """
        try:
            async with self.get_session() as session:
                # Build query
                stmt = select(DocumentModel).where(
                    DocumentModel.is_deleted == False
                )
                
                # Apply filters
                if filters:
                    if 'status' in filters:
                        stmt = stmt.where(DocumentModel.processing_status == filters['status'])
                    if 'document_type' in filters:
                        stmt = stmt.where(DocumentModel.document_type == filters['document_type'])
                    if 'created_by' in filters:
                        stmt = stmt.where(DocumentModel.created_by == filters['created_by'])
                
                # Get total count
                count_stmt = select(func.count()).select_from(stmt.alias())
                total_count = await session.scalar(count_stmt)
                
                # Get paginated results
                stmt = stmt.order_by(DocumentModel.upload_date.desc()).offset(skip).limit(limit)
                result = await session.execute(stmt)
                db_documents = result.scalars().all()
                
                documents = [self._convert_document_model(d) for d in db_documents]
                return documents, total_count
                
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise StorageError(f"Failed to list documents: {e}")
    
    async def delete_document(
        self,
        document_id: str,
        soft_delete: bool = True
    ) -> None:
        """
        Delete document (soft or hard delete).
        
        Args:
            document_id: Document ID
            soft_delete: If True, mark as deleted; if False, physically delete
        """
        try:
            async with self.get_session() as session:
                if soft_delete:
                    # Soft delete
                    stmt = update(DocumentModel).where(
                        DocumentModel.id == document_id
                    ).values(
                        is_deleted=True,
                        deleted_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    await session.execute(stmt)
                else:
                    # Hard delete (cascades to related tables)
                    stmt = delete(DocumentModel).where(
                        DocumentModel.id == document_id
                    )
                    await session.execute(stmt)
                
                await session.commit()
                logger.info(f"Deleted document {document_id} (soft={soft_delete})")
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise StorageError(f"Failed to delete document: {e}")
    
    # ========================================================================
    # PATTERN OPERATIONS
    # ========================================================================
    
    async def save_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Save learned pattern to database.
        
        Args:
            pattern: Pattern dictionary with name, type, features, etc.
        """
        try:
            async with self.get_session() as session:
                # Check if pattern exists
                stmt = select(PatternModel).where(
                    PatternModel.pattern_name == pattern['name']
                )
                result = await session.execute(stmt)
                db_pattern = result.scalar_one_or_none()
                
                if db_pattern:
                    # Update existing
                    db_pattern.feature_vector = pattern.get('features', [])
                    db_pattern.examples_json = pattern.get('examples', {})
                    db_pattern.accuracy_score = pattern.get('accuracy', 0.0)
                    db_pattern.usage_count += 1
                    db_pattern.version += 1
                    db_pattern.updated_at = datetime.now()
                else:
                    # Create new
                    db_pattern = PatternModel(
                        pattern_name=pattern['name'],
                        pattern_type=pattern.get('type', 'unknown'),
                        feature_vector=pattern.get('features', []),
                        examples_json=pattern.get('examples', {}),
                        accuracy_score=pattern.get('accuracy', 0.0),
                        usage_count=1,
                    )
                    session.add(db_pattern)
                
                await session.commit()
                logger.info(f"Saved pattern: {pattern['name']}")
                
        except Exception as e:
            logger.error(f"Error saving pattern: {e}")
            raise StorageError(f"Failed to save pattern: {e}")
    
    async def get_patterns(
        self,
        pattern_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve learned patterns.
        
        Args:
            pattern_type: Optional pattern type filter
            
        Returns:
            List of pattern dictionaries
        """
        try:
            async with self.get_session() as session:
                stmt = select(PatternModel)
                
                if pattern_type:
                    stmt = stmt.where(PatternModel.pattern_type == pattern_type)
                
                stmt = stmt.order_by(
                    PatternModel.accuracy_score.desc(),
                    PatternModel.usage_count.desc()
                )
                
                result = await session.execute(stmt)
                db_patterns = result.scalars().all()
                
                patterns = []
                for p in db_patterns:
                    patterns.append({
                        'id': p.id,
                        'name': p.pattern_name,
                        'type': p.pattern_type,
                        'features': p.feature_vector,
                        'examples': p.examples_json,
                        'usage_count': p.usage_count,
                        'accuracy': p.accuracy_score,
                        'version': p.version,
                    })
                
                return patterns
                
        except Exception as e:
            logger.error(f"Error retrieving patterns: {e}")
            raise StorageError(f"Failed to retrieve patterns: {e}")
    
    # ========================================================================
    # LOGGING OPERATIONS
    # ========================================================================
    
    async def log_processing(
        self,
        document_id: str,
        stage: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log processing stage to audit trail.
        
        Args:
            document_id: Document ID
            stage: Processing stage name
            status: Stage status
            details: Optional details dictionary
        """
        try:
            async with self.get_session() as session:
                log_entry = ProcessingLogModel(
                    document_id=document_id,
                    processing_stage=stage,
                    status=status,
                    start_time=datetime.now(),
                    details_json=details or {},
                )
                
                session.add(log_entry)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error logging processing stage: {e}")
            # Don't raise exception for logging errors
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health.
        
        Returns:
            Health status dictionary
        """
        try:
            async with self.get_session() as session:
                # Simple query to test connection
                result = await session.execute(text("SELECT 1"))
                result.scalar()
                
                # Get pool status if available
                pool_status = {}
                if hasattr(self.engine.pool, 'size'):
                    pool_status = {
                        'pool_size': self.engine.pool.size(),
                        'checked_in': self.engine.pool.checkedin(),
                        'checked_out': self.engine.pool.checkedout(),
                        'overflow': self.engine.pool.overflow(),
                    }
                
                return {
                    'status': 'healthy',
                    'database': 'connected',
                    'pool': pool_status,
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        try:
            async with self.get_session() as session:
                # Document counts
                total_docs = await session.scalar(
                    select(func.count()).select_from(DocumentModel).where(
                        DocumentModel.is_deleted == False
                    )
                )
                
                completed_docs = await session.scalar(
                    select(func.count()).select_from(DocumentModel).where(
                        DocumentModel.processing_status == ProcessingStatus.COMPLETED,
                        DocumentModel.is_deleted == False
                    )
                )
                
                # Section, metadata, Q&A counts
                total_sections = await session.scalar(select(func.count()).select_from(SectionModel))
                total_metadata = await session.scalar(select(func.count()).select_from(MetadataModel))
                total_qa = await session.scalar(select(func.count()).select_from(QAPairModel))
                
                return {
                    'total_documents': total_docs,
                    'completed_documents': completed_docs,
                    'total_sections': total_sections,
                    'total_metadata': total_metadata,
                    'total_qa_pairs': total_qa,
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    # ========================================================================
    # CONVERSION HELPERS
    # ========================================================================
    
    def _convert_document_model(self, db_doc: DocumentModel) -> Document:
        """Convert DocumentModel to Document Pydantic model."""
        return Document(
            id=db_doc.id,
            filename=db_doc.filename,
            file_size=db_doc.file_size,
            upload_date=db_doc.upload_date,
            processed_date=db_doc.processed_date,
            document_type=db_doc.document_type,
            processing_status=db_doc.processing_status,
            confidence_score=db_doc.confidence_score,
            total_pages=db_doc.total_pages,
            total_sections=db_doc.total_sections,
            total_qa_pairs=db_doc.total_qa_pairs,
            raw_text=db_doc.raw_text,
            metadata=db_doc.metadata_json or {},
            processing_time_seconds=db_doc.processing_time_seconds,
            error_message=db_doc.error_message,
            created_by=db_doc.created_by,
        )
    
    def _convert_section_model(self, db_section: SectionModel) -> Section:
        """Convert SectionModel to Section Pydantic model."""
        return Section(
            id=db_section.id,
            document_id=db_section.document_id,
            parent_id=db_section.parent_id,
            title=db_section.title,
            level=db_section.level,
            content=db_section.content or "",
            section_number=db_section.section_number,
            order_index=db_section.order_index,
            page_numbers=db_section.page_numbers or [],
            start_position=db_section.start_position,
            end_position=db_section.end_position,
            confidence_score=db_section.confidence_score,
            formatting=db_section.formatting_json or {},
        )
    
    def _convert_metadata_model(self, db_metadata: MetadataModel) -> Metadata:
        """Convert MetadataModel to Metadata Pydantic model."""
        return Metadata(
            id=db_metadata.id,
            document_id=db_metadata.document_id,
            key=db_metadata.key,
            value=db_metadata.value,
            data_type=db_metadata.data_type,
            normalized_value=db_metadata.normalized_value,
            extraction_method=db_metadata.extraction_method,
            confidence=db_metadata.confidence,
            page_number=db_metadata.page_number,
            section_id=db_metadata.section_id,
        )
    
    def _convert_qa_model(self, db_qa: QAPairModel) -> QAPair:
        """Convert QAPairModel to QAPair Pydantic model."""
        return QAPair(
            id=db_qa.id,
            document_id=db_qa.document_id,
            section_id=db_qa.section_id,
            question_number=db_qa.question_number,
            question_text=db_qa.question_text,
            answer_text=db_qa.answer_text,
            question_type=db_qa.question_type,
            is_answered=db_qa.is_answered,
            is_required=db_qa.is_required,
            confidence_score=db_qa.confidence_score,
            answer_quality_score=db_qa.answer_quality_score,
            page_numbers=db_qa.page_numbers or [],
            parent_question_id=db_qa.parent_question_id,
            indentation_level=db_qa.indentation_level,
        )
    
    def _convert_content_element_model(self, db_element: ContentElementModel) -> ContentElement:
        """Convert ContentElementModel to ContentElement Pydantic model."""
        return ContentElement(
            id=db_element.id,
            document_id=db_element.document_id,
            section_id=db_element.section_id,
            page_number=db_element.page_number,
            content_type=db_element.content_type,
            text=db_element.text,
            bbox=tuple(db_element.bbox) if db_element.bbox else (0, 0, 0, 0),
            font_size=db_element.font_size,
            font_name=db_element.font_name,
            is_bold=db_element.is_bold,
            is_italic=db_element.is_italic,
            confidence=db_element.confidence,
            line_number=db_element.line_number,
            paragraph_id=db_element.paragraph_id,
        )
    
    async def close(self):
        """Close database connections."""
        if self.is_async:
            await self.engine.dispose()
        else:
            self.engine.dispose()
        logger.info("StorageService connections closed")


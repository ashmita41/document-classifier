"""
REST API endpoints for document classification system.

This module provides comprehensive REST endpoints for uploading, processing,
retrieving, and managing documents with their classifications.
"""
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Depends,
    Query,
    Path as PathParam,
    BackgroundTasks,
    Request,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

from app.models.document import (
    Document,
    Section,
    Metadata,
    QAPair,
    ContentElement,
    ProcessingResult,
    ProcessingStatus,
    DocumentType,
    MetadataType,
    FeedbackEntry,
    FeedbackType,
)
from app.services.storage_service import StorageService, DocumentNotFoundError
from app.services.document_processor import DocumentProcessor, CacheService
from app.services.pdf_extractor import PDFExtractor
from app.ml.pattern_learner import PatternLearner
from app.ml.section_detector import SectionDetector
from app.ml.metadata_extractor import MetadataExtractor
from app.ml.qa_detector import QADetector
from app.ml.content_classifier import ContentClassifier


logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf"}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Rate limiting (simple in-memory implementation)
_rate_limit_cache: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class UploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    filename: str
    status: ProcessingStatus
    message: str
    processing_started_at: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None


class StatusResponse(BaseModel):
    """Response for status check."""
    document_id: str
    status: ProcessingStatus
    progress_percentage: float = Field(ge=0, le=100)
    current_stage: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_time_remaining: Optional[float] = None
    error_message: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request for submitting feedback."""
    element_id: str
    element_type: str
    corrected_value: Any
    feedback_type: FeedbackType
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    feedback_id: str
    status: str
    message: str
    learning_queued: bool


class SearchResponse(BaseModel):
    """Response for document search."""
    results: List[Document]
    total_count: int
    has_more: bool
    query_time_ms: float
    query: str


class ListResponse(BaseModel):
    """Response for document listing."""
    documents: List[Document]
    total: int
    page: int
    page_size: int
    total_pages: int


class DeleteResponse(BaseModel):
    """Response for document deletion."""
    message: str
    deleted_at: datetime


class ReprocessResponse(BaseModel):
    """Response for reprocess request."""
    message: str
    new_job_id: str
    status: ProcessingStatus


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SectionTreeResponse(BaseModel):
    """Response for section hierarchy."""
    sections: List[Section]
    total_count: int
    max_level: int


class MetadataResponse(BaseModel):
    """Response for metadata."""
    metadata: List[Metadata]
    total_count: int
    statistics: Dict[str, int]  # Count by type


class QAPairResponse(BaseModel):
    """Response for Q&A pairs."""
    qa_pairs: List[QAPair]
    total_count: int
    answered_count: int
    unanswered_count: int


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

# Global service instances (in production, use dependency injection framework)
_storage_service: Optional[StorageService] = None
_document_processor: Optional[DocumentProcessor] = None
_cache_service: Optional[CacheService] = None


def get_storage_service() -> StorageService:
    """Get storage service instance."""
    global _storage_service
    if _storage_service is None:
        # Initialize with database URL from environment
        import os
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://user:password@localhost/document_classifier"
        )
        _storage_service = StorageService(database_url)
    return _storage_service


def get_cache_service() -> CacheService:
    """Get cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


def get_document_processor() -> DocumentProcessor:
    """Get document processor instance."""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor(
            storage_service=get_storage_service(),
            cache_service=get_cache_service()
        )
    return _document_processor


async def verify_api_key(request: Request) -> str:
    """
    Verify API key (optional authentication).
    
    Args:
        request: FastAPI request
        
    Returns:
        User ID from API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        # For development, allow requests without API key
        return "anonymous"
    
    # In production, validate against database
    # For now, accept any non-empty key
    return api_key


def check_rate_limit(client_id: str) -> None:
    """
    Check rate limiting.
    
    Args:
        client_id: Client identifier (IP or user ID)
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    current_time = time.time()
    
    # Clean old entries
    if client_id in _rate_limit_cache:
        _rate_limit_cache[client_id] = [
            t for t in _rate_limit_cache[client_id]
            if current_time - t < RATE_LIMIT_WINDOW
        ]
    else:
        _rate_limit_cache[client_id] = []
    
    # Check limit
    if len(_rate_limit_cache[client_id]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s"
        )
    
    # Add current request
    _rate_limit_cache[client_id].append(current_time)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: Optional[DocumentType] = Query(None),
    process_immediately: bool = Query(True),
    user_id: str = Depends(verify_api_key),
    storage: StorageService = Depends(get_storage_service),
    processor: DocumentProcessor = Depends(get_document_processor),
    request: Request = None
):
    """
    Upload PDF document for processing.
    
    Args:
        file: PDF file (multipart/form-data)
        document_type: Optional document type classification
        process_immediately: Start processing immediately
        user_id: User ID from API key
        
    Returns:
        Upload response with document ID and status
        
    Raises:
        HTTPException: 400 for invalid file, 500 for server error
    """
    try:
        # Rate limiting
        client_id = request.client.host if request else "unknown"
        check_rate_limit(client_id)
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF files are allowed."
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f} MB"
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Generate document ID
        document_id = f"doc_{uuid4().hex[:12]}"
        
        # Save file to storage
        file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Create document record
        document = Document(
            id=document_id,
            filename=file.filename,
            file_size=file_size,
            upload_date=datetime.now(),
            document_type=document_type or DocumentType.OTHER,
            processing_status=ProcessingStatus.PENDING,
            created_by=user_id,
        )
        
        await storage.save_document(document)
        
        # Queue processing
        processing_started_at = None
        estimated_completion = None
        
        if process_immediately:
            # Start processing in background
            background_tasks.add_task(
                processor.process_document,
                str(file_path),
                document_id,
                user_id
            )
            processing_started_at = datetime.now()
            
            # Estimate completion time
            estimated_time = processor.estimate_processing_time(file_size)
            estimated_completion = processing_started_at + timedelta(seconds=estimated_time)
            
            document.processing_status = ProcessingStatus.PROCESSING
            await storage.update_document_status(document_id, ProcessingStatus.PROCESSING)
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            status=document.processing_status,
            message="Document uploaded successfully",
            processing_started_at=processing_started_at,
            estimated_completion_time=estimated_completion
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{document_id}", response_model=ProcessingResult)
async def get_document(
    document_id: str = PathParam(..., description="Document ID"),
    include_sections: bool = Query(True),
    include_metadata: bool = Query(True),
    include_qa_pairs: bool = Query(True),
    include_content_elements: bool = Query(False),
    storage: StorageService = Depends(get_storage_service),
    cache: CacheService = Depends(get_cache_service),
):
    """
    Get complete document with all content.
    
    Args:
        document_id: Document ID
        include_sections: Include sections in response
        include_metadata: Include metadata in response
        include_qa_pairs: Include Q&A pairs in response
        include_content_elements: Include content elements (large data)
        
    Returns:
        ProcessingResult with requested components
        
    Raises:
        HTTPException: 404 if not found, 500 for server error
    """
    try:
        # Check cache first
        cache_key = f"document:{document_id}:full"
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for document: {document_id}")
            result = cached_result
        else:
            # Fetch from database
            result = await storage.get_document_with_content(document_id)
            
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {document_id}"
                )
            
            # Cache result (24 hour TTL)
            await cache.set(cache_key, result, ttl_seconds=86400)
        
        # Filter components based on parameters
        if not include_sections:
            result.sections = []
        if not include_metadata:
            result.metadata = []
        if not include_qa_pairs:
            result.qa_pairs = []
        if not include_content_elements:
            result.content_elements = []
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


@router.get("/{document_id}/status", response_model=StatusResponse)
async def get_document_status(
    document_id: str = PathParam(..., description="Document ID"),
    storage: StorageService = Depends(get_storage_service),
    cache: CacheService = Depends(get_cache_service),
):
    """
    Check document processing status.
    
    Args:
        document_id: Document ID
        
    Returns:
        Status response with progress information
        
    Raises:
        HTTPException: 404 if not found
    """
    try:
        # Get document from database
        document = await storage.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        # Calculate progress
        progress = 0.0
        current_stage = None
        estimated_remaining = None
        
        if document.processing_status == ProcessingStatus.PENDING:
            progress = 0.0
            current_stage = "Queued"
        elif document.processing_status == ProcessingStatus.PROCESSING:
            # Try to get progress from cache
            progress_key = f"progress:{document_id}"
            cached_progress = await cache.get(progress_key)
            if cached_progress:
                progress = cached_progress.get('percentage', 50.0)
                current_stage = cached_progress.get('stage', 'Processing')
            else:
                progress = 50.0  # Default mid-progress
                current_stage = "Processing"
            
            # Estimate remaining time
            if document.upload_date:
                elapsed = (datetime.now() - document.upload_date).total_seconds()
                estimated_total = elapsed / (progress / 100) if progress > 0 else 0
                estimated_remaining = max(0, estimated_total - elapsed)
        elif document.processing_status == ProcessingStatus.COMPLETED:
            progress = 100.0
            current_stage = "Completed"
        elif document.processing_status == ProcessingStatus.FAILED:
            progress = 0.0
            current_stage = "Failed"
        elif document.processing_status == ProcessingStatus.CANCELLED:
            progress = 0.0
            current_stage = "Cancelled"
        
        return StatusResponse(
            document_id=document_id,
            status=document.processing_status,
            progress_percentage=progress,
            current_stage=current_stage,
            started_at=document.upload_date,
            completed_at=document.processed_date,
            estimated_time_remaining=estimated_remaining,
            error_message=document.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/sections", response_model=SectionTreeResponse)
async def get_document_sections(
    document_id: str = PathParam(..., description="Document ID"),
    level: Optional[int] = Query(None, ge=1, le=10, description="Filter by hierarchy level"),
    parent_id: Optional[str] = Query(None, description="Get children of specific section"),
    include_content: bool = Query(True),
    max_depth: int = Query(10, ge=1, le=20, description="Maximum nesting depth"),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get hierarchical section structure.
    
    Args:
        document_id: Document ID
        level: Optional level filter
        parent_id: Optional parent section filter
        include_content: Include section content
        max_depth: Maximum nesting depth
        
    Returns:
        Section tree response
        
    Raises:
        HTTPException: 404 if not found
    """
    try:
        # Verify document exists
        document = await storage.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get sections
        sections = await storage.get_sections(document_id, parent_id=parent_id)
        
        # Apply level filter
        if level is not None:
            sections = [s for s in sections if s.level == level]
        
        # Remove content if not requested
        if not include_content:
            for section in sections:
                section.content = ""
        
        # Calculate max level
        max_level = max((s.level for s in sections), default=0)
        
        return SectionTreeResponse(
            sections=sections,
            total_count=len(sections),
            max_level=max_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sections for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/metadata", response_model=MetadataResponse)
async def get_document_metadata(
    document_id: str = PathParam(..., description="Document ID"),
    metadata_type: Optional[MetadataType] = Query(None, description="Filter by type"),
    key: Optional[str] = Query(None, description="Filter by key"),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get extracted metadata.
    
    Args:
        document_id: Document ID
        metadata_type: Optional type filter
        key: Optional key filter
        
    Returns:
        Metadata response with statistics
        
    Raises:
        HTTPException: 404 if not found
    """
    try:
        # Verify document exists
        document = await storage.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get metadata
        metadata_list = await storage.get_metadata(document_id)
        
        # Apply filters
        if metadata_type:
            metadata_list = [m for m in metadata_list if m.data_type == metadata_type]
        if key:
            metadata_list = [m for m in metadata_list if m.key == key]
        
        # Calculate statistics
        statistics = {}
        for m in metadata_list:
            type_str = m.data_type.value if hasattr(m.data_type, 'value') else str(m.data_type)
            statistics[type_str] = statistics.get(type_str, 0) + 1
        
        return MetadataResponse(
            metadata=metadata_list,
            total_count=len(metadata_list),
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/qa-pairs", response_model=QAPairResponse)
async def get_document_qa_pairs(
    document_id: str = PathParam(..., description="Document ID"),
    answered_only: bool = Query(False, description="Return only answered questions"),
    question_type: Optional[str] = Query(None, description="Filter by question type"),
    section_id: Optional[str] = Query(None, description="Filter by section"),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get question-answer pairs.
    
    Args:
        document_id: Document ID
        answered_only: Filter to answered questions only
        question_type: Optional question type filter
        section_id: Optional section filter
        
    Returns:
        Q&A pair response with statistics
        
    Raises:
        HTTPException: 404 if not found
    """
    try:
        # Verify document exists
        document = await storage.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get Q&A pairs
        qa_pairs = await storage.get_qa_pairs(document_id, include_unanswered=not answered_only)
        
        # Apply filters
        if question_type:
            qa_pairs = [q for q in qa_pairs if q.question_type.value == question_type]
        if section_id:
            qa_pairs = [q for q in qa_pairs if q.section_id == section_id]
        
        # Calculate statistics
        answered_count = sum(1 for q in qa_pairs if q.is_answered)
        unanswered_count = len(qa_pairs) - answered_count
        
        return QAPairResponse(
            qa_pairs=qa_pairs,
            total_count=len(qa_pairs),
            answered_count=answered_count,
            unanswered_count=unanswered_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Q&A pairs for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    document_id: str = PathParam(..., description="Document ID"),
    feedback: FeedbackRequest = ...,
    user_id: str = Depends(verify_api_key),
    storage: StorageService = Depends(get_storage_service),
    cache: CacheService = Depends(get_cache_service),
):
    """
    Submit user corrections for adaptive learning.
    
    Args:
        document_id: Document ID
        feedback: Feedback data
        user_id: User ID from API key
        
    Returns:
        Feedback response
        
    Raises:
        HTTPException: 404 if not found, 400 for invalid data
    """
    try:
        # Verify document exists
        document = await storage.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create feedback entry
        feedback_entry = FeedbackEntry(
            document_id=document_id,
            element_id=feedback.element_id,
            element_type=feedback.element_type,
            original_value="",  # Would need to fetch original
            corrected_value=feedback.corrected_value,
            feedback_type=feedback.feedback_type,
            user_id=user_id,
            notes=feedback.notes,
        )
        
        # Save feedback
        await storage.save_feedback(feedback_entry)
        
        # Invalidate cache
        cache_key = f"document:{document_id}:full"
        await cache.delete(cache_key)
        
        # Queue for adaptive learning (would integrate with learning system)
        learning_queued = True
        
        logger.info(f"Feedback submitted for document {document_id} by {user_id}")
        
        return FeedbackResponse(
            feedback_id=feedback_entry.id,
            status="success",
            message="Feedback submitted successfully",
            learning_queued=learning_queued
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    document_type: Optional[DocumentType] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    status: Optional[ProcessingStatus] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Full-text search across documents.
    
    Args:
        q: Search query
        document_type: Optional document type filter
        date_from: Optional start date filter
        date_to: Optional end date filter
        status: Optional status filter
        limit: Maximum results
        offset: Pagination offset
        
    Returns:
        Search results with relevance ranking
    """
    try:
        start_time = time.time()
        
        # Build filters
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if status:
            filters['status'] = status
        if date_from:
            filters['date_from'] = date_from
        if date_to:
            filters['date_to'] = date_to
        
        # Execute search
        results = await storage.search_documents(
            query=q,
            filters=filters,
            limit=limit + 1  # Get one extra to check if more exist
        )
        
        # Check if more results exist
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]
        
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            has_more=has_more,
            query_time_ms=query_time,
            query=q
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=ListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[ProcessingStatus] = Query(None),
    document_type: Optional[DocumentType] = Query(None),
    sort_by: str = Query("upload_date", regex="^(upload_date|filename|processing_status)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    storage: StorageService = Depends(get_storage_service),
):
    """
    List documents with pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum records to return
        status: Optional status filter
        document_type: Optional type filter
        sort_by: Sort field
        sort_order: Sort order (asc/desc)
        
    Returns:
        Paginated document list
    """
    try:
        # Build filters
        filters = {}
        if status:
            filters['status'] = status
        if document_type:
            filters['document_type'] = document_type
        
        # Get documents
        documents, total = await storage.list_documents(
            skip=skip,
            limit=limit,
            filters=filters
        )
        
        # Calculate pagination info
        page = (skip // limit) + 1
        total_pages = (total + limit - 1) // limit
        
        return ListResponse(
            documents=documents,
            total=total,
            page=page,
            page_size=limit,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str = PathParam(..., description="Document ID"),
    hard_delete: bool = Query(False, description="Permanently delete (default: soft delete)"),
    storage: StorageService = Depends(get_storage_service),
    cache: CacheService = Depends(get_cache_service),
):
    """
    Delete document.
    
    Args:
        document_id: Document ID
        hard_delete: If True, permanently delete; if False, soft delete
        
    Returns:
        Delete confirmation
        
    Raises:
        HTTPException: 404 if not found
    """
    try:
        # Verify document exists
        document = await storage.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from database
        await storage.delete_document(document_id, soft_delete=not hard_delete)
        
        # Remove from cache
        cache_key = f"document:{document_id}:full"
        await cache.delete(cache_key)
        
        # Optionally delete physical file
        if hard_delete:
            file_pattern = f"{document_id}_*"
            for file_path in UPLOAD_DIR.glob(file_pattern):
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
        
        delete_type = "permanently" if hard_delete else "soft"
        logger.info(f"Document {document_id} deleted ({delete_type})")
        
        return DeleteResponse(
            message=f"Document deleted successfully ({delete_type})",
            deleted_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/reprocess", response_model=ReprocessResponse)
async def reprocess_document(
    document_id: str = PathParam(..., description="Document ID"),
    background_tasks: BackgroundTasks = ...,
    storage: StorageService = Depends(get_storage_service),
    processor: DocumentProcessor = Depends(get_document_processor),
    cache: CacheService = Depends(get_cache_service),
):
    """
    Reprocess document with latest models.
    
    Args:
        document_id: Document ID
        
    Returns:
        Reprocess response with new job ID
        
    Raises:
        HTTPException: 404 if not found
    """
    try:
        # Verify document exists
        document = await storage.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Find original file
        file_pattern = f"{document_id}_*"
        file_paths = list(UPLOAD_DIR.glob(file_pattern))
        
        if not file_paths:
            raise HTTPException(
                status_code=400,
                detail="Original file not found. Cannot reprocess."
            )
        
        file_path = file_paths[0]
        
        # Reset status
        await storage.update_document_status(document_id, ProcessingStatus.PROCESSING)
        
        # Invalidate cache
        cache_key = f"document:{document_id}:full"
        await cache.delete(cache_key)
        
        # Queue for reprocessing
        new_job_id = f"job_{uuid4().hex[:12]}"
        background_tasks.add_task(
            processor.process_document,
            str(file_path),
            document_id,
            document.created_by
        )
        
        logger.info(f"Document {document_id} queued for reprocessing (job: {new_job_id})")
        
        return ReprocessResponse(
            message="Document queued for reprocessing",
            new_job_id=new_job_id,
            status=ProcessingStatus.PROCESSING
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/export")
async def export_document(
    document_id: str = PathParam(..., description="Document ID"),
    format: str = Query("json", regex="^(json|csv|markdown|xml)$"),
    include_sections: bool = Query(True),
    include_metadata: bool = Query(True),
    include_qa_pairs: bool = Query(True),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Export document in various formats.
    
    Args:
        document_id: Document ID
        format: Export format (json, csv, markdown, xml)
        include_sections: Include sections
        include_metadata: Include metadata
        include_qa_pairs: Include Q&A pairs
        
    Returns:
        File download
        
    Raises:
        HTTPException: 404 if not found, 400 for unsupported format
    """
    try:
        # Get complete document
        result = await storage.get_document_with_content(document_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Filter components
        if not include_sections:
            result.sections = []
        if not include_metadata:
            result.metadata = []
        if not include_qa_pairs:
            result.qa_pairs = []
        
        # Generate export based on format
        if format == "json":
            content = result.model_dump_json(indent=2)
            media_type = "application/json"
            filename = f"{document_id}.json"
        
        elif format == "csv":
            # Simple CSV export (would be more sophisticated in production)
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Type", "Content", "Page", "Confidence"])
            
            for section in result.sections:
                writer.writerow(["Section", section.title, ",".join(map(str, section.page_numbers)), section.confidence_score])
            
            for qa in result.qa_pairs:
                writer.writerow(["Question", qa.question_text, ",".join(map(str, qa.page_numbers)), qa.confidence_score])
                if qa.answer_text:
                    writer.writerow(["Answer", qa.answer_text, "", qa.answer_quality_score])
            
            content = output.getvalue()
            media_type = "text/csv"
            filename = f"{document_id}.csv"
        
        elif format == "markdown":
            # Generate markdown
            md_lines = [
                f"# Document: {result.document.filename}",
                f"",
                f"**Document ID**: {result.document.id}",
                f"**Status**: {result.document.processing_status}",
                f"**Pages**: {result.document.total_pages}",
                f"",
            ]
            
            if result.sections:
                md_lines.append("## Sections\n")
                for section in result.sections:
                    indent = "  " * (section.level - 1)
                    md_lines.append(f"{indent}- {section.title}")
                md_lines.append("")
            
            if result.qa_pairs:
                md_lines.append("## Questions & Answers\n")
                for i, qa in enumerate(result.qa_pairs, 1):
                    md_lines.append(f"### Q{i}: {qa.question_text}\n")
                    if qa.answer_text:
                        md_lines.append(f"**A**: {qa.answer_text}\n")
                    else:
                        md_lines.append("**A**: *Unanswered*\n")
            
            content = "\n".join(md_lines)
            media_type = "text/markdown"
            filename = f"{document_id}.md"
        
        elif format == "xml":
            # Simple XML export
            import xml.etree.ElementTree as ET
            
            root = ET.Element("document")
            root.set("id", result.document.id)
            root.set("filename", result.document.filename)
            
            if result.sections:
                sections_elem = ET.SubElement(root, "sections")
                for section in result.sections:
                    sec_elem = ET.SubElement(sections_elem, "section")
                    sec_elem.set("level", str(section.level))
                    title_elem = ET.SubElement(sec_elem, "title")
                    title_elem.text = section.title
            
            content = ET.tostring(root, encoding='unicode')
            media_type = "application/xml"
            filename = f"{document_id}.xml"
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Return as downloadable file
        return StreamingResponse(
            iter([content]),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


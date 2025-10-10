"""
Services module for document classifier.
"""
from app.services.pdf_extractor import PDFExtractor, PDFExtractorError, PDFCorruptedError
from app.services.document_processor import (
    DocumentProcessor,
    DocumentProcessingError,
    PDFValidationError,
    ProcessingCancelledException,
    ResourceLimitExceeded,
    CacheService,
)
from app.services.storage_service import (
    StorageService,
    StorageError,
    ConnectionError,
    DocumentNotFoundError,
    TransactionError,
)
from app.services.mongodb_storage import (
    MongoDBStorage,
    MongoStorageError,
)

__all__ = [
    # PDF Extractor
    "PDFExtractor",
    "PDFExtractorError", 
    "PDFCorruptedError",
    # Document Processor
    "DocumentProcessor",
    "DocumentProcessingError",
    "PDFValidationError",
    "ProcessingCancelledException",
    "ResourceLimitExceeded",
    "CacheService",
    # Storage Service (PostgreSQL)
    "StorageService",
    "StorageError",
    "ConnectionError",
    "DocumentNotFoundError",
    "TransactionError",
    # MongoDB Storage
    "MongoDBStorage",
    "MongoStorageError",
]


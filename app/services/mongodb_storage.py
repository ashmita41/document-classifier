"""
MongoDB storage service for document classification system.

This module provides MongoDB Atlas integration as an alternative to PostgreSQL.
Uses Motor (async MongoDB driver) for non-blocking operations.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, PyMongoError

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


logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class MongoStorageError(Exception):
    """Base exception for MongoDB storage operations."""
    pass


class DocumentNotFoundError(MongoStorageError):
    """Document not found in database."""
    pass


# ============================================================================
# MONGODB STORAGE SERVICE
# ============================================================================

class MongoDBStorage:
    """
    MongoDB storage service for document classification.
    
    Uses MongoDB Atlas with Motor (async driver) for high-performance
    document storage and retrieval.
    """
    
    def __init__(self, mongodb_uri: str, database_name: str = "document_classifier"):
        """
        Initialize MongoDB storage.
        
        Args:
            mongodb_uri: MongoDB connection string (Atlas URI)
            database_name: Database name
        """
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(mongodb_uri)
        self.db: AsyncIOMotorDatabase = self.client[database_name]
        
        # Collections
        self.documents = self.db.documents
        self.sections = self.db.sections
        self.metadata = self.db.metadata
        self.qa_pairs = self.db.qa_pairs
        self.content_elements = self.db.content_elements
        self.patterns = self.db.patterns
        self.feedback = self.db.feedback
        self.processing_logs = self.db.processing_logs
        
        logger.info(f"MongoDB storage initialized: {database_name}")
    
    async def initialize_indexes(self):
        """Create indexes for optimal query performance."""
        try:
            # Documents collection indexes
            await self.documents.create_index([("document_id", ASCENDING)], unique=True)
            await self.documents.create_index([("processing_status", ASCENDING)])
            await self.documents.create_index([("document_type", ASCENDING)])
            await self.documents.create_index([("upload_date", DESCENDING)])
            await self.documents.create_index([("created_by", ASCENDING)])
            await self.documents.create_index([("is_deleted", ASCENDING)])
            
            # Full-text search index
            await self.documents.create_index([("raw_text", TEXT), ("filename", TEXT)])
            
            # Sections collection indexes
            await self.sections.create_index([("document_id", ASCENDING)])
            await self.sections.create_index([("parent_id", ASCENDING)])
            await self.sections.create_index([("document_id", ASCENDING), ("order_index", ASCENDING)])
            
            # Metadata collection indexes
            await self.metadata.create_index([("document_id", ASCENDING)])
            await self.metadata.create_index([("key", ASCENDING)])
            await self.metadata.create_index([("data_type", ASCENDING)])
            
            # Q&A pairs collection indexes
            await self.qa_pairs.create_index([("document_id", ASCENDING)])
            await self.qa_pairs.create_index([("is_answered", ASCENDING)])
            
            # Content elements collection indexes
            await self.content_elements.create_index([("document_id", ASCENDING)])
            await self.content_elements.create_index([("page_number", ASCENDING)])
            
            # Feedback collection indexes
            await self.feedback.create_index([("document_id", ASCENDING)])
            await self.feedback.create_index([("is_processed", ASCENDING)])
            await self.feedback.create_index([("timestamp", DESCENDING)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    async def save_document(self, document: Document) -> str:
        """
        Save document to MongoDB.
        
        Args:
            document: Document object
            
        Returns:
            Document ID
        """
        try:
            doc_dict = document.model_dump()
            doc_dict["_id"] = document.id
            doc_dict["document_id"] = document.id
            doc_dict["is_deleted"] = False
            doc_dict["created_at"] = datetime.now()
            doc_dict["updated_at"] = datetime.now()
            
            await self.documents.insert_one(doc_dict)
            
            logger.info(f"Saved document to MongoDB: {document.id}")
            return document.id
            
        except DuplicateKeyError:
            # Update existing document
            await self.documents.update_one(
                {"_id": document.id},
                {"$set": {**doc_dict, "updated_at": datetime.now()}}
            )
            return document.id
            
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise MongoStorageError(f"Failed to save document: {e}")
    
    async def save_processing_result(self, result: ProcessingResult) -> None:
        """
        Save complete processing result in MongoDB.
        
        Args:
            result: ProcessingResult with all components
        """
        try:
            # Save document
            await self.save_document(result.document)
            
            # Save sections
            if result.sections:
                section_docs = [
                    {
                        **section.model_dump(),
                        "_id": section.id,
                        "created_at": datetime.now()
                    }
                    for section in result.sections
                ]
                await self.sections.insert_many(section_docs)
            
            # Save metadata
            if result.metadata:
                metadata_docs = [
                    {
                        **meta.model_dump(),
                        "_id": meta.id,
                        "created_at": datetime.now()
                    }
                    for meta in result.metadata
                ]
                await self.metadata.insert_many(metadata_docs)
            
            # Save Q&A pairs
            if result.qa_pairs:
                qa_docs = [
                    {
                        **qa.model_dump(),
                        "_id": qa.id,
                        "created_at": datetime.now()
                    }
                    for qa in result.qa_pairs
                ]
                await self.qa_pairs.insert_many(qa_docs)
            
            # Save content elements
            if result.content_elements:
                element_docs = [
                    {
                        **elem.model_dump(),
                        "_id": elem.id,
                        "created_at": datetime.now()
                    }
                    for elem in result.content_elements
                ]
                await self.content_elements.insert_many(element_docs)
            
            logger.info(
                f"Saved processing result for {result.document.id}: "
                f"{len(result.sections)} sections, {len(result.metadata)} metadata, "
                f"{len(result.qa_pairs)} Q&A pairs"
            )
            
        except Exception as e:
            logger.error(f"Error saving processing result: {e}")
            raise MongoStorageError(f"Failed to save processing result: {e}")
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document object or None
        """
        try:
            doc = await self.documents.find_one({
                "document_id": document_id,
                "is_deleted": False
            })
            
            if not doc:
                return None
            
            # Remove MongoDB _id for Pydantic
            doc.pop("_id", None)
            doc.pop("created_at", None)
            doc.pop("updated_at", None)
            
            return Document(**doc)
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            raise MongoStorageError(f"Failed to retrieve document: {e}")
    
    async def get_document_with_content(self, document_id: str) -> Optional[ProcessingResult]:
        """
        Retrieve document with all related content.
        
        Args:
            document_id: Document ID
            
        Returns:
            ProcessingResult with all components
        """
        try:
            # Get document
            document = await self.get_document(document_id)
            if not document:
                return None
            
            # Get sections
            section_docs = await self.sections.find({"document_id": document_id}).to_list(None)
            sections = [Section(**{k: v for k, v in doc.items() if k != "_id"}) for doc in section_docs]
            
            # Get metadata
            metadata_docs = await self.metadata.find({"document_id": document_id}).to_list(None)
            metadata_list = [Metadata(**{k: v for k, v in doc.items() if k != "_id"}) for doc in metadata_docs]
            
            # Get Q&A pairs
            qa_docs = await self.qa_pairs.find({"document_id": document_id}).to_list(None)
            qa_pairs = [QAPair(**{k: v for k, v in doc.items() if k != "_id"}) for doc in qa_docs]
            
            # Get content elements
            element_docs = await self.content_elements.find({"document_id": document_id}).to_list(None)
            content_elements = [ContentElement(**{k: v for k, v in doc.items() if k != "_id"}) for doc in element_docs]
            
            return ProcessingResult(
                document=document,
                sections=sections,
                metadata=metadata_list,
                qa_pairs=qa_pairs,
                content_elements=content_elements,
                warnings=[],
                processing_stats={}
            )
            
        except Exception as e:
            logger.error(f"Error retrieving document with content: {e}")
            raise MongoStorageError(f"Failed to retrieve document with content: {e}")
    
    async def get_sections(self, document_id: str, parent_id: Optional[str] = None) -> List[Section]:
        """Get sections for document."""
        try:
            query = {"document_id": document_id}
            if parent_id is not None:
                query["parent_id"] = parent_id
            
            section_docs = await self.sections.find(query).sort("order_index", ASCENDING).to_list(None)
            return [Section(**{k: v for k, v in doc.items() if k != "_id"}) for doc in section_docs]
            
        except Exception as e:
            logger.error(f"Error retrieving sections: {e}")
            raise MongoStorageError(f"Failed to retrieve sections: {e}")
    
    async def get_metadata(self, document_id: str) -> List[Metadata]:
        """Get metadata for document."""
        try:
            metadata_docs = await self.metadata.find({"document_id": document_id}).sort("key", ASCENDING).to_list(None)
            return [Metadata(**{k: v for k, v in doc.items() if k != "_id"}) for doc in metadata_docs]
            
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
            raise MongoStorageError(f"Failed to retrieve metadata: {e}")
    
    async def get_qa_pairs(self, document_id: str, include_unanswered: bool = True) -> List[QAPair]:
        """Get Q&A pairs for document."""
        try:
            query = {"document_id": document_id}
            if not include_unanswered:
                query["is_answered"] = True
            
            qa_docs = await self.qa_pairs.find(query).to_list(None)
            return [QAPair(**{k: v for k, v in doc.items() if k != "_id"}) for doc in qa_docs]
            
        except Exception as e:
            logger.error(f"Error retrieving Q&A pairs: {e}")
            raise MongoStorageError(f"Failed to retrieve Q&A pairs: {e}")
    
    async def update_document_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update document processing status."""
        try:
            update_dict = {
                "processing_status": status.value,
                "updated_at": datetime.now()
            }
            
            if error_message:
                update_dict["error_message"] = error_message
            
            if status == ProcessingStatus.COMPLETED:
                update_dict["processed_date"] = datetime.now()
            
            await self.documents.update_one(
                {"document_id": document_id},
                {"$set": update_dict}
            )
            
            logger.info(f"Updated status for {document_id}: {status}")
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            raise MongoStorageError(f"Failed to update status: {e}")
    
    async def update_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Alias for update_document_status for compatibility."""
        return await self.update_document_status(document_id, status, error_message)
    
    # ========================================================================
    # SEARCH AND LIST OPERATIONS
    # ========================================================================
    
    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[Document]:
        """Full-text search using MongoDB text index."""
        try:
            search_query = {
                "$text": {"$search": query},
                "is_deleted": False
            }
            
            # Apply filters
            if filters:
                if "document_type" in filters:
                    search_query["document_type"] = filters["document_type"].value
                if "status" in filters:
                    search_query["processing_status"] = filters["status"].value
                if "date_from" in filters:
                    search_query["upload_date"] = {"$gte": filters["date_from"]}
                if "date_to" in filters:
                    search_query.setdefault("upload_date", {})["$lte"] = filters["date_to"]
            
            # Execute search with text score
            cursor = self.documents.find(
                search_query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            docs = await cursor.to_list(None)
            
            return [Document(**{k: v for k, v in doc.items() if k not in ["_id", "score", "created_at", "updated_at"]}) for doc in docs]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise MongoStorageError(f"Failed to search documents: {e}")
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], int]:
        """List documents with pagination."""
        try:
            query = {"is_deleted": False}
            
            # Apply filters
            if filters:
                if "status" in filters:
                    query["processing_status"] = filters["status"].value
                if "document_type" in filters:
                    query["document_type"] = filters["document_type"].value
                if "created_by" in filters:
                    query["created_by"] = filters["created_by"]
            
            # Get total count
            total = await self.documents.count_documents(query)
            
            # Get paginated results
            cursor = self.documents.find(query).sort("upload_date", DESCENDING).skip(skip).limit(limit)
            docs = await cursor.to_list(None)
            
            documents = [Document(**{k: v for k, v in doc.items() if k not in ["_id", "created_at", "updated_at"]}) for doc in docs]
            
            return documents, total
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise MongoStorageError(f"Failed to list documents: {e}")
    
    async def delete_document(self, document_id: str, soft_delete: bool = True) -> None:
        """Delete document (soft or hard)."""
        try:
            if soft_delete:
                # Soft delete
                await self.documents.update_one(
                    {"document_id": document_id},
                    {"$set": {"is_deleted": True, "deleted_at": datetime.now()}}
                )
            else:
                # Hard delete - remove all related data
                await self.documents.delete_one({"document_id": document_id})
                await self.sections.delete_many({"document_id": document_id})
                await self.metadata.delete_many({"document_id": document_id})
                await self.qa_pairs.delete_many({"document_id": document_id})
                await self.content_elements.delete_many({"document_id": document_id})
            
            logger.info(f"Deleted document {document_id} (soft={soft_delete})")
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise MongoStorageError(f"Failed to delete document: {e}")
    
    # ========================================================================
    # FEEDBACK OPERATIONS
    # ========================================================================
    
    async def save_feedback(self, feedback: FeedbackEntry) -> None:
        """Save user feedback."""
        try:
            feedback_dict = feedback.model_dump()
            feedback_dict["_id"] = feedback.id
            feedback_dict["is_processed"] = False
            
            await self.feedback.insert_one(feedback_dict)
            
            logger.info(f"Saved feedback: {feedback.id}")
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            raise MongoStorageError(f"Failed to save feedback: {e}")
    
    async def get_pending_feedback(self, limit: int = 100) -> List[FeedbackEntry]:
        """Get unprocessed feedback."""
        try:
            cursor = self.feedback.find({"is_processed": False}).sort("timestamp", ASCENDING).limit(limit)
            docs = await cursor.to_list(None)
            
            return [FeedbackEntry(**{k: v for k, v in doc.items() if k != "_id"}) for doc in docs]
            
        except Exception as e:
            logger.error(f"Error retrieving feedback: {e}")
            raise MongoStorageError(f"Failed to retrieve feedback: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Ping database
            await self.db.command("ping")
            
            # Get stats
            stats = await self.db.command("dbStats")
            
            return {
                "status": "healthy",
                "database": "connected",
                "collections": stats.get("collections", 0),
                "data_size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            total_docs = await self.documents.count_documents({"is_deleted": False})
            completed_docs = await self.documents.count_documents({
                "processing_status": ProcessingStatus.COMPLETED.value,
                "is_deleted": False
            })
            total_sections = await self.sections.count_documents({})
            total_metadata = await self.metadata.count_documents({})
            total_qa = await self.qa_pairs.count_documents({})
            
            return {
                "total_documents": total_docs,
                "completed_documents": completed_docs,
                "total_sections": total_sections,
                "total_metadata": total_metadata,
                "total_qa_pairs": total_qa
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")


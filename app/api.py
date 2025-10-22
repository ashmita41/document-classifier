"""
Main application for LLM-based document parsing.

This is the refactored main application that uses GPT-4o-mini for intelligent
document structure extraction and produces structured JSON output.
"""
import logging
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
from dotenv import load_dotenv
import threading
from functools import lru_cache

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.llm_parser import FinalEnhancedDocumentParserPipeline
from app.universal_parser import UniversalDocumentParserPipeline

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Document Parser",
    description="Intelligent document parsing using GPT-4o-mini for structured JSON output",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances
pipeline: Optional[FinalEnhancedDocumentParserPipeline] = None
algorithmic_pipeline: Optional[UniversalDocumentParserPipeline] = None

# Intelligent caching layer
class DocumentCache:
    """Intelligent caching layer for processed documents."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.cache_lock = threading.RLock()
        self.max_size = max_size
        self.access_times = {}
        
    def get_cache_key(self, file_path: str, pipeline_type: str) -> str:
        """Generate cache key for file and pipeline type."""
        try:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                key_data = f"{file_path}:{pipeline_type}:{stat.st_size}:{stat.st_mtime}"
                return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            pass
        return hashlib.md5(f"{file_path}:{pipeline_type}".encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        with self.cache_lock:
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]
            return None
    
    def set(self, cache_key: str, result: Dict[str, Any]):
        """Set cached result with LRU eviction."""
        with self.cache_lock:
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
            
            self.cache[cache_key] = result
            self.access_times[cache_key] = time.time()
    
    def clear(self):
        """Clear cache."""
        with self.cache_lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'cache_keys': list(self.cache.keys())
            }

# Global cache instance
document_cache = DocumentCache(max_size=100)


@app.on_event("startup")
async def startup_event():
    """Initialize the document parser pipeline with optimization."""
    global pipeline, algorithmic_pipeline
    
    # Get configuration from environment
    api_key = os.getenv("OPENAI_API_KEY")
    max_workers = int(os.getenv("MAX_WORKERS", "4"))
    enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # Initialize pipelines with optimization
    if api_key:
        pipeline = FinalEnhancedDocumentParserPipeline(
            api_key, 
            max_workers=max_workers, 
            enable_caching=enable_caching
        )
        logger.info(f"Optimized LLM parser initialized with {max_workers} workers")
    else:
        pipeline = None
        logger.warning("OPENAI_API_KEY not found. LLM parser will not be available.")
    
    # Universal parser doesn't require API key but can be optimized
    algorithmic_pipeline = UniversalDocumentParserPipeline(
        max_workers=max_workers, 
        enable_caching=enable_caching
    )
    logger.info(f"Optimized universal parser initialized with {max_workers} workers")
    logger.info("Optimized document parser pipeline initialized")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Document Parser",
        "version": "2.0.0",
        "status": "operational",
        "description": "Intelligent document parsing using GPT-4o-mini",
        "endpoints": {
            "parse": "/parse",
            "parse_algorithmic": "/parse-algorithmic",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with performance metrics."""
    return {
        "status": "healthy",
        "llm_pipeline_available": pipeline is not None,
        "universal_pipeline_available": algorithmic_pipeline is not None,
        "openai_api_key_configured": os.getenv("OPENAI_API_KEY") is not None,
        "cache_stats": document_cache.get_stats()
    }

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return document_cache.get_stats()

@app.post("/cache/clear")
async def clear_cache():
    """Clear document cache."""
    document_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/performance")
async def get_performance_metrics():
    """Get performance metrics and optimization status."""
    return {
        "optimization_features": {
            "parallel_processing": True,
            "intelligent_caching": True,
            "compiled_regex_patterns": True,
            "batch_processing": True,
            "memory_optimization": True
        },
        "cache_stats": document_cache.get_stats(),
        "pipeline_config": {
            "max_workers": int(os.getenv("MAX_WORKERS", "4")),
            "caching_enabled": os.getenv("ENABLE_CACHING", "true").lower() == "true"
        }
    }


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    """
    Parse PDF document and return structured JSON with caching optimization.
    
    Args:
        file: PDF file to parse
        
    Returns:
        Structured JSON with sections, metadata, and questions
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="LLM parser not available. OpenAI API key required.")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are allowed."
        )
    
    # Save uploaded file temporarily
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Check cache first
        cache_key = document_cache.get_cache_key(str(file_path), "llm")
        cached_result = document_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for document: {file.filename}")
            # Clean up temporary file
            file_path.unlink()
            return JSONResponse(content=cached_result)
        
        # Parse document
        logger.info(f"Parsing document: {file.filename}")
        start_time = time.time()
        result = await pipeline.parse_document(str(file_path))
        processing_time = time.time() - start_time
        
        # Add API-level performance metrics
        result['api_metrics'] = {
            'processing_time_seconds': processing_time,
            'cached': False,
            'file_size_bytes': file_path.stat().st_size
        }
        
        # Cache result
        document_cache.set(cache_key, result)
        
        # Clean up temporary file
        file_path.unlink()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse-file")
async def parse_file_path(file_path: str):
    """
    Parse PDF document from file path.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Structured JSON with sections, metadata, and questions
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="LLM parser not available. OpenAI API key required.")
    
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="File is not a PDF")
    
    try:
        # Parse document
        logger.info(f"Parsing document: {file_path}")
        result = await pipeline.parse_document(file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@app.post("/parse-algorithmic")
async def parse_document_algorithmic(file: UploadFile = File(...)):
    """
    Parse PDF document using algorithmic approach with caching optimization.
    
    Args:
        file: PDF file to parse
        
    Returns:
        Structured JSON with sections, metadata, and content using algorithmic parsing
    """
    if not algorithmic_pipeline:
        raise HTTPException(status_code=500, detail="Algorithmic pipeline not initialized")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are allowed."
        )
    
    # Save uploaded file temporarily
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Check cache first
        cache_key = document_cache.get_cache_key(str(file_path), "algorithmic")
        cached_result = document_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for algorithmic document: {file.filename}")
            # Clean up temporary file
            file_path.unlink()
            return JSONResponse(content=cached_result)
        
        # Parse document using algorithmic approach
        logger.info(f"Parsing document algorithmically: {file.filename}")
        start_time = time.time()
        result = algorithmic_pipeline.process_document(str(file_path))
        processing_time = time.time() - start_time
        
        # Add API-level performance metrics
        result['api_metrics'] = {
            'processing_time_seconds': processing_time,
            'cached': False,
            'file_size_bytes': file_path.stat().st_size
        }
        
        # Cache result
        document_cache.set(cache_key, result)
        
        # Clean up temporary file
        file_path.unlink()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Algorithmic document parsing failed: {e}")
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Algorithmic parsing failed: {str(e)}")


@app.post("/parse-algorithmic-file")
async def parse_algorithmic_file_path(file_path: str):
    """
    Parse PDF document from file path using algorithmic approach.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Structured JSON with sections, metadata, and content using algorithmic parsing
    """
    if not algorithmic_pipeline:
        raise HTTPException(status_code=500, detail="Algorithmic pipeline not initialized")
    
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="File is not a PDF")
    
    try:
        # Parse document using algorithmic approach
        logger.info(f"Parsing document algorithmically: {file_path}")
        result = algorithmic_pipeline.process_document(file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Algorithmic document parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Algorithmic parsing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Set default API key for development
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Please set it in environment variables.")
    
    uvicorn.run(
        "app.main_new:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

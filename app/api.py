"""
Main application for LLM-based document parsing.

This is the refactored main application that uses GPT-4o-mini for intelligent
document structure extraction and produces structured JSON output.
"""
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
from dotenv import load_dotenv

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


@app.on_event("startup")
async def startup_event():
    """Initialize the document parser pipeline."""
    global pipeline, algorithmic_pipeline
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize pipelines
    if api_key:
        pipeline = FinalEnhancedDocumentParserPipeline(api_key)
        logger.info("LLM parser initialized with OpenAI API key")
    else:
        pipeline = None
        logger.warning("OPENAI_API_KEY not found. LLM parser will not be available.")
    
    # Universal parser doesn't require API key
    algorithmic_pipeline = UniversalDocumentParserPipeline()
    logger.info("Universal parser initialized")
    logger.info("Document parser pipeline initialized")


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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_pipeline_available": pipeline is not None,
        "universal_pipeline_available": algorithmic_pipeline is not None,
        "openai_api_key_configured": os.getenv("OPENAI_API_KEY") is not None
    }


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    """
    Parse PDF document and return structured JSON.
    
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
        # Parse document
        logger.info(f"Parsing document: {file.filename}")
        result = await pipeline.parse_document(str(file_path))
        
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
    Parse PDF document using algorithmic approach and return structured JSON.
    
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
        # Parse document using algorithmic approach
        logger.info(f"Parsing document algorithmically: {file.filename}")
        result = algorithmic_pipeline.process_document(str(file_path))
        
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

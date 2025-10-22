from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import time
from app.parser.extractor import extract_ordered_text
from app.parser.section_detector import detect_sections
from app.parser.question_detector import detect_questions
from app.parser.metadata_extractor import extract_metadata
from app.parser.categorizer import build_document_structure
from app.models import DocumentResponse

app = FastAPI(
    title="PDF Parser API",
    description="Intelligent PDF document parser for RFPs, resumes, and structured documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "PDF Parser API"}

@app.post("/parse", response_model=DocumentResponse)
async def parse_pdf(file: UploadFile = File(...)):
    """
    Parse uploaded PDF and return structured content
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Step 1: Extract text with layout
        lines = extract_ordered_text(tmp_path)
        
        if not lines:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Step 2: Detect sections
        section_result = detect_sections(lines)
        sections = section_result["sections"]
        
        # Step 3: Detect questions
        questions = detect_questions(lines, sections)
        
        # Step 4: Extract metadata
        metadata = extract_metadata(lines)
        
        # Step 5: Build document structure
        document = build_document_structure(lines, sections, questions, metadata)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        document['document_info']['processing_time_seconds'] = round(processing_time, 2)
        document['document_info']['file_name'] = file.filename
        document['document_info']['total_lines'] = len(lines)
        
        return JSONResponse(content=document)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from PDF with formatting information
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        lines = extract_ordered_text(tmp_path)
        return JSONResponse(content={
            "success": True,
            "lines": lines,
            "total_lines": len(lines),
            "file_name": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/detect-sections")
async def detect_sections_endpoint(file: UploadFile = File(...)):
    """
    Detect sections in PDF document
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        lines = extract_ordered_text(tmp_path)
        section_result = detect_sections(lines)
        return JSONResponse(content={
            "success": True,
            "sections": section_result["sections"],
            "excluded_headers": section_result["excluded_headers"],
            "debug_info": section_result["debug_info"],
            "total_sections": len(section_result["sections"]),
            "file_name": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting sections: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/classify-questions")
async def classify_questions_endpoint(file: UploadFile = File(...)):
    """
    Classify questions in PDF document
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        lines = extract_ordered_text(tmp_path)
        section_result = detect_sections(lines)
        questions = detect_questions(lines, section_result["sections"])
        return JSONResponse(content={
            "success": True,
            "questions": questions,
            "total_questions": len(questions),
            "file_name": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying questions: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/extract-metadata")
async def extract_metadata_endpoint(file: UploadFile = File(...)):
    """
    Extract metadata from PDF document
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        lines = extract_ordered_text(tmp_path)
        metadata = extract_metadata(lines)
        return JSONResponse(content={
            "success": True,
            "metadata": metadata,
            "file_name": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting metadata: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/")
async def root():
    return {
        "message": "PDF Parser API",
        "endpoints": {
            "POST /parse": "Upload PDF for complete parsing",
            "POST /extract-text": "Extract text with formatting",
            "POST /detect-sections": "Detect document sections",
            "POST /classify-questions": "Classify questions",
            "POST /extract-metadata": "Extract metadata",
            "GET /health": "Health check"
        }
    }

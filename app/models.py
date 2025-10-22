from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ContentItem(BaseModel):
    type: str  # description, question, answer
    text: str
    page: int
    number: Optional[str] = None  # For questions

class Section(BaseModel):
    title: str
    type: str = "section"
    page: int
    confidence: str
    content: List[ContentItem] = []

class DocumentInfo(BaseModel):
    document_type: Optional[str] = None
    organization: Optional[str] = None
    file_name: str
    total_pages: int
    total_lines: int
    processing_time_seconds: float

class Metadata(BaseModel):
    dates: Dict[str, str] = {}
    contacts: Dict[str, str] = {}

class DocumentResponse(BaseModel):
    success: bool = True
    document_info: DocumentInfo
    metadata: Metadata
    preamble: List[ContentItem] = []
    sections: List[Section]
    statistics: Dict[str, Any]

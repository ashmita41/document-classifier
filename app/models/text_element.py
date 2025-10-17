"""
Text element data model for PDF extraction.
"""
from typing import Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Type of document element."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADING = "heading"
    LIST_ITEM = "list_item"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x0: float = Field(..., description="Left x-coordinate")
    y0: float = Field(..., description="Top y-coordinate")
    x1: float = Field(..., description="Right x-coordinate")
    y1: float = Field(..., description="Bottom y-coordinate")

    @property
    def width(self) -> float:
        """Calculate width of bounding box."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Calculate height of bounding box."""
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        """Calculate center x-coordinate."""
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        """Calculate center y-coordinate."""
        return (self.y0 + self.y1) / 2


class TextElement(BaseModel):
    """Structured text element extracted from PDF."""
    text: str = Field(..., description="Text content")
    page_number: int = Field(..., description="Page number (1-indexed)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    
    # Font information
    font_size: Optional[float] = Field(None, description="Font size in points")
    font_name: Optional[str] = Field(None, description="Font name")
    is_bold: bool = Field(False, description="Whether text is bold")
    is_italic: bool = Field(False, description="Whether text is italic")
    
    # Layout information
    line_number: Optional[int] = Field(None, description="Line number on page")
    paragraph_id: Optional[int] = Field(None, description="Paragraph identifier")
    indentation_level: Optional[int] = Field(None, description="Indentation level (0=none)")
    vertical_spacing: Optional[float] = Field(None, description="Vertical spacing to previous element")
    
    # Element classification
    element_type: ElementType = Field(ElementType.TEXT, description="Type of element")
    column_index: Optional[int] = Field(None, description="Column index for multi-column layouts")
    
    # Additional metadata
    confidence: Optional[float] = Field(None, description="Extraction confidence score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic config."""
        use_enum_values = True

    def __str__(self) -> str:
        """String representation."""
        return f"TextElement(page={self.page_number}, type={self.element_type}, text='{self.text[:50]}...')"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"TextElement(text='{self.text[:30]}...', page={self.page_number}, "
            f"bbox=({self.bbox.x0:.1f}, {self.bbox.y0:.1f}, {self.bbox.x1:.1f}, {self.bbox.y1:.1f}), "
            f"font_size={self.font_size}, type={self.element_type})"
        )


class ExtractionResult(BaseModel):
    """Result of PDF extraction."""
    elements: list[TextElement] = Field(default_factory=list, description="Extracted text elements")
    tables: list[dict] = Field(default_factory=list, description="Extracted tables")
    total_pages: int = Field(..., description="Total number of pages")
    extraction_method: str = Field(..., description="Method used for extraction")
    success: bool = Field(True, description="Whether extraction was successful")
    errors: list[str] = Field(default_factory=list, description="List of errors encountered")
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    metadata: dict = Field(default_factory=dict, description="Document metadata")

    class Config:
        """Pydantic config."""
        use_enum_values = True


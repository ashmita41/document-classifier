"""
Services module for document classifier.
"""
from app.services.pdf_extractor import PDFExtractor, PDFExtractorError, PDFCorruptedError
# Removed complex services - using simplified structure

__all__ = [
    # PDF Extractor
    "PDFExtractor",
    "PDFExtractorError", 
    "PDFCorruptedError",
]


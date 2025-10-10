"""
Core document analysis module.

This module contains the standalone document analyzer that can be used
independently of the backend API.
"""

from .document_analyzer_v2 import (
    DocumentAnalyzer,
    analyze_document,
    PDFTextExtractor,
    SectionDetector,
    MetadataExtractor,
    QADetector,
    TextLine,
    Section
)

__all__ = [
    'DocumentAnalyzer',
    'analyze_document',
    'PDFTextExtractor',
    'SectionDetector',
    'MetadataExtractor',
    'QADetector',
    'TextLine',
    'Section'
]

__version__ = '2.0'


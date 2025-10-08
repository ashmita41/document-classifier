"""
Robust PDF extraction service with layout preservation and fallback mechanisms.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import io

import pdfplumber
import PyPDF2
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

from app.models.text_element import (
    TextElement, 
    BoundingBox, 
    ElementType, 
    ExtractionResult
)


logger = logging.getLogger(__name__)


class PDFExtractorError(Exception):
    """Base exception for PDF extraction errors."""
    pass


class PDFCorruptedError(PDFExtractorError):
    """Raised when PDF file is corrupted."""
    pass


class PDFExtractor:
    """
    Robust PDF extraction service with comprehensive layout preservation.
    
    Features:
    - Primary extraction with pdfplumber
    - Fallback to PyPDF2
    - Layout preservation with font and position info
    - Multi-column layout detection
    - Table extraction
    - Memory-efficient page-by-page processing
    """
    
    def __init__(
        self,
        min_font_size: float = 6.0,
        max_font_size: float = 72.0,
        paragraph_spacing_threshold: float = 10.0,
        column_detection_enabled: bool = True,
        column_spacing_threshold: float = 30.0,
    ):
        """
        Initialize PDF extractor.
        
        Args:
            min_font_size: Minimum font size to consider
            max_font_size: Maximum font size to consider
            paragraph_spacing_threshold: Vertical spacing threshold for paragraph detection
            column_detection_enabled: Whether to detect multi-column layouts
            column_spacing_threshold: Horizontal spacing threshold for column detection
        """
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.paragraph_spacing_threshold = paragraph_spacing_threshold
        self.column_detection_enabled = column_detection_enabled
        self.column_spacing_threshold = column_spacing_threshold

    def extract_from_file(self, file_path: str) -> ExtractionResult:
        """
        Extract text and structure from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ExtractionResult with extracted elements and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Starting extraction from: {file_path}")
        
        # Try pdfplumber first
        try:
            result = self._extract_with_pdfplumber(file_path)
            result.extraction_method = "pdfplumber"
            logger.info(f"Successfully extracted with pdfplumber: {len(result.elements)} elements")
            return result
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            logger.info("Falling back to PyPDF2")
        
        # Fall back to PyPDF2
        try:
            result = self._extract_with_pypdf2(file_path)
            result.extraction_method = "pypdf2"
            logger.info(f"Successfully extracted with PyPDF2: {len(result.elements)} elements")
            return result
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise PDFExtractorError(f"All extraction methods failed: {e}")

    def extract_from_bytes(self, pdf_bytes: bytes) -> ExtractionResult:
        """
        Extract text and structure from PDF bytes.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            ExtractionResult with extracted elements and metadata
        """
        logger.info("Starting extraction from bytes")
        
        # Try pdfplumber first
        try:
            with io.BytesIO(pdf_bytes) as pdf_file:
                result = self._extract_with_pdfplumber_stream(pdf_file)
                result.extraction_method = "pdfplumber"
                logger.info(f"Successfully extracted with pdfplumber: {len(result.elements)} elements")
                return result
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            logger.info("Falling back to PyPDF2")
        
        # Fall back to PyPDF2
        try:
            with io.BytesIO(pdf_bytes) as pdf_file:
                result = self._extract_with_pypdf2_stream(pdf_file)
                result.extraction_method = "pypdf2"
                logger.info(f"Successfully extracted with PyPDF2: {len(result.elements)} elements")
                return result
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise PDFExtractorError(f"All extraction methods failed: {e}")

    def _extract_with_pdfplumber(self, file_path: Path) -> ExtractionResult:
        """Extract using pdfplumber with full layout information."""
        elements = []
        tables_data = []
        errors = []
        warnings = []
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = self._extract_metadata(pdf)
                total_pages = len(pdf.pages)
                
                logger.info(f"Processing {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract text elements with layout
                        page_elements = self._extract_page_elements_pdfplumber(page, page_num)
                        elements.extend(page_elements)
                        
                        # Extract tables
                        page_tables = self._extract_tables_pdfplumber(page, page_num)
                        tables_data.extend(page_tables)
                        
                    except Exception as e:
                        error_msg = f"Error processing page {page_num}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                # Post-process elements
                elements = self._post_process_elements(elements)
                
                return ExtractionResult(
                    elements=elements,
                    tables=tables_data,
                    total_pages=total_pages,
                    extraction_method="pdfplumber",
                    success=True,
                    errors=errors,
                    warnings=warnings,
                    metadata=metadata
                )
        except Exception as e:
            logger.error(f"Fatal error in pdfplumber extraction: {e}")
            raise PDFExtractorError(f"pdfplumber extraction failed: {e}")

    def _extract_with_pdfplumber_stream(self, pdf_stream: io.BytesIO) -> ExtractionResult:
        """Extract using pdfplumber from stream."""
        elements = []
        tables_data = []
        errors = []
        warnings = []
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_stream) as pdf:
                metadata = self._extract_metadata(pdf)
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_elements = self._extract_page_elements_pdfplumber(page, page_num)
                        elements.extend(page_elements)
                        
                        page_tables = self._extract_tables_pdfplumber(page, page_num)
                        tables_data.extend(page_tables)
                        
                    except Exception as e:
                        error_msg = f"Error processing page {page_num}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                elements = self._post_process_elements(elements)
                
                return ExtractionResult(
                    elements=elements,
                    tables=tables_data,
                    total_pages=total_pages,
                    extraction_method="pdfplumber",
                    success=True,
                    errors=errors,
                    warnings=warnings,
                    metadata=metadata
                )
        except Exception as e:
            raise PDFExtractorError(f"pdfplumber stream extraction failed: {e}")

    def _extract_page_elements_pdfplumber(
        self, 
        page: Any, 
        page_num: int
    ) -> List[TextElement]:
        """
        Extract text elements from a page with full layout information.
        
        Args:
            page: pdfplumber page object
            page_num: Page number (1-indexed)
            
        Returns:
            List of TextElement objects
        """
        elements = []
        
        try:
            # Extract characters with position and font information
            chars = page.chars
            
            if not chars:
                logger.warning(f"No characters found on page {page_num}")
                return elements
            
            # Group characters into lines
            lines = self._group_chars_into_lines(chars)
            
            # Detect columns if enabled
            column_assignments = None
            if self.column_detection_enabled:
                column_assignments = self._detect_columns(lines)
            
            # Process each line
            for line_idx, line_chars in enumerate(lines):
                if not line_chars:
                    continue
                
                # Combine characters into text
                text = ''.join(char['text'] for char in line_chars)
                
                if not text.strip():
                    continue
                
                # Calculate bounding box
                x0 = min(char['x0'] for char in line_chars)
                y0 = min(char['top'] for char in line_chars)
                x1 = max(char['x1'] for char in line_chars)
                y1 = max(char['bottom'] for char in line_chars)
                
                # Get font information (use most common in line)
                font_sizes = [char.get('size', 0) for char in line_chars]
                font_names = [char.get('fontname', '') for char in line_chars]
                
                font_size = self._get_most_common(font_sizes)
                font_name = self._get_most_common(font_names)
                
                # Detect bold/italic from font name
                is_bold = self._is_bold_font(font_name)
                is_italic = self._is_italic_font(font_name)
                
                # Calculate indentation level
                indentation_level = self._calculate_indentation(x0, page.width)
                
                # Determine column
                column_index = None
                if column_assignments and line_idx < len(column_assignments):
                    column_index = column_assignments[line_idx]
                
                # Create text element
                element = TextElement(
                    text=text,
                    page_number=page_num,
                    bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    font_size=font_size,
                    font_name=font_name,
                    is_bold=is_bold,
                    is_italic=is_italic,
                    line_number=line_idx,
                    indentation_level=indentation_level,
                    element_type=ElementType.TEXT,
                    column_index=column_index,
                    confidence=1.0
                )
                
                elements.append(element)
            
            # Assign paragraph IDs and vertical spacing
            elements = self._assign_paragraphs(elements)
            
        except Exception as e:
            logger.error(f"Error extracting elements from page {page_num}: {e}")
            raise
        
        return elements

    def _group_chars_into_lines(self, chars: List[Dict]) -> List[List[Dict]]:
        """
        Group characters into lines based on vertical position.
        
        Args:
            chars: List of character dictionaries
            
        Returns:
            List of lines, where each line is a list of characters
        """
        if not chars:
            return []
        
        # Sort by vertical position, then horizontal
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        lines = []
        current_line = [sorted_chars[0]]
        current_y = sorted_chars[0]['top']
        
        for char in sorted_chars[1:]:
            # If vertical position is close to current line, add to it
            if abs(char['top'] - current_y) < 3:  # 3 point tolerance
                current_line.append(char)
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = [char]
                current_y = char['top']
        
        # Add last line
        if current_line:
            lines.append(current_line)
        
        return lines

    def _detect_columns(self, lines: List[List[Dict]]) -> List[int]:
        """
        Detect multi-column layout using x-coordinate clustering.
        
        Args:
            lines: List of lines (each line is a list of characters)
            
        Returns:
            List of column indices for each line
        """
        if not lines:
            return []
        
        # Get x-coordinate of start of each line
        x_positions = []
        for line in lines:
            if line:
                x_positions.append(min(char['x0'] for char in line))
        
        if len(x_positions) < 2:
            return [0] * len(lines)
        
        # Use DBSCAN to cluster x-positions
        X = np.array(x_positions).reshape(-1, 1)
        
        try:
            clustering = DBSCAN(eps=self.column_spacing_threshold, min_samples=2).fit(X)
            labels = clustering.labels_
            
            # Map cluster labels to column indices (left to right)
            unique_labels = sorted(set(labels))
            label_to_column = {label: idx for idx, label in enumerate(unique_labels)}
            
            column_indices = [label_to_column.get(label, 0) for label in labels]
            
            return column_indices
        except Exception as e:
            logger.warning(f"Column detection failed: {e}")
            return [0] * len(lines)

    def _assign_paragraphs(self, elements: List[TextElement]) -> List[TextElement]:
        """
        Assign paragraph IDs and calculate vertical spacing.
        
        Args:
            elements: List of text elements
            
        Returns:
            Updated list of text elements
        """
        if not elements:
            return elements
        
        paragraph_id = 0
        prev_element = None
        
        for element in elements:
            if prev_element is None:
                element.paragraph_id = paragraph_id
                element.vertical_spacing = 0.0
            else:
                # Calculate vertical spacing
                spacing = element.bbox.y0 - prev_element.bbox.y1
                element.vertical_spacing = spacing
                
                # Check if new paragraph (large spacing or different column)
                is_new_paragraph = (
                    spacing > self.paragraph_spacing_threshold or
                    (element.column_index != prev_element.column_index and
                     element.column_index is not None and
                     prev_element.column_index is not None)
                )
                
                if is_new_paragraph:
                    paragraph_id += 1
                
                element.paragraph_id = paragraph_id
            
            prev_element = element
        
        return elements

    def _extract_tables_pdfplumber(self, page: Any, page_num: int) -> List[Dict]:
        """
        Extract tables from page using pdfplumber.
        
        Args:
            page: pdfplumber page object
            page_num: Page number
            
        Returns:
            List of table dictionaries
        """
        tables_data = []
        
        try:
            tables = page.find_tables()
            
            for table_idx, table in enumerate(tables):
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    if not table_data:
                        continue
                    
                    # Get bounding box
                    bbox = table.bbox
                    
                    table_dict = {
                        'page_number': page_num,
                        'table_index': table_idx,
                        'data': table_data,
                        'bbox': {
                            'x0': bbox[0],
                            'y0': bbox[1],
                            'x1': bbox[2],
                            'y1': bbox[3]
                        },
                        'rows': len(table_data),
                        'columns': len(table_data[0]) if table_data else 0
                    }
                    
                    tables_data.append(table_dict)
                    
                except Exception as e:
                    logger.warning(f"Error extracting table {table_idx} on page {page_num}: {e}")
        
        except Exception as e:
            logger.warning(f"Error finding tables on page {page_num}: {e}")
        
        return tables_data

    def _extract_with_pypdf2(self, file_path: Path) -> ExtractionResult:
        """
        Extract using PyPDF2 (fallback method with limited layout info).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ExtractionResult
        """
        elements = []
        errors = []
        warnings = []
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                metadata = self._extract_pypdf2_metadata(reader)
                
                for page_num in range(total_pages):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text:
                            # Split into lines and create basic elements
                            lines = text.split('\n')
                            for line_idx, line in enumerate(lines):
                                if line.strip():
                                    element = TextElement(
                                        text=line,
                                        page_number=page_num + 1,
                                        bbox=BoundingBox(x0=0, y0=0, x1=100, y1=10),
                                        line_number=line_idx,
                                        element_type=ElementType.TEXT,
                                        confidence=0.5  # Lower confidence for PyPDF2
                                    )
                                    elements.append(element)
                    
                    except Exception as e:
                        error_msg = f"Error processing page {page_num + 1}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                warnings.append("Limited layout information available with PyPDF2 fallback")
                
                return ExtractionResult(
                    elements=elements,
                    tables=[],
                    total_pages=total_pages,
                    extraction_method="pypdf2",
                    success=True,
                    errors=errors,
                    warnings=warnings,
                    metadata=metadata
                )
        
        except Exception as e:
            raise PDFExtractorError(f"PyPDF2 extraction failed: {e}")

    def _extract_with_pypdf2_stream(self, pdf_stream: io.BytesIO) -> ExtractionResult:
        """Extract using PyPDF2 from stream."""
        elements = []
        errors = []
        warnings = []
        
        try:
            reader = PyPDF2.PdfReader(pdf_stream)
            total_pages = len(reader.pages)
            
            metadata = self._extract_pypdf2_metadata(reader)
            
            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        lines = text.split('\n')
                        for line_idx, line in enumerate(lines):
                            if line.strip():
                                element = TextElement(
                                    text=line,
                                    page_number=page_num + 1,
                                    bbox=BoundingBox(x0=0, y0=0, x1=100, y1=10),
                                    line_number=line_idx,
                                    element_type=ElementType.TEXT,
                                    confidence=0.5
                                )
                                elements.append(element)
                
                except Exception as e:
                    error_msg = f"Error processing page {page_num + 1}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            warnings.append("Limited layout information available with PyPDF2 fallback")
            
            return ExtractionResult(
                elements=elements,
                tables=[],
                total_pages=total_pages,
                extraction_method="pypdf2",
                success=True,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
        
        except Exception as e:
            raise PDFExtractorError(f"PyPDF2 stream extraction failed: {e}")

    def extract_tables_as_dataframes(
        self, 
        file_path: str
    ) -> List[Tuple[int, int, pd.DataFrame]]:
        """
        Extract tables from PDF as pandas DataFrames.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of tuples (page_number, table_index, dataframe)
        """
        file_path = Path(file_path)
        dataframes = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        tables = page.find_tables()
                        
                        for table_idx, table in enumerate(tables):
                            try:
                                table_data = table.extract()
                                
                                if not table_data or len(table_data) < 2:
                                    continue
                                
                                # First row as headers
                                headers = table_data[0]
                                data_rows = table_data[1:]
                                
                                # Create DataFrame
                                df = pd.DataFrame(data_rows, columns=headers)
                                
                                # Clean up None values
                                df = df.fillna('')
                                
                                dataframes.append((page_num, table_idx, df))
                                
                            except Exception as e:
                                logger.warning(
                                    f"Error converting table {table_idx} on page {page_num} "
                                    f"to DataFrame: {e}"
                                )
                    
                    except Exception as e:
                        logger.warning(f"Error processing tables on page {page_num}: {e}")
        
        except Exception as e:
            logger.error(f"Error extracting tables as DataFrames: {e}")
            raise PDFExtractorError(f"Table extraction failed: {e}")
        
        return dataframes

    def _post_process_elements(self, elements: List[TextElement]) -> List[TextElement]:
        """
        Post-process extracted elements to classify types.
        
        Args:
            elements: List of text elements
            
        Returns:
            Updated list of text elements
        """
        for element in elements:
            # Detect headings based on font size
            if element.font_size and element.font_size > 14:
                element.element_type = ElementType.HEADING
            
            # Detect list items
            text_stripped = element.text.strip()
            if text_stripped and (
                text_stripped[0] in ['•', '◦', '▪', '-', '*'] or
                (len(text_stripped) > 2 and text_stripped[0].isdigit() and text_stripped[1] in ['.', ')'])
            ):
                element.element_type = ElementType.LIST_ITEM
        
        return elements

    def _extract_metadata(self, pdf: Any) -> Dict[str, Any]:
        """Extract metadata from pdfplumber PDF object."""
        metadata = {}
        
        try:
            if hasattr(pdf, 'metadata') and pdf.metadata:
                metadata = dict(pdf.metadata)
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata

    def _extract_pypdf2_metadata(self, reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """Extract metadata from PyPDF2 reader."""
        metadata = {}
        
        try:
            if hasattr(reader, 'metadata') and reader.metadata:
                for key, value in reader.metadata.items():
                    metadata[key] = str(value)
        except Exception as e:
            logger.warning(f"Error extracting PyPDF2 metadata: {e}")
        
        return metadata

    def _calculate_indentation(self, x0: float, page_width: float) -> int:
        """
        Calculate indentation level based on x-coordinate.
        
        Args:
            x0: Left x-coordinate
            page_width: Width of page
            
        Returns:
            Indentation level (0 = no indentation)
        """
        # Normalize to percentage of page width
        position_pct = (x0 / page_width) * 100
        
        # Define indentation levels (adjust as needed)
        if position_pct < 15:
            return 0
        elif position_pct < 25:
            return 1
        elif position_pct < 35:
            return 2
        else:
            return 3

    def _is_bold_font(self, font_name: str) -> bool:
        """Check if font name indicates bold style."""
        if not font_name:
            return False
        
        font_name_lower = font_name.lower()
        return any(keyword in font_name_lower for keyword in ['bold', 'heavy', 'black'])

    def _is_italic_font(self, font_name: str) -> bool:
        """Check if font name indicates italic style."""
        if not font_name:
            return False
        
        font_name_lower = font_name.lower()
        return any(keyword in font_name_lower for keyword in ['italic', 'oblique', 'slant'])

    def _get_most_common(self, items: List[Any]) -> Any:
        """Get most common item from list."""
        if not items:
            return None
        
        # Count occurrences
        counts = defaultdict(int)
        for item in items:
            counts[item] += 1
        
        # Return most common
        return max(counts.items(), key=lambda x: x[1])[0] if counts else None


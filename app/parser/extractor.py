"""
PDF Text Extractor using pdfplumber

This module provides functionality to extract text from PDF files with detailed formatting information
including font size, bold detection, and Y-position for each line of text.
"""

import pdfplumber
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def extract_ordered_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with proper ordering using pdfplumber.
    Fixes text extraction order by sorting: page -> Y-position -> X-position
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing text lines with formatting info
    """
    all_lines = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"Processing PDF with {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                logger.debug(f"Processing page {page_num}")
                
                # Extract words with layout information
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False
                )
                
                if not words:
                    logger.warning(f"No words found on page {page_num}")
                    continue
                
                # Group words into lines by Y-position
                lines_dict = {}
                for word in words:
                    y = round(word['top'], 1)  # Round to group nearby words
                    if y not in lines_dict:
                        lines_dict[y] = []
                    lines_dict[y].append(word)
                
                # Sort lines top-to-bottom, words left-to-right
                for y in sorted(lines_dict.keys()):
                    words_in_line = sorted(lines_dict[y], key=lambda w: w['x0'])
                    text = ' '.join(w['text'] for w in words_in_line)
                    
                    if text.strip():  # Only add non-empty lines
                        # Get font info from first word
                        first_word = words_in_line[0]
                        
                        # Check if any word in the line is bold
                        is_bold = any('Bold' in word.get('fontname', '') for word in words_in_line)
                        
                        all_lines.append({
                            'text': text.strip(),
                            'page_number': page_num,
                            'y_position': y,
                            'font_size': first_word.get('height', 10),
                            'is_bold': is_bold,
                            'x_position': first_word['x0'],
                            'width': first_word['x1'] - first_word['x0'],
                            'height': first_word['bottom'] - first_word['top']
                        })
                
                logger.debug(f"Extracted {len(lines_dict)} lines from page {page_num}")
    
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        raise
    
    # Final sort: page number, then Y-position, then X-position
    all_lines.sort(key=lambda line: (
        line.get('page_number', 1),
        line.get('y_position', 0),
        line.get('x_position', 0)
    ))
    
    logger.info(f"Successfully extracted {len(all_lines)} non-empty lines from PDF")
    return all_lines

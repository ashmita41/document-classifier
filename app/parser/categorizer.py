"""
Document Categorizer for PDF Documents

This module provides functionality to categorize every line in a document
and build a hierarchical structure with sections, questions, and descriptions.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentStructure:
    """Represents the complete document structure."""
    document_info: Dict[str, Any]
    metadata: Dict[str, Any]
    preamble: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    statistics: Dict[str, Any]


class DocumentCategorizer:
    """Categorizes document lines and builds hierarchical structure."""
    
    def __init__(self):
        self.metadata_keywords = [
            'due date', 'deadline', 'respond by', 'submit by', 'proposal due',
            'contact:', 'attn:', 'point of contact:', 'email:', 'phone:',
            'company name:', 'address:', 'website:', 'effective date',
            'start date', 'end date', 'submission deadline', 'response due',
            'proposal deadline', 'bid due', 'application due'
        ]
        
        self.subsection_patterns = [
            r'^[A-Z]\.\s+',  # A., B., C.
            r'^\d+\.\d+\s+',  # 1.1, 1.2, 2.1
            r'^[a-z]\)\s+',  # a), b), c)
            r'^\d+\)\s+',  # 1), 2), 3)
            r'^[A-Z][a-z]+\.\s+',  # Background., Overview., etc.
        ]
    
    def categorize_document(self, 
                          lines: List[Dict[str, Any]], 
                          sections: List[Any], 
                          questions: List[Any], 
                          metadata: Any) -> DocumentStructure:
        """
        Categorize every line and build hierarchical document structure.
        
        Args:
            lines: List of line dictionaries with text and formatting info
            sections: List of detected sections
            questions: List of detected questions
            metadata: Extracted metadata
            
        Returns:
            DocumentStructure with complete hierarchical organization
        """
        if not lines:
            return DocumentStructure({}, {}, [], [], {})
        
        logger.info(f"Starting document categorization with {len(lines)} lines, {len(sections)} sections, {len(questions)} questions")
        
        # Create maps for quick lookup
        section_map = {s.get('line_index', 0): s for s in sections}
        question_map = {q.get('line_index', 0): q for q in questions}
        
        # Initialize document structure
        document_info = self._build_document_info(lines, metadata)
        preamble = []
        sections_list = []
        current_section = None
        current_section_content = []
        current_subsection = None
        
        # Process each line
        for i, line in enumerate(lines):
            text = line.get('text', '').strip()
            
            # Skip empty lines
            if not text or len(text) < 3:
                continue
            
            # Check if this is a section header
            if i in section_map:
                # Save previous section if exists
                if current_section:
                    current_section['content'] = current_section_content
                    sections_list.append(current_section)
                
                # Start new section
                current_section = section_map[i].copy()
                current_section['type'] = 'section'
                current_section['page'] = line.get('page_number', 1)
                current_section_content = []
                current_subsection = None
                continue
            
            # Check if this is a question
            if i in question_map:
                question_obj = question_map[i]
                question_item = {
                    "type": "question",
                    "text": question_obj.get('text', ''),
                    "page": line.get('page_number', 1),
                    "number": question_obj.get('question_number', ''),
                    "confidence": question_obj.get('confidence', 'medium')
                }
                
                if current_subsection:
                    # Add to current subsection
                    current_section_content.append(question_item)
                elif current_section:
                    current_section_content.append(question_item)
                else:
                    preamble.append(question_item)
                continue
            
            # Check if metadata line (skip in content)
            if self._is_metadata_line(text):
                continue
            
            # Check if subsection
            subsection_info = self._detect_subsection(text, i, line)
            if subsection_info:
                # Save previous subsection if exists
                if current_subsection and current_section:
                    current_section_content.append(current_subsection)
                
                # Start new subsection
                current_subsection = {
                    "type": "subsection",
                    "title": subsection_info['title'],
                    "page": line.get('page_number', 1),
                    "content": []
                }
                continue
            
            # Otherwise, it's a description
            description_item = {
                "type": "description",
                "text": text,
                "page": line.get('page_number', 1)
            }
            
            if current_subsection:
                # Add to current subsection
                current_subsection['content'].append(description_item)
            elif current_section:
                current_section_content.append(description_item)
            else:
                preamble.append(description_item)
        
        # Add final subsection if exists
        if current_subsection and current_section:
            current_section_content.append(current_subsection)
        
        # Add final section if exists
        if current_section:
            current_section['content'] = current_section_content
            sections_list.append(current_section)
        
        # Build statistics
        statistics = self._build_statistics(lines, sections_list, questions, preamble)
        
        return DocumentStructure(
            document_info=document_info,
            metadata=metadata,
            preamble=preamble,
            sections=sections_list,
            statistics=statistics
        )
    
    def _build_document_info(self, lines: List[Dict[str, Any]], metadata: Any) -> Dict[str, Any]:
        """Build document information."""
        total_pages = max((line.get('page_number', 1) for line in lines), default=1)
        
        return {
            "document_type": metadata.get('document_info', {}).get('document_type'),
            "organization": metadata.get('document_info', {}).get('organization'),
            "title": metadata.get('document_info', {}).get('title'),
            "total_pages": total_pages,
            "total_lines": len(lines),
            "processing_time": datetime.now().isoformat()
        }
    
    def _is_metadata_line(self, text: str) -> bool:
        """Check if line contains metadata that should be excluded from content."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.metadata_keywords)
    
    def _detect_subsection(self, text: str, line_index: int, line: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect if line is a subsection header."""
        for pattern in self.subsection_patterns:
            if re.match(pattern, text):
                return {
                    'title': text,
                    'pattern': pattern,
                    'line_index': line_index
                }
        return None
    
    def _build_statistics(self, lines: List[Dict[str, Any]], sections: List[Dict[str, Any]], 
                         questions: List[Any], preamble: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build document statistics."""
        total_sections = len(sections)
        total_subsections = 0
        total_questions = 0
        total_descriptions = 0
        
        # Count subsections, questions, and descriptions in sections
        for section in sections:
            for item in section.get('content', []):
                if item.get('type') == 'subsection':
                    total_subsections += 1
                    # Count questions and descriptions within subsections
                    for sub_item in item.get('content', []):
                        if sub_item.get('type') == 'question':
                            total_questions += 1
                        elif sub_item.get('type') == 'description':
                            total_descriptions += 1
                elif item.get('type') == 'question':
                    total_questions += 1
                elif item.get('type') == 'description':
                    total_descriptions += 1
        
        # Count questions and descriptions in preamble
        for item in preamble:
            if item.get('type') == 'question':
                total_questions += 1
            elif item.get('type') == 'description':
                total_descriptions += 1
        
        preamble_items = len(preamble)
        
        return {
            "total_sections": total_sections,
            "total_subsections": total_subsections,
            "total_questions": total_questions,
            "total_descriptions": total_descriptions,
            "preamble_items": preamble_items,
            "pages_analyzed": max((line.get('page_number', 1) for line in lines), default=1)
        }


def build_document_structure(lines: List[Dict[str, Any]], 
                            sections: List[Dict[str, Any]], 
                            questions: List[Dict[str, Any]], 
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to build complete document structure.
    
    Args:
        lines: List of line dictionaries with text and formatting info
        sections: List of detected sections
        questions: List of detected questions
        metadata: Extracted metadata
        
    Returns:
        Dictionary with complete document structure
    """
    categorizer = DocumentCategorizer()
    structure = categorizer.categorize_document(lines, sections, questions, metadata)
    
    return {
        "success": True,
        "document_info": structure.document_info,
        "metadata": structure.metadata,
        "preamble": structure.preamble,
        "sections": structure.sections,
        "statistics": structure.statistics
    }

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
            sections: List of detected sections with hierarchy
            questions: List of detected questions
            metadata: Extracted metadata
            
        Returns:
            DocumentStructure with complete hierarchical organization
        """
        if not lines:
            return DocumentStructure({}, {}, [], [], {})
        
        logger.info(f"Starting hierarchical document categorization with {len(lines)} lines, {len(sections)} sections, {len(questions)} questions")
        
        # Create maps for quick lookup
        section_map = {s.get('line_index', 0): s for s in sections}
        question_map = {q.get('line_index', 0): q for q in questions}
        
        # Initialize document structure
        document_info = self._build_document_info(lines, metadata)
        preamble = []
        
        # Build hierarchical sections with proper content assignment
        sections_list = self._build_hierarchical_sections(lines, sections, questions)
        
        # Build statistics
        statistics = self._build_statistics(lines, sections_list, questions, preamble)
        
        return DocumentStructure(
            document_info=document_info,
            metadata=metadata,
            preamble=preamble,
            sections=sections_list,
            statistics=statistics
        )
    
    def _build_hierarchical_sections(self, lines: List[Dict[str, Any]], 
                                   sections: List[Any], 
                                   questions: List[Any]) -> List[Dict[str, Any]]:
        """Build hierarchical sections with proper content assignment."""
        if not sections:
            return []
        
        # Sort sections by line index
        sorted_sections = sorted(sections, key=lambda s: s.get('line_index', 0))
        
        # Create section objects with hierarchy
        section_objects = []
        for section in sorted_sections:
            section_obj = {
                "id": section.get('id', ''),
                "title": section.get('title', ''),
                "level": section.get('level', 1),
                "parent_id": section.get('parent_id'),
                "children": [],
                "content": [],
                "page": section.get('page_number', 1),
                "confidence": section.get('confidence', 'medium'),
                "type": "section"
            }
            section_objects.append(section_obj)
        
        # Assign content to sections
        for i, section in enumerate(section_objects):
            # Find the next section of same or higher level
            next_section_index = len(lines)
            for j in range(i + 1, len(section_objects)):
                if section_objects[j]['level'] <= section['level']:
                    next_section_index = section_objects[j].get('line_index', len(lines))
                    break
            
            # Collect content between current section and next section
            content = self._collect_section_content(
                lines, 
                section.get('line_index', 0) + 1, 
                next_section_index, 
                questions
            )
            section['content'] = content
        
        # Build parent-child relationships
        self._build_parent_child_relationships(section_objects)
        
        # Return only top-level sections (level 1)
        return [s for s in section_objects if s['level'] == 1]
    
    def _collect_section_content(self, lines: List[Dict[str, Any]], 
                               start_index: int, 
                               end_index: int, 
                               questions: List[Any]) -> List[Dict[str, Any]]:
        """Collect content for a section between start and end indices."""
        content = []
        question_map = {q.get('line_index', 0): q for q in questions}
        
        for i in range(start_index, min(end_index, len(lines))):
            line = lines[i]
            text = line.get('text', '').strip()
            
            # Skip empty lines and noise
            if not text or len(text) < 3:
                continue
            
            # Skip metadata lines
            if self._is_metadata_line(text):
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
                content.append(question_item)
            else:
                # Regular content
                content_item = {
                    "type": "description",
                    "text": text,
                    "page": line.get('page_number', 1)
                }
                content.append(content_item)
        
        return content
    
    def _build_parent_child_relationships(self, sections: List[Dict[str, Any]]) -> None:
        """Build parent-child relationships between sections."""
        # Create a map of sections by ID
        section_map = {s['id']: s for s in sections}
        
        # Build relationships
        for section in sections:
            if section['parent_id']:
                parent = section_map.get(section['parent_id'])
                if parent:
                    parent['children'].append(section)
                    # Remove from top-level list
                    if section in sections:
                        sections.remove(section)
    
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

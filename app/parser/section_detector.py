"""
Hierarchical Section Detector for RFP Documents

This module provides functionality to detect section headers in RFP documents
with proper hierarchical structure detection using a 3-pass algorithm.
"""

import re
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class SectionCandidate:
    """Represents a potential section header."""
    id: str
    title: str
    level: int  # 1, 2, or 3
    index: int
    confidence: str  # "high", "medium", "low"
    font_size: float
    is_bold: bool
    y_position: float
    page_number: int
    reason: str  # Why this was identified as a section
    parent_id: Optional[str] = None
    style: Dict[str, Any] = None


class HierarchicalSectionDetector:
    """Enhanced section detector with proper hierarchical structure detection."""
    
    def __init__(self):
        self.avg_font_size = 0.0
        self.mode_font_size = 0.0
        self.section_threshold = 0.0
        self.excluded_headers = set()
        self.page_headers = set()
        self.section_style_profile = None
        self.debug_info = {
            "total_lines_analyzed": 0,
            "section_candidates_found": 0,
            "sections_confirmed": 0,
            "excluded_count": 0
        }
        
        # Pattern definitions for hierarchical detection
        self.level1_pattern = re.compile(r'^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+[A-Z][A-Z\s]+$')
        self.level2_pattern = re.compile(r'^\s*[A-J]\.\s+[A-Z][a-z].*')
        self.level3_pattern = re.compile(r'^\s*\d{1,2}\.\s+')
        
        # Question indicators for Level 3 classification
        self.question_indicators = [
            'please', 'describe', 'what', 'how', 'do you', 'can you', 'will you',
            'explain', 'provide', 'list', 'identify', 'outline', 'detail'
        ]
    
    def detect_sections(self, lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """
        Detect section headers with proper hierarchical structure.
        
        Uses a 3-pass algorithm:
        1. FIRST PASS - Detect document structure markers by pattern
        2. SECOND PASS - Build hierarchy relationships
        3. THIRD PASS - Classify Level 3 items correctly
        
        Args:
            lines: List of line dictionaries with text and formatting info
            
        Returns:
            List of detected sections with proper hierarchy
        """
        if not lines:
            return []
        
        logger.info(f"Starting hierarchical section detection on {len(lines)} lines")
        
        # Step 1: Pre-filtering - identify and exclude noise
        self._pre_filter_noise(lines)
        
        # Step 2: Calculate baseline metrics
        self._calculate_baseline_metrics(lines)
        
        # FIRST PASS - Detect document structure markers by pattern
        logger.debug("FIRST PASS: Detecting structure markers by pattern")
        level1_sections = self._detect_level1_sections(lines)
        level2_sections = self._detect_level2_sections(lines)
        level3_items = self._detect_level3_items(lines)
        
        
        # SECOND PASS - Build hierarchy relationships
        logger.debug("SECOND PASS: Building hierarchy relationships")
        sections = self._build_hierarchy(level1_sections, level2_sections, level3_items)
        
        # THIRD PASS - Classify Level 3 items correctly
        logger.debug("THIRD PASS: Classifying Level 3 items")
        sections = self._classify_level3_items(sections, lines)
        
        self.debug_info["sections_confirmed"] = len(sections)
        logger.info(f"Detected {len(sections)} sections with hierarchy")
        
        return sections
    
    def _pre_filter_noise(self, lines: List[Dict[str, Any]]) -> None:
        """Identify and exclude common noise patterns."""
        logger.debug("Pre-filtering noise patterns")
        
        # Track page headers (lines appearing at top of multiple pages)
        page_header_candidates = {}
        
        for line in lines:
            text = line.get('text', '').strip()
            page_num = line.get('page_number', 1)
            
            # Skip very short lines
            if len(text) < 3:
                self.excluded_headers.add(text)
                continue
            
            # Track potential page headers
            if page_num <= 3:  # Only check first few pages
                if text not in page_header_candidates:
                    page_header_candidates[text] = []
                page_header_candidates[text].append(page_num)
        
        # Identify repeated headers (but be more selective)
        for text, pages in page_header_candidates.items():
            if len(pages) > 1:  # Appears on multiple pages
                # Only exclude if it looks like a page header (short, not section-like)
                if (len(text.split()) <= 3 and 
                    not re.match(r'^[A-Z]\.\s+', text) and  # Not a section header
                    not re.match(r'^\d+\.\s+', text) and   # Not a numbered item
                    not re.match(r'^[IVX]+\.\s+', text)):  # Not a Roman numeral section
                    self.page_headers.add(text)
                    self.excluded_headers.add(text)
        
        # Add common noise patterns
        noise_patterns = [
            r"^Page \d+",
            r"^Page \d+ of \d+",
            r"Request [Ff]or [Pp]roposal",
            r"^[A-Z]{2,}\s+Lab$",  # ACME Lab, etc.
            r"^Generic Company",
            r"^\d{4}$",  # Years
            r"^[A-Z]{1,2}$",  # Single letters
        ]
        
        for pattern in noise_patterns:
            for line in lines:
                text = line.get('text', '').strip()
                if re.match(pattern, text):
                    self.excluded_headers.add(text)
        
        logger.debug(f"Excluded {len(self.excluded_headers)} noise patterns")
    
    def _calculate_baseline_metrics(self, lines: List[Dict[str, Any]]) -> None:
        """Calculate baseline font metrics for section detection."""
        logger.debug("Calculating baseline metrics")
        
        # Filter out excluded headers and get font sizes
        valid_lines = []
        for line in lines:
            text = line.get('text', '').strip()
            if (text not in self.excluded_headers and 
                text not in self.page_headers and
                len(text) >= 3):
                valid_lines.append(line)
        
        if not valid_lines:
            logger.warning("No valid lines for baseline calculation")
            return
        
        # Calculate font size statistics
        font_sizes = [line.get('font_size', 10) for line in valid_lines]
        self.avg_font_size = statistics.mean(font_sizes)
        self.mode_font_size = statistics.mode(font_sizes)
        self.section_threshold = self.avg_font_size + 1.5  # 1.5pt above average
        
        logger.debug(f"Baseline: avg={self.avg_font_size:.1f}, mode={self.mode_font_size:.1f}, threshold={self.section_threshold:.1f}")
    
    def _detect_level1_sections(self, lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """Detect Level 1 sections (Roman numerals + uppercase text)."""
        sections = []
        
        for i, line in enumerate(lines):
            text = line.get('text', '').strip()
            
            # Skip excluded headers
            if text in self.excluded_headers or text in self.page_headers:
                continue
            
            # Check Level 1 pattern: Roman numerals followed by period + uppercase text
            if self.level1_pattern.match(text):
                # Extract Roman numeral for ID
                roman_match = re.match(r'^\s*([IVX]+)\.', text)
                roman_id = roman_match.group(1) if roman_match else f"L1_{i}"
                
                section = SectionCandidate(
                    id=roman_id,
                    title=text,
                    level=1,
                    index=i,
                    confidence="high",
                    font_size=line.get('font_size', 10),
                    is_bold=line.get('is_bold', False),
                    y_position=line.get('y_position', 0),
                    page_number=line.get('page_number', 1),
                    reason="level1_roman_pattern",
                    parent_id=None,
                    style={"pattern": "level1_roman", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
                )
                sections.append(section)
                logger.debug(f"Found Level 1 section: {text}")
        
        return sections
    
    def _detect_level2_sections(self, lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """Detect Level 2 sections (Single letters + title case)."""
        sections = []
        
        for i, line in enumerate(lines):
            text = line.get('text', '').strip()
            
            # Skip excluded headers
            if text in self.excluded_headers or text in self.page_headers:
                continue
            
            
            # Check Level 2 pattern: Single letters followed by period + title case
            if self.level2_pattern.match(text):
                # Extract letter for ID
                letter_match = re.match(r'^\s*([A-J])\.', text)
                letter_id = letter_match.group(1) if letter_match else f"L2_{i}"
                
                section = SectionCandidate(
                    id=letter_id,
                    title=text,
                    level=2,
                    index=i,
                    confidence="high",
                    font_size=line.get('font_size', 10),
                    is_bold=line.get('is_bold', False),
                    y_position=line.get('y_position', 0),
                    page_number=line.get('page_number', 1),
                    reason="level2_letter_pattern",
                    parent_id=None,  # Will be set in hierarchy building
                    style={"pattern": "level2_letter", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
                )
                sections.append(section)
                logger.debug(f"Found Level 2 section: {text}")
        
        return sections
    
    def _detect_level3_items(self, lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """Detect Level 3 items (Numbers followed by period)."""
        items = []
        
        for i, line in enumerate(lines):
            text = line.get('text', '').strip()
            
            # Skip excluded headers
            if text in self.excluded_headers or text in self.page_headers:
                continue
            
            # Check Level 3 pattern: Numbers followed by period
            if self.level3_pattern.match(text):
                # Extract number for ID
                number_match = re.match(r'^\s*(\d{1,2})\.', text)
                number_id = number_match.group(1) if number_match else f"L3_{i}"
                
                item = SectionCandidate(
                    id=number_id,
                    title=text,
                    level=3,
                    index=i,
                    confidence="medium",
                    font_size=line.get('font_size', 10),
                    is_bold=line.get('is_bold', False),
                    y_position=line.get('y_position', 0),
                    page_number=line.get('page_number', 1),
                    reason="level3_number_pattern",
                    parent_id=None,  # Will be set in hierarchy building
                    style={"pattern": "level3_number", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
                )
                items.append(item)
                logger.debug(f"Found Level 3 item: {text}")
        
        return items
    
    def _build_hierarchy(self, level1_sections: List[SectionCandidate], 
                        level2_sections: List[SectionCandidate], 
                        level3_items: List[SectionCandidate]) -> List[SectionCandidate]:
        """Build hierarchy relationships between sections."""
        all_sections = []
        
        # Sort all sections by index
        all_items = sorted(level1_sections + level2_sections + level3_items, key=lambda x: x.index)
        
        current_level1 = None
        current_level2 = None
        
        for item in all_items:
            if item.level == 1:
                # Level 1 section - reset current Level 2
                current_level1 = item
                current_level2 = None
                item.parent_id = None
                all_sections.append(item)
                logger.debug(f"Set Level 1: {item.title}")
                
            elif item.level == 2:
                # Level 2 section - set parent to current Level 1
                if current_level1:
                    item.parent_id = current_level1.id
                    current_level2 = item
                    all_sections.append(item)
                    logger.debug(f"Set Level 2: {item.title} under {current_level1.title}")
                else:
                    # Orphaned Level 2 - treat as Level 1
                    item.level = 1
                    item.parent_id = None
                    all_sections.append(item)
                    logger.debug(f"Promoted orphaned Level 2 to Level 1: {item.title}")
                
            elif item.level == 3:
                # Level 3 item - set parent to current Level 2 or Level 1
                if current_level2:
                    item.parent_id = current_level2.id
                    all_sections.append(item)
                    logger.debug(f"Set Level 3: {item.title} under {current_level2.title}")
                elif current_level1:
                    item.parent_id = current_level1.id
                    all_sections.append(item)
                    logger.debug(f"Set Level 3: {item.title} under {current_level1.title}")
                else:
                    # Orphaned Level 3 - treat as Level 1
                    item.level = 1
                    item.parent_id = None
                    all_sections.append(item)
                    logger.debug(f"Promoted orphaned Level 3 to Level 1: {item.title}")
        
        return all_sections
    
    def _classify_level3_items(self, sections: List[SectionCandidate], lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """Classify Level 3 items as questions or sections."""
        for section in sections:
            if section.level == 3:
                text_lower = section.title.lower()
                
                # Check if it contains question indicators
                is_question = any(indicator in text_lower for indicator in self.question_indicators)
                
                if is_question:
                    section.reason = "level3_question"
                    section.confidence = "high"
                    logger.debug(f"Classified as question: {section.title}")
                else:
                    section.reason = "level3_section"
                    section.confidence = "medium"
                    logger.debug(f"Classified as section: {section.title}")
        
        return sections
    
    def _assign_content_to_sections(self, sections: List[SectionCandidate], lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """Assign content between sections to the appropriate section."""
        # Create a map of section indices
        section_map = {s.index: s for s in sections}
        
        # Sort sections by index
        sorted_sections = sorted(sections, key=lambda x: x.index)
        
        for i, section in enumerate(sorted_sections):
            # Find the next section of same or higher level
            next_section_index = len(lines)
            for j in range(i + 1, len(sorted_sections)):
                if sorted_sections[j].level <= section.level:
                    next_section_index = sorted_sections[j].index
                    break
            
            # Collect content between current section and next section
            content_lines = []
            for line_idx in range(section.index + 1, next_section_index):
                if line_idx < len(lines):
                    line = lines[line_idx]
                    text = line.get('text', '').strip()
                    
                    # Skip empty lines and noise
                    if text and text not in self.excluded_headers and text not in self.page_headers:
                        content_lines.append({
                            'text': text,
                            'page': line.get('page_number', 1),
                            'line_index': line_idx
                        })
            
            # Store content in section (this would need to be added to SectionCandidate)
            # For now, we'll just log the content
            logger.debug(f"Section '{section.title}' has {len(content_lines)} content lines")
        
        return sections


def detect_sections(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to detect sections with hierarchical structure.
    
    Args:
        lines: List of line dictionaries with text and formatting info
        
    Returns:
        Dictionary with hierarchical sections and debug info
    """
    detector = HierarchicalSectionDetector()
    sections = detector.detect_sections(lines)
    
    # Convert to dictionary format with proper hierarchy
    sections_list = []
    for section in sections:
        sections_list.append({
            "id": section.id,
            "title": section.title,
            "level": section.level,
            "parent_id": section.parent_id,
            "line_index": section.index,
            "confidence": section.confidence,
            "style": section.style,
            "page_number": section.page_number,
            "y_position": section.y_position,
            "reason": section.reason
        })
    
    return {
        "sections": sections_list,
        "excluded_headers": list(detector.excluded_headers),
        "debug_info": detector.debug_info
    }
"""
Enhanced Section Detector for RFP Documents

This module provides functionality to detect section headers in RFP documents
with improved accuracy by filtering noise and using RFP-specific patterns.
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
    title: str
    index: int
    confidence: str  # "high", "medium", "low"
    font_size: float
    is_bold: bool
    y_position: float
    page_number: int
    reason: str  # Why this was identified as a section
    style: Dict[str, Any] = None  # Style signature
    parent_section: str = None  # Parent section if hierarchical


class SectionDetector:
    """Enhanced section detector for RFP documents."""
    
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
    
    def detect_sections(self, lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """
        Detect section headers in the document.
        
        Args:
            lines: List of line dictionaries with text and formatting info
            
        Returns:
            List of detected sections with confidence levels
        """
        if not lines:
            return []
        
        logger.info(f"Starting section detection on {len(lines)} lines")
        
        # Step 1: Pre-filtering - identify and exclude noise
        self._pre_filter_noise(lines)
        
        # Step 2: Calculate baseline metrics
        self._calculate_baseline_metrics(lines)
        
        # Step 3: Detect sections using priority rules
        sections = self._detect_sections_with_rules(lines)
        
        # Step 4: Style consistency check
        if sections:
            sections = self._apply_style_consistency_check(lines, sections)
        
        # Step 5: Handle numbered lists vs sections
        sections = self._handle_numbered_lists_vs_sections(lines, sections)
        
        self.debug_info["sections_confirmed"] = len(sections)
        logger.info(f"Detected {len(sections)} sections")
        
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
        
        # Identify repeated headers
        for text, pages in page_header_candidates.items():
            if len(pages) > 1:  # Appears on multiple pages
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
    
    def _detect_sections_with_rules(self, lines: List[Dict[str, Any]]) -> List[SectionCandidate]:
        """Detect sections using priority-ordered rules."""
        logger.debug("Applying section detection rules")
        
        sections = []
        
        for i, line in enumerate(lines):
            text = line.get('text', '').strip()
            
            # Skip excluded headers
            if text in self.excluded_headers or text in self.page_headers:
                continue
            
            # Skip very short or very long lines
            word_count = len(text.split())
            if word_count < 2 or word_count > 15:
                continue
            
            # Skip questions
            if text.endswith('?'):
                continue
            
            # Skip lines with "Page" in them
            if 'Page' in text:
                continue
            
            # Skip organization/company names (all caps, short)
            if text.isupper() and word_count <= 3:
                continue
            
            # Skip single word lines
            if word_count == 1:
                continue
            
            # HIGH CONFIDENCE RULES
            section = self._check_high_confidence_rules(text, i, line)
            if section:
                sections.append(section)
                continue
            
            # MEDIUM CONFIDENCE RULES
            section = self._check_medium_confidence_rules(text, i, line)
            if section:
                sections.append(section)
        
        # Sort by line index
        sections.sort(key=lambda s: s.index)
        
        logger.debug(f"Found {len(sections)} section candidates")
        return sections
    
    def _check_high_confidence_rules(self, text: str, index: int, line: Dict[str, Any]) -> Optional[SectionCandidate]:
        """Check high confidence section patterns."""
        
        # Rule 1: Section [Roman/Number/Letter] pattern
        section_roman_match = re.match(r'^Section\s+([IVX]+|\d+|[A-Z])', text, re.IGNORECASE)
        if section_roman_match:
            return SectionCandidate(
                title=text,
                index=index,
                confidence="high",
                font_size=line.get('font_size', 10),
                is_bold=line.get('is_bold', False),
                y_position=line.get('y_position', 0),
                page_number=line.get('page_number', 1),
                reason="section_roman_pattern",
                style={"pattern": "section_roman", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
            )
        
        # Rule 2: Single capital letter + period + space (A. Background)
        letter_period_match = re.match(r'^[A-Z]\.\s+[A-Z]', text)
        if letter_period_match:
            return SectionCandidate(
                title=text,
                index=index,
                confidence="high",
                font_size=line.get('font_size', 10),
                is_bold=line.get('is_bold', False),
                y_position=line.get('y_position', 0),
                page_number=line.get('page_number', 1),
                reason="letter_period_pattern",
                style={"pattern": "letter_prefix", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
            )
        
        # Rule 3: Bold + Larger than average (2+ pts) + Title Case + 3-10 words
        font_size = line.get('font_size', 10)
        is_bold = line.get('is_bold', False)
        word_count = len(text.split())
        
        if (is_bold and 
            font_size >= (self.avg_font_size + 2.0) and  # 2+ points larger than average
            text.istitle() and 
            3 <= word_count <= 10):
            return SectionCandidate(
                title=text,
                index=index,
                confidence="high",
                font_size=font_size,
                is_bold=is_bold,
                y_position=line.get('y_position', 0),
                page_number=line.get('page_number', 1),
                reason="bold_large_title_case",
                style={"pattern": "bold_title", "font_size": font_size, "bold": is_bold}
            )
        
        return None
    
    def _check_medium_confidence_rules(self, text: str, index: int, line: Dict[str, Any]) -> Optional[SectionCandidate]:
        """Check medium confidence section patterns."""
        
        # Rule 1: All uppercase 3-8 words (but NOT single words like "ACME")
        word_count = len(text.split())
        if (text.isupper() and 
            3 <= word_count <= 8 and 
            word_count > 1):  # Not single word
            return SectionCandidate(
                title=text,
                index=index,
                confidence="medium",
                font_size=line.get('font_size', 10),
                is_bold=line.get('is_bold', False),
                y_position=line.get('y_position', 0),
                page_number=line.get('page_number', 1),
                reason="all_uppercase",
                style={"pattern": "uppercase", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
            )
        
        # Rule 2: Ends with colon + bold + not a question
        if (text.endswith(':') and 
            line.get('is_bold', False) and 
            not text.endswith('?')):
            return SectionCandidate(
                title=text,
                index=index,
                confidence="medium",
                font_size=line.get('font_size', 10),
                is_bold=line.get('is_bold', False),
                y_position=line.get('y_position', 0),
                page_number=line.get('page_number', 1),
                reason="ends_with_colon_bold",
                style={"pattern": "colon_bold", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
            )
        
        # Rule 3: Hierarchical numbering starting section (e.g., "1. Overview" where 2., 3. follow)
        hierarchical_match = re.match(r'^(\d+)\.\s+[A-Z]', text)
        if hierarchical_match:
            # Check if this looks like a section starter (not a question)
            if not text.endswith('?') and len(text.split()) >= 2:
                return SectionCandidate(
                    title=text,
                    index=index,
                    confidence="medium",
                    font_size=line.get('font_size', 10),
                    is_bold=line.get('is_bold', False),
                    y_position=line.get('y_position', 0),
                    page_number=line.get('page_number', 1),
                    reason="hierarchical_numbering",
                    style={"pattern": "hierarchical", "font_size": line.get('font_size', 10), "bold": line.get('is_bold', False)}
                )
        
        return None
    
    def _apply_style_consistency_check(self, lines: List[Dict[str, Any]], sections: List[SectionCandidate]) -> List[SectionCandidate]:
        """Apply style consistency check once first section is found."""
        if not sections:
            return sections
        
        logger.debug("Applying style consistency check")
        
        # Get style profile from first high-confidence section
        first_section = sections[0]
        if first_section.confidence == "high":
            self.section_style_profile = {
                'font_size': first_section.font_size,
                'is_bold': first_section.is_bold,
                'pattern': first_section.style.get('pattern', 'unknown')
            }
            
            # Find other lines with similar styling
            for i, line in enumerate(lines):
                if i == first_section.index:
                    continue
                
                text = line.get('text', '').strip()
                if (text in self.excluded_headers or 
                    text in self.page_headers or
                    len(text.split()) < 2 or 
                    len(text.split()) > 15):
                    continue
                
                font_size = line.get('font_size', 10)
                is_bold = line.get('is_bold', False)
                
                # Check if style matches (±1pt font, same bold status)
                if (abs(font_size - self.section_style_profile['font_size']) <= 1.0 and
                    is_bold == self.section_style_profile['is_bold']):
                    
                    # Check if not already a section
                    if not any(s.index == i for s in sections):
                        sections.append(SectionCandidate(
                            title=text,
                            index=i,
                            confidence="medium",
                            font_size=font_size,
                            is_bold=is_bold,
                            y_position=line.get('y_position', 0),
                            page_number=line.get('page_number', 1),
                            reason="style_consistency",
                            style={"pattern": "style_match", "font_size": font_size, "bold": is_bold}
                        ))
        
        # Sort by index again
        sections.sort(key=lambda s: s.index)
        return sections
    
    def _handle_numbered_lists_vs_sections(self, lines: List[Dict[str, Any]], sections: List[SectionCandidate]) -> List[SectionCandidate]:
        """Handle numbered lists vs sections logic."""
        logger.debug("Handling numbered lists vs sections")
        
        # This is a simplified version - in practice, you'd want more sophisticated logic
        # to determine if numbered items are questions under a section or standalone sections
        
        return sections


def detect_sections(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to detect sections from PDF lines.
    
    Args:
        lines: List of line dictionaries with text and formatting info
        
    Returns:
        Dictionary with sections, excluded headers, and debug info
    """
    detector = SectionDetector()
    sections = detector.detect_sections(lines)
    
    # Convert to dictionary format
    sections_list = []
    for section in sections:
        sections_list.append({
            "title": section.title,
            "line_index": section.index,
            "confidence": section.confidence,
            "style": section.style,
            "parent_section": section.parent_section,
            "page_number": section.page_number,
            "y_position": section.y_position,
            "reason": section.reason
        })
    
    return {
        "sections": sections_list,
        "excluded_headers": list(detector.excluded_headers),
        "debug_info": detector.debug_info
    }

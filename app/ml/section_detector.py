"""
Intelligent document section detection with multi-signal analysis.

This module identifies document sections using:
- Font characteristics and formatting
- Structural patterns and whitespace
- Semantic similarity
- Position-based heuristics
- Hierarchical relationships
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.text_element import TextElement


logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a document section with hierarchical structure."""
    title: str
    level: int  # 1 = H1, 2 = H2, etc.
    children: List['Section'] = field(default_factory=list)
    content: List[TextElement] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.0
    
    # Metadata
    number: Optional[str] = None  # e.g., "1.1", "A.", "I."
    font_size: Optional[float] = None
    is_bold: bool = False
    is_uppercase: bool = False
    page_number: int = 1
    
    def __repr__(self) -> str:
        """String representation."""
        num_prefix = f"{self.number} " if self.number else ""
        return (
            f"Section(level={self.level}, title='{num_prefix}{self.title[:40]}...', "
            f"children={len(self.children)}, content={len(self.content)}, "
            f"confidence={self.confidence:.2f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'level': self.level,
            'number': self.number,
            'children': [child.to_dict() for child in self.children],
            'content_count': len(self.content),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'confidence': self.confidence,
            'font_size': self.font_size,
            'is_bold': self.is_bold,
            'is_uppercase': self.is_uppercase,
            'page_number': self.page_number,
        }


class SectionDetector:
    """
    Intelligent section detector using multi-signal analysis.
    
    Combines font analysis, structural patterns, semantic similarity,
    and position heuristics to identify document sections.
    """
    
    # Numbering patterns
    NUMBERING_PATTERNS = [
        # Arabic numerals: 1., 1.1, 1.1.1
        re.compile(r'^(\d+(?:\.\d+)*)\.\s+'),
        # Roman numerals: I., II., III., IV.
        re.compile(r'^([IVXLCDM]+)\.\s+', re.IGNORECASE),
        # Letters: A., B., C. or a., b., c.
        re.compile(r'^([A-Za-z])\.\s+'),
        # Parenthetical: (1), (a), (i)
        re.compile(r'^\(([0-9a-zA-Z]+)\)\s+'),
        # Section prefix: Section 1, Chapter 1
        re.compile(r'^(?:Section|Chapter|Part|Article)\s+(\d+(?:\.\d+)*)[:\.\s]+', re.IGNORECASE),
    ]
    
    # Known section headings for semantic matching
    KNOWN_SECTION_HEADINGS = [
        "Introduction", "Background", "Overview", "Summary",
        "Methodology", "Methods", "Approach", "Implementation",
        "Results", "Findings", "Analysis", "Discussion",
        "Conclusion", "Recommendations", "Future Work",
        "References", "Bibliography", "Appendix",
        "Abstract", "Executive Summary", "Table of Contents",
        "Objectives", "Goals", "Requirements", "Specifications",
    ]
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.7,
        max_title_length: int = 100,
        debug: bool = False
    ):
        """
        Initialize section detector.
        
        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Minimum similarity for semantic matching
            max_title_length: Maximum length for section titles
            debug: Enable debug output
        """
        self.similarity_threshold = similarity_threshold
        self.max_title_length = max_title_length
        self.debug = debug
        
        # Load sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Precompute embeddings for known section headings
        self.section_embeddings = self.model.encode(
            self.KNOWN_SECTION_HEADINGS,
            show_progress_bar=False
        )
        
        logger.info("SectionDetector initialized")
    
    def detect_sections(
        self, 
        elements: List[TextElement]
    ) -> List[Section]:
        """
        Detect sections in document with hierarchical structure.
        
        Args:
            elements: List of TextElements from document
            
        Returns:
            List of top-level Section objects with nested children
        """
        logger.info(f"Detecting sections in {len(elements)} elements")
        
        if not elements:
            return []
        
        # Step 1: Calculate page statistics for context
        page_stats = self._calculate_page_stats(elements)
        
        # Step 2: Score each element as potential section heading
        candidates = self._identify_candidates(elements, page_stats)
        
        if self.debug:
            logger.debug(f"Found {len(candidates)} section candidates")
        
        # Step 3: Filter and validate candidates
        sections = self._validate_candidates(candidates, elements)
        
        if self.debug:
            logger.debug(f"Validated {len(sections)} sections")
        
        # Step 4: Build hierarchical structure
        hierarchy = self._build_hierarchy(sections)
        
        # Step 5: Assign content to sections
        self._assign_content(hierarchy, elements)
        
        logger.info(f"Detected {len(hierarchy)} top-level sections")
        
        return hierarchy
    
    def _calculate_page_stats(
        self, 
        elements: List[TextElement]
    ) -> Dict[int, Dict[str, Any]]:
        """Calculate statistics for each page."""
        stats = defaultdict(lambda: {
            'font_sizes': [],
            'avg_font_size': 12.0,
            'max_font_size': 12.0,
            'total_elements': 0
        })
        
        for elem in elements:
            page = elem.page_number
            stats[page]['total_elements'] += 1
            
            if elem.font_size:
                stats[page]['font_sizes'].append(elem.font_size)
        
        # Calculate averages
        for page, page_stats in stats.items():
            if page_stats['font_sizes']:
                page_stats['avg_font_size'] = np.mean(page_stats['font_sizes'])
                page_stats['max_font_size'] = np.max(page_stats['font_sizes'])
        
        return dict(stats)
    
    def _identify_candidates(
        self, 
        elements: List[TextElement],
        page_stats: Dict[int, Dict[str, Any]]
    ) -> List[Tuple[TextElement, float, int]]:
        """
        Identify potential section headings with confidence scores.
        
        Returns:
            List of (element, confidence, level) tuples
        """
        candidates = []
        
        for i, elem in enumerate(elements):
            text = elem.text.strip()
            
            # Skip empty or very long text
            if not text or len(text) > self.max_title_length:
                continue
            
            # Calculate multi-signal confidence score
            signals = self._analyze_signals(elem, i, elements, page_stats)
            confidence = self._compute_confidence(signals)
            
            # Determine hierarchy level
            level = self._determine_level(elem, signals, page_stats)
            
            if confidence > 0.3:  # Minimum threshold
                candidates.append((elem, confidence, level))
                
                if self.debug:
                    logger.debug(
                        f"Candidate: '{text[:50]}...' "
                        f"confidence={confidence:.2f}, level={level}"
                    )
        
        return candidates
    
    def _analyze_signals(
        self,
        elem: TextElement,
        index: int,
        elements: List[TextElement],
        page_stats: Dict[int, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze multiple signals for section detection."""
        signals = {}
        text = elem.text.strip()
        
        # 1. Font characteristics
        signals['font_signal'] = self._analyze_font(elem, page_stats)
        
        # 2. Structural patterns
        signals['structure_signal'] = self._analyze_structure(elem, index, elements)
        
        # 3. Semantic similarity
        signals['semantic_signal'] = self._analyze_semantics(text)
        
        # 4. Position heuristics
        signals['position_signal'] = self._analyze_position(elem, index, elements)
        
        # 5. Numbering patterns
        signals['numbering_signal'] = self._analyze_numbering(text)
        
        # 6. Formatting patterns
        signals['format_signal'] = self._analyze_formatting(elem, text)
        
        return signals
    
    def _analyze_font(
        self,
        elem: TextElement,
        page_stats: Dict[int, Dict[str, Any]]
    ) -> float:
        """Analyze font characteristics."""
        score = 0.0
        
        page = elem.page_number
        avg_font = page_stats.get(page, {}).get('avg_font_size', 12.0)
        
        if elem.font_size:
            # Larger than average = potential section
            font_ratio = elem.font_size / avg_font if avg_font > 0 else 1.0
            
            if font_ratio > 1.5:
                score += 0.8
            elif font_ratio > 1.2:
                score += 0.5
            elif font_ratio > 1.0:
                score += 0.2
        
        # Bold text is strong indicator
        if elem.is_bold:
            score += 0.6
        
        # Italic less likely to be section
        if elem.is_italic and not elem.is_bold:
            score -= 0.3
        
        return min(score, 1.0)
    
    def _analyze_structure(
        self,
        elem: TextElement,
        index: int,
        elements: List[TextElement]
    ) -> float:
        """Analyze structural patterns."""
        score = 0.0
        
        # Check if standalone (whitespace before and after)
        spacing_before = elem.vertical_spacing or 0.0
        
        # Look ahead for spacing after
        spacing_after = 0.0
        if index + 1 < len(elements):
            next_elem = elements[index + 1]
            if next_elem.page_number == elem.page_number:
                spacing_after = next_elem.vertical_spacing or 0.0
        
        # Standalone line with spacing
        if spacing_before > 10.0 and spacing_after > 5.0:
            score += 0.6
        elif spacing_before > 5.0:
            score += 0.3
        
        # First element on page
        if elem.line_number == 0:
            score += 0.2
        
        # Left-aligned with no indentation
        if elem.indentation_level == 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_semantics(self, text: str) -> float:
        """Analyze semantic similarity to known sections."""
        # Encode the text
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]
        
        # Calculate cosine similarities
        similarities = np.dot(self.section_embeddings, text_embedding) / (
            np.linalg.norm(self.section_embeddings, axis=1) * 
            np.linalg.norm(text_embedding)
        )
        
        max_similarity = float(np.max(similarities))
        
        if max_similarity > self.similarity_threshold:
            return 1.0
        elif max_similarity > 0.5:
            return max_similarity
        else:
            return 0.0
    
    def _analyze_position(
        self,
        elem: TextElement,
        index: int,
        elements: List[TextElement]
    ) -> float:
        """Analyze position-based heuristics."""
        score = 0.0
        
        # Top of page
        if elem.bbox.y0 < 100:  # Near top
            score += 0.3
        
        # Left-aligned
        if elem.bbox.x0 < 100:  # Near left edge
            score += 0.2
        
        # Followed by multiple elements
        same_page_after = sum(
            1 for e in elements[index+1:index+6]
            if e.page_number == elem.page_number
        )
        
        if same_page_after >= 3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_numbering(self, text: str) -> float:
        """Analyze numbering patterns."""
        for pattern in self.NUMBERING_PATTERNS:
            if pattern.match(text):
                return 0.8
        
        return 0.0
    
    def _analyze_formatting(self, elem: TextElement, text: str) -> float:
        """Analyze formatting patterns."""
        score = 0.0
        
        # All uppercase (but not too long)
        if text.isupper() and len(text) < 50:
            score += 0.5
        
        # Title case
        if text.istitle():
            score += 0.3
        
        # Doesn't end with period (sections usually don't)
        if not text.endswith('.'):
            score += 0.2
        
        # Short text more likely to be section
        if len(text) < 50:
            score += 0.2
        elif len(text) < 30:
            score += 0.3
        
        # Contains keywords
        keywords = [
            'introduction', 'background', 'method', 'result', 
            'conclusion', 'discussion', 'abstract', 'summary',
            'overview', 'analysis', 'chapter', 'section', 'part'
        ]
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                score += 0.4
                break
        
        return min(score, 1.0)
    
    def _compute_confidence(self, signals: Dict[str, float]) -> float:
        """Compute overall confidence from multiple signals."""
        # Weighted combination of signals
        weights = {
            'font_signal': 0.25,
            'structure_signal': 0.20,
            'semantic_signal': 0.20,
            'position_signal': 0.15,
            'numbering_signal': 0.10,
            'format_signal': 0.10,
        }
        
        confidence = sum(
            signals.get(signal, 0.0) * weight
            for signal, weight in weights.items()
        )
        
        return min(confidence, 1.0)
    
    def _determine_level(
        self,
        elem: TextElement,
        signals: Dict[str, float],
        page_stats: Dict[int, Dict[str, Any]]
    ) -> int:
        """Determine hierarchy level (1=H1, 2=H2, etc.)."""
        # Base level on font size
        page = elem.page_number
        avg_font = page_stats.get(page, {}).get('avg_font_size', 12.0)
        max_font = page_stats.get(page, {}).get('max_font_size', 14.0)
        
        if elem.font_size:
            font_ratio = elem.font_size / avg_font if avg_font > 0 else 1.0
            
            if font_ratio >= 1.5 or elem.font_size >= max_font * 0.95:
                level = 1
            elif font_ratio >= 1.3:
                level = 2
            elif font_ratio >= 1.1:
                level = 3
            else:
                level = 4
        else:
            level = 3
        
        # Check numbering for level hints
        text = elem.text.strip()
        for pattern in self.NUMBERING_PATTERNS[:1]:  # Check first pattern
            match = pattern.match(text)
            if match:
                number = match.group(1)
                # Count dots to determine depth: 1 = level 1, 1.1 = level 2
                dot_count = number.count('.')
                level = min(dot_count + 1, level)
                break
        
        return max(1, min(level, 6))  # Clamp between 1-6
    
    def _validate_candidates(
        self,
        candidates: List[Tuple[TextElement, float, int]],
        elements: List[TextElement]
    ) -> List[Section]:
        """Validate and filter candidates to prevent false positives."""
        sections = []
        
        for elem, confidence, level in candidates:
            text = elem.text.strip()
            
            # Skip if confidence too low
            if confidence < 0.4:
                continue
            
            # Validate it's not a bullet point
            if text.startswith(('•', '◦', '▪', '-', '*')) and len(text) < 100:
                if self.debug:
                    logger.debug(f"Rejected (bullet): {text[:50]}")
                continue
            
            # Validate it's not just emphasis
            if elem.is_italic and not elem.is_bold and confidence < 0.6:
                if self.debug:
                    logger.debug(f"Rejected (emphasis): {text[:50]}")
                continue
            
            # Validate length
            if len(text) < 3:
                if self.debug:
                    logger.debug(f"Rejected (too short): {text}")
                continue
            
            # Extract numbering if present
            number = self._extract_number(text)
            title = self._normalize_title(text, number)
            
            section = Section(
                title=title,
                level=level,
                confidence=confidence,
                number=number,
                font_size=elem.font_size,
                is_bold=elem.is_bold,
                is_uppercase=text.isupper(),
                page_number=elem.page_number,
                start_pos=elements.index(elem),
            )
            
            sections.append(section)
        
        return sections
    
    def _extract_number(self, text: str) -> Optional[str]:
        """Extract section number from text."""
        for pattern in self.NUMBERING_PATTERNS:
            match = pattern.match(text)
            if match:
                return match.group(1)
        return None
    
    def _normalize_title(self, text: str, number: Optional[str]) -> str:
        """Normalize section title."""
        # Remove numbering prefix if present
        if number:
            for pattern in self.NUMBERING_PATTERNS:
                text = pattern.sub('', text)
        
        # Trim whitespace
        text = text.strip()
        
        # Optional: Convert to title case if all uppercase
        # (commented out to preserve original formatting)
        # if text.isupper() and len(text) > 5:
        #     text = text.title()
        
        return text
    
    def _build_hierarchy(self, sections: List[Section]) -> List[Section]:
        """Build hierarchical structure from flat section list."""
        if not sections:
            return []
        
        # Sort by start position
        sections.sort(key=lambda s: s.start_pos)
        
        # Build tree structure
        root_sections = []
        stack = []  # Stack of (section, level) tuples
        
        for section in sections:
            # Pop stack until we find parent level
            while stack and stack[-1].level >= section.level:
                stack.pop()
            
            if stack:
                # Add as child of top of stack
                parent = stack[-1]
                parent.children.append(section)
            else:
                # Top-level section
                root_sections.append(section)
            
            stack.append(section)
        
        return root_sections
    
    def _assign_content(
        self,
        sections: List[Section],
        elements: List[TextElement]
    ) -> None:
        """Assign content elements to sections recursively."""
        if not sections:
            return
        
        # Flatten sections for position lookup
        flat_sections = self._flatten_sections(sections)
        
        if not flat_sections:
            return
        
        # Sort by start position
        flat_sections.sort(key=lambda s: s.start_pos)
        
        # Set end positions
        for i, section in enumerate(flat_sections):
            if i + 1 < len(flat_sections):
                section.end_pos = flat_sections[i + 1].start_pos
            else:
                section.end_pos = len(elements)
        
        # Assign content
        for section in flat_sections:
            section.content = [
                elem for i, elem in enumerate(elements)
                if section.start_pos < i < section.end_pos
            ]
    
    def _flatten_sections(self, sections: List[Section]) -> List[Section]:
        """Flatten hierarchical sections into list."""
        result = []
        for section in sections:
            result.append(section)
            if section.children:
                result.extend(self._flatten_sections(section.children))
        return result
    
    def visualize_sections(
        self,
        sections: List[Section],
        indent: int = 0
    ) -> str:
        """
        Generate visual representation of section hierarchy.
        
        Args:
            sections: List of sections to visualize
            indent: Current indentation level
            
        Returns:
            Formatted string representation
        """
        lines = []
        
        for section in sections:
            prefix = "  " * indent
            num_str = f"{section.number} " if section.number else ""
            conf_str = f"[{section.confidence:.2f}]"
            content_str = f"({len(section.content)} elements)"
            
            line = f"{prefix}{'H' + str(section.level)}: {num_str}{section.title} {conf_str} {content_str}"
            lines.append(line)
            
            # Recursively visualize children
            if section.children:
                child_vis = self.visualize_sections(section.children, indent + 1)
                lines.append(child_vis)
        
        return "\n".join(lines)


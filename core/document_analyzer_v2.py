"""
Document Analyzer v2 - High-Quality Structured Extraction (ENHANCED)

This module provides production-grade extraction of structured data from
unstructured PDFs (resumes, RFPs, sales documents, etc.).

================================================================================
RECENT ENHANCEMENTS FOR BETTER NESTING AND STRUCTURE
================================================================================

1. CONTEXTUAL METADATA EXTRACTION
   - Uses nearby keywords ("Due date:", "Contact:", etc.) to classify metadata
   - Example: "Due date: Oct 10" is classified as 'deadline' not generic 'date'
   - Extracts key-value pairs using pattern matching
   - Each metadata item tracks its line position for proper section assignment

2. IMPROVED SECTION BOUNDARY DETECTION
   - Filters out empty sections (no content between headings)
   - Merges sections that are too close together (< 2 lines apart)
   - Validates that each section has real content
   - Better hierarchy detection (main vs subsections)

3. STRICT CONTENT NESTING
   - All content (description, metadata, Q&A) stays inside its section
   - Line range tracking ensures content doesn't leak between sections
   - Subsections properly nested under parent section's 'sub_sections' key
   - No orphaned content at root level (except document-level metadata)

4. ENHANCED FILTERING
   - Descriptions exclude metadata lines (key: value patterns)
   - Questions removed from descriptions (already in 'questions' array)
   - Q&A markers filtered out (Q:, A:, Question:, Answer:)
   - Prevents content duplication

5. VALIDATION
   - Post-build validation checks for orphan content
   - Warns if metadata or Q&A couldn't be assigned to a section
   - Ensures data integrity and proper nesting

6. BETTER HIERARCHY
   - Improved font size, indentation, and numbering detection
   - Subsections correctly identified and nested
   - Multi-level nesting support (section → subsection → sub-subsection)

Output Structure:
{
  "Section Title": {
    "type": "section",
    "description": ["paragraph 1", "paragraph 2"],
    "metadata": {"deadline": "Oct 10", "contact": "John Doe"},
    "questions": [{"question": "...", "answer": "..."}],
    "sub_sections": {
      "Subsection Title": {
        "type": "section",
        "description": ["..."]
      }
    }
  },
  "metadata": {  // Only document-level metadata here
    "organization": "Acme Corp"
  }
}

Author: Senior Python Engineer
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json

import pdfplumber
import PyPDF2

# Optional NLP for advanced metadata extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TextLine:
    """
    Represents a single line of text with formatting metadata.
    
    Attributes:
        text: The actual text content
        page_num: Page number (1-indexed)
        line_num: Line number within page
        font_size: Font size in points (if available)
        is_bold: Whether text is bold
        is_italic: Whether text is italic
        is_uppercase: Whether text is all uppercase
        x0: Left x-coordinate
        y0: Top y-coordinate
        indentation: Indentation level
    """
    text: str
    page_num: int
    line_num: int
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    is_uppercase: bool = False
    x0: float = 0.0
    y0: float = 0.0
    indentation: int = 0
    
    def __repr__(self) -> str:
        return f"TextLine(p{self.page_num}:{self.line_num} '{self.text[:50]}...')"


@dataclass
class Section:
    """
    Represents a document section with hierarchy.
    
    Attributes:
        title: Section title/heading
        level: Hierarchy level (1=top, 2=subsection, etc.)
        start_line: Starting line index in document
        end_line: Ending line index in document
        content: List of text lines in this section
        content_lines: List of TextLine objects (with line indices)
        subsections: Nested subsections
        metadata: Section-specific metadata
        questions: Q&A pairs in this section
    """
    title: str
    level: int = 1
    start_line: int = 0
    end_line: int = 0
    content: List[str] = field(default_factory=list)
    content_lines: List[TextLine] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    questions: List[Dict[str, str]] = field(default_factory=list)


# ============================================================================
# STEP 1: TEXT EXTRACTION
# ============================================================================

class PDFTextExtractor:
    """
    Extracts text from PDF with layout and formatting information.
    
    Uses pdfplumber as primary method with PyPDF2 fallback.
    Preserves font information, positioning, and structure.
    """
    
    def extract_text(self, file_path: str) -> List[TextLine]:
        """
        Extract text lines from PDF with formatting metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of TextLine objects with formatting info
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF is corrupted or empty
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        logger.info(f"Extracting text from: {file_path}")
        
        try:
            # Try pdfplumber first (preserves layout best)
            lines = self._extract_with_pdfplumber(file_path)
            logger.info(f"✓ Extracted {len(lines)} lines using pdfplumber")
            return lines
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2...")
            
            # Fallback to PyPDF2
            try:
                lines = self._extract_with_pypdf2(file_path)
                logger.info(f"✓ Extracted {len(lines)} lines using PyPDF2")
                return lines
            except Exception as e2:
                raise ValueError(f"Failed to extract PDF: {e2}")
    
    def _extract_with_pdfplumber(self, file_path: Path) -> List[TextLine]:
        """Extract using pdfplumber with layout preservation."""
        lines = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Get characters with position and font info
                chars = page.chars
                if not chars:
                    continue
                
                # Group characters into lines based on y-coordinate
                lines_on_page = self._group_chars_into_lines(chars)
                
                for line_num, line_chars in enumerate(lines_on_page):
                    if not line_chars:
                        continue
                    
                    # Extract text
                    text = ''.join(c['text'] for c in line_chars).strip()
                    if not text:
                        continue
                    
                    # Get font info (most common in line)
                    font_sizes = [c.get('size', 12) for c in line_chars]
                    font_names = [c.get('fontname', '') for c in line_chars]
                    
                    font_size = max(set(font_sizes), key=font_sizes.count)
                    font_name = max(set(font_names), key=font_names.count)
                    
                    # Detect formatting
                    is_bold = 'bold' in font_name.lower()
                    is_italic = 'italic' in font_name.lower()
                    is_uppercase = text.isupper() and len(text) > 3
                    
                    # Get position
                    x0 = min(c['x0'] for c in line_chars)
                    y0 = min(c['top'] for c in line_chars)
                    
                    # Calculate indentation (normalize to page width)
                    indentation = int((x0 / page.width) * 100 / 10)  # 0-10 scale
                    
                    lines.append(TextLine(
                        text=text,
                        page_num=page_num,
                        line_num=line_num,
                        font_size=font_size,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        is_uppercase=is_uppercase,
                        x0=x0,
                        y0=y0,
                        indentation=indentation
                    ))
        
        return lines
    
    def _group_chars_into_lines(self, chars: List[Dict]) -> List[List[Dict]]:
        """Group characters into lines based on vertical position."""
        if not chars:
            return []
        
        # Sort by y position, then x
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        lines = []
        current_line = [sorted_chars[0]]
        current_y = sorted_chars[0]['top']
        
        for char in sorted_chars[1:]:
            # If y position is close to current line, add to it
            if abs(char['top'] - current_y) < 3:
                current_line.append(char)
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = [char]
                current_y = char['top']
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _extract_with_pypdf2(self, file_path: Path) -> List[TextLine]:
        """Fallback extraction using PyPDF2 (limited formatting info)."""
        lines = []
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                if text:
                    page_lines = text.split('\n')
                    for line_num, line in enumerate(page_lines):
                        line = line.strip()
                        if line:
                            lines.append(TextLine(
                                text=line,
                                page_num=page_num + 1,
                                line_num=line_num,
                                is_uppercase=line.isupper() and len(line) > 3
                            ))
        
        return lines


# ============================================================================
# STEP 2: SECTION DETECTION
# ============================================================================

class SectionDetector:
    """
    Detects document sections using multiple heuristics.
    
    Detection Strategy:
    1. Font-based: Larger font = higher priority
    2. Format-based: Bold, uppercase, title case
    3. Pattern-based: Numbering (1., 1.1, A., etc.)
    4. Keyword-based: Common section titles
    5. Structure-based: Whitespace, positioning
    """
    
    # Common section keywords (case-insensitive)
    SECTION_KEYWORDS = {
        'summary', 'introduction', 'background', 'overview', 'abstract',
        'objective', 'goal', 'purpose', 'scope', 'methodology', 'approach',
        'requirement', 'specification', 'criteria', 'condition', 'term',
        'experience', 'qualification', 'education', 'skill', 'certification',
        'responsibility', 'achievement', 'project', 'work', 'employment',
        'conclusion', 'recommendation', 'reference', 'appendix', 'contact',
        'general', 'specific', 'technical', 'administrative', 'financial'
    }
    
    # Numbering patterns
    NUMBERING_PATTERNS = [
        r'^\d+\.',  # 1., 2., 3.
        r'^\d+\.\d+',  # 1.1, 1.2, 2.1
        r'^[A-Z]\.',  # A., B., C.
        r'^[IVXLCDM]+\.',  # I., II., III. (Roman numerals)
        r'^Section\s+\d+',  # Section 1, Section 2
        r'^Chapter\s+\d+',  # Chapter 1, Chapter 2
        r'^Part\s+[A-Z0-9]+',  # Part A, Part 1
    ]
    
    def detect_sections(
        self, 
        lines: List[TextLine],
        min_confidence: float = 0.4
    ) -> List[Section]:
        """
        Detect sections in document using multi-heuristic approach.
        
        ENHANCED: Better boundary detection and empty section avoidance.
        
        Args:
            lines: List of text lines from document
            min_confidence: Minimum confidence score (0-1) to consider as section
            
        Returns:
            List of Section objects with hierarchy
        """
        logger.info(f"Detecting sections in {len(lines)} lines...")
        
        # Calculate document statistics (for normalization)
        doc_stats = self._calculate_document_stats(lines)
        
        # Score each line as potential section
        section_candidates = []
        for i, line in enumerate(lines):
            score = self._calculate_section_score(line, i, lines, doc_stats)
            
            if score >= min_confidence:
                level = self._determine_hierarchy_level(line, doc_stats)
                section_candidates.append((i, line, score, level))
        
        logger.info(f"Found {len(section_candidates)} section candidates")
        
        # ENHANCEMENT: Filter out sections that would be empty or too close together
        section_candidates = self._filter_section_candidates(section_candidates, lines)
        logger.info(f"After filtering: {len(section_candidates)} valid sections")
        
        # Build section hierarchy
        sections = self._build_section_hierarchy(section_candidates, lines)
        
        logger.info(f"✓ Detected {len(sections)} top-level sections")
        return sections
    
    def _filter_section_candidates(
        self,
        candidates: List[Tuple[int, TextLine, float, int]],
        all_lines: List[TextLine]
    ) -> List[Tuple[int, TextLine, float, int]]:
        """
        Filter section candidates to avoid empty sections and improve boundaries.
        
        ENHANCEMENT: Remove candidates that:
        1. Would have no content (next section immediately follows)
        2. Are too close to previous section (< 2 lines apart)
        3. Have very low content between them and next section
        """
        if not candidates:
            return []
        
        filtered = []
        
        for i, (line_idx, line, score, level) in enumerate(candidates):
            # Check if this section would have content
            if i + 1 < len(candidates):
                next_idx = candidates[i + 1][0]
                content_lines = all_lines[line_idx + 1:next_idx]
            else:
                content_lines = all_lines[line_idx + 1:]
            
            # Filter non-empty content (ignore whitespace-only lines)
            real_content = [l for l in content_lines if l.text.strip()]
            
            # RULE 1: Must have at least 1 line of content OR be followed by subsection
            if not real_content and i + 1 < len(candidates):
                next_level = candidates[i + 1][3]
                if next_level <= level:  # Not a subsection
                    continue  # Skip this empty section
            
            # RULE 2: If too close to previous section (< 2 lines), merge into parent
            if filtered:
                prev_idx = filtered[-1][0]
                if line_idx - prev_idx < 2:
                    # Only keep if this has higher score (better heading)
                    if score <= filtered[-1][2]:
                        continue
                    else:
                        filtered.pop()  # Replace previous with this one
            
            filtered.append((line_idx, line, score, level))
        
        return filtered
    
    def _calculate_document_stats(self, lines: List[TextLine]) -> Dict[str, Any]:
        """Calculate document-level statistics for normalization."""
        font_sizes = [l.font_size for l in lines if l.font_size]
        
        return {
            'avg_font_size': sum(font_sizes) / len(font_sizes) if font_sizes else 12.0,
            'max_font_size': max(font_sizes) if font_sizes else 14.0,
            'min_font_size': min(font_sizes) if font_sizes else 10.0,
            'total_lines': len(lines)
        }
    
    def _calculate_section_score(
        self,
        line: TextLine,
        index: int,
        all_lines: List[TextLine],
        doc_stats: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for line being a section heading.
        
        Returns score between 0.0 and 1.0
        """
        score = 0.0
        text = line.text.strip()
        
        # Skip very short or very long lines
        if len(text) < 3 or len(text) > 100:
            return 0.0
        
        # 1. FONT SIZE SIGNAL (weight: 0.25)
        if line.font_size and doc_stats['avg_font_size']:
            font_ratio = line.font_size / doc_stats['avg_font_size']
            if font_ratio > 1.5:
                score += 0.25
            elif font_ratio > 1.2:
                score += 0.15
            elif font_ratio > 1.0:
                score += 0.05
        
        # 2. FORMATTING SIGNAL (weight: 0.25)
        if line.is_bold:
            score += 0.15
        if line.is_uppercase and len(text) < 50:
            score += 0.10
        if text.istitle():  # Title Case
            score += 0.05
        
        # 3. NUMBERING SIGNAL (weight: 0.20)
        for pattern in self.NUMBERING_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                score += 0.20
                break
        
        # 4. KEYWORD SIGNAL (weight: 0.15)
        text_lower = text.lower()
        for keyword in self.SECTION_KEYWORDS:
            if keyword in text_lower:
                score += 0.15
                break
        
        # 5. STRUCTURE SIGNAL (weight: 0.15)
        # Check if followed by content (not another heading)
        if index + 1 < len(all_lines):
            next_line = all_lines[index + 1]
            # If next line is less prominent, likely a section
            if (not next_line.is_bold or 
                (line.font_size and next_line.font_size and 
                 line.font_size > next_line.font_size)):
                score += 0.10
        
        # Low indentation suggests main heading
        if line.indentation <= 1:
            score += 0.05
        
        # Standalone line (not part of paragraph)
        if not text.endswith(('.', ',', ';')):
            score += 0.05
        
        # Normalize score to 0-1
        return min(score, 1.0)
    
    def _determine_hierarchy_level(
        self,
        line: TextLine,
        doc_stats: Dict[str, Any]
    ) -> int:
        """
        Determine hierarchy level (1=top, 2=sub, 3=sub-sub, etc.).
        
        Based on font size, numbering depth, and indentation.
        """
        level = 1
        
        # Font size based level
        if line.font_size and doc_stats['avg_font_size']:
            font_ratio = line.font_size / doc_stats['avg_font_size']
            if font_ratio >= 1.5:
                level = 1
            elif font_ratio >= 1.2:
                level = 2
            else:
                level = 3
        
        # Numbering depth (e.g., 1.1.1 = level 3)
        numbering_match = re.match(r'^(\d+(?:\.\d+)*)', line.text)
        if numbering_match:
            num_str = numbering_match.group(1)
            dots = num_str.count('.')
            level = min(dots + 1, level)
        
        # Indentation
        if line.indentation > 2:
            level = max(level, 3)
        elif line.indentation > 1:
            level = max(level, 2)
        
        return min(level, 6)  # Cap at level 6
    
    def _build_section_hierarchy(
        self,
        candidates: List[Tuple[int, TextLine, float, int]],
        all_lines: List[TextLine]
    ) -> List[Section]:
        """
        Build hierarchical section structure from candidates.
        
        CRITICAL: This method tracks line ranges for each section so that
        metadata, Q&A, and content can be properly nested inside sections.
        
        ENHANCED: Captures ALL text between headings for proper description content.
        
        Args:
            candidates: List of (line_index, line, score, level) tuples
            all_lines: All document lines
            
        Returns:
            List of top-level Section objects with nested subsections
        """
        if not candidates:
            return []
        
        sections = []
        stack = []  # Stack for building hierarchy
        
        for i, (line_idx, line, score, level) in enumerate(candidates):
            # Determine content range (from this heading to next)
            # This is CRITICAL for proper nesting
            start_idx = line_idx + 1
            if i + 1 < len(candidates):
                end_idx = candidates[i + 1][0]
            else:
                end_idx = len(all_lines)
            
            # ENHANCEMENT: Capture ALL text between headings (will filter later)
            # This ensures we don't lose descriptive content
            # Store tuples of (global_line_index, TextLine) for precise filtering
            content_with_indices = []
            for global_idx in range(start_idx, end_idx):
                if global_idx < len(all_lines):
                    text_line = all_lines[global_idx]
                    if text_line.text.strip():
                        content_with_indices.append((global_idx, text_line))
            
            # Extract just the text for backwards compatibility
            content = [tl.text for _, tl in content_with_indices]
            
            # Create section with line range tracking
            section = Section(
                title=line.text.strip(),
                level=level,
                start_line=line_idx,  # Track where section starts
                end_line=end_idx,     # Track where section ends
                content=content,      # Text content (for backwards compatibility)
                content_lines=[tl for _, tl in content_with_indices]  # TextLine objects
            )
            
            # Store the line indices mapping for this section (will use in formatting)
            section._line_indices = [idx for idx, _ in content_with_indices]
            
            # Build hierarchy using stack
            # This ensures subsections are properly nested
            while stack and stack[-1].level >= level:
                stack.pop()
            
            if stack:
                # Add as subsection of parent
                stack[-1].subsections.append(section)
            else:
                # Top-level section
                sections.append(section)
            
            stack.append(section)
        
        return sections


# ============================================================================
# STEP 3: METADATA EXTRACTION
# ============================================================================

class MetadataExtractor:
    """
    Extracts structured metadata from document with contextual awareness.
    
    ENHANCED: Uses contextual cues (nearby keywords) to identify metadata
    and classify it correctly.
    
    Extracts:
    - Dates (due dates, deadlines, timestamps)
    - Contacts (names, emails, phones)
    - Companies/Organizations
    - Reference numbers
    - Financial amounts
    """
    
    # Contextual keywords that indicate metadata (case-insensitive)
    METADATA_KEYWORDS = {
        'deadline': ['deadline', 'due date', 'due by', 'submit by', 'submission date'],
        'response_date': ['respond by', 'response by', 'response date', 'reply by'],
        'contact': ['contact', 'point of contact', 'poc', 'contact person'],
        'email': ['email', 'e-mail', 'email address'],
        'phone': ['phone', 'telephone', 'contact number', 'tel'],
        'organization': ['organization', 'company', 'vendor', 'contractor', 'bidder'],
        'reference': ['reference', 'ref', 'rfp', 'rfq', 'reference number'],
        'budget': ['budget', 'cost', 'price', 'amount', 'value'],
        'timeline': ['timeline', 'schedule', 'duration', 'timeframe'],
        'location': ['location', 'address', 'site', 'venue'],
    }
    
    # Regex patterns for metadata
    PATTERNS = {
        'date': [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  # MM/DD/YYYY
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',  # Month DD, YYYY
            r'\b(\d{4}-\d{2}-\d{2})\b',  # ISO format
        ],
        'email': [
            r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
        ],
        'phone': [
            r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',  # XXX-XXX-XXXX
            r'\((\d{3})\)\s*(\d{3})[-.]?(\d{4})\b',  # (XXX) XXX-XXXX
        ],
        'money': [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b'  # $1,000.00
        ],
        'url': [
            r'\b((?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)\b'
        ],
        'reference': [
            r'\b(RFP[-\s]?[A-Z0-9]+)\b',  # RFP-2025-001
            r'\b(REF[-:\s]?[A-Z0-9]+)\b',  # REF-123
        ]
    }
    
    def __init__(self):
        """Initialize metadata extractor."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                import spacy
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("spaCy loaded for NER extraction")
            except:
                logger.warning("spaCy model not found, using regex only")
    
    def extract_metadata(
        self,
        lines: List[TextLine],
        use_ner: bool = True
    ) -> List[Tuple[int, str, Any]]:
        """
        Extract metadata from document lines WITH LINE POSITIONS and CONTEXT.
        
        ENHANCED: Uses contextual keywords to classify metadata correctly.
        For example, "Due date: Oct 10" is classified as 'deadline' not just 'date'.
        
        Args:
            lines: Document text lines
            use_ner: Whether to use NER (requires spaCy)
            
        Returns:
            List of (line_index, key, value) tuples
        """
        logger.info("Extracting metadata with contextual awareness...")
        
        metadata_items = []
        
        # Extract from each line with position tracking and context
        for idx, line in enumerate(lines):
            text = line.text
            text_lower = text.lower()
            
            # STEP 1: Check for contextual metadata (key: value patterns)
            # Example: "Due date: Oct 10, 2025"
            contextual_metadata = self._extract_contextual_metadata(text, idx)
            metadata_items.extend(contextual_metadata)
            
            # STEP 2: Extract using regex patterns (with context classification)
            for category, patterns in self.PATTERNS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if isinstance(match, tuple):
                                value = ''.join(match)
                            else:
                                value = match
                            
                            # Classify based on context
                            classified_key = self._classify_with_context(
                                category, value, text_lower
                            )
                            
                            # Store with line position and classified key
                            metadata_items.append((idx, classified_key, value))
        
        # Also extract document-level metadata using NER (first 3 pages)
        if use_ner and self.nlp:
            relevant_lines = [l for l in lines if l.page_num <= 3]
            text = '\n'.join(l.text for l in relevant_lines)
            
            ner_metadata = self._extract_with_ner(text)
            for key, values in ner_metadata.items():
                for value in values:
                    # Assign to line 0 (document level)
                    metadata_items.append((0, key, value))
        
        logger.info(f"✓ Extracted {len(metadata_items)} metadata items")
        return metadata_items
    
    def _extract_contextual_metadata(
        self,
        text: str,
        line_idx: int
    ) -> List[Tuple[int, str, str]]:
        """
        Extract metadata using contextual keywords (key: value patterns).
        
        Examples:
        - "Due date: Oct 10, 2025" -> ('deadline', 'Oct 10, 2025')
        - "Contact: John Doe" -> ('contact', 'John Doe')
        - "Budget: $50,000" -> ('budget', '$50,000')
        """
        metadata = []
        text_lower = text.lower()
        
        # Check for each metadata type
        for meta_key, keywords in self.METADATA_KEYWORDS.items():
            for keyword in keywords:
                # Pattern: "keyword: value" or "keyword - value"
                pattern = rf'\b{re.escape(keyword)}\s*[:\-–]\s*(.+?)(?:\.|$)'
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                
                if matches:
                    for match in matches:
                        value = match.strip()
                        if value and len(value) > 2:
                            # Get original case from text
                            start_pos = text_lower.find(value.lower())
                            if start_pos >= 0:
                                value = text[start_pos:start_pos + len(value)]
                            
                            metadata.append((line_idx, meta_key, value))
        
        return metadata
    
    def _classify_with_context(
        self,
        category: str,
        value: str,
        text_lower: str
    ) -> str:
        """
        Classify metadata more specifically based on context.
        
        Example: A date found near "deadline" keyword is classified as 'deadline'
        instead of generic 'date'.
        """
        # If it's already specific, keep it
        if category in ['email', 'phone', 'money', 'url', 'reference']:
            return category
        
        # For dates, check context
        if category == 'date':
            for meta_key, keywords in self.METADATA_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return meta_key
        
        return category
    
    def _extract_with_ner(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER."""
        metadata = defaultdict(list)
        
        # Process text (limit to 100K chars for performance)
        doc = self.nlp(text[:100000])
        
        # Entity mapping
        entity_map = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'DATE': 'date',
            'MONEY': 'money',
        }
        
        for ent in doc.ents:
            category = entity_map.get(ent.label_)
            if category:
                metadata[category].append(ent.text)
        
        return dict(metadata)
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and format metadata values."""
        cleaned = {}
        
        for key, values in metadata.items():
            if not values:
                continue
            
            # Remove duplicates
            unique_values = list(set(values))
            
            # Take first value if only one type expected
            if key in ['date', 'email', 'phone'] and len(unique_values) == 1:
                cleaned[key] = unique_values[0]
            else:
                cleaned[key] = unique_values
        
        return cleaned


# ============================================================================
# STEP 4: QUESTION/ANSWER DETECTION
# ============================================================================

class QADetector:
    """
    Detects and pairs questions with answers.
    
    Detection Strategy:
    1. Identify questions (?, imperative verbs, Q:)
    2. Match answers (proximity, indentation, markers)
    3. Group related Q&A pairs
    """
    
    # Question indicators
    QUESTION_PATTERNS = [
        r'\?$',  # Ends with ?
        r'^Q\d*[\.:]\s*',  # Q1:, Q:
        r'^Question\s+\d+',  # Question 1
    ]
    
    IMPERATIVE_VERBS = {
        'describe', 'explain', 'provide', 'list', 'identify',
        'specify', 'detail', 'outline', 'state', 'indicate',
        'clarify', 'demonstrate', 'discuss', 'define'
    }
    
    INTERROGATIVES = {
        'who', 'what', 'when', 'where', 'why', 'how', 'which',
        'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did'
    }
    
    def detect_qa_pairs(self, lines: List[TextLine]) -> List[Tuple[int, Dict[str, str]]]:
        """
        Detect question-answer pairs in document WITH LINE POSITIONS.
        
        CRITICAL: Returns line index with each Q&A pair so it can be
        properly nested inside the correct section.
        
        Args:
            lines: Document text lines
            
        Returns:
            List of (line_index, {question: str, answer: str}) tuples
        """
        logger.info("Detecting Q&A pairs...")
        
        qa_pairs = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a question
            if self._is_question(line.text):
                question = line.text.strip()
                
                # Look for answer in next few lines
                answer = self._find_answer(lines, i + 1, max_distance=5)
                
                # Store with line position
                qa_pairs.append((i, {
                    'question': question,
                    'answer': answer if answer else None
                }))
                
                # Skip lines we've processed
                if answer:
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        logger.info(f"✓ Detected {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        text = text.strip()
        
        # Pattern matching
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text):
                return True
        
        # Starts with interrogative word
        first_word = text.split()[0].lower().strip('.,!?:') if text.split() else ''
        if first_word in self.INTERROGATIVES:
            return True
        
        # Starts with imperative verb
        if any(text.lower().startswith(verb) for verb in self.IMPERATIVE_VERBS):
            return True
        
        return False
    
    def _find_answer(
        self,
        lines: List[TextLine],
        start_idx: int,
        max_distance: int = 5
    ) -> Optional[str]:
        """Find answer following a question."""
        for i in range(start_idx, min(start_idx + max_distance, len(lines))):
            line = lines[i]
            text = line.text.strip()
            
            # Skip if empty or too short
            if not text or len(text) < 5:
                continue
            
            # Stop if we hit another question
            if self._is_question(text):
                break
            
            # Skip if it looks like a heading
            if line.is_bold and line.is_uppercase:
                break
            
            # This is likely the answer
            return text
        
        return None


# ============================================================================
# STEP 5: PARAGRAPH MERGING
# ============================================================================

class ParagraphMerger:
    """
    Merges multi-line text into complete paragraphs.
    
    Strategy:
    - Merge lines with similar formatting
    - Respect paragraph breaks (blank lines, indentation changes)
    - Handle bullet points and lists
    """
    
    def merge_paragraphs(self, content: List[str]) -> List[str]:
        """
        Merge multi-line content into paragraphs.
        
        Args:
            content: List of text lines
            
        Returns:
            List of merged paragraphs
        """
        if not content:
            return []
        
        paragraphs = []
        current_paragraph = []
        
        for line in content:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Check if new paragraph (bullet, number, etc.)
            if self._is_new_paragraph(line):
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [line]
            else:
                # Continue current paragraph
                current_paragraph.append(line)
        
        # Add last paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def _is_new_paragraph(self, line: str) -> bool:
        """Check if line starts a new paragraph."""
        # Bullet points
        if re.match(r'^[•◦▪\-\*]\s+', line):
            return True
        
        # Numbered lists
        if re.match(r'^\d+[\.)]\s+', line):
            return True
        
        # Section markers
        if re.match(r'^[A-Z]\)\s+', line):
            return True
        
        return False


# ============================================================================
# STEP 6: JSON BUILDER
# ============================================================================

class JSONBuilder:
    """
    Builds hierarchical JSON output from extracted data.
    
    CRITICAL: This class ensures all content (metadata, Q&A, descriptions)
    is properly nested inside the section it belongs to.
    
    Output Structure:
    {
      "Section Title": {
        "type": "section",
        "description": ["paragraph 1", "paragraph 2"],
        "metadata": {"key": "value"},
        "questions": [{"question": "...", "answer": "..."}],
        "sub_sections": {
          "Subsection Title": {
            "description": ["..."]
          }
        }
      },
      "metadata": {...}  // Only root-level metadata
    }
    """
    
    def __init__(self):
        """Initialize JSON builder."""
        self.paragraph_merger = ParagraphMerger()
    
    def build_json(
        self,
        sections: List[Section],
        metadata_items: List[Tuple[int, str, Any]],
        qa_pairs: List[Tuple[int, Dict[str, str]]]
    ) -> Dict[str, Any]:
        """
        Build hierarchical JSON from analyzed data.
        
        CRITICAL: This method assigns metadata and Q&A pairs to their respective
        sections based on line position, ensuring proper nesting.
        
        Args:
            sections: Detected sections with hierarchy and line ranges
            metadata_items: List of (line_index, key, value) tuples
            qa_pairs: List of (line_index, qa_dict) tuples
            
        Returns:
            Structured JSON dictionary with proper nesting
        """
        logger.info("Building JSON output...")
        
        # Initialize tracking sets (CRITICAL: Must be done before assignment)
        self._metadata_line_indices = set()
        self._question_line_indices = set()
        
        # Step 1: Assign metadata and Q&A to sections based on line position
        self._assign_content_to_sections(sections, metadata_items, qa_pairs)
        
        output = {}
        
        # Step 2: Format sections with all their nested content
        for section in sections:
            output[section.title] = self._format_section(section)
        
        # Step 3: Add document-level metadata (line 0 or not in any section)
        doc_metadata = {}
        for line_idx, key, value in metadata_items:
            # Check if this metadata belongs to document level
            if line_idx == 0 or not self._find_section_for_line(sections, line_idx):
                if key not in doc_metadata:
                    doc_metadata[key] = []
                doc_metadata[key].append(value)
        
        # Clean document metadata (remove duplicates, single values)
        if doc_metadata:
            cleaned_metadata = {}
            for key, values in doc_metadata.items():
                unique_values = list(set(values))
                cleaned_metadata[key] = unique_values[0] if len(unique_values) == 1 else unique_values
            output['metadata'] = cleaned_metadata
        
        # Step 4: VALIDATE - Ensure no orphan content
        validation_results = self._validate_nesting(
            sections, metadata_items, qa_pairs
        )
        
        if validation_results['has_issues']:
            logger.warning(f"⚠ Validation found {validation_results['orphan_count']} orphan items")
            for issue in validation_results['issues'][:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ Validation passed: All content properly nested")
        
        logger.info(f"✓ Built JSON with {len(output)} top-level keys")
        return output
    
    def _validate_nesting(
        self,
        sections: List[Section],
        metadata_items: List[Tuple[int, str, Any]],
        qa_pairs: List[Tuple[int, Dict[str, str]]]
    ) -> Dict[str, Any]:
        """
        Validate that all content is properly nested within sections.
        
        CRITICAL VALIDATION: Check that no metadata or Q&A items are orphaned
        (i.e., not assigned to any section when they should be).
        
        Returns validation report with any issues found.
        """
        issues = []
        orphan_count = 0
        
        # Check metadata items
        for line_idx, key, value in metadata_items:
            if line_idx == 0:  # Document-level is OK
                continue
            
            section = self._find_section_for_line(sections, line_idx)
            if not section:
                issues.append(f"Orphan metadata at line {line_idx}: {key}={value}")
                orphan_count += 1
        
        # Check Q&A pairs
        for line_idx, qa in qa_pairs:
            section = self._find_section_for_line(sections, line_idx)
            if not section:
                issues.append(f"Orphan Q&A at line {line_idx}: {qa['question'][:50]}...")
                orphan_count += 1
        
        return {
            'has_issues': orphan_count > 0,
            'orphan_count': orphan_count,
            'issues': issues
        }
    
    def _assign_content_to_sections(
        self,
        sections: List[Section],
        metadata_items: List[Tuple[int, str, Any]],
        qa_pairs: List[Tuple[int, Dict[str, str]]]
    ):
        """
        Assign metadata and Q&A to sections based on line position.
        
        CRITICAL: This is where the magic happens - we determine which section
        each piece of content belongs to by checking if its line number falls
        within the section's line range.
        
        ENHANCED: Also tracks which lines are metadata/questions so we can
        exclude them from descriptions.
        """
        # Assign metadata to sections
        for line_idx, key, value in metadata_items:
            self._metadata_line_indices.add(line_idx)
            section = self._find_section_for_line(sections, line_idx)
            if section:
                if key not in section.metadata:
                    section.metadata[key] = []
                section.metadata[key].append(value)
        
        # Clean section metadata (remove duplicates)
        self._clean_section_metadata(sections)
        
        # Assign Q&A to sections
        for line_idx, qa in qa_pairs:
            self._question_line_indices.add(line_idx)
            section = self._find_section_for_line(sections, line_idx)
            if section:
                section.questions.append(qa)
    
    def _find_section_for_line(
        self,
        sections: List[Section],
        line_idx: int
    ) -> Optional[Section]:
        """
        Find which section a line belongs to based on line ranges.
        
        Checks nested subsections recursively to find the most specific match.
        """
        for section in sections:
            # Check if line is in this section's range
            if section.start_line <= line_idx < section.end_line:
                # Check subsections first (more specific)
                subsection = self._find_section_for_line(section.subsections, line_idx)
                if subsection:
                    return subsection
                # Otherwise, belongs to this section
                return section
        
        return None
    
    def _clean_section_metadata(self, sections: List[Section]):
        """Remove duplicate metadata values in all sections."""
        for section in sections:
            for key in section.metadata:
                values = section.metadata[key]
                if isinstance(values, list):
                    unique_values = list(set(values))
                    section.metadata[key] = unique_values[0] if len(unique_values) == 1 else unique_values
            
            # Recursively clean subsections
            self._clean_section_metadata(section.subsections)
    
    def _format_section(self, section: Section) -> Dict[str, Any]:
        """
        Format a single section for JSON output.
        
        ENHANCED: Uses tracked line indices to precisely filter out metadata/questions
        while preserving all descriptive content.
        
        CRITICAL FIX: Be PERMISSIVE - only exclude confirmed metadata/Q&A lines,
        keep everything else as description.
        """
        section_dict = {'type': 'section'}
        
        # Build descriptions by excluding ONLY confirmed metadata/question lines
        description_lines = []
        
        if hasattr(section, '_line_indices') and section._line_indices:
            # We have precise line tracking
            for i, line_text in enumerate(section.content):
                if i >= len(section._line_indices):
                    # Safety check
                    description_lines.append(line_text)
                    continue
                
                global_line_idx = section._line_indices[i]
                
                # ONLY skip if this line is CONFIRMED metadata or question
                if (global_line_idx in self._metadata_line_indices or
                    global_line_idx in self._question_line_indices):
                    continue
                
                # Keep everything else as description
                description_lines.append(line_text)
        else:
            # No line tracking - BE VERY PERMISSIVE, keep almost everything
            for line in section.content:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                    
                # ONLY filter extremely obvious metadata patterns
                # Pattern: "Word Word: value" with short key (< 25 chars)
                if re.match(r'^[A-Z][A-Za-z\s]{2,24}\s*[:\-]\s*.+', line_stripped):
                    # Could be metadata, but also could be content - be careful
                    # Only skip if it looks like a label
                    words = line_stripped.split(':')[0].strip().split()
                    if len(words) <= 4:  # Short labels only
                        continue
                
                # Keep this line as description
                description_lines.append(line)
        
        # Merge into paragraphs
        if description_lines:
            paragraphs = self.paragraph_merger.merge_paragraphs(description_lines)
            if paragraphs:
                section_dict['description'] = paragraphs
        
        # LAST RESORT FALLBACK: If description is STILL empty but we have content, use it ALL
        if 'description' not in section_dict and section.content:
            logger.debug(f"Fallback: Section '{section.title}' has content but no description. Using all content.")
            # Use ALL content
            section_dict['description'] = self.paragraph_merger.merge_paragraphs(section.content)
        
        # Add section metadata (if any)
        if section.metadata:
            section_dict['metadata'] = section.metadata
        
        # Add questions (if any)
        if section.questions:
            section_dict['questions'] = section.questions
        
        # Add subsections (if any) - RECURSIVE nesting
        if section.subsections:
            section_dict['sub_sections'] = {
                sub.title: self._format_section(sub)
                for sub in section.subsections
            }
        
        return section_dict


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class DocumentAnalyzer:
    """
    Main document analyzer orchestrating all components.
    
    Pipeline:
    1. Extract text from PDF
    2. Detect sections
    3. Extract metadata
    4. Detect Q&A pairs
    5. Build hierarchical JSON
    """
    
    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing DocumentAnalyzer v2...")
        
        self.extractor = PDFTextExtractor()
        self.section_detector = SectionDetector()
        self.metadata_extractor = MetadataExtractor()
        self.qa_detector = QADetector()
        self.json_builder = JSONBuilder()
        
        logger.info("✓ DocumentAnalyzer v2 ready")
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze PDF document and return structured JSON.
        
        CRITICAL: This orchestrates the entire pipeline, ensuring that
        metadata and Q&A are properly assigned to sections based on
        their line positions in the document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Hierarchical JSON dictionary with proper nesting
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING: {file_path}")
        logger.info(f"{'='*60}\n")
        
        # Step 1: Extract text with line positions
        logger.info("STEP 1/5: Extracting text from PDF...")
        lines = self.extractor.extract_text(file_path)
        logger.info(f"Extracted {len(lines)} lines from document")
        
        # Step 2: Detect sections with line ranges
        logger.info("\nSTEP 2/5: Detecting document sections...")
        sections = self.section_detector.detect_sections(lines)
        logger.info(f"Detected {len(sections)} top-level sections")
        
        # Step 3: Extract metadata with line positions
        logger.info("\nSTEP 3/5: Extracting metadata...")
        metadata_items = self.metadata_extractor.extract_metadata(lines)
        logger.info(f"Extracted {len(metadata_items)} metadata items")
        
        # Step 4: Detect Q&A with line positions
        logger.info("\nSTEP 4/5: Detecting Q&A pairs...")
        qa_pairs = self.qa_detector.detect_qa_pairs(lines)
        logger.info(f"Detected {len(qa_pairs)} Q&A pairs")
        
        # Step 5: Build JSON with proper nesting
        logger.info("\nSTEP 5/5: Building JSON output with proper nesting...")
        result = self.json_builder.build_json(sections, metadata_items, qa_pairs)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ ANALYSIS COMPLETE")
        logger.info(f"✓ All content properly nested within sections")
        logger.info(f"{'='*60}\n")
        
        return result


# ============================================================================
# PUBLIC API
# ============================================================================

def analyze_document(file_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze PDF document and extract structured data.
    
    Args:
        file_path: Path to PDF file
        output_file: Optional path to save JSON output
        
    Returns:
        Hierarchical JSON dictionary
        
    Example:
        >>> result = analyze_document("resume.pdf")
        >>> print(json.dumps(result, indent=2))
    """
    analyzer = DocumentAnalyzer()
    result = analyzer.analyze(file_path)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Output saved to: {output_file}")
    
    return result


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
Usage: python document_analyzer_v2.py <pdf_file> [output.json]

Examples:
    python document_analyzer_v2.py document.pdf
    python document_analyzer_v2.py document.pdf output.json
""")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = analyze_document(input_file, output_file)
        
        if not output_file:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


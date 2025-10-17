"""
Algorithmic Document Parser - Universal document structure extraction.

This parser uses algorithmic approaches to analyze and structure any document type
without relying on hardcoded patterns or document-specific rules.

Key Features:
- Universal document type support
- Algorithmic text analysis and classification
- Automatic structure detection
- Hierarchical content organization
- Pattern-based content classification
"""

import logging
import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path
import statistics

from app.models.text_element import TextElement, ExtractionResult
from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


@dataclass
class DocumentPattern:
    """Represents a detected pattern in the document."""
    pattern_type: str
    confidence: float
    text: str
    position: int
    metadata: Dict[str, Any] = None


@dataclass
class DocumentSection:
    """Represents a document section with hierarchical structure."""
    title: str
    level: int
    content: List[str]
    subsections: List['DocumentSection']
    metadata: Dict[str, Any]
    start_position: int
    end_position: int


class TextAnalyzer:
    """Advanced text analysis algorithms for document understanding."""
    
    def __init__(self):
        self.pattern_cache = {}
        
    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text for various patterns and characteristics."""
        if not text:
            return {}
            
        analysis = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'paragraph_count': len(text.split('\n\n')),
            'has_numbers': bool(re.search(r'\d', text)),
            'has_dates': bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', text)),
            'has_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            'has_urls': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'has_phone': bool(re.search(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', text)),
            'capitalization_ratio': self._calculate_capitalization_ratio(text),
            'punctuation_density': self._calculate_punctuation_density(text),
            'special_char_ratio': self._calculate_special_char_ratio(text),
            'readability_score': self._calculate_readability_score(text)
        }
        
        return analysis
    
    def _calculate_capitalization_ratio(self, text: str) -> float:
        """Calculate ratio of capitalized characters."""
        if not text:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)
    
    def _calculate_punctuation_density(self, text: str) -> float:
        """Calculate punctuation density."""
        if not text:
            return 0.0
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        return punctuation_count / len(text)
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters."""
        if not text:
            return 0.0
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special_count / len(text)
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score."""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = self._count_syllables(text) / len(words)
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        total_syllables = 0
        
        for word in words:
            # Simple syllable counting
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent 'e'
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            total_syllables += max(1, syllable_count)
        
        return total_syllables


class StructureDetector:
    """Algorithmic document structure detection."""
    
    def __init__(self):
        self.heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z]',   # Numbered headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$',  # Title with colon
        ]
        
        self.list_patterns = [
            r'^[•\-\*]\s+',      # Bullet points
            r'^\d+\.\s+',        # Numbered lists
            r'^[a-z]\)\s+',      # Letter lists
            r'^[ivx]+\)\s+',     # Roman numerals
        ]
        
        self.metadata_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'time': r'\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%',
        }
    
    def detect_heading_level(self, text: str, font_size: float = None, is_bold: bool = False) -> Tuple[int, float]:
        """Detect heading level algorithmically."""
        if not text:
            return 0, 0.0
        
        confidence = 0.0
        level = 0
        
        # Pattern-based detection
        for i, pattern in enumerate(self.heading_patterns):
            if re.match(pattern, text.strip()):
                confidence += 0.3
                level = max(level, i + 1)
        
        # Font-based detection
        if font_size:
            if font_size > 16:
                confidence += 0.4
                level = max(level, 1)
            elif font_size > 14:
                confidence += 0.3
                level = max(level, 2)
            elif font_size > 12:
                confidence += 0.2
                level = max(level, 3)
        
        # Bold text detection
        if is_bold:
            confidence += 0.2
            level = max(level, 1)
        
        # Length-based detection
        word_count = len(text.split())
        if word_count <= 3:
            confidence += 0.3
            level = max(level, 1)
        elif word_count <= 6:
            confidence += 0.2
            level = max(level, 2)
        elif word_count <= 10:
            confidence += 0.1
            level = max(level, 3)
        
        # Capitalization
        if text.isupper():
            confidence += 0.2
            level = max(level, 1)
        elif text.istitle():
            confidence += 0.1
            level = max(level, 2)
        
        return level, min(1.0, confidence)
    
    def detect_list_item(self, text: str) -> Tuple[bool, str, int]:
        """Detect if text is a list item and return type and level."""
        if not text:
            return False, "", 0
        
        text = text.strip()
        
        for i, pattern in enumerate(self.list_patterns):
            match = re.match(pattern, text)
            if match:
                return True, pattern, i + 1
        
        return False, "", 0
    
    def extract_metadata(self, text: str) -> Dict[str, List[str]]:
        """Extract metadata from text using pattern matching."""
        metadata = {}
        
        for key, pattern in self.metadata_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metadata[key] = [match if isinstance(match, str) else ' '.join(match) for match in matches]
        
        return metadata
    
    def detect_section_boundaries(self, elements: List[TextElement]) -> List[int]:
        """Detect section boundaries based on text analysis."""
        boundaries = [0]  # Start of document
        
        for i, element in enumerate(elements):
            # Check for heading indicators
            heading_level, confidence = self.detect_heading_level(
                element.text, 
                element.font_size, 
                element.is_bold
            )
            
            if heading_level > 0 and confidence > 0.5:
                boundaries.append(i)
        
        boundaries.append(len(elements))  # End of document
        return boundaries


class ContentClassifier:
    """Algorithmic content classification based on text analysis."""
    
    def __init__(self):
        self.classifier_cache = {}
        
        # Content type indicators
        self.content_indicators = {
            'contact_info': [
                'email', 'phone', 'address', 'location', 'contact', 'reach'
            ],
            'education': [
                'education', 'degree', 'university', 'college', 'school', 'graduated',
                'bachelor', 'master', 'phd', 'diploma', 'certificate'
            ],
            'experience': [
                'experience', 'work', 'employment', 'job', 'position', 'role',
                'company', 'employer', 'career', 'professional'
            ],
            'skills': [
                'skills', 'technical', 'programming', 'languages', 'tools',
                'technologies', 'expertise', 'competencies'
            ],
            'projects': [
                'projects', 'portfolio', 'work samples', 'case studies',
                'implemented', 'developed', 'built', 'created'
            ],
            'achievements': [
                'achievements', 'awards', 'honors', 'recognition', 'accomplishments',
                'success', 'results', 'impact'
            ],
            'references': [
                'references', 'recommendations', 'testimonials', 'endorsements'
            ]
        }
    
    def classify_content(self, text: str, context: Dict[str, Any] = None) -> Tuple[str, float]:
        """Classify content based on text analysis."""
        if not text:
            return "unknown", 0.0
        
        text_lower = text.lower()
        scores = {}
        
        # Keyword-based scoring
        for content_type, keywords in self.content_indicators.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            scores[content_type] = score / len(keywords)
        
        # Pattern-based scoring
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            scores['contact_info'] = scores.get('contact_info', 0) + 0.5
        
        if re.search(r'\b\d{4}\s*[-–]\s*(Present|\d{4})\b', text):
            scores['experience'] = scores.get('experience', 0) + 0.3
        
        if re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', text):
            scores['experience'] = scores.get('experience', 0) + 0.2
        
        # Context-based scoring
        if context:
            if context.get('is_heading'):
                scores['section_title'] = 0.8
            if context.get('is_list_item'):
                scores['list_item'] = 0.7
        
        # Find best match
        if not scores:
            return "description", 0.5
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, min(1.0, confidence)


class HierarchicalBuilder:
    """Build hierarchical document structure algorithmically."""
    
    def __init__(self):
        self.sections = []
        self.current_section = None
        self.current_subsection = None
        self.section_stack = []
    
    def build_structure(self, elements: List[TextElement]) -> Dict[str, Any]:
        """Build hierarchical document structure."""
        structure = {
            'metadata': {},
            'sections': [],
            'content': []
        }
        
        # Process elements sequentially
        for element in elements:
            self._process_element(element, structure)
        
        # Finalize structure
        self._finalize_structure(structure)
        
        return structure
    
    def _process_element(self, element: TextElement, structure: Dict[str, Any]):
        """Process a single element and update structure."""
        text = element.text.strip()
        if not text:
            return
        
        # Detect element type
        element_type, confidence = self._classify_element(element)
        
        if element_type == 'heading':
            self._handle_heading(text, element, structure)
        elif element_type == 'list_item':
            self._handle_list_item(text, element, structure)
        elif element_type == 'metadata':
            self._handle_metadata(text, element, structure)
        else:
            self._handle_content(text, element, structure)
    
    def _classify_element(self, element: TextElement) -> Tuple[str, float]:
        """Classify element type algorithmically."""
        text = element.text.strip()
        
        # Heading detection
        if self._is_heading(text, element):
            return 'heading', 0.9
        
        # List item detection
        if self._is_list_item(text):
            return 'list_item', 0.8
        
        # Metadata detection
        if self._is_metadata(text):
            return 'metadata', 0.7
        
        # Default to content
        return 'content', 0.5
    
    def _is_heading(self, text: str, element: TextElement) -> bool:
        """Check if text is a heading."""
        # Short text with high confidence
        if len(text.split()) <= 6:
            # Check font characteristics
            if element.font_size and element.font_size > 12:
                return True
            if element.is_bold:
                return True
        
        # Pattern matching
        heading_patterns = [
            r'^[A-Z][A-Z\s]+$',
            r'^\d+\.\s+[A-Z]',
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item."""
        list_patterns = [
            r'^[•\-\*]\s+',
            r'^\d+\.\s+',
            r'^[a-z]\)\s+',
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_metadata(self, text: str) -> bool:
        """Check if text contains metadata."""
        metadata_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _handle_heading(self, text: str, element: TextElement, structure: Dict[str, Any]):
        """Handle heading element."""
        # Determine heading level
        level = self._determine_heading_level(text, element)
        
        # Create new section
        section = {
            'title': text,
            'level': level,
            'content': [],
            'subsections': [],
            'metadata': {},
            'start_position': element.page_number
        }
        
        # Add to appropriate level
        if level == 1:
            structure['sections'].append(section)
            self.current_section = section
            self.section_stack = [section]
        else:
            # Find appropriate parent
            parent = self._find_parent_section(level)
            if parent:
                parent['subsections'].append(section)
                self.section_stack.append(section)
            else:
                structure['sections'].append(section)
                self.section_stack = [section]
    
    def _handle_list_item(self, text: str, element: TextElement, structure: Dict[str, Any]):
        """Handle list item element."""
        # Add to current section or create content section
        if self.current_section:
            self.current_section['content'].append(text)
        else:
            structure['content'].append(text)
    
    def _handle_metadata(self, text: str, element: TextElement, structure: Dict[str, Any]):
        """Handle metadata element."""
        # Extract metadata patterns
        metadata = self._extract_metadata(text)
        
        if self.current_section:
            self.current_section['metadata'].update(metadata)
        else:
            structure['metadata'].update(metadata)
    
    def _handle_content(self, text: str, element: TextElement, structure: Dict[str, Any]):
        """Handle content element."""
        if self.current_section:
            self.current_section['content'].append(text)
        else:
            structure['content'].append(text)
    
    def _determine_heading_level(self, text: str, element: TextElement) -> int:
        """Determine heading level."""
        level = 1
        
        # Font size based
        if element.font_size:
            if element.font_size > 16:
                level = 1
            elif element.font_size > 14:
                level = 2
            elif element.font_size > 12:
                level = 3
            else:
                level = 4
        
        # Pattern based
        if re.match(r'^\d+\.\s+', text):
            level = 1
        elif re.match(r'^\d+\.\d+\.\s+', text):
            level = 2
        elif re.match(r'^\d+\.\d+\.\d+\.\s+', text):
            level = 3
        
        return level
    
    def _find_parent_section(self, level: int) -> Optional[Dict[str, Any]]:
        """Find appropriate parent section for given level."""
        for i in range(len(self.section_stack) - 1, -1, -1):
            if self.section_stack[i]['level'] < level:
                return self.section_stack[i]
        return None
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text."""
        metadata = {}
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            metadata['email'] = email_match.group()
        
        # Phone
        phone_match = re.search(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', text)
        if phone_match:
            metadata['phone'] = phone_match.group()
        
        # URL
        url_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if url_match:
            metadata['url'] = url_match.group()
        
        return metadata
    
    def _finalize_structure(self, structure: Dict[str, Any]):
        """Finalize document structure."""
        # Clean up empty sections
        self._clean_empty_sections(structure['sections'])
        
        # Normalize content
        self._normalize_content(structure)
    
    def _clean_empty_sections(self, sections: List[Dict[str, Any]]):
        """Remove empty sections."""
        sections_to_remove = []
        
        for i, section in enumerate(sections):
            if not section['content'] and not section['subsections']:
                sections_to_remove.append(i)
            else:
                # Recursively clean subsections
                self._clean_empty_sections(section['subsections'])
        
        # Remove empty sections (in reverse order)
        for i in reversed(sections_to_remove):
            sections.pop(i)
    
    def _normalize_content(self, structure: Dict[str, Any]):
        """Normalize content structure."""
        # Merge short content blocks
        for section in structure['sections']:
            self._merge_short_content(section)
    
    def _merge_short_content(self, section: Dict[str, Any]):
        """Merge short content blocks."""
        if len(section['content']) <= 1:
            return
        
        merged_content = []
        current_block = ""
        
        for content in section['content']:
            if len(content.split()) < 10:  # Short content
                if current_block:
                    current_block += " " + content
                else:
                    current_block = content
            else:
                if current_block:
                    merged_content.append(current_block)
                    current_block = ""
                merged_content.append(content)
        
        if current_block:
            merged_content.append(current_block)
        
        section['content'] = merged_content
        
        # Recursively process subsections
        for subsection in section['subsections']:
            self._merge_short_content(subsection)


class AlgorithmicDocumentParser:
    """Main algorithmic document parser."""
    
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.structure_detector = StructureDetector()
        self.content_classifier = ContentClassifier()
        self.hierarchical_builder = HierarchicalBuilder()
        self.extractor = PDFExtractor()
        
        logger.info("Algorithmic Document Parser initialized")
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse document using algorithmic approaches."""
        logger.info(f"Starting algorithmic document parsing: {file_path}")
        
        # Extract text elements
        extraction_result = self.extractor.extract_from_file(file_path)
        logger.info(f"Extracted {len(extraction_result.elements)} text elements")
        
        # Analyze text patterns
        analyzed_elements = self._analyze_elements(extraction_result.elements)
        logger.info(f"Analyzed {len(analyzed_elements)} elements")
        
        # Detect document structure
        structure = self.hierarchical_builder.build_structure(analyzed_elements)
        logger.info("Built hierarchical structure")
        
        # Add document metadata
        structure['document_metadata'] = {
            'total_pages': extraction_result.total_pages,
            'extraction_method': extraction_result.extraction_method,
            'total_elements': len(extraction_result.elements),
            'file_path': file_path
        }
        
        logger.info("Algorithmic document parsing completed")
        return structure
    
    def _analyze_elements(self, elements: List[TextElement]) -> List[TextElement]:
        """Analyze text elements and add analysis metadata."""
        analyzed_elements = []
        
        for element in elements:
            # Analyze text patterns
            analysis = self.text_analyzer.analyze_text_patterns(element.text)
            
            # Add analysis to element metadata
            element.metadata.update(analysis)
            
            # Classify content
            content_type, confidence = self.content_classifier.classify_content(
                element.text, 
                {
                    'font_size': element.font_size,
                    'is_bold': element.is_bold,
                    'is_heading': analysis.get('capitalization_ratio', 0) > 0.5
                }
            )
            
            element.metadata['content_type'] = content_type
            element.metadata['classification_confidence'] = confidence
            
            analyzed_elements.append(element)
        
        return analyzed_elements


class UniversalDocumentParserPipeline:
    """Universal document parsing pipeline."""
    
    def __init__(self):
        self.parser = AlgorithmicDocumentParser()
        logger.info("Universal Document Parser Pipeline initialized")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process any document type and return structured output."""
        try:
            return self.parser.parse_document(file_path)
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise

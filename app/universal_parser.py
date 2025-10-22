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
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path
import statistics
import asyncio
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

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


class UniversalTextAnalyzer:
    """Universal text analysis with adaptive heuristics for any document type."""
    
    def __init__(self):
        self.pattern_cache = {}
        self._analysis_cache = {}
        self._cache_lock = threading.RLock()
        self._compiled_patterns = self._compile_patterns()
        
        # Adaptive thresholds that adjust per document
        self.adaptive_thresholds = {
            'heading_confidence_min': 0.6,
            'font_size_variance_threshold': 2.0,
            'indentation_cluster_threshold': 10.0,
            'capitalization_ratio_threshold': 0.7,
            'line_spacing_threshold': 5.0,
            'word_count_threshold': 15
        }
        
        # Document context for adaptive analysis
        self.document_context = {
            'font_sizes': [],
            'indentation_levels': [],
            'line_spacings': [],
            'capitalization_patterns': [],
            'mean_font_size': 12,
            'font_size_variance': 2.0
        }
        
    def analyze_text_patterns(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Universal text analysis with contextual awareness and adaptive heuristics."""
        if not text:
            return {}
        
        # Check cache first
        cache_key = f"{text}:{context}"
        text_hash = hashlib.md5(cache_key.encode()).hexdigest()
        with self._cache_lock:
            if text_hash in self._analysis_cache:
                return self._analysis_cache[text_hash]
        
        analysis = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(self._compiled_patterns['sentences'].findall(text)),
            'paragraph_count': len(text.split('\n\n')),
            'has_numbers': bool(self._compiled_patterns['numbers'].search(text)),
            'has_dates': bool(self._compiled_patterns['dates'].search(text)),
            'has_emails': bool(self._compiled_patterns['emails'].search(text)),
            'has_urls': bool(self._compiled_patterns['urls'].search(text)),
            'has_phone': bool(self._compiled_patterns['phone'].search(text)),
            'capitalization_ratio': self._calculate_capitalization_ratio(text),
            'punctuation_density': self._calculate_punctuation_density(text),
            'special_char_ratio': self._calculate_special_char_ratio(text),
            'readability_score': self._calculate_readability_score(text),
            
            # Universal document analysis
            'structure_signals': self._analyze_structure_signals(text, context),
            'content_type': self._classify_content_type(text, context),
            'metadata_indicators': self._extract_metadata_indicators(text),
            'linguistic_features': self._analyze_linguistic_features(text)
        }
        
        # Cache result
        with self._cache_lock:
            if len(self._analysis_cache) < 1000:
                self._analysis_cache[text_hash] = analysis
        
        return analysis
    
    def _analyze_structure_signals(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze structural signals for universal document understanding."""
        signals = {
            'is_potential_heading': False,
            'is_potential_list_item': False,
            'is_potential_metadata': False,
            'is_potential_section_title': False,
            'heading_confidence': 0.0,
            'list_item_confidence': 0.0,
            'metadata_confidence': 0.0
        }
        
        # Heading detection with adaptive thresholds
        heading_signals = self._detect_heading_signals(text, context)
        signals.update(heading_signals)
        
        # List item detection
        list_signals = self._detect_list_signals(text)
        signals.update(list_signals)
        
        # Metadata detection
        metadata_signals = self._detect_metadata_signals(text)
        signals.update(metadata_signals)
        
        return signals
    
    def _detect_heading_signals(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect heading signals with adaptive heuristics."""
        signals = {
            'is_potential_heading': False,
            'is_potential_section_title': False,
            'heading_confidence': 0.0
        }
        
        if not text or len(text.strip()) == 0:
            return signals
        
        text = text.strip()
        confidence = 0.0
        
        # Length-based analysis
        word_count = len(text.split())
        if word_count <= 3:
            confidence += 0.4
        elif word_count <= 6:
            confidence += 0.3
        elif word_count <= 10:
            confidence += 0.2
        
        # Capitalization analysis
        if text.isupper():
            confidence += 0.3
        elif text.istitle():
            confidence += 0.4
        elif any(word.istitle() for word in text.split()):
            confidence += 0.2
        
        # Pattern-based detection
        if self._compiled_patterns['heading_patterns']:
            for pattern in self._compiled_patterns['heading_patterns']:
                if pattern.match(text):
                    confidence += 0.3
                    break
        
        # Context-based analysis
        if context:
            font_size = context.get('font_size')
            if font_size and font_size > (self.document_context.get('mean_font_size', 12) + 2):
                confidence += 0.2
            
            if context.get('is_bold'):
                confidence += 0.2
            
            if context.get('indentation_level', 0) == 0:
                confidence += 0.1
        
        # Apply adaptive threshold
        threshold = self.adaptive_thresholds['heading_confidence_min']
        if confidence >= threshold:
            signals['is_potential_heading'] = True
            signals['is_potential_section_title'] = True
        
        signals['heading_confidence'] = min(1.0, confidence)
        return signals
    
    def _detect_list_signals(self, text: str) -> Dict[str, Any]:
        """Detect list item signals."""
        signals = {
            'is_potential_list_item': False,
            'list_item_confidence': 0.0
        }
        
        if not text:
            return signals
        
        text = text.strip()
        confidence = 0.0
        
        # Check for list patterns
        for pattern in self._compiled_patterns['list_patterns']:
            if pattern.match(text):
                confidence += 0.8
                signals['is_potential_list_item'] = True
                break
        
        signals['list_item_confidence'] = confidence
        return signals
    
    def _detect_metadata_signals(self, text: str) -> Dict[str, Any]:
        """Detect metadata signals."""
        signals = {
            'is_potential_metadata': False,
            'metadata_confidence': 0.0
        }
        
        if not text:
            return signals
        
        confidence = 0.0
        
        # Check for metadata patterns
        metadata_patterns = ['emails', 'phone', 'urls', 'dates']
        for pattern_name in metadata_patterns:
            if self._compiled_patterns[pattern_name].search(text):
                confidence += 0.3
        
        if confidence > 0:
            signals['is_potential_metadata'] = True
        
        signals['metadata_confidence'] = min(1.0, confidence)
        return signals
    
    def _classify_content_type(self, text: str, context: Dict[str, Any] = None) -> str:
        """Classify content type for universal document understanding."""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # Universal content type indicators
        content_indicators = {
            'contact_info': ['email', 'phone', 'address', 'contact', 'location'],
            'education': ['education', 'degree', 'university', 'college', 'school', 'graduated'],
            'experience': ['experience', 'work', 'employment', 'job', 'position', 'career'],
            'skills': ['skills', 'technical', 'programming', 'languages', 'tools', 'technologies'],
            'projects': ['projects', 'portfolio', 'work samples', 'case studies'],
            'achievements': ['achievements', 'awards', 'honors', 'recognition'],
            'summary': ['summary', 'objective', 'profile', 'overview'],
            'references': ['references', 'recommendations', 'testimonials']
        }
        
        scores = {}
        for content_type, keywords in content_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[content_type] = score / len(keywords)
        
        if not scores or max(scores.values()) < 0.1:
            return 'description'
        
        return max(scores, key=scores.get)
    
    def _extract_metadata_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract metadata indicators with enhanced patterns."""
        metadata = {}
        
        # Enhanced metadata patterns
        patterns = {
            'email': self._compiled_patterns['emails'],
            'phone': self._compiled_patterns['phone'],
            'url': self._compiled_patterns['urls'],
            'date': self._compiled_patterns['dates'],
            'address': re.compile(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)', re.IGNORECASE)
        }
        
        for key, pattern in patterns.items():
            matches = pattern.findall(text)
            if matches:
                metadata[key] = [match if isinstance(match, str) else ' '.join(match) for match in matches]
        
        return metadata
    
    def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features for document understanding."""
        features = {
            'sentence_complexity': 0.0,
            'vocabulary_richness': 0.0,
            'formality_level': 0.0,
            'technical_density': 0.0
        }
        
        if not text:
            return features
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Sentence complexity
        if sentences and words:
            avg_words_per_sentence = len(words) / len(sentences)
            features['sentence_complexity'] = min(1.0, avg_words_per_sentence / 20)
        
        # Vocabulary richness (unique words ratio)
        if words:
            unique_words = len(set(word.lower() for word in words))
            features['vocabulary_richness'] = unique_words / len(words)
        
        # Formality level (based on formal language patterns)
        formal_patterns = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
        formal_count = sum(1 for pattern in formal_patterns if pattern in text.lower())
        features['formality_level'] = min(1.0, formal_count / 5)
        
        # Technical density (technical terms ratio)
        technical_patterns = ['algorithm', 'implementation', 'optimization', 'framework', 'architecture']
        technical_count = sum(1 for pattern in technical_patterns if pattern in text.lower())
        features['technical_density'] = min(1.0, technical_count / 5)
        
        return features
    
    def adapt_thresholds(self, document_stats: Dict[str, Any]):
        """Adapt thresholds based on document statistics."""
        self.document_context.update(document_stats)
        
        # Adapt thresholds based on document characteristics
        if self.document_context.get('font_size_variance', 0) > 5:
            self.adaptive_thresholds['heading_confidence_min'] = 0.5  # More lenient for varied documents
        
        if self.document_context.get('mean_font_size', 12) > 14:
            self.adaptive_thresholds['font_size_variance_threshold'] = 3.0  # Larger fonts need more variance
        
        if self.document_context.get('indentation_clusters', 0) > 3:
            self.adaptive_thresholds['indentation_cluster_threshold'] = 15.0  # More complex indentation
    
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
    
    def _compile_patterns(self) -> Dict[str, Any]:
        """Compile regex patterns for better performance."""
        return {
            'sentences': re.compile(r'[.!?]+'),
            'numbers': re.compile(r'\d'),
            'dates': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            
            # Universal document patterns
            'heading_patterns': [
                re.compile(r'^[A-Z][A-Z\s]+$'),  # ALL CAPS
                re.compile(r'^\d+\.\s+[A-Z]'),   # Numbered headings
                re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'),  # Title Case
                re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$'),  # Title with colon
            ],
            'list_patterns': [
                re.compile(r'^[•\-\*]\s+'),      # Bullet points
                re.compile(r'^\d+\.\s+'),        # Numbered lists
                re.compile(r'^[a-z]\)\s+'),      # Letter lists
                re.compile(r'^[ivx]+\)\s+'),     # Roman numerals
            ]
        }


class UniversalStructureDetector:
    """Universal structure detection with contextual analysis and adaptive heuristics."""
    
    def __init__(self):
        # Universal document patterns
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
        
        # Compile patterns for better performance
        self._compiled_heading_patterns = [re.compile(p) for p in self.heading_patterns]
        self._compiled_list_patterns = [re.compile(p) for p in self.list_patterns]
        self._compiled_metadata_patterns = {
            key: re.compile(pattern) for key, pattern in self.metadata_patterns.items()
        }
        
        # Cache for detection results
        self._detection_cache = {}
        self._cache_lock = threading.RLock()
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'heading_confidence_min': 0.6,
            'font_size_difference_threshold': 2.0,
            'indentation_cluster_threshold': 10.0,
            'spacing_variance_threshold': 5.0
        }
        
        # Document context for adaptive analysis
        self.document_context = {
            'font_sizes': [],
            'indentation_levels': [],
            'line_spacings': [],
            'capitalization_patterns': []
        }
    
    def detect_heading_level(self, text: str, font_size: float = None, is_bold: bool = False, context: Dict[str, Any] = None) -> Tuple[int, float]:
        """Detect heading level with universal contextual analysis."""
        if not text:
            return 0, 0.0
        
        # Check cache first
        cache_key = f"{text}:{font_size}:{is_bold}:{context}"
        with self._cache_lock:
            if cache_key in self._detection_cache:
                return self._detection_cache[cache_key]
        
        confidence = 0.0
        level = 0
        
        # Universal heading detection with contextual analysis
        heading_analysis = self._analyze_heading_context(text, font_size, is_bold, context)
        confidence = heading_analysis['confidence']
        level = heading_analysis['level']
        
        result = (level, min(1.0, confidence))
        
        # Cache result
        with self._cache_lock:
            if len(self._detection_cache) < 1000:
                self._detection_cache[cache_key] = result
        
        return result
    
    def _analyze_heading_context(self, text: str, font_size: float = None, is_bold: bool = False, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze heading context with universal document understanding."""
        analysis = {'confidence': 0.0, 'level': 0}
        
        if not text:
            return analysis
        
        text = text.strip()
        confidence = 0.0
        level = 0
        
        # Pattern-based detection
        for i, pattern in enumerate(self._compiled_heading_patterns):
            if pattern.match(text):
                confidence += 0.3
                level = max(level, i + 1)
        
        # Font-based analysis with adaptive thresholds
        if font_size:
            font_analysis = self._analyze_font_context(font_size)
            confidence += font_analysis['confidence']
            level = max(level, font_analysis['level'])
        
        # Bold text analysis
        if is_bold:
            confidence += 0.2
            level = max(level, 1)
        
        # Length-based analysis
        word_count = len(text.split())
        length_analysis = self._analyze_length_context(word_count)
        confidence += length_analysis['confidence']
        level = max(level, length_analysis['level'])
        
        # Capitalization analysis
        capitalization_analysis = self._analyze_capitalization_context(text)
        confidence += capitalization_analysis['confidence']
        level = max(level, capitalization_analysis['level'])
        
        # Contextual analysis
        if context:
            contextual_analysis = self._analyze_contextual_signals(text, context)
            confidence += contextual_analysis['confidence']
            level = max(level, contextual_analysis['level'])
        
        analysis['confidence'] = min(1.0, confidence)
        analysis['level'] = level
        return analysis
    
    def _analyze_font_context(self, font_size: float) -> Dict[str, Any]:
        """Analyze font context with adaptive thresholds."""
        analysis = {'confidence': 0.0, 'level': 0}
        
        # Use pre-calculated values from document context
        mean_font_size = self.document_context.get('mean_font_size', 12)
        font_variance = self.document_context.get('font_size_variance', 2.0)
        
        # Adaptive font size analysis
        font_diff = font_size - mean_font_size
        if font_diff > font_variance * 1.5:
            analysis['confidence'] = 0.4
            analysis['level'] = 1
        elif font_diff > font_variance:
            analysis['confidence'] = 0.3
            analysis['level'] = 2
        elif font_diff > font_variance * 0.5:
            analysis['confidence'] = 0.2
            analysis['level'] = 3
        
        return analysis
    
    def _analyze_length_context(self, word_count: int) -> Dict[str, Any]:
        """Analyze length context for heading detection."""
        analysis = {'confidence': 0.0, 'level': 0}
        
        if word_count <= 3:
            analysis['confidence'] = 0.3
            analysis['level'] = 1
        elif word_count <= 6:
            analysis['confidence'] = 0.2
            analysis['level'] = 2
        elif word_count <= 10:
            analysis['confidence'] = 0.1
            analysis['level'] = 3
        
        return analysis
    
    def _analyze_capitalization_context(self, text: str) -> Dict[str, Any]:
        """Analyze capitalization context for heading detection."""
        analysis = {'confidence': 0.0, 'level': 0}
        
        if text.isupper():
            analysis['confidence'] = 0.3
            analysis['level'] = 1
        elif text.istitle():
            analysis['confidence'] = 0.2
            analysis['level'] = 2
        elif any(word.istitle() for word in text.split()):
            analysis['confidence'] = 0.1
            analysis['level'] = 3
        
        return analysis
    
    def _analyze_contextual_signals(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual signals for heading detection."""
        analysis = {'confidence': 0.0, 'level': 0}
        
        # Indentation analysis
        indentation_level = context.get('indentation_level', 0)
        if indentation_level == 0:
            analysis['confidence'] += 0.1
            analysis['level'] = max(analysis['level'], 1)
        
        # Line spacing analysis
        line_spacing = context.get('line_spacing', 0)
        if line_spacing > self.adaptive_thresholds['spacing_variance_threshold']:
            analysis['confidence'] += 0.1
        
        # Position analysis (beginning of section)
        if context.get('is_section_start', False):
            analysis['confidence'] += 0.2
            analysis['level'] = max(analysis['level'], 1)
        
        return analysis
    
    def detect_list_item(self, text: str, context: Dict[str, Any] = None) -> Tuple[bool, str, int]:
        """Detect list item with universal contextual analysis."""
        if not text:
            return False, "", 0
        
        text = text.strip()
        
        # Check cache first
        cache_key = f"{text}:{context}"
        with self._cache_lock:
            if cache_key in self._detection_cache:
                return self._detection_cache[cache_key]
        
        # Universal list item detection
        for i, pattern in enumerate(self._compiled_list_patterns):
            match = pattern.match(text)
            if match:
                result = (True, self.list_patterns[i], i + 1)
                # Cache result
                with self._cache_lock:
                    if len(self._detection_cache) < 1000:
                        self._detection_cache[cache_key] = result
                return result
        
        # Contextual list item detection
        if context:
            contextual_analysis = self._analyze_list_context(text, context)
            if contextual_analysis['is_list_item']:
                result = (True, contextual_analysis['pattern'], contextual_analysis['level'])
                with self._cache_lock:
                    if len(self._detection_cache) < 1000:
                        self._detection_cache[cache_key] = result
                return result
        
        result = (False, "", 0)
        # Cache result
        with self._cache_lock:
            if len(self._detection_cache) < 1000:
                self._detection_cache[cache_key] = result
        return result
    
    def _analyze_list_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze list context for universal document understanding."""
        analysis = {'is_list_item': False, 'pattern': '', 'level': 0}
        
        # Indentation-based list detection
        indentation_level = context.get('indentation_level', 0)
        if indentation_level > 0 and len(text.split()) <= 10:
            analysis['is_list_item'] = True
            analysis['pattern'] = 'indented_list'
            analysis['level'] = indentation_level
        
        # Font-based list detection (smaller font for sub-items)
        font_size = context.get('font_size', 12)
        if font_size < context.get('parent_font_size', 12) * 0.9:
            analysis['is_list_item'] = True
            analysis['pattern'] = 'font_size_list'
            analysis['level'] = 2
        
        return analysis
    
    def update_document_context(self, elements: List[TextElement]):
        """Update document context for adaptive analysis."""
        font_sizes = [elem.font_size for elem in elements if elem.font_size]
        indentation_levels = [elem.indentation_level for elem in elements]
        line_spacings = [elem.vertical_spacing for elem in elements if elem.vertical_spacing]
        capitalization_patterns = [elem.text.upper() == elem.text for elem in elements]
        
        self.document_context.update({
            'font_sizes': font_sizes,
            'indentation_levels': indentation_levels,
            'line_spacings': line_spacings,
            'capitalization_patterns': capitalization_patterns
        })
        
        # Adapt thresholds based on document characteristics
        self._adapt_thresholds()
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on document statistics."""
        # Calculate mean font size and store it for later use
        if self.document_context['font_sizes']:
            mean_font_size = statistics.mean(self.document_context['font_sizes'])
            font_variance = statistics.stdev(self.document_context['font_sizes']) if len(self.document_context['font_sizes']) > 1 else 2.0
            
            # Store calculated values in document context
            self.document_context['mean_font_size'] = mean_font_size
            self.document_context['font_size_variance'] = font_variance
            
            # Adapt font size difference threshold
            if font_variance > 3.0:
                self.adaptive_thresholds['font_size_difference_threshold'] = font_variance * 0.8
        else:
            # Default values if no font sizes
            self.document_context['mean_font_size'] = 12
            self.document_context['font_size_variance'] = 2.0
        
        # Adapt indentation cluster threshold
        if self.document_context['indentation_levels']:
            max_indentation = max(self.document_context['indentation_levels'])
            if max_indentation > 3:
                self.adaptive_thresholds['indentation_cluster_threshold'] = max_indentation * 5.0
        
        # Adapt spacing variance threshold
        if self.document_context['line_spacings']:
            mean_spacing = statistics.mean(self.document_context['line_spacings'])
            self.adaptive_thresholds['spacing_variance_threshold'] = mean_spacing * 1.5
    
    def extract_metadata(self, text: str) -> Dict[str, List[str]]:
        """Extract metadata from text using pattern matching with caching."""
        if not text:
            return {}
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        with self._cache_lock:
            if text_hash in self._detection_cache:
                return self._detection_cache[text_hash]
        
        metadata = {}
        
        for key, pattern in self._compiled_metadata_patterns.items():
            matches = pattern.findall(text)
            if matches:
                metadata[key] = [match if isinstance(match, str) else ' '.join(match) for match in matches]
        
        # Cache result
        with self._cache_lock:
            if len(self._detection_cache) < 1000:
                self._detection_cache[text_hash] = metadata
        
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
    """Algorithmic content classification based on text analysis with optimization."""
    
    def __init__(self):
        self.classifier_cache = {}
        self._classification_cache = {}
        self._cache_lock = threading.RLock()
        
        # Pre-compile patterns for better performance
        self._email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self._date_range_pattern = re.compile(r'\b\d{4}\s*[-–]\s*(Present|\d{4})\b')
        self._date_pattern = re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b')
        
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
        """Classify content based on text analysis with caching."""
        if not text:
            return "unknown", 0.0
        
        # Check cache first
        cache_key = f"{text}:{context}"
        with self._cache_lock:
            if cache_key in self._classification_cache:
                return self._classification_cache[cache_key]
        
        text_lower = text.lower()
        scores = {}
        
        # Keyword-based scoring
        for content_type, keywords in self.content_indicators.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            scores[content_type] = score / len(keywords)
        
        # Pattern-based scoring using compiled patterns
        if self._email_pattern.search(text):
            scores['contact_info'] = scores.get('contact_info', 0) + 0.5
        
        if self._date_range_pattern.search(text):
            scores['experience'] = scores.get('experience', 0) + 0.3
        
        if self._date_pattern.search(text):
            scores['experience'] = scores.get('experience', 0) + 0.2
        
        # Context-based scoring
        if context:
            if context.get('is_heading'):
                scores['section_title'] = 0.8
            if context.get('is_list_item'):
                scores['list_item'] = 0.7
        
        # Find best match
        if not scores:
            result = ("description", 0.5)
        else:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            result = (best_type, min(1.0, confidence))
        
        # Cache result
        with self._cache_lock:
            if len(self._classification_cache) < 1000:
                self._classification_cache[cache_key] = result
        
        return result


class UniversalHierarchicalBuilder:
    """Universal hierarchical builder with nested structure and adaptive heuristics."""
    
    def __init__(self):
        self.sections = []
        self.current_section = None
        self.current_subsection = None
        self.section_stack = []
        
        # Pre-compile patterns for better performance
        self._heading_patterns = [
            re.compile(r'^[A-Z][A-Z\s]+$'),
            re.compile(r'^\d+\.\s+[A-Z]'),
            re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'),
        ]
        
        self._list_patterns = [
            re.compile(r'^[•\-\*]\s+'),
            re.compile(r'^\d+\.\s+'),
            re.compile(r'^[a-z]\)\s+'),
        ]
        
        self._metadata_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'),
            'time': re.compile(r'\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b'),
            'currency': re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?'),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
        }
        
        # Universal document structure tracking
        self.document_structure = {
            'metadata': {},
            'sections': [],
            'document_metadata': {},
            'processing_metrics': {}
        }
        
        # Content aggregation for better organization
        self.content_buffer = []
        self.metadata_buffer = []
    
    def build_structure(self, elements: List[TextElement]) -> Dict[str, Any]:
        """Build universal hierarchical document structure with nested sections."""
        # Initialize universal document structure
        structure = {
            'metadata': {},
            'sections': [],
            'document_metadata': {
                'total_elements': len(elements),
                'document_type': self._detect_document_type(elements),
                'structure_confidence': 0.0
            },
            'processing_metrics': {
                'sections_created': 0,
                'subsections_created': 0,
                'metadata_extracted': 0,
                'content_organized': 0
            }
        }
        
        # Process elements sequentially with universal structure detection
        for element in elements:
            self._process_element_universal(element, structure)
        
        # Finalize universal structure
        self._finalize_universal_structure(structure)
        
        return structure
    
    def _detect_document_type(self, elements: List[TextElement]) -> str:
        """Detect document type based on content patterns."""
        if not elements:
            return 'unknown'
        
        # Analyze content patterns
        text_content = ' '.join([elem.text for elem in elements[:50]])  # Sample first 50 elements
        
        # Document type indicators
        document_indicators = {
            'resume': ['experience', 'education', 'skills', 'objective', 'summary'],
            'report': ['introduction', 'methodology', 'results', 'conclusion', 'analysis'],
            'agreement': ['terms', 'conditions', 'agreement', 'contract', 'signature'],
            'research_paper': ['abstract', 'introduction', 'literature', 'methodology', 'references'],
            'proposal': ['proposal', 'objective', 'budget', 'timeline', 'deliverables']
        }
        
        scores = {}
        for doc_type, indicators in document_indicators.items():
            score = sum(1 for indicator in indicators if indicator.lower() in text_content.lower())
            scores[doc_type] = score / len(indicators)
        
        if not scores or max(scores.values()) < 0.1:
            return 'general_document'
        
        return max(scores, key=scores.get)
    
    def _process_element_universal(self, element: TextElement, structure: Dict[str, Any]):
        """Process element with universal document understanding."""
        text = element.text.strip()
        if not text:
            return
        
        # Analyze element with universal context
        element_analysis = self._analyze_element_universal(element)
        
        # Route element based on analysis
        if element_analysis['is_section_title']:
            self._handle_section_title_universal(text, element, structure, element_analysis)
        elif element_analysis['is_subsection_title']:
            self._handle_subsection_title_universal(text, element, structure, element_analysis)
        elif element_analysis['is_list_item']:
            self._handle_list_item_universal(text, element, structure, element_analysis)
        elif element_analysis['is_metadata']:
            self._handle_metadata_universal(text, element, structure, element_analysis)
        else:
            self._handle_content_universal(text, element, structure, element_analysis)
    
    def _analyze_element_universal(self, element: TextElement) -> Dict[str, Any]:
        """Analyze element for universal document understanding."""
        text = element.text.strip()
        analysis = {
            'is_section_title': False,
            'is_subsection_title': False,
            'is_list_item': False,
            'is_metadata': False,
            'is_content': True,
            'confidence': 0.0,
            'level': 0
        }
        
        # Heading detection
        heading_level, confidence = self._detect_heading_universal(text, element)
        if confidence > 0.7:
            analysis['is_section_title'] = True
            analysis['confidence'] = confidence
            analysis['level'] = heading_level
        
        # Subsection detection
        elif confidence > 0.5:
            analysis['is_subsection_title'] = True
            analysis['confidence'] = confidence
            analysis['level'] = heading_level
        
        # List item detection
        is_list, list_type, list_level = self._detect_list_item_universal(text, element)
        if is_list:
            analysis['is_list_item'] = True
            analysis['confidence'] = 0.8
            analysis['level'] = list_level
        
        # Metadata detection
        metadata = self._extract_metadata_universal(text)
        if metadata:
            analysis['is_metadata'] = True
            analysis['confidence'] = 0.9
        
        # Content classification
        if not any([analysis['is_section_title'], analysis['is_subsection_title'], 
                   analysis['is_list_item'], analysis['is_metadata']]):
            analysis['is_content'] = True
            analysis['confidence'] = 0.5
        
        return analysis
    
    def _detect_heading_universal(self, text: str, element: TextElement) -> Tuple[int, float]:
        """Detect heading with universal document understanding."""
        confidence = 0.0
        level = 0
        
        # Pattern-based detection
        for i, pattern in enumerate(self._heading_patterns):
            if pattern.match(text):
                confidence += 0.3
                level = max(level, i + 1)
        
        # Font-based detection
        if element.font_size:
            if element.font_size > 16:
                confidence += 0.4
                level = max(level, 1)
            elif element.font_size > 14:
                confidence += 0.3
                level = max(level, 2)
            elif element.font_size > 12:
                confidence += 0.2
                level = max(level, 3)
        
        # Bold text detection
        if element.is_bold:
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
    
    def _detect_list_item_universal(self, text: str, element: TextElement) -> Tuple[bool, str, int]:
        """Detect list item with universal understanding."""
        for i, pattern in enumerate(self._list_patterns):
            if pattern.match(text):
                return True, pattern.pattern, i + 1
        
        # Indentation-based detection
        if element.indentation_level > 0 and len(text.split()) <= 10:
            return True, 'indented_list', element.indentation_level
        
        return False, "", 0
    
    def _extract_metadata_universal(self, text: str) -> Dict[str, List[str]]:
        """Extract metadata with universal patterns."""
        metadata = {}
        
        for pattern_name, pattern in self._metadata_patterns.items():
            matches = pattern.findall(text)
            if matches:
                metadata[pattern_name] = [match if isinstance(match, str) else ' '.join(match) for match in matches]
        
        return metadata
    
    def _handle_section_title_universal(self, text: str, element: TextElement, structure: Dict[str, Any], analysis: Dict[str, Any]):
        """Handle section title with universal structure."""
        # Create new section with nested structure
        section = {
            'title': text,
            'level': analysis['level'],
            'description': [],
            'metadata': {},
            'sub_sections': [],
            'confidence': analysis['confidence'],
            'page_number': element.page_number,
            'line_number': element.line_number
        }
        
        structure['sections'].append(section)
        self.current_section = section
        self.section_stack = [section]
        
        # Update metrics
        structure['processing_metrics']['sections_created'] += 1
    
    def _handle_subsection_title_universal(self, text: str, element: TextElement, structure: Dict[str, Any], analysis: Dict[str, Any]):
        """Handle subsection title with universal structure."""
        if not self.current_section:
            # Treat as main section if no current section
            self._handle_section_title_universal(text, element, structure, analysis)
            return
        
        # Create subsection
        subsection = {
            'title': text,
            'level': analysis['level'],
            'description': [],
            'metadata': {},
            'sub_sections': [],
            'confidence': analysis['confidence'],
            'page_number': element.page_number,
            'line_number': element.line_number
        }
        
        self.current_section['sub_sections'].append(subsection)
        self.current_subsection = subsection
        
        # Update metrics
        structure['processing_metrics']['subsections_created'] += 1
    
    def _handle_list_item_universal(self, text: str, element: TextElement, structure: Dict[str, Any], analysis: Dict[str, Any]):
        """Handle list item with universal structure."""
        # Add to current subsection or section
        if self.current_subsection:
            self.current_subsection['description'].append(text)
        elif self.current_section:
            self.current_section['description'].append(text)
        else:
            # Add to global content if no section context
            if 'content' not in structure:
                structure['content'] = []
            structure['content'].append(text)
        
        # Update metrics
        structure['processing_metrics']['content_organized'] += 1
    
    def _handle_metadata_universal(self, text: str, element: TextElement, structure: Dict[str, Any], analysis: Dict[str, Any]):
        """Handle metadata with universal structure."""
        metadata = self._extract_metadata_universal(text)
        
        if self.current_subsection:
            self.current_subsection['metadata'].update(metadata)
        elif self.current_section:
            self.current_section['metadata'].update(metadata)
        else:
            structure['metadata'].update(metadata)
        
        # Update metrics
        structure['processing_metrics']['metadata_extracted'] += len(metadata)
    
    def _handle_content_universal(self, text: str, element: TextElement, structure: Dict[str, Any], analysis: Dict[str, Any]):
        """Handle content with universal structure."""
        # Add to current subsection or section
        if self.current_subsection:
            self.current_subsection['description'].append(text)
        elif self.current_section:
            self.current_section['description'].append(text)
        else:
            # Add to global content if no section context
            if 'content' not in structure:
                structure['content'] = []
            structure['content'].append(text)
        
        # Update metrics
        structure['processing_metrics']['content_organized'] += 1
    
    def _finalize_universal_structure(self, structure: Dict[str, Any]):
        """Finalize universal document structure."""
        # Clean up empty sections
        self._clean_empty_sections_universal(structure['sections'])
        
        # Normalize content structure
        self._normalize_content_universal(structure)
        
        # Calculate structure confidence
        structure['document_metadata']['structure_confidence'] = self._calculate_structure_confidence(structure)
        
        # Merge fragmented content
        self._merge_fragmented_content_universal(structure)
    
    def _clean_empty_sections_universal(self, sections: List[Dict[str, Any]]):
        """Clean up empty sections in universal structure."""
        sections_to_remove = []
        
        for i, section in enumerate(sections):
            # Clean subsections recursively
            if section.get('sub_sections'):
                self._clean_empty_sections_universal(section['sub_sections'])
            
            # Remove if empty
            if (not section.get('description') and 
                not section.get('metadata') and 
                not section.get('sub_sections')):
                sections_to_remove.append(i)
        
        # Remove empty sections (in reverse order)
        for i in reversed(sections_to_remove):
            sections.pop(i)
    
    def _normalize_content_universal(self, structure: Dict[str, Any]):
        """Normalize content structure for universal documents."""
        for section in structure['sections']:
            self._normalize_section_content_universal(section)
    
    def _normalize_section_content_universal(self, section: Dict[str, Any]):
        """Normalize section content."""
        # Ensure all required fields exist
        if 'description' not in section:
            section['description'] = []
        if 'metadata' not in section:
            section['metadata'] = {}
        if 'sub_sections' not in section:
            section['sub_sections'] = []
        
        # Normalize subsections recursively
        for subsection in section['sub_sections']:
            self._normalize_section_content_universal(subsection)
    
    def _calculate_structure_confidence(self, structure: Dict[str, Any]) -> float:
        """Calculate confidence in structure detection."""
        total_sections = len(structure['sections'])
        if total_sections == 0:
            return 0.0
        
        # Calculate average confidence
        total_confidence = 0.0
        section_count = 0
        
        for section in structure['sections']:
            if 'confidence' in section:
                total_confidence += section['confidence']
                section_count += 1
        
        if section_count == 0:
            return 0.5  # Default confidence
        
        return total_confidence / section_count
    
    def _merge_fragmented_content_universal(self, structure: Dict[str, Any]):
        """Merge fragmented content in universal structure."""
        for section in structure['sections']:
            self._merge_section_content_universal(section)
    
    def _merge_section_content_universal(self, section: Dict[str, Any]):
        """Merge fragmented content within a section."""
        if not section.get('description'):
            return
        
        # Merge short content blocks
        merged_content = []
        current_block = ""
        
        for content in section['description']:
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
        
        section['description'] = merged_content
        
        # Process subsections recursively
        for subsection in section.get('sub_sections', []):
            self._merge_section_content_universal(subsection)
    
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
        """Check if text is a heading using compiled patterns."""
        # Short text with high confidence
        if len(text.split()) <= 6:
            # Check font characteristics
            if element.font_size and element.font_size > 12:
                return True
            if element.is_bold:
                return True
        
        # Pattern matching using compiled patterns
        for pattern in self._heading_patterns:
            if pattern.match(text):
                return True
        
        return False
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item using compiled patterns."""
        for pattern in self._list_patterns:
            if pattern.match(text):
                return True
        
        return False
    
    def _is_metadata(self, text: str) -> bool:
        """Check if text contains metadata using compiled patterns."""
        for pattern in self._metadata_patterns:
            if pattern.search(text):
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


class UniversalAlgorithmicDocumentParser:
    """Universal algorithmic document parser with adaptive heuristics for any document type."""
    
    def __init__(self, max_workers: int = 4, enable_caching: bool = True):
        self.text_analyzer = UniversalTextAnalyzer()
        self.structure_detector = UniversalStructureDetector()
        self.content_classifier = ContentClassifier()
        self.hierarchical_builder = UniversalHierarchicalBuilder()
        self.extractor = PDFExtractor(max_workers=max_workers, enable_caching=enable_caching)
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        
        logger.info(f"Universal Algorithmic Document Parser initialized with {max_workers} workers")
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse document using universal algorithmic approaches."""
        logger.info(f"Starting universal algorithmic document parsing: {file_path}")
        
        # Extract text elements
        extraction_result = self.extractor.extract_from_file(file_path)
        logger.info(f"Extracted {len(extraction_result.elements)} text elements")
        
        # Update document context for adaptive analysis
        self.structure_detector.update_document_context(extraction_result.elements)
        
        # Analyze text patterns with universal context
        analyzed_elements = self._analyze_elements_universal(extraction_result.elements)
        logger.info(f"Analyzed {len(analyzed_elements)} elements with universal context")
        
        # Build universal document structure
        structure = self.hierarchical_builder.build_structure(analyzed_elements)
        logger.info("Built universal hierarchical structure")
        
        # Add comprehensive document metadata
        structure['document_metadata'].update({
            'total_pages': extraction_result.total_pages,
            'extraction_method': extraction_result.extraction_method,
            'total_elements': len(extraction_result.elements),
            'file_path': file_path,
            'parser_version': 'universal_v2.0',
            'adaptive_thresholds': self.structure_detector.adaptive_thresholds
        })
        
        # Add processing metrics
        structure['processing_metrics'].update({
            'max_workers_used': self.max_workers,
            'caching_enabled': True,
            'adaptive_analysis': True
        })
        
        logger.info("Universal algorithmic document parsing completed")
        return structure
    
    def _analyze_elements_universal(self, elements: List[TextElement]) -> List[TextElement]:
        """Analyze elements with universal document understanding."""
        if len(elements) < 50:
            # For small documents, process sequentially
            return self._analyze_elements_sequential_universal(elements)
        
        # For larger documents, use parallel processing
        analyzed_elements = []
        batch_size = max(1, len(elements) // self.max_workers)
        
        # Create batches
        batches = [elements[i:i + batch_size] for i in range(0, len(elements), batch_size)]
        
        # Process batches in parallel
        future_to_batch = {
            self._executor.submit(self._analyze_batch_universal, batch): batch 
            for batch in batches
        }
        
        # Collect results
        for future in as_completed(future_to_batch):
            try:
                batch_result = future.result()
                analyzed_elements.extend(batch_result)
            except Exception as e:
                logger.error(f"Error analyzing batch: {e}")
                # Fallback to sequential processing for this batch
                batch = future_to_batch[future]
                batch_result = self._analyze_elements_sequential_universal(batch)
                analyzed_elements.extend(batch_result)
        
        # Sort by original order
        analyzed_elements.sort(key=lambda x: (x.page_number, x.line_number))
        return analyzed_elements
    
    def _analyze_elements_sequential_universal(self, elements: List[TextElement]) -> List[TextElement]:
        """Sequential analysis of elements with universal context."""
        analyzed_elements = []
        
        for element in elements:
            # Analyze text patterns with universal context
            context = {
                'font_size': element.font_size,
                'is_bold': element.is_bold,
                'indentation_level': element.indentation_level,
                'page_number': element.page_number,
                'line_number': element.line_number
            }
            
            analysis = self.text_analyzer.analyze_text_patterns(element.text, context)
            
            # Add analysis to element metadata
            element.metadata.update(analysis)
            
            # Classify content with universal understanding
            content_type, confidence = self.content_classifier.classify_content(
                element.text, 
                {
                    'font_size': element.font_size,
                    'is_bold': element.is_bold,
                    'is_heading': analysis.get('structure_signals', {}).get('is_potential_heading', False)
                }
            )
            
            element.metadata['content_type'] = content_type
            element.metadata['classification_confidence'] = confidence
            
            analyzed_elements.append(element)
        
        return analyzed_elements
    
    def _analyze_batch_universal(self, elements: List[TextElement]) -> List[TextElement]:
        """Analyze a batch of elements with universal context."""
        return self._analyze_elements_sequential_universal(elements)
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _analyze_elements(self, elements: List[TextElement]) -> List[TextElement]:
        """Analyze text elements and add analysis metadata with parallel processing."""
        if len(elements) < 50:
            # For small documents, process sequentially
            return self._analyze_elements_sequential(elements)
        
        # For larger documents, use parallel processing
        analyzed_elements = []
        batch_size = max(1, len(elements) // self.max_workers)
        
        # Create batches
        batches = [elements[i:i + batch_size] for i in range(0, len(elements), batch_size)]
        
        # Process batches in parallel
        future_to_batch = {
            self._executor.submit(self._analyze_batch, batch): batch 
            for batch in batches
        }
        
        # Collect results
        for future in as_completed(future_to_batch):
            try:
                batch_result = future.result()
                analyzed_elements.extend(batch_result)
            except Exception as e:
                logger.error(f"Error analyzing batch: {e}")
                # Fallback to sequential processing for this batch
                batch = future_to_batch[future]
                batch_result = self._analyze_elements_sequential(batch)
                analyzed_elements.extend(batch_result)
        
        # Sort by original order
        analyzed_elements.sort(key=lambda x: (x.page_number, x.line_number))
        return analyzed_elements
    
    def _analyze_elements_sequential(self, elements: List[TextElement]) -> List[TextElement]:
        """Sequential analysis of elements."""
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
    
    def _analyze_batch(self, elements: List[TextElement]) -> List[TextElement]:
        """Analyze a batch of elements."""
        return self._analyze_elements_sequential(elements)


class UniversalDocumentParserPipeline:
    """Universal document parsing pipeline with adaptive heuristics for any document type."""
    
    def __init__(self, max_workers: int = 4, enable_caching: bool = True):
        self.parser = UniversalAlgorithmicDocumentParser(max_workers=max_workers, enable_caching=enable_caching)
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        logger.info(f"Universal Document Parser Pipeline initialized with {max_workers} workers")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process any document type with universal adaptive parsing."""
        try:
            logger.info(f"Starting universal document processing: {file_path}")
            start_time = time.time()
            
            result = self.parser.parse_document(file_path)
            
            processing_time = time.time() - start_time
            logger.info(f"Universal document processing completed in {processing_time:.2f} seconds")
            
            # Add comprehensive performance metrics
            result['processing_metrics'].update({
                'processing_time_seconds': processing_time,
                'max_workers': self.max_workers,
                'caching_enabled': self.enable_caching,
                'universal_parsing': True,
                'adaptive_heuristics': True
            })
            
            return result
        except Exception as e:
            logger.error(f"Universal document processing failed: {e}")
            raise

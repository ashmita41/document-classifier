"""
Sophisticated question-answer identification system.

This module identifies Q&A pairs using:
- Multi-signal question detection
- Proximity and semantic answer matching
- Hierarchical Q&A structure support
- Answer quality assessment
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.text_element import TextElement


logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Type of question."""
    OPEN_ENDED = "open_ended"
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    TECHNICAL = "technical"
    ADMINISTRATIVE = "administrative"
    FINANCIAL = "financial"
    UNKNOWN = "unknown"


@dataclass
class QAPair:
    """Represents a question-answer pair."""
    question_text: str
    answer_text: Optional[str] = None
    question_number: Optional[str] = None
    question_type: QuestionType = QuestionType.UNKNOWN
    is_answered: bool = False
    confidence_score: float = 0.0
    
    # Context
    section_context: Optional[str] = None
    page_numbers: List[int] = field(default_factory=list)
    parent_question_id: Optional[str] = None
    sub_questions: List['QAPair'] = field(default_factory=list)
    
    # Metadata
    is_required: bool = True
    indentation_level: int = 0
    question_start_pos: int = 0
    answer_start_pos: Optional[int] = None
    
    # Quality metrics
    answer_length: int = 0
    answer_quality_score: float = 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        num_prefix = f"{self.question_number}. " if self.question_number else ""
        answered = "✓" if self.is_answered else "✗"
        return (
            f"QAPair({num_prefix}{self.question_text[:40]}... "
            f"[{answered}] conf={self.confidence_score:.2f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question_text': self.question_text,
            'answer_text': self.answer_text,
            'question_number': self.question_number,
            'question_type': self.question_type,
            'is_answered': self.is_answered,
            'confidence_score': self.confidence_score,
            'section_context': self.section_context,
            'page_numbers': self.page_numbers,
            'parent_question_id': self.parent_question_id,
            'sub_questions': [sq.to_dict() for sq in self.sub_questions],
            'is_required': self.is_required,
            'answer_quality_score': self.answer_quality_score,
        }


class QADetector:
    """
    Sophisticated Q&A detector for document analysis.
    
    Identifies questions and matches them with answers using
    multiple detection strategies and semantic understanding.
    """
    
    # Question patterns
    QUESTION_PATTERNS = [
        # Explicit question marks
        re.compile(r'([^.!?]*\?)', re.MULTILINE),
        # Interrogative words at start
        re.compile(
            r'\b(Who|What|When|Where|Why|How|Which|Can|Could|Would|Should|Will|Do|Does|Did|Is|Are|Was|Were)\b[^.!?]*\?',
            re.IGNORECASE
        ),
        # Imperative verbs
        re.compile(
            r'\b(Describe|Explain|Provide|List|Identify|Specify|Detail|Outline|State|Define|Discuss)\b[^.!?]*[:\?]?',
            re.IGNORECASE
        ),
        # Question-like phrases
        re.compile(
            r'\b(Please\s+(?:provide|describe|explain|list|identify)|'
            r'We\s+need\s+to\s+know|'
            r'Indicate\s+(?:whether|if|how)|'
            r'Clarify)\b[^.!?]*',
            re.IGNORECASE
        ),
    ]
    
    # Question numbering patterns
    NUMBERING_PATTERNS = [
        # 1., 1.1, 1.1.1
        re.compile(r'^(\d+(?:\.\d+)*)\.\s+'),
        # Q1:, Question 1:
        re.compile(r'^(?:Q|Question)\s*(\d+(?:\.\d+)*)[:\.\s]+', re.IGNORECASE),
        # a), b), (i), (ii)
        re.compile(r'^([a-z]|[ivxlcdm]+)\)\s+', re.IGNORECASE),
        # (a), (b), (1), (2)
        re.compile(r'^\(([a-z0-9]+)\)\s+', re.IGNORECASE),
    ]
    
    # Answer markers
    ANSWER_MARKERS = [
        re.compile(r'^(?:Answer|Response|A|Reply)[:\.\s]+', re.IGNORECASE),
        re.compile(r'^([a-z])\)\s+', re.IGNORECASE),  # a), b), c)
        re.compile(r'^\(([a-z])\)\s+', re.IGNORECASE),  # (a), (b), (c)
    ]
    
    # Placeholder patterns
    PLACEHOLDER_PATTERNS = [
        re.compile(r'^\s*(?:TBD|N/?A|To\s+be\s+(?:determined|filled|provided)|'
                   r'\[.*?\]|___+|\.\.\.|pending)\s*$', re.IGNORECASE),
    ]
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.6,
        max_answer_distance: int = 5,
        min_answer_length: int = 10
    ):
        """
        Initialize Q&A detector.
        
        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Minimum similarity for semantic matching
            max_answer_distance: Maximum lines between question and answer
            min_answer_length: Minimum length for valid answer
        """
        self.similarity_threshold = similarity_threshold
        self.max_answer_distance = max_answer_distance
        self.min_answer_length = min_answer_length
        
        # Load sentence transformer
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        logger.info("QADetector initialized")
    
    def detect_qa_pairs(
        self,
        elements: List[TextElement],
        section_context: Optional[str] = None
    ) -> List[QAPair]:
        """
        Detect Q&A pairs in document.
        
        Args:
            elements: List of TextElements
            section_context: Optional section name for context
            
        Returns:
            List of QAPair objects
        """
        logger.info(f"Detecting Q&A pairs in {len(elements)} elements")
        
        # Step 1: Identify questions
        questions = self._identify_questions(elements)
        logger.info(f"Found {len(questions)} potential questions")
        
        # Step 2: Find answers for each question
        qa_pairs = []
        for q_elem, q_number, q_confidence in questions:
            answer = self._find_answer(q_elem, elements, questions)
            
            # Create Q&A pair
            qa_pair = self._create_qa_pair(
                q_elem, answer, q_number, q_confidence, section_context
            )
            
            # Check for sub-questions
            sub_questions = self._detect_sub_questions(q_elem.text)
            if sub_questions:
                qa_pair.sub_questions = sub_questions
            
            qa_pairs.append(qa_pair)
        
        # Step 3: Build hierarchy
        qa_pairs = self._build_hierarchy(qa_pairs)
        
        logger.info(f"Detected {len(qa_pairs)} Q&A pairs")
        
        return qa_pairs
    
    def _identify_questions(
        self,
        elements: List[TextElement]
    ) -> List[Tuple[TextElement, Optional[str], float]]:
        """
        Identify questions using multiple signals.
        
        Returns:
            List of (element, question_number, confidence) tuples
        """
        questions = []
        
        for i, elem in enumerate(elements):
            text = elem.text.strip()
            
            if not text or len(text) < 5:
                continue
            
            # Calculate confidence from multiple signals
            confidence = 0.0
            question_number = None
            
            # Signal 1: Explicit question mark
            if '?' in text:
                confidence += 0.8
            
            # Signal 2: Check question patterns
            for pattern in self.QUESTION_PATTERNS:
                if pattern.search(text):
                    confidence += 0.5
                    break
            
            # Signal 3: Starts with interrogative word
            if self._starts_with_interrogative(text):
                confidence += 0.6
            
            # Signal 4: Has imperative verb
            if self._has_imperative_verb(text):
                confidence += 0.4
            
            # Signal 5: Check numbering
            number = self._extract_question_number(text)
            if number:
                question_number = number
                confidence += 0.7
            
            # Signal 6: Formatting (bold, larger font)
            if elem.is_bold:
                confidence += 0.2
            
            if elem.font_size and elem.font_size > 11:
                confidence += 0.1
            
            # Normalize confidence
            confidence = min(confidence, 1.0)
            
            # Threshold filter
            if confidence >= 0.5:
                questions.append((elem, question_number, confidence))
        
        return questions
    
    def _starts_with_interrogative(self, text: str) -> bool:
        """Check if text starts with interrogative word."""
        interrogatives = [
            'who', 'what', 'when', 'where', 'why', 'how', 'which',
            'can', 'could', 'would', 'should', 'will', 'do', 'does',
            'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had'
        ]
        
        first_word = text.split()[0].lower().strip('.,!?;:')
        return first_word in interrogatives
    
    def _has_imperative_verb(self, text: str) -> bool:
        """Check if text has imperative verb."""
        imperatives = [
            'describe', 'explain', 'provide', 'list', 'identify',
            'specify', 'detail', 'outline', 'state', 'define',
            'discuss', 'indicate', 'clarify', 'demonstrate'
        ]
        
        text_lower = text.lower()
        return any(verb in text_lower for verb in imperatives)
    
    def _extract_question_number(self, text: str) -> Optional[str]:
        """Extract question number from text."""
        for pattern in self.NUMBERING_PATTERNS:
            match = pattern.match(text)
            if match:
                return match.group(1)
        return None
    
    def _find_answer(
        self,
        question_elem: TextElement,
        all_elements: List[TextElement],
        questions: List[Tuple[TextElement, Optional[str], float]]
    ) -> Optional[Tuple[TextElement, float, str]]:
        """
        Find answer for a question.
        
        Returns:
            Tuple of (answer_element, confidence, method) or None
        """
        q_index = all_elements.index(question_elem)
        
        # Strategy 1: Proximity-based (immediate next elements)
        answer = self._find_answer_proximity(
            question_elem, all_elements, q_index, questions
        )
        if answer:
            return answer
        
        # Strategy 2: Indentation-based
        answer = self._find_answer_indentation(
            question_elem, all_elements, q_index, questions
        )
        if answer:
            return answer
        
        # Strategy 3: Explicit markers
        answer = self._find_answer_markers(
            question_elem, all_elements, q_index
        )
        if answer:
            return answer
        
        # Strategy 4: Semantic matching
        answer = self._find_answer_semantic(
            question_elem, all_elements, q_index, questions
        )
        if answer:
            return answer
        
        return None
    
    def _find_answer_proximity(
        self,
        question_elem: TextElement,
        all_elements: List[TextElement],
        q_index: int,
        questions: List[Tuple[TextElement, Optional[str], float]]
    ) -> Optional[Tuple[TextElement, float, str]]:
        """Find answer immediately following question."""
        question_elements = [q[0] for q in questions]
        
        # Look at next few elements
        for i in range(q_index + 1, min(q_index + self.max_answer_distance + 1, len(all_elements))):
            candidate = all_elements[i]
            
            # Skip if it's another question
            if candidate in question_elements:
                break
            
            # Check if valid answer
            if self._is_valid_answer(candidate):
                return (candidate, 0.8, 'proximity')
        
        return None
    
    def _find_answer_indentation(
        self,
        question_elem: TextElement,
        all_elements: List[TextElement],
        q_index: int,
        questions: List[Tuple[TextElement, Optional[str], float]]
    ) -> Optional[Tuple[TextElement, float, str]]:
        """Find answer based on indentation."""
        question_elements = [q[0] for q in questions]
        q_indent = question_elem.indentation_level or 0
        
        # Look for indented text after question
        candidates = []
        for i in range(q_index + 1, min(q_index + self.max_answer_distance + 1, len(all_elements))):
            candidate = all_elements[i]
            
            # Stop at next question
            if candidate in question_elements:
                break
            
            # Check if indented relative to question
            c_indent = candidate.indentation_level or 0
            if c_indent > q_indent and self._is_valid_answer(candidate):
                candidates.append(candidate)
        
        if candidates:
            # Combine multiple indented elements
            combined_text = ' '.join(c.text for c in candidates)
            # Return first element with combined confidence
            return (candidates[0], 0.85, 'indentation')
        
        return None
    
    def _find_answer_markers(
        self,
        question_elem: TextElement,
        all_elements: List[TextElement],
        q_index: int
    ) -> Optional[Tuple[TextElement, float, str]]:
        """Find answer with explicit markers."""
        # Look at next few elements
        for i in range(q_index + 1, min(q_index + self.max_answer_distance + 1, len(all_elements))):
            candidate = all_elements[i]
            text = candidate.text.strip()
            
            # Check for answer markers
            for pattern in self.ANSWER_MARKERS:
                if pattern.match(text):
                    return (candidate, 0.95, 'explicit_marker')
        
        return None
    
    def _find_answer_semantic(
        self,
        question_elem: TextElement,
        all_elements: List[TextElement],
        q_index: int,
        questions: List[Tuple[TextElement, Optional[str], float]]
    ) -> Optional[Tuple[TextElement, float, str]]:
        """Find answer using semantic similarity."""
        question_elements = [q[0] for q in questions]
        
        # Encode question
        q_embedding = self.model.encode([question_elem.text])[0]
        
        # Search within reasonable distance
        best_candidate = None
        best_similarity = 0.0
        
        for i in range(q_index + 1, min(q_index + 15, len(all_elements))):
            candidate = all_elements[i]
            
            # Skip other questions
            if candidate in question_elements:
                continue
            
            # Skip if too short
            if len(candidate.text.strip()) < self.min_answer_length:
                continue
            
            # Calculate similarity
            c_embedding = self.model.encode([candidate.text])[0]
            similarity = float(np.dot(q_embedding, c_embedding) / (
                np.linalg.norm(q_embedding) * np.linalg.norm(c_embedding)
            ))
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_candidate = candidate
        
        if best_candidate:
            return (best_candidate, best_similarity, 'semantic')
        
        return None
    
    def _is_valid_answer(self, element: TextElement) -> bool:
        """Check if element is a valid answer."""
        text = element.text.strip()
        
        # Too short
        if len(text) < self.min_answer_length:
            return False
        
        # Check for placeholders
        for pattern in self.PLACEHOLDER_PATTERNS:
            if pattern.match(text):
                return False
        
        return True
    
    def _create_qa_pair(
        self,
        question_elem: TextElement,
        answer: Optional[Tuple[TextElement, float, str]],
        question_number: Optional[str],
        question_confidence: float,
        section_context: Optional[str]
    ) -> QAPair:
        """Create Q&A pair from elements."""
        # Extract clean question text
        question_text = self._clean_question_text(question_elem.text, question_number)
        
        # Extract answer if found
        answer_text = None
        answer_confidence = 0.0
        answer_method = None
        answer_elem = None
        is_answered = False
        
        if answer:
            answer_elem, answer_confidence, answer_method = answer
            answer_text = self._clean_answer_text(answer_elem.text)
            is_answered = True
        
        # Classify question type
        question_type = self._classify_question_type(question_text)
        
        # Check if required
        is_required = self._is_required_question(question_text)
        
        # Calculate quality score
        answer_quality = 0.0
        answer_length = 0
        if answer_text:
            answer_length = len(answer_text)
            answer_quality = self._assess_answer_quality(answer_text, question_text)
        
        # Create Q&A pair
        qa_pair = QAPair(
            question_text=question_text,
            answer_text=answer_text,
            question_number=question_number,
            question_type=question_type,
            is_answered=is_answered,
            confidence_score=min(question_confidence, answer_confidence) if answer else question_confidence,
            section_context=section_context,
            page_numbers=[question_elem.page_number],
            is_required=is_required,
            indentation_level=question_elem.indentation_level or 0,
            question_start_pos=0,  # Would need index from full list
            answer_length=answer_length,
            answer_quality_score=answer_quality,
        )
        
        if answer_elem:
            if answer_elem.page_number not in qa_pair.page_numbers:
                qa_pair.page_numbers.append(answer_elem.page_number)
        
        return qa_pair
    
    def _clean_question_text(self, text: str, number: Optional[str]) -> str:
        """Clean question text."""
        # Remove numbering
        if number:
            for pattern in self.NUMBERING_PATTERNS:
                text = pattern.sub('', text)
        
        return text.strip()
    
    def _clean_answer_text(self, text: str) -> str:
        """Clean answer text."""
        # Remove answer markers
        for pattern in self.ANSWER_MARKERS:
            text = pattern.sub('', text)
        
        return text.strip()
    
    def _classify_question_type(self, question_text: str) -> QuestionType:
        """Classify question type."""
        text_lower = question_text.lower()
        
        # Yes/No questions
        if any(text_lower.startswith(word) for word in ['is', 'are', 'do', 'does', 'can', 'will', 'would']):
            return QuestionType.YES_NO
        
        # Technical keywords
        technical_keywords = [
            'technical', 'specification', 'architecture', 'design',
            'implementation', 'algorithm', 'system', 'software'
        ]
        if any(keyword in text_lower for keyword in technical_keywords):
            return QuestionType.TECHNICAL
        
        # Financial keywords
        financial_keywords = [
            'cost', 'price', 'budget', 'fee', 'payment', 'financial',
            'dollar', 'expense', 'revenue'
        ]
        if any(keyword in text_lower for keyword in financial_keywords):
            return QuestionType.FINANCIAL
        
        # Administrative keywords
        admin_keywords = [
            'company', 'organization', 'contact', 'address', 'phone',
            'email', 'representative', 'certificate', 'license'
        ]
        if any(keyword in text_lower for keyword in admin_keywords):
            return QuestionType.ADMINISTRATIVE
        
        return QuestionType.OPEN_ENDED
    
    def _is_required_question(self, question_text: str) -> bool:
        """Check if question is required."""
        optional_indicators = [
            'if applicable', 'if any', 'optional', 'if available',
            'if relevant', 'as appropriate'
        ]
        
        text_lower = question_text.lower()
        return not any(indicator in text_lower for indicator in optional_indicators)
    
    def _assess_answer_quality(self, answer_text: str, question_text: str) -> float:
        """Assess answer quality."""
        score = 0.0
        
        # Length score
        if len(answer_text) >= 50:
            score += 0.4
        elif len(answer_text) >= 20:
            score += 0.2
        
        # Not a placeholder
        is_placeholder = any(
            pattern.match(answer_text)
            for pattern in self.PLACEHOLDER_PATTERNS
        )
        if not is_placeholder:
            score += 0.3
        
        # Has substance (not just "Yes" or "No")
        if len(answer_text.split()) > 3:
            score += 0.2
        
        # Semantic relevance (simple keyword overlap)
        q_words = set(question_text.lower().split())
        a_words = set(answer_text.lower().split())
        overlap = len(q_words & a_words) / max(len(q_words), 1)
        score += min(overlap * 0.3, 0.3)
        
        return min(score, 1.0)
    
    def _detect_sub_questions(self, question_text: str) -> List[QAPair]:
        """Detect sub-questions within main question."""
        sub_questions = []
        
        # Pattern: "Describe: a) X, b) Y"
        sub_pattern = re.compile(
            r'([a-z]|[ivxlcdm]+)\)\s*([^,;]+?)(?=[a-z]\)|$)',
            re.IGNORECASE
        )
        
        matches = sub_pattern.findall(question_text)
        
        for letter, sub_text in matches:
            sub_qa = QAPair(
                question_text=sub_text.strip(),
                question_number=letter,
                confidence_score=0.7,
                is_answered=False,
            )
            sub_questions.append(sub_qa)
        
        return sub_questions
    
    def _build_hierarchy(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """Build hierarchical structure from Q&A pairs."""
        if not qa_pairs:
            return []
        
        # Identify parent-child relationships based on numbering
        root_pairs = []
        
        for qa in qa_pairs:
            if not qa.question_number:
                root_pairs.append(qa)
                continue
            
            # Check if this is a sub-question (has dot or letter)
            if '.' in qa.question_number or len(qa.question_number) == 1:
                # Try to find parent
                parent = self._find_parent_question(qa, qa_pairs)
                if parent:
                    parent.sub_questions.append(qa)
                    qa.parent_question_id = parent.question_number
                else:
                    root_pairs.append(qa)
            else:
                root_pairs.append(qa)
        
        return root_pairs
    
    def _find_parent_question(
        self,
        child: QAPair,
        all_pairs: List[QAPair]
    ) -> Optional[QAPair]:
        """Find parent question for a child."""
        if not child.question_number:
            return None
        
        # If numbered like 1.1, parent is 1
        if '.' in child.question_number:
            parent_num = child.question_number.rsplit('.', 1)[0]
            for qa in all_pairs:
                if qa.question_number == parent_num:
                    return qa
        
        return None
    
    def get_unanswered_questions(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """Get list of unanswered questions."""
        unanswered = []
        
        def collect_unanswered(pairs):
            for qa in pairs:
                if not qa.is_answered:
                    unanswered.append(qa)
                if qa.sub_questions:
                    collect_unanswered(qa.sub_questions)
        
        collect_unanswered(qa_pairs)
        return unanswered
    
    def get_statistics(self, qa_pairs: List[QAPair]) -> Dict[str, Any]:
        """Get statistics about Q&A pairs."""
        def count_recursive(pairs):
            total = len(pairs)
            answered = sum(1 for qa in pairs if qa.is_answered)
            for qa in pairs:
                if qa.sub_questions:
                    sub_total, sub_answered = count_recursive(qa.sub_questions)
                    total += sub_total
                    answered += sub_answered
            return total, answered
        
        total, answered = count_recursive(qa_pairs)
        
        # Count by type
        type_counts = {}
        for qa in qa_pairs:
            type_counts[qa.question_type] = type_counts.get(qa.question_type, 0) + 1
        
        return {
            'total_questions': total,
            'answered': answered,
            'unanswered': total - answered,
            'answer_rate': answered / total if total > 0 else 0,
            'by_type': type_counts,
        }


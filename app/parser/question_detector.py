"""
Enhanced Question and Answer Detector for RFP Documents

This module provides comprehensive functionality to detect questions, sub-questions,
and answers in RFP documents with proper multi-line support and hierarchy.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Represents a detected question in the document."""
    question_number: str
    text: str
    line_index: int
    parent_section: str
    confidence: str  # "high", "medium", "low"
    page_number: int
    y_position: float
    is_multiline: bool = False
    merged_lines: List[int] = None  # Line indices of merged lines
    is_sub_question: bool = False
    parent_question_id: Optional[str] = None
    answer_text: Optional[str] = None
    is_answered: bool = False


class EnhancedQuestionDetector:
    """Enhanced question detector with comprehensive pattern matching and multi-line support."""
    
    def __init__(self):
        self.sections = []
        self.questions = []
        self.debug_info = {
            "total_lines_analyzed": 0,
            "question_candidates_found": 0,
            "questions_confirmed": 0,
            "multiline_questions": 0,
            "sub_questions": 0,
            "answers_detected": 0
        }
        
        # Expanded question patterns
        self.interrogative_patterns = [
            r'^Who\s+', r'^What\s+', r'^When\s+', r'^Where\s+', r'^Why\s+', 
            r'^How\s+', r'^Which\s+'
        ]
        
        self.imperative_patterns = [
            r'^Please\s+(describe|provide|explain|list|identify|detail|outline)',
            r'^Describe\s+', r'^Provide\s+', r'^List\s+', r'^Identify\s+',
            r'^Explain\s+', r'^Detail\s+', r'^Outline\s+'
        ]
        
        self.request_patterns = [
            r'^Is\s+your\s+', r'^Do\s+you\s+', r'^Does\s+your\s+', 
            r'^Can\s+you\s+', r'^Will\s+you\s+', r'^Are\s+you\s+', 
            r'^Have\s+you\s+'
        ]
        
        self.bullet_patterns = [
            r'^[●•▪▫◦‣⁃]\s+',  # Various bullet symbols
            r'^[a-z]\)\s+',      # Lowercase letters with parenthesis
            r'^[ivx]+\)\s+',     # Roman numerals with parenthesis
            r'^-\s+',            # Dash bullets
            r'^\*\s+'            # Asterisk bullets
        ]
    
    def detect_questions_and_answers(self, lines: List[Dict[str, Any]], sections: List[Any]) -> List[Question]:
        """
        Comprehensive question and answer detection with multi-line support.
        
        Args:
            lines: List of line dictionaries with text and formatting info
            sections: List of detected sections for context
            
        Returns:
            List of detected questions with answers and hierarchy
        """
        if not lines:
            return []
        
        self.sections = sections
        self.debug_info["total_lines_analyzed"] = len(lines)
        
        logger.info(f"Starting enhanced question detection on {len(lines)} lines with {len(sections)} sections")
        
        # Step 1: Detect all question candidates (including multi-line)
        question_candidates = self._detect_question_candidates(lines)
        
        # Step 2: Process multi-line questions
        questions = self._process_multiline_questions(question_candidates, lines)
        
        # Step 3: Detect sub-questions and bullet points
        questions = self._detect_sub_questions(questions, lines)
        
        # Step 4: Detect answers for answered RFPs
        questions = self._detect_answers(questions, lines)
        
        # Step 5: Link to parent sections
        questions = self._link_to_parent_sections(questions)
        
        self.debug_info["questions_confirmed"] = len(questions)
        logger.info(f"Detected {len(questions)} questions with enhanced patterns")
        
        return questions
    
    def _detect_question_candidates(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect all potential question candidates using expanded patterns."""
        candidates = []
        
        for i, line in enumerate(lines):
            text = line.get('text', '').strip()
            
            # Skip empty or very short lines
            if len(text) < 3:
                continue
            
            # RULE 1: Numbered questions (highest priority)
            numbered_match = re.match(r'^(\d{1,2})\.\s+(.+)', text)
            if numbered_match:
                q_num = numbered_match.group(1)
                q_text = numbered_match.group(2)
                
                candidates.append({
                    'type': 'numbered',
                    'number': q_num,
                    'text': q_text,
                    'line_index': i,
                    'confidence': 'high',
                    'line': line
                })
                continue
            
            # RULE 2: Explicit question marks (but not bullet questions)
            if text.endswith('?') and not any(re.match(pattern, text) for pattern in self.bullet_patterns):
                candidates.append({
                    'type': 'explicit',
                    'number': '',
                    'text': text,
                    'line_index': i,
                    'confidence': 'high',
                    'line': line
                })
                continue
            
            # RULE 3: Interrogative patterns (but not bullet questions)
            if not any(re.match(pattern, text) for pattern in self.bullet_patterns):
                for pattern in self.interrogative_patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        candidates.append({
                            'type': 'interrogative',
                            'number': '',
                            'text': text,
                            'line_index': i,
                            'confidence': 'medium',
                            'line': line
                        })
                        break
                
                # RULE 4: Imperative patterns
                for pattern in self.imperative_patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        candidates.append({
                            'type': 'imperative',
                            'number': '',
                            'text': text,
                            'line_index': i,
                            'confidence': 'medium',
                            'line': line
                        })
                        break
                
                # RULE 5: Request patterns
                for pattern in self.request_patterns:
                    if re.match(pattern, text, re.IGNORECASE):
                        candidates.append({
                            'type': 'request',
                            'number': '',
                            'text': text,
                            'line_index': i,
                            'confidence': 'medium',
                            'line': line
                        })
                        break
        
        return candidates
    
    def _process_multiline_questions(self, candidates: List[Dict[str, Any]], lines: List[Dict[str, Any]]) -> List[Question]:
        """Process multi-line questions by joining continuation lines."""
        questions = []
        processed_lines = set()
        
        for candidate in candidates:
            if candidate['line_index'] in processed_lines:
                continue
                
            line_index = candidate['line_index']
            text = candidate['text']
            merged_lines = [line_index]
            
            # Check for continuation lines
            i = line_index + 1
            while i < len(lines):
                next_line = lines[i]
                next_text = next_line.get('text', '').strip()
                
                # Stop if we hit another question or section
                if (self._is_question_start(next_text) or 
                    self._is_section_start(next_text) or
                    not next_text):
                    break
                
                # Check if this looks like a continuation
                if self._is_continuation_line(next_text, text):
                    text += ' ' + next_text
                    merged_lines.append(i)
                    processed_lines.add(i)
                else:
                    break
                
                i += 1
            
            # Create question object
            question = Question(
                question_number=candidate['number'],
                text=text,
                line_index=line_index,
                parent_section="",  # Will be set later
                confidence=candidate['confidence'],
                page_number=candidate['line'].get('page_number', 1),
                y_position=candidate['line'].get('y_position', 0),
                is_multiline=len(merged_lines) > 1,
                merged_lines=merged_lines
            )
            questions.append(question)
            processed_lines.add(line_index)
        
        return questions
    
    def _is_question_start(self, text: str) -> bool:
        """Check if text starts a new question."""
        return (re.match(r'^\d+\.\s+', text) or 
                text.endswith('?') or
                any(re.match(pattern, text, re.IGNORECASE) for pattern in self.interrogative_patterns) or
                any(re.match(pattern, text, re.IGNORECASE) for pattern in self.imperative_patterns) or
                any(re.match(pattern, text, re.IGNORECASE) for pattern in self.request_patterns))
    
    def _is_section_start(self, text: str) -> bool:
        """Check if text starts a new section."""
        return (re.match(r'^[IVX]+\.\s+[A-Z]', text) or  # Roman numerals
                re.match(r'^[A-J]\.\s+[A-Z]', text))     # Letters
    
    def _is_continuation_line(self, next_text: str, current_text: str) -> bool:
        """Check if next line is a continuation of current text."""
        # Skip empty lines
        if not next_text:
            return False
        
        # Must not start with capital letter (likely new sentence)
        if next_text and next_text[0].isupper():
            return False
        
        # Must not end with period (likely complete sentence)
        if next_text.endswith('.'):
            return False
        
        # Must not be a question or section start
        if (self._is_question_start(next_text) or 
            self._is_section_start(next_text)):
            return False
        
        # Must be reasonable length (not too short or too long)
        return 3 <= len(next_text) <= 150
    
    def _detect_sub_questions(self, questions: List[Question], lines: List[Dict[str, Any]]) -> List[Question]:
        """Detect sub-questions and bullet points."""
        enhanced_questions = []
        processed_lines = set()
        
        for question in questions:
            enhanced_questions.append(question)
            processed_lines.update(question.merged_lines)
            
            # Look for sub-questions after this question
            start_line = max(question.merged_lines) + 1
            end_line = self._find_next_question_line(start_line, questions, lines)
            
            sub_questions = self._find_sub_questions_in_range(
                lines, start_line, end_line, question, processed_lines
            )
            enhanced_questions.extend(sub_questions)
            # Mark sub-question lines as processed
            for sub_q in sub_questions:
                processed_lines.update(sub_q.merged_lines)
        
        return enhanced_questions
    
    def _find_next_question_line(self, start_line: int, questions: List[Question], lines: List[Dict[str, Any]]) -> int:
        """Find the line index of the next question."""
        next_question_line = len(lines)
        
        for q in questions:
            if q.line_index > start_line and q.line_index < next_question_line:
                next_question_line = q.line_index
        
        return next_question_line
    
    def _find_sub_questions_in_range(self, lines: List[Dict[str, Any]], 
                                   start_line: int, end_line: int, 
                                   parent_question: Question, processed_lines: set) -> List[Question]:
        """Find sub-questions in a specific line range."""
        sub_questions = []
        
        for i in range(start_line, min(end_line, len(lines))):
            if i in processed_lines:
                continue
                
            line = lines[i]
            text = line.get('text', '').strip()
            
            if not text:
                continue
            
            # Check for bullet patterns
            for pattern in self.bullet_patterns:
                if re.match(pattern, text):
                    # Extract text after bullet
                    bullet_text = re.sub(pattern, '', text).strip()
                    
                    sub_question = Question(
                        question_number="",
                        text=bullet_text,
                        line_index=i,
                        parent_section=parent_question.parent_section,
                        confidence="medium",
                        page_number=line.get('page_number', 1),
                        y_position=line.get('y_position', 0),
                        is_multiline=False,
                        merged_lines=[i],
                        is_sub_question=True,
                        parent_question_id=f"q_{parent_question.line_index}"
                    )
                    sub_questions.append(sub_question)
                    break
        
        return sub_questions
    
    def _detect_answers(self, questions: List[Question], lines: List[Dict[str, Any]]) -> List[Question]:
        """Detect answers for answered RFPs."""
        for question in questions:
            if question.is_sub_question:
                continue
                
            # Look for answer patterns after the question
            start_line = max(question.merged_lines) + 1
            end_line = self._find_next_question_line(start_line, questions, lines)
            
            answer_text = self._extract_answer(lines, start_line, end_line)
            if answer_text:
                question.answer_text = answer_text
                question.is_answered = True
                self.debug_info["answers_detected"] += 1
        
        return questions
    
    def _extract_answer(self, lines: List[Dict[str, Any]], start_line: int, end_line: int) -> Optional[str]:
        """Extract answer text from a line range."""
        answer_lines = []
        
        for i in range(start_line, min(end_line, len(lines))):
            line = lines[i]
            text = line.get('text', '').strip()
            
            if not text:
                continue
            
            # Look for answer indicators
            if re.match(r'^(Answer|Response|A):\s*', text, re.IGNORECASE):
                # Extract text after indicator
                answer_text = re.sub(r'^(Answer|Response|A):\s*', '', text, flags=re.IGNORECASE)
                if answer_text:
                    answer_lines.append(answer_text)
            elif answer_lines:  # Continuation of answer
                answer_lines.append(text)
        
        return ' '.join(answer_lines) if answer_lines else None
    
    def _link_to_parent_sections(self, questions: List[Question]) -> List[Question]:
        """Link questions to their parent sections."""
        for question in questions:
            question.parent_section = self._find_parent_section(question.line_index)
        
        return questions
    
    def _is_valid_question_text(self, text: str) -> bool:
        """Validate that text is actually a question, not a description."""
        # Basic validation - most filtering is done by pattern matching
        if len(text) < 5:
            return False
        
        # Skip obvious descriptions
        description_indicators = [
            "Acme Lab has", "research program", "hospital in which",
            "system/database", "existing system", "shadow systems"
        ]
        
        text_lower = text.lower()
        if any(indicator.lower() in text_lower for indicator in description_indicators):
            return False
        
        return True
    
    def _find_parent_section(self, line_index: int) -> str:
        """Find the most recent section above the given line index."""
        if not self.sections:
            return "preamble"
        
        # Find the most recent section before this line
        parent_section = "preamble"
        for section in self.sections:
            if section.get('line_index', 0) < line_index:
                parent_section = section.get('title', 'unknown')
            else:
                break
        
        return parent_section


def detect_questions(lines: List[Dict[str, Any]], sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhanced function to detect questions and answers from PDF lines with section context.
    
    Args:
        lines: List of line dictionaries with text and formatting info
        sections: List of detected sections for context
        
    Returns:
        List of question dictionaries with detection results
    """
    detector = EnhancedQuestionDetector()
    questions = detector.detect_questions_and_answers(lines, sections)
    
    # Convert to dictionary format
    result = []
    for question in questions:
        result.append({
            "question_number": question.question_number,
            "text": question.text,
            "line_index": question.line_index,
            "parent_section": question.parent_section,
            "confidence": question.confidence,
            "page_number": question.page_number,
            "y_position": question.y_position,
            "is_multiline": question.is_multiline,
            "merged_lines": question.merged_lines or [question.line_index],
            "is_sub_question": question.is_sub_question,
            "parent_question_id": question.parent_question_id,
            "answer_text": question.answer_text,
            "is_answered": question.is_answered
        })
    
    return result

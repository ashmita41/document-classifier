"""
Question Classifier for RFP Documents

This module provides functionality to detect and classify questions in RFP documents
by working with the section detector to establish proper context and hierarchy.
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


class QuestionClassifier:
    """Classifies questions in RFP documents with section context."""
    
    def __init__(self):
        self.sections = []
        self.questions = []
        self.debug_info = {
            "total_lines_analyzed": 0,
            "question_candidates_found": 0,
            "questions_confirmed": 0,
            "multiline_questions": 0
        }
    
    def classify_questions(self, lines: List[Dict[str, Any]], sections: List[Any]) -> List[Question]:
        """
        Classify questions in the document with section context.
        Uses strict single-line detection to avoid false positives.
        
        Args:
            lines: List of line dictionaries with text and formatting info
            sections: List of detected sections for context
            
        Returns:
            List of detected questions with parent section information
        """
        if not lines:
            return []
        
        self.sections = sections
        self.debug_info["total_lines_analyzed"] = len(lines)
        
        logger.info(f"Starting question classification on {len(lines)} lines with {len(sections)} sections")
        
        # Detect questions using strict rules (NO multiline merging)
        questions = self._detect_questions_strict(lines)
        
        self.debug_info["questions_confirmed"] = len(questions)
        logger.info(f"Detected {len(questions)} questions using strict single-line rules")
        
        return questions
    
    def _detect_questions_strict(self, lines: List[Dict[str, Any]]) -> List[Question]:
        """Detect questions using strict rules to avoid false positives."""
        questions = []
        
        for i, line in enumerate(lines):
            text = line.get('text', '').strip()
            
            # Skip empty or very short lines
            if len(text) < 5:
                continue
            
            # RULE 1: Numbered questions (highest priority)
            numbered_match = re.match(r'^(\d{1,2})\.\s+(.+)', text)
            if numbered_match:
                q_num = numbered_match.group(1)
                q_text = numbered_match.group(2)
                
                # Additional validation for numbered questions
                if self._is_valid_question_text(q_text):
                    parent = self._find_parent_section(i)
                    
                    question = Question(
                        question_number=q_num,
                        text=q_text,
                        line_index=i,
                        parent_section=parent,
                        confidence="high",
                        page_number=line.get('page_number', 1),
                        y_position=line.get('y_position', 0),
                        is_multiline=False,
                        merged_lines=[i]
                    )
                    questions.append(question)
                continue
            
            # RULE 2: Ends with "?"
            if text.endswith('?') and len(text) < 200:
                if self._is_valid_question_text(text):
                    parent = self._find_parent_section(i)
                    
                    question = Question(
                        question_number="",
                        text=text,
                        line_index=i,
                        parent_section=parent,
                        confidence="medium",
                        page_number=line.get('page_number', 1),
                        y_position=line.get('y_position', 0),
                        is_multiline=False,
                        merged_lines=[i]
                    )
                    questions.append(question)
                continue
            
            # RULE 3: Starts with question word + be under 100 characters
            question_starters = [
                'How ', 'What ', 'When ', 'Where ', 'Why ', 'Who ',
                'Please provide', 'Please describe', 'Please list',
                'Would you', 'Will you', 'Can you', 'Do you'
            ]
            
            if any(text.startswith(starter) for starter in question_starters):
                # Additional checks to avoid false positives
                if (len(text) < 100 and 
                    text.count('. ') < 2 and  # Not multiple sentences
                    self._is_valid_question_text(text)):
                    parent = self._find_parent_section(i)
                    
                    question = Question(
                        question_number="",
                        text=text,
                        line_index=i,
                        parent_section=parent,
                        confidence="medium",
                        page_number=line.get('page_number', 1),
                        y_position=line.get('y_position', 0),
                        is_multiline=False,
                        merged_lines=[i]
                    )
                    questions.append(question)
        
        return questions
    
    def _is_valid_question_text(self, text: str) -> bool:
        """Validate that text is actually a question, not a description."""
        # NOT QUESTIONS if:
        # - Line is longer than 200 characters (likely paragraph)
        if len(text) > 200:
            return False
        
        # - Contains multiple sentences (check for ". " in middle)
        if text.count('. ') > 1:
            return False
        
        # - Starts with lowercase letter (likely continuation)
        if text and text[0].islower():
            return False
        
        # - Contains description indicators
        description_indicators = [
            "Acme Lab has", "research program", "hospital in which",
            "system/database", "existing system", "shadow systems",
            "absence of an", "integrated system", "proposal and award",
            "track proposal", "award information", "grant management",
            "electronic research", "administration system", "grants management",
            "The research administration", "processes at", "hospital in which",
            "system that", "database that", "existing system", "shadow systems",
            "absence of", "integrated system", "proposal and", "award information"
        ]
        
        text_lower = text.lower()
        if any(indicator.lower() in text_lower for indicator in description_indicators):
            return False
        
        # - Contains company/organization names (likely descriptions)
        org_indicators = ["Acme Lab", "Generic Company", "hospital", "clinicians", "research program"]
        if any(indicator in text for indicator in org_indicators):
            return False
        
        # - Contains paragraph-like indicators
        paragraph_indicators = [
            "The research", "processes at", "hospital in which", "system that",
            "database that", "existing system", "shadow systems", "absence of",
            "integrated system", "proposal and", "award information", "grant management"
        ]
        
        if any(indicator in text for indicator in paragraph_indicators):
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
    Main function to detect questions from PDF lines with section context.
    
    Args:
        lines: List of line dictionaries with text and formatting info
        sections: List of detected sections for context
        
    Returns:
        List of question dictionaries with detection results
    """
    classifier = QuestionClassifier()
    questions = classifier.classify_questions(lines, sections)
    
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
            "merged_lines": question.merged_lines or [question.line_index]
        })
    
    return result

"""
Metadata Extractor for PDF Documents

This module provides functionality to extract document-level metadata including
dates, contact information, and document information from PDF text.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import dateparser
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Represents extracted document metadata."""
    dates: Dict[str, str]
    contacts: Dict[str, str]
    document_info: Dict[str, str]


class MetadataExtractor:
    """Extracts metadata from PDF documents."""
    
    # Date-related keywords
    DATE_KEYWORDS = [
        'due date', 'deadline', 'respond by', 'submit by', 'proposal due',
        'effective date', 'start date', 'end date', 'submission deadline',
        'response due', 'proposal deadline', 'bid due', 'application due'
    ]
    
    # Contact-related keywords
    CONTACT_KEYWORDS = [
        'contact:', 'attn:', 'point of contact:', 'contact person:',
        'for questions contact:', 'inquiries:', 'questions:'
    ]
    
    # Document type patterns
    DOCUMENT_TYPES = [
        'Request For Proposal', 'RFP', 'RFI', 'Resume', 'CV', 
        'Proposal', 'Statement of Work', 'SOW', 'Contract',
        'Agreement', 'Terms and Conditions', 'Terms of Service'
    ]
    
    def __init__(self):
        self.metadata = {
            'dates': {},
            'contacts': {},
            'document_info': {}
        }
    
    def extract_metadata(self, lines: List[Dict[str, Any]]) -> DocumentMetadata:
        """
        Extract metadata from PDF lines.
        
        Args:
            lines: List of line dictionaries with text and formatting info
            
        Returns:
            DocumentMetadata object with extracted information
        """
        if not lines:
            return DocumentMetadata({}, {}, {})
        
        logger.info(f"Starting metadata extraction from {len(lines)} lines")
        
        # Extract dates
        self._extract_dates(lines)
        
        # Extract contact information
        self._extract_contacts(lines)
        
        # Extract document information
        self._extract_document_info(lines)
        
        logger.info(f"Extracted metadata: {len(self.metadata['dates'])} dates, {len(self.metadata['contacts'])} contacts, {len(self.metadata['document_info'])} doc info")
        
        return DocumentMetadata(
            dates=self.metadata['dates'],
            contacts=self.metadata['contacts'],
            document_info=self.metadata['document_info']
        )
    
    def _extract_dates(self, lines: List[Dict[str, Any]]) -> None:
        """Extract dates and deadlines from text."""
        logger.debug("Extracting dates and deadlines")
        
        for line in lines:
            text = line.get('text', '').strip()
            text_lower = text.lower()
            
            # Check for date-related keywords
            for keyword in self.DATE_KEYWORDS:
                if keyword in text_lower:
                    # Try to parse date using dateparser
                    date_obj = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
                    if date_obj:
                        # Create a clean key from the keyword
                        key = keyword.replace(' ', '_').replace('_by', '')
                        self.metadata['dates'][key] = date_obj.strftime('%Y-%m-%d')
                        logger.debug(f"Found date: {key} = {date_obj.strftime('%Y-%m-%d')}")
                    break
    
    def _extract_contacts(self, lines: List[Dict[str, Any]]) -> None:
        """Extract contact information from text."""
        logger.debug("Extracting contact information")
        
        for line in lines:
            text = line.get('text', '').strip()
            text_lower = text.lower()
            
            # Check for contact-related keywords
            for keyword in self.CONTACT_KEYWORDS:
                if keyword in text_lower:
                    # Extract name after the keyword
                    pattern = rf"{re.escape(keyword)}\s*(.+?)(?:\n|$)"
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        contact_name = match.group(1).strip()
                        if contact_name and len(contact_name) > 2:
                            self.metadata['contacts']['contact_person'] = contact_name
                            logger.debug(f"Found contact person: {contact_name}")
                    break
            
            # Extract email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_match = re.search(email_pattern, text)
            if email_match:
                self.metadata['contacts']['email'] = email_match.group(0)
                logger.debug(f"Found email: {email_match.group(0)}")
            
            # Extract phone numbers
            phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            phone_match = re.search(phone_pattern, text)
            if phone_match:
                self.metadata['contacts']['phone'] = phone_match.group(0)
                logger.debug(f"Found phone: {phone_match.group(0)}")
            
            # Extract company names (all caps, 3-5 words)
            if (text.isupper() and 
                3 <= len(text.split()) <= 5 and 
                not text.startswith('PAGE') and
                not text.startswith('SECTION') and
                not text.startswith('REQUEST')):
                self.metadata['contacts']['company'] = text
                logger.debug(f"Found company: {text}")
    
    def _extract_document_info(self, lines: List[Dict[str, Any]]) -> None:
        """Extract document information from first page."""
        logger.debug("Extracting document information")
        
        # Get first page lines
        first_page_lines = [line for line in lines if line.get('page_number', 1) == 1][:10]
        
        if not first_page_lines:
            return
        
        # Find document title (largest font size, top of page)
        sorted_by_font = sorted(first_page_lines, key=lambda x: x.get('font_size', 0), reverse=True)
        if sorted_by_font:
            title_candidate = sorted_by_font[0].get('text', '').strip()
            if title_candidate and len(title_candidate) > 5:
                self.metadata['document_info']['title'] = title_candidate
                logger.debug(f"Found title: {title_candidate}")
        
        # Find organization name (header - all caps, short)
        for line in first_page_lines:
            text = line.get('text', '').strip()
            if (text.isupper() and 
                3 <= len(text.split()) <= 5 and
                not text.startswith('PAGE') and
                not text.startswith('SECTION') and
                not text.startswith('REQUEST')):
                self.metadata['document_info']['organization'] = text
                logger.debug(f"Found organization: {text}")
                break
        
        # Detect document type from all text
        all_text = ' '.join(line.get('text', '') for line in lines[:20])
        for doc_type in self.DOCUMENT_TYPES:
            if doc_type.lower() in all_text.lower():
                self.metadata['document_info']['document_type'] = doc_type
                logger.debug(f"Found document type: {doc_type}")
                break


def extract_metadata(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to extract metadata from PDF lines.
    
    Args:
        lines: List of line dictionaries with text and formatting info
        
    Returns:
        Dictionary with extracted metadata in the expected format
    """
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(lines)
    
    # Return in the expected format
    return {
        'dates': metadata.dates,
        'contacts': metadata.contacts,
        'document_info': metadata.document_info
    }

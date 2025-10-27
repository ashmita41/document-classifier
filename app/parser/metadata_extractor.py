"""
Comprehensive Metadata Extractor for PDF Documents

This module provides comprehensive functionality to extract all types of metadata including
dates, emails, contact details, phone numbers, and other information from PDF text.
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
    emails: List[str]
    phone_numbers: List[str]
    contact_details: Dict[str, str]
    numeric_data: Dict[str, str]
    document_type: str
    total_pages: int


class ComprehensiveMetadataExtractor:
    """Comprehensive metadata extractor for all types of information."""
    
    def __init__(self):
        self.metadata = {
            'dates': {},
            'emails': [],
            'phone_numbers': [],
            'contact_details': {},
            'numeric_data': {},
            'document_type': 'Unknown',
            'total_pages': 0
        }
        
        # Comprehensive date patterns
        self.date_patterns = [
            # Month DD, YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            # Month DD
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}',
            # MM/DD/YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',
            # YYYY-MM-DD
            r'\d{4}-\d{2}-\d{2}',
            # DD/MM/YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',
            # Month YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            # Day of week patterns
            r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        
        # Date context keywords
        self.date_context_keywords = [
            'due date', 'deadline', 'respond by', 'submit by', 'proposal due',
            'effective date', 'start date', 'end date', 'submission deadline',
            'response due', 'proposal deadline', 'bid due', 'application due',
            'questions due', 'questions deadline', 'presentations', 'vendor selection',
            'go live', 'implementation', 'contract start', 'contract end',
            'by', 'on or before', 'no later than', 'until', 'through'
        ]
        
        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Phone number patterns
        self.phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (123) 456-7890, 123-456-7890, 123.456.7890
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',       # 123-456-7890, 123.456.7890, 123 456 7890
            r'\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # +1-123-456-7890
        ]
        
        # Numeric data patterns
        self.numeric_patterns = [
            r'\$[\d,]+[KMB]?',  # $37M, $27M, $1.2B, $1,000
            r'\d+[KMB]?\s*(?:beds|employees|users|contracts|years|months)',  # 189 beds, 3-year agreement
            r'(?:annual|yearly|monthly)\s*(?:spend|budget|revenue)',  # annual spend
            r'\d+\s*(?:year|month|day)s?\s*(?:agreement|contract|term)',  # 3-year agreement
        ]
        
        # Document type patterns
        self.document_types = [
            'Request For Proposal', 'RFP', 'RFI', 'Resume', 'CV', 
            'Proposal', 'Statement of Work', 'SOW', 'Contract',
            'Agreement', 'Terms and Conditions', 'Terms of Service'
        ]
    
    def extract_metadata(self, lines: List[Dict[str, Any]]) -> DocumentMetadata:
        """
        Extract comprehensive metadata from PDF lines.
        
        Args:
            lines: List of line dictionaries with text and formatting info
            
        Returns:
            DocumentMetadata object with extracted information
        """
        if not lines:
            return DocumentMetadata({}, [], [], {}, {}, 'Unknown', 0)
        
        logger.info(f"Starting comprehensive metadata extraction from {len(lines)} lines")
        
        # Extract all types of metadata
        self._extract_dates_with_context(lines)
        self._extract_emails(lines)
        self._extract_phone_numbers(lines)
        self._extract_contact_details(lines)
        self._extract_numeric_data(lines)
        self._extract_document_type(lines)
        self._extract_total_pages(lines)
        
        logger.info(f"Extracted metadata: {len(self.metadata['dates'])} dates, {len(self.metadata['emails'])} emails, {len(self.metadata['phone_numbers'])} phones")
        
        return DocumentMetadata(
            dates=self.metadata['dates'],
            emails=self.metadata['emails'],
            phone_numbers=self.metadata['phone_numbers'],
            contact_details=self.metadata['contact_details'],
            numeric_data=self.metadata['numeric_data'],
            document_type=self.metadata['document_type'],
            total_pages=self.metadata['total_pages']
        )
    
    def _extract_dates_with_context(self, lines: List[Dict[str, Any]]) -> None:
        """Extract dates with proper context and categorization."""
        logger.debug("Extracting dates with context")
        
        for line in lines:
            text = line.get('text', '').strip()
            if not text:
                continue
            
            # Check each date pattern
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    date_str = match.group(0)
                    
                    # Parse the date
                    parsed_date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
                    if not parsed_date:
                        continue
                    
                    # Extract context around the date
                    context = self._extract_date_context(text, match.start(), match.end())
                    
                    # Categorize the date based on context
                    date_key = self._categorize_date(context, date_str)
                    
                    if date_key:
                        formatted_date = parsed_date.strftime('%Y-%m-%d')
                        self.metadata['dates'][date_key] = formatted_date
                        logger.debug(f"Found date: {date_key} = {formatted_date} (context: {context})")
                    else:
                        # Store as generic date if no specific context
                        generic_key = f"date_{len(self.metadata['dates']) + 1}"
                        formatted_date = parsed_date.strftime('%Y-%m-%d')
                        self.metadata['dates'][generic_key] = formatted_date
                        logger.debug(f"Found generic date: {generic_key} = {formatted_date}")
    
    def _extract_date_context(self, text: str, start: int, end: int) -> str:
        """Extract context around a date match."""
        # Get 50 characters before and after the date
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end].lower()
        
        return context
    
    def _categorize_date(self, context: str, date_str: str) -> Optional[str]:
        """Categorize a date based on its context."""
        context_lower = context.lower()
        
        # RFP distribution
        if any(keyword in context_lower for keyword in ['rfp sent', 'rfp distribution', 'rfp release']):
            return 'rfp_distribution'
        
        # Questions deadline
        if any(keyword in context_lower for keyword in ['questions due', 'questions deadline', 'questions submitted']):
            return 'questions_deadline'
        
        # Responses deadline
        if any(keyword in context_lower for keyword in ['responses due', 'proposal due', 'submission deadline', 'response deadline']):
            return 'responses_deadline'
        
        # Presentations
        if any(keyword in context_lower for keyword in ['presentations', 'vendor presentations', 'oral presentations']):
            return 'presentations'
        
        # Vendor selection
        if any(keyword in context_lower for keyword in ['vendor selection', 'award decision', 'selection']):
            return 'vendor_selection'
        
        # Go live
        if any(keyword in context_lower for keyword in ['go live', 'implementation', 'start date', 'effective date']):
            return 'go_live'
        
        # Contract start/end
        if any(keyword in context_lower for keyword in ['contract start', 'contract begins']):
            return 'contract_start'
        
        if any(keyword in context_lower for keyword in ['contract end', 'contract expires']):
            return 'contract_end'
        
        # Default to general deadline if context contains deadline keywords
        if any(keyword in context_lower for keyword in ['deadline', 'due', 'by', 'no later than', 'on or before']):
            return 'general_deadline'
        
        return None
    
    def _extract_emails(self, lines: List[Dict[str, Any]]) -> None:
        """Extract email addresses from text."""
        logger.debug("Extracting email addresses")
        
        for line in lines:
            text = line.get('text', '').strip()
            if not text:
                continue
            
            # Find all email addresses in the line
            emails = re.findall(self.email_pattern, text)
            for email in emails:
                if email not in self.metadata['emails']:
                    self.metadata['emails'].append(email)
                    logger.debug(f"Found email: {email}")
    
    def _extract_phone_numbers(self, lines: List[Dict[str, Any]]) -> None:
        """Extract phone numbers from text."""
        logger.debug("Extracting phone numbers")
        
        for line in lines:
            text = line.get('text', '').strip()
            if not text:
                continue
            
            # Check each phone pattern
            for pattern in self.phone_patterns:
                phones = re.findall(pattern, text)
                for phone in phones:
                    # Clean up the phone number
                    clean_phone = re.sub(r'[^\d]', '', phone)
                    if len(clean_phone) >= 10 and clean_phone not in self.metadata['phone_numbers']:
                        self.metadata['phone_numbers'].append(phone)
                        logger.debug(f"Found phone: {phone}")
    
    def _extract_contact_details(self, lines: List[Dict[str, Any]]) -> None:
        """Extract contact details and names."""
        logger.debug("Extracting contact details")
        
        for line in lines:
            text = line.get('text', '').strip()
            if not text:
                continue
            
            # Look for name patterns (First Last)
            name_pattern = r'^[A-Z][a-z]+\s+[A-Z][a-z]+'
            if re.match(name_pattern, text):
                if 'contact_person' not in self.metadata['contact_details']:
                    self.metadata['contact_details']['contact_person'] = text
                    logger.debug(f"Found contact person: {text}")
            
            # Look for organization names
            if (text.isupper() and 
                3 <= len(text.split()) <= 5 and 
                not text.startswith('PAGE') and
                not text.startswith('SECTION') and
                not text.startswith('REQUEST')):
                if 'organization' not in self.metadata['contact_details']:
                    self.metadata['contact_details']['organization'] = text
                    logger.debug(f"Found organization: {text}")
    
    def _extract_numeric_data(self, lines: List[Dict[str, Any]]) -> None:
        """Extract numeric data and financial information."""
        logger.debug("Extracting numeric data")
        
        for line in lines:
            text = line.get('text', '').strip()
            if not text:
                continue
            
            # Check each numeric pattern
            for pattern in self.numeric_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(0)
                    context = text.lower()
                    
                    # Categorize based on context
                    if 'annual' in context or 'yearly' in context:
                        self.metadata['numeric_data']['annual_spend'] = value
                    elif 'beds' in context:
                        self.metadata['numeric_data']['bed_count'] = value
                    elif 'year' in context or 'agreement' in context:
                        self.metadata['numeric_data']['contract_term'] = value
                    else:
                        # Store as generic numeric data
                        key = f"numeric_{len(self.metadata['numeric_data']) + 1}"
                        self.metadata['numeric_data'][key] = value
                    
                    logger.debug(f"Found numeric data: {value}")
    
    def _extract_document_type(self, lines: List[Dict[str, Any]]) -> None:
        """Extract document type from text."""
        logger.debug("Extracting document type")
        
        # Get first few lines to check for document type
        first_lines = lines[:10]
        all_text = ' '.join(line.get('text', '') for line in first_lines)
        
        # Check for document types
        for doc_type in self.document_types:
            if doc_type.lower() in all_text.lower():
                self.metadata['document_type'] = doc_type
                logger.debug(f"Found document type: {doc_type}")
                break
    
    def _extract_total_pages(self, lines: List[Dict[str, Any]]) -> None:
        """Extract total number of pages."""
        logger.debug("Extracting total pages")
        
        if not lines:
            self.metadata['total_pages'] = 0
            return
        
        # Get the maximum page number from all lines
        max_page = max(line.get('page_number', 1) for line in lines)
        self.metadata['total_pages'] = max_page
        logger.debug(f"Found total pages: {max_page}")


def extract_metadata(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Comprehensive function to extract metadata from PDF lines.
    
    Args:
        lines: List of line dictionaries with text and formatting info
        
    Returns:
        Dictionary with extracted metadata in the expected format
    """
    extractor = ComprehensiveMetadataExtractor()
    metadata = extractor.extract_metadata(lines)
    
    # Return in the expected format
    return {
        'dates': metadata.dates,
        'emails': metadata.emails,
        'phone_numbers': metadata.phone_numbers,
        'contact_details': metadata.contact_details,
        'numeric_data': metadata.numeric_data,
        'document_type': metadata.document_type,
        'total_pages': metadata.total_pages
    }
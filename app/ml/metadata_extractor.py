"""
Intelligent metadata extraction with multiple strategies.

This module extracts metadata from documents using:
- Regex-based pattern matching
- Named Entity Recognition (spaCy)
- Key-value pair detection
- Table extraction
- Contextual analysis
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from urllib.parse import urlparse

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

from app.models.text_element import TextElement


logger = logging.getLogger(__name__)


@dataclass
class MetadataValue:
    """Represents an extracted metadata value with provenance."""
    value: Any
    data_type: str  # date, float, int, string, email, phone, url
    confidence: float
    source: str  # regex, ner, key-value, table, contextual
    location: Dict[str, Any] = field(default_factory=dict)  # page, position, etc.
    raw_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'value': str(self.value),
            'data_type': self.data_type,
            'confidence': self.confidence,
            'source': self.source,
            'location': self.location,
            'raw_text': self.raw_text,
        }


class MetadataExtractor:
    """
    Intelligent metadata extractor with multiple strategies.
    
    Combines regex patterns, NER, key-value detection, and
    contextual analysis to extract comprehensive metadata.
    """
    
    # Regex patterns for structured data
    PATTERNS = {
        'date': [
            # MM/DD/YYYY, M/D/YY
            re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'),
            # Month DD, YYYY
            re.compile(
                r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
                re.IGNORECASE
            ),
            # ISO format: YYYY-MM-DD
            re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
            # DD Month YYYY
            re.compile(
                r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
                re.IGNORECASE
            ),
        ],
        'time': [
            # HH:MM AM/PM
            re.compile(r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b'),
            # 24-hour format
            re.compile(r'\b(\d{2}:\d{2}(?::\d{2})?)\b'),
        ],
        'money': [
            # $X,XXX.XX
            re.compile(r'([$£€¥]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b'),
            # X.XX USD/EUR/GBP
            re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY))\b', re.IGNORECASE),
        ],
        'phone': [
            # (XXX) XXX-XXXX
            re.compile(r'\((\d{3})\)\s*(\d{3})-(\d{4})\b'),
            # XXX-XXX-XXXX
            re.compile(r'\b(\d{3})-(\d{3})-(\d{4})\b'),
            # +1-XXX-XXX-XXXX
            re.compile(r'\+(\d{1,3})-(\d{3})-(\d{3})-(\d{4})\b'),
            # XXX.XXX.XXXX
            re.compile(r'\b(\d{3})\.(\d{3})\.(\d{4})\b'),
        ],
        'email': [
            re.compile(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'),
        ],
        'url': [
            re.compile(r'\b(https?://[^\s]+)\b'),
            re.compile(r'\b(www\.[^\s]+\.[^\s]{2,})\b'),
        ],
        'reference': [
            # RFP #, ID:, Case #, etc.
            re.compile(r'\b((?:RFP|ID|Case|Order|Invoice|PO|Ref(?:erence)?)\s*[#:]?\s*[A-Z0-9-]+)\b', re.IGNORECASE),
        ],
        'percentage': [
            re.compile(r'\b(\d+(?:\.\d+)?%)\b'),
        ],
        'version': [
            # v1.0, Version 2.1, Rev. 3
            re.compile(r'\b([vV](?:ersion)?\s*\.?\s*\d+(?:\.\d+)*)\b'),
            re.compile(r'\b(Rev(?:ision)?\.?\s*\d+)\b', re.IGNORECASE),
        ],
    }
    
    # Common metadata keys
    METADATA_KEYS = {
        'temporal': [
            'date', 'deadline', 'due date', 'start date', 'end date',
            'effective date', 'expiration', 'timestamp', 'created',
            'modified', 'updated', 'published'
        ],
        'financial': [
            'budget', 'cost', 'price', 'amount', 'total', 'subtotal',
            'fee', 'charge', 'payment', 'value'
        ],
        'contact': [
            'contact', 'name', 'author', 'client', 'vendor', 'company',
            'organization', 'phone', 'email', 'address', 'fax'
        ],
        'document': [
            'title', 'subject', 'description', 'version', 'revision',
            'status', 'type', 'category', 'reference', 'id', 'number'
        ],
        'classification': [
            'confidential', 'classification', 'security', 'level',
            'status', 'priority', 'urgency'
        ],
    }
    
    # Key-value pattern
    KEY_VALUE_PATTERN = re.compile(
        r'(\w+(?:\s+\w+){0,3})\s*[:|\-]\s*(.+?)(?:\n|$)',
        re.IGNORECASE
    )
    
    def __init__(
        self,
        spacy_model: str = 'en_core_web_sm',
        ner_confidence_threshold: float = 0.8,
        enable_ner: bool = True
    ):
        """
        Initialize metadata extractor.
        
        Args:
            spacy_model: spaCy model name
            ner_confidence_threshold: Minimum confidence for NER entities
            enable_ner: Whether to enable NER extraction
        """
        self.ner_confidence_threshold = ner_confidence_threshold
        self.enable_ner = enable_ner
        
        # Load spaCy model
        if self.enable_ner and SPACY_AVAILABLE:
            try:
                logger.info(f"Loading spaCy model: {spacy_model}")
                self.nlp = spacy.load(spacy_model)
                logger.info("spaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}. NER disabled.")
                self.enable_ner = False
                self.nlp = None
        else:
            if not SPACY_AVAILABLE:
                logger.warning("spaCy not installed. NER disabled.")
            self.enable_ner = False
            self.nlp = None
        
        logger.info("MetadataExtractor initialized")
    
    def extract_metadata(
        self,
        elements: List[TextElement],
        tables: Optional[List[Dict]] = None
    ) -> Dict[str, List[MetadataValue]]:
        """
        Extract metadata from document elements.
        
        Args:
            elements: List of TextElements
            tables: Optional list of table data
            
        Returns:
            Dictionary mapping metadata keys to lists of MetadataValue objects
        """
        logger.info(f"Extracting metadata from {len(elements)} elements")
        
        metadata = defaultdict(list)
        
        # Combine all text for certain analyses
        full_text = "\n".join(elem.text for elem in elements)
        
        # 1. Regex-based extraction
        regex_metadata = self._extract_with_regex(elements)
        self._merge_metadata(metadata, regex_metadata)
        
        # 2. Named Entity Recognition
        if self.enable_ner and self.nlp:
            ner_metadata = self._extract_with_ner(full_text, elements)
            self._merge_metadata(metadata, ner_metadata)
        
        # 3. Key-value pair detection
        kv_metadata = self._extract_key_value_pairs(elements)
        self._merge_metadata(metadata, kv_metadata)
        
        # 4. Table extraction
        if tables:
            table_metadata = self._extract_from_tables(tables)
            self._merge_metadata(metadata, table_metadata)
        
        # 5. Contextual extraction
        contextual_metadata = self._extract_contextual(elements)
        self._merge_metadata(metadata, contextual_metadata)
        
        # 6. Deduplicate and rank
        metadata = self._deduplicate_metadata(metadata)
        
        # 7. Normalize keys
        metadata = self._normalize_metadata(metadata)
        
        logger.info(f"Extracted {len(metadata)} metadata keys")
        
        return dict(metadata)
    
    def _extract_with_regex(
        self,
        elements: List[TextElement]
    ) -> Dict[str, List[MetadataValue]]:
        """Extract metadata using regex patterns."""
        metadata = defaultdict(list)
        
        for elem in elements:
            text = elem.text
            
            # Try each pattern category
            for category, patterns in self.PATTERNS.items():
                for pattern in patterns:
                    matches = pattern.findall(text)
                    
                    for match in matches:
                        # Handle tuple matches (from grouped patterns)
                        if isinstance(match, tuple):
                            value = ''.join(match)
                        else:
                            value = match
                        
                        # Validate and parse
                        parsed_value, data_type = self._validate_and_parse(
                            value, category
                        )
                        
                        if parsed_value is not None:
                            meta_value = MetadataValue(
                                value=parsed_value,
                                data_type=data_type,
                                confidence=0.9,
                                source='regex',
                                location={
                                    'page': elem.page_number,
                                    'bbox': {
                                        'x0': elem.bbox.x0,
                                        'y0': elem.bbox.y0,
                                    }
                                },
                                raw_text=value
                            )
                            
                            metadata[category].append(meta_value)
        
        return metadata
    
    def _extract_with_ner(
        self,
        text: str,
        elements: List[TextElement]
    ) -> Dict[str, List[MetadataValue]]:
        """Extract metadata using Named Entity Recognition."""
        metadata = defaultdict(list)
        
        # Process text with spaCy
        doc = self.nlp(text[:1000000])  # Limit to 1M chars
        
        # Entity type mapping
        entity_mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'DATE': 'date',
            'MONEY': 'money',
            'TIME': 'time',
            'PERCENT': 'percentage',
            'CARDINAL': 'number',
        }
        
        for ent in doc.ents:
            # Filter by confidence (if available)
            # Note: spaCy doesn't provide confidence by default
            
            entity_type = entity_mapping.get(ent.label_, ent.label_.lower())
            
            meta_value = MetadataValue(
                value=ent.text,
                data_type=entity_type,
                confidence=0.8,  # Default NER confidence
                source='ner',
                location={'char_start': ent.start_char, 'char_end': ent.end_char},
                raw_text=ent.text
            )
            
            metadata[entity_type].append(meta_value)
        
        return metadata
    
    def _extract_key_value_pairs(
        self,
        elements: List[TextElement]
    ) -> Dict[str, List[MetadataValue]]:
        """Extract key-value pairs from text."""
        metadata = defaultdict(list)
        
        # Focus on first few pages (metadata usually at start)
        relevant_elements = [e for e in elements if e.page_number <= 3]
        
        for elem in relevant_elements:
            text = elem.text
            
            # Find key-value patterns
            matches = self.KEY_VALUE_PATTERN.findall(text)
            
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                
                # Skip if value is empty or too long
                if not value or len(value) > 200:
                    continue
                
                # Normalize key
                normalized_key = self._normalize_key(key)
                
                # Determine data type
                data_type = self._infer_data_type(value, key)
                
                # Parse value
                parsed_value = self._parse_value(value, data_type)
                
                meta_value = MetadataValue(
                    value=parsed_value,
                    data_type=data_type,
                    confidence=0.85,
                    source='key-value',
                    location={
                        'page': elem.page_number,
                        'bbox': {'x0': elem.bbox.x0, 'y0': elem.bbox.y0}
                    },
                    raw_text=f"{key}: {value}"
                )
                
                metadata[normalized_key].append(meta_value)
        
        return metadata
    
    def _extract_from_tables(
        self,
        tables: List[Dict]
    ) -> Dict[str, List[MetadataValue]]:
        """Extract metadata from tables."""
        metadata = defaultdict(list)
        
        for table in tables:
            table_data = table.get('data', [])
            page_number = table.get('page_number', 1)
            
            # Look for two-column tables (key-value format)
            if table_data and len(table_data[0]) == 2:
                for row in table_data:
                    if len(row) == 2:
                        key, value = row[0], row[1]
                        
                        if not key or not value:
                            continue
                        
                        key = str(key).strip()
                        value = str(value).strip()
                        
                        # Normalize key
                        normalized_key = self._normalize_key(key)
                        
                        # Infer data type
                        data_type = self._infer_data_type(value, key)
                        
                        # Parse value
                        parsed_value = self._parse_value(value, data_type)
                        
                        meta_value = MetadataValue(
                            value=parsed_value,
                            data_type=data_type,
                            confidence=0.9,
                            source='table',
                            location={'page': page_number, 'table': True},
                            raw_text=f"{key}: {value}"
                        )
                        
                        metadata[normalized_key].append(meta_value)
        
        return metadata
    
    def _extract_contextual(
        self,
        elements: List[TextElement]
    ) -> Dict[str, List[MetadataValue]]:
        """Extract metadata from contextual locations."""
        metadata = defaultdict(list)
        
        # Header (top 10% of first page)
        header_elements = [
            e for e in elements
            if e.page_number == 1 and e.bbox.y0 < 80
        ]
        
        # Footer (bottom 10% of pages)
        footer_elements = [
            e for e in elements
            if e.bbox.y0 > 700  # Near bottom
        ]
        
        # Look for specific patterns in headers/footers
        for context, context_elements in [
            ('header', header_elements),
            ('footer', footer_elements)
        ]:
            for elem in context_elements:
                text = elem.text
                
                # Document status
                status_patterns = [
                    r'\b(Draft|Final|Approved|Confidential|Internal|Public)\b',
                    r'\b(Status:\s*\w+)\b',
                ]
                
                for pattern_str in status_patterns:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    matches = pattern.findall(text)
                    
                    for match in matches:
                        meta_value = MetadataValue(
                            value=match,
                            data_type='string',
                            confidence=0.85,
                            source='contextual',
                            location={
                                'page': elem.page_number,
                                'context': context
                            },
                            raw_text=match
                        )
                        
                        metadata['status'].append(meta_value)
        
        return metadata
    
    def _validate_and_parse(
        self,
        value: str,
        category: str
    ) -> Tuple[Optional[Any], str]:
        """Validate and parse extracted value."""
        if category == 'date':
            return self._parse_date(value), 'date'
        elif category == 'time':
            return value, 'time'
        elif category == 'money':
            return self._parse_money(value), 'float'
        elif category == 'phone':
            return value, 'phone'
        elif category == 'email':
            if self._validate_email(value):
                return value, 'email'
        elif category == 'url':
            if self._validate_url(value):
                return value, 'url'
        elif category == 'percentage':
            return self._parse_percentage(value), 'float'
        
        return value, 'string'
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format."""
        date_formats = [
            '%m/%d/%Y', '%m/%d/%y', '%m-%d-%Y', '%m-%d-%y',
            '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return date_str  # Return as-is if can't parse
    
    def _parse_money(self, money_str: str) -> Optional[float]:
        """Parse money string to float."""
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$£€¥,]', '', money_str)
        cleaned = re.sub(r'\s+(?:USD|EUR|GBP|JPY)', '', cleaned, flags=re.IGNORECASE)
        
        try:
            return float(cleaned.strip())
        except ValueError:
            return None
    
    def _parse_percentage(self, pct_str: str) -> Optional[float]:
        """Parse percentage string to float."""
        cleaned = pct_str.replace('%', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
        return bool(pattern.match(email))
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme or url.startswith('www'), result.netloc or '.' in url])
        except Exception:
            return False
    
    def _normalize_key(self, key: str) -> str:
        """Normalize metadata key."""
        # Convert to lowercase
        key = key.lower().strip()
        
        # Remove special characters
        key = re.sub(r'[^\w\s]', '', key)
        
        # Replace spaces with underscores
        key = key.replace(' ', '_')
        
        # Common synonyms
        synonyms = {
            'due_date': 'deadline',
            'effective_date': 'start_date',
            'exp_date': 'expiration_date',
            'phone_number': 'phone',
            'email_address': 'email',
            'company_name': 'company',
            'org': 'organization',
            'ref': 'reference',
            'ref_number': 'reference',
            'id_number': 'id',
        }
        
        return synonyms.get(key, key)
    
    def _infer_data_type(self, value: str, key: str) -> str:
        """Infer data type from value and key context."""
        key_lower = key.lower()
        
        # Date-related keys
        if any(word in key_lower for word in ['date', 'deadline', 'time']):
            return 'date'
        
        # Financial keys
        if any(word in key_lower for word in ['cost', 'price', 'budget', 'amount', 'fee']):
            return 'float'
        
        # Contact keys
        if 'email' in key_lower:
            return 'email'
        if 'phone' in key_lower:
            return 'phone'
        if 'url' in key_lower or 'website' in key_lower:
            return 'url'
        
        # Try to infer from value
        if re.match(r'^\d+\.\d+$', value):
            return 'float'
        if re.match(r'^\d+$', value):
            return 'int'
        if re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', value):
            return 'email'
        
        return 'string'
    
    def _parse_value(self, value: str, data_type: str) -> Any:
        """Parse value according to data type."""
        try:
            if data_type == 'int':
                return int(value)
            elif data_type == 'float':
                # Remove commas
                return float(value.replace(',', ''))
            elif data_type == 'date':
                return self._parse_date(value) or value
            else:
                return value
        except (ValueError, AttributeError):
            return value
    
    def _merge_metadata(
        self,
        target: Dict[str, List[MetadataValue]],
        source: Dict[str, List[MetadataValue]]
    ) -> None:
        """Merge source metadata into target."""
        for key, values in source.items():
            target[key].extend(values)
    
    def _deduplicate_metadata(
        self,
        metadata: Dict[str, List[MetadataValue]]
    ) -> Dict[str, List[MetadataValue]]:
        """Remove duplicate metadata entries."""
        deduplicated = {}
        
        for key, values in metadata.items():
            if not values:
                continue
            
            # Group by value
            value_groups = defaultdict(list)
            for meta_val in values:
                # Normalize value for comparison
                norm_value = str(meta_val.value).lower().strip()
                value_groups[norm_value].append(meta_val)
            
            # Keep highest confidence from each group
            unique_values = []
            for norm_value, group in value_groups.items():
                # Sort by confidence (descending)
                group.sort(key=lambda x: x.confidence, reverse=True)
                unique_values.append(group[0])
            
            deduplicated[key] = unique_values
        
        return deduplicated
    
    def _normalize_metadata(
        self,
        metadata: Dict[str, List[MetadataValue]]
    ) -> Dict[str, List[MetadataValue]]:
        """Normalize metadata keys and values."""
        normalized = {}
        
        for key, values in metadata.items():
            # Normalize key
            norm_key = self._normalize_key(key)
            
            if norm_key not in normalized:
                normalized[norm_key] = []
            
            normalized[norm_key].extend(values)
        
        return normalized
    
    def to_dict(
        self,
        metadata: Dict[str, List[MetadataValue]]
    ) -> Dict[str, Any]:
        """
        Convert metadata to dictionary format.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Simplified dictionary format
        """
        result = {}
        
        for key, values in metadata.items():
            if len(values) == 1:
                # Single value
                result[key] = values[0].to_dict()
            else:
                # Multiple values
                result[key] = [v.to_dict() for v in values]
        
        return result
    
    def get_summary(
        self,
        metadata: Dict[str, List[MetadataValue]]
    ) -> Dict[str, Any]:
        """
        Get summary of extracted metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_keys': len(metadata),
            'total_values': sum(len(values) for values in metadata.values()),
            'by_source': defaultdict(int),
            'by_type': defaultdict(int),
            'keys': list(metadata.keys()),
        }
        
        for values in metadata.values():
            for meta_val in values:
                summary['by_source'][meta_val.source] += 1
                summary['by_type'][meta_val.data_type] += 1
        
        summary['by_source'] = dict(summary['by_source'])
        summary['by_type'] = dict(summary['by_type'])
        
        return summary


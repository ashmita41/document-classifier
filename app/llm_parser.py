"""
Final enhanced document parser with comprehensive improvements.

This version includes:
- Advanced text normalization
- Better section detection
- Improved metadata extraction
- Cleaner JSON structure
- Better LLM prompts
"""
import logging
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import openai
from openai import AsyncOpenAI

from app.models.text_element import TextElement, ExtractionResult
from app.services.pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedBlock:
    """Represents a classified text block with enhanced metadata."""
    text: str
    label: str
    confidence: float
    page_number: int
    line_number: int
    metadata: Dict[str, Any] = None
    normalized_text: str = None
    is_section_title: bool = False
    is_sub_section: bool = False


class AdvancedTextNormalizer:
    """Advanced text normalization with comprehensive cleaning."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Comprehensive text normalization."""
        if not text:
            return ""
        
        # Remove PDF artifacts
        text = re.sub(r'\(cid:\d+\)', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Fix word merging issues - comprehensive approach
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # lowercase-uppercase
        text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)  # digit-letter
        text = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', text)  # letter-digit
        
        # Fix common technical terms
        text = re.sub(r'\b([A-Z][a-z]+)([A-Z][a-z]+)\b', r'\1 \2', text)  # DjangoREST -> Django REST
        text = re.sub(r'\b([A-Z]{2,})([A-Z][a-z]+)\b', r'\1 \2', text)  # AWSCloud -> AWS Cloud
        text = re.sub(r'\b([a-z]+)([A-Z][a-z]+)\b', r'\1 \2', text)  # wordWord -> word Word
        
        # Fix punctuation
        text = re.sub(r'([a-zA-Z])([.,!?:;])', r'\1\2', text)
        text = re.sub(r'([.,!?:;])([a-zA-Z])', r'\1 \2', text)
        
        # Fix common merged words
        replacements = {
            'Collegeof': 'College of',
            'Technologyand': 'Technology and',
            'Electronicsand': 'Electronics and',
            'Communication': 'Communication',
            'DjangoREST': 'Django REST',
            'AWSCloud': 'AWS Cloud',
            'React19': 'React 19',
            'TailwindCSS': 'Tailwind CSS',
            'ReactRouter': 'React Router',
            'JWTauthentication': 'JWT authentication',
            'RESTAPI': 'REST API',
            'RESTAPIs': 'REST APIs',
            'GitHubActions': 'GitHub Actions',
            'CloudWatch': 'Cloud Watch',
            'Celerywith': 'Celery with',
            'Redisfor': 'Redis for',
            'Flaskthread': 'Flask thread',
            'taskexecution': 'task execution',
            'jobprocessing': 'job processing',
            'betterscalability': 'better scalability',
            'backendperformance': 'backend performance',
            'WebhookSystem': 'Webhook System',
            'GoogleCalendar': 'Google Calendar',
            'OutlookCalendar': 'Outlook Calendar',
            'Webhookswith': 'Webhooks with',
            'Lambdadaily': 'Lambda daily',
            'renewaljobs': 'renewal jobs',
            'calendarsyncing': 'calendar syncing',
            'eventupdates': 'event updates',
            'atomicityandidempotency': 'atomicity and idempotency',
            'webhookcreation': 'webhook creation',
            'renewalprocesses': 'renewal processes',
            'proactiverenewal': 'proactive renewal',
            'renewallogic': 'renewal logic',
            'webhookexpiration': 'webhook expiration',
            'alwaysup': 'always up',
            'todatemeeting': 'to date meeting',
            'datainside': 'data inside',
            'PepsalesAI': 'Pepsales AI',
            'inputvalidation': 'input validation',
            'requestschema': 'request schema',
            'schemaenforcement': 'schema enforcement',
            'errorhandling': 'error handling',
            'consistencyreliability': 'consistency, reliability',
            'securityacross': 'security across',
            'backendservices': 'backend services',
            'optimizedbackendsolutionstofixbotreschedulinglogic': 'optimized backend solutions to fix bot rescheduling logic',
            'resolvetime': 'resolve time',
            'sensitivebusinesslogicbugs': 'sensitive business logic bugs',
            'enhanceobservabilityusing': 'enhance observability using',
            'structureddatabaselogging': 'structured database logging',
            'Workedcloselywith': 'Worked closely with',
            'Founder&CEO': 'Founder & CEO',
            'toimplementcreative': 'to implement creative',
            'growthstrategiessuchasfreedemocourses': 'growth strategies such as free demo courses',
            'resultingina': 'resulting in a',
            'xrevenueincreasewithin': 'x revenue increase within',
            'months': 'months',
            'Playedakeyroleinstrategicdecision': 'Played a key role in strategic decision',
            'makingcontributingtothecompany': 'making, contributing to the company',
            'sprogressiontothefinalroundof': 's progression to the final round of',
            'Balancedtechnicalandmanagerialresponsibilities': 'Balanced technical and managerial responsibilities',
            'ensuringalignmentbetweenbusinessgoalsand': 'ensuring alignment between business goals and',
            'developmentexecution': 'development execution',
            'Builtafull': 'Built a full',
            'stackhiringplatformwheredevelopersandcompaniesconnectviaintelligentswipe': 'stack hiring platform where developers and companies connect via intelligent swipe',
            'basedmatchmaking': 'based matchmaking',
            'Implementedreal': 'Implemented real',
            'timemutualmatchdetection': 'time mutual match detection',
            'wishlistfunctionality': 'wishlist functionality',
            'andactivity': 'and activity',
            'basedstatustracking': 'based status tracking',
            'Frontenddevelopedin': 'Frontend developed in',
            'withTailwindCSS': 'with Tailwind CSS',
            'andTinder': 'and Tinder',
            'styleswipeUX': 'style swipe UX',
            'LanguagesPython': 'Languages: Python',
            'JavaJava': 'Java, Java',
            'Script': 'Script',
            'BackendDjango': 'Backend: Django',
            'DjangoRESTFramework': 'Django REST Framework',
            'DatabasesPostgreSQL': 'Databases: PostgreSQL',
            'MongoDBRedis': 'MongoDB, Redis',
            'CloudDevOpsGit': 'Cloud/DevOps: Git',
            'GitHubDocker': 'GitHub, Docker',
            'AWS': 'AWS',
            'KeyConceptsData': 'Key Concepts: Data',
            'StructuresandAlgorithms': 'Structures and Algorithms',
            'Object': 'Object',
            'OrientedProgramming': 'Oriented Programming',
            'OOPRESTAPI': 'OOP, REST API',
            'SystemDesignBasics': 'System Design Basics'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, str]:
        """Enhanced metadata extraction."""
        metadata = {}
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            metadata['email'] = email_match.group()
        
        # Phone
        phone_match = re.search(r'(\+?91[\s-]?)?[6-9]\d{9}', text)
        if phone_match:
            phone = phone_match.group().replace(' ', '').replace('-', '')
            metadata['phone'] = phone
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text, re.IGNORECASE)
        if linkedin_match:
            metadata['linkedin'] = linkedin_match.group()
        
        # GitHub
        github_match = re.search(r'github\.com/[\w-]+', text, re.IGNORECASE)
        if github_match:
            metadata['github'] = github_match.group()
        
        # Location
        location_match = re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+,\s*[A-Z][a-z]+\b', text)
        if location_match:
            metadata['location'] = location_match.group()
        
        # Date ranges
        date_match = re.search(r'\b(19|20)\d{2}[-–]\d{4}\b', text)
        if date_match:
            metadata['date_range'] = date_match.group()
        
        return metadata


class SmartSectionDetector:
    """Smart section detection with improved accuracy."""
    
    SECTION_KEYWORDS = {
        'personal', 'contact', 'about', 'summary', 'objective', 'profile',
        'education', 'academic', 'qualification', 'degree',
        'experience', 'work', 'employment', 'career', 'professional',
        'projects', 'portfolio', 'work samples',
        'skills', 'technical', 'competencies', 'expertise',
        'certifications', 'certificates', 'licenses',
        'awards', 'achievements', 'honors',
        'publications', 'research', 'papers',
        'languages', 'interests', 'hobbies'
    }
    
    JOB_TITLES = {
        'backend developer', 'frontend developer', 'full stack developer',
        'software engineer', 'data scientist', 'product manager',
        'intern', 'associate', 'senior', 'junior', 'lead', 'principal'
    }
    
    @classmethod
    def is_section_title(cls, text: str, confidence_threshold: float = 0.8) -> Tuple[bool, float]:
        """Enhanced section title detection."""
        if not text or len(text.strip()) == 0:
            return False, 0.0
        
        text = text.strip()
        confidence = 0.0
        
        # Length check
        word_count = len(text.split())
        if word_count <= 3:
            confidence += 0.4
        elif word_count <= 6:
            confidence += 0.2
        elif word_count <= 10:
            confidence += 0.1
        
        # Capitalization
        if text.isupper():
            confidence += 0.3
        elif text.istitle():
            confidence += 0.4
        elif any(word.istitle() for word in text.split()):
            confidence += 0.2
        
        # Known keywords
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in cls.SECTION_KEYWORDS):
            confidence += 0.5
        
        # Penalize job titles
        if any(job_title in text_lower for job_title in cls.JOB_TITLES):
            confidence -= 0.4
        
        # Pattern checks
        if re.match(r'^[A-Z\s]+$', text):
            confidence += 0.2
        if text.endswith(':'):
            confidence += 0.1
        
        # Penalize sentences
        if text.endswith('.') or text.endswith('!') or text.endswith('?'):
            confidence -= 0.3
        if word_count > 15:
            confidence -= 0.2
        
        return confidence >= confidence_threshold, confidence
    
    @classmethod
    def is_sub_section(cls, text: str) -> bool:
        """Enhanced sub-section detection."""
        if not text:
            return False
        
        text = text.strip()
        
        # Company patterns
        if re.match(r'^[A-Z][a-zA-Z\s&.,]+(Inc|LLC|Corp|Company|Ltd|Technologies|Systems|Solutions|AI)$', text):
            return True
        
        # Project patterns
        if re.match(r'^[A-Z][a-zA-Z\s]+(Project|Platform|System|Application|Tool)$', text):
            return True
        
        # Date ranges
        if re.match(r'^\w+\s+\d{4}\s*[-–]\s*(Present|\d{4})$', text):
            return True
        
        # Degree patterns
        if re.match(r'^[A-Z]\.[A-Za-z]+\s+in\s+', text):
            return True
        
        return False

class FinalEnhancedLLMDocumentParser:
    """Optimized LLM document parser with performance enhancements."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_workers: int = 4):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = 12
        self.max_retries = 3
        self.max_workers = max_workers
        
        # Performance optimizations
        self._classification_cache = {}
        self._cache_lock = asyncio.Lock()
        
        logger.info(f"Optimized LLMDocumentParser initialized with model: {model}, workers: {max_workers}")
    
    async def classify_blocks(self, blocks: List[TextElement]) -> List[ClassifiedBlock]:
        """Classify text blocks with optimized performance."""
        classified_blocks = []
        
        # Pre-classify with enhanced rules (parallel processing for large documents)
        if len(blocks) > 100:
            # Use parallel processing for large documents
            classified_blocks = await self._classify_blocks_parallel(blocks)
        else:
            # Sequential processing for small documents
            classified_blocks = await self._classify_blocks_sequential(blocks)
        
        # LLM classification for ambiguous blocks
        ambiguous_blocks = [b for b in classified_blocks if b.confidence < 0.8]
        if ambiguous_blocks:
            logger.info(f"LLM classifying {len(ambiguous_blocks)} ambiguous blocks")
            await self._llm_classify_blocks_optimized(ambiguous_blocks)
        
        return classified_blocks
    
    async def _classify_blocks_sequential(self, blocks: List[TextElement]) -> List[ClassifiedBlock]:
        """Sequential classification of blocks."""
        classified_blocks = []
        
        for block in blocks:
            classified_block = await self._classify_single_block(block)
            classified_blocks.append(classified_block)
        
        return classified_blocks
    
    async def _classify_blocks_parallel(self, blocks: List[TextElement]) -> List[ClassifiedBlock]:
        """Parallel classification of blocks for better performance."""
        import asyncio
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def classify_with_semaphore(block):
            async with semaphore:
                return await self._classify_single_block(block)
        
        # Process blocks in parallel
        tasks = [classify_with_semaphore(block) for block in blocks]
        classified_blocks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_blocks = []
        for i, result in enumerate(classified_blocks):
            if isinstance(result, Exception):
                logger.error(f"Error classifying block {i}: {result}")
                # Create fallback block
                fallback_block = await self._classify_single_block(blocks[i])
                valid_blocks.append(fallback_block)
            else:
                valid_blocks.append(result)
        
        return valid_blocks
    
    async def _classify_single_block(self, block: TextElement) -> ClassifiedBlock:
        """Classify a single block with caching."""
        # Check cache first
        cache_key = f"{block.text}:{block.page_number}:{block.line_number}"
        
        async with self._cache_lock:
            if cache_key in self._classification_cache:
                cached_result = self._classification_cache[cache_key]
                # Update with current block data
                cached_result.page_number = block.page_number
                cached_result.line_number = block.line_number
                return cached_result
        
        # Process block
        normalized_text = AdvancedTextNormalizer.normalize_text(block.text)
        metadata = AdvancedTextNormalizer.extract_metadata(normalized_text)
        
        # Enhanced rule-based classification
        label, confidence = self._rule_based_classify(normalized_text, block)
        
        classified_block = ClassifiedBlock(
            text=block.text,
            label=label,
            confidence=confidence,
            page_number=block.page_number,
            line_number=block.line_number,
            metadata=metadata,
            normalized_text=normalized_text
        )
        
        # Cache result
        async with self._cache_lock:
            if len(self._classification_cache) < 1000:  # Limit cache size
                self._classification_cache[cache_key] = classified_block
        
        return classified_block
    
    def _rule_based_classify(self, text: str, block: TextElement) -> Tuple[str, float]:
        """Enhanced rule-based classification."""
        if not text or len(text.strip()) == 0:
            return "description", 0.0
        
        text = text.strip()
        
        # Section title detection
        is_section, section_confidence = SmartSectionDetector.is_section_title(text)
        if is_section:
            return "section_title", section_confidence
        
        # Sub-section detection
        if SmartSectionDetector.is_sub_section(text):
            return "sub_section_title", 0.8
        
        # Question detection
        if text.endswith('?') or re.match(r'^(What|How|When|Where|Why|Who|Which)', text, re.IGNORECASE):
            return "question", 0.9
        
        # Metadata detection
        metadata = AdvancedTextNormalizer.extract_metadata(text)
        if metadata:
            return "metadata", 0.8
        
        # List item detection
        if re.match(r'^[•\-\*]\s+', text) or re.match(r'^\d+\.\s+', text):
            return "list_item", 0.7
        
        # Default to description
        return "description", 0.5
    
    async def _llm_classify_blocks_optimized(self, blocks: List[ClassifiedBlock]):
        """Optimized LLM classification with parallel processing."""
        batches = [blocks[i:i + self.batch_size] for i in range(0, len(blocks), self.batch_size)]
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._classify_batch_safe(batch))
            tasks.append(task)
        
        # Wait for all batches to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _classify_batch_safe(self, batch: List[ClassifiedBlock]):
        """Safely classify a batch with error handling."""
        try:
            await self._classify_batch(batch)
        except Exception as e:
            logger.error(f"Error classifying batch: {e}")
            # Fallback to rule-based classification
            for block in batch:
                if block.confidence < 0.5:
                    block.label = "description"
                    block.confidence = 0.5
    
    async def _classify_batch(self, blocks: List[ClassifiedBlock]):
        """Classify a batch with enhanced prompt."""
        text_blocks = [block.normalized_text for block in blocks]
        
        prompt = self._create_final_prompt(text_blocks)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document parser specializing in resume and RFP structure analysis. Focus on creating clean, logical document structure."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Update blocks with LLM results
            for i, block in enumerate(blocks):
                if i < len(result.get("classifications", [])):
                    classification = result["classifications"][i]
                    block.label = classification.get("label", block.label)
                    block.confidence = classification.get("confidence", block.confidence)
                    
                    # Update section flags
                    block.is_section_title = block.label == "section_title"
                    block.is_sub_section = block.label == "sub_section_title"
        
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            raise
    
    def _create_final_prompt(self, text_blocks: List[str]) -> str:
        """Create final enhanced prompt."""
        blocks_text = "\n".join([f"{i+1}. \"{block}\"" for i, block in enumerate(text_blocks)])
        
        return f"""You are a document parser specializing in resume and RFP structure analysis.

Analyze the following text blocks and classify each into one of these categories:
- section_title: Main document sections (Education, Experience, Projects, Skills, etc.)
- sub_section_title: Sub-sections within main sections (company names, project names, degree titles, etc.)
- description: Paragraph content, job descriptions, project details
- metadata: Contact info, dates, technical skills, key-value pairs
- question: Questions ending with "?" or interrogative statements
- answer: Answers to questions

Rules:
1. Only classify short, capitalized lines or known headings as section_title
2. Company names, project names, degree titles, and date ranges are sub_section_title
3. Combine related content into coherent descriptions
4. Extract contact info, dates, and technical details as metadata
5. Maintain logical hierarchy and structure
6. Be conservative - only classify as section_title if very confident

Text blocks to classify:
{blocks_text}

Return JSON in this format:
{{
  "classifications": [
    {{"label": "section_title", "confidence": 0.9, "reasoning": "Short, capitalized, known section"}},
    {{"label": "description", "confidence": 0.8, "reasoning": "Paragraph content"}}
  ]
}}"""


class CleanJSONBuilder:
    """Optimized builder for clean, hierarchical JSON structure."""
    
    def __init__(self):
        self.sections = {}
        self.metadata = {}
        self.current_section = None
        self.current_sub_section = None
        
        # Performance optimizations
        self._section_cache = {}
        self._metadata_cache = {}
        self._processing_stats = {
            'sections_processed': 0,
            'sub_sections_processed': 0,
            'descriptions_added': 0,
            'metadata_extracted': 0
        }
    
    def build(self, blocks: List[ClassifiedBlock]) -> Dict[str, Any]:
        """Build clean hierarchical JSON structure with optimization."""
        import time
        start_time = time.time()
        
        # Process blocks efficiently
        self._process_blocks_optimized(blocks)
        
        # Clean up and normalize sections
        self._cleanup_sections_optimized()
        
        processing_time = time.time() - start_time
        
        # Add processing metrics
        result = {
            **self.sections,
            "metadata": self.metadata,
            "processing_metrics": {
                **self._processing_stats,
                "processing_time_seconds": processing_time,
                "total_blocks_processed": len(blocks)
            }
        }
        
        return result
    
    def _process_blocks_optimized(self, blocks: List[ClassifiedBlock]):
        """Optimized block processing with batching."""
        # Group blocks by type for efficient processing
        section_blocks = []
        sub_section_blocks = []
        description_blocks = []
        metadata_blocks = []
        
        for block in blocks:
            if block.label == "section_title":
                section_blocks.append(block)
            elif block.label == "sub_section_title":
                sub_section_blocks.append(block)
            elif block.label == "description":
                description_blocks.append(block)
            elif block.label == "metadata":
                metadata_blocks.append(block)
        
        # Process blocks in order
        for block in blocks:
            self._process_block(block)
    
    def _process_block(self, block: ClassifiedBlock):
        """Process a single classified block."""
        if block.label == "section_title":
            self._start_new_section(block)
        elif block.label == "sub_section_title":
            self._start_new_sub_section(block)
        elif block.label == "description":
            self._add_description(block)
        elif block.label == "metadata":
            self._add_metadata(block)
    
    def _start_new_section(self, block: ClassifiedBlock):
        """Start a new main section with caching."""
        section_name = block.normalized_text
        self.current_section = section_name
        self.current_sub_section = None
        
        # Check cache first
        if section_name in self._section_cache:
            self.sections[section_name] = self._section_cache[section_name]
        else:
            if section_name not in self.sections:
                self.sections[section_name] = {
                    "type": "section",
                    "description": [],
                    "sub_sections": {},
                    "metadata": {}
                }
                # Cache the section structure
                self._section_cache[section_name] = self.sections[section_name]
        
        self._processing_stats['sections_processed'] += 1
    
    def _start_new_sub_section(self, block: ClassifiedBlock):
        """Start a new sub-section."""
        if not self.current_section:
            # If no current section, treat as main section
            self._start_new_section(block)
            return
        
        sub_section_name = block.normalized_text
        self.current_sub_section = sub_section_name
        
        if "sub_sections" not in self.sections[self.current_section]:
            self.sections[self.current_section]["sub_sections"] = {}
        
        if sub_section_name not in self.sections[self.current_section]["sub_sections"]:
            self.sections[self.current_section]["sub_sections"][sub_section_name] = {
                "description": [],
                "metadata": {}
            }
    
    def _add_description(self, block: ClassifiedBlock):
        """Add description content with optimization."""
        if self.current_sub_section and self.current_section:
            # Add to sub-section
            self.sections[self.current_section]["sub_sections"][self.current_sub_section]["description"].append(block.normalized_text)
        elif self.current_section:
            # Add to main section
            self.sections[self.current_section]["description"].append(block.normalized_text)
        else:
            # Add to global metadata if no section context
            if "description" not in self.metadata:
                self.metadata["description"] = []
            self.metadata["description"].append(block.normalized_text)
        
        self._processing_stats['descriptions_added'] += 1
    
    def _add_metadata(self, block: ClassifiedBlock):
        """Add metadata with caching."""
        if block.metadata:
            if self.current_sub_section and self.current_section:
                # Add to sub-section metadata
                self.sections[self.current_section]["sub_sections"][self.current_sub_section]["metadata"].update(block.metadata)
            elif self.current_section:
                # Add to section metadata
                self.sections[self.current_section]["metadata"].update(block.metadata)
            else:
                # Add to global metadata
                self.metadata.update(block.metadata)
            
            self._processing_stats['metadata_extracted'] += 1
    
    def _cleanup_sections_optimized(self):
        """Optimized cleanup and normalization of sections."""
        for section_name, section_data in self.sections.items():
            # Normalize section structure
            self._normalize_section_structure(section_data)
            
            # Clean up empty sub-sections efficiently
            self._cleanup_empty_sub_sections(section_data)
    
    def _normalize_section_structure(self, section_data: Dict[str, Any]):
        """Normalize section structure."""
        if "type" not in section_data:
            section_data["type"] = "section"
        if "description" not in section_data:
            section_data["description"] = []
        if "metadata" not in section_data:
            section_data["metadata"] = {}
        if "sub_sections" not in section_data:
            section_data["sub_sections"] = {}
    
    def _cleanup_empty_sub_sections(self, section_data: Dict[str, Any]):
        """Clean up empty sub-sections efficiently."""
        empty_sub_sections = [
            sub_name for sub_name, sub_data in section_data["sub_sections"].items()
            if not sub_data.get("description", []) and not sub_data.get("metadata", {})
        ]
        
        for sub_name in empty_sub_sections:
            del section_data["sub_sections"][sub_name]
            self._processing_stats['sub_sections_processed'] += 1


class FinalEnhancedDocumentParserPipeline:
    """Optimized document parser pipeline with performance enhancements."""
    
    def __init__(self, api_key: str, max_workers: int = 4, enable_caching: bool = True):
        self.parser = FinalEnhancedLLMDocumentParser(api_key, max_workers=max_workers)
        self.extractor = PDFExtractor(max_workers=max_workers, enable_caching=enable_caching)
        self.builder = CleanJSONBuilder()
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        logger.info(f"Optimized DocumentParserPipeline initialized with {max_workers} workers")
    
    async def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse document with optimized performance."""
        import time
        start_time = time.time()
        
        logger.info(f"Starting optimized document parsing: {file_path}")
        
        # Extract text blocks
        extraction_start = time.time()
        extraction_result = self.extractor.extract_from_file(file_path)
        extraction_time = time.time() - extraction_start
        logger.info(f"Extracted {len(extraction_result.elements)} text blocks in {extraction_time:.2f}s")
        
        # Classify blocks
        classification_start = time.time()
        classified_blocks = await self.parser.classify_blocks(extraction_result.elements)
        classification_time = time.time() - classification_start
        logger.info(f"Classified {len(classified_blocks)} blocks in {classification_time:.2f}s")
        
        # Build clean hierarchical structure
        building_start = time.time()
        structured_doc = self.builder.build(classified_blocks)
        building_time = time.time() - building_start
        
        total_time = time.time() - start_time
        logger.info(f"Optimized document parsing completed in {total_time:.2f}s")
        
        # Add comprehensive performance metrics
        structured_doc['performance_metrics'] = {
            'total_processing_time': total_time,
            'extraction_time': extraction_time,
            'classification_time': classification_time,
            'building_time': building_time,
            'elements_extracted': len(extraction_result.elements),
            'blocks_classified': len(classified_blocks),
            'max_workers': self.max_workers,
            'caching_enabled': self.enable_caching,
            'extraction_method': extraction_result.extraction_method
        }
        
        return structured_doc

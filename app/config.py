# Section Detection Config
SECTION_KEYWORDS = [
    "summary", "introduction", "overview", "background",
    "scope", "requirements", "terms", "conditions",
    "timeline", "deliverables", "pricing", "contact",
    "experience", "education", "skills", "projects",
    "vendor information", "questionnaire", "proposal"
]

METADATA_KEYWORDS = [
    "due date", "deadline", "respond by", "submit by",
    "contact", "email", "phone", "address", "effective date"
]

# Detection Thresholds
MIN_SECTION_FONT_DELTA = 1.5  # Font size above average
MIN_SECTION_WORDS = 2
MAX_SECTION_WORDS = 15
EXCLUDE_SINGLE_WORD_SECTIONS = True

# File Upload Limits
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ['.pdf']

# Header Exclusion Patterns
COMMON_HEADERS = [
    r"^Page \d+",
    r"^Page \d+ of \d+",
    r"Request [Ff]or [Pp]roposal",
    r"^[A-Z]{2,}\s+Lab$",
]

# Section Patterns (Priority Order)
SECTION_PATTERNS = [
    r"^Section\s+[IVX]+",           # Section II, Section IV
    r"^Section\s+\d+",              # Section 1, Section 2
    r"^[A-Z]\.\s+[A-Z]",           # A. Background
    r"^Part\s+[IVX]+",             # Part I, Part II
]

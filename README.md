# PDF Parser API

Intelligent PDF document parser for RFPs, resumes, and structured documents.

## Features

- **Text Extraction**: Extract text with proper ordering and formatting
- **Section Detection**: Identify document sections and subsections
- **Question Classification**: Detect and categorize questions
- **Metadata Extraction**: Extract dates, contacts, and document info
- **Hierarchical Structure**: Build complete document structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Start the API server
uvicorn app.main:app --reload

# Or run directly
python -m uvicorn app.main:app --reload
```

## API Endpoints

- `POST /parse` - Upload PDF for parsing
- `GET /health` - Health check
- `GET /` - API information

## Example Usage

```bash
curl -X POST "http://localhost:8000/parse" \
  -F "file=@document.pdf"
```

## Project Structure

```
app/
├── main.py                  # FastAPI app
├── models.py                # Pydantic schemas
├── config.py                # Configuration
├── parser/
│   ├── extractor.py         # PDF text extraction
│   ├── section_detector.py  # Section detection
│   ├── question_detector.py # Question detection
│   ├── metadata_extractor.py # Metadata extraction
│   └── categorizer.py       # Content categorization
└── utils/
    └── validators.py        # Validation helpers
```
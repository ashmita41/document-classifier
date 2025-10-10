# Document Classifier

Intelligent PDF document parser that automatically extracts structured content from unstructured documents (RFPs, resumes, proposals, contracts).

## Features

- Automatic section detection with hierarchical nesting
- Contextual metadata extraction (deadlines, contacts, references)
- Question-answer pair detection
- 100% section coverage with full text descriptions
- REST API with async processing
- Web interface for document upload and analysis
- Standalone CLI tool

## Quick Start

### Backend + Frontend
```bash
# Windows
RUN_BOTH.bat

# Docker
docker-compose up -d
```

### Standalone CLI
```bash
python -m core.document_analyzer_v2 document.pdf output.json
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Enhanced NER
pip install spacy
python -m spacy download en_core_web_sm

# Configure environment
cp env.template .env
# Edit .env with your MongoDB URI
```

## API Endpoints

### Upload Document
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

Parameters:
- file: PDF file (required)
- process_immediately: boolean (optional, default: false)

Response: 202 Accepted
{
  "id": "doc_abc123",
  "filename": "document.pdf",
  "status": "processing"
}
```

### Get Analysis Results
```http
GET /api/v1/documents/{document_id}/simplified

Response: 200 OK
{
  "Section Name": {
    "type": "section",
    "description": ["Full paragraph text..."],
    "metadata": {
      "deadline": "Oct 10, 2025",
      "contact": "John Doe"
    },
    "questions": [
      {
        "question": "What is...?",
        "answer": "The answer is..."
      }
    ],
    "sub_sections": {
      "Subsection Name": {
        "type": "section",
        "description": ["..."]
      }
    }
  },
  "metadata": {
    "organization": "Company Name"
  }
}
```

### Check Status
```http
GET /api/v1/documents/{document_id}/status

Response: 200 OK
{
  "id": "doc_abc123",
  "status": "completed",
  "filename": "document.pdf",
  "created_at": "2025-10-10T12:00:00"
}
```

### Get Sections Only
```http
GET /api/v1/documents/{document_id}/sections

Response: 200 OK
{
  "sections": [
    {
      "title": "Section Name",
      "level": 1,
      "content": "..."
    }
  ]
}
```

### Get Metadata Only
```http
GET /api/v1/documents/{document_id}/metadata

Response: 200 OK
{
  "metadata": [
    {
      "key": "deadline",
      "value": "Oct 10, 2025",
      "type": "date"
    }
  ]
}
```

### Get Q&A Pairs Only
```http
GET /api/v1/documents/{document_id}/qa

Response: 200 OK
{
  "qa_pairs": [
    {
      "question": "What is...?",
      "answer": "The answer is...",
      "confidence": 0.95
    }
  ]
}
```

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
document-classifier/
├── core/                    # Standalone analyzer
│   ├── __init__.py
│   └── document_analyzer_v2.py
├── app/                     # FastAPI backend
│   ├── api/v1/             # REST endpoints
│   ├── ml/                 # ML components
│   ├── services/           # Business logic
│   └── models/             # Data models
├── frontend/               # Web UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container image
└── docker-compose.yml     # Service orchestration
```

## Configuration

Environment variables (`.env` file):
```bash
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=document_classifier
API_PORT=8000
MAX_FILE_SIZE_MB=50
```

## Docker Deployment

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Performance

- Processing: ~9 seconds for 30-page RFP
- Section coverage: 100%
- Metadata extraction: 150+ items per document
- Q&A detection: 95+ pairs per document

## Output Format

All content is properly nested within sections:

```json
{
  "Section Title": {
    "type": "section",
    "description": ["paragraph 1", "paragraph 2"],
    "metadata": {"key": "value"},
    "questions": [{"question": "...", "answer": "..."}],
    "sub_sections": {...}
  }
}
```

## License

MIT

## Version

2.0 - Production Ready

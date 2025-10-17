# Document Classifier API

A document parsing system that uses both LLM-based and algorithmic approaches to extract structured information from any document type.

## Features

- **Document Support**: Resumes, RFPs, Proposals, Manuals, Reports
- **Dual Parsing Engines**: LLM-powered and Algorithmic parsing
- **Advanced Question Detection**: Intelligent question identification and extraction
- **Robust PDF Processing**: Multiple extraction methods with fallbacks
- **Production Ready**: FastAPI, CORS, Health checks, Error handling

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key (for LLM parser)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd document-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
```bash
# Copy and edit config file
cp config.env.example config.env
# Add your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" >> config.env
```

4. **Run the server**
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```bash
GET /health
```

### LLM Parser (AI-powered)
```bash
# Upload file
POST /parse
Content-Type: multipart/form-data
Body: file (PDF)

# Parse from file path
POST /parse-file?file_path=/path/to/document.pdf
```

### Universal Parser (Algorithmic)
```bash
# Upload file
POST /parse-algorithmic
Content-Type: multipart/form-data
Body: file (PDF)

# Parse from file path
POST /parse-algorithmic-file?file_path=/path/to/document.pdf
```

## Usage Examples

### Python Client
```python
import requests

# Upload and parse document
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/parse',
        files={'file': f}
    )
    result = response.json()

# Parse from file path
response = requests.post(
    'http://localhost:8000/parse-algorithmic-file',
    params={'file_path': '/path/to/document.pdf'}
)
result = response.json()
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health

# Parse uploaded file
curl -X POST -F "file=@document.pdf" http://localhost:8000/parse

# Parse from file path
curl -X POST "http://localhost:8000/parse-algorithmic-file?file_path=/path/to/document.pdf"
```

## Response Format

```json
{
  "document_metadata": {
    "total_pages": 5,
    "extraction_method": "pdfplumber",
    "total_elements": 150,
    "file_path": "document.pdf"
  },
  "metadata": {
    "email": "contact@example.com",
    "phone": "(555) 123-4567",
    "url": "https://example.com"
  },
  "sections": [
    {
      "title": "Executive Summary",
      "level": 1,
      "content": [
        "This document provides...",
        "Key findings include..."
      ],
      "subsections": [
        {
          "title": "Key Metrics",
          "level": 2,
          "content": ["Revenue: $2.5M", "Growth: 15%"],
          "subsections": [],
          "metadata": {}
        }
      ],
      "metadata": {}
    }
  ],
  "content": [
    "Standalone content blocks"
  ]
}
```

## Architecture

The system consists of two main parsing engines:

### LLM Parser (GPT-4o-mini)
- **Best for**: Complex documents requiring AI understanding
- **Features**: Advanced text normalization, smart section detection, question detection
- **Performance**: High accuracy, moderate speed, requires API key

### Universal Parser (Algorithmic)
- **Best for**: Any document type, production environments
- **Features**: Pattern recognition, structure detection, content classification
- **Performance**: Fast processing, no API dependencies, high reliability

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Server Configuration
```python
# app/api.py
app = FastAPI(
    title="Document Classifier API",
    description="Intelligent document parsing system",
    version="1.0.0"
)
```

## Production Deployment

### Docker Deployment
```bash
# Build image
docker build -t document-classifier .

# Run container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key document-classifier
```

### Docker Compose
```bash
docker-compose up -d
```

### Environment Setup
```bash
# Production environment
export OPENAI_API_KEY=your_production_key
export ENVIRONMENT=production
uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Monitoring

### Health Checks
```bash
curl http://localhost:8000/health
```

### Logs
- Application logs: Console output
- Error tracking: FastAPI error handling
- Processing metrics: Built-in logging

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid file type, missing parameters)
- `404`: File Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Security

- **File Validation**: PDF files only
- **CORS**: Configurable origins
- **Input Sanitization**: Automatic cleanup
- **Temporary Files**: Automatic cleanup after processing

**Document Classifier API** - Intelligent document parsing for any use case.
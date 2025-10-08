# Document Classification API

Intelligent document parsing system for unstructured PDFs (RFPs, resumes, contracts).

## Features

- Automatic section detection
- Metadata extraction (dates, amounts, contacts)
- Question-answer identification
- Content type classification
- MongoDB Atlas storage
- REST API endpoints

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
export MONGODB_URI="mongodb+srv://user:password@cluster.mongodb.net/document_classifier"
```

### 3. Run API Server

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf"
```

### Get Results
```bash
curl "http://localhost:8000/api/v1/documents/{document_id}"
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

Set environment variables:
- `MONGODB_URI` - MongoDB connection string (required)
- `API_PORT` - API port (default: 8000)
- `MAX_FILE_SIZE_MB` - Max upload size (default: 50)

## Production Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
ENV MONGODB_URI=""
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t document-classifier .
docker run -p 8000:8000 -e MONGODB_URI="your_uri" document-classifier
```

## License

MIT


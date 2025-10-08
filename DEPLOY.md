# Production Deployment Guide

## Prerequisites

- Python 3.11+
- MongoDB Atlas account
- 2GB RAM minimum
- 10GB disk space

## Environment Setup

### 1. MongoDB Atlas

1. Create account at https://cloud.mongodb.com
2. Create cluster (M0 free tier or higher)
3. Create database user
4. Whitelist IP addresses
5. Get connection string

### 2. Environment Variables

Create `.env` file:
```bash
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/document_classifier
API_PORT=8000
MAX_FILE_SIZE_MB=50
DEBUG=false
```

## Deployment Options

### Option 1: Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run server
python -m app.main
```

### Option 2: Docker

```bash
# Build image
docker build -t document-classifier .

# Run container
docker run -d \
  -p 8000:8000 \
  -e MONGODB_URI="your_uri" \
  --name doc-classifier \
  document-classifier
```

### Option 3: Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - API_PORT=8000
    restart: unless-stopped
```

Run: `docker-compose up -d`

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "api": "operational",
    "database": "operational"
  }
}
```

## Monitoring

- API logs: Check console output
- Database: MongoDB Atlas dashboard
- Performance: `/api/v1/documents` response times

## Scaling

### Horizontal Scaling

Deploy multiple API instances behind load balancer:

```bash
# Instance 1
docker run -p 8001:8000 -e MONGODB_URI="..." document-classifier

# Instance 2
docker run -p 8002:8000 -e MONGODB_URI="..." document-classifier

# Instance 3
docker run -p 8003:8000 -e MONGODB_URI="..." document-classifier
```

### Vertical Scaling

Upgrade MongoDB Atlas tier:
- M0 (512MB) → M2 (2GB) → M5 (5GB)

## Security

1. **Never commit secrets** - Use environment variables
2. **Use HTTPS** - Deploy behind reverse proxy (nginx)
3. **Limit CORS** - Set specific origins in production
4. **Rate limiting** - Already implemented (10 req/min default)
5. **API authentication** - Enable API key validation

## Troubleshooting

### MongoDB connection failed
- Check URI format
- Verify IP whitelist
- Test with `mongosh` CLI

### API not responding
- Check logs: `docker logs doc-classifier`
- Verify port not in use: `netstat -an | findstr :8000`
- Check MongoDB connection

### High memory usage
- Reduce MAX_CONCURRENT_BATCH (default: 5)
- Process smaller documents
- Upgrade server RAM

## Production Checklist

- [ ] MongoDB Atlas configured
- [ ] Environment variables set
- [ ] SSL/TLS enabled
- [ ] CORS origins restricted
- [ ] Rate limiting configured
- [ ] Logging enabled
- [ ] Health checks working
- [ ] Backup strategy in place
- [ ] Monitoring setup
- [ ] API documentation accessible

## Support

For issues, check:
1. Application logs
2. MongoDB Atlas metrics
3. API documentation at `/docs`


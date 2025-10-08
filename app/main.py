"""
Main FastAPI application for document classification system.

This module initializes the FastAPI app with all routes, middleware,
and configuration.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import documents_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Document Classification API...")
    
    # Initialize services (database connections, etc.)
    # This would be done here in production
    
    logger.info("API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Classification API...")
    
    # Cleanup (close database connections, etc.)
    
    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Document Classification API",
    description="""
    Intelligent document classification and extraction system.
    
    Features:
    - PDF document upload and processing
    - Section detection and hierarchical structure
    - Metadata extraction (dates, amounts, contacts, etc.)
    - Question-answer identification
    - Content type classification
    - Full-text search
    - Adaptive learning from user feedback
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request, call_next):
    """Add request ID to all requests for tracing."""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests."""
    import time
    
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )
    
    return response


# ============================================================================
# ROUTES
# ============================================================================

# Include API routers
app.include_router(documents_router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information and available endpoints
    """
    return {
        "name": "Document Classification API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "documents": "/api/v1/documents",
            "upload": "/api/v1/documents/upload",
            "search": "/api/v1/documents/search",
        }
    }


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API and services
    """
    # In production, check database, cache, etc.
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "api": "operational",
            "database": "operational",
            "cache": "operational",
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred"
        }
    )


# ============================================================================
# STARTUP MESSAGE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


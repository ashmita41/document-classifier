"""
Document Classifier API - Main Entry Point

Production-ready document parsing system with dual parsing engines.
"""

import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Set default configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("ENVIRONMENT", "production") == "development"
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Info: OPENAI_API_KEY not set. LLM parser will not be available. Universal parser will work without API key.")
    else:
        print("Info: OPENAI_API_KEY found. Both LLM and Universal parsers will be available.")
    
    # Run the server
    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )

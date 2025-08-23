"""
Research Agent Web Backend
FastAPI application for the Research Agent web interface.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add research_agent to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'research_agent'))

from .api.research import router as research_router
from .api.websocket import router as websocket_router
from .core.config import settings
from .core.logging_config import setup_logging
from .services.research_service import research_service

# Setup logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    print("🚀 Research Agent Web Backend starting up...")
    
    # Validate required environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set. Enhanced features may not work.")
    else:
        print(f"✅ OpenAI API Key loaded (length: {len(api_key)})")
    
    yield
    
    # Shutdown
    print("👋 Research Agent Web Backend shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Research Agent API",
    description="Advanced Research Agent with AI-powered source discovery and validation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(research_router, prefix="/api/research", tags=["research"])
app.include_router(websocket_router, prefix="/api/ws", tags=["websocket"])

# Health check endpoint
@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with research service validation."""
    # Get research service validation status
    service_status = research_service.validate_setup()
    
    return {
        "status": "healthy" if service_status["setup_valid"] else "degraded",
        "service": "research-agent-backend",
        "version": "2.0.0",
        "features": {
            "unified_research": service_status["agents_available"],
            "web_search": True,
            "gpt4o_mini": True,
            "inline_citations": True,
            "websocket_support": True,
            "openai_configured": service_status["api_key_configured"],
            "openai_valid_format": service_status["api_key_configured"]  # If configured, format is valid
        },
        "research_service": {
            "setup_valid": service_status["setup_valid"],
            "issues": service_status["issues"],
            "recommendations": []  # Add empty recommendations for compatibility
        }
    }

# Root endpoint
@app.get("/api")
async def root() -> Dict[str, str]:
    """Root API endpoint."""
    return {
        "message": "Research Agent API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
"""
Research Agent Web Backend
FastAPI application for the Research Agent web interface.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add research_agent to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'research_agent'))

from .api.research import router as research_router
from .api.websocket import router as websocket_router
from .core.config import settings
from .core.logging_config import setup_logging

# Setup logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    print("🚀 Research Agent Web Backend starting up...")
    
    # Validate required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Enhanced features may not work.")
    
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "research-agent-backend",
        "version": "1.0.0",
        "features": {
            "basic_research": True,
            "enhanced_research": True,
            "websocket_support": True,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
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
    return {"error": "Endpoint not found", "path": str(request.url)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    return {"error": "Internal server error", "message": str(exc)}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
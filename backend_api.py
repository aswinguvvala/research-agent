"""
Backend API Service for Hybrid Architecture
Runs the full research system on a paid cloud instance and provides REST API endpoints.
Frontend (Streamlit Cloud) communicates with this backend via HTTP requests.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
import os
import json
import uuid
from datetime import datetime
import logging

# Import both lightweight and full systems
try:
    from src.enhanced_research_system import EnhancedResearchSystem
    FULL_SYSTEM_AVAILABLE = True
except ImportError:
    FULL_SYSTEM_AVAILABLE = False

from lightweight_research_system import LightweightResearchSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AI Research Agent API",
    description="Backend API for the AI Research Agent system",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ResearchRequest(BaseModel):
    query: str
    agents: Optional[List[str]] = ["academic", "technical", "business"]
    system_type: Optional[str] = "lightweight"  # or "full"
    max_duration: Optional[int] = 300

class ResearchResponse(BaseModel):
    session_id: str
    query: str
    results: Dict[str, Any]
    status: str
    timestamp: str

class SystemStatus(BaseModel):
    status: str
    system_type: str
    agents_available: List[str]
    stats: Dict[str, Any]
    timestamp: str

# Global system instances
lightweight_system = None
full_system = None
active_sessions = {}


async def initialize_systems():
    """Initialize research systems on startup"""
    global lightweight_system, full_system
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # Always initialize lightweight system
    try:
        lightweight_system = LightweightResearchSystem(api_key)
        logger.info("✅ Lightweight research system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize lightweight system: {e}")
    
    # Initialize full system if available
    if FULL_SYSTEM_AVAILABLE:
        try:
            config = {
                "chroma_db_path": "./api_research_db",
                "api_keys": {
                    "semantic_scholar": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
                    "news_api": os.getenv("NEWS_API_KEY")
                }
            }
            full_system = EnhancedResearchSystem(api_key, config)
            await full_system.initialize_system()
            logger.info("✅ Full research system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize full system: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    await initialize_systems()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Research Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/research": "POST - Conduct research",
            "/status": "GET - System status",
            "/sessions/{session_id}": "GET - Get session results",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "lightweight": lightweight_system is not None,
            "full": full_system is not None
        }
    }


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status and statistics"""
    global lightweight_system, full_system
    
    # Determine which systems are available
    available_systems = []
    if lightweight_system:
        available_systems.append("lightweight")
    if full_system:
        available_systems.append("full")
    
    if not available_systems:
        raise HTTPException(status_code=503, detail="No research systems available")
    
    # Get stats from primary system
    primary_system = full_system if full_system else lightweight_system
    
    if hasattr(primary_system, 'get_system_stats'):
        stats = primary_system.get_system_stats()
    else:
        stats = {"message": "Stats not available"}
    
    return SystemStatus(
        status="operational",
        system_type=", ".join(available_systems),
        agents_available=["academic", "technical", "business"] if lightweight_system else [],
        stats=stats,
        timestamp=datetime.now().isoformat()
    )


@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Conduct research using specified system"""
    global lightweight_system, full_system, active_sessions
    
    # Validate request
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Select system
    if request.system_type == "full" and full_system:
        system = full_system
        system_type = "full"
    elif lightweight_system:
        system = lightweight_system
        system_type = "lightweight"
    else:
        raise HTTPException(status_code=503, detail="No research system available")
    
    # Generate session ID
    session_id = str(uuid.uuid4())[:12]
    
    try:
        # Store session as active
        active_sessions[session_id] = {
            "query": request.query,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "system_type": system_type
        }
        
        logger.info(f"Starting research session {session_id} with {system_type} system")
        
        # Conduct research
        if system_type == "full":
            # Use full system method
            results = await system.conduct_enhanced_research(
                request.query, 
                max_duration=request.max_duration
            )
        else:
            # Use lightweight system method
            results = await system.research(
                request.query, 
                selected_agents=request.agents
            )
        
        # Update session
        active_sessions[session_id] = {
            "query": request.query,
            "results": results,
            "status": "completed",
            "started_at": active_sessions[session_id]["started_at"],
            "completed_at": datetime.now().isoformat(),
            "system_type": system_type
        }
        
        logger.info(f"Research session {session_id} completed successfully")
        
        return ResearchResponse(
            session_id=session_id,
            query=request.query,
            results=results,
            status="completed",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Research session {session_id} failed: {e}")
        
        # Update session with error
        active_sessions[session_id] = {
            "query": request.query,
            "status": "failed",
            "error": str(e),
            "started_at": active_sessions[session_id]["started_at"],
            "failed_at": datetime.now().isoformat(),
            "system_type": system_type
        }
        
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session results by ID"""
    global active_sessions
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    
    return {
        "session_id": session_id,
        "query": session_data["query"],
        "status": session_data["status"],
        "system_type": session_data["system_type"],
        "started_at": session_data["started_at"],
        "results": session_data.get("results"),
        "error": session_data.get("error"),
        "completed_at": session_data.get("completed_at"),
        "failed_at": session_data.get("failed_at")
    }


@app.get("/sessions")
async def list_sessions(limit: int = 10):
    """List recent sessions"""
    global active_sessions
    
    sessions = []
    for session_id, data in list(active_sessions.items())[-limit:]:
        sessions.append({
            "session_id": session_id,
            "query": data["query"][:100] + "..." if len(data["query"]) > 100 else data["query"],
            "status": data["status"],
            "system_type": data["system_type"],
            "started_at": data["started_at"]
        })
    
    return {
        "sessions": sessions,
        "total": len(active_sessions)
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    global active_sessions
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    
    return {"message": f"Session {session_id} deleted"}


@app.get("/systems/available")
async def get_available_systems():
    """Get available research systems"""
    systems = []
    
    if lightweight_system:
        systems.append({
            "type": "lightweight",
            "name": "Lightweight Research System",
            "agents": ["academic", "technical", "business"],
            "memory_usage": "~800MB",
            "features": ["3 specialized agents", "In-memory storage", "Cost-optimized"]
        })
    
    if full_system:
        systems.append({
            "type": "full",
            "name": "Enhanced Research System",
            "agents": ["academic", "technical", "business", "social", "fact_checker", "synthesis"],
            "memory_usage": "~2GB",
            "features": ["6 specialized agents", "Vector database", "Advanced reasoning", "Modern APIs"]
        })
    
    return {"available_systems": systems}


# Development server configuration
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )
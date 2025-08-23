"""
Configuration settings for Research Agent Web Backend
"""

import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings

# Load environment variables from .env file early
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Vite dev server (alternate port)
        "http://localhost:3002",  # Vite dev server (another alternate port)
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:5173"
    ]
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Research Agent settings
    RESEARCH_AGENT_ROOT_PATH: Optional[str] = os.getenv("RESEARCH_AGENT_ROOT_PATH", "/Users/aswin/new_research_agent")
    MAX_SOURCES: int = 10
    RELEVANCE_THRESHOLD: float = 0.35
    CONTENT_VALIDATION_THRESHOLD: float = 0.65
    CONSENSUS_THRESHOLD: float = 0.6
    RESEARCH_TIMEOUT_SECONDS: int = 900  # 15 minutes for research operations
    PROGRESS_UPDATE_INTERVAL: int = 15   # Send progress updates every 15 seconds
    
    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 100
    
    # Background task settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Database settings (SQLite for simplicity)
    DATABASE_URL: str = "sqlite:///./research_agent_web.db"
    
    # File storage
    UPLOAD_DIR: str = "uploads"
    EXPORT_DIR: str = "exports"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_PATH: Optional[str] = os.getenv("LOG_FILE_PATH", "logs/research_agent_web.log")
    LOG_FILE_ENABLED: bool = True  # Can be disabled for production
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB per log file
    LOG_BACKUP_COUNT: int = 5  # Keep 5 backup files
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Validate critical settings
if settings.DEBUG:
    print(f"🔧 Debug mode enabled")
    print(f"📍 Server: {settings.HOST}:{settings.PORT}")
    print(f"🔑 OpenAI API Key: {'✅ Set' if settings.OPENAI_API_KEY else '❌ Not set'}")

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EXPORT_DIR, exist_ok=True)

# Ensure logs directory exists if log file is configured
if settings.LOG_FILE_ENABLED and settings.LOG_FILE_PATH:
    log_dir = os.path.dirname(settings.LOG_FILE_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
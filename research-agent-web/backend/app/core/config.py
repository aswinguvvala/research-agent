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

class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    
    # Research Agent settings
    MAX_SOURCES: int = 10
    RELEVANCE_THRESHOLD: float = 0.35
    CONTENT_VALIDATION_THRESHOLD: float = 0.65
    CONSENSUS_THRESHOLD: float = 0.6
    
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
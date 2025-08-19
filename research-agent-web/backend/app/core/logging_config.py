"""
Logging configuration for Research Agent Web Backend
"""

import logging
import sys
from .config import settings

def setup_logging():
    """Setup logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=settings.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for errors
    file_handler = logging.FileHandler("research_agent_web.log")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Specific loggers
    
    # FastAPI logger
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.INFO)
    
    # Research Agent logger
    research_logger = logging.getLogger("research_agent")
    research_logger.setLevel(logging.INFO)
    
    # WebSocket logger
    ws_logger = logging.getLogger("websocket")
    ws_logger.setLevel(logging.INFO)
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    if settings.DEBUG:
        root_logger.info("🔧 Logging configured in debug mode")
    else:
        root_logger.info("📝 Logging configured")
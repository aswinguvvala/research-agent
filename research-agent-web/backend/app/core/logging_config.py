"""
Logging configuration for Research Agent Web Backend
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from .config import settings

def setup_logging():
    """Setup logging configuration with rotation and proper file handling."""
    
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
    
    # File handler with rotation (only if enabled)
    if settings.LOG_FILE_ENABLED and settings.LOG_FILE_PATH:
        try:
            # Ensure log directory exists
            log_file_path = Path(settings.LOG_FILE_PATH)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use RotatingFileHandler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                filename=settings.LOG_FILE_PATH,
                maxBytes=settings.LOG_MAX_BYTES,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            if settings.DEBUG:
                root_logger.info(f"📝 File logging enabled: {settings.LOG_FILE_PATH}")
                root_logger.info(f"📁 Max file size: {settings.LOG_MAX_BYTES / 1024 / 1024:.1f}MB")
                root_logger.info(f"📦 Backup count: {settings.LOG_BACKUP_COUNT}")
        except Exception as e:
            root_logger.error(f"❌ Failed to setup file logging: {e}")
            root_logger.warning("⚠️ Continuing with console logging only")
    
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
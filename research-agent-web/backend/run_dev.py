#!/usr/bin/env python3
"""
Development server runner for Research Agent Web Backend
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables for development
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("HOST", "127.0.0.1")
os.environ.setdefault("PORT", "8000")

# Ensure OPENAI_API_KEY is available
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY environment variable not set!")
    print("   Enhanced research features may not work properly.")
    print("   Set your OpenAI API key with: export OPENAI_API_KEY='your-key-here'")
    print()

def main():
    """Run the development server."""
    print("🚀 Starting Research Agent Web Backend (Development)")
    print("📍 http://127.0.0.1:8000")
    print("📚 API Documentation: http://127.0.0.1:8000/api/docs")
    print("🔄 Auto-reload enabled")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(current_dir)],
        log_level="info"
    )

if __name__ == "__main__":
    main()
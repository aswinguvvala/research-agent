#!/bin/bash

# Start the Research Agent Web Backend
# This script activates the virtual environment and starts the FastAPI server

echo "🚀 Starting Research Agent Web Backend..."

# Check if we're in the right directory
if [[ ! -f "app/main.py" ]]; then
    echo "❌ Error: Must run from the backend directory"
    echo "   cd research-agent-web/backend"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    echo "📥 Installing dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "❌ Error: .env file not found"
    echo "   Please copy .env.example to .env and configure your API keys"
    exit 1
fi

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "❌ Error: OpenAI API key not configured in .env file"
    echo "   Please add your OpenAI API key to the .env file"
    exit 1
fi

echo "✅ Environment configured"
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📝 API documentation: http://localhost:8000/docs"
echo "💬 WebSocket endpoint: ws://localhost:8000/api/ws/research/{client_id}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
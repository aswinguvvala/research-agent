# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Quick Start & Testing
```bash
# Simple demo (fastest way to test)
python simple_demo.py

# Full system demo
python quick_start.py

# Web dashboard (requires additional dependencies)
pip install streamlit plotly
streamlit run src/research_dashboard.py
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Minimal dependencies for basic functionality
pip install openai

# Full system with optional components
pip install -r requirements.txt
```

### Testing Individual Components
```bash
# Test individual agents
python -c "from src.specialized_agents import AcademicSpecialist; print('Agents working')"

# Test vector database
python -c "from src.vector_knowledge_system import VectorKnowledgeSystem; print('Vector system working')"

# Test cost optimizer
python -c "from src.cost_optimizer import CostOptimizer; print('Cost optimizer working')"
```

## Architecture Overview

### Multi-Agent Research System
The system is built around a **multi-agent architecture** with specialized AI agents that coordinate to conduct comprehensive research:

- **EnhancedResearchSystem**: Main orchestrator that coordinates all components
- **MultiAgentResearchSystem**: Manages agent communication and task distribution
- **BaseAgent**: Abstract base class for all specialized agents

### Specialized Research Agents
Four domain-specific agents that provide different perspectives:

1. **AcademicSpecialist**: Academic papers, citations, literature reviews
2. **TechnicalSpecialist**: Implementation details, technical analysis, best practices  
3. **BusinessSpecialist**: Market analysis, business implications, ROI assessment
4. **SocialSpecialist**: Social trends, ethics, human factors, adoption patterns

### Quality Assurance Layer
- **FactChecker**: Source verification, bias detection, consistency validation
- **SynthesisAgent**: Cross-domain integration, pattern recognition, comprehensive reporting

### Supporting Systems
- **VectorKnowledgeSystem**: Semantic search using Chroma DB and sentence transformers
- **ModernAPIManager**: Integration with Semantic Scholar, OpenAlex, enhanced arXiv
- **ChainOfThoughtEngine**: Advanced reasoning with metacognitive reflection
- **CostOptimizer**: Token usage optimization and cost tracking

## Key Features

### Agent Communication
- Agents use a message-passing system with **MessageType** enum (research_request, coordination, fact_check, etc.)
- **ResearchTask** dataclass manages task assignment and status tracking
- **MessageBroker** handles inter-agent coordination

### Vector Knowledge System
- Uses **ChromaDB** for persistent semantic search
- **KnowledgeItem** stores research with embeddings for similarity search
- Maintains research session history across multiple queries

### Cost Optimization
- Defaults to **GPT-4o mini** (cheapest OpenAI model)
- Token usage tracking with real-time cost estimation
- Context-aware token limits (quick: 100, summary: 150, analysis: 250, etc.)
- Request optimization with frequency/presence penalties

### Modern API Integration
- **Semantic Scholar API**: 200M+ academic papers with metadata
- **OpenAlex**: Comprehensive academic database with citations
- **Enhanced arXiv**: Smart categorization and trend analysis
- **Real-time News API**: Current events correlation

## Project Structure

```
research-agent/
├── src/                                    # Core system modules
│   ├── enhanced_research_system.py        # Main system orchestrator
│   ├── multi_agent_research_system.py     # Agent coordination framework
│   ├── specialized_agents.py              # Domain expert agents
│   ├── quality_agents.py                  # Fact-checking & synthesis
│   ├── vector_knowledge_system.py         # Semantic search & memory
│   ├── modern_api_sources.py              # External data sources
│   ├── advanced_reasoning.py              # Chain-of-thought engine
│   ├── cost_optimizer.py                  # Cost-efficient operations
│   └── research_dashboard.py              # Streamlit web interface
├── simple_demo.py                         # Minimal working example
├── quick_start.py                         # Full system demo
├── research_sessions/                     # Saved research sessions
├── reports/                              # Generated research reports
└── enhanced_research_db/                 # ChromaDB vector storage
```

## Configuration

### API Keys
The system uses OpenAI API keys for core functionality, with optional keys for enhanced features:
- **OpenAI**: Required for all LLM operations
- **Semantic Scholar**: Enhanced academic search
- **News API**: Real-time news integration

API keys are typically set in the demo files or passed to the system initializer.

### System Configuration
Common configuration options:
```python
config = {
    "cost_optimized": True,
    "model": "gpt-4o-mini",
    "max_tokens": 150,
    "chroma_db_path": "./enhanced_research_db",
    "api_keys": {
        "semantic_scholar": "optional-key",
        "news_api": "optional-key"
    }
}
```

## Performance & Quality Metrics

### Research Quality
- **Source Diversity**: 5-10 different data sources per session
- **Cross-Domain Integration**: 4+ specialized agent perspectives  
- **Fact-Checking Coverage**: 90%+ of claims verified
- **Reasoning Depth**: 6-step chain-of-thought analysis

### Performance Benchmarks
- **Research Speed**: 2-5 minutes for comprehensive analysis
- **Cost Efficiency**: $0.01-$0.05 per research session with GPT-4o mini
- **Token Optimization**: 30-50% reduction through intelligent context management

## Development Notes

### Async/Await Pattern
The system is built with **asyncio** for concurrent agent operations. Most core functions are async and should be awaited.

### Error Handling
- Graceful degradation when external APIs are unavailable
- Cost optimization prevents runaway token usage
- Agent coordination includes retry logic and timeout handling

### Extensibility
- **BaseAgent** abstract class for custom specialized agents
- **ModernAPIManager** supports adding new data sources
- **ChainOfThoughtEngine** allows custom reasoning modules
- **ResearchDashboard** can be extended with new visualizations

When working with this codebase, prioritize the multi-agent coordination system and understand how agents communicate through the message broker. The vector knowledge system maintains context across sessions, and the cost optimizer ensures efficient resource usage.
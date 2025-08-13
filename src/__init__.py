"""
Enhanced Multi-Agent Research System
A revolutionary AI research platform that combines specialized agents, 
vector knowledge systems, modern APIs, and advanced reasoning.
"""

__version__ = "2.0.0"
__author__ = "Enhanced Research Team"

# Core system exports
from .enhanced_research_system import EnhancedResearchSystem

# Multi-agent components
from .multi_agent_research_system import (
    MultiAgentResearchSystem,
    BaseAgent, 
    ResearchCoordinator,
    MessageBroker,
    ResearchTask,
    AgentMessage,
    MessageType,
    AgentStatus
)

# Specialized agents
from .specialized_agents import (
    AcademicSpecialist,
    TechnicalSpecialist,
    BusinessSpecialist,
    SocialSpecialist
)

# Quality agents
from .quality_agents import (
    FactChecker,
    SynthesisAgent
)

# Vector knowledge system
from .vector_knowledge_system import (
    VectorKnowledgeSystem,
    KnowledgeItem
)

# Modern API sources
from .modern_api_sources import (
    SemanticScholarAPI,
    OpenAlexAPI,
    EnhancedArxivSearch,
    RealTimeNewsAPI,
    ModernAPIManager
)

# Advanced reasoning
from .advanced_reasoning import (
    ChainOfThoughtEngine,
    ReasoningEnhancedAgent,
    ReasoningType,
    ReasoningStep,
    ReasoningChain
)

# Dashboard (optional import)
try:
    from .research_dashboard import ResearchDashboard
    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False

__all__ = [
    # Core system
    'EnhancedResearchSystem',
    
    # Multi-agent framework
    'MultiAgentResearchSystem',
    'BaseAgent',
    'ResearchCoordinator', 
    'MessageBroker',
    'ResearchTask',
    'AgentMessage',
    'MessageType',
    'AgentStatus',
    
    # Specialized agents
    'AcademicSpecialist',
    'TechnicalSpecialist',
    'BusinessSpecialist', 
    'SocialSpecialist',
    
    # Quality agents
    'FactChecker',
    'SynthesisAgent',
    
    # Vector knowledge
    'VectorKnowledgeSystem',
    'KnowledgeItem',
    
    # Modern APIs
    'SemanticScholarAPI',
    'OpenAlexAPI',
    'EnhancedArxivSearch',
    'RealTimeNewsAPI',
    'ModernAPIManager',
    
    # Advanced reasoning
    'ChainOfThoughtEngine',
    'ReasoningEnhancedAgent',
    'ReasoningType',
    'ReasoningStep',
    'ReasoningChain',
]

if _DASHBOARD_AVAILABLE:
    __all__.append('ResearchDashboard')


def get_system_info() -> dict:
    """Get information about the enhanced research system"""
    return {
        "version": __version__,
        "name": "Enhanced Multi-Agent Research System",
        "description": "AI research platform with specialized agents, vector knowledge, and advanced reasoning",
        "capabilities": [
            "Multi-agent coordination with specialized domains",
            "Vector-based semantic search and knowledge management", 
            "Modern API integration (Semantic Scholar, OpenAlex, enhanced arXiv)",
            "Chain-of-thought reasoning with metacognitive reflection",
            "Real-time fact-checking and quality assurance",
            "Interactive web dashboard (if Streamlit available)",
            "Cross-domain synthesis and pattern recognition"
        ],
        "agents": [
            "Academic Research Specialist",
            "Technical Implementation Expert", 
            "Business Applications Analyst",
            "Social Impact Researcher",
            "Fact Verification Specialist",
            "Research Synthesis Specialist"
        ],
        "dashboard_available": _DASHBOARD_AVAILABLE
    }

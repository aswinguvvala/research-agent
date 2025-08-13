"""
Hybrid Frontend for AI Research Agent
Lightweight frontend that runs on Streamlit Cloud (free) and communicates 
with backend API service running on paid cloud infrastructure.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

# Page configuration
st.set_page_config(
    page_title="AI Research Agent - Hybrid",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c62828;
    }
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
    }
    .system-status {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef6c00;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


class APIClient:
    """Client for communicating with backend API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minute timeout
    
    def health_check(self) -> bool:
        """Check if backend API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_status(self) -> Optional[Dict]:
        """Get system status from backend"""
        try:
            response = self.session.get(f"{self.base_url}/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get system status: {e}")
            return None
    
    def get_available_systems(self) -> Optional[Dict]:
        """Get available research systems"""
        try:
            response = self.session.get(f"{self.base_url}/systems/available", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get available systems: {e}")
            return None
    
    def conduct_research(self, query: str, agents: List[str], system_type: str = "lightweight") -> Optional[Dict]:
        """Conduct research via backend API"""
        try:
            payload = {
                "query": query,
                "agents": agents,
                "system_type": system_type,
                "max_duration": 300
            }
            
            response = self.session.post(
                f"{self.base_url}/research",
                json=payload,
                timeout=360  # 6 minute timeout for research
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"Research failed: {error_detail}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Research request timed out. Please try again with a simpler query.")
            return None
        except Exception as e:
            st.error(f"Research request failed: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session results by ID"""
        try:
            response = self.session.get(f"{self.base_url}/sessions/{session_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get session: {e}")
            return None
    
    def list_sessions(self, limit: int = 10) -> Optional[Dict]:
        """List recent sessions"""
        try:
            response = self.session.get(f"{self.base_url}/sessions?limit={limit}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to list sessions: {e}")
            return None


def initialize_session_state():
    """Initialize session state variables"""
    if "api_client" not in st.session_state:
        st.session_state.api_client = None
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = None
    if "available_systems" not in st.session_state:
        st.session_state.available_systems = None
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    if "current_research" not in st.session_state:
        st.session_state.current_research = None


def get_backend_url() -> str:
    """Get backend API URL from secrets or environment"""
    try:
        # Try Streamlit secrets first
        return st.secrets.get("BACKEND_API_URL", "")
    except:
        # Try environment variable
        return os.getenv("BACKEND_API_URL", "")


def setup_api_client():
    """Setup API client with backend URL"""
    backend_url = get_backend_url()
    
    if not backend_url:
        st.error("""
        ⚠️ Backend API URL not configured. Please add it to your Streamlit secrets:
        
        ```toml
        BACKEND_API_URL = "https://your-backend-api.com"
        ```
        
        Or set the BACKEND_API_URL environment variable.
        """)
        return None
    
    try:
        client = APIClient(backend_url)
        
        # Test connection
        if not client.health_check():
            st.error(f"❌ Backend API at {backend_url} is not responding. Please check the URL and ensure the backend service is running.")
            return None
        
        return client
    except Exception as e:
        st.error(f"Failed to setup API client: {e}")
        return None


def display_backend_status(client: APIClient):
    """Display backend system status"""
    status = client.get_status()
    
    if status:
        st.markdown(f"""
        <div class="system-status">
            🔗 <strong>Backend Status:</strong> {status['status']} | 
            <strong>System:</strong> {status['system_type']} | 
            <strong>Agents:</strong> {len(status['agents_available'])}
        </div>
        """, unsafe_allow_html=True)
        
        return status
    else:
        st.markdown("""
        <div class="error-message">
            ❌ <strong>Backend Unavailable:</strong> Unable to connect to backend API
        </div>
        """, unsafe_allow_html=True)
        return None


def display_system_selection(available_systems: List[Dict]) -> str:
    """Display system selection interface"""
    if not available_systems:
        return "lightweight"
    
    system_options = {}
    for system in available_systems:
        name = f"{system['name']} ({system['memory_usage']})"
        system_options[name] = system['type']
    
    if len(system_options) > 1:
        selected_name = st.sidebar.selectbox(
            "🖥️ Select Research System",
            options=list(system_options.keys()),
            help="Choose between lightweight (free tier) or full system (paid tier)"
        )
        return system_options[selected_name]
    else:
        # Only one system available
        system_name = list(system_options.keys())[0]
        st.sidebar.info(f"🖥️ **System:** {system_name}")
        return list(system_options.values())[0]


def display_agent_selection(system_type: str, available_systems: List[Dict]) -> List[str]:
    """Display agent selection interface"""
    st.sidebar.markdown("### 🤖 Select Research Agents")
    
    # Find available agents for selected system
    available_agents = []
    for system in available_systems:
        if system['type'] == system_type:
            available_agents = system['agents']
            break
    
    if not available_agents:
        available_agents = ["academic", "technical", "business"]
    
    agent_descriptions = {
        "academic": "📚 Academic research, papers, and scholarly analysis",
        "technical": "⚙️ Technical implementation and practical solutions",
        "business": "💼 Business applications and market analysis",
        "social": "👥 Social impact and human factors analysis",
        "fact_checker": "✅ Fact verification and quality assurance",
        "synthesis": "🔄 Cross-domain synthesis and integration"
    }
    
    selected_agents = []
    
    for agent in available_agents:
        description = agent_descriptions.get(agent, f"{agent.title()} specialist")
        if st.sidebar.checkbox(
            f"{agent.title()} Agent",
            value=True,  # All selected by default
            help=description
        ):
            selected_agents.append(agent)
    
    return selected_agents


def display_research_results(results: Dict[str, Any]):
    """Display research results"""
    if "error" in results or results.get("status") == "failed":
        error_msg = results.get("error", "Unknown error occurred")
        st.markdown(f"""
        <div class="error-message">
            <strong>❌ Research Failed:</strong> {error_msg}
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display success message
    session_id = results.get("session_id", "unknown")
    st.markdown(f"""
    <div class="success-message">
        <strong>✅ Research Completed!</strong> Session ID: {session_id}
    </div>
    """, unsafe_allow_html=True)
    
    research_data = results.get("results", {})
    
    # Handle both lightweight and full system response formats
    if "synthesis" in research_data:
        st.markdown("### 🔄 Research Synthesis")
        st.markdown(f"**Key Insights:** {research_data['synthesis']}")
    elif "integrated_findings" in research_data:
        # Full system format
        synthesis = research_data.get("integrated_findings", {}).get("synthesis", {})
        if synthesis:
            st.markdown("### 🔄 Research Synthesis")
            st.markdown(f"**Executive Summary:** {synthesis.get('executive_summary', 'N/A')}")
            
            insights = synthesis.get("main_insights", [])
            if insights:
                st.markdown("**Key Insights:**")
                for i, insight in enumerate(insights[:5], 1):
                    st.markdown(f"{i}. {insight}")
    
    # Display agent results
    st.markdown("### 📊 Agent Findings")
    
    # Handle different result formats
    if "results_by_agent" in research_data:
        # Lightweight system format
        results_by_agent = research_data["results_by_agent"]
        
        for agent_type, data in results_by_agent.items():
            with st.expander(f"{agent_type.title()} Agent Results", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Summary:** {data.get('summary', 'No summary available')}")
                    
                    findings = data.get('findings', [])
                    if findings:
                        st.markdown("**Key Findings:**")
                        for i, finding in enumerate(findings[:5], 1):
                            st.markdown(f"{i}. {finding}")
                
                with col2:
                    confidence = data.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.0%}")
                    
                    sources = data.get('sources', [])
                    if sources:
                        st.markdown("**Sources:**")
                        for source in sources:
                            st.markdown(f"• {source}")
    
    elif "agent_research" in research_data:
        # Full system format
        agent_results = research_data.get("agent_research", {})
        tasks_by_domain = agent_results.get("tasks_by_domain", {})
        
        for domain, tasks in tasks_by_domain.items():
            with st.expander(f"{domain.title()} Research", expanded=False):
                for task in tasks[:3]:  # Show first 3 tasks
                    st.markdown(f"**Task:** {task.get('task', 'N/A')}")
                    results_list = task.get('results', [])
                    if results_list:
                        for result in results_list[:1]:  # Show first result
                            findings = result.get('findings', [])
                            if findings:
                                st.markdown("**Findings:**")
                                for finding in findings[:3]:
                                    st.markdown(f"• {finding}")


def display_session_history(client: APIClient):
    """Display session history from backend"""
    sessions = client.list_sessions(limit=5)
    
    if sessions and sessions.get("sessions"):
        st.markdown("### 📚 Recent Sessions")
        
        for session in sessions["sessions"]:
            with st.expander(f"Session: {session['query'][:60]}..."):
                st.markdown(f"**ID:** {session['session_id']}")
                st.markdown(f"**System:** {session['system_type']}")
                st.markdown(f"**Status:** {session['status']}")
                st.markdown(f"**Started:** {session['started_at']}")
                
                if st.button(f"Load Results", key=f"load_{session['session_id']}"):
                    session_data = client.get_session(session['session_id'])
                    if session_data and session_data.get('results'):
                        st.session_state.current_research = {
                            "session_id": session['session_id'],
                            "results": session_data['results'],
                            "status": "completed"
                        }
                        st.rerun()


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Research Agent - Hybrid</h1>', unsafe_allow_html=True)
    st.markdown("**Frontend runs on Streamlit Cloud (free) • Backend runs on paid cloud infrastructure**")
    
    # Setup API client
    if st.session_state.api_client is None:
        st.session_state.api_client = setup_api_client()
    
    client = st.session_state.api_client
    if not client:
        st.stop()
    
    # Display backend status
    status = display_backend_status(client)
    if not status:
        st.stop()
    
    # Get available systems
    available_systems_data = client.get_available_systems()
    if not available_systems_data:
        st.error("Failed to get available systems from backend")
        st.stop()
    
    available_systems = available_systems_data.get("available_systems", [])
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # System selection
    selected_system = display_system_selection(available_systems)
    
    # Agent selection
    selected_agents = display_agent_selection(selected_system, available_systems)
    
    if not selected_agents:
        st.warning("⚠️ Please select at least one research agent.")
        return
    
    # Display system info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 System Info")
    for system in available_systems:
        if system['type'] == selected_system:
            st.sidebar.markdown(f"**{system['name']}**")
            st.sidebar.markdown(f"Memory: {system['memory_usage']}")
            st.sidebar.markdown("**Features:**")
            for feature in system['features']:
                st.sidebar.markdown(f"• {feature}")
            break
    
    # Main interface
    st.markdown("### 🔍 Research Query")
    
    # Query input
    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the latest developments in artificial intelligence for healthcare?",
        height=100,
        help="Ask any research question. The system will analyze it from multiple expert perspectives."
    )
    
    # Research button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Research", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner(f"Researching with {len(selected_agents)} agents on {selected_system} system..."):
                    
                    # Conduct research via API
                    results = client.conduct_research(query, selected_agents, selected_system)
                    
                    if results:
                        # Store results
                        st.session_state.current_research = results
                        st.session_state.research_history.append({
                            "query": query,
                            "results": results,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "system_type": selected_system
                        })
                        
                        st.success(f"Research completed! Session ID: {results.get('session_id', 'N/A')}")
                    else:
                        st.error("Research failed. Please try again.")
            else:
                st.warning("Please enter a research question.")
    
    # Display current research results
    if st.session_state.current_research:
        st.markdown("---")
        st.markdown("## 📋 Research Results")
        display_research_results(st.session_state.current_research)
    
    # Display session history
    st.markdown("---")
    display_session_history(client)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🤖 Hybrid AI Research System<br>
        <small>Frontend: Streamlit Cloud (Free) • Backend: Paid Cloud Infrastructure</small><br>
        <small>Powered by GPT-4o-mini for cost-effective research</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
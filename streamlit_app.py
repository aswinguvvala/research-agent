"""
Streamlit Dashboard for Lightweight Research System
Optimized for cloud deployment with minimal memory usage.
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import os

# Import our lightweight system
from lightweight_research_system import LightweightResearchSystem, SimpleResearchResult

# Page configuration
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
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
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "research_system" not in st.session_state:
        st.session_state.research_system = None
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    if "current_research" not in st.session_state:
        st.session_state.current_research = None
    if "system_stats" not in st.session_state:
        st.session_state.system_stats = {}


def get_api_key() -> str:
    """Get OpenAI API key from secrets or environment"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets["OPENAI_API_KEY"]
    except:
        # Try environment variable
        return os.getenv("OPENAI_API_KEY", "")


@st.cache_resource
def initialize_research_system(api_key: str):
    """Initialize the research system (cached for performance)"""
    if not api_key:
        return None
    
    try:
        system = LightweightResearchSystem(api_key)
        return system
    except Exception as e:
        st.error(f"Failed to initialize research system: {e}")
        return None


def display_agent_selection():
    """Display agent selection interface"""
    st.sidebar.markdown("### 🤖 Select Research Agents")
    
    agent_descriptions = {
        "academic": "📚 Academic research, papers, and scholarly analysis",
        "technical": "⚙️ Technical implementation and practical solutions",
        "business": "💼 Business applications and market analysis"
    }
    
    selected_agents = []
    
    for agent_type, description in agent_descriptions.items():
        if st.sidebar.checkbox(
            f"{agent_type.title()} Agent",
            value=True,  # All selected by default
            help=description
        ):
            selected_agents.append(agent_type)
    
    return selected_agents


def display_system_stats(system):
    """Display system statistics"""
    if system is None:
        return
    
    stats = system.get_system_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Sessions</h3>
            <h2>{}</h2>
        </div>
        """.format(stats.get("sessions_completed", 0)), unsafe_allow_html=True)
    
    with col2:
        avg_duration = stats.get("avg_session_duration", 0)
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Duration</h3>
            <h2>{:.1f}s</h2>
        </div>
        """.format(avg_duration), unsafe_allow_html=True)
    
    with col3:
        total_cost = stats.get("llm_usage", {}).get("estimated_cost", 0)
        st.markdown("""
        <div class="metric-card">
            <h3>Total Cost</h3>
            <h2>${:.4f}</h2>
        </div>
        """.format(total_cost), unsafe_allow_html=True)
    
    with col4:
        agents_count = stats.get("agents_available", 0)
        st.markdown("""
        <div class="metric-card">
            <h3>Agents</h3>
            <h2>{}</h2>
        </div>
        """.format(agents_count), unsafe_allow_html=True)


def display_research_results(results: Dict[str, Any]):
    """Display research results in a user-friendly format"""
    if "error" in results:
        st.markdown(f"""
        <div class="error-message">
            <strong>❌ Research Failed:</strong> {results["error"]}
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display metadata
    metadata = results.get("metadata", {})
    duration = metadata.get("duration_seconds", 0)
    agents_used = metadata.get("agents_used", [])
    
    st.markdown(f"""
    <div class="success-message">
        <strong>✅ Research Completed!</strong> Duration: {duration:.1f}s | Agents: {', '.join(agents_used)}
    </div>
    """, unsafe_allow_html=True)
    
    # Display synthesis
    if "synthesis" in results:
        st.markdown("### 🔄 Research Synthesis")
        st.markdown(f"**Key Insights:** {results['synthesis']}")
    
    # Display agent results
    st.markdown("### 📊 Agent Findings")
    
    results_by_agent = results.get("results_by_agent", {})
    
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


async def run_research(system, query: str, selected_agents: List[str]):
    """Run research asynchronously"""
    try:
        results = await system.research(query, selected_agents)
        return results
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Research Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Lightweight multi-agent research system optimized for cloud deployment**")
    
    # API Key Setup
    api_key = get_api_key()
    
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please add it to your Streamlit secrets or environment variables.")
        st.info("""
        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Add `OPENAI_API_KEY = "your-key-here"` to secrets
        
        **For local development:**
        Set the environment variable: `export OPENAI_API_KEY="your-key-here"`
        """)
        return
    
    # Initialize system
    if st.session_state.research_system is None:
        with st.spinner("Initializing research system..."):
            st.session_state.research_system = initialize_research_system(api_key)
    
    system = st.session_state.research_system
    
    if system is None:
        st.error("❌ Failed to initialize research system. Please check your API key.")
        return
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    selected_agents = display_agent_selection()
    
    if not selected_agents:
        st.warning("⚠️ Please select at least one research agent.")
        return
    
    # System stats
    st.sidebar.markdown("## 📈 System Stats")
    display_system_stats(system)
    
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
                with st.spinner(f"Researching with {len(selected_agents)} agents..."):
                    # Run research
                    start_time = time.time()
                    
                    # Create event loop for async function
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(
                            run_research(system, query, selected_agents)
                        )
                        loop.close()
                        
                        # Store results
                        st.session_state.current_research = results
                        st.session_state.research_history.append({
                            "query": query,
                            "results": results,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                    except Exception as e:
                        st.error(f"Research failed: {e}")
                        results = {"error": str(e)}
            else:
                st.warning("Please enter a research question.")
    
    # Display current research results
    if st.session_state.current_research:
        st.markdown("---")
        st.markdown("## 📋 Research Results")
        display_research_results(st.session_state.current_research)
    
    # Research history
    if st.session_state.research_history:
        st.markdown("---")
        st.markdown("## 📚 Research History")
        
        for i, session in enumerate(reversed(st.session_state.research_history[-5:])):  # Show last 5
            with st.expander(f"Session {len(st.session_state.research_history) - i}: {session['query'][:60]}..."):
                st.markdown(f"**Timestamp:** {session['timestamp']}")
                st.markdown(f"**Query:** {session['query']}")
                
                if "synthesis" in session["results"]:
                    st.markdown(f"**Synthesis:** {session['results']['synthesis']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🤖 Lightweight AI Research System | Optimized for Cloud Deployment<br>
        <small>Powered by GPT-4o-mini for cost-effective research</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
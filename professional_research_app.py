"""
Professional Self-Initiated Research Agent Interface
Clean, modern, enterprise-grade design without emojis or casual elements.
"""

import streamlit as st
import asyncio
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
import sys
import pandas as pd
import networkx as nx
from io import StringIO

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, current_dir)

# Import the self-initiated research agent
try:
    # Try importing from src directory
    import sys
    sys.path.append(src_path)
    from src.self_initiated_research_agent import SelfInitiatedResearchAgent, ResearchState, ResearchPlan
except ImportError as e:
    try:
        # Fallback: try direct import if module is available
        from self_initiated_research_agent import SelfInitiatedResearchAgent, ResearchState, ResearchPlan
    except ImportError:
        st.error(f"Could not import SelfInitiatedResearchAgent. Error: {e}")
        st.error("Please ensure you're in the correct directory and all dependencies are installed.")
        st.info("Try running the lightweight app instead: `streamlit run streamlit_app.py`")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Research Agent Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Professional CSS
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Remove Streamlit Default Styling */
    .stApp > header {visibility: hidden;}
    .stApp {background: transparent;}
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    #stDecoration {display: none;}
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0f0f23, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating Particles Background Effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.1), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.05), transparent),
            radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.08), transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.03), transparent);
        background-repeat: repeat;
        background-size: 150px 100px;
        animation: particleMove 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-150px, -100px); }
    }
    
    /* Global Text and Container Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        position: relative;
        z-index: 1;
    }
    
    h1, h2, h3, h4, h5, h6, p, div, span, li {
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    
    /* Glass Morphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-5px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.8) !important;
        margin-bottom: 3rem;
    }
    
    /* Modern Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        font-family: 'Inter', sans-serif;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.5);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress Cards */
    .progress-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Knowledge Graph Card */
    .knowledge-graph-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    /* Tutorial Section */
    .tutorial-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .tutorial-step {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-active {
        background: #4ade80;
        box-shadow: 0 0 10px rgba(74, 222, 128, 0.5);
    }
    
    .status-planning {
        background: #facc15;
        box-shadow: 0 0 10px rgba(250, 204, 21, 0.5);
    }
    
    .status-complete {
        background: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Example Query Cards */
    .example-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.3);
        transform: translateX(5px);
    }
    
    /* Hide Streamlit Elements */
    .stAlert, .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for the professional research agent"""
    if "research_agent" not in st.session_state:
        st.session_state.research_agent = None
    if "current_research_session" not in st.session_state:
        st.session_state.current_research_session = None
    if "research_plan" not in st.session_state:
        st.session_state.research_plan = None
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    if "questions_for_user" not in st.session_state:
        st.session_state.questions_for_user = []
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    if "show_tutorial" not in st.session_state:
        st.session_state.show_tutorial = True

def get_api_key() -> str:
    """Get OpenAI API key from secrets or environment"""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY", "")

@st.cache_resource
def initialize_research_agent(api_key: str):
    """Initialize the self-initiated research agent"""
    if not api_key:
        return None
    
    try:
        agent = SelfInitiatedResearchAgent(api_key)
        return agent
    except Exception as e:
        st.error(f"Failed to initialize research agent: {e}")
        return None

def display_tutorial():
    """Display how to use tutorial"""
    st.markdown("""
    <div class="tutorial-section">
        <h2 style="color: #667eea; margin-bottom: 1.5rem;">How to Use the AI Research Agent</h2>
        
        <div class="tutorial-step">
            <h4>Step 1: Enter Research Goal</h4>
            <p>Provide a high-level research question or topic. The more specific, the better the results.</p>
        </div>
        
        <div class="tutorial-step">
            <h4>Step 2: Start Research</h4>
            <p>Click "Start Autonomous Research" and watch the agent create its own research plan.</p>
        </div>
        
        <div class="tutorial-step">
            <h4>Step 3: Monitor Progress</h4>
            <p>Observe real-time knowledge graph building and autonomous gap identification.</p>
        </div>
        
        <div class="tutorial-step">
            <h4>Step 4: Interact</h4>
            <p>Answer any clarifying questions the agent asks to refine research direction.</p>
        </div>
        
        <div class="tutorial-step">
            <h4>Step 5: Generate Report</h4>
            <p>Export comprehensive research findings in professional report format.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_example_queries():
    """Display example research queries"""
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Example Research Queries</h3>
        <p style="margin-bottom: 1.5rem;">Click any example to get started quickly:</p>
    </div>
    """, unsafe_allow_html=True)
    
    examples = [
        "Latest advances in quantum computing for cryptography",
        "AI impact on healthcare diagnostics and treatment",
        "Sustainable energy storage technologies comparison",
        "Blockchain applications beyond cryptocurrency",
        "Machine learning in autonomous vehicle systems",
        "Future of renewable energy grid integration"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{example}", help="Click to use this query"):
            return example
    
    return None

def display_system_status(agent):
    """Display professional system status"""
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #667eea; margin-bottom: 1.5rem;">System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Active</div>
            <div class="metric-label">System</div>
            <div class="status-indicator status-active"></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sessions = len(st.session_state.research_history)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sessions}</div>
            <div class="metric-label">Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        state = "Ready"
        if agent and hasattr(agent, 'state'):
            state = agent.state.value.title()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{state}</div>
            <div class="metric-label">Agent State</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Pro</div>
            <div class="metric-label">Version</div>
        </div>
        """, unsafe_allow_html=True)

def display_research_plan(plan: ResearchPlan):
    """Display the autonomous research plan"""
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Autonomous Research Plan</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="progress-card">
        <h4>Research Goal</h4>
        <p>{plan.goal}</p>
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <span>Created: {plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}</span>
            <span>Budget: {plan.time_budget}s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sub-Goals**")
        for i, sub_goal in enumerate(plan.sub_goals, 1):
            st.markdown(f"{i}. {sub_goal}")
        
        st.markdown("**Research Questions**")
        for i, question in enumerate(plan.research_questions, 1):
            st.markdown(f"{i}. {question}")
    
    with col2:
        st.markdown("**Priority Topics**")
        for i, topic in enumerate(plan.priority_topics, 1):
            st.markdown(f"{i}. {topic}")
        
        st.markdown("**Search Strategies**")
        for i, strategy in enumerate(plan.search_strategies, 1):
            st.markdown(f"{i}. {strategy}")

def visualize_knowledge_graph(agent):
    """Create professional knowledge graph visualization"""
    if not agent or not agent.knowledge_graph.nodes:
        st.markdown("""
        <div class="knowledge-graph-card">
            <h3 style="color: #667eea;">Knowledge Graph</h3>
            <p>Start a research session to see the autonomous knowledge graph build in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="knowledge-graph-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Autonomous Knowledge Graph</h3>
        <p style="margin-bottom: 2rem;">Real-time visualization of agent understanding</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create network graph data
    nodes = list(agent.knowledge_graph.nodes.values())
    
    if not nodes:
        st.info("Knowledge graph is empty. Start research to see it grow!")
        return
    
    # Display graph statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(nodes)}</div>
            <div class="metric-label">Total Nodes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        facts = [n for n in nodes if n.node_type == 'fact']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(facts)}</div>
            <div class="metric-label">Facts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        questions = [n for n in nodes if n.node_type == 'question']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(questions)}</div>
            <div class="metric-label">Questions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        gaps = agent.knowledge_graph.find_knowledge_gaps()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(gaps)}</div>
            <div class="metric-label">Gaps</div>
        </div>
        """, unsafe_allow_html=True)

def display_research_progress(results: Dict[str, Any]):
    """Display professional research progress"""
    if not results:
        return
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Research Progress</h3>
    </div>
    """, unsafe_allow_html=True)
    
    discoveries = results.get('discoveries', [])
    questions_raised = results.get('questions_raised', [])
    gaps_identified = results.get('gaps_identified', [])
    
    # Progress metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="progress-card">
            <div class="metric-value">{len(discoveries)}</div>
            <div class="metric-label">Discoveries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="progress-card">
            <div class="metric-value">{len(questions_raised)}</div>
            <div class="metric-label">Questions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="progress-card">
            <div class="metric-value">{len(gaps_identified)}</div>
            <div class="metric-label">Gaps</div>
        </div>
        """, unsafe_allow_html=True)

async def run_autonomous_research(agent, goal: str):
    """Run the autonomous research process"""
    try:
        # Create research plan
        plan = agent.initiate_research(goal)
        st.session_state.research_plan = plan
        
        # Execute research
        results = await agent.execute_research(max_iterations=5)
        st.session_state.research_results = results
        
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">AI Research Agent Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Autonomous intelligence for comprehensive research analysis</p>', unsafe_allow_html=True)
    
    # API Key Setup
    api_key = get_api_key()
    
    if not api_key:
        st.error("OpenAI API key not found. Please configure your API key in Streamlit secrets.")
        st.info("""
        **Setup Instructions:**
        1. Add `OPENAI_API_KEY = "your-key-here"` to Streamlit secrets
        2. Or set environment variable: `export OPENAI_API_KEY="your-key-here"`
        """)
        return
    
    # Initialize agent
    if st.session_state.research_agent is None:
        with st.spinner("Initializing AI Research Agent..."):
            st.session_state.research_agent = initialize_research_agent(api_key)
    
    agent = st.session_state.research_agent
    
    if agent is None:
        st.error("Failed to initialize research agent. Please check your API key.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #667eea;">Agent Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        display_system_status(agent)
        
        if st.button("Show Tutorial", help="Learn how to use the research agent"):
            st.session_state.show_tutorial = not st.session_state.show_tutorial
        
        if st.button("Clear Session", help="Start a new research session"):
            st.session_state.research_plan = None
            st.session_state.research_results = None
            st.session_state.questions_for_user = []
    
    # Main interface
    if st.session_state.show_tutorial:
        display_tutorial()
    
    # Example queries
    selected_example = display_example_queries()
    
    # Research input
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Research Query</h3>
    </div>
    """, unsafe_allow_html=True)
    
    goal = st.text_area(
        "Enter your research goal:",
        value=selected_example if selected_example else "",
        placeholder="e.g., Latest advances in quantum computing applications",
        height=100,
        help="Provide a specific research question or topic for autonomous investigation"
    )
    
    # Research button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("Start Autonomous Research", type="primary", use_container_width=True):
            if goal.strip():
                with st.spinner("AI agent is planning and executing research..."):
                    start_time = time.time()
                    
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(
                            run_autonomous_research(agent, goal)
                        )
                        loop.close()
                        
                        duration = time.time() - start_time
                        
                        if "error" not in results:
                            st.success(f"Research completed in {duration:.1f} seconds!")
                        else:
                            st.error(f"Research failed: {results['error']}")
                        
                    except Exception as e:
                        st.error(f"Research failed: {e}")
            else:
                st.warning("Please enter a research goal.")
    
    # Display results
    if st.session_state.research_plan:
        display_research_plan(st.session_state.research_plan)
    
    if st.session_state.research_results:
        display_research_progress(st.session_state.research_results)
        
        # Display synthesis
        synthesis = st.session_state.research_results.get('synthesis')
        if synthesis:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #667eea; margin-bottom: 1rem;">Research Synthesis</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(synthesis)
    
    # Knowledge graph
    visualize_knowledge_graph(agent)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; opacity: 0.6;">
        <p>AI Research Agent Pro | Autonomous Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
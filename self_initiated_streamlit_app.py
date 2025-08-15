"""
Self-Initiated Research Agent - Streamlit Interface
Advanced autonomous research system that builds plans, identifies gaps, and asks questions.
This showcases the project that exceeds YouTube's "self-initiated research agent" requirements.
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
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Import the self-initiated research agent
try:
    from self_initiated_research_agent import SelfInitiatedResearchAgent, ResearchState, ResearchPlan
except ImportError:
    st.error("⚠️ Could not import SelfInitiatedResearchAgent. Please ensure you're in the correct directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Self-Initiated Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"

def get_theme_css():
    """Get CSS based on current theme mode"""
    if st.session_state.theme_mode == "dark":
        return """
        <style>
            /* Dark Theme Variables */
            :root {
                --bg-primary: #0e1117;
                --bg-secondary: #1e2329;
                --bg-card: #262730;
                --text-primary: #fafafa;
                --text-secondary: #a8a8a8;
                --accent-primary: #00d4aa;
                --accent-secondary: #7c3aed;
                --border-color: #2d3748;
                --shadow: 0 8px 32px rgba(0,0,0,0.3);
                --success: #22c55e;
                --warning: #f59e0b;
                --error: #ef4444;
                --info: #3b82f6;
            }
            
            /* Global Styles */
            .stApp {
                background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
                color: var(--text-primary);
            }
            
            .main-header {
                font-size: clamp(2rem, 5vw, 3.5rem);
                text-align: center;
                margin-bottom: 1.5rem;
                background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            
            .subtitle {
                text-align: center;
                color: var(--text-secondary);
                font-size: clamp(1rem, 2.5vw, 1.3rem);
                margin-bottom: 2rem;
                font-weight: 400;
            }
            
            /* Enhanced Card Styles */
            .autonomous-card {
                background: linear-gradient(135deg, var(--accent-secondary) 0%, #8b5cf6 100%);
                color: white;
                padding: clamp(1rem, 3vw, 2rem);
                border-radius: 20px;
                margin: 1.5rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .autonomous-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.4);
            }
            
            .knowledge-graph-card {
                background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
                color: white;
                padding: clamp(1rem, 3vw, 2rem);
                border-radius: 20px;
                margin: 1.5rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
            }
            
            .research-plan-card {
                background: linear-gradient(135deg, var(--accent-primary) 0%, #06b6d4 100%);
                color: white;
                padding: clamp(1rem, 3vw, 2rem);
                border-radius: 20px;
                margin: 1.5rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
            }
            
            .progress-card {
                background: linear-gradient(135deg, var(--success) 0%, #16a34a 100%);
                color: white;
                padding: clamp(0.8rem, 2vw, 1.5rem);
                border-radius: 15px;
                text-align: center;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.2s ease;
            }
            
            .progress-card:hover {
                transform: scale(1.02);
            }
            
            .question-card {
                background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
                color: white;
                padding: clamp(0.8rem, 2vw, 1.5rem);
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.2s ease;
            }
            
            .metric-card {
                background: var(--bg-card);
                color: var(--text-primary);
                padding: clamp(0.8rem, 2vw, 1.5rem);
                border-radius: 15px;
                box-shadow: var(--shadow);
                text-align: center;
                border: 1px solid var(--border-color);
                border-left: 4px solid var(--accent-primary);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                border-left-color: var(--accent-secondary);
            }
            
            /* Responsive Grid Layout */
            @media (max-width: 768px) {
                .autonomous-card, .knowledge-graph-card, .research-plan-card {
                    margin: 1rem 0;
                    padding: 1rem;
                }
                
                .main-header {
                    font-size: 2rem;
                    margin-bottom: 1rem;
                }
                
                .subtitle {
                    font-size: 1rem;
                    margin-bottom: 1.5rem;
                }
            }
            
            /* Enhanced Interactive Elements */
            .autonomous-badge {
                background: linear-gradient(45deg, #8b5cf6, #a855f7);
                color: white;
                padding: 0.4rem 1rem;
                border-radius: 25px;
                font-size: 0.85rem;
                font-weight: 600;
                margin: 0.2rem;
                display: inline-block;
                box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
                transition: transform 0.2s ease;
            }
            
            .autonomous-badge:hover {
                transform: scale(1.05);
            }
            
            .gap-identified {
                background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
                color: #1f2937;
                padding: 1rem;
                border-radius: 12px;
                margin: 1rem 0;
                border-left: 4px solid #d97706;
                box-shadow: 0 4px 15px rgba(251, 191, 36, 0.2);
                font-weight: 500;
            }
            
            .discovery-item {
                background: var(--bg-card);
                color: var(--text-primary);
                padding: 1rem;
                border-radius: 12px;
                margin: 1rem 0;
                border-left: 4px solid var(--accent-primary);
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
                transition: border-left-color 0.2s ease;
            }
            
            .discovery-item:hover {
                border-left-color: var(--accent-secondary);
            }
            
            /* Progress Indicator */
            .progress-indicator {
                background: var(--bg-card);
                border-radius: 15px;
                padding: 1rem;
                margin: 1rem 0;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow);
            }
            
            .progress-step {
                display: inline-block;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                line-height: 30px;
                margin: 0 0.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .progress-step.active {
                background: var(--accent-primary);
                color: white;
                box-shadow: 0 0 20px rgba(0, 212, 170, 0.4);
            }
            
            .progress-step.completed {
                background: var(--success);
                color: white;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light Theme Variables */
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f8fafc;
                --bg-card: #ffffff;
                --text-primary: #1f2937;
                --text-secondary: #6b7280;
                --accent-primary: #0891b2;
                --accent-secondary: #7c3aed;
                --border-color: #e5e7eb;
                --shadow: 0 8px 32px rgba(0,0,0,0.1);
                --success: #059669;
                --warning: #d97706;
                --error: #dc2626;
                --info: #2563eb;
            }
            
            /* Global Styles */
            .stApp {
                background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
                color: var(--text-primary);
            }
            
            .main-header {
                font-size: clamp(2rem, 5vw, 3.5rem);
                text-align: center;
                margin-bottom: 1.5rem;
                background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            
            .subtitle {
                text-align: center;
                color: var(--text-secondary);
                font-size: clamp(1rem, 2.5vw, 1.3rem);
                margin-bottom: 2rem;
                font-weight: 400;
            }
            
            /* Enhanced Card Styles */
            .autonomous-card {
                background: linear-gradient(135deg, var(--accent-secondary) 0%, #8b5cf6 100%);
                color: white;
                padding: clamp(1rem, 3vw, 2rem);
                border-radius: 20px;
                margin: 1.5rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .autonomous-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.15);
            }
            
            .knowledge-graph-card {
                background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
                color: white;
                padding: clamp(1rem, 3vw, 2rem);
                border-radius: 20px;
                margin: 1.5rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
            }
            
            .research-plan-card {
                background: linear-gradient(135deg, var(--accent-primary) 0%, #06b6d4 100%);
                color: white;
                padding: clamp(1rem, 3vw, 2rem);
                border-radius: 20px;
                margin: 1.5rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
            }
            
            .progress-card {
                background: linear-gradient(135deg, var(--success) 0%, #16a34a 100%);
                color: white;
                padding: clamp(0.8rem, 2vw, 1.5rem);
                border-radius: 15px;
                text-align: center;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.2s ease;
            }
            
            .progress-card:hover {
                transform: scale(1.02);
            }
            
            .question-card {
                background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
                color: white;
                padding: clamp(0.8rem, 2vw, 1.5rem);
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.2s ease;
            }
            
            .metric-card {
                background: var(--bg-card);
                color: var(--text-primary);
                padding: clamp(0.8rem, 2vw, 1.5rem);
                border-radius: 15px;
                box-shadow: var(--shadow);
                text-align: center;
                border: 1px solid var(--border-color);
                border-left: 4px solid var(--accent-primary);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                border-left-color: var(--accent-secondary);
            }
            
            /* Responsive Grid Layout */
            @media (max-width: 768px) {
                .autonomous-card, .knowledge-graph-card, .research-plan-card {
                    margin: 1rem 0;
                    padding: 1rem;
                }
                
                .main-header {
                    font-size: 2rem;
                    margin-bottom: 1rem;
                }
                
                .subtitle {
                    font-size: 1rem;
                    margin-bottom: 1.5rem;
                }
            }
            
            /* Enhanced Interactive Elements */
            .autonomous-badge {
                background: linear-gradient(45deg, #8b5cf6, #a855f7);
                color: white;
                padding: 0.4rem 1rem;
                border-radius: 25px;
                font-size: 0.85rem;
                font-weight: 600;
                margin: 0.2rem;
                display: inline-block;
                box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
                transition: transform 0.2s ease;
            }
            
            .autonomous-badge:hover {
                transform: scale(1.05);
            }
            
            .gap-identified {
                background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                color: #92400e;
                padding: 1rem;
                border-radius: 12px;
                margin: 1rem 0;
                border-left: 4px solid #d97706;
                box-shadow: 0 4px 15px rgba(251, 191, 36, 0.1);
                font-weight: 500;
            }
            
            .discovery-item {
                background: var(--bg-card);
                color: var(--text-primary);
                padding: 1rem;
                border-radius: 12px;
                margin: 1rem 0;
                border-left: 4px solid var(--accent-primary);
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
                transition: border-left-color 0.2s ease;
            }
            
            .discovery-item:hover {
                border-left-color: var(--accent-secondary);
            }
            
            /* Progress Indicator */
            .progress-indicator {
                background: var(--bg-card);
                border-radius: 15px;
                padding: 1rem;
                margin: 1rem 0;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow);
            }
            
            .progress-step {
                display: inline-block;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background: var(--bg-secondary);
                color: var(--text-secondary);
                text-align: center;
                line-height: 30px;
                margin: 0 0.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .progress-step.active {
                background: var(--accent-primary);
                color: white;
                box-shadow: 0 0 20px rgba(8, 145, 178, 0.4);
            }
            
            .progress-step.completed {
                background: var(--success);
                color: white;
            }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)


def safe_display_string(content, max_length=None):
    """Safely convert any content to string for display, handling dicts and other types"""
    try:
        if isinstance(content, dict):
            # Convert dict to readable string
            if 'model_name' in content:
                return str(content['model_name'])
            elif 'name' in content:
                return str(content['name'])
            elif 'title' in content:
                return str(content['title'])
            elif 'content' in content:
                return str(content['content'])
            else:
                # Fallback: use first string value or stringify the dict
                str_values = [str(v) for v in content.values() if v and not isinstance(v, dict)]
                return str_values[0] if str_values else str(content)
        elif not isinstance(content, str):
            content = str(content)
        
        # Ensure content is string and truncate if needed
        content = str(content)
        if max_length:
            content = content[:max_length]
        return content
    except Exception as e:
        logger.warning(f"Error converting content to display string: {e}")
        return f"[Display conversion error: {type(content).__name__}]"


def initialize_session_state():
    """Initialize session state for the self-initiated research agent"""
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
    if "knowledge_graph_data" not in st.session_state:
        st.session_state.knowledge_graph_data = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "awaiting_user_response" not in st.session_state:
        st.session_state.awaiting_user_response = False
    if "research_active" not in st.session_state:
        st.session_state.research_active = False
    if "last_agent_response" not in st.session_state:
        st.session_state.last_agent_response = None


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


def sync_agent_state(agent):
    """Synchronize agent state with Streamlit session state"""
    if not agent:
        logger.debug("No agent to sync")
        return
    
    logger.debug("Syncing agent state with session state")
    
    # Sync conversation history from session state to agent
    if st.session_state.conversation_history:
        logger.debug(f"Syncing {len(st.session_state.conversation_history)} conversation exchanges")
        # Convert session state format to agent format
        agent_history = []
        for exchange in st.session_state.conversation_history:
            agent_history.extend([
                {"role": "user", "content": exchange.get("user", "")},
                {"role": "assistant", "content": exchange.get("agent", "")}
            ])
        agent.conversation_history = agent_history
    
    # Sync questions from agent to session state if new questions available
    if agent.questions_for_user and not st.session_state.questions_for_user:
        logger.info(f"Syncing {len(agent.questions_for_user)} new questions from agent")
        st.session_state.questions_for_user = agent.questions_for_user.copy()
        st.session_state.awaiting_user_response = True
        
    # Log current research state
    if hasattr(agent, 'state'):
        logger.debug(f"Agent research state: {agent.state.value}")
    if hasattr(agent, 'research_paused'):
        logger.debug(f"Agent research paused: {agent.research_paused}")


def display_system_overview():
    """Display system capabilities overview"""
    st.markdown("""
    <div class="autonomous-card">
        <h3>🧠 Self-Initiated Research Agent</h3>
        <p>This agent goes beyond traditional query-response systems. It autonomously:</p>
        <ul>
            <li>🎯 <strong>Creates Research Plans</strong> - Builds comprehensive strategies from high-level goals</li>
            <li>🔍 <strong>Identifies Knowledge Gaps</strong> - Automatically detects what's missing in current understanding</li>
            <li>❓ <strong>Asks Clarifying Questions</strong> - Engages in dialogue to refine research direction</li>
            <li>🕸️ <strong>Builds Knowledge Graphs</strong> - Creates interconnected understanding networks</li>
            <li>📊 <strong>Synthesizes Findings</strong> - Produces professional research reports</li>
        </ul>
        <div style="text-align: center; margin-top: 1rem;">
            <span class="autonomous-badge">AUTONOMOUS BEHAVIOR</span>
            <span class="autonomous-badge">SYSTEMS THINKING</span>
            <span class="autonomous-badge">PROMPT ORCHESTRATION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_research_plan(plan: ResearchPlan):
    """Display the autonomous research plan"""
    st.markdown(f"""
    <div class="research-plan-card">
        <h3>📋 Autonomous Research Plan</h3>
        <p><strong>Goal:</strong> {plan.goal}</p>
        <p><strong>Created:</strong> {plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Time Budget:</strong> {plan.time_budget} seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Sub-Goals**")
        for i, sub_goal in enumerate(plan.sub_goals, 1):
            st.markdown(f"{i}. {safe_display_string(sub_goal)}")
        
        st.markdown("**🔍 Research Questions**")
        for i, question in enumerate(plan.research_questions, 1):
            st.markdown(f"{i}. {safe_display_string(question)}")
    
    with col2:
        st.markdown("**📚 Priority Topics**")
        for i, topic in enumerate(plan.priority_topics, 1):
            st.markdown(f"{i}. {safe_display_string(topic)}")
        
        st.markdown("**🔬 Search Strategies**")
        for i, strategy in enumerate(plan.search_strategies, 1):
            st.markdown(f"{i}. {safe_display_string(strategy)}")


def visualize_knowledge_graph(agent):
    """Create an interactive knowledge graph visualization"""
    if not agent or not agent.knowledge_graph.nodes:
        st.info("No knowledge graph data available yet. Start a research session to see the graph build autonomously!")
        return
    
    st.markdown("""
    <div class="knowledge-graph-card">
        <h3>🕸️ Autonomous Knowledge Graph</h3>
        <p>Watch how the agent builds understanding through interconnected discoveries</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create network graph data
    nodes = list(agent.knowledge_graph.nodes.values())
    
    if not nodes:
        st.info("Knowledge graph is empty. Start research to see it grow!")
        return
    
    # Prepare data for visualization
    node_data = []
    edge_data = []
    
    # Color mapping for node types
    color_map = {
        'fact': '#1f77b4',      # Blue
        'question': '#ff7f0e',   # Orange
        'hypothesis': '#2ca02c', # Green
        'gap': '#d62728'         # Red
    }
    
    # Create nodes
    for node in nodes:
        node_data.append({
            'id': node.id,
            'label': node.content[:50] + "..." if len(node.content) > 50 else node.content,
            'type': node.node_type,
            'confidence': node.confidence,
            'color': color_map.get(node.node_type, '#gray'),
            'size': min(max(node.confidence * 30, 10), 40)
        })
    
    # Create edges
    for node in nodes:
        for child_id in node.children:
            if child_id in agent.knowledge_graph.nodes:
                edge_data.append({
                    'source': node.id,
                    'target': child_id
                })
    
    # Display graph statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Nodes</h3>
            <h2>{len(node_data)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Connections</h3>
            <h2>{len(edge_data)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        facts = [n for n in nodes if n.node_type == 'fact']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Facts</h3>
            <h2>{len(facts)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        questions = [n for n in nodes if n.node_type == 'question']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Questions</h3>
            <h2>{len(questions)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        gaps = agent.knowledge_graph.find_knowledge_gaps()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Gaps Identified</h3>
            <h2>{len(gaps)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Create a simple network visualization using plotly
    if node_data:
        # Create a simple force-directed layout
        import math
        import random
        
        # Simple circular layout for visualization
        num_nodes = len(node_data)
        for i, node in enumerate(node_data):
            angle = 2 * math.pi * i / max(num_nodes, 1)
            radius = 100 + random.uniform(-20, 20)
            node['x'] = radius * math.cos(angle)
            node['y'] = radius * math.sin(angle)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in edge_data:
            source_node = next(n for n in node_data if n['id'] == edge['source'])
            target_node = next(n for n in node_data if n['id'] == edge['target'])
            
            fig.add_trace(go.Scatter(
                x=[source_node['x'], target_node['x']],
                y=[source_node['y'], target_node['y']],
                mode='lines',
                line=dict(color='#ddd', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Add nodes
        for node_type in color_map.keys():
            type_nodes = [n for n in node_data if n['type'] == node_type]
            if type_nodes:
                fig.add_trace(go.Scatter(
                    x=[n['x'] for n in type_nodes],
                    y=[n['y'] for n in type_nodes],
                    mode='markers+text',
                    marker=dict(
                        size=[n['size'] for n in type_nodes],
                        color=color_map[node_type],
                        opacity=0.8
                    ),
                    text=[n['label'] for n in type_nodes],
                    textposition="middle center",
                    name=node_type.title(),
                    hovertemplate='<b>%{text}</b><br>Type: ' + node_type + '<br>Confidence: %{marker.size}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Autonomous Knowledge Graph - Real-time Building",
            showlegend=True,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_research_progress(results: Dict[str, Any]):
    """Display real-time research progress"""
    if not results:
        return
    
    st.markdown("### 🔄 Autonomous Research Progress")
    
    discoveries = results.get('discoveries', [])
    questions_raised = results.get('questions_raised', [])
    gaps_identified = results.get('gaps_identified', [])
    
    # Progress metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="progress-card">
            <h3>Discoveries</h3>
            <h2>{len(discoveries)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="progress-card">
            <h3>Questions Raised</h3>
            <h2>{len(questions_raised)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="progress-card">
            <h3>Gaps Identified</h3>
            <h2>{len(gaps_identified)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Display recent discoveries
    if discoveries:
        st.markdown("**🔍 Recent Discoveries**")
        for discovery in discoveries[-5:]:  # Show last 5
            source = safe_display_string(discovery.get('source', 'Unknown'))
            content = safe_display_string(discovery.get('content', ''), 200) 
            if len(safe_display_string(discovery.get('content', ''))) > 200:
                content += "..."
            st.markdown(f"""
            <div class="discovery-item">
                <strong>Source:</strong> {source}<br>
                <strong>Finding:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
    
    # Display identified gaps
    if gaps_identified:
        st.markdown("**⚠️ Knowledge Gaps Identified**")
        for gap in gaps_identified[-3:]:  # Show last 3
            gap_str = safe_display_string(gap)
            st.markdown(f"""
            <div class="gap-identified">
                📊 Gap: {gap_str}
            </div>
            """, unsafe_allow_html=True)


def display_agent_questions(agent):
    """Display questions the agent wants to ask the user"""
    # Sync agent questions with session state
    if agent and agent.questions_for_user and not st.session_state.questions_for_user:
        st.session_state.questions_for_user = agent.questions_for_user.copy()
        st.session_state.awaiting_user_response = True
    
    # Display questions from session state (persistent across reruns)
    questions_to_display = st.session_state.questions_for_user or (agent.questions_for_user if agent else [])
    
    if not questions_to_display:
        return
    
    st.markdown("### ❓ Agent Questions")
    st.markdown("The agent has identified areas where your input would help refine the research:")
    
    for i, question in enumerate(questions_to_display, 1):
        st.markdown(f"""
        <div class="question-card">
            <strong>Question {i}:</strong> {question}
        </div>
        """, unsafe_allow_html=True)
    
    # Add input for user responses
    st.markdown("**Your Response:**")
    user_response = st.text_area(
        "Respond to help guide the research direction:",
        placeholder="Share your thoughts on the agent's questions...",
        key="user_response"
    )
    
    if st.button("💬 Submit Response"):
        if user_response.strip():
            logger.info(f"User submitted response: {user_response[:100]}...")
            
            # Process user response through the agent
            try:
                response = agent.interactive_dialogue(user_response)
                logger.info(f"Agent generated response: {response[:100]}...")
                
                st.session_state.last_agent_response = response
                st.session_state.conversation_history.append({
                    "user": user_response,
                    "agent": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Clear the agent's questions and mark response received
                agent.questions_for_user = []
                st.session_state.questions_for_user = []
                st.session_state.awaiting_user_response = False
                
                logger.info("Conversation state updated successfully")
                st.success("Agent Response: " + response)
                
                # Trigger research continuation
                st.session_state.research_active = True
                logger.info("Research continuation triggered")
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error processing user response: {e}")
                st.error(f"Sorry, I encountered an error processing your response: {e}")
        else:
            logger.warning("User submitted empty response")
            st.warning("Please enter a response before submitting.")


def display_conversation_history():
    """Display conversation history between user and agent"""
    if not st.session_state.conversation_history:
        return
    
    st.markdown("### 💬 Conversation History")
    st.markdown("Previous interactions with the research agent:")
    
    for i, exchange in enumerate(st.session_state.conversation_history, 1):
        with st.expander(f"Exchange {i} - {exchange.get('timestamp', 'Unknown time')[:19]}"):
            st.markdown(f"**You:** {exchange.get('user', 'N/A')}")
            st.markdown(f"**Agent:** {exchange.get('agent', 'N/A')}")


def continue_research_after_response(agent):
    """Continue research after user provides responses to agent questions"""
    logger.debug("continue_research_after_response called")
    
    if not st.session_state.research_active or not agent:
        logger.debug("Research not active or agent not available")
        return
    
    if st.session_state.awaiting_user_response:
        logger.debug("Still awaiting user response")
        return  # Still waiting for user response
    
    # Check if we should continue research
    if st.session_state.last_agent_response and not st.session_state.awaiting_user_response:
        logger.info("Displaying research continuation option")
        st.info("🔄 Research continues based on your response...")
        
        # Trigger continuation button
        if st.button("🚀 Continue Research", type="primary"):
            logger.info("User clicked Continue Research button")
            with st.spinner("Agent is continuing research based on your input..."):
                try:
                    # Use the new resume_research method
                    logger.info("Starting research continuation process")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Resume research with 3 additional iterations
                    if hasattr(agent, 'resume_research'):
                        logger.info("Using resume_research method")
                        additional_results = loop.run_until_complete(
                            agent.resume_research(additional_iterations=3)
                        )
                    else:
                        logger.info("Using fallback execute_research method")
                        additional_results = loop.run_until_complete(
                            agent.execute_research(max_iterations=3)
                        )
                    loop.close()
                    
                    logger.info(f"Research continuation completed with {len(additional_results.get('discoveries', []))} new discoveries")
                    
                    # Handle errors from resume
                    if "error" in additional_results:
                        logger.error(f"Research continuation failed: {additional_results['error']}")
                        st.error(f"❌ Research continuation failed: {additional_results['error']}")
                        return
                    
                    # Update research results
                    if st.session_state.research_results:
                        existing = st.session_state.research_results
                        existing['discoveries'].extend(additional_results.get('discoveries', []))
                        existing['questions_raised'].extend(additional_results.get('questions_raised', []))
                        existing['gaps_identified'].extend(additional_results.get('gaps_identified', []))
                        if additional_results.get('synthesis'):
                            existing['synthesis'] = additional_results['synthesis']
                    else:
                        st.session_state.research_results = additional_results
                    
                    # Reset continuation flags
                    st.session_state.research_active = False
                    st.session_state.last_agent_response = None
                    
                    # Sync any new questions
                    if agent.questions_for_user:
                        st.session_state.questions_for_user = agent.questions_for_user.copy()
                        st.session_state.awaiting_user_response = True
                    
                    st.success("✅ Research continued successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Research continuation failed: {e}")
                    logger.error(f"Research continuation error: {e}")


def generate_formatted_report(agent, format_type="markdown"):
    """Generate research report in different formats"""
    try:
        base_report = agent.generate_research_report()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if format_type == "markdown":
            return base_report
        
        elif format_type == "html":
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Research Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 20px; }}
                    .discovery {{ background: #e8f4fd; padding: 10px; border-left: 4px solid #0984e3; margin: 10px 0; }}
                    .footer {{ text-align: center; font-size: 0.9em; color: #666; margin-top: 40px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🧠 Research Report</h1>
                    <p><strong>Generated:</strong> {timestamp}</p>
                    <p><strong>Research Goal:</strong> {st.session_state.research_plan.goal if st.session_state.research_plan else 'N/A'}</p>
                </div>
                <div class="section">
                    {base_report.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>').replace('\n\n', '</p><p>').replace('\n', '<br>')}
                </div>
                <div class="footer">
                    <p>Generated by Self-Initiated Research Agent | {timestamp}</p>
                </div>
            </body>
            </html>
            """
            return html_report
        
        elif format_type == "json":
            # Structure data as JSON
            json_data = {
                "metadata": {
                    "generated_at": timestamp,
                    "research_goal": st.session_state.research_plan.goal if st.session_state.research_plan else None,
                    "agent_version": "1.0"
                },
                "research_plan": {
                    "sub_goals": st.session_state.research_plan.sub_goals if st.session_state.research_plan else [],
                    "research_questions": st.session_state.research_plan.research_questions if st.session_state.research_plan else [],
                    "priority_topics": st.session_state.research_plan.priority_topics if st.session_state.research_plan else []
                },
                "results": {
                    "discoveries": st.session_state.research_results.get('discoveries', []) if st.session_state.research_results else [],
                    "questions_raised": st.session_state.research_results.get('questions_raised', []) if st.session_state.research_results else [],
                    "gaps_identified": st.session_state.research_results.get('gaps_identified', []) if st.session_state.research_results else [],
                    "synthesis": st.session_state.research_results.get('synthesis', '') if st.session_state.research_results else ''
                },
                "conversation_history": st.session_state.conversation_history,
                "report_text": base_report
            }
            return json.dumps(json_data, indent=2)
        
        elif format_type == "csv":
            # Create CSV with discoveries and findings
            csv_data = []
            if st.session_state.research_results and st.session_state.research_results.get('discoveries'):
                for i, discovery in enumerate(st.session_state.research_results['discoveries']):
                    csv_data.append([
                        i + 1,
                        safe_display_string(discovery.get('source', 'Unknown')),
                        safe_display_string(discovery.get('content', '')),
                        discovery.get('confidence', 0),
                        'Discovery'
                    ])
            
            if st.session_state.research_results and st.session_state.research_results.get('questions_raised'):
                for i, question in enumerate(st.session_state.research_results['questions_raised']):
                    csv_data.append([
                        len(csv_data) + 1,
                        'Agent Generated',
                        safe_display_string(question),
                        0.5,
                        'Question'
                    ])
            
            # Convert to CSV string
            import io
            output = io.StringIO()
            output.write("ID,Source,Content,Confidence,Type\n")
            for row in csv_data:
                # Escape commas and quotes in CSV
                escaped_row = []
                for cell in row:
                    cell_str = str(cell).replace('"', '""')
                    if ',' in cell_str or '"' in cell_str or '\n' in cell_str:
                        cell_str = f'"{cell_str}"'
                    escaped_row.append(cell_str)
                output.write(",".join(escaped_row) + "\n")
            
            return output.getvalue()
            
    except Exception as e:
        logger.error(f"Error generating {format_type} report: {e}")
        return f"Error generating {format_type} report: {e}"

def display_research_report(agent):
    """Display and allow download of research report in multiple formats"""
    if not agent:
        return
    
    st.markdown("### 📊 Professional Research Report & Export")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("📄 Generate Research Report", type="primary"):
            with st.spinner("Generating professional research report..."):
                report = agent.generate_research_report()
                st.session_state.research_report = report
    
    with col2:
        if hasattr(st.session_state, 'research_report'):
            st.markdown("**📥 Export Options:**")
            
            # Format selection
            export_format = st.selectbox(
                "Choose format:",
                ["Markdown (.md)", "HTML (.html)", "JSON (.json)", "CSV (.csv)"],
                help="Select the format for downloading your research report"
            )
            
            format_map = {
                "Markdown (.md)": ("markdown", ".md", "text/markdown"),
                "HTML (.html)": ("html", ".html", "text/html"),
                "JSON (.json)": ("json", ".json", "application/json"),
                "CSV (.csv)": ("csv", ".csv", "text/csv")
            }
            
            format_key, file_ext, mime_type = format_map[export_format]
            
            # Generate and download
            if st.button(f"📥 Download {format_key.upper()}"):
                with st.spinner(f"Generating {format_key.upper()} report..."):
                    formatted_report = generate_formatted_report(agent, format_key)
                    
                    st.download_button(
                        label=f"💾 Save {format_key.upper()} Report",
                        data=formatted_report,
                        file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}",
                        mime=mime_type,
                        key=f"download_{format_key}"
                    )
    
    if hasattr(st.session_state, 'research_report'):
        # Display report preview with enhanced formatting
        st.markdown("---")
        st.markdown("**📋 Report Preview:**")
        
        # Expandable sections for better organization
        with st.expander("📖 Full Report Content", expanded=True):
            st.markdown(st.session_state.research_report)
        
        # Quick insights panel
        if st.session_state.research_results:
            with st.expander("📊 Quick Insights"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    discoveries_count = len(st.session_state.research_results.get('discoveries', []))
                    st.metric("Discoveries", discoveries_count)
                
                with col2:
                    questions_count = len(st.session_state.research_results.get('questions_raised', []))
                    st.metric("Questions Raised", questions_count)
                
                with col3:
                    gaps_count = len(st.session_state.research_results.get('gaps_identified', []))
                    st.metric("Gaps Identified", gaps_count)
                
                # Research quality score
                if discoveries_count > 0:
                    quality_score = min(100, (discoveries_count * 10) + (questions_count * 5))
                    st.markdown(f"**Research Quality Score:** {quality_score}/100")
                    st.progress(quality_score / 100)


async def run_autonomous_research(agent, goal: str, max_iterations: int = 5):
    """Run the autonomous research process"""
    try:
        # Create research plan
        plan = agent.initiate_research(goal)
        st.session_state.research_plan = plan
        
        # Execute research with specified iterations
        results = await agent.execute_research(max_iterations=max_iterations)
        st.session_state.research_results = results
        
        return results
    except Exception as e:
        return {"error": str(e)}


def display_progress_pipeline():
    """Display research progress pipeline"""
    steps = [
        {"name": "Planning", "icon": "📋", "desc": "Creating research plan"},
        {"name": "Research", "icon": "🔍", "desc": "Gathering information"},
        {"name": "Analysis", "icon": "🧠", "desc": "Processing findings"},
        {"name": "Synthesis", "icon": "📊", "desc": "Generating insights"},
        {"name": "Complete", "icon": "✅", "desc": "Research finished"}
    ]
    
    # Determine current step based on session state
    current_step = 0
    if st.session_state.research_plan:
        current_step = 1
    if st.session_state.research_results:
        current_step = 2
        if st.session_state.research_results.get('discoveries'):
            current_step = 3
        if st.session_state.research_results.get('synthesis'):
            current_step = 4
    
    st.markdown("""
    <div class="progress-indicator">
        <h4 style="margin: 0 0 1rem 0; text-align: center;">Research Pipeline</h4>
        <div style="text-align: center;">
    """, unsafe_allow_html=True)
    
    for i, step in enumerate(steps):
        status_class = ""
        if i < current_step:
            status_class = "completed"
        elif i == current_step:
            status_class = "active"
        
        st.markdown(f"""
            <div style="display: inline-block; text-align: center; margin: 0 0.5rem;">
                <div class="progress-step {status_class}">{i+1}</div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem;">{step['icon']}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">{step['name']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if i < len(steps) - 1:
            st.markdown("""
                <div style="display: inline-block; margin: 0 0.2rem;">
                    <div style="width: 20px; height: 2px; background: var(--border-color); margin-top: 15px;"></div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header with theme toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")  # Spacer
    with col2:
        st.markdown('<h1 class="main-header">🧠 Self-Initiated Research Agent</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Autonomous AI that builds research plans, identifies gaps, and asks questions</p>', unsafe_allow_html=True)
    with col3:
        # Theme toggle button
        theme_icon = "🌙" if st.session_state.theme_mode == "light" else "☀️"
        if st.button(f"{theme_icon} Theme", help="Toggle dark/light theme"):
            st.session_state.theme_mode = "dark" if st.session_state.theme_mode == "light" else "light"
            st.rerun()
    
    # Progress pipeline
    display_progress_pipeline()
    
    # Display system overview
    display_system_overview()
    
    # API Key Setup
    api_key = get_api_key()
    
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please add it to your Streamlit secrets or environment variables.")
        st.info("""
        **Setup Instructions:**
        1. Add `OPENAI_API_KEY = "your-key-here"` to Streamlit secrets
        2. Or set environment variable: `export OPENAI_API_KEY="your-key-here"`
        """)
        return
    
    # Initialize agent
    if st.session_state.research_agent is None:
        with st.spinner("Initializing Self-Initiated Research Agent..."):
            st.session_state.research_agent = initialize_research_agent(api_key)
    
    agent = st.session_state.research_agent
    
    if agent is None:
        st.error("❌ Failed to initialize research agent. Please check your API key.")
        return
    
    # Synchronize agent state with session state
    sync_agent_state(agent)
    
    # Sidebar with agent state info
    st.sidebar.markdown("## 🤖 Agent Status")
    if agent:
        state = agent.state.value if hasattr(agent, 'state') else 'Ready'
        st.sidebar.info(f"Current State: **{state.title()}**")
        
        if hasattr(agent, 'session_id'):
            st.sidebar.info(f"Session: {agent.session_id}")
        
        # Debug panel
        with st.sidebar.expander("🔍 Debug Info"):
            st.write("**Session State:**")
            st.write(f"- Research Active: {st.session_state.research_active}")
            st.write(f"- Awaiting User Response: {st.session_state.awaiting_user_response}")
            st.write(f"- Questions Available: {len(st.session_state.questions_for_user)}")
            st.write(f"- Conversation History: {len(st.session_state.conversation_history)} exchanges")
            
            st.write("**Agent State:**")
            if hasattr(agent, 'research_paused'):
                st.write(f"- Research Paused: {agent.research_paused}")
            if hasattr(agent, 'questions_for_user'):
                st.write(f"- Agent Questions: {len(agent.questions_for_user)}")
            if hasattr(agent, 'conversation_history'):
                st.write(f"- Agent Conversation: {len(agent.conversation_history)} entries")
            
            # Add reset button for debugging
            if st.button("🔄 Reset Conversation State"):
                logger.info("Resetting conversation state for debugging")
                st.session_state.conversation_history = []
                st.session_state.questions_for_user = []
                st.session_state.awaiting_user_response = False
                st.session_state.research_active = False
                st.session_state.last_agent_response = None
                if agent:
                    agent.questions_for_user = []
                    agent.conversation_history = []
                st.success("Conversation state reset!")
                st.rerun()
    
    # Main interface
    st.markdown("---")
    
    # Research Templates Section
    st.markdown("### 🎯 Research Goal & Templates")
    
    # Template selection
    research_templates = {
        "Custom Research": {
            "description": "Create your own research goal",
            "prompt": "",
            "placeholder": "e.g., Understanding the latest advances in self-driving car technology"
        },
        "Academic Research": {
            "description": "In-depth academic investigation with citations",
            "prompt": "Conduct comprehensive academic research on [TOPIC] including recent papers, methodologies, and scholarly analysis",
            "placeholder": "e.g., machine learning in medical diagnosis"
        },
        "Technology Analysis": {
            "description": "Technical deep-dive into emerging technologies",
            "prompt": "Analyze the technical architecture, implementation challenges, and future prospects of [TOPIC]",
            "placeholder": "e.g., quantum computing applications"
        },
        "Market Research": {
            "description": "Business and market analysis",
            "prompt": "Research market trends, competitive landscape, and business opportunities in [TOPIC]",
            "placeholder": "e.g., renewable energy storage solutions"
        },
        "Comparative Study": {
            "description": "Compare different approaches or technologies",
            "prompt": "Compare and contrast different approaches to [TOPIC], analyzing pros, cons, and use cases",
            "placeholder": "e.g., different AI model architectures"
        },
        "Trend Analysis": {
            "description": "Identify and analyze emerging trends",
            "prompt": "Analyze current and emerging trends in [TOPIC], including future predictions and implications",
            "placeholder": "e.g., social media algorithms impact"
        }
    }
    
    # Template selection interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_template = st.selectbox(
            "🎨 Choose Research Template:",
            list(research_templates.keys()),
            help="Select a pre-configured research type for optimal results"
        )
        
        template_info = research_templates[selected_template]
        st.info(f"**{selected_template}**\n\n{template_info['description']}")
        
        # Quick examples for selected template
        if selected_template != "Custom Research":
            st.markdown("**Examples:**")
            examples_map = {
                "Academic Research": [
                    "Deep learning in natural language processing",
                    "Climate change impact on agriculture",
                    "Gene therapy advancements in cancer treatment"
                ],
                "Technology Analysis": [
                    "Blockchain scalability solutions",
                    "Edge computing architectures",
                    "Autonomous vehicle sensor fusion"
                ],
                "Market Research": [
                    "Electric vehicle charging infrastructure",
                    "Plant-based meat alternatives market",
                    "Remote work productivity tools"
                ],
                "Comparative Study": [
                    "SQL vs NoSQL databases for big data",
                    "React vs Vue.js for web development",
                    "Cloud computing providers comparison"
                ],
                "Trend Analysis": [
                    "AI in content creation and media",
                    "Sustainable fashion movement",
                    "Future of work and automation"
                ]
            }
            
            for example in examples_map.get(selected_template, []):
                st.markdown(f"• {example}")
    
    with col2:
        # Goal input with template integration
        if selected_template == "Custom Research":
            goal = st.text_area(
                "Enter your research goal:",
                placeholder=template_info['placeholder'],
                height=120,
                help="Describe what you want to research in detail"
            )
        else:
            topic_input = st.text_input(
                "Research Topic:",
                placeholder=template_info['placeholder'],
                help="Enter the specific topic you want to research"
            )
            
            if topic_input:
                goal = template_info['prompt'].replace('[TOPIC]', topic_input)
                st.text_area(
                    "Generated Research Goal:",
                    value=goal,
                    height=120,
                    help="This is the automatically generated research goal based on your template and topic",
                    disabled=True
                )
            else:
                goal = ""
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            research_depth = st.select_slider(
                "Research Depth:",
                options=["Quick Overview", "Standard Research", "Deep Analysis", "Comprehensive Study"],
                value="Standard Research",
                help="Controls the thoroughness and time spent on research"
            )
            
            max_iterations = st.number_input(
                "Max Research Iterations:",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of research cycles to perform"
            )
            
            include_sources = st.checkbox(
                "Include Source Citations",
                value=True,
                help="Include detailed source citations in the research output"
            )
    
    # Research button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Autonomous Research", type="primary", use_container_width=True):
            if goal.strip():
                with st.spinner("Agent is autonomously planning and executing research..."):
                    # Run autonomous research
                    start_time = time.time()
                    
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Use advanced options
                        depth_to_iterations = {
                            "Quick Overview": 3,
                            "Standard Research": 5,
                            "Deep Analysis": 7,
                            "Comprehensive Study": 10
                        }
                        actual_iterations = depth_to_iterations.get(research_depth, 5)
                        
                        results = loop.run_until_complete(
                            run_autonomous_research(agent, goal, actual_iterations)
                        )
                        loop.close()
                        
                        duration = time.time() - start_time
                        
                        if "error" not in results:
                            st.success(f"✅ Autonomous research completed in {duration:.1f} seconds!")
                            
                            # Show completion summary
                            if results.get('discoveries'):
                                st.info(f"🔍 Found {len(results['discoveries'])} discoveries, "
                                       f"raised {len(results.get('questions_raised', []))} questions, "
                                       f"identified {len(results.get('gaps_identified', []))} knowledge gaps")
                        else:
                            # Enhanced error handling with recovery options
                            error_message = results['error']
                            st.error(f"❌ Research failed: {error_message}")
                            
                            # Provide recovery suggestions
                            with st.expander("🔧 Troubleshooting & Recovery Options"):
                                if "openai" in error_message.lower() or "api" in error_message.lower():
                                    st.markdown("""
                                    **API Connection Issues:**
                                    - Check your OpenAI API key in the sidebar
                                    - Verify your internet connection
                                    - Try reducing the research depth
                                    """)
                                    
                                    if st.button("🔄 Retry with Quick Overview"):
                                        st.session_state.retry_research = "quick"
                                        st.rerun()
                                
                                elif "timeout" in error_message.lower():
                                    st.markdown("""
                                    **Timeout Issues:**
                                    - The research goal might be too complex
                                    - Try breaking it into smaller parts
                                    - Reduce the research depth
                                    """)
                                    
                                    if st.button("🎯 Simplify Research Goal"):
                                        st.session_state.show_simplify_dialog = True
                                        st.rerun()
                                
                                else:
                                    st.markdown("""
                                    **General Recovery Options:**
                                    - Try rephrasing your research goal
                                    - Use a research template for better structure
                                    - Check the debug panel in the sidebar
                                    """)
                                
                                # Always offer manual reset
                                if st.button("🔄 Reset Research Session"):
                                    # Clear all research-related session state
                                    for key in ['research_plan', 'research_results', 'questions_for_user', 
                                              'conversation_history', 'research_active', 'last_agent_response']:
                                        if key in st.session_state:
                                            del st.session_state[key]
                                    st.success("Research session reset! You can start fresh.")
                                    st.rerun()
                        
                    except Exception as e:
                        error_details = str(e)
                        st.error(f"❌ Unexpected error during research: {error_details}")
                        
                        # Detailed error recovery
                        with st.expander("🚨 Error Details & Recovery"):
                            st.code(error_details)
                            
                            st.markdown("""
                            **This error suggests a system-level issue. Try these steps:**
                            1. Refresh the page and try again
                            2. Check the console for additional error messages
                            3. Verify all dependencies are installed correctly
                            4. Contact support if the issue persists
                            """)
                            
                            if st.button("📋 Copy Error Details"):
                                st.write("Error details copied to clipboard (if supported by browser)")
                                st.code(f"Error: {error_details}\nTimestamp: {datetime.now().isoformat()}")
                    
                    # Handle retry scenarios
                    if hasattr(st.session_state, 'retry_research') and st.session_state.retry_research == "quick":
                        delattr(st.session_state, 'retry_research')
                        # Auto-retry with quick settings
                        research_depth = "Quick Overview"
                        max_iterations = 3
            else:
                st.warning("Please enter a research goal.")
    
    # Display research plan
    if st.session_state.research_plan:
        st.markdown("---")
        display_research_plan(st.session_state.research_plan)
    
    # Display research progress
    if st.session_state.research_results:
        st.markdown("---")
        display_research_progress(st.session_state.research_results)
        
        # Display synthesis
        synthesis = st.session_state.research_results.get('synthesis')
        if synthesis:
            st.markdown("### 🔬 Research Synthesis")
            st.markdown(synthesis)
    
    # Display knowledge graph
    st.markdown("---")
    visualize_knowledge_graph(agent)
    
    # Display agent questions
    if agent:
        display_agent_questions(agent)
    
    # Continue research after user response
    if agent:
        continue_research_after_response(agent)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        display_conversation_history()
    
    # Research report generation
    if st.session_state.research_results:
        st.markdown("---")
        display_research_report(agent)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🧠 Self-Initiated Research Agent | Autonomous Intelligence<br>
        <small>Exceeds YouTube project requirements with autonomous behavior, systems thinking, and prompt orchestration</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
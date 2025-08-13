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

# Custom CSS for advanced interface
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .autonomous-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .knowledge-graph-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .research-plan-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .progress-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .question-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .autonomous-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .gap-identified {
        background: #ffeaa7;
        color: #2d3436;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #fdcb6e;
    }
    .discovery-item {
        background: #e8f4fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #0984e3;
    }
</style>
""", unsafe_allow_html=True)


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
            st.markdown(f"{i}. {sub_goal}")
        
        st.markdown("**🔍 Research Questions**")
        for i, question in enumerate(plan.research_questions, 1):
            st.markdown(f"{i}. {question}")
    
    with col2:
        st.markdown("**📚 Priority Topics**")
        for i, topic in enumerate(plan.priority_topics, 1):
            st.markdown(f"{i}. {topic}")
        
        st.markdown("**🔬 Search Strategies**")
        for i, strategy in enumerate(plan.search_strategies, 1):
            st.markdown(f"{i}. {strategy}")


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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Nodes</h3>
            <h2>{len(node_data)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        facts = [n for n in nodes if n.node_type == 'fact']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Facts</h3>
            <h2>{len(facts)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        questions = [n for n in nodes if n.node_type == 'question']
        st.markdown(f"""
        <div class="metric-card">
            <h3>Questions</h3>
            <h2>{len(questions)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
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
            source = discovery.get('source', 'Unknown')
            content = discovery.get('content', '')[:200] + "..." if len(discovery.get('content', '')) > 200 else discovery.get('content', '')
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
            st.markdown(f"""
            <div class="gap-identified">
                📊 Gap: {gap}
            </div>
            """, unsafe_allow_html=True)


def display_agent_questions(agent):
    """Display questions the agent wants to ask the user"""
    if not agent or not agent.questions_for_user:
        return
    
    st.markdown("### ❓ Agent Questions")
    st.markdown("The agent has identified areas where your input would help refine the research:")
    
    for i, question in enumerate(agent.questions_for_user, 1):
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
            # Process user response through the agent
            response = agent.interactive_dialogue(user_response)
            st.success("Agent Response: " + response)
            # Clear questions after response
            agent.questions_for_user = []
            st.experimental_rerun()


def display_research_report(agent):
    """Display and allow download of research report"""
    if not agent:
        return
    
    st.markdown("### 📊 Professional Research Report")
    
    if st.button("📄 Generate Research Report"):
        with st.spinner("Generating professional research report..."):
            report = agent.generate_research_report()
            st.session_state.research_report = report
    
    if hasattr(st.session_state, 'research_report'):
        # Display report preview
        st.markdown("**Report Preview:**")
        st.text_area("Research Report", st.session_state.research_report, height=300)
        
        # Download button
        st.download_button(
            label="📥 Download Full Report",
            data=st.session_state.research_report,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


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
    st.markdown('<h1 class="main-header">🧠 Self-Initiated Research Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Autonomous AI that builds research plans, identifies gaps, and asks questions</p>', unsafe_allow_html=True)
    
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
    
    # Sidebar with agent state info
    st.sidebar.markdown("## 🤖 Agent Status")
    if agent:
        state = agent.state.value if hasattr(agent, 'state') else 'Ready'
        st.sidebar.info(f"Current State: **{state.title()}**")
        
        if hasattr(agent, 'session_id'):
            st.sidebar.info(f"Session: {agent.session_id}")
    
    # Main interface
    st.markdown("---")
    st.markdown("### 🎯 Research Goal")
    
    # Example goals
    example_goals = [
        "Understanding the latest advances in self-driving car technology",
        "Investigating the impact of artificial intelligence on healthcare",
        "Exploring quantum computing applications in cryptography",
        "Analyzing the future of renewable energy storage",
        "Examining blockchain technology beyond cryptocurrency"
    ]
    
    selected_example = st.selectbox(
        "Select an example or enter your own:",
        [""] + example_goals,
        help="Choose a complex research goal that requires autonomous investigation"
    )
    
    goal = st.text_area(
        "Enter your high-level research goal:",
        value=selected_example if selected_example else "",
        placeholder="e.g., Understanding the latest advances in self-driving car technology",
        height=100,
        help="The agent will autonomously create a research plan and identify knowledge gaps"
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
                        results = loop.run_until_complete(
                            run_autonomous_research(agent, goal)
                        )
                        loop.close()
                        
                        duration = time.time() - start_time
                        
                        if "error" not in results:
                            st.success(f"✅ Autonomous research completed in {duration:.1f} seconds!")
                        else:
                            st.error(f"❌ Research failed: {results['error']}")
                        
                    except Exception as e:
                        st.error(f"Research failed: {e}")
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
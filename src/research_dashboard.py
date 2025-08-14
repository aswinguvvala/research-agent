"""
Interactive Research Dashboard
Modern web interface for the Enhanced Multi-Agent Research System
using Streamlit for real-time research interaction and visualization.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import sys
import os

# Import the enhanced research system with robust fallback
try:
    from .enhanced_research_system import EnhancedResearchSystem
    from .advanced_reasoning import ReasoningType
except ImportError:
    # For standalone execution without package context
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from enhanced_research_system import EnhancedResearchSystem
    from advanced_reasoning import ReasoningType

# Configure logging for dashboard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchDashboard:
    """Streamlit-based interactive research dashboard"""
    
    def __init__(self):
        self.system: Optional[EnhancedResearchSystem] = None
        self.session_state = st.session_state
        
        # Initialize session state
        if "initialized" not in self.session_state:
            self.session_state.initialized = False
            self.session_state.research_history = []
            self.session_state.current_research = None
            self.session_state.system_metrics = {}
    
    def run(self):
        """Main dashboard interface"""
        st.set_page_config(
            page_title="Enhanced Research System",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._apply_custom_styling()
        
        # Main header
        st.title("🤖 Enhanced Multi-Agent Research System")
        st.markdown("*Powered by specialized AI agents, vector knowledge, and advanced reasoning*")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content area
        if not self.session_state.initialized:
            self._render_setup_page()
        else:
            self._render_main_dashboard()
    
    def _apply_custom_styling(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin: 0.5rem 0;
        }
        
        .research-result {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .agent-status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .agent-active { background: #d4edda; color: #155724; }
        .agent-idle { background: #f8d7da; color: #721c24; }
        
        .reasoning-step {
            background: #f1f3f4;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 3px solid #4285f4;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with system controls and metrics"""
        st.sidebar.title("System Control")
        
        # API Key configuration
        if not self.session_state.initialized:
            st.sidebar.subheader("🔑 Configuration")
            openai_key = st.sidebar.text_input(
                "OpenAI API Key", 
                type="password",
                help="Required for AI agent functionality"
            )
            
            # Optional API keys
            with st.sidebar.expander("Optional API Keys"):
                semantic_scholar_key = st.sidebar.text_input(
                    "Semantic Scholar API Key", 
                    type="password",
                    help="For enhanced academic search"
                )
                news_api_key = st.sidebar.text_input(
                    "News API Key", 
                    type="password", 
                    help="For real-time news integration"
                )
            
            if st.sidebar.button("🚀 Initialize System"):
                if openai_key:
                    self._initialize_system(openai_key, semantic_scholar_key, news_api_key)
                else:
                    st.sidebar.error("OpenAI API Key is required")
        
        else:
            # System status
            st.sidebar.subheader("📊 System Status")
            
            if self.session_state.system_metrics:
                metrics = self.session_state.system_metrics
                
                st.sidebar.metric(
                    "Research Sessions",
                    metrics.get('total_sessions', 0),
                    delta=None
                )
                
                success_rate = metrics.get('success_rate', 0)
                st.sidebar.metric(
                    "Success Rate", 
                    f"{success_rate:.1%}",
                    delta=None
                )
                
                st.sidebar.metric(
                    "Active Agents",
                    metrics.get('agents_available', 0)
                )
                
                st.sidebar.metric(
                    "Knowledge Items",
                    metrics.get('knowledge_system', {}).get('total_items', 0)
                )
            
            # Research history
            st.sidebar.subheader("📚 Research History")
            if self.session_state.research_history:
                for i, research in enumerate(reversed(self.session_state.research_history[-5:])):
                    with st.sidebar.expander(f"Session {len(self.session_state.research_history)-i}"):
                        st.write(f"**Goal:** {research['research_goal'][:50]}...")
                        st.write(f"**Duration:** {research['session_metrics']['duration_seconds']:.1f}s")
                        st.write(f"**Quality:** {research['quality_assurance']['quality_level']}")
            else:
                st.sidebar.info("No research sessions yet")
    
    def _render_setup_page(self):
        """Render the initial setup page"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class="main-header">
                <h2>🤖 Enhanced Research System Setup</h2>
                <p>Configure your advanced AI research assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### System Capabilities
            
            🔬 **Specialized AI Agents**
            - Academic Research Specialist (papers, citations, literature reviews)  
            - Technical Implementation Expert (code, tools, best practices)
            - Business Applications Analyst (market research, ROI analysis)
            - Social Impact Researcher (trends, ethics, human factors)
            
            🧠 **Advanced Intelligence**
            - Vector-based semantic search and knowledge management
            - Chain-of-thought reasoning with metacognitive reflection
            - Cross-domain pattern recognition and synthesis
            - Real-time fact-checking and quality assurance
            
            🔍 **Modern Data Sources**
            - Semantic Scholar (better than Google Scholar)
            - OpenAlex comprehensive academic database
            - Enhanced arXiv with smart classification
            - Real-time news and trend monitoring
            
            ### Getting Started
            
            1. Enter your OpenAI API key in the sidebar
            2. Optionally add Semantic Scholar and News API keys for enhanced capabilities
            3. Click "Initialize System" to start your research assistant
            4. Begin asking research questions!
            """)
            
            if st.button("🎯 See Example Research Questions"):
                st.markdown("""
                ### Example Research Questions
                
                **Academic Research:**
                - "What are the latest advances in transformer architecture for natural language processing?"
                - "How effective are CRISPR gene editing techniques for treating genetic diseases?"
                
                **Technical Implementation:**
                - "What are the best practices for implementing microservices architecture in Python?"
                - "How can I optimize deep learning model inference for edge devices?"
                
                **Business Analysis:**
                - "What is the market opportunity for AI-powered healthcare diagnostics?"
                - "How are companies using blockchain for supply chain management?"
                
                **Social Impact:**
                - "What are the ethical implications of facial recognition technology?"
                - "How is remote work affecting employee productivity and well-being?"
                """)
    
    def _render_main_dashboard(self):
        """Render the main research dashboard"""
        # Research input section
        st.subheader("🎯 Research Query")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            research_goal = st.text_area(
                "What would you like to research?",
                height=100,
                placeholder="Enter your research question here...",
                key="research_input"
            )
        
        with col2:
            st.write("**Reasoning Type:**")
            reasoning_type = st.selectbox(
                "Select reasoning approach",
                options=list(ReasoningType),
                format_func=lambda x: x.value.title(),
                key="reasoning_type"
            )
            
            max_duration = st.slider(
                "Max Duration (minutes)",
                min_value=2,
                max_value=15,
                value=8,
                key="max_duration"
            )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Start Research", type="primary"):
                if research_goal.strip():
                    self._start_research(research_goal.strip(), reasoning_type, max_duration * 60)
                else:
                    st.error("Please enter a research question")
        
        with col2:
            if st.button("📊 Update Metrics"):
                self._update_system_metrics()
        
        # Current research status
        if hasattr(self.session_state, 'research_in_progress') and self.session_state.research_in_progress:
            self._render_research_progress()
        
        # Research results
        if self.session_state.current_research:
            self._render_research_results()
        
        # System analytics
        if self.session_state.research_history:
            self._render_analytics()
    
    def _render_research_progress(self):
        """Render research progress indicators"""
        st.subheader("🔄 Research in Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress updates (in real implementation, this would be connected to actual progress)
        phases = [
            "🧠 Advanced problem reasoning...",
            "📚 Gathering contextual knowledge...", 
            "🔍 Comprehensive source search...",
            "🤖 Multi-agent research execution...",
            "🔀 Knowledge integration...",
            "✅ Quality assurance...",
            "💾 Storing new knowledge..."
        ]
        
        for i, phase in enumerate(phases):
            progress = (i + 1) / len(phases)
            progress_bar.progress(progress)
            status_text.text(phase)
            # In real implementation, this would be actual progress updates
    
    def _render_research_results(self):
        """Render comprehensive research results"""
        results = self.session_state.current_research
        
        st.markdown("---")
        st.subheader("📋 Research Results")
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Duration", 
                f"{results['session_metrics']['duration_seconds']:.1f}s"
            )
        
        with col2:
            qa_score = results['quality_assurance']['overall_qa_score']
            st.metric(
                "Quality Score",
                f"{qa_score:.1%}",
                delta=f"{qa_score - 0.7:.1%}" if qa_score > 0.7 else None
            )
        
        with col3:
            st.metric(
                "Sources Used",
                results['session_metrics']['total_sources']
            )
        
        with col4:
            reasoning_confidence = results['reasoning_analysis']['confidence']
            st.metric(
                "Reasoning Confidence",
                f"{reasoning_confidence:.1%}"
            )
        
        # Tabs for different result sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧠 Reasoning Analysis", 
            "💡 Key Findings", 
            "📊 Source Analysis",
            "🎯 Recommendations", 
            "📈 Quality Metrics"
        ])
        
        with tab1:
            self._render_reasoning_analysis(results['reasoning_analysis'])
        
        with tab2:
            self._render_key_findings(results['integrated_findings'])
        
        with tab3:
            self._render_source_analysis(results)
        
        with tab4:
            self._render_recommendations(results['integrated_findings'])
        
        with tab5:
            self._render_quality_metrics(results['quality_assurance'])
    
    def _render_reasoning_analysis(self, reasoning_data: Dict[str, Any]):
        """Render reasoning analysis results"""
        st.markdown("### 🧠 Advanced Reasoning Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Final Conclusion:**")
            st.info(reasoning_data['final_conclusion'])
            
            # Reasoning steps (if available)
            if reasoning_data.get('reasoning_steps', 0) > 0:
                st.markdown(f"**Reasoning Process:** {reasoning_data['reasoning_steps']} steps")
                
                # Create a simple reasoning flow visualization
                steps = ['Problem Analysis', 'Evidence Assessment', 'Primary Reasoning', 'Critical Evaluation', 'Meta-Reflection', 'Final Synthesis']
                
                fig = go.Figure(data=go.Scatter(
                    x=list(range(len(steps))),
                    y=[1] * len(steps),
                    mode='markers+lines+text',
                    text=steps,
                    textposition="top center",
                    marker=dict(size=20, color='lightblue'),
                    line=dict(width=2, color='blue')
                ))
                
                fig.update_layout(
                    title="Reasoning Process Flow",
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False, range=[0.5, 1.5]),
                    height=200
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = reasoning_data['confidence'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Reasoning Confidence"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_key_findings(self, integrated_findings: Dict[str, Any]):
        """Render key research findings"""
        synthesis = integrated_findings.get('synthesis', {})
        
        st.markdown("### 💡 Key Research Findings")
        
        # Executive summary
        exec_summary = synthesis.get('executive_summary', 'No summary available')
        st.markdown("**Executive Summary:**")
        st.success(exec_summary)
        
        # Main insights
        insights = synthesis.get('main_insights', [])
        if insights:
            st.markdown("**Main Insights:**")
            for i, insight in enumerate(insights, 1):
                st.markdown(f"{i}. {insight}")
        
        # Cross-source patterns
        patterns = synthesis.get('cross_source_patterns', [])
        if patterns:
            st.markdown("**Cross-Source Patterns:**")
            for pattern in patterns:
                st.info(f"🔍 {pattern}")
        
        # Research gaps
        gaps = synthesis.get('research_gaps', [])
        if gaps:
            st.markdown("**Identified Research Gaps:**")
            for gap in gaps:
                st.warning(f"❓ {gap}")
    
    def _render_source_analysis(self, results: Dict[str, Any]):
        """Render analysis of data sources used"""
        st.markdown("### 📊 Source Analysis")
        
        # API sources breakdown
        api_sources = results['api_sources']
        
        if api_sources:
            # Create source distribution chart
            source_data = pd.DataFrame([
                {'Source': source.replace('_', ' ').title(), 'Items': count}
                for source, count in api_sources.items()
            ])
            
            fig = px.bar(
                source_data, 
                x='Source', 
                y='Items',
                title='Research Sources Distribution',
                color='Items',
                color_continuous_scale='blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Knowledge context
        knowledge_context = results['knowledge_context']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Knowledge Base Context:**")
            st.metric("Relevant Items Found", knowledge_context['items_found'])
            
            if knowledge_context['relevance_scores']:
                avg_relevance = sum(knowledge_context['relevance_scores']) / len(knowledge_context['relevance_scores'])
                st.metric("Average Relevance", f"{avg_relevance:.2f}")
        
        with col2:
            st.markdown("**Integration Statistics:**")
            integrated = results['integrated_findings']
            st.metric("Total Findings", integrated['total_findings'])
            st.metric("Sources Integrated", len(integrated['sources_integrated']))
    
    def _render_recommendations(self, integrated_findings: Dict[str, Any]):
        """Render actionable recommendations"""
        st.markdown("### 🎯 Actionable Recommendations")
        
        synthesis = integrated_findings.get('synthesis', {})
        recommendations = synthesis.get('actionable_recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{i}. {rec}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific recommendations generated")
        
        # Research gaps as potential next steps
        gaps = synthesis.get('research_gaps', [])
        if gaps:
            st.markdown("**Suggested Next Research Steps:**")
            for gap in gaps:
                st.markdown(f"• {gap}")
    
    def _render_quality_metrics(self, qa_data: Dict[str, Any]):
        """Render quality assurance metrics"""
        st.markdown("### 📈 Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall quality score
            overall_score = qa_data.get('overall_qa_score', 0)
            quality_level = qa_data.get('quality_level', 'unknown')
            
            st.metric("Overall QA Score", f"{overall_score:.1%}")
            st.metric("Quality Level", quality_level.title())
        
        with col2:
            # Detailed metrics
            completeness = qa_data.get('completeness_score', 0)
            diversity = qa_data.get('source_diversity', 0)
            consistency = qa_data.get('confidence_consistency', 0)
            
            st.metric("Completeness", f"{completeness:.1%}")
            st.metric("Source Diversity", diversity)
            st.metric("Confidence Consistency", f"{consistency:.1%}")
        
        # Quality visualization
        quality_metrics = {
            'Completeness': completeness,
            'Source Diversity': min(diversity / 5, 1),  # Normalize to 0-1
            'Confidence': consistency,
            'Overall': overall_score
        }
        
        fig = px.bar(
            x=list(quality_metrics.keys()),
            y=list(quality_metrics.values()),
            title='Quality Metrics Breakdown',
            color=list(quality_metrics.values()),
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_analytics(self):
        """Render system analytics and historical data"""
        st.markdown("---")
        st.subheader("📊 Research Analytics")
        
        if len(self.session_state.research_history) < 2:
            st.info("More research sessions needed for analytics")
            return
        
        # Prepare analytics data
        history_data = []
        for i, session in enumerate(self.session_state.research_history):
            history_data.append({
                'Session': i + 1,
                'Duration': session['session_metrics']['duration_seconds'],
                'Quality Score': session['quality_assurance']['overall_qa_score'],
                'Sources Used': session['session_metrics']['total_sources'],
                'Reasoning Confidence': session['reasoning_analysis']['confidence']
            })
        
        df = pd.DataFrame(history_data)
        
        # Analytics tabs
        tab1, tab2 = st.tabs(["📈 Trends", "📋 Session Comparison"])
        
        with tab1:
            # Quality trend over time
            fig = px.line(
                df, 
                x='Session', 
                y='Quality Score',
                title='Research Quality Over Time',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Duration vs Quality correlation
            fig = px.scatter(
                df,
                x='Duration',
                y='Quality Score',
                title='Duration vs Quality Correlation',
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.dataframe(df, use_container_width=True)
    
    def _initialize_system(self, openai_key: str, semantic_scholar_key: str = None, news_api_key: str = None):
        """Initialize the enhanced research system"""
        try:
            with st.spinner("🚀 Initializing Enhanced Research System..."):
                config = {
                    "api_keys": {
                        "semantic_scholar": semantic_scholar_key,
                        "news_api": news_api_key,
                        "email": "dashboard@enhanced-research.ai"
                    },
                    "chroma_db_path": "./dashboard_research_db"
                }
                
                self.system = EnhancedResearchSystem(openai_key, config)
                
                # Initialize asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.system.initialize_system())
                
                self.session_state.initialized = True
                self._update_system_metrics()
                
                st.sidebar.success("✅ System initialized successfully!")
                st.rerun()
        
        except Exception as e:
            st.sidebar.error(f"❌ Initialization failed: {str(e)}")
            logger.error(f"System initialization error: {e}")
    
    def _start_research(self, research_goal: str, reasoning_type: ReasoningType, max_duration: int):
        """Start a research session"""
        try:
            self.session_state.research_in_progress = True
            
            with st.spinner("🔬 Conducting comprehensive research..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                results = loop.run_until_complete(
                    self.system.conduct_enhanced_research(research_goal, max_duration)
                )
                
                self.session_state.current_research = results
                self.session_state.research_history.append(results)
                self.session_state.research_in_progress = False
                
                # Update metrics
                self._update_system_metrics()
                
                st.success("✅ Research completed!")
                st.rerun()
        
        except Exception as e:
            self.session_state.research_in_progress = False
            st.error(f"❌ Research failed: {str(e)}")
            logger.error(f"Research error: {e}")
    
    def _update_system_metrics(self):
        """Update system metrics"""
        if self.system:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # In a real async context, this would be properly awaited
                metrics = self.system.get_system_metrics()
                self.session_state.system_metrics = metrics
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")


def main():
    """Main dashboard application"""
    dashboard = ResearchDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
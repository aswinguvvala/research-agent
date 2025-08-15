"""
Enhanced UI Components for AI Research Agent Pro
Advanced Streamlit components with modern design and enhanced functionality.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import base64
from io import BytesIO
import pandas as pd

def create_animated_progress_ring(percentage: float, size: int = 120, color: str = "#3b82f6"):
    """Create an animated progress ring using Plotly"""
    
    fig = go.Figure()
    
    # Background circle
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(
            size=size,
            color='rgba(229, 231, 235, 0.3)',
            line=dict(width=0)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Progress arc (simplified representation)
    theta = [i * 360 / 100 for i in range(int(percentage) + 1)]
    r = [size/2] * len(theta)
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='lines',
        line=dict(color=color, width=8),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Center text
    fig.add_annotation(
        x=0, y=0,
        text=f"<b>{percentage:.0f}%</b>",
        showarrow=False,
        font=dict(size=16, color=color)
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, size/2]),
            angularaxis=dict(visible=False)
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=size + 40,
        width=size + 40,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_research_timeline(events: List[Dict], height: int = 400):
    """Create an interactive research timeline"""
    
    if not events:
        return None
    
    # Sort events by timestamp
    events_sorted = sorted(events, key=lambda x: x.get('timestamp', datetime.now()))
    
    fig = go.Figure()
    
    # Add timeline events
    for i, event in enumerate(events_sorted):
        timestamp = event.get('timestamp', datetime.now())
        title = event.get('title', f'Event {i+1}')
        description = event.get('description', '')
        event_type = event.get('type', 'info')
        
        # Color mapping for event types
        color_map = {
            'discovery': '#10b981',
            'question': '#f59e0b', 
            'gap': '#ef4444',
            'synthesis': '#3b82f6',
            'info': '#6b7280'
        }
        
        color = color_map.get(event_type, '#6b7280')
        
        # Add event marker
        fig.add_trace(go.Scatter(
            x=[timestamp],
            y=[i],
            mode='markers+text',
            marker=dict(
                size=15,
                color=color,
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            text=[title],
            textposition="middle right",
            textfont=dict(size=12),
            name=event_type.title(),
            hovertemplate=f'<b>{title}</b><br>{description}<br>%{{x}}<extra></extra>',
            showlegend=False
        ))
    
    # Add connecting line
    timestamps = [event.get('timestamp', datetime.now()) for event in events_sorted]
    y_positions = list(range(len(events_sorted)))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=y_positions,
        mode='lines',
        line=dict(color='rgba(156, 163, 175, 0.5)', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Research Timeline",
        xaxis=dict(title="Time"),
        yaxis=dict(visible=False),
        height=height,
        margin=dict(t=40, b=40, l=100, r=100),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_knowledge_metrics_dashboard(agent):
    """Create a comprehensive metrics dashboard"""
    
    if not agent or not agent.knowledge_graph.nodes:
        return None
    
    nodes = list(agent.knowledge_graph.nodes.values())
    
    # Calculate metrics
    total_nodes = len(nodes)
    fact_nodes = [n for n in nodes if n.node_type == 'fact']
    question_nodes = [n for n in nodes if n.node_type == 'question']
    hypothesis_nodes = [n for n in nodes if n.node_type == 'hypothesis']
    gaps = agent.knowledge_graph.find_knowledge_gaps()
    
    # Average confidence
    avg_confidence = sum(n.confidence for n in nodes) / total_nodes if total_nodes > 0 else 0
    
    # Connection density
    total_edges = agent.knowledge_graph.graph.number_of_edges()
    max_possible_edges = total_nodes * (total_nodes - 1) / 2 if total_nodes > 1 else 1
    connection_density = (total_edges / max_possible_edges) * 100 if max_possible_edges > 0 else 0
    
    # Create metrics visualization
    fig = go.Figure()
    
    # Metrics pie chart
    fig.add_trace(go.Pie(
        labels=['Facts', 'Questions', 'Hypotheses', 'Gaps'],
        values=[len(fact_nodes), len(question_nodes), len(hypothesis_nodes), len(gaps)],
        hole=.3,
        marker_colors=['#10b981', '#f59e0b', '#3b82f6', '#ef4444']
    ))
    
    fig.update_layout(
        title="Knowledge Distribution",
        annotations=[dict(text=f'{total_nodes}<br>Total', x=0.5, y=0.5, font_size=16, showarrow=False)],
        height=300,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    return fig, {
        'total_nodes': total_nodes,
        'avg_confidence': avg_confidence,
        'connection_density': connection_density,
        'facts': len(fact_nodes),
        'questions': len(question_nodes),
        'hypotheses': len(hypothesis_nodes),
        'gaps': len(gaps)
    }

def create_source_analysis_chart(results: Dict[str, Any]):
    """Create a chart analyzing research sources"""
    
    discoveries = results.get('discoveries', [])
    if not discoveries:
        return None
    
    # Analyze sources
    source_counts = {}
    source_confidence = {}
    
    for discovery in discoveries:
        source = discovery.get('source', 'Unknown')
        confidence = discovery.get('confidence', 0.5)
        
        # Simplify source names
        if 'arxiv' in source.lower():
            source_type = 'arXiv Papers'
        elif 'scholar' in source.lower():
            source_type = 'Google Scholar'
        elif 'wikipedia' in source.lower():
            source_type = 'Wikipedia'
        elif 'web' in source.lower():
            source_type = 'Web Sources'
        else:
            source_type = 'Other'
        
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        if source_type not in source_confidence:
            source_confidence[source_type] = []
        source_confidence[source_type].append(confidence)
    
    # Calculate average confidence per source
    avg_confidence = {
        source: sum(confidences) / len(confidences) 
        for source, confidences in source_confidence.items()
    }
    
    # Create combined chart
    fig = go.Figure()
    
    sources = list(source_counts.keys())
    counts = list(source_counts.values())
    confidences = [avg_confidence[source] * 100 for source in sources]
    
    # Bar chart for counts
    fig.add_trace(go.Bar(
        x=sources,
        y=counts,
        name='Discovery Count',
        marker_color='#3b82f6',
        yaxis='y'
    ))
    
    # Line chart for confidence
    fig.add_trace(go.Scatter(
        x=sources,
        y=confidences,
        mode='lines+markers',
        name='Avg Confidence (%)',
        line=dict(color='#ef4444', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Research Sources Analysis',
        xaxis=dict(title='Source Type'),
        yaxis=dict(title='Number of Discoveries', side='left'),
        yaxis2=dict(title='Average Confidence (%)', side='right', overlaying='y'),
        height=400,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    return fig

def create_research_quality_score(results: Dict[str, Any], agent):
    """Calculate and visualize research quality score"""
    
    if not results or not agent:
        return 0, None
    
    discoveries = results.get('discoveries', [])
    questions = results.get('questions_raised', [])
    gaps = results.get('gaps_identified', [])
    synthesis = results.get('synthesis', '')
    
    # Calculate quality components
    discovery_score = min(len(discoveries) * 2, 30)  # Max 30 points
    question_score = min(len(questions) * 1.5, 20)  # Max 20 points
    gap_score = min(len(gaps) * 2, 20)  # Max 20 points
    synthesis_score = 20 if synthesis and len(synthesis) > 100 else 0  # 20 points
    
    # Source diversity bonus
    unique_sources = set()
    for discovery in discoveries:
        source = discovery.get('source', '')
        if 'arxiv' in source.lower():
            unique_sources.add('academic')
        elif 'scholar' in source.lower():
            unique_sources.add('scholarly')
        elif 'wikipedia' in source.lower():
            unique_sources.add('encyclopedia')
        elif 'web' in source.lower():
            unique_sources.add('web')
    
    diversity_score = min(len(unique_sources) * 2.5, 10)  # Max 10 points
    
    total_score = discovery_score + question_score + gap_score + synthesis_score + diversity_score
    
    # Create quality breakdown chart
    components = ['Discoveries', 'Questions', 'Gaps', 'Synthesis', 'Source Diversity']
    scores = [discovery_score, question_score, gap_score, synthesis_score, diversity_score]
    max_scores = [30, 20, 20, 20, 10]
    
    fig = go.Figure()
    
    # Add bars for actual scores
    fig.add_trace(go.Bar(
        x=components,
        y=scores,
        name='Actual Score',
        marker_color='#3b82f6',
        text=[f'{s:.0f}' for s in scores],
        textposition='auto'
    ))
    
    # Add bars for maximum possible scores
    fig.add_trace(go.Bar(
        x=components,
        y=[m - s for m, s in zip(max_scores, scores)],
        name='Remaining',
        marker_color='rgba(229, 231, 235, 0.5)',
        base=scores
    ))
    
    fig.update_layout(
        title=f'Research Quality Score: {total_score:.0f}/100',
        xaxis=dict(title='Quality Components'),
        yaxis=dict(title='Score'),
        barmode='stack',
        height=400,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    return total_score, fig

def create_download_report_section(agent, results: Dict[str, Any]):
    """Create enhanced download section with multiple formats"""
    
    if not agent or not results:
        return
    
    st.markdown("""
    <div class="research-card">
        <h2 style="margin-bottom: 1rem;">📊 Export Research Results</h2>
        <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
            Download your research in professional formats suitable for sharing and presentation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate reports in different formats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        # JSON Export
        if st.button("📄 JSON Data", use_container_width=True, help="Structured data export"):
            json_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "research_goal": st.session_state.research_plan.goal if st.session_state.research_plan else None,
                    "agent_version": "2.0-enhanced"
                },
                "results": results,
                "conversation_history": st.session_state.conversation_history,
                "knowledge_graph": {
                    "total_nodes": len(agent.knowledge_graph.nodes),
                    "node_types": {
                        node_type: len([n for n in agent.knowledge_graph.nodes.values() if n.node_type == node_type])
                        for node_type in ['fact', 'question', 'hypothesis', 'gap']
                    }
                }
            }
            
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(json_data, indent=2, default=str),
                file_name=f"research_data_{timestamp}.json",
                mime="application/json"
            )
    
    with col2:
        # Markdown Report
        if st.button("📝 Markdown", use_container_width=True, help="Human-readable report"):
            if hasattr(agent, 'generate_research_report'):
                markdown_report = agent.generate_research_report()
            else:
                markdown_report = f"""# Research Report
                
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Goal: {st.session_state.research_plan.goal if st.session_state.research_plan else 'N/A'}

## Summary
{results.get('synthesis', 'No synthesis available')}

## Key Findings
- Discoveries: {len(results.get('discoveries', []))}
- Questions Raised: {len(results.get('questions_raised', []))}
- Gaps Identified: {len(results.get('gaps_identified', []))}
"""
            
            st.download_button(
                label="💾 Download MD",
                data=markdown_report,
                file_name=f"research_report_{timestamp}.md",
                mime="text/markdown"
            )
    
    with col3:
        # CSV Export
        if st.button("📊 CSV Data", use_container_width=True, help="Spreadsheet-compatible data"):
            discoveries = results.get('discoveries', [])
            if discoveries:
                df_data = []
                for i, discovery in enumerate(discoveries):
                    df_data.append({
                        'ID': i + 1,
                        'Content': discovery.get('content', ''),
                        'Source': discovery.get('source', ''),
                        'Confidence': discovery.get('confidence', 0),
                        'Type': 'Discovery'
                    })
                
                df = pd.DataFrame(df_data)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"research_discoveries_{timestamp}.csv",
                    mime="text/csv"
                )
    
    with col4:
        # PowerPoint Summary (simplified)
        if st.button("🎯 Executive Summary", use_container_width=True, help="Executive summary format"):
            exec_summary = f"""
EXECUTIVE RESEARCH SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OBJECTIVE:
{st.session_state.research_plan.goal if st.session_state.research_plan else 'N/A'}

KEY METRICS:
• Discoveries: {len(results.get('discoveries', []))}
• Questions Raised: {len(results.get('questions_raised', []))}
• Knowledge Gaps: {len(results.get('gaps_identified', []))}
• Research Quality: {create_research_quality_score(results, agent)[0]:.0f}/100

SYNTHESIS:
{results.get('synthesis', 'No synthesis available')[:500]}...

RECOMMENDATIONS:
• Continue investigation into identified gaps
• Validate findings through additional sources
• Consider expert consultation on open questions
"""
            
            st.download_button(
                label="💾 Download Summary",
                data=exec_summary,
                file_name=f"executive_summary_{timestamp}.txt",
                mime="text/plain"
            )

def display_research_insights(results: Dict[str, Any], agent):
    """Display advanced research insights and analytics"""
    
    if not results or not agent:
        return
    
    st.markdown("""
    <div class="research-card">
        <h2 style="margin-bottom: 1rem;">🔍 Research Insights & Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Quality score
    quality_score, quality_fig = create_research_quality_score(results, agent)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Quality score display
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem;">
                {quality_score:.0f}
            </div>
            <div style="color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                Quality Score
            </div>
            <div style="margin-top: 1rem; font-size: 0.9rem; color: var(--text-muted);">
                Based on discoveries, insights, and source diversity
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Research completeness
        discoveries = results.get('discoveries', [])
        questions = results.get('questions_raised', [])
        gaps = results.get('gaps_identified', [])
        
        completeness = min((len(discoveries) + len(questions)) / 10 * 100, 100)
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 1.5rem; margin-top: 1rem;">
            <div style="font-size: 2rem; font-weight: 700; color: var(--success-color); margin-bottom: 0.5rem;">
                {completeness:.0f}%
            </div>
            <div style="color: var(--text-secondary); font-weight: 600;">
                Research Completeness
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if quality_fig:
            st.plotly_chart(quality_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Source analysis
    source_fig = create_source_analysis_chart(results)
    if source_fig:
        st.plotly_chart(source_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Knowledge metrics
    metrics_result = create_knowledge_metrics_dashboard(agent)
    if metrics_result:
        metrics_fig, metrics_data = metrics_result
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(metrics_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("### 📈 Knowledge Metrics")
            
            metrics_display = [
                ("Total Knowledge Nodes", metrics_data['total_nodes']),
                ("Average Confidence", f"{metrics_data['avg_confidence']*100:.0f}%"),
                ("Connection Density", f"{metrics_data['connection_density']:.1f}%"),
                ("Facts Discovered", metrics_data['facts']),
                ("Questions Generated", metrics_data['questions']),
                ("Gaps Identified", metrics_data['gaps'])
            ]
            
            for label, value in metrics_display:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                    <span style="color: var(--text-secondary);">{label}:</span>
                    <span style="font-weight: 600; color: var(--text-primary);">{value}</span>
                </div>
                """, unsafe_allow_html=True)

def create_floating_action_menu():
    """Create a floating action menu for quick actions"""
    
    # Add floating action menu CSS and HTML
    st.markdown("""
    <style>
        .fab-container {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 1000;
        }
        
        .fab-main {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            color: white;
            border: none;
            box-shadow: var(--shadow-lg);
            font-size: 1.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .fab-main:hover {
            transform: scale(1.1) rotate(90deg);
            box-shadow: var(--shadow-xl);
        }
        
        .fab-menu {
            position: absolute;
            bottom: 70px;
            right: 0;
            background: white;
            border-radius: var(--border-radius-lg);
            box-shadow: var(--shadow-xl);
            border: 1px solid var(--border-color);
            min-width: 200px;
            overflow: hidden;
        }
        
        .fab-item {
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background-color 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .fab-item:hover {
            background: var(--background-secondary);
        }
        
        .fab-item:last-child {
            border-bottom: none;
        }
        
        .fab-icon {
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Additional utility functions for enhanced components
def format_research_time(start_time: datetime, end_time: datetime = None) -> str:
    """Format research duration in a human-readable way"""
    if end_time is None:
        end_time = datetime.now()
    
    duration = end_time - start_time
    
    if duration.total_seconds() < 60:
        return f"{duration.total_seconds():.0f} seconds"
    elif duration.total_seconds() < 3600:
        return f"{duration.total_seconds() / 60:.1f} minutes"
    else:
        return f"{duration.total_seconds() / 3600:.1f} hours"

def calculate_research_velocity(results: Dict[str, Any]) -> float:
    """Calculate research velocity (discoveries per minute)"""
    discoveries = results.get('discoveries', [])
    
    if not discoveries:
        return 0.0
    
    # Estimate research duration (simplified)
    research_duration = 5.0  # minutes, could be calculated from actual timestamps
    
    return len(discoveries) / research_duration

def generate_research_recommendations(results: Dict[str, Any], agent) -> List[str]:
    """Generate intelligent recommendations for next steps"""
    
    recommendations = []
    
    if not results or not agent:
        return recommendations
    
    discoveries = results.get('discoveries', [])
    questions = results.get('questions_raised', [])
    gaps = results.get('gaps_identified', [])
    
    # Quality-based recommendations
    if len(discoveries) < 5:
        recommendations.append("🔍 Expand research scope to gather more comprehensive insights")
    
    if len(questions) > len(discoveries):
        recommendations.append("💡 Focus on answering existing questions before expanding scope")
    
    if len(gaps) > 3:
        recommendations.append("🎯 Prioritize filling the most critical knowledge gaps identified")
    
    # Source diversity recommendations
    sources = set()
    for discovery in discoveries:
        source = discovery.get('source', '')
        if 'arxiv' in source.lower():
            sources.add('academic')
        elif 'scholar' in source.lower():
            sources.add('scholarly')
        elif 'wikipedia' in source.lower():
            sources.add('general')
    
    if len(sources) < 3:
        recommendations.append("📚 Diversify research sources for more comprehensive coverage")
    
    # Confidence-based recommendations
    low_confidence_items = [d for d in discoveries if d.get('confidence', 1.0) < 0.6]
    if len(low_confidence_items) > len(discoveries) * 0.3:
        recommendations.append("🔬 Validate findings with additional authoritative sources")
    
    # Default recommendations
    if not recommendations:
        recommendations.extend([
            "✅ Current research shows good depth and coverage",
            "🚀 Consider exploring related topics for broader understanding",
            "📊 Review synthesis for actionable insights"
        ])
    
    return recommendations[:4]  # Limit to 4 recommendations

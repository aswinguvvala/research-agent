#!/usr/bin/env python3
"""
Test script for Self-Initiated Research Agent Streamlit App
Verifies that all components can be imported and basic functionality works.
"""

import sys
import os

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from self_initiated_research_agent import SelfInitiatedResearchAgent, ResearchState, ResearchPlan
        print("✅ SelfInitiatedResearchAgent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SelfInitiatedResearchAgent: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Plotly: {e}")
        return False
    
    try:
        import networkx as nx
        print("✅ NetworkX imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NetworkX: {e}")
        return False
    
    return True


def test_agent_initialization():
    """Test basic agent initialization (without API key)"""
    print("\n🧪 Testing agent initialization...")
    
    try:
        from self_initiated_research_agent import SelfInitiatedResearchAgent
        
        # Test with dummy key (will fail but should not crash)
        agent = SelfInitiatedResearchAgent("dummy-key")
        print("✅ Agent initialization code works")
        return True
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False


def test_knowledge_graph():
    """Test knowledge graph functionality"""
    print("\n🧪 Testing knowledge graph...")
    
    try:
        from self_initiated_research_agent import KnowledgeGraph, ResearchNode
        from datetime import datetime
        
        # Create test knowledge graph
        kg = KnowledgeGraph()
        
        # Add test node
        test_node = ResearchNode(
            id="test_1",
            content="Test research finding",
            node_type="fact",
            source="test",
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        kg.add_node(test_node)
        print(f"✅ Knowledge graph works - {len(kg.nodes)} nodes")
        return True
    except Exception as e:
        print(f"❌ Knowledge graph test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Self-Initiated Research Agent Streamlit Integration")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_agent_initialization,
        test_knowledge_graph
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The self-initiated research agent app is ready to run.")
        print("\nTo start the app, run:")
        print("streamlit run self_initiated_streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements-self-initiated.txt")


if __name__ == "__main__":
    main()
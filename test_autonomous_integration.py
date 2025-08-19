#!/usr/bin/env python3
"""
Test script to verify autonomous research agent integration
"""
import os
import asyncio
import sys

# Add paths
sys.path.append('/Users/aswin/new_research_agent')
sys.path.append('/Users/aswin/new_research_agent/research-agent-web/backend')

async def test_autonomous_integration():
    """Test the autonomous research agent integration."""
    
    print("🧪 Testing Autonomous Research Agent Integration")
    print("=" * 60)
    
    try:
        # Test import
        from autonomous_research_agent import AutonomousResearchAgent
        print("✅ AutonomousResearchAgent imported successfully")
        
        # Test AI component imports
        from ai_query_analyzer import AIQueryAnalyzer
        from ai_relevance_judge import AIRelevanceJudge
        from ai_honest_synthesizer import AIHonestSynthesizer
        from ai_search_strategist import AISearchStrategist
        
        print("✅ All AI components imported successfully")
        
        # Test web service integration
        from app.services.research_service import research_service
        print("✅ Research service imported successfully")
        
        # Check if autonomous agent is available
        from app.services.research_service import AUTONOMOUS_AVAILABLE
        print(f"🤖 Autonomous agent available: {AUTONOMOUS_AVAILABLE}")
        
        # Test API key validation
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-test-key-for-research-demo":
            print("⚠️  Using demo API key - some tests may fail")
        else:
            print("✅ Real API key detected")
        
        # Basic functionality test (without making actual API calls)
        if api_key and api_key != "sk-test-key-for-research-demo":
            print("\n🔬 Testing basic autonomous agent functionality...")
            
            agent = AutonomousResearchAgent(
                openai_api_key=api_key,
                max_sources=3,
                debug_mode=True
            )
            print("✅ Autonomous agent created successfully")
            
            # Test query analysis
            query_analyzer = AIQueryAnalyzer(api_key)
            print("✅ Query analyzer created successfully")
            
        print("\n🎉 All integration tests passed!")
        print("\n💡 Next steps:")
        print("   1. Set a real OpenAI API key for full functionality")
        print("   2. Test with actual research queries")
        print("   3. Verify autonomous behavior vs hardcoded patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_autonomous_integration())
    sys.exit(0 if success else 1)
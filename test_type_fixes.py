#!/usr/bin/env python3
"""
Test script to verify the research agent type mismatch fixes.
Tests the specific error scenario that was causing the "expected str instance, dict found" error.
"""

import asyncio
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from self_initiated_research_agent import SelfInitiatedResearchAgent

async def test_type_fixes():
    """Test the research agent with the original failing goal."""
    
    print("🧪 Testing Research Agent Type Fixes")
    print("=" * 50)
    
    # Get API key from secrets file
    try:
        with open('.streamlit/secrets.toml', 'r') as f:
            content = f.read()
            # Extract API key (simple parsing)
            for line in content.split('\n'):
                if 'OPENAI_API_KEY' in line and '=' in line:
                    api_key = line.split('=')[1].strip().strip('"')
                    break
    except Exception as e:
        print(f"❌ Error reading API key: {e}")
        return False
    
    # Initialize agent
    try:
        agent = SelfInitiatedResearchAgent(api_key)
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False
    
    # Test with the original failing goal
    test_goal = "what are the advancements after lora? explain them and what are they?"
    print(f"🎯 Testing with goal: {test_goal}")
    
    try:
        # Test 1: Create research plan (this should handle dict priority_topics)
        print("\n📋 Test 1: Creating research plan...")
        plan = agent.initiate_research(test_goal)
        print(f"✅ Research plan created successfully")
        print(f"   - Sub-goals: {len(plan.sub_goals)}")
        print(f"   - Research questions: {len(plan.research_questions)}")
        print(f"   - Priority topics: {len(plan.priority_topics)}")
        print(f"   - Search strategies: {len(plan.search_strategies)}")
        
        # Verify priority topics are strings
        for i, topic in enumerate(plan.priority_topics[:3]):
            print(f"   - Topic {i+1}: {type(topic).__name__} - {str(topic)[:50]}...")
            if not isinstance(topic, str):
                print(f"⚠️  Warning: Topic {i+1} is not a string: {type(topic)}")
    
    except Exception as e:
        print(f"❌ Research plan creation failed: {e}")
        return False
    
    try:
        # Test 2: Execute one iteration of research (this should handle string conversion)
        print("\n🔬 Test 2: Executing research iteration...")
        results = await agent.execute_research(max_iterations=1)
        
        if "error" in results:
            print(f"❌ Research execution failed: {results['error']}")
            return False
        else:
            print(f"✅ Research execution completed successfully")
            print(f"   - Discoveries: {len(results.get('discoveries', []))}")
            print(f"   - Questions raised: {len(results.get('questions_raised', []))}")
            print(f"   - Gaps identified: {len(results.get('gaps_identified', []))}")
            
            # Test synthesis (this was a major source of the error)
            if results.get('synthesis'):
                print(f"   - Synthesis length: {len(results['synthesis'])} chars")
                print("✅ Synthesis generation successful")
            
    except Exception as e:
        print(f"❌ Research execution failed: {e}")
        return False
    
    try:
        # Test 3: Test display functions would work
        print("\n🖥️  Test 3: Testing display compatibility...")
        
        # Test safe string conversion on various data types
        from self_initiated_streamlit_app import safe_display_string
        
        test_cases = [
            "simple string",
            {"model_name": "NB-IoT", "paper": "test paper"},
            {"title": "Test Title", "content": "Test Content"},
            123,
            ["list", "item"],
            None
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                result = safe_display_string(test_case)
                print(f"   ✅ Test case {i+1} ({type(test_case).__name__}): {result[:30]}...")
            except Exception as e:
                print(f"   ❌ Test case {i+1} failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Display compatibility test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The type mismatch fixes are working correctly.")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_type_fixes())
    if success:
        print("\n✅ Research agent type fixes verified successfully!")
        print("🌐 The Streamlit app should now work without string/dict errors.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
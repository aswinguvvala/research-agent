#!/usr/bin/env python3
"""
SIMPLE CLEAN DEMO - Test with environment variables
No hardcoded keys - secure and production-ready!

Run: export OPENAI_API_KEY='your-key' && python simple_demo_clean.py
"""

import openai
import asyncio
import json
import os
from datetime import datetime

# Your API Key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your-key-here'")
    exit(1)

# Set up OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Cost tracking
total_cost = 0.0

async def simple_research(query: str) -> dict:
    """Simple research function using GPT-4o mini"""
    global total_cost
    
    print(f"🔍 Researching: {query}")
    
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",  # Cheapest model
                messages=[
                    {"role": "system", "content": "You are a research assistant. Provide concise, factual research."},
                    {"role": "user", "content": f"Research and analyze: {query}. Provide key findings in JSON format with 'findings' and 'summary' fields."}
                ],
                max_tokens=200,
                temperature=0.3
            )
        )
        
        # Cost calculation (GPT-4o mini pricing)
        if response.usage:
            input_cost = response.usage.prompt_tokens * 0.000150 / 1000
            output_cost = response.usage.completion_tokens * 0.000600 / 1000
            session_cost = input_cost + output_cost
            total_cost += session_cost
            
            print(f"💰 Session cost: ${session_cost:.4f} | Total: ${total_cost:.4f}")
        
        content = response.choices[0].message.content
        
        # Try to parse JSON, fallback to text
        try:
            return json.loads(content)
        except:
            return {"findings": [content], "summary": content[:100] + "..."}
            
    except Exception as e:
        return {"error": str(e), "findings": [], "summary": "Research failed"}

async def main():
    """Main demo function"""
    print("🧠 Simple Research Agent Demo")
    print("🔐 Using environment variables (secure)")
    print("💰 Cost-optimized with GPT-4o mini")
    print("-" * 50)
    
    # Demo queries
    queries = [
        "What are the benefits of renewable energy?",
        "How does machine learning work in healthcare?",
        "What is the future of electric vehicles?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}/3:")
        result = await simple_research(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Summary: {result.get('summary', 'No summary')}")
            findings = result.get('findings', [])
            if findings:
                print(f"🔍 Key findings: {len(findings)} items")
        
        if i < len(queries):
            await asyncio.sleep(1)  # Brief pause
    
    print(f"\n🎯 Demo completed!")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"📊 Average per query: ${total_cost/len(queries):.4f}")

if __name__ == "__main__":
    asyncio.run(main())
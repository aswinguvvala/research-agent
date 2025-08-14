#!/usr/bin/env python3
"""
Clean Demo Script - Uses Environment Variables for API Keys
Uses GPT-4o mini for maximum cost efficiency
"""

import asyncio
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_research_system import EnhancedResearchSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Cost-optimized configuration
config = {
    "model": "gpt-4o-mini",  # Cheapest model
    "max_tokens": 150,       # Conservative limit
    "chroma_db_path": "./enhanced_research_db",
    "cost_optimized": True
}

async def demo():
    """Run research system demo"""
    print("🚀 Starting Enhanced Research System Demo")
    print("💰 Cost-optimized with GPT-4o mini")
    print("-" * 50)
    
    # Initialize system
    system = EnhancedResearchSystem(OPENAI_API_KEY, config)
    
    # Demo query
    query = "What are the latest advances in quantum computing?"
    
    print(f"📋 Query: {query}")
    print("⏳ Processing... (this may take 30-60 seconds)")
    
    try:
        results = await system.research(query)
        
        print(f"\n✅ Research completed!")
        print(f"💰 Estimated cost: ${system.cost_optimizer.estimated_cost:.4f}")
        
        if 'synthesis' in results:
            print(f"\n📝 SYNTHESIS:")
            print(results['synthesis'])
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(demo())
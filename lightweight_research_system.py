"""
Lightweight Research System for Free Tier Deployment
Optimized version for Streamlit Cloud and low-memory environments.
Removes ChromaDB, reduces agents to 3 core specialists, and simplifies operations.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os

# Core dependencies only
import openai
import requests
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimpleResearchResult:
    """Simplified research result structure"""
    agent_type: str
    findings: List[str]
    confidence: float
    sources: List[str]
    summary: str
    timestamp: str


class CostOptimizedLLM:
    """Cost-optimized LLM wrapper using GPT-4o-mini"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        
        # Track usage
        self.total_requests = 0
        self.estimated_cost = 0.0
        
        # GPT-4o mini pricing
        self.cost_per_1k_input = 0.000150
        self.cost_per_1k_output = 0.000600
    
    async def request(self, messages: List[Dict], max_tokens: int = 150, context: str = "general") -> str:
        """Make cost-optimized request"""
        
        # Context-based token limits for memory efficiency
        context_limits = {
            "quick": 100,
            "summary": 150,
            "analysis": 200,
            "synthesis": 250
        }
        
        actual_max_tokens = min(max_tokens, context_limits.get(context, 150))
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=actual_max_tokens,
                    temperature=0.3,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
            )
            
            # Track usage
            self.total_requests += 1
            if hasattr(response, 'usage') and response.usage:
                cost = (response.usage.prompt_tokens / 1000 * self.cost_per_1k_input + 
                       response.usage.completion_tokens / 1000 * self.cost_per_1k_output)
                self.estimated_cost += cost
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            return ""
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            "requests": self.total_requests,
            "estimated_cost": self.estimated_cost,
            "cost_per_request": self.estimated_cost / max(self.total_requests, 1)
        }


class SimpleMemoryStore:
    """Simple in-memory storage to replace ChromaDB"""
    
    def __init__(self):
        self.research_history = []
        self.session_cache = {}
        self.max_items = 50  # Limit memory usage
    
    def add_research(self, session_id: str, query: str, results: List[SimpleResearchResult]):
        """Store research results"""
        item = {
            "id": str(uuid.uuid4())[:8],
            "session_id": session_id,
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.research_history.append(item)
        
        # Keep only recent items to manage memory
        if len(self.research_history) > self.max_items:
            self.research_history = self.research_history[-self.max_items:]
    
    def get_recent_research(self, limit: int = 5) -> List[Dict]:
        """Get recent research for context"""
        return self.research_history[-limit:] if self.research_history else []
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        return {
            "total_items": len(self.research_history),
            "memory_usage": "lightweight",
            "storage_type": "in_memory"
        }


class LightweightAgent:
    """Base lightweight agent with minimal overhead"""
    
    def __init__(self, agent_type: str, name: str, llm: CostOptimizedLLM):
        self.agent_type = agent_type
        self.name = name
        self.llm = llm
        self.tasks_completed = 0
    
    async def research(self, query: str) -> SimpleResearchResult:
        """Conduct research on the given query"""
        try:
            findings = await self._generate_findings(query)
            summary = await self._generate_summary(query, findings)
            
            self.tasks_completed += 1
            
            return SimpleResearchResult(
                agent_type=self.agent_type,
                findings=findings,
                confidence=0.8,
                sources=[f"{self.agent_type}_research"],
                summary=summary,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Research failed for {self.name}: {e}")
            return SimpleResearchResult(
                agent_type=self.agent_type,
                findings=[f"Research failed: {str(e)}"],
                confidence=0.0,
                sources=[],
                summary="Error occurred during research",
                timestamp=datetime.now().isoformat()
            )
    
    async def _generate_findings(self, query: str) -> List[str]:
        """Generate findings specific to agent type"""
        prompt = self._get_research_prompt(query)
        
        messages = [
            {"role": "system", "content": f"You are a {self.name}. Be concise and focus on key insights."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm.request(messages, context="analysis")
        
        # Parse response into findings list
        try:
            # Try JSON first
            findings = json.loads(response)
            if isinstance(findings, list):
                return findings[:5]  # Limit findings
        except:
            # Fallback: split by lines or sentences
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return lines[:5] if lines else ["Analysis completed"]
    
    async def _generate_summary(self, query: str, findings: List[str]) -> str:
        """Generate a concise summary"""
        if not findings:
            return "No significant findings to summarize."
        
        findings_text = ". ".join(findings[:3])  # Use top 3 findings
        
        prompt = f"Summarize these key findings for '{query}' in 2-3 sentences: {findings_text}"
        
        messages = [
            {"role": "system", "content": "You are a research summarizer. Be concise and clear."},
            {"role": "user", "content": prompt}
        ]
        
        return await self.llm.request(messages, context="summary")
    
    def _get_research_prompt(self, query: str) -> str:
        """Override in subclasses"""
        return f"Research and analyze: {query}. Provide 3-5 key insights."


class AcademicAgent(LightweightAgent):
    """Academic research specialist - lightweight version"""
    
    def __init__(self, llm: CostOptimizedLLM):
        super().__init__("academic", "Academic Research Specialist", llm)
    
    def _get_research_prompt(self, query: str) -> str:
        return f"""
        As an academic researcher, analyze: "{query}"
        
        Focus on:
        - Current research findings and consensus
        - Key methodologies and approaches  
        - Academic implications and significance
        - Research gaps or future directions
        
        Provide 4-5 key academic insights. Be scholarly but accessible.
        Return as JSON array: ["insight1", "insight2", "insight3", "insight4"]
        """


class TechnicalAgent(LightweightAgent):
    """Technical implementation specialist - lightweight version"""
    
    def __init__(self, llm: CostOptimizedLLM):
        super().__init__("technical", "Technical Implementation Specialist", llm)
    
    def _get_research_prompt(self, query: str) -> str:
        return f"""
        As a technical expert, analyze: "{query}"
        
        Focus on:
        - Implementation approaches and best practices
        - Technical challenges and solutions
        - Tools, frameworks, and technologies
        - Performance and scalability considerations
        
        Provide 4-5 key technical insights. Be practical and actionable.
        Return as JSON array: ["insight1", "insight2", "insight3", "insight4"]
        """


class BusinessAgent(LightweightAgent):
    """Business applications specialist - lightweight version"""
    
    def __init__(self, llm: CostOptimizedLLM):
        super().__init__("business", "Business Applications Specialist", llm)
    
    def _get_research_prompt(self, query: str) -> str:
        return f"""
        As a business analyst, evaluate: "{query}"
        
        Focus on:
        - Market opportunities and applications
        - Business value and ROI potential
        - Competitive landscape and positioning
        - Adoption challenges and success factors
        
        Provide 4-5 key business insights. Be strategic and market-focused.
        Return as JSON array: ["insight1", "insight2", "insight3", "insight4"]
        """


class LightweightResearchSystem:
    """Main lightweight research system for free tier deployment"""
    
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        
        # Initialize core components
        self.llm = CostOptimizedLLM(openai_api_key)
        self.memory = SimpleMemoryStore()
        
        # Initialize lightweight agents
        self.agents = {
            "academic": AcademicAgent(self.llm),
            "technical": TechnicalAgent(self.llm),
            "business": BusinessAgent(self.llm)
        }
        
        # System metrics
        self.sessions_completed = 0
        self.total_research_time = 0.0
        
        logger.info("Lightweight Research System initialized with 3 agents")
    
    async def research(self, query: str, selected_agents: List[str] = None) -> Dict[str, Any]:
        """Conduct research using selected agents"""
        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        # Use all agents if none specified
        if not selected_agents:
            selected_agents = list(self.agents.keys())
        
        logger.info(f"Starting research session {session_id}: {query}")
        logger.info(f"Using agents: {', '.join(selected_agents)}")
        
        try:
            # Run agents concurrently for speed
            tasks = []
            for agent_type in selected_agents:
                if agent_type in self.agents:
                    agent = self.agents[agent_type]
                    task = asyncio.create_task(agent.research(query))
                    tasks.append((agent_type, task))
            
            # Collect results
            results = []
            for agent_type, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=30.0)  # 30 second timeout per agent
                    results.append(result)
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {agent_type} timed out")
                    results.append(SimpleResearchResult(
                        agent_type=agent_type,
                        findings=["Research timed out"],
                        confidence=0.0,
                        sources=[],
                        summary="Agent timed out during research",
                        timestamp=datetime.now().isoformat()
                    ))
            
            # Generate cross-agent synthesis
            synthesis = await self._synthesize_results(query, results)
            
            # Calculate session metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Store in memory
            self.memory.add_research(session_id, query, results)
            
            # Update system metrics
            self.sessions_completed += 1
            self.total_research_time += duration
            
            # Compile final response
            response = {
                "session_id": session_id,
                "query": query,
                "results_by_agent": {r.agent_type: {
                    "findings": r.findings,
                    "summary": r.summary,
                    "confidence": r.confidence,
                    "sources": r.sources
                } for r in results},
                "synthesis": synthesis,
                "metadata": {
                    "agents_used": selected_agents,
                    "duration_seconds": duration,
                    "timestamp": end_time.isoformat(),
                    "results_count": len(results)
                },
                "system_stats": self.get_system_stats()
            }
            
            logger.info(f"Research session {session_id} completed in {duration:.1f}s")
            return response
            
        except Exception as e:
            logger.error(f"Research session {session_id} failed: {e}")
            return {
                "session_id": session_id,
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _synthesize_results(self, query: str, results: List[SimpleResearchResult]) -> str:
        """Synthesize results from multiple agents"""
        if not results:
            return "No results to synthesize."
        
        # Combine all findings
        all_findings = []
        for result in results:
            all_findings.extend(result.findings[:3])  # Top 3 from each agent
        
        if not all_findings:
            return "No findings available for synthesis."
        
        # Create synthesis prompt
        findings_text = ". ".join(all_findings)
        prompt = f"""
        Synthesize research findings for: "{query}"
        
        Key findings from {len(results)} specialist agents:
        {findings_text}
        
        Create a comprehensive 3-4 sentence synthesis that:
        1. Identifies common themes and patterns
        2. Highlights the most significant insights
        3. Provides actionable conclusions
        
        Be concise and focused on practical value.
        """
        
        messages = [
            {"role": "system", "content": "You are a research synthesizer. Create coherent, valuable syntheses."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            return await self.llm.request(messages, context="synthesis")
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Synthesis of {len(results)} agent results completed with {len(all_findings)} key findings."
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        memory_stats = self.memory.get_stats()
        llm_stats = self.llm.get_stats()
        
        return {
            "sessions_completed": self.sessions_completed,
            "avg_session_duration": self.total_research_time / max(self.sessions_completed, 1),
            "agents_available": len(self.agents),
            "memory_system": memory_stats,
            "llm_usage": llm_stats,
            "system_type": "lightweight",
            "deployment_ready": True
        }
    
    def get_recent_research(self, limit: int = 5) -> List[Dict]:
        """Get recent research history"""
        return self.memory.get_recent_research(limit)


# Demo function for testing
async def demo_lightweight_system():
    """Demo the lightweight research system"""
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🚀 Initializing Lightweight Research System...")
    system = LightweightResearchSystem(api_key)
    
    # Example queries
    test_queries = [
        "What is the current state of artificial intelligence in healthcare?",
        "How can small businesses implement automation effectively?",
        "What are the emerging trends in sustainable technology?"
    ]
    
    for query in test_queries:
        print(f"\n📋 Researching: {query}")
        print("⏳ This should take 10-30 seconds...")
        
        results = await system.research(query)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            continue
        
        print(f"\n✅ Research completed in {results['metadata']['duration_seconds']:.1f}s")
        print(f"💰 Estimated cost: ${system.llm.estimated_cost:.4f}")
        
        # Display results
        print("\n📊 AGENT FINDINGS:")
        for agent_type, data in results["results_by_agent"].items():
            print(f"\n{agent_type.upper()}:")
            print(f"Summary: {data['summary']}")
            print(f"Confidence: {data['confidence']:.0%}")
        
        print(f"\n🔄 SYNTHESIS:")
        print(results["synthesis"])
        
        # Ask if user wants to continue
        if query != test_queries[-1]:  # Not the last query
            continue_demo = input("\n⏭️  Continue with next query? (y/n): ").strip().lower()
            if continue_demo != 'y':
                break
    
    # Display final stats
    print("\n" + "="*50)
    print("📈 FINAL SYSTEM STATS")
    print("="*50)
    stats = system.get_system_stats()
    print(f"Sessions completed: {stats['sessions_completed']}")
    print(f"Average duration: {stats['avg_session_duration']:.1f}s")
    print(f"Total LLM requests: {stats['llm_usage']['requests']}")
    print(f"Total estimated cost: ${stats['llm_usage']['estimated_cost']:.4f}")
    print(f"Cost per session: ${stats['llm_usage']['cost_per_request']:.4f}")
    
    print("\n✅ Lightweight Research System demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_lightweight_system())


# Export main components
__all__ = [
    'LightweightResearchSystem',
    'SimpleResearchResult',
    'CostOptimizedLLM',
    'SimpleMemoryStore'
]
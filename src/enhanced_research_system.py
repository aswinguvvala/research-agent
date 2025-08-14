"""
Enhanced Multi-Agent Research System
Complete integration of all advanced components: multi-agent architecture,
vector knowledge system, modern APIs, and advanced reasoning.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import sys
import os

# Import all components with robust fallback to support running without package context
try:
    # Package-relative imports (when executed as part of the src package)
    from .multi_agent_research_system import (
        MultiAgentResearchSystem, ResearchTask, AgentMessage, MessageType
    )
    from .specialized_agents import (
        AcademicSpecialist, TechnicalSpecialist, BusinessSpecialist, SocialSpecialist
    )
    from .quality_agents import FactChecker, SynthesisAgent
    from .vector_knowledge_system import VectorKnowledgeSystem, KnowledgeItem
    from .modern_api_sources import ModernAPIManager
    from .advanced_reasoning import ChainOfThoughtEngine, ReasoningType
except ImportError:
    # Direct script execution fallback
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from multi_agent_research_system import (
        MultiAgentResearchSystem, ResearchTask, AgentMessage, MessageType
    )
    from specialized_agents import (
        AcademicSpecialist, TechnicalSpecialist, BusinessSpecialist, SocialSpecialist
    )
    from quality_agents import FactChecker, SynthesisAgent
    from vector_knowledge_system import VectorKnowledgeSystem, KnowledgeItem
    from modern_api_sources import ModernAPIManager
    from advanced_reasoning import ChainOfThoughtEngine, ReasoningType

logger = logging.getLogger(__name__)


class EnhancedResearchSystem:
    """Complete enhanced research system with all modern capabilities"""
    
    def __init__(self, openai_api_key: str, config: Dict[str, Any] = None):
        self.openai_api_key = openai_api_key
        self.config = config or {}
        
        # Core multi-agent system
        self.multi_agent_system = MultiAgentResearchSystem(openai_api_key)
        
        # Vector knowledge system
        self.knowledge_system = VectorKnowledgeSystem(
            openai_api_key, 
            persist_directory=self.config.get("chroma_db_path", "./enhanced_research_db")
        )
        
        # Modern API manager
        api_keys = self.config.get("api_keys", {})
        self.api_manager = ModernAPIManager(api_keys)
        
        # Advanced reasoning engine
        self.reasoning_engine = ChainOfThoughtEngine(openai_api_key)
        
        # System state
        self.is_initialized = False
        self.current_session = None
        self.session_history = []
        
        # Performance metrics
        self.metrics = {
            "total_sessions": 0,
            "successful_sessions": 0,
            "total_agents_used": 0,
            "total_knowledge_items": 0,
            "avg_session_duration": 0.0
        }
    
    async def initialize_system(self):
        """Initialize all system components"""
        if self.is_initialized:
            return
        
        logger.info("Initializing Enhanced Research System...")
        
        try:
            # Initialize specialized agents
            await self._setup_specialized_agents()
            
            # Initialize quality agents
            await self._setup_quality_agents()
            
            # Start multi-agent system (don't await - let it run in background)
            asyncio.create_task(self._start_background_systems())
            
            # Wait for knowledge system to initialize
            await self._wait_for_knowledge_system()
            
            self.is_initialized = True
            logger.info("Enhanced Research System initialized successfully")
            
            # Print system status
            await self._print_system_status()
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Research System: {e}")
            raise
    
    async def _setup_specialized_agents(self):
        """Set up all specialized research agents"""
        agents = [
            AcademicSpecialist(self.openai_api_key),
            TechnicalSpecialist(self.openai_api_key),
            BusinessSpecialist(self.openai_api_key),
            SocialSpecialist(self.openai_api_key)
        ]
        
        for agent in agents:
            self.multi_agent_system.add_agent(agent)
        
        logger.info(f"Added {len(agents)} specialized research agents")
    
    async def _setup_quality_agents(self):
        """Set up quality assurance agents"""
        quality_agents = [
            FactChecker(self.openai_api_key),
            SynthesisAgent(self.openai_api_key)
        ]
        
        for agent in quality_agents:
            self.multi_agent_system.add_agent(agent)
        
        logger.info(f"Added {len(quality_agents)} quality assurance agents")
    
    async def _start_background_systems(self):
        """Start background systems"""
        try:
            # This will run the multi-agent system in the background
            await self.multi_agent_system.start_system()
        except Exception as e:
            logger.error(f"Background systems error: {e}")
    
    async def _wait_for_knowledge_system(self):
        """Wait for knowledge system to initialize"""
        max_attempts = 30  # 30 seconds max wait
        for attempt in range(max_attempts):
            if self.knowledge_system.is_initialized:
                return
            await asyncio.sleep(1)
        
        logger.warning("Knowledge system initialization timeout - proceeding with fallback")
    
    async def _print_system_status(self):
        """Print current system status"""
        agent_count = len(self.multi_agent_system.agents)
        knowledge_stats = self.knowledge_system.get_system_stats()
        
        print("\n" + "="*70)
        print("🤖 ENHANCED MULTI-AGENT RESEARCH SYSTEM STATUS")
        print("="*70)
        print(f"✅ System initialized: {self.is_initialized}")
        print(f"🔬 Active agents: {agent_count}")
        print(f"🧠 Knowledge system: {knowledge_stats['system_type']}")
        print(f"📊 Knowledge items: {knowledge_stats['total_items']}")
        print(f"🔍 API sources: Semantic Scholar, OpenAlex, Enhanced arXiv, News")
        print(f"⚡ Advanced reasoning: Chain-of-thought enabled")
        print("="*70)
    
    async def conduct_enhanced_research(self, research_goal: str, max_duration: int = 600) -> Dict[str, Any]:
        """Conduct comprehensive research using all enhanced capabilities"""
        if not self.is_initialized:
            await self.initialize_system()
        
        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        logger.info(f"🚀 Starting enhanced research session {session_id}")
        logger.info(f"🎯 Research goal: {research_goal}")
        
        try:
            # Phase 1: Advanced reasoning about the research problem
            logger.info("🧠 Phase 1: Advanced problem reasoning...")
            reasoning_chain = await self.reasoning_engine.reason_through_problem(
                research_goal, 
                context={"session_id": session_id, "enhanced_system": True},
                reasoning_type=ReasoningType.STRATEGIC
            )
            
            # Phase 2: Gather contextual knowledge
            logger.info("📚 Phase 2: Gathering contextual knowledge...")
            context_items = await self.knowledge_system.get_research_context(research_goal)
            
            # Phase 3: Modern API search across sources
            logger.info("🔍 Phase 3: Comprehensive source search...")
            api_results = await self.api_manager.comprehensive_search(research_goal)
            
            # Phase 4: Multi-agent research coordination
            logger.info("🤖 Phase 4: Multi-agent research execution...")
            agent_results = await self.multi_agent_system.conduct_research(research_goal, max_duration)
            
            # Phase 5: Knowledge integration and synthesis
            logger.info("🔀 Phase 5: Knowledge integration...")
            integrated_results = await self._integrate_all_results(
                reasoning_chain, context_items, api_results, agent_results, research_goal
            )
            
            # Phase 6: Final quality assurance
            logger.info("✅ Phase 6: Quality assurance...")
            qa_results = await self._perform_quality_assurance(integrated_results, research_goal)
            
            # Phase 7: Knowledge storage
            logger.info("💾 Phase 7: Storing new knowledge...")
            await self._store_session_knowledge(integrated_results, session_id)
            
            # Compile final results
            final_results = {
                "session_id": session_id,
                "research_goal": research_goal,
                "reasoning_analysis": {
                    "chain_id": reasoning_chain.chain_id,
                    "final_conclusion": reasoning_chain.final_conclusion,
                    "confidence": reasoning_chain.overall_confidence,
                    "reasoning_steps": len(reasoning_chain.reasoning_steps)
                },
                "knowledge_context": {
                    "items_found": len(context_items),
                    "relevance_scores": [item.get("similarity_score", 0) for item in context_items]
                },
                "api_sources": {
                    source: len(results) for source, results in api_results.items()
                },
                "agent_research": agent_results,
                "integrated_findings": integrated_results,
                "quality_assurance": qa_results,
                "session_metrics": {
                    "duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "agents_used": len(self.multi_agent_system.agents),
                    "knowledge_items_added": integrated_results.get("knowledge_items_added", 0),
                    "total_sources": sum(len(results) for results in api_results.values())
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Update system metrics
            self._update_metrics(final_results, True)
            
            # Store session
            self.current_session = final_results
            self.session_history.append(final_results)
            
            logger.info(f"✨ Enhanced research session {session_id} completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced research session {session_id} failed: {e}")
            
            # Store failed session info
            failed_results = {
                "session_id": session_id,
                "research_goal": research_goal,
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
            
            self._update_metrics(failed_results, False)
            return failed_results
    
    async def _integrate_all_results(self, reasoning_chain, context_items, api_results, 
                                   agent_results, research_goal) -> Dict[str, Any]:
        """Integrate results from all sources into comprehensive findings"""
        
        # Prepare data for synthesis agent
        all_findings = []
        
        # Add reasoning insights
        if reasoning_chain:
            all_findings.append({
                "source": "reasoning_engine",
                "finding": reasoning_chain.final_conclusion,
                "confidence": reasoning_chain.overall_confidence,
                "type": "reasoning_analysis"
            })
        
        # Add contextual knowledge
        for item in context_items[:5]:  # Top 5 contextual items
            all_findings.append({
                "source": "knowledge_base",
                "finding": item.get("content", "")[:300],
                "confidence": item.get("similarity_score", 0.5),
                "type": "contextual_knowledge"
            })
        
        # Add API results
        for source, results in api_results.items():
            for result in results[:3]:  # Top 3 from each source
                finding_text = ""
                if isinstance(result, dict):
                    finding_text = result.get("title", "") + " " + result.get("abstract", result.get("summary", ""))
                else:
                    finding_text = str(result)
                
                all_findings.append({
                    "source": f"api_{source}",
                    "finding": finding_text[:300],
                    "confidence": 0.8,
                    "type": "external_source"
                })
        
        # Add agent results
        if agent_results.get("key_findings"):
            for finding in agent_results["key_findings"][:10]:
                all_findings.append({
                    "source": "multi_agent_research",
                    "finding": str(finding)[:300],
                    "confidence": 0.8,
                    "type": "agent_research"
                })
        
        # Create synthesis using all findings
        synthesis_prompt = f"""
        Integrate and synthesize comprehensive research findings for: "{research_goal}"
        
        Total findings: {len(all_findings)}
        Sources: {set(f["source"] for f in all_findings)}
        
        Key findings:
        {chr(10).join([f"- {f['finding'][:150]}..." for f in all_findings[:15]])}
        
        Create a comprehensive synthesis that:
        1. Integrates insights from all sources
        2. Identifies key themes and patterns
        3. Highlights contradictions or uncertainties
        4. Provides actionable conclusions
        5. Notes areas needing further research
        
        Provide synthesis as JSON:
        {{
            "executive_summary": "concise overview of key findings",
            "main_insights": ["insight1", "insight2", "insight3"],
            "cross_source_patterns": ["pattern1", "pattern2"],
            "confidence_assessment": 0.0-1.0,
            "research_gaps": ["gap1", "gap2"],
            "actionable_recommendations": ["rec1", "rec2", "rec3"],
            "synthesis_methodology": "how the integration was performed"
        }}
        """
        
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert research synthesizer. Create comprehensive, accurate syntheses. Return valid JSON only."},
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
            )
            
            synthesis = json.loads(response.choices[0].message.content)
            
            return {
                "total_findings": len(all_findings),
                "sources_integrated": list(set(f["source"] for f in all_findings)),
                "synthesis": synthesis,
                "raw_findings": all_findings,
                "integration_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Integration synthesis failed: {e}")
            return {
                "total_findings": len(all_findings),
                "sources_integrated": list(set(f["source"] for f in all_findings)),
                "synthesis": {
                    "executive_summary": f"Integration of {len(all_findings)} findings from multiple sources",
                    "main_insights": ["Cross-source research conducted", "Multiple perspectives gathered"],
                    "confidence_assessment": 0.6,
                    "synthesis_methodology": "Automated integration with fallback"
                },
                "raw_findings": all_findings[:10],  # Limit for storage
                "integration_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _perform_quality_assurance(self, integrated_results: Dict[str, Any], research_goal: str) -> Dict[str, Any]:
        """Perform quality assurance on integrated results"""
        
        try:
            synthesis = integrated_results.get("synthesis", {})
            main_insights = synthesis.get("main_insights", [])
            
            # Simple quality checks
            qa_results = {
                "completeness_score": min(len(main_insights) / 5.0, 1.0),  # Expect at least 5 insights
                "source_diversity": len(integrated_results.get("sources_integrated", [])),
                "confidence_consistency": synthesis.get("confidence_assessment", 0.5),
                "synthesis_quality": "high" if len(str(synthesis)) > 500 else "medium",
                "timestamp": datetime.now().isoformat()
            }
            
            # Overall QA score
            qa_score = (
                qa_results["completeness_score"] * 0.3 +
                min(qa_results["source_diversity"] / 5.0, 1.0) * 0.3 +
                qa_results["confidence_consistency"] * 0.4
            )
            
            qa_results["overall_qa_score"] = qa_score
            qa_results["quality_level"] = "high" if qa_score > 0.8 else "medium" if qa_score > 0.6 else "needs_improvement"
            
            return qa_results
            
        except Exception as e:
            logger.error(f"Quality assurance failed: {e}")
            return {
                "overall_qa_score": 0.5,
                "quality_level": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _store_session_knowledge(self, integrated_results: Dict[str, Any], session_id: str) -> int:
        """Store new knowledge from the session"""
        try:
            findings = integrated_results.get("raw_findings", [])
            return await self.knowledge_system.batch_add_research_findings(findings, session_id)
        except Exception as e:
            logger.error(f"Failed to store session knowledge: {e}")
            return 0
    
    def _update_metrics(self, results: Dict[str, Any], success: bool):
        """Update system performance metrics"""
        self.metrics["total_sessions"] += 1
        if success:
            self.metrics["successful_sessions"] += 1
        
        duration = results.get("session_metrics", {}).get("duration_seconds", 0)
        if duration > 0:
            # Update running average
            total_duration = self.metrics["avg_session_duration"] * (self.metrics["total_sessions"] - 1) + duration
            self.metrics["avg_session_duration"] = total_duration / self.metrics["total_sessions"]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        knowledge_stats = self.knowledge_system.get_system_stats()
        
        return {
            **self.metrics,
            "success_rate": self.metrics["successful_sessions"] / max(self.metrics["total_sessions"], 1),
            "knowledge_system": knowledge_stats,
            "agents_available": len(self.multi_agent_system.agents),
            "last_updated": datetime.now().isoformat()
        }
    
    async def interactive_research_session(self):
        """Run an interactive research session"""
        if not self.is_initialized:
            await self.initialize_system()
        
        print("\n" + "="*70)
        print("🤖 ENHANCED MULTI-AGENT RESEARCH SYSTEM")
        print("="*70)
        print("Welcome to the most advanced AI research system!")
        print("This system uses multiple specialized AI agents, vector knowledge,")
        print("modern APIs, and chain-of-thought reasoning for comprehensive research.\n")
        
        while True:
            try:
                research_goal = input("🎯 Enter your research question (or 'quit' to exit): ").strip()
                
                if research_goal.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not research_goal:
                    print("Please enter a research question.")
                    continue
                
                print(f"\n🚀 Starting enhanced research for: {research_goal}")
                print("This may take 2-5 minutes depending on complexity...\n")
                
                # Conduct research
                results = await self.conduct_enhanced_research(research_goal)
                
                if results.get("status") == "failed":
                    print(f"❌ Research failed: {results.get('error', 'Unknown error')}")
                    continue
                
                # Display results
                self._display_research_results(results)
                
                # Ask for follow-up
                follow_up = input("\n💬 Would you like to ask a follow-up question about these results? (y/n): ").strip().lower()
                if follow_up in ['y', 'yes']:
                    continue
                else:
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Research session interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        # Display final metrics
        print("\n" + "="*50)
        print("📊 SESSION METRICS")
        print("="*50)
        metrics = self.get_system_metrics()
        print(f"Total sessions: {metrics['total_sessions']}")
        print(f"Success rate: {metrics['success_rate']:.1%}")
        print(f"Average duration: {metrics['avg_session_duration']:.1f}s")
        print(f"Knowledge items: {metrics['knowledge_system']['total_items']}")
        
        print("\n👋 Thank you for using the Enhanced Research System!")
    
    def _display_research_results(self, results: Dict[str, Any]):
        """Display research results in a user-friendly format"""
        print("\n" + "="*70)
        print("📋 RESEARCH RESULTS")
        print("="*70)
        
        # Session info
        session_id = results["session_id"]
        duration = results["session_metrics"]["duration_seconds"]
        print(f"Session ID: {session_id}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Sources used: {results['session_metrics']['total_sources']}")
        
        # Reasoning analysis
        reasoning = results.get("reasoning_analysis", {})
        print(f"\n🧠 REASONING ANALYSIS:")
        print(f"Conclusion: {reasoning.get('final_conclusion', 'N/A')}")
        print(f"Confidence: {reasoning.get('confidence', 0):.1%}")
        
        # Synthesis
        synthesis = results.get("integrated_findings", {}).get("synthesis", {})
        print(f"\n📝 KEY FINDINGS:")
        print(synthesis.get("executive_summary", "No summary available"))
        
        insights = synthesis.get("main_insights", [])
        if insights:
            print(f"\n💡 MAIN INSIGHTS:")
            for i, insight in enumerate(insights[:5], 1):
                print(f"{i}. {insight}")
        
        recommendations = synthesis.get("actionable_recommendations", [])
        if recommendations:
            print(f"\n🎯 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec}")
        
        # Quality assessment
        qa = results.get("quality_assurance", {})
        print(f"\n✅ QUALITY ASSESSMENT:")
        print(f"Overall score: {qa.get('overall_qa_score', 0):.1%}")
        print(f"Quality level: {qa.get('quality_level', 'unknown').title()}")
        
        print("="*70)
    
    async def close_system(self):
        """Clean shutdown of all system components"""
        try:
            await self.api_manager.close_all()
            logger.info("Enhanced Research System shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Demo function for standalone execution
async def demo_enhanced_research_system():
    """Demo the enhanced research system"""
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Configuration
    config = {
        "chroma_db_path": "./demo_research_db",
        "api_keys": {
            "semantic_scholar": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            "news_api": os.getenv("NEWS_API_KEY"),
            "email": "demo@enhanced-research.ai"
        }
    }
    
    # Create and run system
    system = EnhancedResearchSystem(openai_api_key, config)
    
    try:
        await system.interactive_research_session()
    finally:
        await system.close_system()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('enhanced_research.log')
        ]
    )
    
    # Run demo
    asyncio.run(demo_enhanced_research_system())


# Export the enhanced system
__all__ = ['EnhancedResearchSystem']
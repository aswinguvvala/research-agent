"""
Autonomous Research Agent
A truly intelligent research agent that uses GPT-4o mini for all decision-making.
No hardcoded patterns, search terms, or thresholds - purely AI-driven research.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import AI components
from ai_query_analyzer import AIQueryAnalyzer, QueryAnalysis
from ai_relevance_judge import AIRelevanceJudge, RelevanceAssessment
from ai_honest_synthesizer import AIHonestSynthesizer, SynthesisResult
from ai_search_strategist import AISearchStrategist, AdaptiveSearchResult

# Import citation manager for bibliography
try:
    from citation_manager import CitationManager, Source
except ImportError:
    logger.warning("Citation manager not available")
    CitationManager = None
    Source = None

logger = logging.getLogger(__name__)


@dataclass
class AutonomousResearchResult:
    """Complete autonomous research result."""
    query: str
    research_time: float
    confidence_score: float
    
    # AI Analysis Results
    query_analysis: Dict[str, Any]
    search_results: Dict[str, Any]
    relevance_evaluations: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    synthesis_result: Dict[str, Any]
    
    # Final Outputs
    synthesis: str
    sources: List[Dict[str, Any]]
    bibliography: str
    
    # Quality Assessment
    quality_assessment: Dict[str, Any]
    research_notes: List[str]
    recommendations: List[str]
    
    timestamp: str


class AutonomousResearchAgent:
    """
    Fully autonomous research agent powered by GPT-4o mini.
    Makes all decisions intelligently without hardcoded assumptions.
    """
    
    def __init__(self, openai_api_key: str, max_sources: int = 10, debug_mode: bool = False):
        """
        Initialize the autonomous research agent.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o mini
            max_sources: Maximum number of sources to process
            debug_mode: Enable detailed logging
        """
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_api_key = openai_api_key
        self.max_sources = max_sources
        self.debug_mode = debug_mode
        
        # Initialize AI components
        self.query_analyzer = AIQueryAnalyzer(openai_api_key)
        self.relevance_judge = AIRelevanceJudge(openai_api_key)
        self.synthesizer = AIHonestSynthesizer(openai_api_key)
        self.search_strategist = AISearchStrategist(openai_api_key)
        
        # Initialize citation manager if available
        self.citation_manager = CitationManager() if CitationManager else None
        
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        logger.info("🤖 Autonomous Research Agent initialized - fully AI-driven")
    
    async def conduct_autonomous_research(self, query: str, citation_style: str = "apa") -> AutonomousResearchResult:
        """
        Conduct completely autonomous research using AI for all decisions.
        
        Args:
            query: The research question
            citation_style: Citation format (apa, mla, ieee)
            
        Returns:
            AutonomousResearchResult with comprehensive findings
        """
        start_time = time.time()
        research_notes = []
        
        logger.info(f"🤖 Starting autonomous research: {query}")
        
        try:
            # Phase 1: AI Query Analysis
            logger.info("🧠 Phase 1: AI Query Understanding")
            query_analysis = await self.query_analyzer.analyze_query(query)
            research_notes.append(f"Query analyzed as {query_analysis.research_type} in {query_analysis.domain_detected} domain")
            
            if self.debug_mode:
                logger.debug(f"🧠 Query Intent: {query_analysis.research_intent}")
                logger.debug(f"🧠 Key Concepts: {query_analysis.key_concepts}")
                logger.debug(f"🧠 Generated {len(query_analysis.search_strategies)} search strategies")
            
            # Phase 2: AI-Driven Adaptive Search
            logger.info("🎯 Phase 2: AI Adaptive Search Execution")
            search_results = await self.search_strategist.execute_adaptive_search(
                query=query,
                query_analysis=vars(query_analysis),
                max_iterations=3
            )
            research_notes.append(f"Executed {search_results.total_strategies_executed} search strategies")
            research_notes.append(f"Found {search_results.total_sources_found} potential sources")
            
            # Collect all sources from search executions
            all_sources = []
            for execution in search_results.search_executions:
                all_sources.extend(execution.sources_found)
            
            # Remove duplicates based on title
            unique_sources = self._deduplicate_sources(all_sources)
            logger.info(f"🎯 Found {len(unique_sources)} unique sources")
            
            if not unique_sources:
                return self._create_no_sources_result(query, query_analysis, search_results, start_time, research_notes)
            
            # Phase 3: AI Relevance Evaluation
            logger.info("🔍 Phase 3: AI Relevance Assessment")
            relevance_evaluations = await self.relevance_judge.batch_evaluate_papers(
                query=query,
                papers=unique_sources[:self.max_sources]
            )
            
            relevant_papers = [(p, a) for p, a in relevance_evaluations if a.is_relevant]
            maybe_relevant = [(p, a) for p, a in relevance_evaluations if a.recommendation == "maybe"]
            
            logger.info(f"🔍 Relevance Assessment: {len(relevant_papers)} relevant, {len(maybe_relevant)} maybe relevant")
            research_notes.append(f"AI identified {len(relevant_papers)} highly relevant sources")
            
            # Phase 4: Citation Management
            logger.info("📚 Phase 4: Citation Processing")
            processed_sources = self._process_citations(relevance_evaluations, citation_style)
            
            # Phase 5: AI Honest Synthesis
            logger.info("📝 Phase 5: AI Honest Synthesis")
            synthesis_result = await self.synthesizer.synthesize_findings(
                query=query,
                paper_evaluations=relevance_evaluations,
                search_metadata={
                    "strategies_used": [e.strategy_name for e in search_results.search_executions],
                    "total_sources_searched": search_results.total_sources_found
                }
            )
            
            logger.info(f"📝 Synthesis completed: {synthesis_result.synthesis_type} (confidence: {synthesis_result.confidence_score:.2f})")
            research_notes.append(f"Synthesis type: {synthesis_result.synthesis_type}")
            
            # Phase 6: Quality Assessment
            logger.info("📊 Phase 6: Final Quality Assessment")
            quality_assessment = self._assess_research_quality(
                query_analysis, search_results, relevance_evaluations, synthesis_result
            )
            
            # Generate bibliography
            bibliography = self._generate_bibliography(processed_sources, citation_style)
            
            # Calculate final metrics
            end_time = time.time()
            research_time = round(end_time - start_time, 2)
            confidence_score = self._calculate_confidence_score(synthesis_result, relevance_evaluations)
            
            # Compile final recommendations
            final_recommendations = self._compile_final_recommendations(
                query_analysis, search_results, synthesis_result, quality_assessment
            )
            
            logger.info(f"✅ Autonomous research completed in {research_time}s (confidence: {confidence_score:.3f})")
            
            # Safe conversion to dict for complex objects
            def safe_vars(obj):
                """Safely convert object to dict."""
                if hasattr(obj, '__dict__'):
                    return vars(obj)
                elif hasattr(obj, '_asdict'):  # namedtuple
                    return obj._asdict()
                elif isinstance(obj, dict):
                    return obj
                else:
                    # For simple objects, create a basic dict representation
                    return {
                        'type': type(obj).__name__,
                        'str_representation': str(obj)
                    }
            
            return AutonomousResearchResult(
                query=query,
                research_time=research_time,
                confidence_score=confidence_score,
                
                # AI Results (with safe conversion)
                query_analysis=safe_vars(query_analysis),
                search_results=safe_vars(search_results),
                relevance_evaluations=[(safe_vars(p), safe_vars(a)) for p, a in relevance_evaluations],
                synthesis_result=safe_vars(synthesis_result),
                
                # Final Outputs
                synthesis=synthesis_result.synthesis_text,
                sources=processed_sources,
                bibliography=bibliography,
                
                # Quality Assessment
                quality_assessment=quality_assessment,
                research_notes=research_notes,
                recommendations=final_recommendations,
                
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Autonomous research failed: {e}")
            return self._create_error_result(query, str(e), start_time)
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources based on title similarity."""
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            title = source.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_sources.append(source)
        
        return unique_sources
    
    def _process_citations(self, 
                          evaluations: List[Tuple[Dict[str, Any], RelevanceAssessment]], 
                          citation_style: str) -> List[Dict[str, Any]]:
        """Process sources for citation management."""
        processed_sources = []
        
        if not self.citation_manager:
            logger.warning("Citation manager not available")
            return [p for p, a in evaluations if a.is_relevant]
        
        self.citation_manager.clear_sources()
        
        for paper, assessment in evaluations:
            if assessment.is_relevant or assessment.recommendation == "maybe":
                try:
                    # Create citation source
                    citation_source = Source(
                        title=paper.get("title", "Unknown Title"),
                        authors=paper.get("authors", []),
                        year=self._extract_year(paper.get("published", "")),
                        url=paper.get("url", ""),
                        doi=paper.get("doi"),
                        journal=paper.get("journal"),
                        source_type=paper.get("source_type", "article")
                    )
                    
                    citation_num = self.citation_manager.add_source(citation_source)
                    
                    # Add citation info to paper
                    processed_paper = {
                        **paper,
                        "citation_number": citation_num,
                        "relevance_score": assessment.relevance_score,
                        "is_relevant": assessment.is_relevant
                    }
                    
                    processed_sources.append(processed_paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to process citation for {paper.get('title', 'Unknown')}: {e}")
                    # Add without citation processing
                    processed_sources.append({
                        **paper,
                        "citation_number": len(processed_sources) + 1,
                        "relevance_score": assessment.relevance_score,
                        "is_relevant": assessment.is_relevant
                    })
        
        return processed_sources
    
    def _generate_bibliography(self, sources: List[Dict[str, Any]], citation_style: str) -> str:
        """Generate bibliography in specified format."""
        if not self.citation_manager:
            return "Bibliography generation unavailable"
        
        try:
            return self.citation_manager.generate_bibliography(citation_style)
        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return f"Bibliography error: {str(e)}"
    
    def _assess_research_quality(self, 
                               query_analysis: QueryAnalysis,
                               search_results: AdaptiveSearchResult,
                               relevance_evaluations: List[Tuple[Dict[str, Any], RelevanceAssessment]],
                               synthesis_result: SynthesisResult) -> Dict[str, Any]:
        """Assess overall research quality."""
        relevant_count = sum(1 for _, a in relevance_evaluations if a.is_relevant)
        total_count = len(relevance_evaluations)
        
        relevance_rate = relevant_count / total_count if total_count > 0 else 0
        search_success_rate = search_results.total_sources_found / max(len(search_results.search_executions), 1)
        
        # Overall quality scoring
        quality_score = (
            query_analysis.confidence * 0.2 +
            search_success_rate * 0.2 +
            relevance_rate * 0.3 +
            synthesis_result.confidence_score * 0.3
        )
        
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "acceptable"
        else:
            quality_level = "poor"
        
        return {
            "overall_score": quality_score,
            "quality_level": quality_level,
            "relevant_sources": relevant_count,
            "total_sources": total_count,
            "relevance_rate": relevance_rate,
            "search_success_rate": search_success_rate,
            "synthesis_confidence": synthesis_result.confidence_score,
            "synthesis_type": synthesis_result.synthesis_type
        }
    
    def _calculate_confidence_score(self, 
                                  synthesis_result: SynthesisResult,
                                  relevance_evaluations: List[Tuple[Dict[str, Any], RelevanceAssessment]]) -> float:
        """Calculate overall confidence in research results."""
        if not relevance_evaluations:
            return 0.0
        
        # Average relevance confidence
        relevance_confidence = sum(a.confidence for _, a in relevance_evaluations) / len(relevance_evaluations)
        
        # Weight synthesis confidence higher
        overall_confidence = (
            synthesis_result.confidence_score * 0.7 +
            relevance_confidence * 0.3
        )
        
        return round(overall_confidence, 3)
    
    def _compile_final_recommendations(self, 
                                     query_analysis: QueryAnalysis,
                                     search_results: AdaptiveSearchResult,
                                     synthesis_result: SynthesisResult,
                                     quality_assessment: Dict[str, Any]) -> List[str]:
        """Compile final recommendations for the user."""
        recommendations = []
        
        # Add synthesis recommendations
        recommendations.extend(synthesis_result.recommendations)
        
        # Add search recommendations
        recommendations.extend(search_results.final_recommendations)
        
        # Add quality-based recommendations
        if quality_assessment["quality_level"] == "poor":
            recommendations.append("Research quality is low - consider alternative search approaches")
        elif quality_assessment["quality_level"] == "acceptable":
            recommendations.append("Research provides basic information but could be enhanced")
        
        # Add search improvement suggestions
        if synthesis_result.search_suggestions:
            recommendations.extend(synthesis_result.search_suggestions)
        
        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]
    
    def _extract_year(self, date_string: str) -> str:
        """Extract year from date string."""
        if not date_string:
            return str(datetime.now().year)
        
        import re
        year_match = re.search(r'\\b(19|20)\\d{2}\\b', str(date_string))
        return year_match.group(0) if year_match else str(datetime.now().year)
    
    def _create_no_sources_result(self, query: str, query_analysis: QueryAnalysis, search_results: AdaptiveSearchResult, start_time: float, research_notes: List[str]) -> AutonomousResearchResult:
        """Create result when no sources found."""
        end_time = time.time()
        research_time = round(end_time - start_time, 2)
        
        return AutonomousResearchResult(
            query=query,
            research_time=research_time,
            confidence_score=0.0,
            
            query_analysis=vars(query_analysis),
            search_results=vars(search_results),
            relevance_evaluations=[],
            synthesis_result={
                "synthesis_text": f"No sources found for: {query}\\n\\nThe search strategies did not locate relevant academic papers. This could indicate:\\n1. The topic may be too specific or new\\n2. Different search terms might be needed\\n3. The research area may not have substantial academic literature",
                "synthesis_type": "no_relevant_sources",
                "confidence_score": 0.0,
                "identified_gaps": ["No academic sources found"],
                "recommendations": ["Try broader search terms", "Consult specialized databases", "Consider alternative research approaches"]
            },
            
            synthesis=f"No relevant sources found for: {query}",
            sources=[],
            bibliography="No sources available",
            
            quality_assessment={"overall_score": 0.0, "quality_level": "failed"},
            research_notes=research_notes + ["No sources found"],
            recommendations=["Try different search terms", "Expand search scope"],
            
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_result(self, query: str, error_message: str, start_time: float) -> AutonomousResearchResult:
        """Create result when research fails."""
        end_time = time.time()
        research_time = round(end_time - start_time, 2)
        
        return AutonomousResearchResult(
            query=query,
            research_time=research_time,
            confidence_score=0.0,
            
            query_analysis={},
            search_results={},
            relevance_evaluations=[],
            synthesis_result={
                "synthesis_text": f"Research failed: {error_message}",
                "synthesis_type": "error",
                "confidence_score": 0.0
            },
            
            synthesis=f"Research error: {error_message}",
            sources=[],
            bibliography="Error occurred",
            
            quality_assessment={"overall_score": 0.0, "quality_level": "error"},
            research_notes=[f"Error: {error_message}"],
            recommendations=["Check system configuration", "Retry with different query"],
            
            timestamp=datetime.now().isoformat()
        )


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_autonomous_agent():
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-test-key-for-research-demo":
            print("⚠️ Please set a real OPENAI_API_KEY for testing")
            return
        
        agent = AutonomousResearchAgent(api_key, max_sources=5, debug_mode=True)
        
        # Test query
        query = "explain the optimizers used in CNNs and RNNs"
        
        print(f"🤖 Testing Autonomous Research Agent")
        print("=" * 70)
        print(f"Query: {query}")
        print("=" * 70)
        
        # Conduct research
        result = await agent.conduct_autonomous_research(query, "apa")
        
        print(f"\\n📊 Research Summary:")
        print(f"Research Time: {result.research_time}s")
        print(f"Confidence Score: {result.confidence_score:.3f}")
        print(f"Quality Level: {result.quality_assessment.get('quality_level', 'unknown')}")
        print(f"Sources Found: {len(result.sources)}")
        print(f"Synthesis Type: {result.synthesis_result.get('synthesis_type', 'unknown')}")
        
        print(f"\\n📝 Synthesis:")
        print(result.synthesis[:500] + "..." if len(result.synthesis) > 500 else result.synthesis)
        
        print(f"\\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations[:5], 1):
            print(f"{i}. {rec}")
    
    # Run test
    asyncio.run(test_autonomous_agent())
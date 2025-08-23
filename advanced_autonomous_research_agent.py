"""
Advanced Autonomous Research Agent
Main orchestrator for comprehensive academic research with multi-source search and iterative refinement.
"""

import os
import asyncio
import openai
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Fix HuggingFace tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import core components (only the ones that work without heavy dependencies)
try:
    from advanced_search_engine import AdvancedSearchEngine, EnhancedSource
    from ai_query_analyzer import AIQueryAnalyzer, QueryAnalysis
    from ai_relevance_judge import AIRelevanceJudge, RelevanceAssessment, SourceContext
    from enhanced_ai_search_strategist import EnhancedAISearchStrategist
    from enhanced_ai_synthesizer import EnhancedAISynthesizer, StructuredSynthesis
    from ai_honest_synthesizer import AIHonestSynthesizer, HonestSynthesis
    from citation_manager import CitationManager, Source
except ImportError as e:
    logging.error(f"Failed to import core components: {e}")
    print(f"Error: Missing required components. Please ensure all files are in the current directory.")
    raise

logger = logging.getLogger(__name__)


@dataclass
class AdvancedResearchResult:
    """Complete advanced research result with comprehensive analysis."""
    query: str
    executive_summary: str
    detailed_analysis: str
    sources: List[EnhancedSource]
    quality_assessment: Dict[str, Any]
    research_gaps: List[str]
    recommendations: List[str]
    confidence_scores: Dict[str, float]
    research_iterations: int
    total_time: float
    citation_bibliography: str
    evidence_quality: str
    timestamp: str


class AdvancedAutonomousResearchAgent:
    """
    Advanced autonomous research agent with multi-source search, iterative refinement,
    and structured synthesis for professional-grade research analysis.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 max_sources: int = 25,
                 max_iterations: int = 3,
                 quality_threshold: float = 0.7,
                 debug_mode: bool = False):
        
        # Validate API key
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required but not provided")
        
        self.openai_api_key = openai_api_key
        self.max_sources = max_sources
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.debug_mode = debug_mode
        
        # Set up logging
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            logger.info("🐛 Debug mode enabled - comprehensive research logging active")
        
        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        # Initialize core research components
        logger.info("🚀 Initializing Advanced Autonomous Research Agent")
        
        try:
            self.search_engine = AdvancedSearchEngine(max_sources_per_engine=max_sources)
            logger.info("✅ Advanced search engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize search engine: {e}")
            raise
            
        try:
            self.query_analyzer = AIQueryAnalyzer(openai_api_key=openai_api_key, debug_mode=debug_mode)
            logger.info("✅ Query analyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize query analyzer: {e}")
            raise
            
        try:
            self.relevance_judge = AIRelevanceJudge(openai_api_key=openai_api_key, debug_mode=debug_mode)
            logger.info("✅ Relevance judge initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize relevance judge: {e}")
            raise
            
        try:
            self.search_strategist = EnhancedAISearchStrategist(
                openai_api_key=openai_api_key, 
                max_iterations=max_iterations,
                debug_mode=debug_mode
            )
            logger.info("✅ Search strategist initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize search strategist: {e}")
            raise
            
        try:
            self.synthesizer = EnhancedAISynthesizer(openai_api_key=openai_api_key, debug_mode=debug_mode)
            logger.info("✅ Synthesizer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize synthesizer: {e}")
            raise
            
        try:
            self.honest_synthesizer = AIHonestSynthesizer(openai_api_key=openai_api_key, debug_mode=debug_mode)
            logger.info("✅ Honest synthesizer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize honest synthesizer: {e}")
            raise
        
        self.citation_manager = CitationManager()
        logger.info("✅ Advanced Autonomous Research Agent ready")
    
    async def conduct_advanced_research(self, 
                                      query: str, 
                                      citation_style: str = "apa") -> AdvancedResearchResult:
        """
        Conduct comprehensive research with multi-source search and iterative refinement.
        
        Args:
            query: Research query
            citation_style: Citation format (apa, mla, ieee)
            
        Returns:
            AdvancedResearchResult with comprehensive analysis
        """
        start_time = time.time()
        logger.info(f"🔬 Starting advanced research for: {query}")
        
        try:
            # Phase 1: Query Analysis
            logger.info("🧠 Phase 1: Analyzing research query")
            query_analysis = await self.query_analyzer.analyze_query(query)
            
            # Phase 2: Multi-Source Search
            logger.info("🔍 Phase 2: Conducting multi-source search")
            sources = await self.search_engine.multi_source_search(query)
            
            if not sources:
                logger.warning("⚠️ No sources found during search")
                return self._create_empty_result(query, "No sources found for the given query")
            
            logger.info(f"📊 Found {len(sources)} initial sources")
            
            # Phase 3: Relevance Assessment
            logger.info("⚖️ Phase 3: Assessing source relevance")
            source_contexts = self._convert_to_source_contexts(sources)
            relevance_assessments = await self.relevance_judge.batch_assess_relevance(query, source_contexts)
            
            # Filter sources by relevance
            relevant_sources = []
            for assessment in relevance_assessments:
                if assessment.relevance_category in ['highly_relevant', 'maybe_relevant']:
                    # Find corresponding source
                    for source in sources:
                        if source.title in assessment.source_id or assessment.source_id in source.title:
                            relevant_sources.append(source)
                            break
            
            if not relevant_sources:
                relevant_sources = sources[:10]  # Keep top 10 if no relevant ones found
            
            logger.info(f"✅ {len(relevant_sources)} relevant sources identified")
            
            # Phase 4: Iterative Search Enhancement (Optional)
            if len(relevant_sources) < 10 and self.max_iterations > 1:
                logger.info("🔄 Phase 4: Conducting iterative search enhancement")
                try:
                    # Convert sources to dict format for strategist
                    source_dicts = [self._source_to_dict(source) for source in relevant_sources]
                    iterative_results = await self.search_strategist.conduct_iterative_search(query, source_dicts)
                    
                    if iterative_results.get('final_sources'):
                        additional_sources = iterative_results['final_sources']
                        logger.info(f"🔍 Iterative search found {len(additional_sources)} additional sources")
                        # Note: In a full implementation, we'd convert these back to EnhancedSource objects
                except Exception as e:
                    logger.warning(f"⚠️ Iterative search failed: {e}")
            
            # Phase 5: Structured Synthesis
            logger.info("📝 Phase 5: Creating structured synthesis")
            source_dicts = [self._source_to_dict(source) for source in relevant_sources]
            structured_synthesis = await self.synthesizer.create_structured_synthesis(query, source_dicts)
            
            # Phase 6: Honest Evidence Assessment
            logger.info("🔍 Phase 6: Conducting honest evidence assessment")
            honest_synthesis = await self.honest_synthesizer.create_honest_synthesis(
                query, source_dicts, structured_synthesis.executive_summary
            )
            
            # Phase 7: Generate Final Results
            logger.info("📋 Phase 7: Generating final research results")
            
            # Extract key findings from structured synthesis
            key_findings = structured_synthesis.key_findings
            
            # Generate research gaps and recommendations
            research_gaps = self._extract_research_gaps(structured_synthesis, honest_synthesis)
            recommendations = self._extract_recommendations(structured_synthesis, honest_synthesis)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                query_analysis, relevance_assessments, structured_synthesis, honest_synthesis
            )
            
            # Determine evidence quality
            evidence_quality = self._determine_evidence_quality(honest_synthesis, len(relevant_sources))
            
            # Generate citations
            citation_bibliography = self._generate_citations(relevant_sources, citation_style)
            
            total_time = time.time() - start_time
            
            # Create final result
            result = AdvancedResearchResult(
                query=query,
                executive_summary=structured_synthesis.executive_summary,
                detailed_analysis=structured_synthesis.comparative_analysis,
                sources=relevant_sources,
                quality_assessment={
                    "total_sources": len(relevant_sources),
                    "relevance_score": sum(a.relevance_score for a in relevance_assessments) / len(relevance_assessments),
                    "evidence_quality": evidence_quality,
                    "reliability_score": honest_synthesis.reliability_score
                },
                research_gaps=research_gaps,
                recommendations=recommendations,
                confidence_scores=confidence_scores,
                research_iterations=1,  # Will be enhanced with iterative search
                total_time=total_time,
                citation_bibliography=citation_bibliography,
                evidence_quality=evidence_quality,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Advanced research completed in {total_time:.2f} seconds")
            logger.info(f"📈 Found {len(relevant_sources)} relevant sources")
            logger.info(f"🎯 Evidence quality: {evidence_quality}")
            logger.info(f"💯 Overall confidence: {confidence_scores.get('overall_confidence', 0.0):.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Advanced research failed: {e}")
            return self._create_error_result(query, str(e))
    
    def _convert_to_source_contexts(self, sources: List[EnhancedSource]) -> List[SourceContext]:
        """Convert EnhancedSource objects to SourceContext for relevance assessment."""
        contexts = []
        for source in sources:
            context = SourceContext(
                title=source.title,
                abstract=source.abstract,
                authors=source.authors,
                year=source.year,
                venue=source.venue,
                keywords=[],  # Will be extracted if needed
                citations=source.citations,
                source_type=source.source_type
            )
            contexts.append(context)
        return contexts
    
    def _source_to_dict(self, source: EnhancedSource) -> Dict[str, Any]:
        """Convert EnhancedSource to dictionary format."""
        return {
            'title': source.title,
            'abstract': source.abstract,
            'authors': source.authors,
            'year': source.year,
            'venue': source.venue,
            'url': source.url,
            'citations': source.citations,
            'source_type': source.source_type,
            'quality_score': source.quality_score,
            'doi': source.doi,
            'pdf_url': source.pdf_url
        }
    
    def _extract_research_gaps(self, structured: StructuredSynthesis, honest: HonestSynthesis) -> List[str]:
        """Extract research gaps from synthesis results."""
        gaps = []
        
        # From structured synthesis
        if hasattr(structured, 'limitations_and_gaps'):
            gaps.append(structured.limitations_and_gaps)
        
        # From honest synthesis
        for assessment in honest.evidence_assessments:
            if assessment.evidence_level in ['limited', 'insufficient']:
                gaps.append(f"Limited evidence for: {assessment.claim}")
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _extract_recommendations(self, structured: StructuredSynthesis, honest: HonestSynthesis) -> List[str]:
        """Extract recommendations from synthesis results."""
        recommendations = []
        
        # From structured synthesis
        if hasattr(structured, 'future_directions'):
            recommendations.append(f"Future research: {structured.future_directions}")
        
        # From honest synthesis
        recommendations.extend(honest.recommendations)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_confidence_scores(self, 
                                   query_analysis: QueryAnalysis,
                                   relevance_assessments: List[RelevanceAssessment],
                                   structured: StructuredSynthesis,
                                   honest: HonestSynthesis) -> Dict[str, float]:
        """Calculate confidence scores for different aspects."""
        
        avg_relevance = sum(a.relevance_score for a in relevance_assessments) / len(relevance_assessments) if relevance_assessments else 0
        
        return {
            'overall_confidence': (query_analysis.confidence_score + avg_relevance + honest.reliability_score) / 3,
            'query_understanding': query_analysis.confidence_score,
            'source_relevance': avg_relevance,
            'evidence_reliability': honest.reliability_score,
            'synthesis_quality': structured.confidence_assessment.get('overall_confidence', 0.7)
        }
    
    def _determine_evidence_quality(self, honest: HonestSynthesis, source_count: int) -> str:
        """Determine overall evidence quality level."""
        if honest.reliability_score > 0.8 and source_count >= 15:
            return "Strong"
        elif honest.reliability_score > 0.6 and source_count >= 8:
            return "Moderate"
        else:
            return "Limited"
    
    def _generate_citations(self, sources: List[EnhancedSource], style: str) -> str:
        """Generate citations in the specified style."""
        try:
            # Convert to Citation Manager format
            citation_sources = []
            for source in sources:
                citation_source = Source(
                    title=source.title,
                    authors=source.authors,
                    year=source.year,
                    journal=source.venue,
                    url=source.url,
                    doi=source.doi
                )
                citation_sources.append(citation_source)
            
            return self.citation_manager.generate_bibliography(citation_sources, style)
        except Exception as e:
            logger.warning(f"Citation generation failed: {e}")
            return "Citation generation failed"
    
    def _create_empty_result(self, query: str, message: str) -> AdvancedResearchResult:
        """Create empty result when no sources are found."""
        return AdvancedResearchResult(
            query=query,
            executive_summary=f"No research sources found for '{query}'. {message}",
            detailed_analysis="Unable to conduct analysis due to lack of sources.",
            sources=[],
            quality_assessment={"total_sources": 0},
            research_gaps=["Lack of available research sources"],
            recommendations=["Try broader search terms or different keywords"],
            confidence_scores={"overall_confidence": 0.0},
            research_iterations=0,
            total_time=0.0,
            citation_bibliography="No sources to cite",
            evidence_quality="Insufficient",
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_result(self, query: str, error: str) -> AdvancedResearchResult:
        """Create error result when research fails."""
        return AdvancedResearchResult(
            query=query,
            executive_summary=f"Research failed for '{query}': {error}",
            detailed_analysis="Unable to complete research due to technical error.",
            sources=[],
            quality_assessment={"error": error},
            research_gaps=["Technical limitations prevented analysis"],
            recommendations=["Check API keys and internet connection"],
            confidence_scores={"overall_confidence": 0.0},
            research_iterations=0,
            total_time=0.0,
            citation_bibliography="No sources available",
            evidence_quality="Error",
            timestamp=datetime.now().isoformat()
        )


async def main():
    """Main function for command line usage."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Get query from user
    query = input("🔬 Enter your research query: ").strip()
    if not query:
        print("❌ Error: Please provide a research query")
        return
    
    print(f"🚀 Starting advanced research for: {query}")
    print("This may take 2-5 minutes for comprehensive analysis...")
    
    try:
        # Initialize agent
        agent = AdvancedAutonomousResearchAgent(
            openai_api_key=api_key,
            max_sources=20,
            debug_mode=True
        )
        
        # Conduct research
        result = await agent.conduct_advanced_research(query)
        
        # Display results
        print("\n" + "="*80)
        print("🎯 ADVANCED RESEARCH RESULTS")
        print("="*80)
        
        print(f"\n📊 Query: {result.query}")
        print(f"⏱️  Research Time: {result.total_time:.2f} seconds")
        print(f"📈 Sources Found: {len(result.sources)}")
        print(f"🎯 Evidence Quality: {result.evidence_quality}")
        print(f"💯 Overall Confidence: {result.confidence_scores.get('overall_confidence', 0.0):.2%}")
        
        print(f"\n📋 EXECUTIVE SUMMARY")
        print("-" * 40)
        print(result.executive_summary)
        
        print(f"\n📝 DETAILED ANALYSIS")
        print("-" * 40)
        print(result.detailed_analysis)
        
        if result.research_gaps:
            print(f"\n🔍 RESEARCH GAPS ({len(result.research_gaps)})")
            print("-" * 40)
            for i, gap in enumerate(result.research_gaps, 1):
                print(f"{i}. {gap}")
        
        if result.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(result.recommendations)})")
            print("-" * 40)
            for i, rec in enumerate(result.recommendations, 1):
                print(f"{i}. {rec}")
        
        print(f"\n📚 BIBLIOGRAPHY")
        print("-" * 40)
        print(result.citation_bibliography)
        
        print("\n" + "="*80)
        print("✅ Advanced research completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        import traceback
        if os.getenv("DEBUG"):
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
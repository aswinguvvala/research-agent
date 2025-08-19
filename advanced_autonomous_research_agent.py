"""
Advanced Autonomous Research Agent
Next-generation research agent with multi-source search, iterative refinement, and structured synthesis.
Designed to compete with premium AI research tools and provide comprehensive academic research capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import enhanced AI components
from ai_query_analyzer import AIQueryAnalyzer, QueryAnalysis
from ai_relevance_judge import AIRelevanceJudge, RelevanceAssessment
from enhanced_ai_search_strategist import EnhancedAISearchStrategist, ComprehensiveSearchResult
from enhanced_ai_synthesizer import EnhancedAISynthesizer, StructuredSynthesis
from advanced_search_engine import EnhancedSource

# Import citation manager for bibliography
try:
    from citation_manager import CitationManager, Source
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Citation manager not available")
    CitationManager = None
    Source = None

logger = logging.getLogger(__name__)


@dataclass
class AdvancedResearchResult:
    """Advanced research result with comprehensive analysis."""
    query: str
    research_time: float
    confidence_score: float
    
    # AI Analysis Results
    query_analysis: Dict[str, Any]
    search_results: Dict[str, Any]
    relevance_evaluations: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    synthesis_result: Dict[str, Any]
    
    # Enhanced Outputs
    executive_summary: str
    detailed_analysis: str
    structured_synthesis: Dict[str, Any]
    sources: List[Dict[str, Any]]
    bibliography: str
    
    # Quality Assessment
    quality_assessment: Dict[str, Any]
    coverage_assessment: Dict[str, Any]
    research_insights: Dict[str, Any]
    recommendations: List[str]
    
    # Metadata
    source_diversity: Dict[str, int]
    research_notes: List[str]
    timestamp: str


class AdvancedAutonomousResearchAgent:
    """
    Advanced autonomous research agent with multi-source search and comprehensive analysis.
    Designed to provide research quality comparable to premium AI research tools.
    """
    
    def __init__(self, 
                 openai_api_key: str, 
                 google_scholar_api_key: Optional[str] = None,
                 max_sources: int = 50, 
                 debug_mode: bool = False):
        """
        Initialize the advanced autonomous research agent.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o mini
            google_scholar_api_key: Optional Google Scholar API key (SerpAPI)
            max_sources: Maximum number of sources to process
            debug_mode: Enable detailed logging
        """
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_api_key = openai_api_key
        self.google_scholar_api_key = google_scholar_api_key
        self.max_sources = max_sources
        self.debug_mode = debug_mode
        
        # Initialize enhanced AI components
        self.query_analyzer = AIQueryAnalyzer(openai_api_key)
        self.search_strategist = EnhancedAISearchStrategist(openai_api_key, google_scholar_api_key)
        self.relevance_judge = AIRelevanceJudge(openai_api_key)
        self.synthesizer = EnhancedAISynthesizer(openai_api_key)
        
        # Initialize citation manager if available
        self.citation_manager = CitationManager() if CitationManager else None
        
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        logger.info("🚀 Advanced Autonomous Research Agent initialized")
        logger.info(f"🔍 Multi-source search enabled: {'Google Scholar' if google_scholar_api_key else 'Semantic Scholar + ArXiv'}")
    
    async def conduct_advanced_research(self, query: str, citation_style: str = "apa") -> AdvancedResearchResult:
        """
        Conduct advanced autonomous research with comprehensive multi-source analysis.
        
        Args:
            query: The research question
            citation_style: Citation format (apa, mla, ieee)
            
        Returns:
            AdvancedResearchResult with comprehensive findings
        """
        start_time = time.time()
        research_notes = []
        
        logger.info(f"🚀 Starting advanced research: {query}")
        research_notes.append(f"Advanced multi-source research initiated")
        
        try:
            # Phase 1: Enhanced AI Query Analysis
            logger.info("🧠 Phase 1: Enhanced Query Understanding & Strategy Generation")
            query_analysis = await self.query_analyzer.analyze_query(query)
            research_notes.append(f"Query analyzed: {query_analysis.research_type} in {query_analysis.domain_detected}")
            research_notes.append(f"Generated {len(query_analysis.search_strategies)} specialized search strategies")
            
            if self.debug_mode:
                logger.debug(f"🧠 Research Intent: {query_analysis.research_intent}")
                logger.debug(f"🧠 Domain: {query_analysis.domain_detected}")
                logger.debug(f"🧠 Key Concepts: {query_analysis.key_concepts}")
                logger.debug(f"🧠 Temporal Focus: {query_analysis.temporal_focus}")
            
            # Phase 2: Comprehensive Multi-Source Search with Iterative Refinement
            logger.info("🎯 Phase 2: Comprehensive Multi-Source Search")
            search_results = await self.search_strategist.execute_comprehensive_search(
                query=query,
                query_analysis=vars(query_analysis),
                max_sources=self.max_sources
            )
            
            research_notes.append(f"Executed {search_results.total_iterations} search iterations")
            research_notes.append(f"Found {search_results.total_sources_found} sources ({search_results.high_quality_sources} high-quality)")
            research_notes.append(f"Coverage level: {search_results.coverage_assessment.get('coverage_level', 'unknown')}")
            
            # Collect all sources from search iterations
            all_sources = []
            for iteration in search_results.search_iterations:
                all_sources.extend(iteration.sources_found)
            
            # Remove duplicates and limit to max_sources
            unique_sources = self._deduplicate_enhanced_sources(all_sources)[:self.max_sources]
            
            logger.info(f"🎯 Collected {len(unique_sources)} unique high-quality sources")
            
            if not unique_sources:
                return self._create_no_sources_result(query, query_analysis, search_results, start_time, research_notes)
            
            # Phase 3: Advanced AI Relevance Assessment
            logger.info("🔍 Phase 3: Advanced Relevance Assessment & Quality Filtering")
            
            # Convert EnhancedSource objects to dictionaries for relevance assessment
            source_dicts = [self._enhanced_source_to_dict(source) for source in unique_sources]
            
            relevance_evaluations = await self.relevance_judge.batch_evaluate_papers(
                query=query,
                papers=source_dicts
            )
            
            # Filter for relevant sources
            relevant_sources = []
            maybe_relevant_sources = []
            
            for (source_dict, assessment) in relevance_evaluations:
                # Find the corresponding EnhancedSource
                enhanced_source = next(
                    (s for s in unique_sources if s.title == source_dict.get("title")),
                    None
                )
                
                if enhanced_source and assessment.is_relevant:
                    relevant_sources.append((enhanced_source, assessment))
                elif enhanced_source and assessment.recommendation == "maybe":
                    maybe_relevant_sources.append((enhanced_source, assessment))
            
            logger.info(f"🔍 Relevance Assessment: {len(relevant_sources)} highly relevant, {len(maybe_relevant_sources)} possibly relevant")
            research_notes.append(f"AI identified {len(relevant_sources)} highly relevant sources")
            
            # Combine relevant and maybe relevant for comprehensive analysis
            all_evaluated_sources = relevant_sources + maybe_relevant_sources
            
            # Phase 4: Citation Processing & Metadata Enhancement
            logger.info("📚 Phase 4: Citation Processing & Metadata Enhancement")
            processed_sources = self._process_enhanced_citations(all_evaluated_sources, citation_style)
            
            # Phase 5: Advanced Structured Synthesis
            logger.info("📝 Phase 5: Advanced Structured Synthesis")
            synthesis_result = await self.synthesizer.synthesize_comprehensive_findings(
                query=query,
                sources=[source for source, _ in all_evaluated_sources],
                query_analysis=vars(query_analysis),
                search_metadata={
                    "total_iterations": search_results.total_iterations,
                    "sources_searched": search_results.total_sources_found,
                    "search_strategies": [iteration.search_strategy.get("strategy_name") for iteration in search_results.search_iterations]
                }
            )
            
            logger.info(f"📝 Advanced synthesis completed: {synthesis_result.synthesis_type} (confidence: {synthesis_result.confidence_score:.3f})")
            research_notes.append(f"Synthesis type: {synthesis_result.synthesis_type}")
            research_notes.append(f"Synthesis confidence: {synthesis_result.confidence_score:.3f}")
            
            # Phase 6: Comprehensive Quality Assessment
            logger.info("📊 Phase 6: Comprehensive Quality Assessment")
            quality_assessment = self._assess_advanced_research_quality(
                query_analysis, search_results, all_evaluated_sources, synthesis_result
            )
            
            # Generate enhanced bibliography
            bibliography = self._generate_enhanced_bibliography(processed_sources, citation_style)
            
            # Calculate comprehensive metrics
            end_time = time.time()
            research_time = round(end_time - start_time, 2)
            confidence_score = self._calculate_advanced_confidence_score(
                synthesis_result, search_results, quality_assessment
            )
            
            # Compile comprehensive recommendations
            final_recommendations = self._compile_comprehensive_recommendations(
                query_analysis, search_results, synthesis_result, quality_assessment
            )
            
            # Prepare research insights
            research_insights = {
                "key_findings": synthesis_result.key_findings,
                "knowledge_gaps": synthesis_result.knowledge_gaps,
                "future_directions": synthesis_result.future_directions,
                "practical_applications": synthesis_result.practical_applications
            }
            
            logger.info(f"✅ Advanced research completed in {research_time}s")
            logger.info(f"📊 Final metrics: {len(processed_sources)} sources, confidence: {confidence_score:.3f}")
            
            # Safe conversion for complex objects
            def safe_vars(obj):
                """Safely convert object to dict."""
                if hasattr(obj, '__dict__'):
                    return vars(obj)
                elif hasattr(obj, '_asdict'):
                    return obj._asdict()
                elif isinstance(obj, dict):
                    return obj
                else:
                    return {
                        'type': type(obj).__name__,
                        'str_representation': str(obj)
                    }
            
            return AdvancedResearchResult(
                query=query,
                research_time=research_time,
                confidence_score=confidence_score,
                
                # AI Results
                query_analysis=safe_vars(query_analysis),
                search_results=safe_vars(search_results),
                relevance_evaluations=[(self._enhanced_source_to_dict(s), safe_vars(a)) for s, a in all_evaluated_sources],
                synthesis_result=safe_vars(synthesis_result),
                
                # Enhanced Outputs
                executive_summary=synthesis_result.executive_summary,
                detailed_analysis=synthesis_result.detailed_analysis,
                structured_synthesis=synthesis_result.structured_content,
                sources=processed_sources,
                bibliography=bibliography,
                
                # Quality Assessment
                quality_assessment=quality_assessment,
                coverage_assessment=search_results.coverage_assessment,
                research_insights=research_insights,
                recommendations=final_recommendations,
                
                # Metadata
                source_diversity=search_results.source_diversity,
                research_notes=research_notes,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Advanced research failed: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return self._create_error_result(query, str(e), start_time)
    
    def _enhanced_source_to_dict(self, source: EnhancedSource) -> Dict[str, Any]:
        """Convert EnhancedSource to dictionary for compatibility."""
        return {
            "title": source.title,
            "authors": source.authors,
            "abstract": source.abstract,
            "url": source.url,
            "doi": source.doi,
            "pdf_url": source.pdf_url,
            "publication_year": source.publication_year,
            "venue": source.venue,
            "source_type": source.source_type,
            "source_id": source.source_id,
            "quality_score": source.quality_score,
            "keywords": source.keywords,
            "categories": source.categories,
            "citation_count": source.metrics.citation_count if source.metrics else 0,
            "is_highly_cited": source.metrics.is_highly_cited if source.metrics else False
        }
    
    def _deduplicate_enhanced_sources(self, sources: List[EnhancedSource]) -> List[EnhancedSource]:
        """Remove duplicate enhanced sources based on title similarity."""
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            title = source.title.lower().strip()
            title = ' '.join(title.split())  # Normalize whitespace
            
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_sources.append(source)
        
        # Sort by quality score (descending)
        unique_sources.sort(key=lambda x: x.quality_score, reverse=True)
        
        return unique_sources
    
    def _process_enhanced_citations(self, 
                                  evaluations: List[Tuple[EnhancedSource, RelevanceAssessment]], 
                                  citation_style: str) -> List[Dict[str, Any]]:
        """Process enhanced sources for citation management."""
        processed_sources = []
        
        if not self.citation_manager:
            logger.warning("Citation manager not available")
            return [self._enhanced_source_to_dict(source) for source, _ in evaluations]
        
        self.citation_manager.clear_sources()
        
        for enhanced_source, assessment in evaluations:
            try:
                # Create citation source
                citation_source = Source(
                    title=enhanced_source.title,
                    authors=enhanced_source.authors,
                    year=str(enhanced_source.publication_year),
                    url=enhanced_source.url,
                    doi=enhanced_source.doi,
                    journal=enhanced_source.venue,
                    source_type=enhanced_source.source_type
                )
                
                citation_num = self.citation_manager.add_source(citation_source)
                
                # Create enhanced processed source
                processed_source = {
                    **self._enhanced_source_to_dict(enhanced_source),
                    "citation_number": citation_num,
                    "relevance_score": assessment.relevance_score,
                    "relevance_explanation": assessment.explanation,
                    "is_relevant": assessment.is_relevant,
                    "quality_metrics": {
                        "citation_count": enhanced_source.metrics.citation_count if enhanced_source.metrics else 0,
                        "quality_score": enhanced_source.quality_score,
                        "venue_score": enhanced_source.metrics.venue_score if enhanced_source.metrics else 0,
                        "is_highly_cited": enhanced_source.metrics.is_highly_cited if enhanced_source.metrics else False
                    }
                }
                
                processed_sources.append(processed_source)
                
            except Exception as e:
                logger.warning(f"Failed to process enhanced citation for {enhanced_source.title}: {e}")
                # Add without citation processing
                processed_source = {
                    **self._enhanced_source_to_dict(enhanced_source),
                    "citation_number": len(processed_sources) + 1,
                    "relevance_score": assessment.relevance_score,
                    "is_relevant": assessment.is_relevant,
                    "error": "Citation processing failed"
                }
                processed_sources.append(processed_source)
        
        return processed_sources
    
    def _generate_enhanced_bibliography(self, sources: List[Dict[str, Any]], citation_style: str) -> str:
        """Generate enhanced bibliography with quality indicators."""
        if not self.citation_manager:
            return "Enhanced bibliography generation unavailable"
        
        try:
            base_bibliography = self.citation_manager.generate_bibliography(citation_style)
            
            # Add quality indicators
            enhanced_lines = []
            for line in base_bibliography.split('\n'):
                if line.strip():
                    # Find corresponding source for quality info
                    source_info = ""
                    for source in sources:
                        if any(author in line for author in source.get("authors", [])):
                            quality_score = source.get("quality_score", 0)
                            citation_count = source.get("citation_count", 0)
                            
                            if quality_score >= 0.8:
                                source_info = " [High Quality]"
                            elif citation_count > 100:
                                source_info = " [Highly Cited]"
                            
                            break
                    
                    enhanced_lines.append(line + source_info)
                else:
                    enhanced_lines.append(line)
            
            return '\n'.join(enhanced_lines)
            
        except Exception as e:
            logger.error(f"Enhanced bibliography generation failed: {e}")
            return f"Enhanced bibliography error: {str(e)}"
    
    def _assess_advanced_research_quality(self, 
                                        query_analysis: QueryAnalysis,
                                        search_results: ComprehensiveSearchResult,
                                        evaluated_sources: List[Tuple[EnhancedSource, RelevanceAssessment]],
                                        synthesis_result: StructuredSynthesis) -> Dict[str, Any]:
        """Assess comprehensive research quality with advanced metrics."""
        
        if not evaluated_sources:
            return {
                "overall_score": 0.0,
                "quality_level": "insufficient",
                "detailed_metrics": {},
                "strengths": [],
                "areas_for_improvement": ["No sources found"]
            }
        
        # Advanced quality metrics
        source_count = len(evaluated_sources)
        high_quality_count = sum(1 for source, _ in evaluated_sources if source.quality_score >= 0.7)
        avg_quality = sum(source.quality_score for source, _ in evaluated_sources) / source_count
        
        # Citation metrics
        total_citations = sum(source.metrics.citation_count if source.metrics else 0 for source, _ in evaluated_sources)
        highly_cited_count = sum(1 for source, _ in evaluated_sources 
                               if source.metrics and source.metrics.is_highly_cited)
        
        # Relevance metrics
        highly_relevant_count = sum(1 for _, assessment in evaluated_sources if assessment.relevance_score >= 0.8)
        avg_relevance = sum(assessment.relevance_score for _, assessment in evaluated_sources) / source_count
        
        # Coverage metrics
        search_coverage = search_results.coverage_assessment.get("overall_score", 0.0)
        synthesis_confidence = synthesis_result.confidence_score
        completeness = synthesis_result.completeness_score
        
        # Calculate comprehensive quality score
        quality_score = (
            min(source_count / 30, 1.0) * 0.15 +      # Source quantity (normalize to 30)
            (high_quality_count / source_count) * 0.20 + # Quality ratio
            avg_quality * 0.15 +                       # Average quality
            avg_relevance * 0.15 +                     # Average relevance
            search_coverage * 0.15 +                   # Search coverage
            synthesis_confidence * 0.10 +              # Synthesis confidence
            completeness * 0.10                        # Completeness
        )
        
        # Determine quality level
        if quality_score >= 0.85:
            quality_level = "exceptional"
        elif quality_score >= 0.75:
            quality_level = "excellent"
        elif quality_score >= 0.65:
            quality_level = "good"
        elif quality_score >= 0.50:
            quality_level = "acceptable"
        else:
            quality_level = "insufficient"
        
        # Identify strengths and areas for improvement
        strengths = []
        areas_for_improvement = []
        
        if source_count >= 20:
            strengths.append("Comprehensive source collection")
        elif source_count < 10:
            areas_for_improvement.append("Limited source coverage")
        
        if high_quality_count >= 10:
            strengths.append("Multiple high-quality sources")
        elif high_quality_count < 5:
            areas_for_improvement.append("Need more high-quality sources")
        
        if highly_cited_count >= 5:
            strengths.append("Well-cited foundational papers")
        
        if search_coverage >= 0.8:
            strengths.append("Excellent search coverage")
        elif search_coverage < 0.6:
            areas_for_improvement.append("Improve search comprehensiveness")
        
        if synthesis_confidence >= 0.8:
            strengths.append("High-confidence synthesis")
        elif synthesis_confidence < 0.6:
            areas_for_improvement.append("Synthesis requires more evidence")
        
        return {
            "overall_score": round(quality_score, 3),
            "quality_level": quality_level,
            "detailed_metrics": {
                "source_count": source_count,
                "high_quality_sources": high_quality_count,
                "average_quality": round(avg_quality, 3),
                "total_citations": total_citations,
                "highly_cited_papers": highly_cited_count,
                "highly_relevant_sources": highly_relevant_count,
                "average_relevance": round(avg_relevance, 3),
                "search_coverage": round(search_coverage, 3),
                "synthesis_confidence": round(synthesis_confidence, 3),
                "completeness_score": round(completeness, 3)
            },
            "strengths": strengths,
            "areas_for_improvement": areas_for_improvement
        }
    
    def _calculate_advanced_confidence_score(self, 
                                           synthesis_result: StructuredSynthesis,
                                           search_results: ComprehensiveSearchResult,
                                           quality_assessment: Dict[str, Any]) -> float:
        """Calculate advanced confidence score considering multiple factors."""
        
        # Component confidence scores
        synthesis_confidence = synthesis_result.confidence_score
        search_confidence = search_results.coverage_assessment.get("overall_score", 0.0)
        quality_confidence = quality_assessment.get("overall_score", 0.0)
        
        # Weight synthesis confidence higher as it's the final output
        overall_confidence = (
            synthesis_confidence * 0.5 +
            search_confidence * 0.3 +
            quality_confidence * 0.2
        )
        
        return round(overall_confidence, 3)
    
    def _compile_comprehensive_recommendations(self, 
                                            query_analysis: QueryAnalysis,
                                            search_results: ComprehensiveSearchResult,
                                            synthesis_result: StructuredSynthesis,
                                            quality_assessment: Dict[str, Any]) -> List[str]:
        """Compile comprehensive recommendations for users and future research."""
        recommendations = []
        
        # Add synthesis recommendations
        recommendations.extend(synthesis_result.recommendations[:3])
        
        # Add search-based recommendations
        recommendations.extend(search_results.recommendations[:2])
        
        # Add quality-based recommendations
        quality_level = quality_assessment.get("quality_level")
        if quality_level in ["exceptional", "excellent"]:
            recommendations.append("Research provides comprehensive coverage suitable for academic or professional use")
        elif quality_level == "good":
            recommendations.append("Research provides solid foundation with good evidence support")
        elif quality_level == "acceptable":
            recommendations.append("Research provides basic coverage but could benefit from additional sources")
        else:
            recommendations.append("Consider expanding research scope or refining search strategy")
        
        # Add gap-based recommendations
        if synthesis_result.knowledge_gaps:
            top_gaps = synthesis_result.knowledge_gaps[:2]
            recommendations.append(f"Future research should explore: {', '.join(top_gaps)}")
        
        # Add practical recommendations
        if synthesis_result.future_directions:
            recommendations.append(f"Emerging areas to watch: {', '.join(synthesis_result.future_directions[:2])}")
        
        # Remove duplicates and limit
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:8]
    
    def _create_no_sources_result(self, 
                                query: str, 
                                query_analysis: QueryAnalysis, 
                                search_results: ComprehensiveSearchResult,
                                start_time: float, 
                                research_notes: List[str]) -> AdvancedResearchResult:
        """Create result when no sources found."""
        end_time = time.time()
        research_time = round(end_time - start_time, 2)
        
        # Create empty synthesis result
        empty_synthesis = StructuredSynthesis(
            executive_summary=f"No relevant sources found for: {query}",
            detailed_analysis=f"Despite comprehensive multi-source search, no relevant academic sources were found.",
            structured_content={},
            main_categories=[],
            synthesis_type="no_sources",
            confidence_score=0.0,
            recommendations=["Try broader search terms", "Consult specialized databases", "Consider alternative research approaches"]
        )
        
        return AdvancedResearchResult(
            query=query,
            research_time=research_time,
            confidence_score=0.0,
            
            query_analysis=vars(query_analysis),
            search_results=vars(search_results),
            relevance_evaluations=[],
            synthesis_result=vars(empty_synthesis),
            
            executive_summary=empty_synthesis.executive_summary,
            detailed_analysis=empty_synthesis.detailed_analysis,
            structured_synthesis={},
            sources=[],
            bibliography="No sources available",
            
            quality_assessment={"overall_score": 0.0, "quality_level": "insufficient"},
            coverage_assessment=search_results.coverage_assessment,
            research_insights={"key_findings": [], "knowledge_gaps": [], "future_directions": []},
            recommendations=empty_synthesis.recommendations,
            
            source_diversity={},
            research_notes=research_notes + ["No sources found despite comprehensive search"],
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_result(self, query: str, error_message: str, start_time: float) -> AdvancedResearchResult:
        """Create result when research fails."""
        end_time = time.time()
        research_time = round(end_time - start_time, 2)
        
        # Create error synthesis
        error_synthesis = StructuredSynthesis(
            executive_summary=f"Research failed due to technical error: {error_message}",
            detailed_analysis=f"Advanced research for '{query}' encountered a technical error and could not be completed.",
            structured_content={},
            main_categories=[],
            synthesis_type="error",
            confidence_score=0.0,
            recommendations=["Check system configuration", "Retry with simpler query", "Contact support if issue persists"]
        )
        
        return AdvancedResearchResult(
            query=query,
            research_time=research_time,
            confidence_score=0.0,
            
            query_analysis={},
            search_results={},
            relevance_evaluations=[],
            synthesis_result=vars(error_synthesis),
            
            executive_summary=error_synthesis.executive_summary,
            detailed_analysis=error_synthesis.detailed_analysis,
            structured_synthesis={},
            sources=[],
            bibliography="Error occurred",
            
            quality_assessment={"overall_score": 0.0, "quality_level": "error"},
            coverage_assessment={},
            research_insights={"key_findings": [], "knowledge_gaps": [], "future_directions": []},
            recommendations=error_synthesis.recommendations,
            
            source_diversity={},
            research_notes=[f"Error: {error_message}"],
            timestamp=datetime.now().isoformat()
        )


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_advanced_agent():
        # Get API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        scholar_key = os.getenv("SERPAPI_API_KEY")  # Optional
        
        if not openai_key:
            print("⚠️ Please set OPENAI_API_KEY for testing")
            return
        
        agent = AdvancedAutonomousResearchAgent(
            openai_api_key=openai_key,
            google_scholar_api_key=scholar_key,
            max_sources=20,
            debug_mode=True
        )
        
        # Test query
        query = "types of activation functions in machine learning and deep learning"
        
        print(f"🚀 Testing Advanced Autonomous Research Agent")
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        
        # Conduct research
        result = await agent.conduct_advanced_research(query, "apa")
        
        print(f"\n📊 Advanced Research Summary:")
        print(f"Research Time: {result.research_time}s")
        print(f"Confidence Score: {result.confidence_score:.3f}")
        print(f"Quality Level: {result.quality_assessment.get('quality_level', 'unknown')}")
        print(f"Sources Found: {len(result.sources)}")
        print(f"Source Diversity: {result.source_diversity}")
        print(f"Coverage Assessment: {result.coverage_assessment.get('coverage_level', 'unknown')}")
        
        print(f"\n📈 Quality Metrics:")
        metrics = result.quality_assessment.get("detailed_metrics", {})
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
        print(f"\n📝 Executive Summary:")
        print(result.executive_summary[:300] + "..." if len(result.executive_summary) > 300 else result.executive_summary)
        
        print(f"\n🔍 Key Findings:")
        for i, finding in enumerate(result.research_insights.get("key_findings", [])[:3], 1):
            print(f"{i}. {finding}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"{i}. {rec}")
    
    # Run test
    asyncio.run(test_advanced_agent())
"""
Enhanced AI-Driven Search Strategist
Uses advanced multi-source search with iterative refinement and gap identification.
"""

import openai
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from advanced_search_engine import AdvancedSearchEngine, EnhancedSource

logger = logging.getLogger(__name__)


@dataclass
class SearchIteration:
    """Results of a single search iteration."""
    iteration_number: int
    search_strategy: Dict[str, Any]
    sources_found: List[EnhancedSource]
    gaps_identified: List[str]
    refinement_suggestions: List[str]
    quality_metrics: Dict[str, float]
    execution_time: float


@dataclass
class ComprehensiveSearchResult:
    """Complete multi-iteration search results."""
    total_iterations: int
    total_sources_found: int
    high_quality_sources: int
    search_iterations: List[SearchIteration]
    final_gaps: List[str]
    coverage_assessment: Dict[str, Any]
    source_diversity: Dict[str, int]
    recommendations: List[str]


class EnhancedAISearchStrategist:
    """
    Enhanced search strategist with multi-source capability and iterative refinement.
    """
    
    def __init__(self, openai_api_key: str, google_scholar_api_key: Optional[str] = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.search_engine = AdvancedSearchEngine(google_scholar_api_key)
        self.max_iterations = 4
        self.quality_threshold = 0.6
        
        logger.info("🚀 Enhanced AI Search Strategist initialized with multi-source capability")
    
    async def execute_comprehensive_search(self, 
                                         query: str,
                                         query_analysis: Dict[str, Any],
                                         max_sources: int = 50) -> ComprehensiveSearchResult:
        """
        Execute comprehensive multi-iteration search with gap identification.
        
        Args:
            query: Original research question
            query_analysis: Analysis from AIQueryAnalyzer
            max_sources: Maximum total sources to collect
            
        Returns:
            ComprehensiveSearchResult with detailed findings
        """
        logger.info(f"🎯 Starting comprehensive search for: {query[:50]}...")
        
        search_iterations = []
        all_sources = []
        cumulative_gaps = []
        
        # Extract initial search strategies
        initial_strategies = query_analysis.get("search_strategies", [])
        research_type = query_analysis.get("research_type", "explanation")
        domain = query_analysis.get("domain_detected", "general")
        
        for iteration in range(self.max_iterations):
            logger.info(f"🔄 Search iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate search strategy for this iteration
            if iteration == 0:
                # Use initial strategy from query analysis
                current_strategy = await self._select_initial_strategy(initial_strategies, query)
            else:
                # Generate adaptive strategy based on previous results
                current_strategy = await self._generate_adaptive_strategy(
                    query, search_iterations, all_sources, cumulative_gaps
                )
            
            # Execute search iteration
            iteration_result = await self._execute_search_iteration(
                iteration + 1, current_strategy, query, max_sources - len(all_sources)
            )
            
            search_iterations.append(iteration_result)
            
            # Add new sources (avoid duplicates)
            new_sources = self._filter_new_sources(iteration_result.sources_found, all_sources)
            all_sources.extend(new_sources)
            
            # Update cumulative gaps
            cumulative_gaps.extend(iteration_result.gaps_identified)
            
            logger.info(f"✅ Iteration {iteration + 1}: {len(iteration_result.sources_found)} sources, {len(new_sources)} new")
            
            # Check if we should continue
            if len(all_sources) >= max_sources:
                logger.info(f"🎯 Reached maximum sources limit ({max_sources})")
                break
            
            # AI decision on whether to continue
            should_continue = await self._should_continue_search(
                query, all_sources, cumulative_gaps, iteration_result
            )
            
            if not should_continue:
                logger.info("🤖 AI determined search is sufficiently comprehensive")
                break
        
        # Perform final analysis
        coverage_assessment = await self._assess_coverage(query, all_sources, cumulative_gaps, research_type)
        source_diversity = self._analyze_source_diversity(all_sources)
        final_recommendations = await self._generate_final_recommendations(
            query, all_sources, cumulative_gaps, coverage_assessment
        )
        
        # Count high-quality sources
        high_quality_count = sum(1 for source in all_sources if source.quality_score >= self.quality_threshold)
        
        result = ComprehensiveSearchResult(
            total_iterations=len(search_iterations),
            total_sources_found=len(all_sources),
            high_quality_sources=high_quality_count,
            search_iterations=search_iterations,
            final_gaps=list(set(cumulative_gaps)),  # Remove duplicates
            coverage_assessment=coverage_assessment,
            source_diversity=source_diversity,
            recommendations=final_recommendations
        )
        
        logger.info(f"🎯 Comprehensive search completed: {len(all_sources)} sources, {high_quality_count} high-quality")
        return result
    
    async def _select_initial_strategy(self, 
                                     initial_strategies: List[Dict[str, Any]], 
                                     query: str) -> Dict[str, Any]:
        """Select the best initial search strategy."""
        if initial_strategies:
            # Use the highest priority strategy
            return max(initial_strategies, key=lambda s: s.get("priority", 0.5))
        else:
            # Generate fallback strategy
            return await self._generate_fallback_strategy(query)
    
    async def _generate_adaptive_strategy(self, 
                                        query: str,
                                        previous_iterations: List[SearchIteration],
                                        current_sources: List[EnhancedSource],
                                        gaps: List[str]) -> Dict[str, Any]:
        """Generate adaptive search strategy based on previous results."""
        
        # Analyze what we have so far
        sources_analysis = self._analyze_current_sources(current_sources)
        gaps_analysis = list(set(gaps))  # Remove duplicates
        
        strategy_prompt = f"""Based on previous search results, generate an improved search strategy:

ORIGINAL QUERY: "{query}"

CURRENT SITUATION:
- Total sources found: {len(current_sources)}
- High-quality sources: {sum(1 for s in current_sources if s.quality_score >= 0.6)}
- Source types: {sources_analysis.get('source_types', {})}
- Publication years: {sources_analysis.get('year_range', 'Unknown')}

IDENTIFIED GAPS:
{chr(10).join(['- ' + gap for gap in gaps_analysis[:5]])}

PREVIOUS STRATEGIES USED:
{chr(10).join(['- ' + iteration.search_strategy.get('strategy_name', 'Unknown') for iteration in previous_iterations])}

Generate a new search strategy to fill gaps and improve coverage:

{{
    "strategy_name": "Targeted strategy name",
    "search_terms": ["specific_term1", "specific_term2", "specific_term3"],
    "target_sources": ["scholar", "semantic", "arxiv"],
    "focus_areas": ["specific aspect to target"],
    "temporal_focus": "recent|foundational|comprehensive",
    "expected_paper_types": ["survey", "technical", "comparative"],
    "priority": 0.8,
    "reasoning": "Why this strategy will fill the gaps"
}}

Focus on:
1. Terms that haven't been tried yet
2. Specific subtopics that are missing
3. Different paper types (surveys, comparisons, tutorials)
4. Balancing recent and foundational work
5. Addressing the specific gaps identified"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research strategist who excels at identifying gaps and generating targeted search strategies. Always respond with valid JSON."},
                    {"role": "user", "content": strategy_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"❌ Adaptive strategy generation failed: {e}")
            return await self._generate_fallback_strategy(query)
    
    async def _execute_search_iteration(self, 
                                      iteration_num: int,
                                      strategy: Dict[str, Any],
                                      query: str,
                                      max_results: int) -> SearchIteration:
        """Execute a single search iteration."""
        start_time = asyncio.get_event_loop().time()
        
        strategy_name = strategy.get("strategy_name", f"Iteration {iteration_num}")
        search_terms = strategy.get("search_terms", [query])
        target_sources = strategy.get("target_sources", ["semantic", "arxiv"])
        
        logger.info(f"🔍 Executing: {strategy_name}")
        
        all_sources = []
        
        # Execute searches for each term
        for search_term in search_terms[:3]:  # Limit to 3 terms per iteration
            try:
                sources = await self.search_engine.comprehensive_search(
                    query=search_term,
                    max_results_per_source=max(5, max_results // len(search_terms)),
                    sources=target_sources
                )
                all_sources.extend(sources)
                
                # Brief pause between searches
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"Search failed for '{search_term}': {e}")
                continue
        
        # Remove duplicates
        unique_sources = self._remove_duplicates(all_sources)
        
        # Analyze gaps and quality
        gaps_identified = await self._identify_gaps(query, unique_sources, strategy)
        quality_metrics = self._calculate_quality_metrics(unique_sources)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return SearchIteration(
            iteration_number=iteration_num,
            search_strategy=strategy,
            sources_found=unique_sources[:max_results],  # Limit results
            gaps_identified=gaps_identified,
            refinement_suggestions=[],  # To be filled by analysis
            quality_metrics=quality_metrics,
            execution_time=execution_time
        )
    
    async def _identify_gaps(self, 
                           query: str,
                           sources: List[EnhancedSource],
                           strategy: Dict[str, Any]) -> List[str]:
        """Identify gaps in current search results."""
        
        if not sources:
            return ["No sources found for this search strategy"]
        
        # Analyze source content for gaps
        titles = [source.title for source in sources]
        abstracts = [source.abstract for source in sources if source.abstract]
        
        gap_analysis_prompt = f"""Analyze these search results for gaps in coverage:

QUERY: "{query}"
SEARCH STRATEGY: {strategy.get('strategy_name', 'Unknown')}

FOUND PAPERS ({len(sources)}):
{chr(10).join([f"- {title}" for title in titles[:10]])}

SAMPLE ABSTRACTS:
{chr(10).join([f"- {abstract[:200]}..." for abstract in abstracts[:3]])}

Identify specific gaps that should be addressed in future searches:

{{
    "content_gaps": ["specific topic not covered", "missing aspect"],
    "methodological_gaps": ["missing research approach", "lack of comparative studies"],
    "temporal_gaps": ["missing recent work", "lack of foundational papers"],
    "source_type_gaps": ["need more surveys", "missing practical implementations"]
}}

Focus on:
1. Important subtopics not represented
2. Missing research methodologies
3. Temporal coverage issues
4. Source type imbalances
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying research gaps and suggesting improvements. Always respond with valid JSON."},
                    {"role": "user", "content": gap_analysis_prompt}
                ],
                max_tokens=600,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            gaps_data = json.loads(response_text)
            
            # Flatten all gap types into a single list
            all_gaps = []
            for gap_type, gaps in gaps_data.items():
                if isinstance(gaps, list):
                    all_gaps.extend(gaps)
            
            return all_gaps
            
        except Exception as e:
            logger.error(f"❌ Gap analysis failed: {e}")
            return ["Unable to analyze gaps due to processing error"]
    
    async def _should_continue_search(self, 
                                    query: str,
                                    current_sources: List[EnhancedSource],
                                    gaps: List[str],
                                    last_iteration: SearchIteration) -> bool:
        """AI decision on whether to continue searching."""
        
        # Simple heuristics for now (can be enhanced with AI decision later)
        if len(current_sources) >= 30:  # Sufficient sources
            return False
        
        if last_iteration.iteration_number >= self.max_iterations:
            return False
        
        if len(last_iteration.sources_found) < 2:  # Very few new sources
            return False
        
        # Check if we're finding high-quality sources
        high_quality_in_last = sum(1 for s in last_iteration.sources_found if s.quality_score >= self.quality_threshold)
        if high_quality_in_last == 0 and last_iteration.iteration_number > 2:
            return False
        
        return True
    
    async def _assess_coverage(self, 
                             query: str,
                             sources: List[EnhancedSource],
                             gaps: List[str],
                             research_type: str) -> Dict[str, Any]:
        """Assess overall coverage quality of the search."""
        
        if not sources:
            return {
                "overall_score": 0.0,
                "coverage_level": "insufficient",
                "strengths": [],
                "weaknesses": ["No sources found"],
                "completeness": 0.0
            }
        
        # Calculate various coverage metrics
        source_diversity = len(set(source.source_type for source in sources))
        temporal_span = max(source.publication_year for source in sources) - min(source.publication_year for source in sources)
        avg_quality = sum(source.quality_score for source in sources) / len(sources)
        high_quality_ratio = sum(1 for source in sources if source.quality_score >= self.quality_threshold) / len(sources)
        
        # Calculate overall coverage score
        coverage_score = (
            min(len(sources) / 20, 1.0) * 0.3 +  # Number of sources (normalize to 20)
            (source_diversity / 3.0) * 0.2 +      # Source diversity (max 3 types)
            avg_quality * 0.3 +                   # Average quality
            high_quality_ratio * 0.2              # High-quality ratio
        )
        
        # Determine coverage level
        if coverage_score >= 0.8:
            coverage_level = "excellent"
        elif coverage_score >= 0.6:
            coverage_level = "good"
        elif coverage_score >= 0.4:
            coverage_level = "acceptable"
        else:
            coverage_level = "insufficient"
        
        return {
            "overall_score": coverage_score,
            "coverage_level": coverage_level,
            "total_sources": len(sources),
            "high_quality_sources": sum(1 for s in sources if s.quality_score >= self.quality_threshold),
            "source_diversity": source_diversity,
            "temporal_span": temporal_span,
            "average_quality": avg_quality,
            "identified_gaps": len(set(gaps)),
            "strengths": self._identify_coverage_strengths(sources),
            "weaknesses": gaps[:5]  # Top 5 gaps
        }
    
    def _identify_coverage_strengths(self, sources: List[EnhancedSource]) -> List[str]:
        """Identify strengths in the source collection."""
        strengths = []
        
        if len(sources) >= 20:
            strengths.append("Comprehensive source collection")
        
        high_quality_count = sum(1 for s in sources if s.quality_score >= self.quality_threshold)
        if high_quality_count >= 10:
            strengths.append("Multiple high-quality sources")
        
        source_types = set(source.source_type for source in sources)
        if len(source_types) >= 2:
            strengths.append("Diverse source types")
        
        recent_sources = sum(1 for s in sources if s.publication_year >= 2020)
        if recent_sources >= 5:
            strengths.append("Recent research coverage")
        
        if any(s.metrics and s.metrics.citation_count > 100 for s in sources):
            strengths.append("Highly cited foundational papers")
        
        return strengths
    
    def _analyze_source_diversity(self, sources: List[EnhancedSource]) -> Dict[str, int]:
        """Analyze diversity of sources."""
        diversity = {}
        
        # Source type distribution
        for source in sources:
            source_type = source.source_type
            diversity[source_type] = diversity.get(source_type, 0) + 1
        
        return diversity
    
    def _analyze_current_sources(self, sources: List[EnhancedSource]) -> Dict[str, Any]:
        """Analyze current source collection."""
        if not sources:
            return {"source_types": {}, "year_range": "None"}
        
        source_types = {}
        years = []
        
        for source in sources:
            source_types[source.source_type] = source_types.get(source.source_type, 0) + 1
            if source.publication_year:
                years.append(source.publication_year)
        
        year_range = f"{min(years)}-{max(years)}" if years else "Unknown"
        
        return {
            "source_types": source_types,
            "year_range": year_range
        }
    
    def _filter_new_sources(self, new_sources: List[EnhancedSource], existing_sources: List[EnhancedSource]) -> List[EnhancedSource]:
        """Filter out sources that are already in the collection."""
        existing_titles = {source.title.lower().strip() for source in existing_sources}
        
        filtered = []
        for source in new_sources:
            title = source.title.lower().strip()
            if title not in existing_titles:
                filtered.append(source)
                existing_titles.add(title)
        
        return filtered
    
    def _remove_duplicates(self, sources: List[EnhancedSource]) -> List[EnhancedSource]:
        """Remove duplicate sources from a list."""
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            title = source.title.lower().strip()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_sources.append(source)
        
        return unique_sources
    
    def _calculate_quality_metrics(self, sources: List[EnhancedSource]) -> Dict[str, float]:
        """Calculate quality metrics for a source collection."""
        if not sources:
            return {"average_quality": 0.0, "high_quality_ratio": 0.0}
        
        avg_quality = sum(source.quality_score for source in sources) / len(sources)
        high_quality_count = sum(1 for source in sources if source.quality_score >= self.quality_threshold)
        high_quality_ratio = high_quality_count / len(sources)
        
        return {
            "average_quality": avg_quality,
            "high_quality_ratio": high_quality_ratio,
            "total_sources": len(sources),
            "high_quality_sources": high_quality_count
        }
    
    async def _generate_final_recommendations(self, 
                                            query: str,
                                            sources: List[EnhancedSource],
                                            gaps: List[str],
                                            coverage: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on search results."""
        recommendations = []
        
        # Coverage-based recommendations
        if coverage["coverage_level"] == "excellent":
            recommendations.append("Comprehensive research coverage achieved")
        elif coverage["coverage_level"] == "good":
            recommendations.append("Good research coverage with minor gaps")
        else:
            recommendations.append("Research coverage could be improved")
        
        # Gap-based recommendations
        if gaps:
            recommendations.append(f"Consider exploring: {', '.join(gaps[:3])}")
        
        # Source quality recommendations
        high_quality_ratio = coverage.get("high_quality_sources", 0) / max(coverage.get("total_sources", 1), 1)
        if high_quality_ratio < 0.5:
            recommendations.append("Focus on higher-quality sources for better research foundation")
        
        # Diversity recommendations
        if coverage.get("source_diversity", 0) < 2:
            recommendations.append("Expand search to include diverse source types")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def _generate_fallback_strategy(self, query: str) -> Dict[str, Any]:
        """Generate fallback search strategy."""
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        return {
            "strategy_name": "Fallback Keyword Search",
            "search_terms": keywords[:5],
            "target_sources": ["semantic", "arxiv"],
            "priority": 0.5,
            "reasoning": "Fallback strategy using query keywords"
        }


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_enhanced_strategist():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Please set OPENAI_API_KEY for testing")
            return
        
        strategist = EnhancedAISearchStrategist(api_key)
        
        # Mock query analysis
        query = "types of activation functions in machine learning"
        mock_analysis = {
            "research_type": "survey",
            "domain_detected": "machine learning",
            "search_strategies": [{
                "strategy_name": "Comprehensive Activation Functions Survey",
                "search_terms": ["activation functions", "neural network activation", "deep learning activation"],
                "target_sources": ["semantic", "arxiv"],
                "priority": 0.9
            }]
        }
        
        print(f"🧪 Testing Enhanced Search Strategist")
        print("=" * 70)
        print(f"Query: {query}")
        print("=" * 70)
        
        result = await strategist.execute_comprehensive_search(
            query=query,
            query_analysis=mock_analysis,
            max_sources=20
        )
        
        print(f"\n📊 Search Results Summary:")
        print(f"Total Iterations: {result.total_iterations}")
        print(f"Total Sources: {result.total_sources_found}")
        print(f"High-Quality Sources: {result.high_quality_sources}")
        print(f"Coverage Level: {result.coverage_assessment.get('coverage_level')}")
        print(f"Source Diversity: {result.source_diversity}")
        print(f"Final Gaps: {len(result.final_gaps)}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
    
    # Run test
    asyncio.run(test_enhanced_strategist())
"""
Enhanced AI Search Strategist
Iterative search with gap identification and progressive improvement.
"""

import openai
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class SearchIteration:
    """Represents one iteration of the search process."""
    iteration_number: int
    search_query: str
    search_strategy: str
    sources_found: int
    quality_score: float
    gaps_identified: List[str]
    coverage_areas: List[str]
    next_strategy: Optional[str]


@dataclass
class GapAnalysis:
    """Analysis of research gaps in current findings."""
    identified_gaps: List[str]
    coverage_assessment: Dict[str, float]
    missing_topics: List[str]
    underrepresented_areas: List[str]
    suggested_searches: List[str]
    confidence_score: float


@dataclass
class SearchStrategy:
    """Definition of a search strategy."""
    name: str
    description: str
    query_template: str
    expected_source_types: List[str]
    priority: float
    estimated_coverage: List[str]


class EnhancedAISearchStrategist:
    """
    Enhanced AI search strategist that implements iterative search with gap identification,
    progressive improvement, and adaptive strategy selection.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 max_iterations: int = 4,
                 quality_threshold: float = 0.7,
                 debug_mode: bool = False):
        
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.debug_mode = debug_mode
        
        # Initialize search strategies
        self.search_strategies = self._initialize_search_strategies()
        
        # Track search history
        self.search_history: List[SearchIteration] = []
        self.covered_topics: Set[str] = set()
        self.used_strategies: Set[str] = set()
        
        logger.info("🎯 Enhanced AI Search Strategist initialized")
    
    async def conduct_iterative_search(self, 
                                     original_query: str,
                                     initial_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Conduct iterative search with gap identification and progressive improvement.
        
        Args:
            original_query: Original research query
            initial_sources: Sources from initial search
            
        Returns:
            Complete iterative search results with gap analysis
        """
        logger.info(f"🔄 Starting iterative search for: {original_query}")
        
        # Initialize search state
        self.search_history = []
        self.covered_topics = set()
        self.used_strategies = set()
        
        current_sources = initial_sources.copy()
        current_iteration = 1
        
        # Record initial search as first iteration
        initial_iteration = SearchIteration(
            iteration_number=0,
            search_query=original_query,
            search_strategy="initial_broad_search",
            sources_found=len(initial_sources),
            quality_score=self._assess_source_quality(initial_sources),
            gaps_identified=[],
            coverage_areas=await self._identify_coverage_areas(initial_sources),
            next_strategy=None
        )
        
        self.search_history.append(initial_iteration)
        self.covered_topics.update(initial_iteration.coverage_areas)
        
        try:
            while current_iteration < self.max_iterations:
                logger.info(f"🔍 Starting iteration {current_iteration}")
                
                # Analyze gaps in current findings
                gap_analysis = await self._analyze_research_gaps(
                    original_query, current_sources
                )
                
                # Check if we have sufficient coverage
                if self._is_coverage_sufficient(gap_analysis):
                    logger.info(f"✅ Sufficient coverage achieved after {current_iteration} iterations")
                    break
                
                # Select next search strategy
                next_strategy = await self._select_next_strategy(
                    original_query, gap_analysis, current_sources
                )
                
                if not next_strategy:
                    logger.info("🚫 No suitable next strategy found")
                    break
                
                # Generate search query for next iteration
                search_query = await self._generate_iterative_query(
                    original_query, gap_analysis, next_strategy
                )
                
                # This would integrate with the search engine in practice
                # For now, we simulate the search results
                new_sources = await self._simulate_iterative_search(
                    search_query, next_strategy, gap_analysis
                )
                
                # Record this iteration
                iteration = SearchIteration(
                    iteration_number=current_iteration,
                    search_query=search_query,
                    search_strategy=next_strategy['name'],
                    sources_found=len(new_sources),
                    quality_score=self._assess_source_quality(new_sources),
                    gaps_identified=gap_analysis.identified_gaps,
                    coverage_areas=await self._identify_coverage_areas(new_sources),
                    next_strategy=None
                )
                
                self.search_history.append(iteration)
                self.covered_topics.update(iteration.coverage_areas)
                self.used_strategies.add(next_strategy['name'])
                
                # Merge new sources with existing ones
                current_sources.extend(new_sources)
                current_sources = self._deduplicate_sources(current_sources)
                
                current_iteration += 1
                
                logger.info(f"📊 Iteration {current_iteration-1} completed: {len(new_sources)} new sources")
            
            # Final gap analysis
            final_gap_analysis = await self._analyze_research_gaps(
                original_query, current_sources
            )
            
            # Generate search summary
            search_summary = self._generate_search_summary()
            
            results = {
                'original_query': original_query,
                'total_iterations': len(self.search_history),
                'final_sources': current_sources,
                'total_sources': len(current_sources),
                'search_history': self.search_history,
                'final_gap_analysis': final_gap_analysis,
                'search_summary': search_summary,
                'coverage_areas': list(self.covered_topics),
                'quality_progression': [iter.quality_score for iter in self.search_history]
            }
            
            logger.info(f"✅ Iterative search completed: {len(current_sources)} total sources across {len(self.search_history)} iterations")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Iterative search failed: {e}")
            return {
                'error': str(e),
                'partial_results': {
                    'search_history': self.search_history,
                    'sources': current_sources
                }
            }
    
    async def _analyze_research_gaps(self, 
                                   query: str, 
                                   sources: List[Dict[str, Any]]) -> GapAnalysis:
        """Analyze gaps in current research findings."""
        try:
            # Extract key information from sources
            source_summaries = []
            for source in sources[:15]:  # Limit for API efficiency
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', 'No abstract')[:200]
                source_summaries.append(f"- {title}: {abstract}")
            
            sources_text = "\n".join(source_summaries)
            
            prompt = f"""
            Analyze the research gaps for this query: "{query}"
            
            Current sources found:
            {sources_text}
            
            Identify:
            1. Research gaps not covered by current sources
            2. Underrepresented topics or perspectives
            3. Missing methodological approaches
            4. Temporal gaps (time periods not covered)
            5. Geographic or demographic gaps
            6. Interdisciplinary connections missing
            
            For each gap, suggest specific search strategies to address it.
            
            Format your response as:
            GAPS:
            - [gap description]
            
            MISSING_TOPICS:
            - [topic description]
            
            SUGGESTED_SEARCHES:
            - [search suggestion]
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.4
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Parse the response
            gaps = self._extract_section(analysis_text, "GAPS:")
            missing_topics = self._extract_section(analysis_text, "MISSING_TOPICS:")
            suggested_searches = self._extract_section(analysis_text, "SUGGESTED_SEARCHES:")
            
            # Assess coverage
            coverage_assessment = await self._assess_topic_coverage(query, sources)
            
            # Calculate confidence
            confidence = self._calculate_gap_analysis_confidence(gaps, sources)
            
            return GapAnalysis(
                identified_gaps=gaps,
                coverage_assessment=coverage_assessment,
                missing_topics=missing_topics,
                underrepresented_areas=self._identify_underrepresented_areas(sources),
                suggested_searches=suggested_searches,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return GapAnalysis(
                identified_gaps=[],
                coverage_assessment={},
                missing_topics=[],
                underrepresented_areas=[],
                suggested_searches=[],
                confidence_score=0.0
            )
    
    async def _select_next_strategy(self, 
                                  query: str, 
                                  gap_analysis: GapAnalysis, 
                                  current_sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the next search strategy based on gap analysis."""
        try:
            # Get unused strategies
            available_strategies = [
                strategy for strategy in self.search_strategies
                if strategy['name'] not in self.used_strategies
            ]
            
            if not available_strategies:
                return None
            
            # Score strategies based on gap analysis
            strategy_scores = {}
            
            for strategy in available_strategies:
                score = 0.0
                
                # Base priority
                score += strategy['priority']
                
                # Gap coverage potential
                for gap in gap_analysis.identified_gaps:
                    if any(coverage in gap.lower() for coverage in strategy['estimated_coverage']):
                        score += 0.2
                
                # Missing topic alignment
                for topic in gap_analysis.missing_topics:
                    if any(coverage in topic.lower() for coverage in strategy['estimated_coverage']):
                        score += 0.15
                
                # Underrepresented area coverage
                for area in gap_analysis.underrepresented_areas:
                    if any(coverage in area.lower() for coverage in strategy['estimated_coverage']):
                        score += 0.1
                
                strategy_scores[strategy['name']] = score
            
            # Select highest scoring strategy
            if strategy_scores:
                best_strategy_name = max(strategy_scores, key=strategy_scores.get)
                best_strategy = next(s for s in available_strategies if s['name'] == best_strategy_name)
                
                logger.info(f"🎯 Selected strategy: {best_strategy_name} (score: {strategy_scores[best_strategy_name]:.2f})")
                return best_strategy
            
            return None
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return None
    
    async def _generate_iterative_query(self, 
                                      original_query: str, 
                                      gap_analysis: GapAnalysis, 
                                      strategy: Dict[str, Any]) -> str:
        """Generate search query for the next iteration."""
        try:
            gaps_text = "; ".join(gap_analysis.identified_gaps[:3])
            missing_topics_text = "; ".join(gap_analysis.missing_topics[:3])
            
            prompt = f"""
            Generate a focused search query for iterative research improvement.
            
            Original Query: "{original_query}"
            Strategy: {strategy['description']}
            Query Template: {strategy['query_template']}
            
            Identified Gaps: {gaps_text}
            Missing Topics: {missing_topics_text}
            
            Create a search query that:
            1. Addresses the most important gaps
            2. Follows the strategy template
            3. Maintains relevance to the original query
            4. Uses specific keywords for the target areas
            
            Return only the search query, no explanation.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            
            query = response.choices[0].message.content.strip()
            
            # Clean up the query
            query = re.sub(r'^["\'](.*)["\']$', r'\1', query)  # Remove quotes
            query = query.strip()
            
            return query
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback to template-based generation
            return strategy['query_template'].replace('{topic}', original_query)
    
    async def _simulate_iterative_search(self, 
                                       query: str, 
                                       strategy: Dict[str, Any], 
                                       gap_analysis: GapAnalysis) -> List[Dict[str, Any]]:
        """Simulate iterative search results (placeholder for real search integration)."""
        # In practice, this would call the actual search engine
        # For now, we simulate realistic search results
        
        simulated_sources = []
        
        # Generate simulated sources based on strategy and gaps
        for i in range(3, 8):  # 3-7 sources per iteration
            source = {
                'title': f"Simulated Paper {i}: {query[:30]}...",
                'abstract': f"This paper addresses aspects of {query} with focus on {strategy['name']}...",
                'authors': [f"Author {i}A", f"Author {i}B"],
                'year': 2023 - (i % 3),
                'venue': f"Journal of {strategy['name'].title()}",
                'citations': max(0, 50 - i * 5),
                'source_type': 'simulated',
                'iteration': len(self.search_history),
                'strategy': strategy['name']
            }
            simulated_sources.append(source)
        
        return simulated_sources
    
    async def _identify_coverage_areas(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Identify what research areas are covered by the sources."""
        try:
            if not sources:
                return []
            
            # Extract titles and abstracts
            content_samples = []
            for source in sources[:10]:  # Limit for efficiency
                title = source.get('title', '')
                abstract = source.get('abstract', '')[:100]  # First 100 chars
                content_samples.append(f"{title} - {abstract}")
            
            content_text = "; ".join(content_samples)
            
            prompt = f"""
            Identify the main research areas covered by these sources:
            
            {content_text}
            
            Extract 3-7 main research areas/topics that are well covered.
            Return as a simple list, one area per line.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            areas_text = response.choices[0].message.content.strip()
            areas = [area.strip() for area in areas_text.split('\n') if area.strip()]
            
            # Clean up areas
            cleaned_areas = []
            for area in areas:
                clean_area = re.sub(r'^\d+\.?\s*', '', area)  # Remove numbering
                clean_area = clean_area.strip('- ')
                if clean_area:
                    cleaned_areas.append(clean_area)
            
            return cleaned_areas[:7]
            
        except Exception as e:
            logger.warning(f"Coverage area identification failed: {e}")
            return []
    
    async def _assess_topic_coverage(self, 
                                   query: str, 
                                   sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess how well different aspects of the query are covered."""
        try:
            # Extract key aspects of the query
            prompt = f"""
            Break down this research query into 3-5 key aspects/components:
            "{query}"
            
            Return as a simple list, one aspect per line.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            aspects_text = response.choices[0].message.content.strip()
            aspects = [a.strip() for a in aspects_text.split('\n') if a.strip()]
            
            # Assess coverage for each aspect
            coverage_scores = {}
            
            for aspect in aspects:
                # Count sources that cover this aspect
                covering_sources = 0
                for source in sources:
                    title = source.get('title', '').lower()
                    abstract = source.get('abstract', '').lower()
                    
                    if aspect.lower() in title or aspect.lower() in abstract:
                        covering_sources += 1
                
                # Calculate coverage score
                if sources:
                    coverage_scores[aspect] = covering_sources / len(sources)
                else:
                    coverage_scores[aspect] = 0.0
            
            return coverage_scores
            
        except Exception as e:
            logger.warning(f"Topic coverage assessment failed: {e}")
            return {}
    
    def _is_coverage_sufficient(self, gap_analysis: GapAnalysis) -> bool:
        """Determine if current coverage is sufficient to stop iterating."""
        # Check if confidence is high enough
        if gap_analysis.confidence_score < 0.6:
            return False
        
        # Check if there are significant gaps
        if len(gap_analysis.identified_gaps) > 3:
            return False
        
        # Check coverage assessment
        coverage_values = list(gap_analysis.coverage_assessment.values())
        if coverage_values:
            avg_coverage = sum(coverage_values) / len(coverage_values)
            if avg_coverage < 0.7:
                return False
        
        return True
    
    def _assess_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """Assess the overall quality of a set of sources."""
        if not sources:
            return 0.0
        
        quality_scores = []
        
        for source in sources:
            score = 0.0
            
            # Citation count
            citations = source.get('citations', 0)
            if citations > 100:
                score += 0.3
            elif citations > 10:
                score += 0.2
            elif citations > 0:
                score += 0.1
            
            # Publication year
            year = source.get('year', 0)
            current_year = datetime.now().year
            if year >= current_year - 2:
                score += 0.3
            elif year >= current_year - 5:
                score += 0.2
            elif year >= current_year - 10:
                score += 0.1
            
            # Venue quality
            venue = source.get('venue', '').lower()
            if any(top in venue for top in ['nature', 'science', 'cell']):
                score += 0.2
            elif 'journal' in venue:
                score += 0.1
            
            # Abstract availability
            if source.get('abstract') and source['abstract'] != 'No abstract available':
                score += 0.2
            
            quality_scores.append(min(1.0, score))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources from the list."""
        seen_titles = set()
        unique_sources = []
        
        for source in sources:
            title = source.get('title', '').lower().strip()
            if title and title not in seen_titles:
                unique_sources.append(source)
                seen_titles.add(title)
        
        return unique_sources
    
    def _identify_underrepresented_areas(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Identify underrepresented research areas."""
        # Simple implementation based on venue and keyword analysis
        venues = [source.get('venue', '') for source in sources]
        venue_counts = Counter(venues)
        
        # Areas with only 1-2 sources might be underrepresented
        underrepresented = []
        for venue, count in venue_counts.items():
            if count <= 2 and venue:
                underrepresented.append(venue)
        
        return underrepresented[:5]  # Limit to top 5
    
    def _calculate_gap_analysis_confidence(self, 
                                         gaps: List[str], 
                                         sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the gap analysis."""
        confidence = 0.5  # Base confidence
        
        # More sources generally mean more confident analysis
        if len(sources) >= 15:
            confidence += 0.2
        elif len(sources) >= 8:
            confidence += 0.1
        
        # Fewer gaps might indicate better coverage
        if len(gaps) <= 2:
            confidence += 0.2
        elif len(gaps) <= 5:
            confidence += 0.1
        
        # Recent sources increase confidence
        recent_sources = sum(1 for s in sources if s.get('year', 0) >= datetime.now().year - 3)
        if recent_sources >= 5:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _generate_search_summary(self) -> Dict[str, Any]:
        """Generate a summary of the iterative search process."""
        if not self.search_history:
            return {}
        
        total_sources = sum(iter.sources_found for iter in self.search_history)
        avg_quality = sum(iter.quality_score for iter in self.search_history) / len(self.search_history)
        
        strategies_used = [iter.search_strategy for iter in self.search_history]
        
        return {
            'total_iterations': len(self.search_history),
            'total_sources_found': total_sources,
            'average_quality_score': avg_quality,
            'strategies_used': strategies_used,
            'coverage_areas_identified': len(self.covered_topics),
            'final_quality_score': self.search_history[-1].quality_score if self.search_history else 0.0
        }
    
    def _extract_section(self, text: str, section_header: str) -> List[str]:
        """Extract items from a specific section of formatted text."""
        items = []
        lines = text.split('\n')
        in_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith(section_header):
                in_section = True
                continue
            elif in_section and line.startswith(('GAPS:', 'MISSING_TOPICS:', 'SUGGESTED_SEARCHES:')):
                break
            elif in_section and line.startswith('- '):
                items.append(line[2:].strip())
        
        return items[:10]  # Limit to 10 items per section
    
    def _initialize_search_strategies(self) -> List[Dict[str, Any]]:
        """Initialize search strategies for iterative search."""
        return [
            {
                'name': 'methodological_focus',
                'description': 'Focus on research methods and techniques',
                'query_template': '{topic} methods techniques approaches',
                'expected_source_types': ['methodology_papers', 'technical_reports'],
                'priority': 0.8,
                'estimated_coverage': ['methods', 'techniques', 'approaches', 'implementation']
            },
            {
                'name': 'application_focus',
                'description': 'Focus on practical applications and case studies',
                'query_template': '{topic} applications practical use cases deployment',
                'expected_source_types': ['case_studies', 'application_papers'],
                'priority': 0.7,
                'estimated_coverage': ['applications', 'practical', 'deployment', 'real-world']
            },
            {
                'name': 'comparative_analysis',
                'description': 'Focus on comparisons and benchmarks',
                'query_template': '{topic} comparison benchmark evaluation analysis',
                'expected_source_types': ['comparative_studies', 'benchmark_papers'],
                'priority': 0.6,
                'estimated_coverage': ['comparison', 'benchmark', 'evaluation', 'analysis']
            },
            {
                'name': 'recent_developments',
                'description': 'Focus on recent developments and trends',
                'query_template': '{topic} recent latest 2023 2024 emerging',
                'expected_source_types': ['recent_papers', 'trend_analysis'],
                'priority': 0.9,
                'estimated_coverage': ['recent', 'latest', 'emerging', 'trends']
            },
            {
                'name': 'theoretical_foundations',
                'description': 'Focus on theoretical and foundational work',
                'query_template': '{topic} theory theoretical foundations mathematical',
                'expected_source_types': ['theoretical_papers', 'foundational_work'],
                'priority': 0.5,
                'estimated_coverage': ['theory', 'theoretical', 'foundations', 'mathematical']
            },
            {
                'name': 'interdisciplinary_connections',
                'description': 'Explore interdisciplinary connections',
                'query_template': '{topic} interdisciplinary cross-domain multi-disciplinary',
                'expected_source_types': ['interdisciplinary_papers', 'cross_domain'],
                'priority': 0.4,
                'estimated_coverage': ['interdisciplinary', 'cross-domain', 'multi-disciplinary']
            }
        ]


async def test_search_strategist():
    """Test function for the search strategist."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    strategist = EnhancedAISearchStrategist(api_key, debug_mode=True)
    
    query = "machine learning transformer architectures"
    
    # Simulate initial sources
    initial_sources = [
        {
            'title': 'Attention Is All You Need',
            'abstract': 'We propose a new simple network architecture, the Transformer...',
            'year': 2017,
            'venue': 'NIPS',
            'citations': 45000
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'abstract': 'We introduce a new language representation model called BERT...',
            'year': 2019,
            'venue': 'NAACL',
            'citations': 35000
        }
    ]
    
    print(f"🎯 Testing iterative search for: {query}")
    results = await strategist.conduct_iterative_search(query, initial_sources)
    
    print(f"\nTotal Iterations: {results['total_iterations']}")
    print(f"Final Sources: {results['total_sources']}")
    print(f"Coverage Areas: {', '.join(results['coverage_areas'])}")


if __name__ == "__main__":
    asyncio.run(test_search_strategist())
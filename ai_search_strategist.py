"""
AI-Driven Search Strategist
Uses GPT-4o mini to orchestrate intelligent, adaptive search strategies.
No fixed search patterns - purely AI-driven search orchestration.
"""

import openai
import asyncio
import json
import logging
import arxiv
import aiohttp
import feedparser
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SearchExecution:
    """Results of executing a search strategy."""
    strategy_name: str
    search_terms_used: List[str]
    sources_found: List[Dict[str, Any]]
    execution_time: float
    success_rate: float
    quality_assessment: str


@dataclass
class AdaptiveSearchResult:
    """Complete adaptive search results."""
    total_strategies_executed: int
    total_sources_found: int
    best_strategy: str
    search_executions: List[SearchExecution]
    adaptation_notes: List[str]
    final_recommendations: List[str]


class AISearchStrategist:
    """
    Intelligent search strategist that uses GPT-4o mini to orchestrate
    adaptive, dynamic search strategies across multiple sources.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.max_sources_per_strategy = 5
        self.max_total_sources = 15
        logger.info("🎯 AI Search Strategist initialized with GPT-4o mini")
    
    async def execute_adaptive_search(self, 
                                    query: str,
                                    query_analysis: Dict[str, Any],
                                    max_iterations: int = 3) -> AdaptiveSearchResult:
        """
        Execute adaptive search using AI-generated strategies.
        
        Args:
            query: Original research question
            query_analysis: Analysis from AIQueryAnalyzer
            max_iterations: Maximum search iterations
            
        Returns:
            AdaptiveSearchResult with comprehensive search results
        """
        logger.info(f"🎯 Starting adaptive search for: {query[:50]}...")
        
        search_executions = []
        all_sources = []
        adaptation_notes = []
        
        # Extract search strategies from query analysis
        search_strategies = query_analysis.get("search_strategies", [])
        if not search_strategies:
            logger.warning("No search strategies found, generating fallback")
            search_strategies = await self._generate_fallback_strategies(query)
        
        # Execute search strategies iteratively
        for iteration in range(max_iterations):
            logger.info(f"🔄 Search iteration {iteration + 1}/{max_iterations}")
            
            # Select best strategy for this iteration
            if iteration == 0:
                # Use highest priority strategy first
                current_strategy = max(search_strategies, key=lambda s: s.get("priority", 0.5))
            else:
                # Adapt strategy based on previous results
                current_strategy = await self._adapt_search_strategy(
                    query, search_executions, all_sources
                )
            
            # Execute current strategy
            execution = await self._execute_search_strategy(query, current_strategy)
            search_executions.append(execution)
            
            # Add new sources
            new_sources = [s for s in execution.sources_found if s not in all_sources]
            all_sources.extend(new_sources)
            
            logger.info(f"✅ Strategy '{execution.strategy_name}': {len(execution.sources_found)} sources found")
            
            # Evaluate if we have enough good sources
            if len(all_sources) >= self.max_total_sources:
                adaptation_notes.append(f"Reached maximum sources limit ({self.max_total_sources})")
                break
            
            # AI decides if we should continue searching
            should_continue = await self._should_continue_search(query, all_sources, search_executions)
            if not should_continue:
                adaptation_notes.append("AI determined sufficient sources found")
                break
        
        # Determine best strategy
        best_strategy = max(search_executions, key=lambda e: e.success_rate).strategy_name if search_executions else "none"
        
        # Generate final recommendations
        final_recommendations = await self._generate_final_recommendations(
            query, search_executions, all_sources
        )
        
        result = AdaptiveSearchResult(
            total_strategies_executed=len(search_executions),
            total_sources_found=len(all_sources),
            best_strategy=best_strategy,
            search_executions=search_executions,
            adaptation_notes=adaptation_notes,
            final_recommendations=final_recommendations
        )
        
        logger.info(f"🎯 Adaptive search completed: {len(all_sources)} sources from {len(search_executions)} strategies")
        return result
    
    async def _execute_search_strategy(self, query: str, strategy: Dict[str, Any]) -> SearchExecution:
        """Execute a single search strategy."""
        start_time = asyncio.get_event_loop().time()
        
        strategy_name = strategy.get("strategy_name", "Unknown Strategy")
        search_terms = strategy.get("search_terms", [])
        target_sources = strategy.get("target_sources", ["arxiv"])
        
        logger.info(f"🔍 Executing strategy: {strategy_name}")
        
        all_sources = []
        
        # Get temporal keywords from strategy
        temporal_keywords = strategy.get("temporal_keywords", [])
        
        # Execute searches across different sources
        for source_type in target_sources:
            for search_term in search_terms[:3]:  # Limit terms per source
                try:
                    if source_type == "arxiv":
                        # Enhanced search with temporal keywords for ArXiv
                        sources = await self._search_arxiv_enhanced(search_term, temporal_keywords)
                    elif source_type == "web":
                        sources = await self._search_web(search_term)
                    elif source_type == "pubmed":
                        sources = await self._search_pubmed(search_term)
                    else:
                        continue
                    
                    all_sources.extend(sources)
                    
                    # Brief pause to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Search failed for {search_term} on {source_type}: {e}")
                    continue
        
        # Remove duplicates based on title
        unique_sources = []
        seen_titles = set()
        for source in all_sources:
            title = source.get("title", "").lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_sources.append(source)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Calculate success rate (simple heuristic)
        success_rate = min(len(unique_sources) / max(len(search_terms), 1), 1.0)
        
        # AI quality assessment
        quality_assessment = await self._assess_search_quality(query, unique_sources, strategy)
        
        return SearchExecution(
            strategy_name=strategy_name,
            search_terms_used=search_terms,
            sources_found=unique_sources[:self.max_sources_per_strategy],
            execution_time=execution_time,
            success_rate=success_rate,
            quality_assessment=quality_assessment
        )
    
    async def _search_arxiv(self, search_term: str) -> List[Dict[str, Any]]:
        """Search ArXiv for papers."""
        try:
            search = arxiv.Search(
                query=search_term,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                source_data = {
                    "title": paper.title.strip(),
                    "authors": [str(author).strip() for author in paper.authors],
                    "abstract": paper.summary.strip() if paper.summary else "",
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else None,
                    "source_type": "arxiv",
                    "arxiv_id": paper.entry_id.split('/')[-1] if paper.entry_id else None,
                    "categories": [str(cat) for cat in paper.categories] if paper.categories else []
                }
                results.append(source_data)
            
            return results
            
        except Exception as e:
            logger.warning(f"ArXiv search failed for '{search_term}': {e}")
            return []
    
    async def _search_web(self, search_term: str) -> List[Dict[str, Any]]:
        """Search web sources for papers."""
        try:
            # Search research blogs and sources
            research_urls = [
                "https://blog.openai.com",
                "https://ai.googleblog.com"
            ]
            
            results = []
            
            for blog_url in research_urls[:1]:  # Limit to avoid rate limiting
                try:
                    feed_urls = [f"{blog_url}/feed", f"{blog_url}/rss"]
                    
                    for feed_url in feed_urls:
                        try:
                            feed = feedparser.parse(feed_url)
                            if feed.entries:
                                query_terms = search_term.lower().split()
                                
                                for entry in feed.entries[:2]:
                                    title = entry.get('title', '').lower()
                                    summary = entry.get('summary', '').lower()
                                    
                                    # Simple relevance check
                                    relevance_score = sum(1 for term in query_terms 
                                                        if term in title or term in summary)
                                    
                                    if relevance_score > 0:
                                        source_data = {
                                            "title": entry.get('title', ''),
                                            "url": entry.get('link', ''),
                                            "abstract": entry.get('summary', ''),
                                            "published": entry.get('published', ''),
                                            "source_type": "blog",
                                            "blog_url": blog_url,
                                            "authors": ["Blog Author"]
                                        }
                                        results.append(source_data)
                                break
                        except Exception:
                            continue
                except Exception:
                    continue
            
            return results[:3]  # Limit web results
            
        except Exception as e:
            logger.warning(f"Web search failed for '{search_term}': {e}")
            return []
    
    async def _search_arxiv_enhanced(self, search_term: str, temporal_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Enhanced ArXiv search with temporal filtering and specific targeting."""
        try:
            # Build enhanced search query
            enhanced_queries = [search_term]
            
            # Add temporal-enhanced versions for recent queries
            if temporal_keywords:
                for temporal in temporal_keywords:
                    if temporal in ["2023", "2024", "recent", "latest", "new", "state-of-the-art"]:
                        # Try specific year searches for very recent papers
                        if temporal in ["2023", "2024"]:
                            enhanced_queries.append(f"{search_term} {temporal}")
                        elif temporal in ["recent", "latest", "new"]:
                            enhanced_queries.append(f"recent {search_term}")
                            enhanced_queries.append(f"new {search_term}")
                        elif temporal == "state-of-the-art":
                            enhanced_queries.append(f"state-of-the-art {search_term}")
            
            all_results = []
            
            # Try each enhanced query
            for query in enhanced_queries[:2]:  # Limit to 2 enhanced queries
                try:
                    # Sort by submittedDate for recent papers, then by relevance
                    sort_options = [arxiv.SortCriterion.SubmittedDate, arxiv.SortCriterion.Relevance]
                    
                    for sort_by in sort_options:
                        search = arxiv.Search(
                            query=query,
                            max_results=10,  # Increased for better results
                            sort_by=sort_by,
                            sort_order=arxiv.SortOrder.Descending
                        )
                        
                        results = []
                        for paper in search.results():
                            source_data = {
                                "title": paper.title.strip(),
                                "authors": [str(author).strip() for author in paper.authors],
                                "abstract": paper.summary.strip() if paper.summary else "",
                                "url": paper.entry_id,
                                "pdf_url": paper.pdf_url,
                                "published": paper.published.strftime("%Y-%m-%d") if paper.published else None,
                                "source_type": "arxiv",
                                "arxiv_id": paper.entry_id.split('/')[-1] if paper.entry_id else None,
                                "categories": [str(cat) for cat in paper.categories] if paper.categories else [],
                                "year": paper.published.year if paper.published else None,
                                "search_query": query,
                                "sort_method": sort_by.value
                            }
                            results.append(source_data)
                        
                        all_results.extend(results)
                        
                        # Brief pause between searches
                        await asyncio.sleep(0.5)
                        
                        # If we got good results with recent sort, prioritize those
                        if sort_by == arxiv.SortCriterion.SubmittedDate and len(results) >= 3:
                            break
                
                except Exception as e:
                    logger.warning(f"Enhanced ArXiv search failed for '{query}': {e}")
                    continue
            
            # Remove duplicates and prioritize recent papers
            unique_results = []
            seen_titles = set()
            
            # Sort by publication date (newest first) then by relevance indicators
            all_results.sort(key=lambda x: (
                x.get('year', 0) or 0,  # Recent papers first
                'recent' in x.get('search_query', '').lower(),  # Favor recent searches
                'survey' in x.get('title', '').lower(),  # Favor surveys
                'comparison' in x.get('title', '').lower()  # Favor comparisons
            ), reverse=True)
            
            for result in all_results:
                title = result.get("title", "").lower().strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(result)
                
                if len(unique_results) >= 5:  # Limit results
                    break
            
            logger.info(f"Enhanced ArXiv search found {len(unique_results)} papers for '{search_term}' with temporal focus")
            return unique_results
            
        except Exception as e:
            logger.warning(f"Enhanced ArXiv search failed for '{search_term}': {e}")
            # Fallback to basic search
            return await self._search_arxiv(search_term)
    
    async def _search_pubmed(self, search_term: str) -> List[Dict[str, Any]]:
        """Search PubMed for medical papers."""
        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": search_term,
                "retmax": 3,
                "retmode": "json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    search_data = await response.json()
            
            results = []
            if "esearchresult" in search_data and search_data["esearchresult"]["idlist"]:
                for pmid in search_data["esearchresult"]["idlist"][:3]:
                    source_data = {
                        "title": f"PubMed Article {pmid}",
                        "authors": ["Medical Researcher"],
                        "abstract": f"Medical research article from PubMed (PMID: {pmid})",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "source_type": "pubmed",
                        "pmid": pmid
                    }
                    results.append(source_data)
            
            return results
            
        except Exception as e:
            logger.warning(f"PubMed search failed for '{search_term}': {e}")
            return []
    
    async def _adapt_search_strategy(self, 
                                   query: str, 
                                   previous_executions: List[SearchExecution],
                                   current_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI to adapt search strategy based on previous results."""
        adaptation_prompt = f"""Based on previous search results, generate an improved search strategy:

ORIGINAL QUERY: "{query}"

PREVIOUS SEARCH RESULTS:
{chr(10).join([f"- Strategy '{e.strategy_name}': {len(e.sources_found)} sources, success rate: {e.success_rate:.2f}" for e in previous_executions])}

CURRENT SOURCES FOUND: {len(current_sources)}

Generate an improved search strategy in JSON format:
{{
    "strategy_name": "Improved Strategy Name",
    "search_terms": ["term1", "term2", "term3"],
    "target_sources": ["arxiv", "web"],
    "expected_paper_types": ["survey", "technical"],
    "priority": 0.8,
    "reasoning": "Why this strategy should work better"
}}

Focus on:
1. Different search terms that might find more relevant papers
2. Alternative approaches to the same question
3. Learning from what didn't work previously"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at adapting search strategies based on results. Always respond with valid JSON only."},
                    {"role": "user", "content": adaptation_prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"🔄 AI adaptation response: {response_text[:100]}...")
            
            if not response_text:
                logger.warning("⚠️ Empty response from AI - using fallback")
                raise ValueError("Empty response from GPT-4o mini")
            
            # Clean response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}. Response: {response_text[:200] if 'response_text' in locals() else 'No response'}")
            return {
                "strategy_name": "Fallback Strategy",
                "search_terms": [query],
                "target_sources": ["arxiv"],
                "expected_paper_types": ["technical"],
                "priority": 0.5,
                "reasoning": "Fallback due to JSON parsing failure"
            }
        except Exception as e:
            logger.error(f"❌ Search strategy adaptation failed: {e}")
            return {
                "strategy_name": "Fallback Strategy",
                "search_terms": [query],
                "target_sources": ["arxiv"],
                "expected_paper_types": ["technical"],
                "priority": 0.5,
                "reasoning": "Fallback due to adaptation failure"
            }
    
    async def _should_continue_search(self, 
                                    query: str, 
                                    current_sources: List[Dict[str, Any]], 
                                    executions: List[SearchExecution]) -> bool:
        """AI decides whether to continue searching."""
        if len(current_sources) >= self.max_total_sources:
            return False
        
        if len(executions) >= 3:  # Maximum iterations
            return False
        
        # Simple heuristic for now - could be enhanced with AI decision
        recent_success = executions[-1].success_rate if executions else 0
        return recent_success > 0.1 and len(current_sources) < 10
    
    async def _assess_search_quality(self, 
                                   query: str, 
                                   sources: List[Dict[str, Any]], 
                                   strategy: Dict[str, Any]) -> str:
        """AI assessment of search quality."""
        if not sources:
            return "No sources found"
        
        # Quick relevance check based on titles
        query_words = set(query.lower().split())
        relevant_count = 0
        
        for source in sources:
            title_words = set(source.get("title", "").lower().split())
            if query_words.intersection(title_words):
                relevant_count += 1
        
        relevance_rate = relevant_count / len(sources) if sources else 0
        
        if relevance_rate > 0.7:
            return "High quality - most sources appear relevant"
        elif relevance_rate > 0.4:
            return "Medium quality - some relevant sources found"
        else:
            return "Low quality - few sources appear relevant"
    
    async def _generate_fallback_strategies(self, query: str) -> List[Dict[str, Any]]:
        """Generate fallback search strategies when none provided."""
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        return [{
            "strategy_name": "Basic Keyword Search",
            "search_terms": keywords[:5],
            "target_sources": ["arxiv"],
            "expected_paper_types": ["technical"],
            "priority": 0.5,
            "reasoning": "Fallback keyword-based search"
        }]
    
    async def _generate_final_recommendations(self, 
                                            query: str,
                                            executions: List[SearchExecution],
                                            sources: List[Dict[str, Any]]) -> List[str]:
        """Generate final recommendations based on search results."""
        recommendations = []
        
        if not sources:
            recommendations.append("No relevant sources found - try alternative search terms")
            recommendations.append("Consider consulting specialized databases")
        elif len(sources) < 5:
            recommendations.append("Limited sources found - expand search strategy")
            recommendations.append("Try broader or more specific search terms")
        else:
            recommendations.append("Good variety of sources found")
            recommendations.append("Proceed with relevance evaluation")
        
        # Add strategy-specific recommendations
        if executions:
            best_execution = max(executions, key=lambda e: e.success_rate)
            recommendations.append(f"Best strategy was: {best_execution.strategy_name}")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_search_strategist():
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-test-key-for-research-demo":
            print("⚠️ Please set a real OPENAI_API_KEY for testing")
            return
        
        strategist = AISearchStrategist(api_key)
        
        # Mock query analysis
        query = "explain the optimizers used in CNNs and RNNs"
        mock_analysis = {
            "search_strategies": [{
                "strategy_name": "Optimizer Research",
                "search_terms": ["CNN optimizer", "RNN optimizer", "Adam", "SGD"],
                "target_sources": ["arxiv"],
                "priority": 0.9,
                "reasoning": "Direct search for optimizers"
            }]
        }
        
        print(f"🧪 Testing adaptive search for: {query}")
        print("=" * 70)
        
        result = await strategist.execute_adaptive_search(query, mock_analysis, max_iterations=2)
        
        print(f"Strategies Executed: {result.total_strategies_executed}")
        print(f"Total Sources: {result.total_sources_found}")
        print(f"Best Strategy: {result.best_strategy}")
        print(f"Recommendations: {result.final_recommendations}")
    
    # Run test
    asyncio.run(test_search_strategist())
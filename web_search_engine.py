"""
Comprehensive Web Search Engine
Like Claude/Gemini Deep Research - searches multiple web sources for comprehensive results.
"""

import requests
import asyncio
import aiohttp
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """Web search result."""
    title: str
    url: str
    snippet: str
    source: str  # 'duckduckgo', 'wikipedia', 'reddit', 'news'
    date: str
    relevance_score: float


class ComprehensiveWebSearchEngine:
    """
    Comprehensive web search engine that searches multiple sources
    like Claude/Gemini Deep Research.
    """
    
    def __init__(self, max_results_per_source: int = 5, timeout: int = 10):
        self.max_results_per_source = max_results_per_source
        self.timeout = timeout
        
        # User agent for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    
    def _calculate_relevance_score(self, query: str, title: str, snippet: str) -> float:
        """
        Calculate relevance score between query and content.
        Like Gemini Deep Research - intelligent relevance scoring.
        """
        try:
            query_terms = set(query.lower().split())
            
            # Clean and tokenize content
            title_terms = set(title.lower().split())
            snippet_terms = set(snippet.lower().split())
            content_terms = title_terms.union(snippet_terms)
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(content_terms))
            total_query_terms = len(query_terms)
            
            if total_query_terms == 0:
                return 0.0
            
            # Base score from term overlap
            base_score = overlap / total_query_terms
            
            # Boost score if query appears as phrase in content
            query_phrase = query.lower()
            content_text = f"{title} {snippet}".lower()
            
            if query_phrase in content_text:
                base_score += 0.3
            
            # Penalize placeholder/generic content
            placeholder_indicators = [
                "lorem ipsum", "example.com", "placeholder", "dummy",
                "generic", "template", "sample", "test content",
                f"about {query.lower()}", f"information about {query.lower()}"
            ]
            
            for indicator in placeholder_indicators:
                if indicator in content_text:
                    base_score *= 0.3  # Heavy penalty for placeholder content
            
            # Cap at 1.0
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            return 0.5  # Default neutral score
    
    
    def _search_duckduckgo(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """
        Search DuckDuckGo for web results.
        DuckDuckGo doesn't require API keys and provides good results.
        """
        try:
            results = []
            
            # DuckDuckGo instant answers API
            ddg_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(ddg_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get instant answer if available
                if data.get('Abstract'):
                    result = WebSearchResult(
                        title=data.get('Heading', f"DuckDuckGo: {query}"),
                        url=data.get('AbstractURL', f"https://duckduckgo.com/?q={quote_plus(query)}"),
                        snippet=data['Abstract'][:300] + "..." if len(data['Abstract']) > 300 else data['Abstract'],
                        source='duckduckgo',
                        date=datetime.now().strftime('%Y-%m-%d'),
                        relevance_score=0.9
                    )
                    results.append(result)
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        result = WebSearchResult(
                            title=topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', '')[:100],
                            url=topic.get('FirstURL', f"https://duckduckgo.com/?q={quote_plus(query)}"),
                            snippet=topic.get('Text', '')[:300],
                            source='duckduckgo',
                            date=datetime.now().strftime('%Y-%m-%d'),
                            relevance_score=0.8
                        )
                        results.append(result)
                        if len(results) >= max_results:
                            break
            
            logger.info(f"🦆 DuckDuckGo found {len(results)} results")
            return results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []
    
    
    def _search_wikipedia(self, query: str, max_results: int = 3) -> List[WebSearchResult]:
        """
        Search Wikipedia for comprehensive, reliable information.
        """
        try:
            results = []
            
            # Wikipedia search API
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
            
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('extract'):
                    result = WebSearchResult(
                        title=data.get('title', query),
                        url=data.get('content_urls', {}).get('desktop', {}).get('page', f"https://en.wikipedia.org/wiki/{quote_plus(query)}"),
                        snippet=data['extract'][:400] + "..." if len(data['extract']) > 400 else data['extract'],
                        source='wikipedia',
                        date=datetime.now().strftime('%Y-%m-%d'),
                        relevance_score=0.95  # Wikipedia is highly reliable
                    )
                    results.append(result)
            
            # If direct search fails, try Wikipedia search API
            if not results:
                search_api_url = f"https://en.wikipedia.org/api/rest_v1/page/search/{quote_plus(query)}"
                response = requests.get(search_api_url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for page in data.get('pages', [])[:max_results]:
                        result = WebSearchResult(
                            title=page.get('title', ''),
                            url=f"https://en.wikipedia.org/wiki/{quote_plus(page.get('key', ''))}",
                            snippet=page.get('description', '') + ". " + page.get('extract', '')[:200],
                            source='wikipedia',
                            date=datetime.now().strftime('%Y-%m-%d'),
                            relevance_score=0.9
                        )
                        results.append(result)
            
            logger.info(f"📖 Wikipedia found {len(results)} results")
            return results
            
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
            return []
    
    
    def _search_reddit(self, query: str, max_results: int = 3) -> List[WebSearchResult]:
        """
        Search Reddit for community insights and discussions.
        """
        try:
            results = []
            
            # Reddit search API (no auth required for search)
            reddit_url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&sort=relevance&limit={max_results}"
            
            response = requests.get(reddit_url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data.get('data', {}).get('children', [])[:max_results]:
                    post_data = post.get('data', {})
                    
                    # Skip if no content
                    if not post_data.get('title'):
                        continue
                    
                    snippet = post_data.get('selftext', '')[:300]
                    if not snippet:
                        snippet = f"Discussion: {post_data.get('title', '')}"
                    
                    result = WebSearchResult(
                        title=post_data.get('title', '')[:100],
                        url=f"https://www.reddit.com{post_data.get('permalink', '')}",
                        snippet=snippet,
                        source='reddit',
                        date=datetime.fromtimestamp(post_data.get('created_utc', time.time())).strftime('%Y-%m-%d'),
                        relevance_score=0.7  # Reddit is good for insights but less authoritative
                    )
                    results.append(result)
            
            logger.info(f"🔴 Reddit found {len(results)} results")
            return results
            
        except Exception as e:
            logger.warning(f"Reddit search failed: {e}")
            return []
    
    
    def _search_news(self, query: str, max_results: int = 3) -> List[WebSearchResult]:
        """
        Search for recent news articles.
        DISABLED: Fake news generation removed for citation integrity.
        """
        # Disable fake news generation to prevent citation integrity issues
        logger.info("📰 News search disabled (prevents fake results)")
        return []
        
        # TODO: Implement real news API integration:
        # - NewsAPI (newsapi.org) for real news articles
        # - Google News RSS feeds
        # - Bing News API
        # Only return results with actual news content validation
    
    
    def _filter_by_strategy(self, results: List[WebSearchResult], strategy: str) -> List[WebSearchResult]:
        """
        Filter and prioritize results based on search strategy.
        """
        if strategy == 'recent':
            # Prioritize news and recent content
            results.sort(key=lambda x: (
                1 if x.source == 'news' else 0,
                x.relevance_score,
                x.date
            ), reverse=True)
        
        elif strategy == 'historical':
            # Prioritize Wikipedia and established sources
            results.sort(key=lambda x: (
                1 if x.source == 'wikipedia' else 0,
                x.relevance_score
            ), reverse=True)
        
        else:  # comprehensive
            # Balanced approach
            results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    
    def search_comprehensive(self, query: str, strategy: str = 'comprehensive') -> List[WebSearchResult]:
        """
        Comprehensive web search like Claude/Gemini Deep Research.
        Searches multiple sources and combines results.
        """
        logger.info(f"🔍 Starting comprehensive web search for: {query} (strategy: {strategy})")
        
        all_results = []
        
        # Search multiple sources in parallel for speed
        search_tasks = [
            ('duckduckgo', self._search_duckduckgo(query, self.max_results_per_source)),
            ('wikipedia', self._search_wikipedia(query, 3)),
            ('reddit', self._search_reddit(query, 2)),
            ('news', self._search_news(query, 2))
        ]
        
        # Combine all results
        for source_name, results in search_tasks:
            all_results.extend(results)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # CRITICAL: Apply relevance scoring and quality gates
        quality_filtered_results = []
        for result in unique_results:
            # Recalculate relevance score for all results
            actual_relevance = self._calculate_relevance_score(query, result.title, result.snippet)
            result.relevance_score = actual_relevance
            
            # Quality gate: Only include results with relevance > 0.3
            if actual_relevance > 0.3:
                quality_filtered_results.append(result)
            else:
                logger.info(f"🚫 Filtered low-relevance result: '{result.title}' (score: {actual_relevance:.2f})")
        
        logger.info(f"🎯 Quality filtering: {len(unique_results)} → {len(quality_filtered_results)} results")
        
        # Filter and prioritize based on strategy
        filtered_results = self._filter_by_strategy(quality_filtered_results, strategy)
        
        logger.info(f"✅ Comprehensive web search completed: {len(filtered_results)} unique results")
        return filtered_results
    
    
    async def search_comprehensive_async(self, query: str, strategy: str = 'comprehensive') -> List[WebSearchResult]:
        """
        Async version of comprehensive search for better performance.
        """
        # For now, use the sync version
        # In production, implement async versions of search methods
        return self.search_comprehensive(query, strategy)


# Factory function
def create_web_search_engine(max_results_per_source: int = 5) -> ComprehensiveWebSearchEngine:
    """Create a comprehensive web search engine."""
    return ComprehensiveWebSearchEngine(max_results_per_source=max_results_per_source)


if __name__ == "__main__":
    # Test the web search engine
    engine = create_web_search_engine()
    results = engine.search_comprehensive("machine learning applications")
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"[{result.source}] {result.title}")
        print(f"  URL: {result.url}")
        print(f"  Snippet: {result.snippet[:100]}...")
        print()
"""
Advanced Search Engine
Multi-source search engine with Semantic Scholar, ArXiv, and Google Scholar integration.
"""

import asyncio
import aiohttp
import arxiv
import requests
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import quote
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class SearchSource:
    """Represents a search source with metadata."""
    name: str
    base_url: str
    api_key_required: bool
    rate_limit: float  # seconds between requests
    max_results: int


@dataclass
class EnhancedSource:
    """Enhanced source with quality metrics."""
    title: str
    abstract: str
    authors: List[str]
    year: int
    venue: str
    url: str
    citations: int
    source_type: str  # 'semantic_scholar', 'arxiv', 'google_scholar'
    quality_score: float
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    metadata: Dict[str, Any] = None


class AdvancedSearchEngine:
    """
    Advanced multi-source search engine for academic research.
    Integrates Semantic Scholar, ArXiv, and Google Scholar.
    """
    
    def __init__(self, 
                 max_sources_per_engine: int = 15,
                 semantic_scholar_api_key: Optional[str] = None,
                 serpapi_api_key: Optional[str] = None):
        
        self.max_sources_per_engine = max_sources_per_engine
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.serpapi_api_key = serpapi_api_key
        
        # Define search sources
        self.search_sources = {
            'semantic_scholar': SearchSource(
                name='Semantic Scholar',
                base_url='https://api.semanticscholar.org/graph/v1/paper/search',
                api_key_required=False,
                rate_limit=1.0,  # 1 second between requests
                max_results=20
            ),
            'arxiv': SearchSource(
                name='ArXiv',
                base_url='https://export.arxiv.org/api/query',
                api_key_required=False,
                rate_limit=3.0,  # 3 seconds (ArXiv requirement)
                max_results=15
            ),
            'google_scholar': SearchSource(
                name='Google Scholar',
                base_url='https://serpapi.com/search',
                api_key_required=True,
                rate_limit=1.0,
                max_results=10
            )
        }
        
        self.last_request_time = defaultdict(float)
        
        logger.info("🔍 Advanced Search Engine initialized")
        if semantic_scholar_api_key:
            logger.info("✅ Semantic Scholar API key provided")
        if serpapi_api_key:
            logger.info("✅ Google Scholar (SerpAPI) key provided")
        else:
            logger.info("⚠️ Google Scholar unavailable (no SerpAPI key)")
    
    async def multi_source_search(self, query: str) -> List[EnhancedSource]:
        """
        Conduct comprehensive search across multiple academic sources.
        
        Args:
            query: Search query
            
        Returns:
            List of EnhancedSource objects ranked by quality
        """
        logger.info(f"🔍 Starting multi-source search for: {query}")
        
        # Execute searches in parallel
        search_tasks = [
            self._search_semantic_scholar(query),
            self._search_arxiv(query)
        ]
        
        # Add Google Scholar if API key available
        if self.serpapi_api_key:
            search_tasks.append(self._search_google_scholar(query))
        
        try:
            # Execute all searches concurrently
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and process results
            all_sources = []
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    source_name = list(self.search_sources.keys())[i]
                    logger.error(f"❌ {source_name} search failed: {result}")
                    continue
                
                if isinstance(result, list):
                    all_sources.extend(result)
            
            # Remove duplicates and rank by quality
            unique_sources = self._remove_duplicates(all_sources)
            ranked_sources = self._rank_sources_by_quality(unique_sources)
            
            logger.info(f"✅ Found {len(ranked_sources)} unique sources across all engines")
            
            return ranked_sources[:self.max_sources_per_engine * 2]  # Return top results
            
        except Exception as e:
            logger.error(f"❌ Multi-source search failed: {e}")
            return []
    
    async def _search_semantic_scholar(self, query: str) -> List[EnhancedSource]:
        """Search Semantic Scholar API."""
        try:
            await self._rate_limit('semantic_scholar')
            
            # Prepare query parameters
            params = {
                'query': query,
                'limit': self.search_sources['semantic_scholar'].max_results,
                'fields': 'title,abstract,authors,year,venue,citationCount,url,externalIds'
            }
            
            headers = {}
            if self.semantic_scholar_api_key:
                headers['x-api-key'] = self.semantic_scholar_api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.search_sources['semantic_scholar'].base_url,
                    params=params,
                    headers=headers
                ) as response:
                    
                    if response.status != 200:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = data.get('data', [])
                    
                    sources = []
                    for paper in papers:
                        try:
                            source = self._process_semantic_scholar_paper(paper)
                            if source:
                                sources.append(source)
                        except Exception as e:
                            logger.warning(f"Failed to process Semantic Scholar paper: {e}")
                            continue
                    
                    logger.info(f"📊 Semantic Scholar: {len(sources)} sources found")
                    return sources
                    
        except Exception as e:
            logger.error(f"❌ Semantic Scholar search failed: {e}")
            return []
    
    async def _search_arxiv(self, query: str) -> List[EnhancedSource]:
        """Search ArXiv API."""
        try:
            await self._rate_limit('arxiv')
            
            # Use arxiv library for search
            search = arxiv.Search(
                query=query,
                max_results=self.search_sources['arxiv'].max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            sources = []
            for paper in search.results():
                try:
                    source = self._process_arxiv_paper(paper)
                    if source:
                        sources.append(source)
                except Exception as e:
                    logger.warning(f"Failed to process ArXiv paper: {e}")
                    continue
            
            logger.info(f"📊 ArXiv: {len(sources)} sources found")
            return sources
            
        except Exception as e:
            logger.error(f"❌ ArXiv search failed: {e}")
            return []
    
    async def _search_google_scholar(self, query: str) -> List[EnhancedSource]:
        """Search Google Scholar via SerpAPI."""
        if not self.serpapi_api_key:
            logger.warning("⚠️ Google Scholar search skipped (no SerpAPI key)")
            return []
        
        try:
            await self._rate_limit('google_scholar')
            
            params = {
                'engine': 'google_scholar',
                'q': query,
                'api_key': self.serpapi_api_key,
                'num': self.search_sources['google_scholar'].max_results
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.search_sources['google_scholar'].base_url,
                    params=params
                ) as response:
                    
                    if response.status != 200:
                        logger.error(f"Google Scholar API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    results = data.get('organic_results', [])
                    
                    sources = []
                    for result in results:
                        try:
                            source = self._process_google_scholar_result(result)
                            if source:
                                sources.append(source)
                        except Exception as e:
                            logger.warning(f"Failed to process Google Scholar result: {e}")
                            continue
                    
                    logger.info(f"📊 Google Scholar: {len(sources)} sources found")
                    return sources
                    
        except Exception as e:
            logger.error(f"❌ Google Scholar search failed: {e}")
            return []
    
    def _process_semantic_scholar_paper(self, paper: Dict[str, Any]) -> Optional[EnhancedSource]:
        """Process Semantic Scholar paper into EnhancedSource."""
        try:
            # Extract basic information
            title = paper.get('title', 'Unknown Title').strip()
            abstract = paper.get('abstract', 'No abstract available').strip()
            
            if not title or title == 'Unknown Title':
                return None
            
            # Extract authors
            authors = []
            for author in paper.get('authors', []):
                name = author.get('name', '').strip()
                if name:
                    authors.append(name)
            
            # Extract metadata
            year = paper.get('year', 0)
            venue = paper.get('venue', 'Unknown Venue').strip()
            citations = paper.get('citationCount', 0)
            url = paper.get('url', '')
            
            # Extract DOI
            doi = None
            external_ids = paper.get('externalIds', {})
            if external_ids and 'DOI' in external_ids:
                doi = external_ids['DOI']
            
            # Calculate quality score
            quality_score = self._calculate_semantic_scholar_quality(paper)
            
            return EnhancedSource(
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue=venue,
                url=url,
                citations=citations,
                source_type='semantic_scholar',
                quality_score=quality_score,
                doi=doi,
                metadata=paper
            )
            
        except Exception as e:
            logger.warning(f"Failed to process Semantic Scholar paper: {e}")
            return None
    
    def _process_arxiv_paper(self, paper) -> Optional[EnhancedSource]:
        """Process ArXiv paper into EnhancedSource."""
        try:
            title = paper.title.strip()
            abstract = paper.summary.strip()
            
            if not title:
                return None
            
            # Extract authors
            authors = [author.name for author in paper.authors]
            
            # Extract metadata
            year = paper.published.year
            venue = 'arXiv'  # ArXiv preprint
            url = paper.entry_id
            pdf_url = paper.pdf_url
            
            # ArXiv papers don't have citation counts, estimate quality differently
            quality_score = self._calculate_arxiv_quality(paper)
            
            return EnhancedSource(
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue=venue,
                url=url,
                citations=0,  # ArXiv doesn't provide citation counts
                source_type='arxiv',
                quality_score=quality_score,
                pdf_url=pdf_url,
                metadata={'arxiv_id': paper.get_short_id()}
            )
            
        except Exception as e:
            logger.warning(f"Failed to process ArXiv paper: {e}")
            return None
    
    def _process_google_scholar_result(self, result: Dict[str, Any]) -> Optional[EnhancedSource]:
        """Process Google Scholar result into EnhancedSource."""
        try:
            title = result.get('title', '').strip()
            snippet = result.get('snippet', 'No abstract available').strip()
            
            if not title:
                return None
            
            # Extract publication info
            publication_info = result.get('publication_info', {})
            authors_str = publication_info.get('authors', '')
            authors = [a.strip() for a in authors_str.split(',') if a.strip()] if authors_str else []
            
            # Extract year from publication info
            year = 0
            summary = publication_info.get('summary', '')
            if summary:
                # Try to extract year from summary
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', summary)
                if year_match:
                    year = int(year_match.group())
            
            venue = publication_info.get('summary', 'Unknown Venue').strip()
            url = result.get('link', '')
            
            # Extract citations from "Cited by X" info
            citations = 0
            cited_by = result.get('inline_links', {}).get('cited_by', {})
            if cited_by and 'total' in cited_by:
                citations = cited_by['total']
            
            # Calculate quality score
            quality_score = self._calculate_google_scholar_quality(result)
            
            return EnhancedSource(
                title=title,
                abstract=snippet,
                authors=authors,
                year=year,
                venue=venue,
                url=url,
                citations=citations,
                source_type='google_scholar',
                quality_score=quality_score,
                metadata=result
            )
            
        except Exception as e:
            logger.warning(f"Failed to process Google Scholar result: {e}")
            return None
    
    def _calculate_semantic_scholar_quality(self, paper: Dict[str, Any]) -> float:
        """Calculate quality score for Semantic Scholar paper."""
        score = 0.0
        
        # Citation count (40% of score)
        citations = paper.get('citationCount', 0)
        if citations > 100:
            score += 0.4
        elif citations > 50:
            score += 0.3
        elif citations > 10:
            score += 0.2
        elif citations > 0:
            score += 0.1
        
        # Venue quality (30% of score)
        venue = paper.get('venue', '').lower()
        if any(top_venue in venue for top_venue in ['nature', 'science', 'cell', 'pnas']):
            score += 0.3
        elif any(conf in venue for conf in ['neurips', 'icml', 'iclr', 'acl', 'emnlp']):
            score += 0.25
        elif venue:
            score += 0.1
        
        # Recency (20% of score)
        year = paper.get('year', 0)
        if year >= 2023:
            score += 0.2
        elif year >= 2020:
            score += 0.15
        elif year >= 2015:
            score += 0.1
        elif year >= 2010:
            score += 0.05
        
        # Abstract availability (10% of score)
        if paper.get('abstract'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_arxiv_quality(self, paper) -> float:
        """Calculate quality score for ArXiv paper."""
        score = 0.0
        
        # Recency is more important for ArXiv (50% of score)
        year = paper.published.year
        if year >= 2023:
            score += 0.5
        elif year >= 2021:
            score += 0.4
        elif year >= 2019:
            score += 0.3
        elif year >= 2017:
            score += 0.2
        
        # Author count (20% of score) - more authors might indicate collaborative work
        author_count = len(paper.authors)
        if author_count >= 5:
            score += 0.2
        elif author_count >= 3:
            score += 0.15
        elif author_count >= 2:
            score += 0.1
        
        # Abstract quality (30% of score)
        abstract_length = len(paper.summary.split())
        if abstract_length >= 200:
            score += 0.3
        elif abstract_length >= 100:
            score += 0.2
        elif abstract_length >= 50:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_google_scholar_quality(self, result: Dict[str, Any]) -> float:
        """Calculate quality score for Google Scholar result."""
        score = 0.0
        
        # Citation count (50% of score)
        cited_by = result.get('inline_links', {}).get('cited_by', {})
        citations = cited_by.get('total', 0) if cited_by else 0
        
        if citations > 500:
            score += 0.5
        elif citations > 100:
            score += 0.4
        elif citations > 50:
            score += 0.3
        elif citations > 10:
            score += 0.2
        elif citations > 0:
            score += 0.1
        
        # Publication info quality (30% of score)
        pub_info = result.get('publication_info', {})
        if pub_info and pub_info.get('summary'):
            score += 0.3
        elif pub_info:
            score += 0.15
        
        # PDF availability (20% of score)
        resources = result.get('resources', [])
        has_pdf = any('pdf' in resource.get('title', '').lower() for resource in resources)
        if has_pdf:
            score += 0.2
        
        return min(score, 1.0)
    
    def _remove_duplicates(self, sources: List[EnhancedSource]) -> List[EnhancedSource]:
        """Remove duplicate sources based on title similarity."""
        if not sources:
            return []
        
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            # Normalize title for comparison
            normalized_title = source.title.lower().strip()
            
            # Check for exact duplicates
            if normalized_title in seen_titles:
                continue
            
            # Check for near duplicates (simple approach)
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sources.append(source)
                seen_titles.add(normalized_title)
        
        logger.info(f"🔍 Removed {len(sources) - len(unique_sources)} duplicate sources")
        return unique_sources
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar enough to be considered duplicates."""
        # Simple word-based similarity
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold
    
    def _rank_sources_by_quality(self, sources: List[EnhancedSource]) -> List[EnhancedSource]:
        """Rank sources by quality score."""
        return sorted(sources, key=lambda s: s.quality_score, reverse=True)
    
    async def _rate_limit(self, source_name: str):
        """Implement rate limiting for API requests."""
        if source_name not in self.search_sources:
            return
        
        rate_limit = self.search_sources[source_name].rate_limit
        last_request = self.last_request_time[source_name]
        current_time = time.time()
        
        time_since_last = current_time - last_request
        if time_since_last < rate_limit:
            sleep_time = rate_limit - time_since_last
            logger.debug(f"Rate limiting {source_name}: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time[source_name] = time.time()


async def test_advanced_search():
    """Test function for the advanced search engine."""
    import os
    
    # Initialize search engine
    engine = AdvancedSearchEngine(
        max_sources_per_engine=10,
        semantic_scholar_api_key=os.getenv('SEMANTIC_SCHOLAR_API_KEY'),
        serpapi_api_key=os.getenv('SERPAPI_API_KEY')
    )
    
    # Test query
    query = "transformer architecture machine learning"
    print(f"🔍 Testing search for: {query}")
    
    # Conduct search
    sources = await engine.multi_source_search(query)
    
    # Display results
    print(f"\n✅ Found {len(sources)} sources:")
    for i, source in enumerate(sources[:5], 1):
        print(f"\n{i}. {source.title}")
        print(f"   Authors: {', '.join(source.authors[:3])}")
        print(f"   Year: {source.year} | Citations: {source.citations}")
        print(f"   Source: {source.source_type} | Quality: {source.quality_score:.2f}")
        print(f"   Abstract: {source.abstract[:150]}...")


if __name__ == "__main__":
    asyncio.run(test_advanced_search())
"""
Advanced Multi-Source Academic Search Engine
Integrates multiple academic databases for comprehensive research coverage.
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import urllib.parse
import arxiv
import feedparser

logger = logging.getLogger(__name__)


@dataclass
class SourceMetrics:
    """Quality metrics for academic sources."""
    citation_count: int = 0
    h_index: Optional[int] = None
    venue_score: float = 0.0
    publication_year: int = 0
    author_count: int = 0
    is_highly_cited: bool = False
    venue_type: str = "unknown"  # journal, conference, preprint, etc.


@dataclass
class EnhancedSource:
    """Enhanced source with quality metrics and metadata."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    publication_year: int = 0
    venue: Optional[str] = None
    source_type: str = "unknown"  # arxiv, scholar, ieee, semantic, web
    source_id: Optional[str] = None
    
    # Quality metrics
    metrics: Optional[SourceMetrics] = None
    quality_score: float = 0.0
    relevance_score: float = 0.0
    
    # Additional metadata
    keywords: List[str] = None
    categories: List[str] = None
    language: str = "en"
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.categories is None:
            self.categories = []
        if self.metrics is None:
            self.metrics = SourceMetrics()


class GoogleScholarSearcher:
    """Google Scholar search integration using serpapi."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
        self.rate_limit_delay = 1.0  # seconds between requests
        
    async def search_papers(self, query: str, max_results: int = 20) -> List[EnhancedSource]:
        """Search Google Scholar for academic papers."""
        if not self.api_key:
            logger.warning("Google Scholar API key not provided, skipping Scholar search")
            return []
        
        try:
            logger.info(f"🎓 Searching Google Scholar: {query}")
            
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": min(max_results, 20),  # Scholar API limit
                "hl": "en"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_scholar_results(data)
                    else:
                        logger.error(f"Google Scholar API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            return []
    
    def _parse_scholar_results(self, data: Dict[str, Any]) -> List[EnhancedSource]:
        """Parse Google Scholar API response."""
        sources = []
        
        for result in data.get("organic_results", []):
            try:
                # Extract basic information
                title = result.get("title", "Unknown Title")
                authors = self._extract_authors(result.get("publication_info", {}).get("authors", []))
                abstract = result.get("snippet", "")
                url = result.get("link", "")
                
                # Extract citation count
                citation_info = result.get("inline_links", {}).get("cited_by", {})
                citation_count = citation_info.get("total", 0) if citation_info else 0
                
                # Extract publication info
                pub_info = result.get("publication_info", {})
                venue = pub_info.get("summary", "")
                year = self._extract_year(venue)
                
                # Create quality metrics
                metrics = SourceMetrics(
                    citation_count=citation_count,
                    publication_year=year,
                    author_count=len(authors),
                    is_highly_cited=citation_count > 100,
                    venue_type="journal" if "journal" in venue.lower() else "conference"
                )
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(metrics, year)
                
                source = EnhancedSource(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    publication_year=year,
                    venue=venue,
                    source_type="scholar",
                    metrics=metrics,
                    quality_score=quality_score
                )
                
                sources.append(source)
                
            except Exception as e:
                logger.warning(f"Failed to parse Scholar result: {e}")
                continue
        
        logger.info(f"🎓 Found {len(sources)} papers from Google Scholar")
        return sources
    
    def _extract_authors(self, authors_data: List[Dict[str, Any]]) -> List[str]:
        """Extract author names from publication info."""
        if isinstance(authors_data, list):
            return [author.get("name", "") for author in authors_data if isinstance(author, dict)]
        elif isinstance(authors_data, str):
            return [name.strip() for name in authors_data.split(",")]
        return []
    
    def _extract_year(self, text: str) -> int:
        """Extract publication year from text."""
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        return int(year_match.group(0)) if year_match else datetime.now().year
    
    def _calculate_quality_score(self, metrics: SourceMetrics, year: int) -> float:
        """Calculate quality score based on multiple factors."""
        current_year = datetime.now().year
        age_penalty = max(0, (current_year - year) * 0.02)  # 2% penalty per year
        
        citation_score = min(metrics.citation_count / 1000, 1.0)  # Normalize to 0-1
        recency_score = max(0, 1.0 - age_penalty)
        
        return (citation_score * 0.6 + recency_score * 0.4)


class SemanticScholarSearcher:
    """Semantic Scholar API integration for citation data and relationships."""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    async def search_papers(self, query: str, max_results: int = 20) -> List[EnhancedSource]:
        """Search Semantic Scholar for papers."""
        try:
            logger.info(f"🔬 Searching Semantic Scholar: {query}")
            
            search_url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": min(max_results, 100),  # API limit
                "fields": "paperId,title,abstract,authors,year,citationCount,venue,url,openAccessPdf"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_semantic_results(data)
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []
    
    def _parse_semantic_results(self, data: Dict[str, Any]) -> List[EnhancedSource]:
        """Parse Semantic Scholar API response."""
        sources = []
        
        for paper in data.get("data", []):
            try:
                # Extract basic information
                title = paper.get("title", "Unknown Title")
                abstract = paper.get("abstract") or ""
                year = paper.get("year", datetime.now().year)
                citation_count = paper.get("citationCount", 0)
                venue = paper.get("venue") or ""
                url = paper.get("url", "")
                paper_id = paper.get("paperId", "")
                
                # Extract authors
                authors = []
                for author in paper.get("authors", []):
                    if isinstance(author, dict) and "name" in author:
                        authors.append(author["name"])
                
                # PDF URL
                pdf_url = None
                if paper.get("openAccessPdf"):
                    pdf_url = paper["openAccessPdf"].get("url")
                
                # Create quality metrics
                metrics = SourceMetrics(
                    citation_count=citation_count,
                    publication_year=year,
                    author_count=len(authors),
                    is_highly_cited=citation_count > 50,
                    venue_type="journal" if venue else "conference"
                )
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(metrics, year)
                
                source = EnhancedSource(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    pdf_url=pdf_url,
                    publication_year=year,
                    venue=venue,
                    source_type="semantic",
                    source_id=paper_id,
                    metrics=metrics,
                    quality_score=quality_score
                )
                
                sources.append(source)
                
            except Exception as e:
                logger.warning(f"Failed to parse Semantic Scholar result: {e}")
                continue
        
        logger.info(f"🔬 Found {len(sources)} papers from Semantic Scholar")
        return sources
    
    def _calculate_quality_score(self, metrics: SourceMetrics, year: int) -> float:
        """Calculate quality score based on multiple factors."""
        current_year = datetime.now().year
        age_penalty = max(0, (current_year - year) * 0.02)
        
        citation_score = min(metrics.citation_count / 500, 1.0)  # Different scale for Semantic Scholar
        recency_score = max(0, 1.0 - age_penalty)
        
        return (citation_score * 0.7 + recency_score * 0.3)


class EnhancedArxivSearcher:
    """Enhanced ArXiv searcher with better filtering and quality assessment."""
    
    def __init__(self):
        self.rate_limit_delay = 1.0
        
    async def search_papers(self, query: str, max_results: int = 20) -> List[EnhancedSource]:
        """Search ArXiv with enhanced filtering."""
        try:
            logger.info(f"📚 Searching ArXiv (enhanced): {query}")
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            sources = []
            for paper in search.results():
                try:
                    # Extract basic information
                    title = paper.title.strip()
                    authors = [str(author).strip() for author in paper.authors]
                    abstract = paper.summary.strip() if paper.summary else ""
                    url = paper.entry_id
                    pdf_url = paper.pdf_url
                    year = paper.published.year if paper.published else datetime.now().year
                    categories = [str(cat) for cat in paper.categories] if paper.categories else []
                    
                    # Create quality metrics (ArXiv doesn't have citations, use other factors)
                    metrics = SourceMetrics(
                        publication_year=year,
                        author_count=len(authors),
                        venue_type="preprint"
                    )
                    
                    # Calculate quality score (emphasis on recency for preprints)
                    quality_score = self._calculate_quality_score(metrics, year, categories)
                    
                    source = EnhancedSource(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        url=url,
                        pdf_url=pdf_url,
                        publication_year=year,
                        venue="arXiv",
                        source_type="arxiv",
                        source_id=url.split('/')[-1] if url else None,
                        metrics=metrics,
                        quality_score=quality_score,
                        categories=categories
                    )
                    
                    sources.append(source)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse ArXiv result: {e}")
                    continue
            
            logger.info(f"📚 Found {len(sources)} papers from ArXiv")
            return sources
            
        except Exception as e:
            logger.error(f"Enhanced ArXiv search failed: {e}")
            return []
    
    def _calculate_quality_score(self, metrics: SourceMetrics, year: int, categories: List[str]) -> float:
        """Calculate quality score for ArXiv papers."""
        current_year = datetime.now().year
        
        # Recency is more important for preprints
        recency_score = max(0, 1.0 - (current_year - year) * 0.05)  # 5% penalty per year
        
        # Category relevance (boost for ML/AI categories)
        category_boost = 0.0
        relevant_categories = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "stat.ML"]
        for cat in categories:
            if any(rel_cat in cat for rel_cat in relevant_categories):
                category_boost = 0.2
                break
        
        return min(recency_score + category_boost, 1.0)


class AdvancedSearchEngine:
    """Advanced multi-source search engine for comprehensive research."""
    
    def __init__(self, google_scholar_api_key: Optional[str] = None):
        self.google_scholar = GoogleScholarSearcher(google_scholar_api_key)
        self.semantic_scholar = SemanticScholarSearcher()
        self.enhanced_arxiv = EnhancedArxivSearcher()
        
        logger.info("🚀 Advanced Search Engine initialized with multi-source capability")
    
    async def comprehensive_search(self, 
                                 query: str, 
                                 max_results_per_source: int = 10,
                                 sources: List[str] = None) -> List[EnhancedSource]:
        """Perform comprehensive search across multiple academic sources."""
        if sources is None:
            sources = ["scholar", "semantic", "arxiv"]
        
        logger.info(f"🔍 Starting comprehensive search: {query}")
        logger.info(f"🔍 Using sources: {sources}")
        
        # Run searches in parallel
        search_tasks = []
        
        if "scholar" in sources:
            search_tasks.append(self.google_scholar.search_papers(query, max_results_per_source))
        
        if "semantic" in sources:
            search_tasks.append(self.semantic_scholar.search_papers(query, max_results_per_source))
        
        if "arxiv" in sources:
            search_tasks.append(self.enhanced_arxiv.search_papers(query, max_results_per_source))
        
        # Execute searches with staggered timing to respect rate limits
        all_sources = []
        for i, task in enumerate(search_tasks):
            if i > 0:
                await asyncio.sleep(1.0)  # Stagger requests
            try:
                sources_batch = await task
                all_sources.extend(sources_batch)
            except Exception as e:
                logger.error(f"Search task failed: {e}")
                continue
        
        # Remove duplicates and rank by quality
        unique_sources = self._deduplicate_sources(all_sources)
        ranked_sources = self._rank_sources_by_quality(unique_sources)
        
        logger.info(f"✅ Comprehensive search completed: {len(ranked_sources)} unique high-quality sources")
        
        return ranked_sources
    
    def _deduplicate_sources(self, sources: List[EnhancedSource]) -> List[EnhancedSource]:
        """Remove duplicate sources based on title similarity."""
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            # Normalize title for comparison
            normalized_title = source.title.lower().strip()
            normalized_title = ' '.join(normalized_title.split())  # Normalize whitespace
            
            if normalized_title and normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_sources.append(source)
        
        return unique_sources
    
    def _rank_sources_by_quality(self, sources: List[EnhancedSource]) -> List[EnhancedSource]:
        """Rank sources by quality score and other factors."""
        def quality_key(source: EnhancedSource) -> Tuple[float, int, int]:
            return (
                source.quality_score,
                source.metrics.citation_count if source.metrics else 0,
                source.publication_year
            )
        
        # Sort by quality (descending)
        ranked = sorted(sources, key=quality_key, reverse=True)
        
        logger.info(f"📊 Top sources by quality:")
        for i, source in enumerate(ranked[:5], 1):
            logger.info(f"  {i}. {source.title[:60]}... (score: {source.quality_score:.3f}, citations: {source.metrics.citation_count if source.metrics else 0})")
        
        return ranked


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_advanced_search():
        # Initialize search engine
        scholar_api_key = os.getenv("SERPAPI_API_KEY")  # Optional
        search_engine = AdvancedSearchEngine(scholar_api_key)
        
        # Test query
        query = "activation functions in neural networks"
        
        print(f"🧪 Testing Advanced Search Engine")
        print("=" * 70)
        print(f"Query: {query}")
        print("=" * 70)
        
        # Perform comprehensive search
        results = await search_engine.comprehensive_search(
            query=query,
            max_results_per_source=5,
            sources=["semantic", "arxiv"]  # Start with free sources
        )
        
        print(f"\n📊 Search Results Summary:")
        print(f"Total Sources: {len(results)}")
        
        source_types = {}
        for result in results:
            source_types[result.source_type] = source_types.get(result.source_type, 0) + 1
        
        print(f"Source Distribution: {source_types}")
        
        print(f"\n🏆 Top 3 Results:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result.title}")
            print(f"   Authors: {', '.join(result.authors[:3])}{'...' if len(result.authors) > 3 else ''}")
            print(f"   Year: {result.publication_year}")
            print(f"   Source: {result.source_type}")
            print(f"   Quality Score: {result.quality_score:.3f}")
            if result.metrics:
                print(f"   Citations: {result.metrics.citation_count}")
    
    # Run test
    asyncio.run(test_advanced_search())
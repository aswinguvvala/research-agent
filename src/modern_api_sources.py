"""
Modern API Sources
Enhanced data sources with better APIs than the original research agent,
including Semantic Scholar, OpenAlex, enhanced arXiv, and real-time news.
"""

import asyncio
import json
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import time
from urllib.parse import quote

import arxiv
import feedparser

logger = logging.getLogger(__name__)


class SemanticScholarAPI:
    """Enhanced academic search using Semantic Scholar API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.session = aiohttp.ClientSession()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second
        
        # Headers
        self.headers = {
            "User-Agent": "Multi-Agent Research System/1.0"
        }
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
    
    async def search_papers(self, query: str, limit: int = 10, fields: List[str] = None) -> List[Dict]:
        """Search for academic papers using Semantic Scholar"""
        if fields is None:
            fields = [
                "paperId", "title", "authors", "year", "abstract", 
                "venue", "citationCount", "referenceCount", "url",
                "openAccessPdf", "publicationTypes", "publicationDate"
            ]
        
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": min(limit, 100),  # API limit
                "fields": ",".join(fields)
            }
            
            async with self.session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = data.get("data", [])
                    
                    # Enhance papers with additional metadata
                    enhanced_papers = []
                    for paper in papers:
                        enhanced_paper = await self._enhance_paper_data(paper)
                        enhanced_papers.append(enhanced_paper)
                    
                    return enhanced_papers
                else:
                    logger.error(f"Semantic Scholar search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Semantic Scholar API error: {e}")
            return []
    
    async def get_paper_details(self, paper_id: str, fields: List[str] = None) -> Optional[Dict]:
        """Get detailed information about a specific paper"""
        if fields is None:
            fields = [
                "title", "authors", "year", "abstract", "venue", "citationCount",
                "referenceCount", "citations", "references", "influentialCitationCount",
                "tldr", "fieldsOfStudy", "publicationTypes", "publicationDate"
            ]
        
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/paper/{paper_id}"
            params = {"fields": ",".join(fields)}
            
            async with self.session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    paper = await response.json()
                    return await self._enhance_paper_data(paper)
                else:
                    logger.error(f"Failed to get paper details: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Paper details API error: {e}")
            return None
    
    async def get_author_papers(self, author_id: str, limit: int = 20) -> List[Dict]:
        """Get papers by a specific author"""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/author/{author_id}/papers"
            params = {
                "limit": limit,
                "fields": "paperId,title,year,citationCount,abstract,venue"
            }
            
            async with self.session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    logger.error(f"Author papers search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Author papers API error: {e}")
            return []
    
    async def get_recommendations(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """Get recommended papers based on a paper"""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/paper/{paper_id}/recommendations"
            params = {
                "limit": limit,
                "fields": "paperId,title,authors,year,abstract,venue,citationCount"
            }
            
            async with self.session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("recommendedPapers", [])
                else:
                    logger.error(f"Recommendations failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Recommendations API error: {e}")
            return []
    
    async def _enhance_paper_data(self, paper: Dict) -> Dict:
        """Enhance paper data with computed metrics and classifications"""
        enhanced = paper.copy()
        
        # Calculate impact metrics
        citation_count = paper.get("citationCount", 0)
        year = paper.get("year", 0)
        current_year = datetime.now().year
        
        if year and year <= current_year:
            years_since_publication = current_year - year + 1
            enhanced["citations_per_year"] = citation_count / years_since_publication
        else:
            enhanced["citations_per_year"] = 0
        
        # Classify impact level
        if citation_count >= 100:
            enhanced["impact_level"] = "high"
        elif citation_count >= 20:
            enhanced["impact_level"] = "medium"
        else:
            enhanced["impact_level"] = "low"
        
        # Extract key information for research agent
        enhanced["key_findings"] = []
        if paper.get("abstract"):
            # Simple extraction - in production, would use more sophisticated NLP
            abstract = paper["abstract"]
            enhanced["key_findings"] = [
                sentence.strip() for sentence in abstract.split(". ")[:3] 
                if len(sentence.strip()) > 20
            ]
        
        # Add source metadata for research system
        enhanced["source_type"] = "semantic_scholar"
        enhanced["retrieved_at"] = datetime.now().isoformat()
        
        return enhanced
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.close()


class OpenAlexAPI:
    """Research using OpenAlex API for comprehensive academic data"""
    
    def __init__(self, email: str = "research@example.com"):
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.session = aiohttp.ClientSession()
        
        # Rate limiting - OpenAlex is more generous
        self.last_request_time = 0
        self.min_request_interval = 0.01  # 100 requests per second
    
    async def search_works(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for academic works (papers, books, etc.)"""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/works"
            params = {
                "search": query,
                "per-page": min(limit, 200),  # API limit
                "mailto": self.email,
                "sort": "cited_by_count:desc"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    works = data.get("results", [])
                    return [self._process_work(work) for work in works]
                else:
                    logger.error(f"OpenAlex search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"OpenAlex API error: {e}")
            return []
    
    async def search_by_concept(self, concept: str, limit: int = 20) -> List[Dict]:
        """Search works by research concept/field"""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/works"
            params = {
                "filter": f"concepts.display_name:{concept}",
                "per-page": min(limit, 200),
                "mailto": self.email,
                "sort": "cited_by_count:desc"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    works = data.get("results", [])
                    return [self._process_work(work) for work in works]
                else:
                    logger.error(f"OpenAlex concept search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"OpenAlex concept search error: {e}")
            return []
    
    async def get_trending_topics(self, field: str = None) -> List[Dict]:
        """Get trending research topics"""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/works"
            
            # Get recent papers with high citation velocity
            one_year_ago = (datetime.now() - timedelta(days=365)).year
            params = {
                "filter": f"from_publication_date:{one_year_ago}",
                "per-page": 50,
                "mailto": self.email,
                "sort": "cited_by_count:desc"
            }
            
            if field:
                params["filter"] += f",concepts.display_name:{field}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    works = data.get("results", [])
                    
                    # Extract trending concepts
                    concept_counts = {}
                    for work in works:
                        for concept in work.get("concepts", [])[:5]:  # Top concepts only
                            name = concept.get("display_name", "")
                            if name and concept.get("level", 0) <= 2:  # Not too specific
                                concept_counts[name] = concept_counts.get(name, 0) + 1
                    
                    # Sort by frequency
                    trending = [
                        {"topic": topic, "frequency": count, "category": "trending"}
                        for topic, count in sorted(concept_counts.items(), 
                                                 key=lambda x: x[1], reverse=True)[:10]
                    ]
                    
                    return trending
                else:
                    logger.error(f"OpenAlex trending search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"OpenAlex trending topics error: {e}")
            return []
    
    def _process_work(self, work: Dict) -> Dict:
        """Process and enhance OpenAlex work data"""
        processed = {
            "id": work.get("id", ""),
            "title": work.get("title", ""),
            "abstract": work.get("abstract", ""),
            "publication_year": work.get("publication_year"),
            "publication_date": work.get("publication_date"),
            "type": work.get("type", ""),
            "citation_count": work.get("cited_by_count", 0),
            "authors": [],
            "venue": "",
            "concepts": [],
            "open_access": work.get("open_access", {}).get("is_oa", False),
            "url": work.get("doi", ""),
            "source_type": "openalex"
        }
        
        # Process authors
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            if author.get("display_name"):
                processed["authors"].append(author["display_name"])
        
        # Process venue
        host_venue = work.get("host_venue", {})
        if host_venue.get("display_name"):
            processed["venue"] = host_venue["display_name"]
        
        # Process concepts
        for concept in work.get("concepts", [])[:5]:
            processed["concepts"].append({
                "name": concept.get("display_name", ""),
                "level": concept.get("level", 0),
                "score": concept.get("score", 0.0)
            })
        
        return processed
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.close()


class EnhancedArxivSearch:
    """Enhanced arXiv search with better metadata and classification"""
    
    def __init__(self):
        self.category_descriptions = {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning", 
            "cs.CL": "Computational Linguistics",
            "cs.CV": "Computer Vision",
            "stat.ML": "Machine Learning (Statistics)",
            "math.CO": "Combinatorics",
            "physics.data-an": "Data Analysis",
            "q-bio.NC": "Neuroscience",
            "econ.GN": "General Economics"
        }
    
    async def search(self, query: str, category: str = None, max_results: int = 10) -> List[Dict]:
        """Enhanced arXiv search with better processing"""
        try:
            # Build search query
            search_query = query
            if category:
                search_query = f"cat:{category} AND {query}"
            
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for paper in search.results():
                processed_paper = self._process_paper(paper)
                papers.append(processed_paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Enhanced arXiv search error: {e}")
            return []
    
    async def get_recent_papers(self, category: str, days: int = 7, max_results: int = 20) -> List[Dict]:
        """Get recent papers from a specific category"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Search recent papers
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for paper in search.results():
                if paper.published >= start_date.replace(tzinfo=paper.published.tzinfo):
                    processed_paper = self._process_paper(paper)
                    papers.append(processed_paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Recent papers search error: {e}")
            return []
    
    def _process_paper(self, paper) -> Dict:
        """Process arXiv paper with enhanced metadata"""
        processed = {
            "id": paper.entry_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "published": paper.published.isoformat(),
            "updated": paper.updated.isoformat() if paper.updated else None,
            "categories": paper.categories,
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "source_type": "arxiv_enhanced"
        }
        
        # Add category descriptions
        processed["category_descriptions"] = [
            self.category_descriptions.get(cat, cat) for cat in paper.categories
        ]
        
        # Classify paper type based on title and abstract
        processed["paper_type"] = self._classify_paper_type(paper.title, paper.summary)
        
        # Extract key terms (simple implementation)
        processed["key_terms"] = self._extract_key_terms(paper.title, paper.summary)
        
        # Calculate recency score
        days_old = (datetime.now() - paper.published.replace(tzinfo=None)).days
        processed["recency_score"] = max(0, 1 - (days_old / 365))  # Decays over a year
        
        return processed
    
    def _classify_paper_type(self, title: str, abstract: str) -> str:
        """Classify the type of research paper"""
        text = (title + " " + abstract).lower()
        
        if any(word in text for word in ["survey", "review", "systematic", "comprehensive"]):
            return "survey"
        elif any(word in text for word in ["novel", "new", "propose", "introduce"]):
            return "novel_method"
        elif any(word in text for word in ["empirical", "experimental", "evaluation", "benchmark"]):
            return "empirical"
        elif any(word in text for word in ["theoretical", "analysis", "proof", "mathematical"]):
            return "theoretical"
        elif any(word in text for word in ["application", "case study", "real-world"]):
            return "application"
        else:
            return "general"
    
    def _extract_key_terms(self, title: str, abstract: str) -> List[str]:
        """Extract key terms from title and abstract"""
        text = title + " " + abstract
        
        # Simple term extraction (in production, use proper NLP)
        import re
        
        # Extract capitalized phrases (likely to be technical terms)
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text)
        
        # Extract terms in parentheses (often abbreviations or technical terms)
        parenthetical = re.findall(r'\(([^)]+)\)', text)
        
        # Combine and filter
        key_terms = list(set(capitalized_phrases + parenthetical))
        
        # Filter out common words and keep only meaningful terms
        common_words = {"The", "This", "We", "Our", "In", "For", "With", "By", "From", "To"}
        key_terms = [term for term in key_terms if term not in common_words and len(term) > 2]
        
        return key_terms[:10]  # Limit to 10 key terms


class RealTimeNewsAPI:
    """Real-time news and trend monitoring for research context"""
    
    def __init__(self, news_api_key: Optional[str] = None):
        self.news_api_key = news_api_key
        self.session = aiohttp.ClientSession()
        
        # News sources for different domains
        self.tech_sources = [
            "https://feeds.feedburner.com/oreilly/radar",
            "https://aws.amazon.com/blogs/aws/feed/", 
            "https://blog.google/technology/ai/rss/",
            "https://openai.com/blog/rss.xml"
        ]
        
        self.academic_sources = [
            "https://www.nature.com/nature.rss",
            "https://www.sciencemag.org/rss/current.xml",
            "https://feeds.feedburner.com/arxiv/cs"
        ]
        
        self.business_sources = [
            "https://feeds.a16z.com/a16z.rss",
            "https://hbr.org/feed",
            "https://techcrunch.com/feed/"
        ]
    
    async def get_relevant_news(self, topic: str, domain: str = "general", limit: int = 10) -> List[Dict]:
        """Get recent news relevant to a research topic"""
        try:
            # Choose appropriate sources based on domain
            sources = self._get_sources_for_domain(domain)
            
            all_articles = []
            
            # Fetch from RSS feeds
            for source_url in sources[:3]:  # Limit sources to avoid overwhelming
                articles = await self._fetch_rss_feed(source_url, topic)
                all_articles.extend(articles)
            
            # If we have News API key, also search there
            if self.news_api_key:
                news_articles = await self._search_news_api(topic, limit)
                all_articles.extend(news_articles)
            
            # Filter and rank by relevance
            relevant_articles = self._filter_relevant_articles(all_articles, topic)
            
            return relevant_articles[:limit]
            
        except Exception as e:
            logger.error(f"News API error: {e}")
            return []
    
    def _get_sources_for_domain(self, domain: str) -> List[str]:
        """Get appropriate news sources for a domain"""
        source_map = {
            "academic": self.academic_sources,
            "technical": self.tech_sources,
            "business": self.business_sources,
            "general": self.tech_sources + self.business_sources[:2]
        }
        return source_map.get(domain, self.tech_sources)
    
    async def _fetch_rss_feed(self, url: str, topic: str) -> List[Dict]:
        """Fetch and parse RSS feed"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    for entry in feed.entries[:20]:  # Limit entries per feed
                        article = {
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", entry.get("description", "")),
                            "url": entry.get("link", ""),
                            "published": entry.get("published", ""),
                            "source": feed.feed.get("title", url),
                            "source_type": "rss_feed"
                        }
                        
                        # Check relevance
                        if self._is_relevant(article, topic):
                            articles.append(article)
                    
                    return articles
                else:
                    logger.warning(f"RSS feed fetch failed: {url} - {response.status}")
                    return []
                    
        except Exception as e:
            logger.warning(f"RSS feed error for {url}: {e}")
            return []
    
    async def _search_news_api(self, topic: str, limit: int) -> List[Dict]:
        """Search using News API if available"""
        if not self.news_api_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": topic,
                "sortBy": "publishedAt",
                "pageSize": limit,
                "apiKey": self.news_api_key,
                "language": "en"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = []
                    
                    for article in data.get("articles", []):
                        articles.append({
                            "title": article.get("title", ""),
                            "summary": article.get("description", ""),
                            "url": article.get("url", ""),
                            "published": article.get("publishedAt", ""),
                            "source": article.get("source", {}).get("name", ""),
                            "source_type": "news_api"
                        })
                    
                    return articles
                else:
                    logger.error(f"News API search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"News API error: {e}")
            return []
    
    def _is_relevant(self, article: Dict, topic: str) -> bool:
        """Check if article is relevant to the topic"""
        topic_terms = topic.lower().split()
        article_text = (article.get("title", "") + " " + article.get("summary", "")).lower()
        
        # Simple relevance check
        matches = sum(1 for term in topic_terms if term in article_text)
        return matches >= max(1, len(topic_terms) // 2)
    
    def _filter_relevant_articles(self, articles: List[Dict], topic: str) -> List[Dict]:
        """Filter and rank articles by relevance"""
        topic_terms = set(topic.lower().split())
        
        scored_articles = []
        for article in articles:
            # Calculate relevance score
            article_text = (article.get("title", "") + " " + article.get("summary", "")).lower()
            article_terms = set(article_text.split())
            
            # Jaccard similarity
            intersection = len(topic_terms & article_terms)
            union = len(topic_terms | article_terms)
            relevance_score = intersection / union if union > 0 else 0
            
            # Boost score for title matches
            if any(term in article.get("title", "").lower() for term in topic_terms):
                relevance_score *= 1.5
            
            scored_articles.append((relevance_score, article))
        
        # Sort by relevance and return articles
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        return [article for score, article in scored_articles if score > 0.1]
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.close()


class ModernAPIManager:
    """Manager for all modern API sources"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
        # Initialize API clients
        self.semantic_scholar = SemanticScholarAPI(
            api_key=self.api_keys.get("semantic_scholar")
        )
        self.openalex = OpenAlexAPI(
            email=self.api_keys.get("email", "research@example.com")
        )
        self.enhanced_arxiv = EnhancedArxivSearch()
        self.news_api = RealTimeNewsAPI(
            news_api_key=self.api_keys.get("news_api")
        )
    
    async def comprehensive_search(self, query: str, domain: str = "general") -> Dict[str, List[Dict]]:
        """Perform comprehensive search across all modern APIs"""
        results = {
            "semantic_scholar": [],
            "openalex": [], 
            "arxiv": [],
            "news": []
        }
        
        try:
            # Run searches concurrently
            search_tasks = [
                self.semantic_scholar.search_papers(query, limit=5),
                self.openalex.search_works(query, limit=5),
                self.enhanced_arxiv.search(query, max_results=5),
                self.news_api.get_relevant_news(query, domain, limit=5)
            ]
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Assign results
            if not isinstance(search_results[0], Exception):
                results["semantic_scholar"] = search_results[0]
            if not isinstance(search_results[1], Exception):
                results["openalex"] = search_results[1]
            if not isinstance(search_results[2], Exception):
                results["arxiv"] = search_results[2] 
            if not isinstance(search_results[3], Exception):
                results["news"] = search_results[3]
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive search error: {e}")
            return results
    
    async def close_all(self):
        """Close all API client sessions"""
        await asyncio.gather(
            self.semantic_scholar.close(),
            self.openalex.close(), 
            self.news_api.close(),
            return_exceptions=True
        )


# Export the modern API classes
__all__ = [
    'SemanticScholarAPI',
    'OpenAlexAPI', 
    'EnhancedArxivSearch',
    'RealTimeNewsAPI',
    'ModernAPIManager'
]
"""
Autonomous Research Agent
Core research orchestrator that searches real sources and synthesizes findings.
"""

import asyncio
import aiohttp
import feedparser
import arxiv
import requests
from typing import Dict, List, Optional, Any, Tuple
import re
import json
from datetime import datetime, timedelta
import openai
import time

from content_extractor import ContentExtractor
from citation_manager import CitationManager, Source


class ResearchAgent:
    """Autonomous research agent that searches and synthesizes real research."""
    
    def __init__(self, openai_api_key: str, max_sources: int = 10):
        self.openai_api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.max_sources = max_sources
        
        # Initialize components
        self.content_extractor = ContentExtractor()
        self.citation_manager = CitationManager()
        
        # API endpoints
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.crossref_base = "https://api.crossref.org/works"
        
        # Research session data
        self.current_query = ""
        self.research_results = []
        self.synthesis = ""
        
    async def conduct_research(self, query: str, citation_style: str = "apa") -> Dict[str, Any]:
        """Conduct comprehensive research on a query."""
        print(f"🔍 Starting research on: {query}")
        self.current_query = query
        self.citation_manager.clear_sources()
        self.research_results = []
        
        start_time = time.time()
        
        # Step 1: Search multiple sources
        print("📚 Searching research sources...")
        sources = await self._search_all_sources(query)
        
        # Step 2: Extract and process content
        print("📖 Extracting content from sources...")
        processed_sources = await self._process_sources(sources)
        
        # Step 3: Synthesize findings
        print("🧠 Synthesizing research findings...")
        synthesis = await self._synthesize_findings(processed_sources, query)
        
        # Step 4: Generate final report
        print("📄 Generating research report...")
        report = self._generate_report(synthesis, citation_style)
        
        end_time = time.time()
        research_time = round(end_time - start_time, 2)
        
        print(f"✅ Research completed in {research_time} seconds")
        
        return {
            "query": query,
            "synthesis": synthesis,
            "report": report,
            "sources": self.citation_manager.export_sources(),
            "num_sources": len(self.citation_manager.sources),
            "research_time": research_time,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _search_all_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search all available sources for the query."""
        tasks = [
            self._search_arxiv(query),
            self._search_pubmed(query),
            self._search_web_sources(query)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all sources
        all_sources = []
        for result in results:
            if isinstance(result, list):
                all_sources.extend(result)
            elif isinstance(result, Exception):
                print(f"⚠️  Search error: {result}")
        
        # Remove duplicates and limit results
        unique_sources = self._deduplicate_sources(all_sources)
        return unique_sources[:self.max_sources]
    
    async def _search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """Search ArXiv for relevant papers."""
        try:
            # Add rate limiting to respect ArXiv API guidelines
            await asyncio.sleep(1)  # 1 second delay between requests
            
            # Use arxiv library for search
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            sources = []
            for paper in search.results():
                source_data = {
                    "title": paper.title,
                    "authors": [str(author) for author in paper.authors],
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else None,
                    "source_type": "arxiv",
                    "arxiv_id": paper.entry_id.split('/')[-1],
                    "categories": [str(cat) for cat in paper.categories]
                }
                sources.append(source_data)
            
            return sources
            
        except Exception as e:
            print(f"ArXiv search error: {e}")
            # Check if it's a redirect or connection issue
            if "301" in str(e) or "redirect" in str(e).lower():
                print("Note: ArXiv API redirect detected - trying alternative approach...")
            return []
    
    async def _search_pubmed(self, query: str) -> List[Dict[str, Any]]:
        """Search PubMed for medical research."""
        try:
            # Add rate limiting for PubMed API
            await asyncio.sleep(0.5)  # 0.5 second delay for PubMed
            
            # Search PubMed using E-utilities
            search_url = f"{self.pubmed_base}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": 8,  # Increased from 5 to get more sources
                "retmode": "json",
                "sort": "relevance"  # Sort by relevance
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    search_data = await response.json()
            
            if "esearchresult" not in search_data or not search_data["esearchresult"]["idlist"]:
                return []
            
            # Get detailed info for each paper
            pmids = search_data["esearchresult"]["idlist"]
            fetch_url = f"{self.pubmed_base}efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url, params=fetch_params) as response:
                    xml_data = await response.text()
            
            # Parse PubMed XML (simplified)
            sources = self._parse_pubmed_xml(xml_data)
            return sources
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            # Provide better error context for debugging
            if "timeout" in str(e).lower():
                print("Note: PubMed API timeout - server may be busy")
            return []
    
    async def _search_web_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search web sources and research blogs."""
        sources = []
        
        try:
            # Search some known research blogs and sources
            search_urls = [
                f"https://scholar.google.com/scholar?q={requests.utils.quote(query)}",
                f"https://www.semanticscholar.org/search?q={requests.utils.quote(query)}",
            ]
            
            # Add specific research blog searches
            research_blogs = [
                "https://blog.openai.com",
                "https://ai.googleblog.com",
                "https://research.facebook.com/blog",
                "https://blogs.microsoft.com/ai/",
                "https://deepmind.com/blog"
            ]
            
            # Simple approach: look for recent posts that might be relevant
            for blog_url in research_blogs[:2]:  # Limit to avoid rate limiting
                try:
                    blog_sources = await self._search_blog(blog_url, query)
                    sources.extend(blog_sources)
                except Exception as e:
                    print(f"Blog search error for {blog_url}: {e}")
                    continue
            
            return sources[:3]  # Limit web sources
            
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    async def _search_blog(self, blog_url: str, query: str) -> List[Dict[str, Any]]:
        """Search a specific blog for relevant content."""
        try:
            # Try to find RSS feed
            feed_urls = [
                f"{blog_url}/feed",
                f"{blog_url}/rss",
                f"{blog_url}/atom.xml",
                f"{blog_url}/feed.xml"
            ]
            
            for feed_url in feed_urls:
                try:
                    feed = feedparser.parse(feed_url)
                    if feed.entries:
                        # Look for relevant entries
                        relevant_entries = []
                        query_terms = query.lower().split()
                        
                        for entry in feed.entries[:10]:  # Check recent entries
                            title = entry.get('title', '').lower()
                            summary = entry.get('summary', '').lower()
                            
                            # Simple relevance check
                            relevance_score = sum(1 for term in query_terms 
                                                if term in title or term in summary)
                            
                            if relevance_score > 0:
                                source_data = {
                                    "title": entry.get('title', ''),
                                    "url": entry.get('link', ''),
                                    "summary": entry.get('summary', ''),
                                    "published": entry.get('published', ''),
                                    "source_type": "blog",
                                    "blog_url": blog_url,
                                    "relevance_score": relevance_score
                                }
                                relevant_entries.append(source_data)
                        
                        # Sort by relevance and return top entries
                        relevant_entries.sort(key=lambda x: x['relevance_score'], reverse=True)
                        return relevant_entries[:2]  # Top 2 relevant entries
                        
                except Exception:
                    continue
            
            return []
            
        except Exception as e:
            print(f"Blog search error: {e}")
            return []
    
    async def _process_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract content from sources and add to citation manager."""
        processed = []
        
        for source_data in sources:
            try:
                # Extract content based on source type
                if source_data["source_type"] == "arxiv":
                    content = self.content_extractor.extract_arxiv_content(source_data["arxiv_id"])
                elif source_data.get("pdf_url"):
                    content = self.content_extractor.extract_pdf_content(source_data["pdf_url"])
                elif source_data.get("url"):
                    content = self.content_extractor.extract_web_content(source_data["url"])
                else:
                    # Use available metadata
                    content = {
                        "content": source_data.get("abstract", source_data.get("summary", "")),
                        "metadata": source_data,
                        "source_type": source_data["source_type"]
                    }
                
                if "error" not in content:
                    # Create citation source
                    citation_source = Source(
                        title=content["metadata"].get("title", source_data.get("title", "Unknown Title")),
                        authors=content["metadata"].get("authors", source_data.get("authors", [])),
                        year=self._extract_year(content["metadata"].get("published", source_data.get("published", ""))),
                        url=source_data.get("url", source_data.get("pdf_url", "")),
                        doi=content["metadata"].get("doi"),
                        journal=content["metadata"].get("journal"),
                        source_type=source_data["source_type"]
                    )
                    
                    citation_num = self.citation_manager.add_source(citation_source)
                    
                    # Add citation number to content
                    content["citation_number"] = citation_num
                    content["source_data"] = source_data
                    
                    processed.append(content)
                else:
                    print(f"⚠️  Failed to extract content from: {source_data.get('title', 'Unknown')}")
                    
            except Exception as e:
                print(f"⚠️  Error processing source: {e}")
                continue
        
        print(f"📊 Successfully processed {len(processed)} sources")
        return processed
    
    async def _synthesize_findings(self, sources: List[Dict[str, Any]], query: str) -> str:
        """Use AI to synthesize findings from multiple sources."""
        if not sources:
            return "No sources were successfully processed for synthesis."
        
        # Prepare content for synthesis
        source_summaries = []
        for i, source in enumerate(sources, 1):
            content = source.get("content", "")
            title = source.get("metadata", {}).get("title", "Unknown")
            citation_num = source.get("citation_number", i)
            
            # Limit content length for AI processing
            if len(content) > 1500:
                content = content[:1500] + "..."
            
            summary = f"Source {citation_num}: {title}\nContent: {content}\n"
            source_summaries.append(summary)
        
        # Create synthesis prompt
        synthesis_prompt = f"""You are a research analyst tasked with synthesizing findings from multiple sources to answer the research question: "{query}"

Please analyze the following sources and provide a comprehensive synthesis that:
1. Directly addresses the research question
2. Identifies key findings and themes
3. Notes any conflicting information or different perspectives
4. Highlights gaps in the current research
5. Uses in-text citations [Source X] to reference findings

Sources:
{chr(10).join(source_summaries)}

Provide a well-structured synthesis (300-500 words) that demonstrates critical analysis and synthesis of the sources. Focus on substance and insights rather than just summarizing each source."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst who specializes in synthesizing academic and technical literature."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            synthesis = response.choices[0].message.content.strip()
            return synthesis
            
        except Exception as e:
            print(f"⚠️  AI synthesis error: {e}")
            # Fallback to simple concatenation
            return self._create_fallback_synthesis(sources, query)
    
    def _create_fallback_synthesis(self, sources: List[Dict[str, Any]], query: str) -> str:
        """Create a simple synthesis without AI."""
        synthesis_parts = [
            f"Research Summary: {query}",
            f"Based on {len(sources)} sources:\n"
        ]
        
        for source in sources:
            title = source.get("metadata", {}).get("title", "Unknown")
            citation_num = source.get("citation_number", 1)
            content = source.get("content", "")
            
            # Extract key points (simple approach)
            sentences = content.split('. ')
            key_sentences = sentences[:2]  # First two sentences
            
            synthesis_parts.append(f"• {title} [Source {citation_num}]: {'. '.join(key_sentences)}")
        
        return "\n".join(synthesis_parts)
    
    def _generate_report(self, synthesis: str, citation_style: str = "apa") -> str:
        """Generate the final research report."""
        report_parts = [
            f"Research Report: {self.current_query}",
            "=" * 60,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of sources: {len(self.citation_manager.sources)}",
            "",
            "SYNTHESIS",
            "-" * 20,
            synthesis,
            "",
            "BIBLIOGRAPHY",
            "-" * 20,
            self.citation_manager.generate_bibliography(citation_style)
        ]
        
        return "\n".join(report_parts)
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources based on title similarity."""
        if not sources:
            return []
        
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            title = source.get("title", "").lower().strip()
            if not title:
                continue
                
            # Simple deduplication based on title similarity
            is_duplicate = False
            for seen_title in seen_titles:
                if self._title_similarity(title, seen_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sources.append(source)
                seen_titles.add(title)
        
        return unique_sources
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate simple title similarity."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _extract_year(self, date_string: str) -> str:
        """Extract year from various date formats."""
        if not date_string:
            return str(datetime.now().year)
        
        # Look for 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_string))
        if year_match:
            return year_match.group(0)
        
        return str(datetime.now().year)
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response (simplified)."""
        # This is a simplified parser - in production, you'd use xml.etree.ElementTree
        sources = []
        
        # Extract basic information using regex (not ideal but simple)
        title_pattern = r'<ArticleTitle>(.*?)</ArticleTitle>'
        abstract_pattern = r'<AbstractText.*?>(.*?)</AbstractText>'
        author_pattern = r'<LastName>(.*?)</LastName>\s*<ForeName>(.*?)</ForeName>'
        year_pattern = r'<Year>(\d{4})</Year>'
        
        titles = re.findall(title_pattern, xml_data, re.DOTALL)
        abstracts = re.findall(abstract_pattern, xml_data, re.DOTALL)
        authors = re.findall(author_pattern, xml_data)
        years = re.findall(year_pattern, xml_data)
        
        for i, title in enumerate(titles):
            source_data = {
                "title": title.strip(),
                "abstract": abstracts[i].strip() if i < len(abstracts) else "",
                "authors": [f"{first} {last}" for last, first in authors] if authors else [],
                "year": years[i] if i < len(years) else str(datetime.now().year),
                "source_type": "pubmed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/"  # Would need actual PMID
            }
            sources.append(source_data)
        
        return sources[:5]  # Limit results


# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    async def test_research():
        agent = ResearchAgent(api_key)
        
        # Test research
        query = "What are the latest advances in large language model reasoning?"
        result = await agent.conduct_research(query)
        
        print(result["report"])
        print(f"\nResearch completed with {result['num_sources']} sources in {result['research_time']} seconds")
    
    # Run test
    asyncio.run(test_research())
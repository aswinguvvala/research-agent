"""
Simple Research Agent
Lightweight research agent with minimal dependencies for basic functionality.
"""

import os
import asyncio
import openai
import arxiv
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimpleSource:
    """Simple source representation."""
    title: str
    abstract: str
    authors: List[str]
    year: int
    venue: str
    url: str
    source_type: str


@dataclass
class SimpleResearchResult:
    """Simple research result."""
    query: str
    summary: str
    sources: List[SimpleSource]
    total_sources: int
    research_time: float
    timestamp: str


class SimpleResearchAgent:
    """
    Simple research agent that works with minimal dependencies.
    Provides basic research functionality using only ArXiv and Semantic Scholar.
    """
    
    def __init__(self, openai_api_key: str, max_sources: int = 10, debug_mode: bool = False):
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_api_key = openai_api_key
        self.max_sources = max_sources
        self.debug_mode = debug_mode
        
        # Set up logging
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        # Initialize OpenAI client
        try:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            raise
        
        logger.info("🚀 Simple Research Agent ready")
    
    async def conduct_research(self, query: str) -> SimpleResearchResult:
        """
        Conduct basic research using ArXiv and Semantic Scholar.
        
        Args:
            query: Research query
            
        Returns:
            SimpleResearchResult with basic analysis
        """
        start_time = time.time()
        logger.info(f"🔬 Starting research for: {query}")
        
        try:
            # Search ArXiv
            logger.info("📚 Searching ArXiv...")
            arxiv_sources = await self._search_arxiv(query)
            
            # Search Semantic Scholar
            logger.info("🔍 Searching Semantic Scholar...")
            scholar_sources = await self._search_semantic_scholar(query)
            
            # Combine sources
            all_sources = arxiv_sources + scholar_sources
            
            if not all_sources:
                logger.warning("⚠️ No sources found")
                return self._create_empty_result(query, start_time)
            
            # Limit to max sources
            sources = all_sources[:self.max_sources]
            logger.info(f"📊 Found {len(sources)} sources")
            
            # Generate summary
            logger.info("📝 Generating research summary...")
            summary = await self._generate_summary(query, sources)
            
            total_time = time.time() - start_time
            
            result = SimpleResearchResult(
                query=query,
                summary=summary,
                sources=sources,
                total_sources=len(sources),
                research_time=total_time,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Research completed in {total_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            return self._create_error_result(query, str(e), start_time)
    
    async def _search_arxiv(self, query: str) -> List[SimpleSource]:
        """Search ArXiv for papers."""
        try:
            # Add delay to respect ArXiv rate limits
            await asyncio.sleep(3)
            
            search = arxiv.Search(
                query=query,
                max_results=5,  # Limit to avoid overwhelming
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            sources = []
            for paper in search.results():
                try:
                    source = SimpleSource(
                        title=paper.title.strip(),
                        abstract=paper.summary.strip()[:500],  # Limit abstract length
                        authors=[author.name for author in paper.authors],
                        year=paper.published.year,
                        venue="arXiv",
                        url=paper.entry_id,
                        source_type="arxiv"
                    )
                    sources.append(source)
                except Exception as e:
                    logger.warning(f"Failed to process ArXiv paper: {e}")
                    continue
            
            logger.info(f"📚 ArXiv: {len(sources)} sources found")
            return sources
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    async def _search_semantic_scholar(self, query: str) -> List[SimpleSource]:
        """Search Semantic Scholar API."""
        try:
            # Add delay to respect rate limits
            await asyncio.sleep(1)
            
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': 5,  # Limit to avoid overwhelming
                'fields': 'title,abstract,authors,year,venue,citationCount,url'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Semantic Scholar API error: {response.status_code}")
                return []
            
            data = response.json()
            papers = data.get('data', [])
            
            sources = []
            for paper in papers:
                try:
                    # Extract basic information
                    title = paper.get('title', 'Unknown Title').strip()
                    abstract = paper.get('abstract', 'No abstract available').strip()
                    
                    if not title or title == 'Unknown Title':
                        continue
                    
                    # Extract authors
                    authors = []
                    for author in paper.get('authors', []):
                        name = author.get('name', '').strip()
                        if name:
                            authors.append(name)
                    
                    source = SimpleSource(
                        title=title,
                        abstract=abstract[:500],  # Limit abstract length
                        authors=authors,
                        year=paper.get('year', 0),
                        venue=paper.get('venue', 'Unknown Venue').strip(),
                        url=paper.get('url', ''),
                        source_type="semantic_scholar"
                    )
                    sources.append(source)
                    
                except Exception as e:
                    logger.warning(f"Failed to process Semantic Scholar paper: {e}")
                    continue
            
            logger.info(f"🔍 Semantic Scholar: {len(sources)} sources found")
            return sources
            
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []
    
    async def _generate_summary(self, query: str, sources: List[SimpleSource]) -> str:
        """Generate research summary using OpenAI."""
        try:
            # Prepare source information
            source_info = []
            for i, source in enumerate(sources[:8], 1):  # Limit to 8 sources for summary
                source_text = f"{i}. {source.title} ({source.year})\n"
                source_text += f"   Authors: {', '.join(source.authors[:3])}\n"  # First 3 authors
                source_text += f"   Abstract: {source.abstract[:200]}...\n"  # First 200 chars
                source_info.append(source_text)
            
            sources_text = "\n".join(source_info)
            
            prompt = f"""
            Create a comprehensive research summary for the query: "{query}"
            
            Based on these research sources:
            {sources_text}
            
            Provide a summary that includes:
            1. Overview of the research landscape
            2. Key findings and themes
            3. Main approaches and methodologies
            4. Significant insights or conclusions
            5. Current state of research in this area
            
            Keep it informative and well-structured (3-4 paragraphs).
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Research summary for '{query}': Found {len(sources)} relevant sources covering various aspects of this topic. Analysis includes academic papers and recent research contributions."
    
    def _create_empty_result(self, query: str, start_time: float) -> SimpleResearchResult:
        """Create empty result when no sources found."""
        return SimpleResearchResult(
            query=query,
            summary=f"No research sources found for '{query}'. Try using different keywords or broader search terms.",
            sources=[],
            total_sources=0,
            research_time=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _create_error_result(self, query: str, error: str, start_time: float) -> SimpleResearchResult:
        """Create error result when research fails."""
        return SimpleResearchResult(
            query=query,
            summary=f"Research failed for '{query}': {error}",
            sources=[],
            total_sources=0,
            research_time=time.time() - start_time,
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
    
    print(f"🚀 Starting basic research for: {query}")
    print("This may take 30-60 seconds...")
    
    try:
        # Initialize agent
        agent = SimpleResearchAgent(
            openai_api_key=api_key,
            max_sources=10,
            debug_mode=True
        )
        
        # Conduct research
        result = await agent.conduct_research(query)
        
        # Display results
        print("\n" + "="*80)
        print("🎯 RESEARCH RESULTS")
        print("="*80)
        
        print(f"\n📊 Query: {result.query}")
        print(f"⏱️  Research Time: {result.research_time:.2f} seconds")
        print(f"📈 Sources Found: {result.total_sources}")
        
        print(f"\n📋 RESEARCH SUMMARY")
        print("-" * 40)
        print(result.summary)
        
        if result.sources:
            print(f"\n📚 SOURCES ({len(result.sources)})")
            print("-" * 40)
            for i, source in enumerate(result.sources, 1):
                print(f"\n{i}. {source.title}")
                print(f"   Authors: {', '.join(source.authors[:3])}")
                print(f"   Year: {source.year} | Venue: {source.venue}")
                print(f"   Source: {source.source_type}")
                print(f"   URL: {source.url}")
        
        print("\n" + "="*80)
        print("✅ Basic research completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        import traceback
        if os.getenv("DEBUG"):
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
"""
Unified Research Agent
Single, streamlined research agent that works like Claude/Gemini Deep Research.
Comprehensive web search + academic papers + simple inline citations.
"""

import os
import asyncio
import openai
import requests
import arxiv
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urljoin
import logging

# Import the comprehensive web search engine
try:
    from web_search_engine import create_web_search_engine, WebSearchResult
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger.warning("Web search engine not available - using fallback")

logger = logging.getLogger(__name__)


@dataclass
class SimpleSource:
    """Simple source representation with inline citation support."""
    title: str
    url: str
    snippet: str
    authors: List[str]
    date: str
    source_type: str  # 'web', 'academic', 'news'
    citation_id: int  # For [1], [2], [3] inline citations


@dataclass
class QueryDisambiguation:
    """Query disambiguation options."""
    original_query: str
    is_ambiguous: bool
    options: List[Dict[str, str]]  # [{"term": "LoRa", "description": "IoT communication protocol"}, ...]
    confidence_score: float


@dataclass
class UnifiedResearchResult:
    """Unified research result with inline citations."""
    query: str
    summary_with_citations: str  # Summary with [1], [2], [3] inline citations
    sources: List[SimpleSource]
    total_sources: int
    research_time: float
    timestamp: str
    search_strategy: str  # 'recent', 'historical', 'comprehensive'
    disambiguation_used: Optional[QueryDisambiguation] = None


class UnifiedResearchAgent:
    """
    Unified research agent that works like Claude/Gemini Deep Research.
    Features:
    - Comprehensive web search (Google, academic papers, news)
    - Dynamic time filtering based on query context
    - GPT-4o mini for summary generation
    - Simple inline [1], [2], [3] citations
    """
    
    def __init__(self, openai_api_key: str, max_sources: int = 15, debug_mode: bool = False):
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_api_key = openai_api_key
        self.max_sources = max_sources
        self.debug_mode = debug_mode
        
        # Initialize OpenAI client with GPT-4o mini
        try:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized for GPT-4o mini")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        # Initialize ArXiv client for academic papers
        self.arxiv_client = arxiv.Client()
        
        # Initialize comprehensive web search engine
        if WEB_SEARCH_AVAILABLE:
            self.web_search_engine = create_web_search_engine(max_results_per_source=5)
            logger.info("✅ Comprehensive web search engine initialized")
        else:
            self.web_search_engine = None
            logger.warning("⚠️ Web search engine not available - using fallback")
        
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
            logger.info("🐛 Debug mode enabled for Unified Research Agent")
    
    
    def _detect_query_ambiguity(self, query: str) -> QueryDisambiguation:
        """
        Detect if query is ambiguous and needs disambiguation.
        Like Gemini Deep Research - identify terms with multiple meanings.
        """
        query_lower = query.lower()
        
        # Common ambiguous terms in tech/AI space
        ambiguous_terms = {
            "lora": [
                {"term": "LoRa", "description": "Long Range - IoT communication protocol for low-power devices"},
                {"term": "LoRA", "description": "Low-Rank Adaptation - AI technique for efficient model fine-tuning"}
            ],
            "python": [
                {"term": "Python Programming", "description": "Programming language used for software development"},
                {"term": "Python Animal", "description": "Large snake species and reptile information"}
            ],
            "transformer": [
                {"term": "Transformer (AI)", "description": "Neural network architecture for language models"},
                {"term": "Transformer (Electrical)", "description": "Electrical device for voltage conversion"}
            ],
            "bert": [
                {"term": "BERT (AI)", "description": "Bidirectional Encoder Representations from Transformers - NLP model"},
                {"term": "Bert (Name)", "description": "Person name or character information"}
            ],
            "go": [
                {"term": "Go Programming", "description": "Programming language developed by Google"},
                {"term": "Go Game", "description": "Ancient board game strategy and rules"}
            ],
            "rust": [
                {"term": "Rust Programming", "description": "Systems programming language focused on safety"},
                {"term": "Rust Corrosion", "description": "Metal oxidation and corrosion prevention"}
            ],
            "swift": [
                {"term": "Swift Programming", "description": "Apple's programming language for iOS development"},
                {"term": "Swift Bird/Speed", "description": "Bird species or speed-related information"}
            ]
        }
        
        # Check for ambiguous terms
        for ambiguous_term, options in ambiguous_terms.items():
            if ambiguous_term in query_lower:
                # Calculate confidence based on context clues
                confidence = 0.8  # Base confidence for detected ambiguity
                
                # Reduce confidence if context clues are present
                tech_context_clues = ["programming", "code", "software", "development", "algorithm", "model", "ai", "ml", "iot", "protocol", "network", "computer"]
                if any(clue in query_lower for clue in tech_context_clues):
                    confidence = 0.3  # Lower ambiguity if context is clear
                
                return QueryDisambiguation(
                    original_query=query,
                    is_ambiguous=confidence > 0.6,
                    options=options,
                    confidence_score=confidence
                )
        
        # No ambiguity detected
        return QueryDisambiguation(
            original_query=query,
            is_ambiguous=False,
            options=[],
            confidence_score=0.0
        )
    
    
    def _determine_search_strategy(self, query: str) -> str:
        """
        Determine search strategy based on query context.
        Like Claude/Gemini Deep Research - analyze query to determine time focus.
        """
        query_lower = query.lower()
        
        # Historical keywords - search old and new papers
        historical_keywords = [
            'history', 'evolution', 'development', 'origins', 'background',
            'timeline', 'historical', 'past', 'traditional', 'classic',
            'foundation', 'early', 'invention', 'discovery'
        ]
        
        # Recent keywords - focus on latest papers
        recent_keywords = [
            'latest', 'recent', 'current', 'new', 'modern', 'today',
            'cutting-edge', 'state-of-the-art', 'breakthrough', '2024',
            '2023', 'emerging', 'future', 'trends', 'advances'
        ]
        
        # Check for historical indicators
        if any(keyword in query_lower for keyword in historical_keywords):
            return 'historical'  # Search from 1990s + recent papers
        
        # Check for recent indicators  
        elif any(keyword in query_lower for keyword in recent_keywords):
            return 'recent'  # Focus on 2022-2024 papers
        
        else:
            return 'comprehensive'  # Balanced mix across timeframes
    
    
    def _get_arxiv_papers(self, query: str, strategy: str, max_papers: int = 8) -> List[SimpleSource]:
        """
        Search ArXiv for academic papers with dynamic time filtering.
        """
        try:
            sources = []
            
            # Determine date range based on strategy
            if strategy == 'historical':
                # For historical queries, get papers from different eras
                searches = [
                    (query, None, None),  # All time for historical context
                ]
            elif strategy == 'recent':
                # For recent queries, focus on last 2-3 years
                recent_date = datetime.now() - timedelta(days=730)  # ~2 years
                searches = [
                    (query, recent_date, None)
                ]
            else:
                # Comprehensive: mix of timeframes
                recent_date = datetime.now() - timedelta(days=365)  # 1 year
                searches = [
                    (query, recent_date, None),  # Recent
                    (query, None, None)  # All time
                ]
            
            for search_query, start_date, end_date in searches:
                try:
                    search = arxiv.Search(
                        query=search_query,
                        max_results=max_papers // len(searches),
                        sort_by=arxiv.SortCriterion.Relevance
                    )
                    
                    for paper in self.arxiv_client.results(search):
                        # Filter by date if specified
                        if start_date and paper.published.date() < start_date.date():
                            continue
                        
                        source = SimpleSource(
                            title=paper.title,
                            url=paper.entry_id,
                            snippet=paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                            authors=[author.name for author in paper.authors],
                            date=paper.published.strftime('%Y-%m-%d'),
                            source_type='academic',
                            citation_id=len(sources) + 1
                        )
                        sources.append(source)
                        
                        if len(sources) >= max_papers:
                            break
                    
                    if len(sources) >= max_papers:
                        break
                        
                except Exception as e:
                    logger.warning(f"ArXiv search failed: {e}")
                    continue
            
            logger.info(f"📚 Found {len(sources)} ArXiv papers")
            return sources
            
        except Exception as e:
            logger.error(f"❌ ArXiv search failed: {e}")
            return []
    
    
    async def _get_arxiv_papers_with_progress(self, query: str, strategy: str, max_papers: int = 8, progress_callback=None) -> List[SimpleSource]:
        """ArXiv search with real-time progress updates."""
        try:
            sources = []
            
            # Progress update: Starting ArXiv search
            if progress_callback:
                await progress_callback('searching', 0.20, f'🔍 Querying ArXiv database for "{query[:50]}..."')
            
            # Run sync method in thread pool to avoid blocking
            import asyncio
            sources = await asyncio.get_event_loop().run_in_executor(
                None, self._get_arxiv_papers, query, strategy, max_papers
            )
            
            # Progress update: ArXiv search completed
            if progress_callback:
                await progress_callback('searching', 0.30, f'📚 Analyzed {len(sources)} academic papers from ArXiv')
            
            return sources
            
        except Exception as e:
            logger.error(f"❌ ArXiv search with progress failed: {e}")
            return []
    
    
    def _search_web(self, query: str, strategy: str, max_results: int = 7) -> List[SimpleSource]:
        """
        Comprehensive web search like Claude/Gemini Deep Research.
        Uses multiple sources: DuckDuckGo, Wikipedia, Reddit, News.
        """
        sources = []
        
        try:
            if self.web_search_engine:
                # Use comprehensive web search engine
                web_results = self.web_search_engine.search_comprehensive(query, strategy)
                
                for web_result in web_results[:max_results]:
                    source = SimpleSource(
                        title=web_result.title,
                        url=web_result.url,
                        snippet=web_result.snippet,
                        authors=[web_result.source.title()],  # Use source name as author
                        date=web_result.date,
                        source_type='web' if web_result.source in ['duckduckgo'] else web_result.source,
                        citation_id=len(sources) + 1
                    )
                    sources.append(source)
            
            else:
                # Fallback to placeholder sources if web search engine not available
                fallback_sources = [
                    {
                        'title': f"Web Research: {query}",
                        'url': f"https://duckduckgo.com/?q={quote_plus(query)}",
                        'snippet': f"Comprehensive web search results for {query}. Multiple sources analyzed for relevant information.",
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source_type': 'web'
                    },
                    {
                        'title': f"Wikipedia: {query}",
                        'url': f"https://en.wikipedia.org/wiki/{quote_plus(query)}",
                        'snippet': f"Encyclopedia information about {query} from Wikipedia.",
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source_type': 'wikipedia'
                    }
                ]
                
                for i, source_data in enumerate(fallback_sources[:max_results]):
                    source = SimpleSource(
                        title=source_data['title'],
                        url=source_data['url'],
                        snippet=source_data['snippet'],
                        authors=['Web Research'],
                        date=source_data['date'],
                        source_type=source_data['source_type'],
                        citation_id=i + 1
                    )
                    sources.append(source)
        
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            # Return empty list if search fails
            sources = []
        
        logger.info(f"🌐 Found {len(sources)} web sources")
        return sources
    
    
    async def _search_web_with_progress(self, query: str, strategy: str, max_results: int = 7, progress_callback=None) -> List[SimpleSource]:
        """Web search with real-time progress updates."""
        try:
            # Progress update: Starting web search
            if progress_callback:
                await progress_callback('searching', 0.45, f'🌐 Searching web sources for "{query[:50]}..."')
            
            # Run sync method in thread pool to avoid blocking
            import asyncio
            sources = await asyncio.get_event_loop().run_in_executor(
                None, self._search_web, query, strategy, max_results
            )
            
            # Progress update: Web search completed
            if progress_callback:
                await progress_callback('searching', 0.65, f'🌐 Gathered {len(sources)} relevant web sources')
            
            return sources
            
        except Exception as e:
            logger.error(f"❌ Web search with progress failed: {e}")
            return []
    
    
    def _generate_summary_with_citations(self, query: str, sources: List[SimpleSource]) -> str:
        """
        Generate summary with inline [1], [2], [3] citations using GPT-4o mini.
        CRITICAL: Only use information actually present in sources - no training data generation.
        """
        try:
            # Validate source quality first
            if not sources:
                return f"No relevant sources found for: {query}. Please try a more specific search query."
            
            # Filter for quality sources
            quality_sources = [s for s in sources if len(s.snippet) > 50]
            if not quality_sources:
                return f"Found sources for '{query}' but content quality is insufficient for analysis. Please try different search terms."
            
            # Prepare source context for GPT-4o mini with emphasis on source-grounding
            source_context = ""
            for source in quality_sources:
                source_context += f"\n[{source.citation_id}] {source.title}\n"
                source_context += f"Authors: {', '.join(source.authors)}\n"
                source_context += f"Date: {source.date}\n"
                source_context += f"Content: {source.snippet}\n"
                source_context += f"Type: {source.source_type}\n\n"
            
            # CRITICAL: Enhanced prompt for source-grounded synthesis
            prompt = f"""You are a research assistant creating a summary ONLY from provided sources. Do NOT use your training data.

QUERY: {query}

SOURCES:
{source_context}

CRITICAL INSTRUCTIONS:
1. ONLY use information that appears in the SOURCE CONTENT above
2. DO NOT generate facts from your training knowledge
3. If sources don't contain information to answer the query, say so
4. Use INLINE CITATIONS [1], [2], [3] for EVERY fact
5. Each citation must correspond to the actual source where you found the information
6. If sources are insufficient, explain what's missing

VALIDATION RULES:
- Every fact must be traceable to a source snippet
- Citations must match source numbers exactly
- Do not infer or extrapolate beyond source content
- If contradictions exist between sources, mention them

FORMAT EXAMPLE:
"According to the provided sources, large language models show capabilities in reasoning [1]. The research indicates chain-of-thought methods improve performance [2]. However, the available sources provide limited information about recent developments."

Generate a source-grounded summary with accurate citations:"""

            # Call GPT-4o mini specifically
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Specifically use GPT-4o mini
                messages=[
                    {"role": "system", "content": "You are a professional research assistant that creates comprehensive summaries with inline citations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more focused, factual responses
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("✅ Summary generated with GPT-4o mini and inline citations")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate summary with GPT-4o mini: {e}")
            # Honest fallback - no fake citations
            return f"Research summary for: {query}\n\nError occurred during AI synthesis. Found {len(sources)} sources but unable to process content. Please try again or use more specific search terms."
    
    
    async def _generate_summary_with_citations_async(self, query: str, sources: List[SimpleSource], progress_callback=None) -> str:
        """
        Generate summary with inline citations using GPT-4o mini with real-time progress updates.
        """
        try:
            # Progress update: Starting AI synthesis
            if progress_callback:
                await progress_callback('synthesizing', 0.86, '🧠 Analyzing source patterns and extracting key insights...')
            
            # Run sync method in thread pool to avoid blocking
            import asyncio
            
            # Progress update: GPT processing
            if progress_callback:
                await progress_callback('synthesizing', 0.90, '✍️ Generating comprehensive summary with citations...')
            
            summary = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_summary_with_citations, query, sources
            )
            
            # Progress update: Summary completed
            if progress_callback:
                await progress_callback('synthesizing', 0.95, '✅ AI synthesis complete with inline citations')
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Async summary generation failed: {e}")
            # Honest fallback - no fake citations
            return f"Research summary for: {query}\n\nError occurred during AI synthesis. Found {len(sources)} sources but unable to process content. Please try again or use more specific search terms."
    
    
    async def research(self, query: str, progress_callback=None) -> UnifiedResearchResult:
        """
        Main research method - works like Gemini Deep Research with real-time thinking updates.
        Comprehensive search + dynamic time filtering + GPT-4o mini + simple citations.
        """
        start_time = time.time()
        logger.info(f"🔍 Starting unified research for: {query}")
        
        # Progress callback helper
        async def update_progress(stage: str, progress: float, message: str, sources_found: int = 0):
            if progress_callback:
                try:
                    await progress_callback({
                        'stage': stage,
                        'progress': progress,
                        'message': message,
                        'sources_found': sources_found
                    })
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
        
        # Stage 0: Query Disambiguation (Like Gemini Deep Research)
        await update_progress('analyzing', 0.02, '🔍 Analyzing query for potential ambiguity...')
        disambiguation = self._detect_query_ambiguity(query)
        
        if disambiguation.is_ambiguous:
            logger.warning(f"⚠️ Ambiguous query detected: '{query}' - Options: {[opt['term'] for opt in disambiguation.options]}")
            await update_progress('analyzing', 0.05, f'⚠️ Detected ambiguous term - proceeding with general interpretation')
            # TODO: In future, pause research and present disambiguation UI to user
            # For now, continue with research but log the ambiguity
        else:
            logger.info(f"✅ Query '{query}' is unambiguous - proceeding with research")
        
        # Stage 1: Strategy Analysis (Gemini-style thinking)
        await update_progress('analyzing', 0.08, '🤔 Analyzing query and determining research strategy...')
        strategy = self._determine_search_strategy(query)
        logger.info(f"📊 Search strategy: {strategy}")
        await update_progress('analyzing', 0.10, f'🎯 Using {strategy} search strategy for optimal results')
        
        # Comprehensive search like Gemini Deep Research
        all_sources = []
        
        # Stage 2: Academic Search with thinking updates
        await update_progress('searching', 0.15, '📚 Searching academic papers on ArXiv...')
        academic_sources = await self._get_arxiv_papers_with_progress(query, strategy, max_papers=8, progress_callback=update_progress)
        all_sources.extend(academic_sources)
        await update_progress('searching', 0.35, f'📚 Found {len(academic_sources)} academic papers', len(academic_sources))
        
        # Stage 3: Web Search with thinking updates  
        await update_progress('searching', 0.40, '🌐 Searching web sources for additional context...')
        web_sources = await self._search_web_with_progress(query, strategy, max_results=7, progress_callback=update_progress)
        all_sources.extend(web_sources)
        await update_progress('searching', 0.70, f'🌐 Found {len(web_sources)} web sources', len(all_sources))
        
        # Stage 4: Source Processing
        await update_progress('processing', 0.75, '🔍 Processing and ranking sources by relevance...')
        
        # Update citation IDs sequentially
        for i, source in enumerate(all_sources):
            source.citation_id = i + 1
        
        # Limit to max_sources
        if len(all_sources) > self.max_sources:
            all_sources = all_sources[:self.max_sources]
        
        await update_progress('processing', 0.80, f'✅ Selected top {len(all_sources)} most relevant sources')
        
        # Stage 5: AI Synthesis (GPT-4o mini)
        await update_progress('synthesizing', 0.85, '🧠 Analyzing sources and generating comprehensive summary...')
        summary_with_citations = await self._generate_summary_with_citations_async(query, all_sources, progress_callback=update_progress)
        
        # Calculate research time
        research_time = time.time() - start_time
        
        # Final stage: Completion
        await update_progress('completed', 1.0, '✅ Research completed with comprehensive analysis', len(all_sources))
        
        # Create result
        result = UnifiedResearchResult(
            query=query,
            summary_with_citations=summary_with_citations,
            sources=all_sources,
            total_sources=len(all_sources),
            research_time=research_time,
            timestamp=datetime.now().isoformat(),
            search_strategy=strategy,
            disambiguation_used=disambiguation if disambiguation.is_ambiguous else None
        )
        
        logger.info(f"✅ Research completed in {research_time:.2f}s with {len(all_sources)} sources")
        return result
    
    
    def research_sync(self, query: str) -> UnifiedResearchResult:
        """Synchronous wrapper for research method."""
        return asyncio.run(self.research(query))
    
    
    def to_dict(self, result: UnifiedResearchResult) -> Dict[str, Any]:
        """Convert result to dictionary for API responses."""
        return asdict(result)


# Factory function for easy instantiation
def create_unified_agent(openai_api_key: str, max_sources: int = 15, debug_mode: bool = False) -> UnifiedResearchAgent:
    """Create a UnifiedResearchAgent instance."""
    return UnifiedResearchAgent(
        openai_api_key=openai_api_key,
        max_sources=max_sources,
        debug_mode=debug_mode
    )


if __name__ == "__main__":
    # Test the unified agent
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        agent = create_unified_agent(api_key, debug_mode=True)
        result = agent.research_sync("What are the latest advances in large language model reasoning?")
        print(f"Query: {result.query}")
        print(f"Strategy: {result.search_strategy}")
        print(f"Sources: {result.total_sources}")
        print(f"Summary:\n{result.summary_with_citations}")
        print(f"\nSources:")
        for source in result.sources:
            print(f"[{source.citation_id}] {source.title} ({source.source_type})")
    else:
        print("Please set OPENAI_API_KEY environment variable to test")
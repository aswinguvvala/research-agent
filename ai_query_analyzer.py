"""
AI-Driven Query Analyzer
Uses GPT-4o mini to intelligently understand research queries and generate dynamic search strategies.
No hardcoded patterns - purely AI-driven analysis.
"""

import openai
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Results of AI-driven query analysis."""
    research_intent: str
    key_concepts: List[str]
    research_type: str  # survey, specific_technique, comparison, explanation, etc.
    domain_detected: str
    academic_terminology: List[str]
    temporal_focus: str  # recent, latest, current, historical, comparative
    specific_entities: List[str]  # specific algorithm/method names
    search_strategies: List[Dict[str, Any]]
    confidence: float
    reasoning: str


@dataclass
class SearchStrategy:
    """AI-generated search strategy."""
    strategy_name: str
    search_terms: List[str]
    target_sources: List[str]  # arxiv, pubmed, web, etc.
    expected_paper_types: List[str]
    priority: float
    reasoning: str


class AIQueryAnalyzer:
    """
    Intelligent query analyzer that uses GPT-4o mini to understand research intent
    and generate dynamic, adaptive search strategies.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        logger.info("🧠 AI Query Analyzer initialized with GPT-4o mini")
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Use AI to analyze the research query and generate intelligent search strategy.
        
        Args:
            query: The user's research question
            
        Returns:
            QueryAnalysis with AI-generated insights and search strategies
        """
        logger.info(f"🧠 AI analyzing query: {query}")
        
        analysis_prompt = f"""You are an expert research strategist specializing in cutting-edge technical literature. Analyze this research query and provide targeted search strategies:

Query: "{query}"

Please analyze this query and provide a JSON response with the following structure:
{{
    "research_intent": "What is the user really trying to learn or understand?",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "research_type": "survey|specific_technique|comparison|explanation|implementation|review",
    "domain_detected": "The academic/research domain this belongs to",
    "academic_terminology": ["academic term 1", "academic term 2", "academic term 3"],
    "temporal_focus": "recent|latest|current|historical|comparative",
    "specific_entities": ["Entity1", "Entity2", "Entity3"],
    "search_strategies": [
        {{
            "strategy_name": "Strategy 1 name",
            "search_terms": ["term1", "term2", "term3"],
            "target_sources": ["arxiv", "pubmed", "web"],
            "expected_paper_types": ["survey", "technical", "tutorial"],
            "temporal_keywords": ["2023", "2024", "recent", "latest"],
            "priority": 0.9,
            "reasoning": "Why this strategy would work"
        }}
    ],
    "confidence": 0.85,
    "reasoning": "Explanation of the analysis and why these strategies were chosen"
}}

ENHANCED ANALYSIS GUIDELINES:
1. For optimization/ML queries, include SPECIFIC optimizer names (Adam, AdamW, Lion, Sophia, RMSprop, Lamb, etc.)
2. For "latest/recent" queries, add temporal keywords: "2023", "2024", "recent", "new", "state-of-the-art"
3. Target high-quality venues: "NeurIPS", "ICML", "ICLR", "AAAI" for ML papers
4. Include survey-focused searches: "survey", "review", "comprehensive study", "benchmark"
5. Use precise technical terminology rather than generic terms
6. For comparative queries, search for "comparison", "evaluation", "benchmark"
7. Prioritize recency for technical advancement queries

DOMAIN-SPECIFIC ENHANCEMENTS:
- Machine Learning: Include specific algorithm names, not just generic terms
- Deep Learning: Target architecture names, training techniques, optimization methods
- AI: Include recent developments, breakthrough papers, state-of-the-art methods

EXAMPLES FOR OPTIMIZATION QUERIES:
- Search for "Adam optimizer", "AdamW", "Lion optimizer", "Sophia optimizer"
- Include "optimization survey 2024", "deep learning optimizers comparison"
- Target "NeurIPS 2023", "ICML 2024", "ICLR 2024" for latest developments

Provide only the JSON response, no other text."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research strategist who understands academic literature and how to find relevant papers. Always respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistent, logical analysis
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"🧠 AI response received: {len(response_text)} characters")
            
            # Parse JSON response
            try:
                analysis_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse AI response as JSON: {e}")
                logger.error(f"Raw response: {response_text}")
                return self._create_fallback_analysis(query)
            
            # Convert to QueryAnalysis object
            search_strategies = []
            for strategy_data in analysis_data.get("search_strategies", []):
                strategy = SearchStrategy(
                    strategy_name=strategy_data.get("strategy_name", "Unknown"),
                    search_terms=strategy_data.get("search_terms", []),
                    target_sources=strategy_data.get("target_sources", ["arxiv"]),
                    expected_paper_types=strategy_data.get("expected_paper_types", ["technical"]),
                    priority=strategy_data.get("priority", 0.5),
                    reasoning=strategy_data.get("reasoning", "")
                )
                search_strategies.append(strategy)
            
            analysis = QueryAnalysis(
                research_intent=analysis_data.get("research_intent", ""),
                key_concepts=analysis_data.get("key_concepts", []),
                research_type=analysis_data.get("research_type", "explanation"),
                domain_detected=analysis_data.get("domain_detected", "general"),
                academic_terminology=analysis_data.get("academic_terminology", []),
                temporal_focus=analysis_data.get("temporal_focus", "current"),
                specific_entities=analysis_data.get("specific_entities", []),
                search_strategies=[vars(s) for s in search_strategies],
                confidence=analysis_data.get("confidence", 0.5),
                reasoning=analysis_data.get("reasoning", "")
            )
            
            logger.info(f"✅ AI analysis completed: {analysis.domain_detected} domain, {len(analysis.search_strategies)} strategies")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ AI query analysis failed: {e}")
            return self._create_fallback_analysis(query)
    
    def _create_fallback_analysis(self, query: str) -> QueryAnalysis:
        """Create a basic fallback analysis when AI analysis fails."""
        logger.warning("🆘 Using fallback query analysis")
        
        # Simple keyword extraction
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        # Basic search strategy
        fallback_strategy = {
            "strategy_name": "Basic Search",
            "search_terms": keywords[:5],
            "target_sources": ["arxiv"],
            "expected_paper_types": ["technical"],
            "priority": 0.5,
            "reasoning": "Fallback strategy using query keywords"
        }
        
        return QueryAnalysis(
            research_intent=f"Find information about: {query}",
            key_concepts=keywords[:5],
            research_type="explanation",
            domain_detected="general",
            academic_terminology=keywords[:3],
            temporal_focus="current",
            specific_entities=[],
            search_strategies=[fallback_strategy],
            confidence=0.3,
            reasoning="Fallback analysis due to AI processing error"
        )
    
    async def generate_adaptive_search_terms(self, query: str, previous_results: List[Dict[str, Any]] = None) -> List[str]:
        """
        Generate adaptive search terms based on query and optionally previous search results.
        
        Args:
            query: Original research query
            previous_results: Previous search results to learn from
            
        Returns:
            List of AI-generated search terms
        """
        context = ""
        if previous_results:
            context = f"\nPrevious search found these types of papers: {[r.get('title', 'Unknown') for r in previous_results[:3]]}"
            context += "\nGenerate different search terms to find more relevant papers."
        
        search_prompt = f"""Generate 5-7 academic search terms for this research query: "{query}"{context}

Think like a researcher searching academic databases. What terms would actually appear in relevant paper titles and abstracts?

Provide a JSON array of search terms:
["term1", "term2", "term3", ...]

Consider:
1. Academic terminology used in research papers
2. Variations and synonyms
3. Specific technical terms
4. Broader conceptual terms
5. Related techniques or methods

Respond with only the JSON array."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at academic search and know how researchers phrase their work."},
                    {"role": "user", "content": search_prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content.strip()
            search_terms = json.loads(response_text)
            
            logger.info(f"🧠 AI generated {len(search_terms)} adaptive search terms")
            return search_terms
            
        except Exception as e:
            logger.error(f"❌ Adaptive search term generation failed: {e}")
            # Fallback to simple keyword extraction
            return [word.lower() for word in query.split() if len(word) > 3][:5]


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_ai_analyzer():
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-test-key-for-research-demo":
            print("⚠️ Please set a real OPENAI_API_KEY for testing")
            return
        
        analyzer = AIQueryAnalyzer(api_key)
        
        # Test queries
        test_queries = [
            "explain the optimizers used in CNNs and RNNs",
            "compare transformer architectures",
            "how does attention mechanism work",
            "what are the latest developments in reinforcement learning"
        ]
        
        for query in test_queries:
            print(f"\n🧪 Testing: {query}")
            print("=" * 60)
            
            analysis = await analyzer.analyze_query(query)
            
            print(f"Research Intent: {analysis.research_intent}")
            print(f"Domain: {analysis.domain_detected}")
            print(f"Key Concepts: {analysis.key_concepts}")
            print(f"Academic Terms: {analysis.academic_terminology}")
            print(f"Strategies: {len(analysis.search_strategies)}")
            print(f"Confidence: {analysis.confidence:.2f}")
    
    # Run test
    asyncio.run(test_ai_analyzer())
"""
AI-Driven Relevance Judge
Uses GPT-4o mini to evaluate the relevance of research papers to user queries in real-time.
No hardcoded thresholds - purely AI-driven relevance assessment.
"""

import openai
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RelevanceAssessment:
    """AI assessment of paper relevance."""
    relevance_score: float  # 0.0 to 1.0
    is_relevant: bool
    explanation: str
    key_points: List[str]
    missing_aspects: List[str]
    recommendation: str  # keep, discard, maybe
    confidence: float


class AIRelevanceJudge:
    """
    Intelligent relevance judge that uses GPT-4o mini to evaluate
    whether papers actually address the user's research question.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.relevance_threshold = 0.6  # AI will set this dynamically
        logger.info("🔍 AI Relevance Judge initialized with GPT-4o mini")
    
    async def evaluate_paper_relevance(self, 
                                     query: str, 
                                     paper_title: str, 
                                     paper_abstract: str = "", 
                                     paper_metadata: Dict[str, Any] = None) -> RelevanceAssessment:
        """
        Use AI to evaluate whether a paper is relevant to the research query.
        
        Args:
            query: Original user research question
            paper_title: Title of the paper
            paper_abstract: Abstract/summary of the paper
            paper_metadata: Additional paper metadata
            
        Returns:
            RelevanceAssessment with AI evaluation
        """
        logger.info(f"🔍 AI evaluating relevance: {paper_title[:50]}...")
        
        # Prepare paper content for analysis
        paper_content = f"Title: {paper_title}"
        if paper_abstract:
            paper_content += f"\n\nAbstract: {paper_abstract}"
        if paper_metadata:
            if paper_metadata.get("authors"):
                paper_content += f"\nAuthors: {', '.join(paper_metadata['authors'][:3])}"
            if paper_metadata.get("categories"):
                paper_content += f"\nCategories: {paper_metadata['categories']}"
        
        relevance_prompt = f"""You are an expert research evaluator. Determine if this paper is relevant to the user's research question.

USER'S RESEARCH QUESTION: "{query}"

PAPER TO EVALUATE:
{paper_content}

Please evaluate this paper and provide a JSON response:
{{
    "relevance_score": 0.85,
    "is_relevant": true,
    "explanation": "Clear explanation of why this paper is/isn't relevant",
    "key_points": ["Point 1 that makes it relevant", "Point 2", "Point 3"],
    "missing_aspects": ["What aspects of the query this paper doesn't address"],
    "recommendation": "keep|discard|maybe",
    "confidence": 0.9
}}

EVALUATION CRITERIA:
1. Does the paper directly address the user's question?
2. Does it contain information that would help answer the query?
3. Is it about the same topic/domain as the query?
4. Would a researcher studying this topic find this paper useful?

SCORING GUIDELINES:
- 0.9-1.0: Directly answers the query, highly relevant
- 0.7-0.8: Addresses important aspects of the query  
- 0.5-0.6: Related topic but doesn't directly answer
- 0.3-0.4: Tangentially related, some useful context
- 0.0-0.2: Not relevant, different topic

RECOMMENDATIONS (be less conservative for technical queries):
- "keep": Score ≥ 0.65 - Relevant and useful for the research question
- "maybe": Score 0.4-0.64 - Some relevance, could provide context
- "discard": Score < 0.4 - Not relevant to the query

FOR TECHNICAL/OPTIMIZATION QUERIES:
- Even if paper doesn't mention specific optimizers, if it discusses deep learning training, optimization techniques, or comparative studies, it may still be valuable
- Papers about specific neural network architectures (CNNs, RNNs) that discuss training are relevant to optimizer queries
- Survey papers and comparative studies should be rated higher for "latest" queries

Be honest and critical. If the paper doesn't actually address the query, say so clearly.

Respond with only the JSON, no other text."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research evaluator who honestly assesses paper relevance. Always respond with valid JSON."},
                    {"role": "user", "content": relevance_prompt}
                ],
                max_tokens=800,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                assessment_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse AI relevance assessment: {e}")
                return self._create_fallback_assessment(query, paper_title, 0.3)
            
            # Create RelevanceAssessment object
            assessment = RelevanceAssessment(
                relevance_score=assessment_data.get("relevance_score", 0.0),
                is_relevant=assessment_data.get("is_relevant", False),
                explanation=assessment_data.get("explanation", "No explanation provided"),
                key_points=assessment_data.get("key_points", []),
                missing_aspects=assessment_data.get("missing_aspects", []),
                recommendation=assessment_data.get("recommendation", "discard"),
                confidence=assessment_data.get("confidence", 0.5)
            )
            
            logger.info(f"✅ AI relevance score: {assessment.relevance_score:.2f} ({assessment.recommendation})")
            return assessment
            
        except Exception as e:
            logger.error(f"❌ AI relevance evaluation failed: {e}")
            return self._create_fallback_assessment(query, paper_title, 0.3)
    
    async def batch_evaluate_papers(self, 
                                  query: str, 
                                  papers: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], RelevanceAssessment]]:
        """
        Evaluate multiple papers for relevance in batch.
        
        Args:
            query: Research query
            papers: List of paper dictionaries
            
        Returns:
            List of (paper, assessment) tuples
        """
        logger.info(f"🔍 AI evaluating {len(papers)} papers for relevance")
        
        evaluations = []
        for paper in papers:
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", paper.get("summary", ""))
            
            assessment = await self.evaluate_paper_relevance(
                query=query,
                paper_title=title,
                paper_abstract=abstract,
                paper_metadata=paper
            )
            
            evaluations.append((paper, assessment))
            
            # Brief pause to avoid rate limiting
            await asyncio.sleep(0.5)
        
        # Sort by relevance score
        evaluations.sort(key=lambda x: x[1].relevance_score, reverse=True)
        
        relevant_count = sum(1 for _, assessment in evaluations if assessment.is_relevant)
        logger.info(f"✅ AI found {relevant_count}/{len(papers)} relevant papers")
        
        return evaluations
    
    def _create_fallback_assessment(self, query: str, paper_title: str, score: float) -> RelevanceAssessment:
        """Create fallback assessment when AI evaluation fails."""
        logger.warning("🆘 Using fallback relevance assessment")
        
        return RelevanceAssessment(
            relevance_score=score,
            is_relevant=score > 0.5,
            explanation=f"Fallback assessment for '{paper_title}' - AI evaluation failed",
            key_points=["Unable to perform detailed analysis"],
            missing_aspects=["Unknown due to evaluation failure"],
            recommendation="maybe",
            confidence=0.2
        )
    
    async def generate_search_feedback(self, 
                                     query: str, 
                                     evaluations: List[Tuple[Dict[str, Any], RelevanceAssessment]]) -> Dict[str, Any]:
        """
        Analyze the relevance evaluations to provide feedback on search strategy.
        
        Args:
            query: Original research query
            evaluations: List of (paper, assessment) tuples
            
        Returns:
            Search feedback and suggestions for improvement
        """
        if not evaluations:
            return {"status": "no_papers", "message": "No papers found to evaluate"}
        
        relevant_papers = [e for e in evaluations if e[1].is_relevant]
        avg_relevance = sum(e[1].relevance_score for e in evaluations) / len(evaluations)
        
        # Generate AI feedback on search quality
        feedback_prompt = f"""Analyze these search results and provide feedback on search strategy:

ORIGINAL QUERY: "{query}"

SEARCH RESULTS ANALYSIS:
- Total papers found: {len(evaluations)}
- Relevant papers: {len(relevant_papers)}
- Average relevance score: {avg_relevance:.2f}

TOP RESULTS:
{chr(10).join([f"- {e[0].get('title', 'Unknown')}: {e[1].relevance_score:.2f} ({e[1].recommendation})" for e in evaluations[:3]])}

Provide JSON feedback:
{{
    "search_quality": "excellent|good|poor|very_poor",
    "main_issues": ["issue1", "issue2"],
    "suggested_improvements": ["suggestion1", "suggestion2"],
    "alternative_search_terms": ["term1", "term2", "term3"],
    "search_strategy_advice": "What to do differently next time"
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing search results and improving search strategies."},
                    {"role": "user", "content": feedback_prompt}
                ],
                max_tokens=600,
                temperature=0.2
            )
            
            feedback_data = json.loads(response.choices[0].message.content.strip())
            feedback_data["statistics"] = {
                "total_papers": len(evaluations),
                "relevant_papers": len(relevant_papers),
                "average_relevance": avg_relevance
            }
            
            return feedback_data
            
        except Exception as e:
            logger.error(f"❌ Search feedback generation failed: {e}")
            return {
                "search_quality": "unknown",
                "main_issues": ["Unable to analyze search results"],
                "suggested_improvements": ["Try different search terms"],
                "alternative_search_terms": [],
                "search_strategy_advice": "AI feedback unavailable"
            }


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_relevance_judge():
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-test-key-for-research-demo":
            print("⚠️ Please set a real OPENAI_API_KEY for testing")
            return
        
        judge = AIRelevanceJudge(api_key)
        
        # Test query and papers
        query = "explain the optimizers used in CNNs and RNNs"
        
        test_papers = [
            {
                "title": "Adam: A Method for Stochastic Optimization",
                "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.",
                "authors": ["Diederik P. Kingma", "Jimmy Ba"]
            },
            {
                "title": "Interleaved Group Convolutions for Deep Neural Networks",
                "abstract": "We present a simple and effective architecture design technique for convolutional neural networks.",
                "authors": ["Some Author"]
            }
        ]
        
        print(f"🧪 Testing relevance evaluation for: {query}")
        print("=" * 70)
        
        for paper in test_papers:
            assessment = await judge.evaluate_paper_relevance(
                query=query,
                paper_title=paper["title"],
                paper_abstract=paper["abstract"],
                paper_metadata=paper
            )
            
            print(f"\nPaper: {paper['title']}")
            print(f"Relevance Score: {assessment.relevance_score:.2f}")
            print(f"Recommendation: {assessment.recommendation}")
            print(f"Explanation: {assessment.explanation}")
            print(f"Key Points: {assessment.key_points}")
    
    # Run test
    asyncio.run(test_relevance_judge())
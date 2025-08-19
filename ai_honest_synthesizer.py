"""
AI-Driven Honest Synthesizer
Uses GPT-4o mini to provide truthful synthesis based on what was actually found.
No forced answers from irrelevant sources - pure intellectual honesty.
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
class SynthesisResult:
    """Result of AI-driven honest synthesis."""
    synthesis_text: str
    synthesis_type: str  # complete_answer, partial_answer, no_relevant_sources, mixed_relevance
    confidence_score: float
    coverage_assessment: str
    identified_gaps: List[str]
    recommendations: List[str]
    source_quality_summary: str
    search_suggestions: List[str]


class AIHonestSynthesizer:
    """
    Intelligent synthesizer that uses GPT-4o mini to provide honest,
    truthful synthesis based on actual research findings.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        logger.info("📝 AI Honest Synthesizer initialized with GPT-4o mini")
    
    async def synthesize_findings(self, 
                                query: str,
                                paper_evaluations: List[Tuple[Dict[str, Any], Any]],
                                search_metadata: Dict[str, Any] = None) -> SynthesisResult:
        """
        Create honest synthesis based on actual research findings.
        
        Args:
            query: Original research question
            paper_evaluations: List of (paper, relevance_assessment) tuples
            search_metadata: Additional context about the search process
            
        Returns:
            SynthesisResult with honest assessment and synthesis
        """
        logger.info(f"📝 AI synthesizing findings for: {query[:50]}...")
        
        # Categorize papers by relevance
        relevant_papers = [(p, a) for p, a in paper_evaluations if a.is_relevant]
        maybe_relevant = [(p, a) for p, a in paper_evaluations if a.recommendation == "maybe"]
        irrelevant_papers = [(p, a) for p, a in paper_evaluations if not a.is_relevant and a.recommendation == "discard"]
        
        # Prepare synthesis context
        synthesis_context = self._prepare_synthesis_context(
            query, relevant_papers, maybe_relevant, irrelevant_papers, search_metadata
        )
        
        synthesis_prompt = f"""You are an expert research analyst. Provide an honest synthesis of research findings.

RESEARCH QUESTION: "{query}"

{synthesis_context}

Please provide a comprehensive, honest analysis in JSON format:
{{
    "synthesis_text": "Your complete synthesis here (400-600 words)",
    "synthesis_type": "complete_answer|partial_answer|no_relevant_sources|mixed_relevance",
    "confidence_score": 0.75,
    "coverage_assessment": "How well the found sources address the query",
    "identified_gaps": ["gap1", "gap2", "gap3"],
    "recommendations": ["recommendation1", "recommendation2"],
    "source_quality_summary": "Assessment of source quality and relevance",
    "search_suggestions": ["Better search term 1", "Better search term 2"]
}}

SYNTHESIS GUIDELINES:
1. BE COMPLETELY HONEST about what was found vs. what was asked
2. If sources don't address the query, say so clearly
3. Distinguish between what the sources DO discuss vs. what was asked about
4. Provide useful information from available sources, even if not perfectly relevant
5. Identify specific gaps in the research findings
6. Suggest how to find better sources if current ones are inadequate

SYNTHESIS TYPES:
- "complete_answer": Sources fully address the research question
- "partial_answer": Sources address some aspects but miss important parts
- "no_relevant_sources": No sources directly address the query
- "mixed_relevance": Mix of relevant and irrelevant sources

CONFIDENCE SCORING:
- 0.9-1.0: Comprehensive answer with highly relevant sources
- 0.7-0.8: Good answer with mostly relevant sources
- 0.5-0.6: Partial answer with some relevant information
- 0.3-0.4: Limited answer with minimal relevant information
- 0.0-0.2: No relevant information found

Be intellectually honest. Don't force connections that don't exist."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst who values intellectual honesty above all else. Always provide truthful assessments, even if the findings are limited."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=2000,
                temperature=0.2  # Low temperature for consistent, logical synthesis
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"📝 AI synthesis response: {response_text[:100]}...")
            
            if not response_text:
                logger.warning("⚠️ Empty synthesis response from AI")
                return self._create_fallback_synthesis(query, paper_evaluations)
            
            # Clean response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON response
            try:
                synthesis_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse AI synthesis response: {e}. Response: {response_text[:200]}")
                return self._create_fallback_synthesis(query, paper_evaluations)
            
            # Create SynthesisResult object
            result = SynthesisResult(
                synthesis_text=synthesis_data.get("synthesis_text", ""),
                synthesis_type=synthesis_data.get("synthesis_type", "mixed_relevance"),
                confidence_score=synthesis_data.get("confidence_score", 0.3),
                coverage_assessment=synthesis_data.get("coverage_assessment", ""),
                identified_gaps=synthesis_data.get("identified_gaps", []),
                recommendations=synthesis_data.get("recommendations", []),
                source_quality_summary=synthesis_data.get("source_quality_summary", ""),
                search_suggestions=synthesis_data.get("search_suggestions", [])
            )
            
            logger.info(f"✅ AI synthesis completed: {result.synthesis_type} (confidence: {result.confidence_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI synthesis failed: {e}")
            return self._create_fallback_synthesis(query, paper_evaluations)
    
    def _prepare_synthesis_context(self, 
                                 query: str,
                                 relevant_papers: List[Tuple[Dict[str, Any], Any]],
                                 maybe_relevant: List[Tuple[Dict[str, Any], Any]],
                                 irrelevant_papers: List[Tuple[Dict[str, Any], Any]],
                                 search_metadata: Dict[str, Any] = None) -> str:
        """Prepare context for AI synthesis."""
        
        context_parts = []
        
        # Search overview
        total_papers = len(relevant_papers) + len(maybe_relevant) + len(irrelevant_papers)
        context_parts.append(f"SEARCH OVERVIEW:")
        context_parts.append(f"- Total papers found: {total_papers}")
        context_parts.append(f"- Highly relevant: {len(relevant_papers)}")
        context_parts.append(f"- Possibly relevant: {len(maybe_relevant)}")
        context_parts.append(f"- Not relevant: {len(irrelevant_papers)}")
        
        if search_metadata:
            context_parts.append(f"- Search strategies used: {search_metadata.get('strategies_used', 'Unknown')}")
        
        # Relevant papers section
        if relevant_papers:
            context_parts.append(f"\nHIGHLY RELEVANT PAPERS:")
            for i, (paper, assessment) in enumerate(relevant_papers[:5], 1):
                title = paper.get("title", "Unknown Title")
                abstract = paper.get("abstract", paper.get("summary", ""))[:200] + "..."
                relevance_score = assessment.relevance_score
                explanation = assessment.explanation
                
                context_parts.append(f"\nPaper {i}: {title}")
                context_parts.append(f"Relevance Score: {relevance_score:.2f}")
                context_parts.append(f"Why Relevant: {explanation}")
                context_parts.append(f"Content: {abstract}")
        
        # Maybe relevant papers
        if maybe_relevant:
            context_parts.append(f"\nPOSSIBLY RELEVANT PAPERS:")
            for i, (paper, assessment) in enumerate(maybe_relevant[:3], 1):
                title = paper.get("title", "Unknown Title")
                explanation = assessment.explanation
                context_parts.append(f"- {title} (Score: {assessment.relevance_score:.2f}) - {explanation}")
        
        # Irrelevant papers summary
        if irrelevant_papers:
            context_parts.append(f"\nIRRELEVANT PAPERS FOUND:")
            context_parts.append(f"Found {len(irrelevant_papers)} papers not directly related to the query:")
            for i, (paper, assessment) in enumerate(irrelevant_papers[:3], 1):
                title = paper.get("title", "Unknown Title")
                context_parts.append(f"- {title} (Score: {assessment.relevance_score:.2f})")
        
        return "\n".join(context_parts)
    
    def _create_fallback_synthesis(self, 
                                 query: str, 
                                 paper_evaluations: List[Tuple[Dict[str, Any], Any]]) -> SynthesisResult:
        """Create fallback synthesis when AI processing fails."""
        logger.warning("🆘 Using fallback synthesis")
        
        total_papers = len(paper_evaluations)
        relevant_count = sum(1 for _, a in paper_evaluations if a.is_relevant)
        
        fallback_text = f"""Research Analysis for: {query}

⚠️ AI synthesis processing encountered an error, providing basic summary:

Found {total_papers} papers, {relevant_count} appear relevant to your query.

Papers found:
"""
        
        for i, (paper, assessment) in enumerate(paper_evaluations[:5], 1):
            title = paper.get("title", "Unknown Title")
            score = assessment.relevance_score
            fallback_text += f"• {title} (Relevance: {score:.2f})\n"
        
        if relevant_count == 0:
            fallback_text += "\n⚠️ No papers directly address your research question. Consider refining your search terms."
        
        return SynthesisResult(
            synthesis_text=fallback_text,
            synthesis_type="mixed_relevance",
            confidence_score=0.2,
            coverage_assessment="Unable to perform detailed assessment due to processing error",
            identified_gaps=["AI analysis unavailable"],
            recommendations=["Try alternative search terms", "Consult specialized databases"],
            source_quality_summary="Basic assessment only - AI analysis failed",
            search_suggestions=["Refine search terms", "Try different approach"]
        )
    
    async def generate_search_improvement_suggestions(self, 
                                                   query: str, 
                                                   synthesis_result: SynthesisResult,
                                                   search_history: List[str] = None) -> Dict[str, Any]:
        """
        Generate suggestions for improving search results based on synthesis outcome.
        
        Args:
            query: Original research question
            synthesis_result: Results of synthesis
            search_history: Previous search terms tried
            
        Returns:
            Improvement suggestions
        """
        improvement_prompt = f"""Based on this research outcome, suggest improvements for finding better sources:

ORIGINAL QUERY: "{query}"
SYNTHESIS TYPE: {synthesis_result.synthesis_type}
CONFIDENCE SCORE: {synthesis_result.confidence_score}
IDENTIFIED GAPS: {synthesis_result.identified_gaps}

{f"PREVIOUS SEARCH TERMS: {search_history}" if search_history else ""}

Provide JSON suggestions:
{{
    "search_strategy_changes": ["change1", "change2"],
    "alternative_databases": ["database1", "database2"],
    "refined_search_terms": ["term1", "term2", "term3"],
    "query_reformulations": ["reformulation1", "reformulation2"],
    "expert_recommendations": ["recommendation1", "recommendation2"]
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research strategist who helps improve search outcomes."},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content.strip())
            
        except Exception as e:
            logger.error(f"❌ Search improvement suggestions failed: {e}")
            return {
                "search_strategy_changes": ["Try different search terms"],
                "alternative_databases": ["Specialized academic databases"],
                "refined_search_terms": ["More specific technical terms"],
                "query_reformulations": ["Rephrase the research question"],
                "expert_recommendations": ["Consult domain experts"]
            }


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_honest_synthesizer():
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-test-key-for-research-demo":
            print("⚠️ Please set a real OPENAI_API_KEY for testing")
            return
        
        synthesizer = AIHonestSynthesizer(api_key)
        
        # Mock paper evaluations
        query = "explain the optimizers used in CNNs and RNNs"
        
        # Mock relevance assessment class
        class MockAssessment:
            def __init__(self, score, relevant, explanation):
                self.relevance_score = score
                self.is_relevant = relevant
                self.explanation = explanation
                self.recommendation = "keep" if relevant else "discard"
        
        mock_evaluations = [
            ({
                "title": "Adam: A Method for Stochastic Optimization",
                "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization."
            }, MockAssessment(0.9, True, "Directly addresses optimizers for neural networks")),
            ({
                "title": "Interleaved Group Convolutions",
                "abstract": "Architecture design technique for convolutional neural networks."
            }, MockAssessment(0.3, False, "About network architecture, not optimization"))
        ]
        
        print(f"🧪 Testing honest synthesis for: {query}")
        print("=" * 70)
        
        result = await synthesizer.synthesize_findings(query, mock_evaluations)
        
        print(f"Synthesis Type: {result.synthesis_type}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Coverage: {result.coverage_assessment}")
        print(f"Gaps: {result.identified_gaps}")
        print(f"\nSynthesis:\n{result.synthesis_text}")
    
    # Run test
    asyncio.run(test_honest_synthesizer())
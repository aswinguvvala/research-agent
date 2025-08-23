"""
AI Relevance Judge
Intelligent assessment of source relevance to research queries using GPT-4o mini.
"""

import openai
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class RelevanceAssessment:
    """Assessment of source relevance to a query."""
    source_id: str
    query: str
    relevance_score: float  # 0.0 to 1.0
    relevance_category: str  # 'highly_relevant', 'maybe_relevant', 'not_relevant'
    explanation: str
    confidence: float
    assessment_factors: Dict[str, float]
    timestamp: str


@dataclass
class SourceContext:
    """Context information about a source for relevance assessment."""
    title: str
    abstract: str
    authors: List[str]
    year: int
    venue: str
    keywords: List[str]
    citations: int
    source_type: str


class AIRelevanceJudge:
    """
    AI-powered relevance judge that evaluates source relevance to research queries
    using intelligent analysis and contextual understanding.
    """
    
    def __init__(self, openai_api_key: str, debug_mode: bool = False):
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.debug_mode = debug_mode
        
        # Relevance scoring weights
        self.scoring_weights = {
            'title_relevance': 0.25,
            'abstract_relevance': 0.35,
            'keyword_match': 0.15,
            'temporal_relevance': 0.10,
            'authority_score': 0.10,
            'citation_relevance': 0.05
        }
        
        # Relevance thresholds
        self.relevance_thresholds = {
            'highly_relevant': 0.7,
            'maybe_relevant': 0.4,
            'not_relevant': 0.0
        }
        
        logger.info("⚖️ AI Relevance Judge initialized")
    
    async def assess_relevance(self, 
                             query: str, 
                             source_context: SourceContext) -> RelevanceAssessment:
        """
        Assess the relevance of a source to a research query.
        
        Args:
            query: Research query
            source_context: Source information for assessment
            
        Returns:
            RelevanceAssessment with detailed analysis
        """
        try:
            logger.debug(f"🔍 Assessing relevance for: {source_context.title[:50]}...")
            
            # Multi-factor relevance assessment
            assessment_factors = await self._comprehensive_assessment(query, source_context)
            
            # Calculate overall relevance score
            relevance_score = self._calculate_weighted_score(assessment_factors)
            
            # Determine relevance category
            relevance_category = self._categorize_relevance(relevance_score)
            
            # Generate explanation
            explanation = await self._generate_explanation(
                query, source_context, relevance_score, assessment_factors
            )
            
            # Calculate confidence in assessment
            confidence = self._calculate_confidence(assessment_factors, source_context)
            
            assessment = RelevanceAssessment(
                source_id=f"{source_context.title[:20]}_{source_context.year}",
                query=query,
                relevance_score=relevance_score,
                relevance_category=relevance_category,
                explanation=explanation,
                confidence=confidence,
                assessment_factors=assessment_factors,
                timestamp=datetime.now().isoformat()
            )
            
            if self.debug_mode:
                logger.debug(f"📊 Relevance: {relevance_score:.2f} ({relevance_category})")
            
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Relevance assessment failed: {e}")
            return self._create_fallback_assessment(query, source_context)
    
    async def batch_assess_relevance(self, 
                                   query: str, 
                                   sources: List[SourceContext]) -> List[RelevanceAssessment]:
        """
        Assess relevance for multiple sources in parallel.
        
        Args:
            query: Research query
            sources: List of source contexts
            
        Returns:
            List of relevance assessments
        """
        logger.info(f"⚖️ Batch assessing {len(sources)} sources for relevance")
        
        # Create assessment tasks
        assessment_tasks = [
            self.assess_relevance(query, source)
            for source in sources
        ]
        
        try:
            # Execute assessments in parallel
            assessments = await asyncio.gather(*assessment_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_assessments = []
            for assessment in assessments:
                if isinstance(assessment, RelevanceAssessment):
                    valid_assessments.append(assessment)
                else:
                    logger.warning(f"Assessment failed: {assessment}")
            
            # Sort by relevance score
            valid_assessments.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"✅ Completed {len(valid_assessments)} relevance assessments")
            return valid_assessments
            
        except Exception as e:
            logger.error(f"❌ Batch assessment failed: {e}")
            return []
    
    async def _comprehensive_assessment(self, 
                                      query: str, 
                                      source: SourceContext) -> Dict[str, float]:
        """Perform comprehensive multi-factor relevance assessment."""
        try:
            # Parallel assessment of different factors
            assessment_tasks = [
                self._assess_title_relevance(query, source.title),
                self._assess_abstract_relevance(query, source.abstract),
                self._assess_keyword_match(query, source),
                self._assess_temporal_relevance(query, source.year),
                self._assess_authority_score(source),
                self._assess_citation_relevance(query, source)
            ]
            
            results = await asyncio.gather(*assessment_tasks)
            
            return {
                'title_relevance': results[0],
                'abstract_relevance': results[1],
                'keyword_match': results[2],
                'temporal_relevance': results[3],
                'authority_score': results[4],
                'citation_relevance': results[5]
            }
            
        except Exception as e:
            logger.error(f"Comprehensive assessment failed: {e}")
            return {factor: 0.5 for factor in self.scoring_weights.keys()}
    
    async def _assess_title_relevance(self, query: str, title: str) -> float:
        """Assess relevance based on title content."""
        try:
            prompt = f"""
            Assess how relevant this paper title is to the research query.
            
            Query: "{query}"
            Title: "{title}"
            
            Consider:
            - Direct topic alignment
            - Keyword overlap
            - Conceptual relevance
            - Specificity match
            
            Rate relevance from 0.0 (not relevant) to 1.0 (highly relevant).
            Return only the numeric score.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
            return max(0.0, min(1.0, score))
            
        except (ValueError, IndexError):
            # Fallback to keyword matching
            return self._simple_keyword_overlap(query, title)
        except Exception as e:
            logger.warning(f"Title relevance assessment failed: {e}")
            return 0.5
    
    async def _assess_abstract_relevance(self, query: str, abstract: str) -> float:
        """Assess relevance based on abstract content."""
        if not abstract or abstract == "No abstract available":
            return 0.3  # Reduced score for missing abstract
        
        try:
            # Truncate very long abstracts
            truncated_abstract = abstract[:800] if len(abstract) > 800 else abstract
            
            prompt = f"""
            Assess how relevant this paper abstract is to the research query.
            
            Query: "{query}"
            Abstract: "{truncated_abstract}"
            
            Consider:
            - Topic alignment
            - Research focus overlap
            - Methodological relevance
            - Problem domain match
            - Solution approach relevance
            
            Rate relevance from 0.0 (not relevant) to 1.0 (highly relevant).
            Return only the numeric score.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
            return max(0.0, min(1.0, score))
            
        except (ValueError, IndexError):
            # Fallback to keyword matching
            return self._simple_keyword_overlap(query, abstract)
        except Exception as e:
            logger.warning(f"Abstract relevance assessment failed: {e}")
            return 0.5
    
    async def _assess_keyword_match(self, query: str, source: SourceContext) -> float:
        """Assess relevance based on keyword matching."""
        try:
            # Extract keywords from query
            query_keywords = self._extract_keywords(query)
            
            # Combine source keywords with title words
            source_keywords = source.keywords + self._extract_keywords(source.title)
            
            # Calculate overlap
            if not query_keywords or not source_keywords:
                return 0.3
            
            # Convert to lowercase for comparison
            query_set = set(word.lower() for word in query_keywords)
            source_set = set(word.lower() for word in source_keywords)
            
            # Calculate Jaccard similarity
            intersection = query_set.intersection(source_set)
            union = query_set.union(source_set)
            
            if not union:
                return 0.0
            
            jaccard_score = len(intersection) / len(union)
            
            # Also check for partial matches
            partial_matches = 0
            for q_word in query_keywords:
                for s_word in source_keywords:
                    if (q_word.lower() in s_word.lower() or 
                        s_word.lower() in q_word.lower()) and len(q_word) > 3:
                        partial_matches += 1
                        break
            
            partial_score = partial_matches / len(query_keywords) if query_keywords else 0
            
            # Combine scores
            final_score = (jaccard_score * 0.7) + (partial_score * 0.3)
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning(f"Keyword matching failed: {e}")
            return 0.3
    
    async def _assess_temporal_relevance(self, query: str, year: int) -> float:
        """Assess temporal relevance based on query requirements and publication year."""
        try:
            current_year = datetime.now().year
            
            # Check for temporal indicators in query
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['recent', 'latest', 'current', '2023', '2024']):
                # Recent research preferred
                if year >= current_year - 1:
                    return 1.0
                elif year >= current_year - 3:
                    return 0.8
                elif year >= current_year - 5:
                    return 0.6
                else:
                    return 0.3
            
            elif any(word in query_lower for word in ['historical', 'evolution', 'development']):
                # Historical perspective preferred
                age = current_year - year
                if age >= 10:
                    return 1.0
                elif age >= 5:
                    return 0.8
                else:
                    return 0.6
            
            else:
                # General temporal relevance
                age = current_year - year
                if age <= 2:
                    return 1.0
                elif age <= 5:
                    return 0.9
                elif age <= 10:
                    return 0.7
                elif age <= 15:
                    return 0.5
                else:
                    return 0.3
                    
        except Exception as e:
            logger.warning(f"Temporal assessment failed: {e}")
            return 0.6
    
    async def _assess_authority_score(self, source: SourceContext) -> float:
        """Assess source authority based on venue and publication context."""
        try:
            score = 0.0
            
            # Venue assessment
            venue_lower = source.venue.lower()
            
            # Top-tier venues
            if any(top_venue in venue_lower for top_venue in 
                  ['nature', 'science', 'cell', 'pnas', 'nejm']):
                score += 0.4
            
            # High-quality conferences
            elif any(conf in venue_lower for conf in 
                    ['neurips', 'icml', 'iclr', 'acl', 'emnlp', 'cvpr', 'iccv']):
                score += 0.35
            
            # Reputable journals
            elif any(journal_type in venue_lower for journal_type in 
                    ['journal', 'proceedings', 'transactions']):
                score += 0.3
            
            # ArXiv and preprints
            elif 'arxiv' in venue_lower or 'preprint' in venue_lower:
                score += 0.2
            
            # Unknown venue
            else:
                score += 0.1
            
            # Author count (more authors might indicate collaborative work)
            author_count = len(source.authors)
            if author_count >= 5:
                score += 0.1
            elif author_count >= 3:
                score += 0.05
            
            # Publication type bonus
            if source.source_type == 'semantic_scholar':
                score += 0.1  # Generally higher quality curation
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Authority assessment failed: {e}")
            return 0.5
    
    async def _assess_citation_relevance(self, query: str, source: SourceContext) -> float:
        """Assess relevance based on citation metrics."""
        try:
            citations = source.citations
            year = source.year
            current_year = datetime.now().year
            
            # Age-adjusted citation score
            age = max(1, current_year - year)
            citations_per_year = citations / age
            
            # Citation score thresholds (age-adjusted)
            if citations_per_year >= 50:
                return 1.0
            elif citations_per_year >= 20:
                return 0.8
            elif citations_per_year >= 10:
                return 0.6
            elif citations_per_year >= 5:
                return 0.4
            elif citations > 0:
                return 0.2
            else:
                return 0.1  # New papers without citations yet
                
        except Exception as e:
            logger.warning(f"Citation assessment failed: {e}")
            return 0.3
    
    def _calculate_weighted_score(self, factors: Dict[str, float]) -> float:
        """Calculate weighted relevance score from assessment factors."""
        weighted_score = 0.0
        
        for factor, score in factors.items():
            if factor in self.scoring_weights:
                weighted_score += score * self.scoring_weights[factor]
        
        return max(0.0, min(1.0, weighted_score))
    
    def _categorize_relevance(self, score: float) -> str:
        """Categorize relevance score into discrete categories."""
        if score >= self.relevance_thresholds['highly_relevant']:
            return 'highly_relevant'
        elif score >= self.relevance_thresholds['maybe_relevant']:
            return 'maybe_relevant'
        else:
            return 'not_relevant'
    
    async def _generate_explanation(self, 
                                  query: str, 
                                  source: SourceContext, 
                                  score: float, 
                                  factors: Dict[str, float]) -> str:
        """Generate explanation for the relevance assessment."""
        try:
            top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
            factor_descriptions = []
            
            for factor, value in top_factors:
                if value > 0.7:
                    strength = "Strong"
                elif value > 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                factor_name = factor.replace('_', ' ').title()
                factor_descriptions.append(f"{strength} {factor_name.lower()} ({value:.2f})")
            
            explanation = f"Relevance score: {score:.2f}. "
            explanation += f"Key factors: {', '.join(factor_descriptions)}. "
            
            if score >= 0.7:
                explanation += "This source appears highly relevant to the research query."
            elif score >= 0.4:
                explanation += "This source may be relevant with some useful information."
            else:
                explanation += "This source has limited relevance to the research query."
            
            return explanation
            
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return f"Relevance score: {score:.2f}"
    
    def _calculate_confidence(self, 
                            factors: Dict[str, float], 
                            source: SourceContext) -> float:
        """Calculate confidence in the relevance assessment."""
        confidence = 0.5  # Base confidence
        
        # Factor score consistency
        factor_values = list(factors.values())
        if factor_values:
            std_dev = (sum((x - sum(factor_values)/len(factor_values))**2 for x in factor_values) / len(factor_values))**0.5
            consistency_bonus = max(0, 0.3 - std_dev)
            confidence += consistency_bonus
        
        # Abstract availability
        if source.abstract and source.abstract != "No abstract available":
            confidence += 0.1
        
        # Source metadata completeness
        if source.authors and source.venue:
            confidence += 0.1
        
        # Recent publication (more confidence in assessment)
        current_year = datetime.now().year
        if current_year - source.year <= 5:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _simple_keyword_overlap(self, query: str, text: str) -> float:
        """Simple keyword overlap calculation as fallback."""
        query_words = set(self._extract_keywords(query))
        text_words = set(self._extract_keywords(text))
        
        if not query_words or not text_words:
            return 0.3
        
        overlap = len(query_words.intersection(text_words))
        return min(1.0, overlap / len(query_words))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'use', 'way', 'she', 'many', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye', 'bed', 'own', 'say', 'too', 'any', 'try', 'ask', 'man', 'end', 'why', 'let', 'put', 'big', 'got', 'make', 'come', 'here', 'this', 'that', 'what', 'when', 'where', 'which', 'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
        
        return [word for word in words if word not in stop_words and len(word) > 3]
    
    def _create_fallback_assessment(self, 
                                  query: str, 
                                  source: SourceContext) -> RelevanceAssessment:
        """Create fallback assessment when main assessment fails."""
        # Simple keyword-based assessment
        fallback_score = self._simple_keyword_overlap(query, source.title + " " + source.abstract)
        
        return RelevanceAssessment(
            source_id=f"{source.title[:20]}_{source.year}",
            query=query,
            relevance_score=fallback_score,
            relevance_category=self._categorize_relevance(fallback_score),
            explanation=f"Fallback assessment based on keyword overlap: {fallback_score:.2f}",
            confidence=0.3,
            assessment_factors={'keyword_match': fallback_score},
            timestamp=datetime.now().isoformat()
        )


async def test_relevance_judge():
    """Test function for the relevance judge."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    judge = AIRelevanceJudge(api_key, debug_mode=True)
    
    query = "machine learning transformer architectures for natural language processing"
    
    # Create test source
    test_source = SourceContext(
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NIPS",
        keywords=["transformer", "attention", "neural networks"],
        citations=45000,
        source_type="semantic_scholar"
    )
    
    print(f"🔍 Testing relevance assessment for: {query}")
    assessment = await judge.assess_relevance(query, test_source)
    
    print(f"\nRelevance Score: {assessment.relevance_score:.2f}")
    print(f"Category: {assessment.relevance_category}")
    print(f"Confidence: {assessment.confidence:.2f}")
    print(f"Explanation: {assessment.explanation}")


if __name__ == "__main__":
    asyncio.run(test_relevance_judge())
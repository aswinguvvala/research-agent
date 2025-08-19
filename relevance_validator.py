"""
Relevance Validator for Research Agent
Validates source relevance to prevent source-content mismatch problems.
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
import logging

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False
    print("⚠️  Advanced features disabled. Install: pip install sentence-transformers scikit-learn")

logger = logging.getLogger(__name__)


@dataclass
class RelevanceScore:
    """Detailed relevance scoring with explanations."""
    overall_score: float
    semantic_score: float
    keyword_score: float
    domain_score: float
    title_score: float
    content_score: float
    explanation: str
    confidence: float
    passes_threshold: bool


class RelevanceValidator:
    """Validates source relevance to research queries using multiple scoring methods."""
    
    def __init__(self, relevance_threshold: float = 0.4, use_semantic: bool = True):
        self.relevance_threshold = relevance_threshold
        self.fallback_threshold = 0.25  # More lenient fallback threshold
        self.use_semantic = use_semantic and ADVANCED_FEATURES
        
        # Initialize semantic model if available
        self.semantic_model = None
        if self.use_semantic:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic similarity model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.use_semantic = False
        
        # Domain-specific keywords for technical topics
        self.domain_keywords = {
            'docker': [
                'container', 'containerization', 'dockerfile', 'docker-compose',
                'microservices', 'orchestration', 'kubernetes', 'virtualization',
                'deployment', 'devops', 'image', 'registry', 'volume', 'network'
            ],
            'machine_learning': [
                'algorithm', 'neural network', 'deep learning', 'training', 'model',
                'prediction', 'classification', 'regression', 'feature', 'dataset',
                'supervised', 'unsupervised', 'reinforcement', 'accuracy', 'validation'
            ],
            'deep_learning': [
                'cnn', 'rnn', 'lstm', 'gru', 'convolutional', 'recurrent', 'neural network',
                'backpropagation', 'gradient', 'vanishing gradient', 'exploding gradient',
                'deep neural network', 'transformer', 'attention', 'dropout', 'activation',
                'sigmoid', 'relu', 'tanh', 'softmax', 'loss function', 'optimizer'
            ],
            'ai': [
                'artificial intelligence', 'neural', 'cognitive', 'reasoning', 'learning',
                'inference', 'knowledge', 'expert system', 'natural language', 'vision',
                'planning', 'optimization', 'search', 'heuristic', 'automation'
            ],
            'medical': [
                'treatment', 'diagnosis', 'patient', 'clinical', 'therapy', 'disease',
                'symptom', 'medicine', 'healthcare', 'pharmaceutical', 'trial',
                'efficacy', 'dosage', 'adverse', 'intervention'
            ],
            'web_development': [
                'frontend', 'backend', 'javascript', 'html', 'css', 'framework',
                'api', 'database', 'server', 'client', 'responsive', 'security',
                'performance', 'optimization', 'deployment'
            ]
        }
        
        # Technical acronym expansions
        self.acronym_expansions = {
            'cnn': 'convolutional neural network',
            'cnns': 'convolutional neural networks',
            'rnn': 'recurrent neural network', 
            'rnns': 'recurrent neural networks',
            'lstm': 'long short term memory',
            'gru': 'gated recurrent unit',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision'
        }
    
    def expand_query_terms(self, query: str) -> str:
        """Expand acronyms and clean query for better matching."""
        # Clean up common typos and expand acronyms
        expanded_query = query.lower()
        
        # Fix common typos
        expanded_query = re.sub(r'\bawe\b', 'and why do we', expanded_query)
        expanded_query = re.sub(r'\bwhy dowe\b', 'why do we', expanded_query)
        
        # Expand acronyms
        words = expanded_query.split()
        expanded_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.acronym_expansions:
                expanded_words.append(self.acronym_expansions[clean_word])
                expanded_words.append(clean_word)  # Keep original too
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def validate_source_relevance(self, 
                                  query: str, 
                                  source_data: Dict[str, Any]) -> RelevanceScore:
        """
        Validate source relevance using multiple scoring methods.
        
        Args:
            query: Research query
            source_data: Source metadata and content
            
        Returns:
            RelevanceScore with detailed breakdown
        """
        # Expand and clean the query
        expanded_query = self.expand_query_terms(query)
        
        # Extract source text components
        title = source_data.get('title', '')
        abstract = source_data.get('abstract', source_data.get('summary', ''))
        content = source_data.get('content', '')
        
        # Combine all available text
        full_text = f"{title} {abstract} {content}".strip()
        
        if not full_text:
            return RelevanceScore(
                overall_score=0.0,
                semantic_score=0.0,
                keyword_score=0.0,
                domain_score=0.0,
                title_score=0.0,
                content_score=0.0,
                explanation="No content available for relevance analysis",
                confidence=0.0,
                passes_threshold=False
            )
        
        # Calculate individual scores using expanded query
        semantic_score = self._calculate_semantic_similarity(expanded_query, full_text)
        keyword_score = self._calculate_keyword_overlap(expanded_query, full_text)
        domain_score = self._calculate_domain_relevance(expanded_query, full_text)
        title_score = self._calculate_title_relevance(expanded_query, title)
        content_score = self._calculate_content_relevance(expanded_query, content or abstract)
        
        # Weighted overall score
        weights = {
            'semantic': 0.35 if self.use_semantic else 0.0,
            'keyword': 0.25,
            'domain': 0.20,
            'title': 0.15,
            'content': 0.05
        }
        
        # Adjust weights if semantic is disabled
        if not self.use_semantic:
            weights['keyword'] = 0.40
            weights['domain'] = 0.30
            weights['title'] = 0.20
            weights['content'] = 0.10
        
        overall_score = (
            semantic_score * weights['semantic'] +
            keyword_score * weights['keyword'] +
            domain_score * weights['domain'] +
            title_score * weights['title'] +
            content_score * weights['content']
        )
        
        # Calculate confidence based on available information
        confidence = self._calculate_confidence(source_data, full_text)
        
        # Generate explanation
        explanation = self._generate_explanation(
            query, semantic_score, keyword_score, domain_score, title_score, content_score
        )
        
        # Check if passes threshold
        passes_threshold = overall_score >= self.relevance_threshold
        
        return RelevanceScore(
            overall_score=round(overall_score, 3),
            semantic_score=round(semantic_score, 3),
            keyword_score=round(keyword_score, 3),
            domain_score=round(domain_score, 3),
            title_score=round(title_score, 3),
            content_score=round(content_score, 3),
            explanation=explanation,
            confidence=round(confidence, 3),
            passes_threshold=passes_threshold
        )
    
    def _calculate_semantic_similarity(self, query: str, text: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not self.use_semantic or not self.semantic_model:
            return 0.0
        
        try:
            # Encode query and text
            query_embedding = self.semantic_model.encode([query])
            text_embedding = self.semantic_model.encode([text[:1000]])  # Limit text length
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, text_embedding)[0][0]
            return max(0.0, float(similarity))
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """Calculate keyword overlap between query and text."""
        # Normalize text
        query_words = self._normalize_text(query)
        text_words = self._normalize_text(text)
        
        if not query_words or not text_words:
            return 0.0
        
        # Calculate intersection
        query_set = set(query_words)
        text_set = set(text_words)
        
        intersection = query_set.intersection(text_set)
        
        # Jaccard similarity with emphasis on query coverage
        union = query_set.union(text_set)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Query coverage (what percentage of query words appear in text)
        query_coverage = len(intersection) / len(query_set) if query_set else 0.0
        
        # Weighted combination favoring query coverage
        return 0.3 * jaccard + 0.7 * query_coverage
    
    def _calculate_domain_relevance(self, query: str, text: str) -> float:
        """Calculate domain-specific relevance."""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Detect query domain
        detected_domains = []
        for domain, keywords in self.domain_keywords.items():
            domain_score = sum(1 for keyword in keywords if keyword in query_lower)
            if domain_score > 0:
                detected_domains.append((domain, domain_score))
        
        if not detected_domains:
            # Generic scoring if no domain detected
            return self._calculate_generic_domain_score(query_lower, text_lower)
        
        # Calculate domain-specific relevance
        max_score = 0.0
        for domain, _ in detected_domains:
            domain_keywords = self.domain_keywords[domain]
            matches = sum(1 for keyword in domain_keywords if keyword in text_lower)
            domain_score = matches / len(domain_keywords)
            max_score = max(max_score, domain_score)
        
        return max_score
    
    def _calculate_generic_domain_score(self, query: str, text: str) -> float:
        """Calculate generic domain relevance for unknown domains."""
        # Extract potential technical terms from query
        query_terms = re.findall(r'\b[a-zA-Z]{4,}\b', query)
        
        if not query_terms:
            return 0.0
        
        # Count how many query terms appear in text
        matches = sum(1 for term in query_terms if term.lower() in text)
        return matches / len(query_terms)
    
    def _calculate_title_relevance(self, query: str, title: str) -> float:
        """Calculate relevance based on title matching."""
        if not title:
            return 0.0
        
        # Title carries high weight, so use multiple methods
        keyword_score = self._calculate_keyword_overlap(query, title)
        
        # Check for exact phrase matches
        query_phrases = self._extract_phrases(query)
        title_lower = title.lower()
        phrase_matches = sum(1 for phrase in query_phrases if phrase in title_lower)
        phrase_score = phrase_matches / len(query_phrases) if query_phrases else 0.0
        
        # Combine scores
        return 0.7 * keyword_score + 0.3 * phrase_score
    
    def _calculate_content_relevance(self, query: str, content: str) -> float:
        """Calculate relevance based on main content."""
        if not content:
            return 0.0
        
        # Use keyword overlap for content relevance
        return self._calculate_keyword_overlap(query, content[:2000])  # Limit content length
    
    def _calculate_confidence(self, source_data: Dict[str, Any], full_text: str) -> float:
        """Calculate confidence in the relevance assessment."""
        confidence_factors = []
        
        # Text length factor
        text_length = len(full_text)
        if text_length > 500:
            confidence_factors.append(0.9)
        elif text_length > 200:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Metadata completeness
        metadata_score = 0.0
        if source_data.get('title'):
            metadata_score += 0.3
        if source_data.get('authors'):
            metadata_score += 0.2
        if source_data.get('abstract') or source_data.get('summary'):
            metadata_score += 0.3
        if source_data.get('year') or source_data.get('published'):
            metadata_score += 0.1
        if source_data.get('journal') or source_data.get('source_type'):
            metadata_score += 0.1
        
        confidence_factors.append(metadata_score)
        
        # Semantic model availability
        if self.use_semantic:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for keyword matching."""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        return [word for word in words if word not in stop_words]
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text."""
        # Simple phrase extraction (2-3 word combinations)
        words = self._normalize_text(text)
        phrases = []
        
        # 2-word phrases
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        # 3-word phrases
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return phrases
    
    def _generate_explanation(self, 
                              query: str,
                              semantic_score: float,
                              keyword_score: float,
                              domain_score: float,
                              title_score: float,
                              content_score: float) -> str:
        """Generate human-readable explanation of relevance scoring."""
        explanations = []
        
        if semantic_score > 0:
            if semantic_score > 0.7:
                explanations.append(f"High semantic similarity ({semantic_score:.2f})")
            elif semantic_score > 0.4:
                explanations.append(f"Moderate semantic similarity ({semantic_score:.2f})")
            else:
                explanations.append(f"Low semantic similarity ({semantic_score:.2f})")
        
        if keyword_score > 0.6:
            explanations.append(f"Strong keyword overlap ({keyword_score:.2f})")
        elif keyword_score > 0.3:
            explanations.append(f"Moderate keyword overlap ({keyword_score:.2f})")
        else:
            explanations.append(f"Weak keyword overlap ({keyword_score:.2f})")
        
        if domain_score > 0.5:
            explanations.append(f"Good domain relevance ({domain_score:.2f})")
        elif domain_score > 0.2:
            explanations.append(f"Some domain relevance ({domain_score:.2f})")
        else:
            explanations.append(f"Limited domain relevance ({domain_score:.2f})")
        
        if title_score > 0.5:
            explanations.append(f"Relevant title match ({title_score:.2f})")
        
        return "; ".join(explanations)
    
    def batch_validate_sources(self, 
                               query: str, 
                               sources: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], RelevanceScore]]:
        """
        Validate multiple sources and return only those passing threshold.
        
        Args:
            query: Research query
            sources: List of source data dictionaries
            
        Returns:
            List of (source, relevance_score) tuples for sources passing threshold
        """
        if not sources:
            logger.warning("No sources provided for validation")
            return []
        
        # First pass with normal threshold
        validated_sources = []
        all_scores = []
        
        for source in sources:
            relevance_score = self.validate_source_relevance(query, source)
            all_scores.append((source, relevance_score))
            
            if relevance_score.passes_threshold:
                validated_sources.append((source, relevance_score))
            else:
                logger.info(f"Source rejected: {source.get('title', 'Unknown')} "
                          f"(score: {relevance_score.overall_score:.3f}, "
                          f"threshold: {self.relevance_threshold})")
        
        # If no sources pass normal threshold, try fallback threshold
        if not validated_sources:
            logger.warning(f"No sources passed primary threshold {self.relevance_threshold:.2f}, trying fallback threshold {self.fallback_threshold:.2f}")
            
            for source, relevance_score in all_scores:
                if relevance_score.overall_score >= self.fallback_threshold:
                    # Update the score to reflect it passed fallback threshold
                    fallback_score = RelevanceScore(
                        overall_score=relevance_score.overall_score,
                        semantic_score=relevance_score.semantic_score,
                        keyword_score=relevance_score.keyword_score,
                        domain_score=relevance_score.domain_score,
                        title_score=relevance_score.title_score,
                        content_score=relevance_score.content_score,
                        explanation=f"{relevance_score.explanation} (fallback threshold)",
                        confidence=relevance_score.confidence * 0.8,  # Reduce confidence for fallback
                        passes_threshold=True
                    )
                    validated_sources.append((source, fallback_score))
                    logger.info(f"Source accepted via fallback: {source.get('title', 'Unknown')} "
                              f"(score: {relevance_score.overall_score:.3f})")
        
        # If still no sources, take top scoring sources anyway
        if not validated_sources and all_scores:
            logger.warning("No sources passed fallback threshold, taking top scoring sources")
            # Sort by score and take top 2-3 sources
            all_scores.sort(key=lambda x: x[1].overall_score, reverse=True)
            top_sources = all_scores[:min(3, len(all_scores))]
            
            for source, relevance_score in top_sources:
                emergency_score = RelevanceScore(
                    overall_score=relevance_score.overall_score,
                    semantic_score=relevance_score.semantic_score,
                    keyword_score=relevance_score.keyword_score,
                    domain_score=relevance_score.domain_score,
                    title_score=relevance_score.title_score,
                    content_score=relevance_score.content_score,
                    explanation=f"{relevance_score.explanation} (emergency fallback)",
                    confidence=relevance_score.confidence * 0.6,  # Further reduce confidence
                    passes_threshold=True
                )
                validated_sources.append((source, emergency_score))
                logger.info(f"Source accepted via emergency fallback: {source.get('title', 'Unknown')} "
                          f"(score: {relevance_score.overall_score:.3f})")
        
        # Sort by relevance score (highest first)
        validated_sources.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        logger.info(f"Final validation result: {len(validated_sources)}/{len(sources)} sources accepted")
        return validated_sources
    
    def get_validation_stats(self, 
                             query: str, 
                             sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get validation statistics for a set of sources."""
        total_sources = len(sources)
        validated_sources = self.batch_validate_sources(query, sources)
        passed_sources = len(validated_sources)
        
        if validated_sources:
            scores = [score.overall_score for _, score in validated_sources]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            avg_score = max_score = min_score = 0.0
        
        return {
            'total_sources': total_sources,
            'passed_sources': passed_sources,
            'rejection_rate': (total_sources - passed_sources) / total_sources if total_sources > 0 else 0.0,
            'average_score': round(avg_score, 3),
            'max_score': round(max_score, 3),
            'min_score': round(min_score, 3),
            'threshold': self.relevance_threshold
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize validator
    validator = RelevanceValidator(relevance_threshold=0.6)
    
    # Test query
    query = "Docker containerization and deployment best practices"
    
    # Sample sources (mix of relevant and irrelevant)
    test_sources = [
        {
            'title': 'Docker in Production: Best Practices for Container Deployment',
            'abstract': 'This paper discusses best practices for deploying Docker containers in production environments, covering security, scalability, and monitoring.',
            'authors': ['John Smith'],
            'year': '2023',
            'source_type': 'article'
        },
        {
            'title': 'Machine Learning Model Training Optimization',
            'abstract': 'We present novel approaches to optimize machine learning model training using GPU acceleration and distributed computing.',
            'authors': ['Jane Doe'],
            'year': '2023',
            'source_type': 'article'
        },
        {
            'title': 'Container Orchestration with Kubernetes and Docker',
            'abstract': 'A comprehensive guide to orchestrating containerized applications using Kubernetes, focusing on Docker container management.',
            'authors': ['Bob Wilson'],
            'year': '2024',
            'source_type': 'article'
        }
    ]
    
    print(f"🔍 Testing Relevance Validator")
    print(f"Query: {query}")
    print(f"Threshold: {validator.relevance_threshold}")
    print("=" * 60)
    
    # Test individual sources
    for i, source in enumerate(test_sources, 1):
        print(f"\nSource {i}: {source['title']}")
        score = validator.validate_source_relevance(query, source)
        
        print(f"Overall Score: {score.overall_score}")
        print(f"Passes Threshold: {score.passes_threshold}")
        print(f"Explanation: {score.explanation}")
        print(f"Confidence: {score.confidence}")
    
    # Test batch validation
    print(f"\n{'='*60}")
    print("Batch Validation Results:")
    
    validated = validator.batch_validate_sources(query, test_sources)
    stats = validator.get_validation_stats(query, test_sources)
    
    print(f"Total sources: {stats['total_sources']}")
    print(f"Passed sources: {stats['passed_sources']}")
    print(f"Rejection rate: {stats['rejection_rate']:.1%}")
    print(f"Average score: {stats['average_score']}")
    
    print(f"\nSources passing threshold:")
    for source, score in validated:
        print(f"- {source['title']} (score: {score.overall_score})")
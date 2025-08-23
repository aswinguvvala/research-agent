"""
AI Query Analyzer
Advanced analysis of research intent, domain detection, and query understanding.
"""

import openai
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Complete analysis of a research query."""
    original_query: str
    research_intent: str
    domain: str
    key_concepts: List[str]
    temporal_focus: str
    expected_sources: List[str]
    search_strategies: List[str]
    complexity_level: str
    suggested_keywords: List[str]
    related_topics: List[str]
    confidence_score: float
    analysis_metadata: Dict[str, Any]


@dataclass
class DomainProfile:
    """Profile of a research domain."""
    name: str
    keywords: List[str]
    typical_venues: List[str]
    search_patterns: List[str]
    quality_indicators: List[str]
    related_domains: List[str]


class AIQueryAnalyzer:
    """
    AI-powered query analyzer that understands research intent,
    identifies domains, and suggests optimal search strategies.
    """
    
    def __init__(self, openai_api_key: str, debug_mode: bool = False):
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.debug_mode = debug_mode
        
        # Define domain profiles
        self.domain_profiles = self._initialize_domain_profiles()
        
        # Research intent patterns
        self.intent_patterns = {
            'survey': ['overview', 'review', 'survey', 'comprehensive', 'state of the art'],
            'comparison': ['compare', 'vs', 'versus', 'comparison', 'difference', 'contrast'],
            'technical': ['implementation', 'algorithm', 'method', 'technique', 'approach'],
            'evaluation': ['performance', 'evaluation', 'benchmark', 'metrics', 'analysis'],
            'trends': ['recent', 'latest', 'current', 'trends', 'emerging', 'future'],
            'application': ['application', 'use case', 'practical', 'real-world', 'deployment'],
            'theoretical': ['theory', 'theoretical', 'mathematical', 'formal', 'proof']
        }
        
        # Temporal indicators
        self.temporal_indicators = {
            'recent': ['recent', 'latest', 'current', '2023', '2024', 'new', 'emerging'],
            'historical': ['history', 'evolution', 'development', 'origins', 'historical'],
            'foundational': ['foundational', 'seminal', 'classic', 'original', 'fundamental'],
            'future': ['future', 'upcoming', 'next', 'predictions', 'forecast']
        }
        
        logger.info("🧠 AI Query Analyzer initialized")
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Comprehensive analysis of research query.
        
        Args:
            query: Research query to analyze
            
        Returns:
            QueryAnalysis with complete understanding
        """
        logger.info(f"🔍 Analyzing query: {query}")
        
        try:
            # Parallel analysis tasks
            analysis_tasks = [
                self._analyze_research_intent(query),
                self._detect_domain(query),
                self._extract_key_concepts(query),
                self._determine_temporal_focus(query),
                self._suggest_search_strategies(query)
            ]
            
            # Execute analyses in parallel
            results = await asyncio.gather(*analysis_tasks)
            
            research_intent = results[0]
            domain = results[1]
            key_concepts = results[2]
            temporal_focus = results[3]
            search_strategies = results[4]
            
            # Generate additional analysis elements
            expected_sources = self._determine_expected_sources(domain, research_intent)
            complexity_level = self._assess_complexity(query, key_concepts)
            suggested_keywords = await self._generate_keywords(query, domain)
            related_topics = await self._find_related_topics(query, domain)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(query, domain, key_concepts)
            
            # Create analysis result
            analysis = QueryAnalysis(
                original_query=query,
                research_intent=research_intent,
                domain=domain,
                key_concepts=key_concepts,
                temporal_focus=temporal_focus,
                expected_sources=expected_sources,
                search_strategies=search_strategies,
                complexity_level=complexity_level,
                suggested_keywords=suggested_keywords,
                related_topics=related_topics,
                confidence_score=confidence_score,
                analysis_metadata={
                    'analysis_time': datetime.now().isoformat(),
                    'analyzer_version': '1.0',
                    'domain_profile': self.domain_profiles.get(domain, {})
                }
            )
            
            logger.info(f"✅ Query analysis completed - Domain: {domain}, Intent: {research_intent}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            # Return basic analysis on failure
            return self._create_fallback_analysis(query)
    
    async def _analyze_research_intent(self, query: str) -> str:
        """Analyze the research intent behind the query."""
        try:
            # Check for pattern matches first
            query_lower = query.lower()
            for intent, patterns in self.intent_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    return intent
            
            # Use AI for more nuanced analysis
            prompt = f"""
            Analyze the research intent of this query: "{query}"
            
            Classify the intent as one of:
            - survey: Comprehensive overview or review
            - comparison: Comparing different approaches
            - technical: Implementation details or methods
            - evaluation: Performance analysis or benchmarking
            - trends: Recent developments or emerging topics
            - application: Practical use cases or deployment
            - theoretical: Mathematical or theoretical foundations
            
            Return only the classification word.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            intent = response.choices[0].message.content.strip().lower()
            
            # Validate response
            if intent in self.intent_patterns:
                return intent
            else:
                return 'survey'  # Default fallback
                
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            return 'survey'
    
    async def _detect_domain(self, query: str) -> str:
        """Detect the research domain of the query."""
        try:
            # Check domain profiles for keyword matches
            query_lower = query.lower()
            domain_scores = {}
            
            for domain_name, profile in self.domain_profiles.items():
                score = 0
                for keyword in profile['keywords']:
                    if keyword.lower() in query_lower:
                        score += 1
                
                if score > 0:
                    domain_scores[domain_name] = score
            
            # Return highest scoring domain
            if domain_scores:
                return max(domain_scores, key=domain_scores.get)
            
            # Use AI for domain detection if no clear match
            prompt = f"""
            Identify the research domain for this query: "{query}"
            
            Choose from these domains:
            - computer_science: AI, machine learning, algorithms, software
            - medicine: Medical research, healthcare, clinical studies
            - physics: Physics, materials science, engineering
            - biology: Biology, genetics, life sciences
            - social_science: Psychology, sociology, education
            - business: Economics, management, finance
            - interdisciplinary: Cross-domain research
            
            Return only the domain name.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            domain = response.choices[0].message.content.strip().lower()
            
            # Validate response
            if domain in self.domain_profiles:
                return domain
            else:
                return 'interdisciplinary'  # Default fallback
                
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            return 'interdisciplinary'
    
    async def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        try:
            prompt = f"""
            Extract the key concepts from this research query: "{query}"
            
            Identify 3-7 main concepts that are essential to understanding the research topic.
            Focus on:
            - Technical terms
            - Main subjects
            - Important modifiers
            - Domain-specific concepts
            
            Return as a simple list, one concept per line.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            concepts_text = response.choices[0].message.content.strip()
            concepts = [c.strip() for c in concepts_text.split('\n') if c.strip()]
            
            # Clean up concepts (remove numbering, etc.)
            cleaned_concepts = []
            for concept in concepts:
                clean_concept = re.sub(r'^\d+\.?\s*', '', concept)  # Remove numbering
                clean_concept = clean_concept.strip('- ')  # Remove bullets
                if clean_concept:
                    cleaned_concepts.append(clean_concept)
            
            return cleaned_concepts[:7]  # Limit to 7 concepts
            
        except Exception as e:
            logger.warning(f"Concept extraction failed: {e}")
            # Fallback: simple word extraction
            words = query.split()
            return [word for word in words if len(word) > 3][:5]
    
    async def _determine_temporal_focus(self, query: str) -> str:
        """Determine the temporal focus of the research."""
        query_lower = query.lower()
        
        # Check for temporal indicators
        for temporal_type, indicators in self.temporal_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return temporal_type
        
        # Default based on query characteristics
        if any(word in query_lower for word in ['algorithm', 'method', 'technique']):
            return 'foundational'
        elif any(word in query_lower for word in ['application', 'practical', 'deployment']):
            return 'recent'
        else:
            return 'comprehensive'  # Both recent and foundational
    
    async def _suggest_search_strategies(self, query: str) -> List[str]:
        """Suggest optimal search strategies for the query."""
        try:
            prompt = f"""
            Suggest 3-5 search strategies for this research query: "{query}"
            
            Consider strategies like:
            - Broad keyword search
            - Author-specific search
            - Venue-specific search
            - Citation following
            - Time-bound search
            - Method-specific search
            - Application-domain search
            
            Return as a simple list, one strategy per line.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4
            )
            
            strategies_text = response.choices[0].message.content.strip()
            strategies = [s.strip() for s in strategies_text.split('\n') if s.strip()]
            
            # Clean up strategies
            cleaned_strategies = []
            for strategy in strategies:
                clean_strategy = re.sub(r'^\d+\.?\s*', '', strategy)
                clean_strategy = clean_strategy.strip('- ')
                if clean_strategy:
                    cleaned_strategies.append(clean_strategy)
            
            return cleaned_strategies[:5]
            
        except Exception as e:
            logger.warning(f"Strategy suggestion failed: {e}")
            return ['Broad keyword search', 'Recent papers search', 'Citation following']
    
    def _determine_expected_sources(self, domain: str, intent: str) -> List[str]:
        """Determine expected source types based on domain and intent."""
        sources = ['academic_papers']
        
        if domain in self.domain_profiles:
            profile = self.domain_profiles[domain]
            sources.extend(profile.get('typical_venues', []))
        
        # Add intent-specific sources
        if intent == 'survey':
            sources.extend(['review_papers', 'survey_papers'])
        elif intent == 'technical':
            sources.extend(['conference_papers', 'technical_reports'])
        elif intent == 'trends':
            sources.extend(['recent_papers', 'preprints'])
        elif intent == 'application':
            sources.extend(['case_studies', 'application_papers'])
        
        return list(set(sources))  # Remove duplicates
    
    def _assess_complexity(self, query: str, concepts: List[str]) -> str:
        """Assess the complexity level of the research query."""
        complexity_score = 0
        
        # Query length factor
        if len(query.split()) > 10:
            complexity_score += 1
        
        # Number of concepts
        if len(concepts) > 5:
            complexity_score += 1
        
        # Technical terms
        technical_indicators = ['algorithm', 'optimization', 'neural', 'quantum', 'statistical']
        if any(term in query.lower() for term in technical_indicators):
            complexity_score += 1
        
        # Multiple domains
        if 'and' in query.lower() or '&' in query:
            complexity_score += 1
        
        # Interdisciplinary indicators
        interdisciplinary_words = ['interdisciplinary', 'cross-domain', 'multi-disciplinary']
        if any(word in query.lower() for word in interdisciplinary_words):
            complexity_score += 1
        
        if complexity_score >= 3:
            return 'high'
        elif complexity_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    async def _generate_keywords(self, query: str, domain: str) -> List[str]:
        """Generate additional keywords for search optimization."""
        try:
            domain_context = ""
            if domain in self.domain_profiles:
                keywords = self.domain_profiles[domain]['keywords']
                domain_context = f"Domain context: {', '.join(keywords[:5])}"
            
            prompt = f"""
            Generate 5-8 search keywords for this research query: "{query}"
            {domain_context}
            
            Include:
            - Synonyms for key terms
            - Related technical terms
            - Alternative phrasings
            - Domain-specific terminology
            
            Return as a simple list, one keyword per line.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
            
            # Clean up keywords
            cleaned_keywords = []
            for keyword in keywords:
                clean_keyword = re.sub(r'^\d+\.?\s*', '', keyword)
                clean_keyword = clean_keyword.strip('- ')
                if clean_keyword:
                    cleaned_keywords.append(clean_keyword)
            
            return cleaned_keywords[:8]
            
        except Exception as e:
            logger.warning(f"Keyword generation failed: {e}")
            return []
    
    async def _find_related_topics(self, query: str, domain: str) -> List[str]:
        """Find related research topics."""
        try:
            prompt = f"""
            Find 4-6 related research topics for: "{query}"
            
            Suggest topics that are:
            - Closely related but distinct
            - Useful for expanding the research
            - In similar or adjacent domains
            - Potentially complementary
            
            Return as a simple list, one topic per line.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.5
            )
            
            topics_text = response.choices[0].message.content.strip()
            topics = [t.strip() for t in topics_text.split('\n') if t.strip()]
            
            # Clean up topics
            cleaned_topics = []
            for topic in topics:
                clean_topic = re.sub(r'^\d+\.?\s*', '', topic)
                clean_topic = clean_topic.strip('- ')
                if clean_topic:
                    cleaned_topics.append(clean_topic)
            
            return cleaned_topics[:6]
            
        except Exception as e:
            logger.warning(f"Related topics generation failed: {e}")
            return []
    
    def _calculate_confidence(self, query: str, domain: str, concepts: List[str]) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.5  # Base confidence
        
        # Query clarity
        if len(query.split()) >= 3:
            confidence += 0.1
        
        # Domain match
        if domain in self.domain_profiles:
            confidence += 0.2
        
        # Concept extraction success
        if len(concepts) >= 3:
            confidence += 0.1
        
        # Technical specificity
        technical_terms = ['algorithm', 'method', 'technique', 'model', 'system']
        if any(term in query.lower() for term in technical_terms):
            confidence += 0.1
        
        # Query length (not too short or too long)
        word_count = len(query.split())
        if 5 <= word_count <= 15:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_fallback_analysis(self, query: str) -> QueryAnalysis:
        """Create a basic analysis when main analysis fails."""
        return QueryAnalysis(
            original_query=query,
            research_intent='survey',
            domain='interdisciplinary',
            key_concepts=query.split()[:5],
            temporal_focus='comprehensive',
            expected_sources=['academic_papers'],
            search_strategies=['Broad keyword search'],
            complexity_level='medium',
            suggested_keywords=[],
            related_topics=[],
            confidence_score=0.3,
            analysis_metadata={'fallback': True}
        )
    
    def _initialize_domain_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain profiles for research areas."""
        return {
            'computer_science': {
                'keywords': ['algorithm', 'machine learning', 'AI', 'neural network', 'deep learning', 
                           'software', 'programming', 'data science', 'computer vision', 'NLP'],
                'typical_venues': ['conference_papers', 'preprints', 'journal_articles'],
                'quality_indicators': ['peer_reviewed', 'high_citations', 'recent_publication']
            },
            'medicine': {
                'keywords': ['clinical', 'medical', 'patient', 'treatment', 'therapy', 'diagnosis',
                           'healthcare', 'disease', 'drug', 'pharmaceutical'],
                'typical_venues': ['clinical_trials', 'medical_journals', 'case_studies'],
                'quality_indicators': ['randomized_controlled_trial', 'peer_reviewed', 'clinical_evidence']
            },
            'physics': {
                'keywords': ['quantum', 'particle', 'energy', 'electromagnetic', 'materials',
                           'theoretical physics', 'experimental', 'optics', 'mechanics'],
                'typical_venues': ['physics_journals', 'preprints', 'conference_proceedings'],
                'quality_indicators': ['experimental_validation', 'theoretical_rigor', 'reproducibility']
            },
            'biology': {
                'keywords': ['gene', 'protein', 'cell', 'molecular', 'genetics', 'evolution',
                           'organism', 'biological', 'biochemistry', 'biotechnology'],
                'typical_venues': ['biology_journals', 'research_articles', 'review_papers'],
                'quality_indicators': ['experimental_data', 'peer_reviewed', 'reproducible_results']
            },
            'social_science': {
                'keywords': ['social', 'psychology', 'behavior', 'society', 'culture',
                           'education', 'human', 'survey', 'qualitative', 'quantitative'],
                'typical_venues': ['social_science_journals', 'survey_studies', 'ethnographic_studies'],
                'quality_indicators': ['statistical_significance', 'sample_size', 'methodology']
            },
            'business': {
                'keywords': ['economics', 'finance', 'management', 'market', 'business',
                           'strategy', 'organization', 'leadership', 'innovation'],
                'typical_venues': ['business_journals', 'case_studies', 'market_research'],
                'quality_indicators': ['empirical_evidence', 'longitudinal_study', 'peer_reviewed']
            },
            'interdisciplinary': {
                'keywords': ['interdisciplinary', 'cross-domain', 'multi-disciplinary', 'complex systems'],
                'typical_venues': ['interdisciplinary_journals', 'conference_papers', 'review_articles'],
                'quality_indicators': ['cross_domain_validation', 'comprehensive_approach', 'novel_insights']
            }
        }


async def test_query_analyzer():
    """Test function for the query analyzer."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    analyzer = AIQueryAnalyzer(api_key, debug_mode=True)
    
    test_queries = [
        "machine learning transformer architectures for natural language processing",
        "clinical trials for COVID-19 treatment effectiveness",
        "quantum computing applications in cryptography"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Analyzing: {query}")
        analysis = await analyzer.analyze_query(query)
        
        print(f"Domain: {analysis.domain}")
        print(f"Intent: {analysis.research_intent}")
        print(f"Complexity: {analysis.complexity_level}")
        print(f"Temporal Focus: {analysis.temporal_focus}")
        print(f"Key Concepts: {', '.join(analysis.key_concepts)}")
        print(f"Confidence: {analysis.confidence_score:.2f}")


if __name__ == "__main__":
    asyncio.run(test_query_analyzer())
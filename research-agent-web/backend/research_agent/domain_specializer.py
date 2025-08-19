"""
Domain Specializer for Research Agent
Provides domain-specific search patterns and source prioritization.
"""

import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ResearchDomain(Enum):
    """Research domain classifications."""
    COMPUTER_SCIENCE = "computer_science"
    MEDICAL = "medical"
    ENGINEERING = "engineering"
    BUSINESS = "business"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    SOCIAL_SCIENCE = "social_science"
    GENERAL = "general"


@dataclass
class DomainProfile:
    """Profile for a specific research domain."""
    domain: ResearchDomain
    confidence: float
    keywords: List[str]
    source_priorities: Dict[str, float]
    search_strategies: List[str]
    quality_criteria: Dict[str, float]
    specialized_sources: List[str]


@dataclass
class SourcePrioritization:
    """Source prioritization rules for a domain."""
    official_docs_weight: float
    academic_papers_weight: float
    industry_reports_weight: float
    community_content_weight: float
    news_articles_weight: float
    preferred_publishers: List[str]
    avoid_sources: List[str]


class DomainSpecializer:
    """Specializes research approach based on detected domain."""
    
    def __init__(self):
        # Domain detection patterns
        self.domain_patterns = {
            ResearchDomain.COMPUTER_SCIENCE: {
                'keywords': [
                    'software', 'algorithm', 'programming', 'computer', 'machine learning',
                    'artificial intelligence', 'database', 'network', 'security', 'web',
                    'mobile', 'cloud', 'distributed', 'docker', 'kubernetes', 'api',
                    'framework', 'javascript', 'python', 'java', 'react', 'node',
                    'microservices', 'devops', 'agile', 'git', 'performance', 'scalability',
                    # Software Development Lifecycle
                    'sdlc', 'software development', 'development lifecycle', 'methodology',
                    'waterfall', 'scrum', 'kanban', 'sprint', 'iteration', 'requirements',
                    'design', 'implementation', 'testing', 'deployment', 'maintenance',
                    'quality assurance', 'project management', 'software engineering'
                ],
                'technical_terms': [
                    'containerization', 'virtualization', 'orchestration', 'deployment',
                    'frontend', 'backend', 'fullstack', 'restful', 'graphql', 'json',
                    'authentication', 'authorization', 'encryption', 'ssl', 'https',
                    # SDLC specific terms
                    'requirements analysis', 'system design', 'code review', 'unit testing',
                    'integration testing', 'user acceptance testing', 'continuous integration',
                    'continuous deployment', 'version control', 'change management',
                    'risk management', 'spiral model', 'v-model', 'prototype model'
                ]
            },
            ResearchDomain.MEDICAL: {
                'keywords': [
                    'medical', 'clinical', 'patient', 'treatment', 'diagnosis', 'therapy',
                    'pharmaceutical', 'drug', 'medicine', 'healthcare', 'hospital',
                    'disease', 'symptom', 'surgery', 'cancer', 'diabetes', 'covid',
                    'vaccine', 'clinical trial', 'efficacy', 'safety', 'adverse'
                ],
                'technical_terms': [
                    'randomized controlled trial', 'meta-analysis', 'systematic review',
                    'biomarker', 'pharmacokinetics', 'pharmacodynamics', 'dosage',
                    'placebo', 'double-blind', 'peer-reviewed', 'pubmed', 'medline'
                ]
            },
            ResearchDomain.ENGINEERING: {
                'keywords': [
                    'engineering', 'mechanical', 'electrical', 'civil', 'chemical',
                    'design', 'manufacturing', 'materials', 'structural', 'thermal',
                    'fluid', 'control', 'automation', 'robotics', 'sensors', 'actuators'
                ],
                'technical_terms': [
                    'finite element analysis', 'cad', 'simulation', 'modeling',
                    'optimization', 'stress analysis', 'fatigue', 'reliability'
                ]
            },
            ResearchDomain.BUSINESS: {
                'keywords': [
                    'business', 'management', 'marketing', 'finance', 'economics',
                    'strategy', 'operations', 'supply chain', 'leadership', 'innovation',
                    'entrepreneurship', 'investment', 'roi', 'profit', 'revenue',
                    'market', 'customer', 'sales', 'growth', 'competitive'
                ],
                'technical_terms': [
                    'market analysis', 'swot analysis', 'business model', 'value proposition',
                    'competitive advantage', 'market share', 'customer acquisition'
                ]
            },
            ResearchDomain.PHYSICS: {
                'keywords': [
                    'physics', 'quantum', 'mechanics', 'thermodynamics', 'electromagnetism',
                    'optics', 'particle', 'energy', 'force', 'motion', 'wave', 'field'
                ],
                'technical_terms': [
                    'quantum mechanics', 'relativity', 'electromagnetic field',
                    'wave function', 'particle physics', 'condensed matter'
                ]
            },
            ResearchDomain.CHEMISTRY: {
                'keywords': [
                    'chemistry', 'chemical', 'molecule', 'atom', 'reaction', 'catalyst',
                    'synthesis', 'compound', 'element', 'bond', 'organic', 'inorganic'
                ],
                'technical_terms': [
                    'chemical reaction', 'molecular structure', 'spectroscopy',
                    'chromatography', 'catalysis', 'thermodynamics'
                ]
            },
            ResearchDomain.BIOLOGY: {
                'keywords': [
                    'biology', 'biological', 'cell', 'gene', 'protein', 'dna', 'rna',
                    'organism', 'evolution', 'ecology', 'genetics', 'molecular'
                ],
                'technical_terms': [
                    'molecular biology', 'cell biology', 'genetics', 'genomics',
                    'proteomics', 'bioinformatics', 'phylogenetics'
                ]
            },
            ResearchDomain.SOCIAL_SCIENCE: {
                'keywords': [
                    'social', 'psychology', 'sociology', 'anthropology', 'political',
                    'education', 'culture', 'society', 'behavior', 'human', 'community'
                ],
                'technical_terms': [
                    'social research', 'qualitative analysis', 'quantitative analysis',
                    'survey methodology', 'ethnography', 'case study'
                ]
            }
        }
        
        # Domain-specific source prioritization
        self.source_prioritizations = {
            ResearchDomain.COMPUTER_SCIENCE: SourcePrioritization(
                official_docs_weight=0.9,
                academic_papers_weight=0.85,
                industry_reports_weight=0.8,
                community_content_weight=0.7,
                news_articles_weight=0.5,
                preferred_publishers=[
                    'ACM', 'IEEE', 'arXiv', 'GitHub', 'Stack Overflow',
                    'Google Research', 'Microsoft Research', 'OpenAI',
                    'Mozilla Developer Network', 'W3C'
                ],
                avoid_sources=['personal blogs', 'unverified tutorials']
            ),
            ResearchDomain.MEDICAL: SourcePrioritization(
                official_docs_weight=0.95,
                academic_papers_weight=0.9,
                industry_reports_weight=0.7,
                community_content_weight=0.4,
                news_articles_weight=0.3,
                preferred_publishers=[
                    'PubMed', 'Cochrane', 'New England Journal of Medicine',
                    'The Lancet', 'JAMA', 'BMJ', 'Nature Medicine',
                    'FDA', 'WHO', 'CDC', 'NIH'
                ],
                avoid_sources=['health blogs', 'unverified medical advice']
            ),
            ResearchDomain.ENGINEERING: SourcePrioritization(
                official_docs_weight=0.85,
                academic_papers_weight=0.8,
                industry_reports_weight=0.85,
                community_content_weight=0.6,
                news_articles_weight=0.5,
                preferred_publishers=[
                    'IEEE', 'ASME', 'ASCE', 'SAE', 'AIAA',
                    'Engineering standards organizations',
                    'Professional engineering societies'
                ],
                avoid_sources=['non-technical blogs', 'promotional content']
            ),
            ResearchDomain.BUSINESS: SourcePrioritization(
                official_docs_weight=0.8,
                academic_papers_weight=0.75,
                industry_reports_weight=0.9,
                community_content_weight=0.65,
                news_articles_weight=0.7,
                preferred_publishers=[
                    'Harvard Business Review', 'McKinsey', 'Deloitte',
                    'PwC', 'Boston Consulting Group', 'Gartner',
                    'Business journals', 'Financial Times', 'Wall Street Journal'
                ],
                avoid_sources=['promotional content', 'biased industry sources']
            ),
            ResearchDomain.GENERAL: SourcePrioritization(
                official_docs_weight=0.8,
                academic_papers_weight=0.8,
                industry_reports_weight=0.7,
                community_content_weight=0.6,
                news_articles_weight=0.6,
                preferred_publishers=[
                    'Academic journals', 'Government publications',
                    'Established research institutions'
                ],
                avoid_sources=['unreliable sources', 'promotional content']
            )
        }
        
        # Domain-specific search strategies
        self.search_strategies = {
            ResearchDomain.COMPUTER_SCIENCE: [
                'official_documentation_first',
                'academic_papers_for_algorithms',
                'community_best_practices',
                'performance_benchmarks',
                'security_considerations'
            ],
            ResearchDomain.MEDICAL: [
                'systematic_reviews_first',
                'randomized_controlled_trials',
                'meta_analyses',
                'clinical_guidelines',
                'peer_reviewed_only'
            ],
            ResearchDomain.ENGINEERING: [
                'standards_and_specifications',
                'technical_papers',
                'industry_best_practices',
                'case_studies',
                'professional_guidelines'
            ],
            ResearchDomain.BUSINESS: [
                'market_research_reports',
                'case_studies',
                'industry_analyses',
                'academic_business_research',
                'expert_opinions'
            ]
        }
        
        # Domain-specific quality criteria
        self.quality_criteria = {
            ResearchDomain.COMPUTER_SCIENCE: {
                'peer_review': 0.8,
                'recency': 0.9,  # Tech moves fast
                'implementation_details': 0.8,
                'benchmarks': 0.7,
                'open_source': 0.6
            },
            ResearchDomain.MEDICAL: {
                'peer_review': 0.95,
                'sample_size': 0.9,
                'methodology': 0.95,
                'ethics_approval': 0.9,
                'replication': 0.8
            },
            ResearchDomain.ENGINEERING: {
                'peer_review': 0.85,
                'standards_compliance': 0.9,
                'testing_validation': 0.85,
                'practical_application': 0.8,
                'safety_considerations': 0.9
            },
            ResearchDomain.BUSINESS: {
                'data_quality': 0.8,
                'sample_representativeness': 0.8,
                'methodology': 0.75,
                'bias_considerations': 0.8,
                'practical_relevance': 0.9
            }
        }
    
    def detect_domain(self, query: str, sources: Optional[List[Dict[str, Any]]] = None) -> DomainProfile:
        """
        Detect the research domain from query and sources.
        
        Args:
            query: Research query
            sources: Optional list of sources for additional context
            
        Returns:
            DomainProfile with detected domain and confidence
        """
        query_lower = query.lower()
        domain_scores = defaultdict(float)
        
        # Score based on query keywords
        for domain, patterns in self.domain_patterns.items():
            score = 0.0
            
            # Check regular keywords
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    score += 1.0
            
            # Check technical terms (higher weight)
            for term in patterns.get('technical_terms', []):
                if term in query_lower:
                    score += 2.0
            
            # Normalize by keyword count
            total_keywords = len(patterns['keywords']) + len(patterns.get('technical_terms', []))
            if total_keywords > 0:
                domain_scores[domain] = score / total_keywords
        
        # Additional scoring from sources if available
        if sources:
            source_domain_scores = self._analyze_source_domains(sources)
            for domain, score in source_domain_scores.items():
                domain_scores[domain] += score * 0.3  # 30% weight from sources
        
        # Find best domain
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            domain, confidence = best_domain
            
            # If confidence is too low, use general domain
            if confidence < 0.1:
                domain = ResearchDomain.GENERAL
                confidence = 0.5
        else:
            domain = ResearchDomain.GENERAL
            confidence = 0.5
        
        # Get domain-specific configuration
        keywords = self.domain_patterns.get(domain, {}).get('keywords', [])
        source_priorities = self._get_source_priorities(domain)
        search_strategies = self.search_strategies.get(domain, ['comprehensive_search'])
        quality_criteria = self.quality_criteria.get(domain, {})
        specialized_sources = self._get_specialized_sources(domain)
        
        profile = DomainProfile(
            domain=domain,
            confidence=round(confidence, 3),
            keywords=keywords,
            source_priorities=source_priorities,
            search_strategies=search_strategies,
            quality_criteria=quality_criteria,
            specialized_sources=specialized_sources
        )
        
        logger.info(f"Detected domain: {domain.value} (confidence: {confidence:.3f})")
        return profile
    
    def _analyze_source_domains(self, sources: List[Dict[str, Any]]) -> Dict[ResearchDomain, float]:
        """Analyze sources to infer domain."""
        domain_scores = defaultdict(float)
        
        for source in sources:
            # Analyze source type
            source_type = source.get('source_type', '')
            if source_type == 'pubmed':
                domain_scores[ResearchDomain.MEDICAL] += 1.0
            elif source_type == 'arxiv':
                # ArXiv categories can hint at domain
                categories = source.get('categories', [])
                for category in categories:
                    if 'cs.' in category.lower():
                        domain_scores[ResearchDomain.COMPUTER_SCIENCE] += 1.0
                    elif 'physics' in category.lower():
                        domain_scores[ResearchDomain.PHYSICS] += 1.0
                    elif 'math' in category.lower():
                        domain_scores[ResearchDomain.COMPUTER_SCIENCE] += 0.5
            
            # Analyze title and abstract
            text_content = f"{source.get('title', '')} {source.get('abstract', '')}".lower()
            for domain, patterns in self.domain_patterns.items():
                for keyword in patterns['keywords'][:10]:  # Check top keywords
                    if keyword in text_content:
                        domain_scores[domain] += 0.1
        
        # Normalize scores
        if sources:
            for domain in domain_scores:
                domain_scores[domain] /= len(sources)
        
        return dict(domain_scores)
    
    def _get_source_priorities(self, domain: ResearchDomain) -> Dict[str, float]:
        """Get source priority weights for domain."""
        prioritization = self.source_prioritizations.get(domain, 
                                                       self.source_prioritizations[ResearchDomain.GENERAL])
        
        return {
            'official_docs': prioritization.official_docs_weight,
            'academic_papers': prioritization.academic_papers_weight,
            'industry_reports': prioritization.industry_reports_weight,
            'community_content': prioritization.community_content_weight,
            'news_articles': prioritization.news_articles_weight
        }
    
    def _get_specialized_sources(self, domain: ResearchDomain) -> List[str]:
        """Get specialized sources for domain."""
        prioritization = self.source_prioritizations.get(domain,
                                                       self.source_prioritizations[ResearchDomain.GENERAL])
        return prioritization.preferred_publishers
    
    def prioritize_sources(self, sources: List[Dict[str, Any]], domain_profile: DomainProfile) -> List[Tuple[Dict[str, Any], float]]:
        """
        Prioritize sources based on domain-specific criteria.
        
        Args:
            sources: List of source data
            domain_profile: Domain profile with prioritization rules
            
        Returns:
            List of (source, priority_score) tuples, sorted by priority
        """
        prioritized_sources = []
        
        for source in sources:
            priority_score = self._calculate_source_priority(source, domain_profile)
            prioritized_sources.append((source, priority_score))
        
        # Sort by priority score (highest first)
        prioritized_sources.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized_sources
    
    def _calculate_source_priority(self, source: Dict[str, Any], domain_profile: DomainProfile) -> float:
        """Calculate priority score for a source based on domain."""
        score = 0.5  # Base score
        
        # Source type scoring
        source_type = source.get('source_type', '').lower()
        if source_type == 'arxiv' and domain_profile.domain in [ResearchDomain.COMPUTER_SCIENCE, ResearchDomain.PHYSICS]:
            score += 0.3
        elif source_type == 'pubmed' and domain_profile.domain == ResearchDomain.MEDICAL:
            score += 0.4
        elif source_type == 'journal':
            score += 0.2
        elif source_type == 'conference':
            score += 0.15
        
        # Publisher/source reputation
        title = source.get('title', '').lower()
        journal = source.get('journal', '').lower()
        url = source.get('url', '').lower()
        
        # Check against preferred publishers
        for publisher in domain_profile.specialized_sources:
            publisher_lower = publisher.lower()
            if (publisher_lower in title or 
                publisher_lower in journal or 
                publisher_lower in url):
                score += 0.2
                break
        
        # Avoid low-quality sources
        avoid_indicators = ['blog', 'forum', 'social', 'wiki', 'personal']
        for indicator in avoid_indicators:
            if indicator in url or indicator in source.get('source_type', ''):
                score -= 0.1
        
        # Recency bonus (domain-dependent)
        year = source.get('year')
        if year:
            try:
                year_int = int(year)
                current_year = 2024  # Should be dynamic in production
                age = current_year - year_int
                
                if domain_profile.domain == ResearchDomain.COMPUTER_SCIENCE:
                    # Tech values recency highly
                    if age <= 2:
                        score += 0.15
                    elif age <= 5:
                        score += 0.05
                elif domain_profile.domain == ResearchDomain.MEDICAL:
                    # Medical values recency moderately
                    if age <= 5:
                        score += 0.1
                    elif age <= 10:
                        score += 0.05
                else:
                    # Other domains less sensitive to age
                    if age <= 10:
                        score += 0.05
            except ValueError:
                pass
        
        # Author information bonus
        if source.get('authors'):
            score += 0.05
        
        # DOI bonus (indicates formal publication)
        if source.get('doi'):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_domain_specific_quality_weights(self, domain_profile: DomainProfile) -> Dict[str, float]:
        """Get quality criteria weights for the domain."""
        return domain_profile.quality_criteria
    
    def get_search_recommendations(self, domain_profile: DomainProfile, query: str) -> List[str]:
        """Get domain-specific search recommendations."""
        recommendations = []
        
        # Domain-specific strategies
        strategies = domain_profile.search_strategies
        
        if 'official_documentation_first' in strategies:
            recommendations.append("Prioritize official documentation and specifications")
        
        if 'systematic_reviews_first' in strategies:
            recommendations.append("Search for systematic reviews and meta-analyses first")
        
        if 'academic_papers_for_algorithms' in strategies:
            recommendations.append("Include academic papers for algorithmic approaches")
        
        if 'performance_benchmarks' in strategies:
            recommendations.append("Look for performance benchmarks and comparisons")
        
        if 'security_considerations' in strategies:
            recommendations.append("Include security analysis and vulnerability assessments")
        
        if 'clinical_guidelines' in strategies:
            recommendations.append("Search for clinical practice guidelines")
        
        if 'industry_best_practices' in strategies:
            recommendations.append("Include industry best practices and case studies")
        
        # Domain-specific source recommendations
        if domain_profile.specialized_sources:
            sources_str = ', '.join(domain_profile.specialized_sources[:5])
            recommendations.append(f"Prioritize sources from: {sources_str}")
        
        return recommendations
    
    def validate_domain_coverage(self, query: str, sources: List[Dict[str, Any]], domain_profile: DomainProfile) -> Dict[str, Any]:
        """Validate that sources provide adequate domain coverage."""
        coverage_analysis = {
            'domain_relevant_sources': 0,
            'preferred_publisher_coverage': 0,
            'quality_source_ratio': 0.0,
            'recency_coverage': 0,
            'coverage_gaps': [],
            'recommendations': []
        }
        
        if not sources:
            coverage_analysis['coverage_gaps'].append("No sources available")
            return coverage_analysis
        
        # Analyze source domain relevance
        domain_relevant = 0
        preferred_publisher_count = 0
        recent_sources = 0
        high_quality_sources = 0
        
        for source in sources:
            # Check domain relevance
            title_content = f"{source.get('title', '')} {source.get('abstract', '')}".lower()
            relevance_score = 0
            for keyword in domain_profile.keywords[:20]:  # Check top keywords
                if keyword in title_content:
                    relevance_score += 1
            
            if relevance_score > 0:
                domain_relevant += 1
            
            # Check preferred publishers
            for publisher in domain_profile.specialized_sources:
                if publisher.lower() in source.get('journal', '').lower() or \
                   publisher.lower() in source.get('url', '').lower():
                    preferred_publisher_count += 1
                    break
            
            # Check recency
            year = source.get('year')
            if year:
                try:
                    if int(year) >= 2020:
                        recent_sources += 1
                except ValueError:
                    pass
            
            # Check quality indicators
            quality_score = 0
            if source.get('authors'):
                quality_score += 1
            if source.get('doi'):
                quality_score += 1
            if source.get('journal'):
                quality_score += 1
            if source.get('source_type') in ['arxiv', 'pubmed', 'journal']:
                quality_score += 1
            
            if quality_score >= 2:
                high_quality_sources += 1
        
        # Calculate coverage metrics
        total_sources = len(sources)
        coverage_analysis.update({
            'domain_relevant_sources': domain_relevant,
            'preferred_publisher_coverage': preferred_publisher_count,
            'quality_source_ratio': high_quality_sources / total_sources,
            'recency_coverage': recent_sources,
        })
        
        # Identify gaps and recommendations
        if domain_relevant / total_sources < 0.7:
            coverage_analysis['coverage_gaps'].append("Low domain relevance in sources")
            coverage_analysis['recommendations'].append("Search with more domain-specific terms")
        
        if preferred_publisher_count == 0:
            coverage_analysis['coverage_gaps'].append("No sources from preferred publishers")
            coverage_analysis['recommendations'].append(f"Include sources from: {', '.join(domain_profile.specialized_sources[:3])}")
        
        if high_quality_sources / total_sources < 0.5:
            coverage_analysis['coverage_gaps'].append("Low proportion of high-quality sources")
            coverage_analysis['recommendations'].append("Prioritize peer-reviewed and authoritative sources")
        
        if recent_sources == 0:
            coverage_analysis['coverage_gaps'].append("No recent sources found")
            coverage_analysis['recommendations'].append("Include more recent publications if available")
        
        return coverage_analysis
    
    def generate_domain_report(self, domain_profile: DomainProfile, coverage_analysis: Dict[str, Any]) -> str:
        """Generate domain specialization report."""
        report_lines = [
            "Domain Specialization Report",
            "=" * 40,
            f"Detected Domain: {domain_profile.domain.value.replace('_', ' ').title()}",
            f"Confidence: {domain_profile.confidence:.3f}",
            ""
        ]
        
        # Domain characteristics
        report_lines.extend([
            "Domain Characteristics:",
            f"  Key Keywords: {', '.join(domain_profile.keywords[:10])}",
            f"  Specialized Sources: {', '.join(domain_profile.specialized_sources[:5])}",
            f"  Search Strategies: {', '.join(domain_profile.search_strategies)}",
            ""
        ])
        
        # Source prioritization
        report_lines.extend([
            "Source Prioritization:",
            f"  Official Documentation: {domain_profile.source_priorities.get('official_docs', 0):.2f}",
            f"  Academic Papers: {domain_profile.source_priorities.get('academic_papers', 0):.2f}",
            f"  Industry Reports: {domain_profile.source_priorities.get('industry_reports', 0):.2f}",
            f"  Community Content: {domain_profile.source_priorities.get('community_content', 0):.2f}",
            ""
        ])
        
        # Coverage analysis with defensive error handling
        report_lines.extend([
            "Coverage Analysis:",
            f"  Domain Relevant Sources: {coverage_analysis.get('domain_relevant_sources', 'N/A')}",
            f"  Preferred Publisher Coverage: {coverage_analysis.get('preferred_publisher_coverage', 'N/A')}",
            f"  Quality Source Ratio: {coverage_analysis.get('quality_source_ratio', 0.0):.1%}",
            f"  Recent Sources: {coverage_analysis.get('recency_coverage', 'N/A')}",
            ""
        ])
        
        # Gaps and recommendations with defensive error handling
        coverage_gaps = coverage_analysis.get('coverage_gaps', [])
        if coverage_gaps:
            report_lines.append("Coverage Gaps:")
            for gap in coverage_gaps:
                report_lines.append(f"  • {gap}")
            report_lines.append("")
        
        recommendations = coverage_analysis.get('recommendations', [])
        if recommendations:
            report_lines.append("Recommendations:")
            for rec in recommendations:
                report_lines.append(f"  • {rec}")
        
        return "\n".join(report_lines)


# Example usage and testing
if __name__ == "__main__":
    # Initialize domain specializer
    specializer = DomainSpecializer()
    
    # Test queries for different domains
    test_queries = [
        "Docker containerization best practices and performance optimization",
        "COVID-19 vaccine effectiveness in preventing severe disease",
        "Machine learning algorithms for natural language processing",
        "Structural analysis of steel bridges under dynamic loading",
        "Market analysis for electric vehicle adoption trends"
    ]
    
    # Sample sources
    sample_sources = [
        {
            'title': 'Docker Performance Optimization Guide',
            'source_type': 'documentation',
            'url': 'https://docs.docker.com/performance',
            'year': '2023'
        },
        {
            'title': 'Container Security Analysis',
            'source_type': 'arxiv',
            'categories': ['cs.CR'],
            'authors': ['John Doe'],
            'year': '2024'
        }
    ]
    
    print("🎯 Testing Domain Specializer")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Detect domain
        profile = specializer.detect_domain(query, sample_sources)
        
        print(f"Domain: {profile.domain.value}")
        print(f"Confidence: {profile.confidence:.3f}")
        
        # Get prioritized sources
        prioritized = specializer.prioritize_sources(sample_sources, profile)
        print(f"Source Priorities: {[round(score, 3) for _, score in prioritized]}")
        
        # Get recommendations
        recommendations = specializer.get_search_recommendations(profile, query)
        print(f"Recommendations: {len(recommendations)} items")
        
        # Validate coverage
        coverage = specializer.validate_domain_coverage(query, sample_sources, profile)
        print(f"Coverage Gaps: {len(coverage['coverage_gaps'])}")
        
        print("")
"""
Cross Validator for Research Agent
Analyzes multiple source perspectives to identify agreements and conflicts.
"""

import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class Perspective:
    """Represents a viewpoint or claim from a source."""
    source_id: int
    source_title: str
    claim: str
    context: str
    confidence: float
    topic_keywords: List[str]


@dataclass
class TopicAnalysis:
    """Analysis of a specific topic across multiple sources."""
    topic_name: str
    topic_keywords: List[str]
    perspectives: List[Perspective]
    consensus_level: float
    conflicting_views: List[Tuple[Perspective, Perspective]]
    majority_view: Optional[str]
    minority_views: List[str]
    confidence_score: float


@dataclass
class CrossValidationResult:
    """Result of cross-source validation analysis."""
    total_topics: int
    consensus_topics: int
    conflicted_topics: int
    topic_analyses: List[TopicAnalysis]
    overall_consensus_score: float
    source_agreement_matrix: Dict[Tuple[int, int], float]
    validation_summary: Dict[str, Any]


class CrossValidator:
    """Validates and compares perspectives across multiple sources."""
    
    def __init__(self, consensus_threshold: float = 0.7, similarity_threshold: float = 0.6):
        self.consensus_threshold = consensus_threshold
        self.similarity_threshold = similarity_threshold
        
        # Topic keywords for different domains
        self.topic_keywords = {
            'performance': ['performance', 'speed', 'efficiency', 'optimization', 'benchmark', 'latency'],
            'security': ['security', 'vulnerability', 'attack', 'protection', 'encryption', 'authentication'],
            'scalability': ['scalability', 'scale', 'growth', 'capacity', 'load', 'distributed'],
            'cost': ['cost', 'price', 'expensive', 'budget', 'economic', 'financial'],
            'usability': ['usability', 'user experience', 'ease of use', 'interface', 'intuitive'],
            'reliability': ['reliability', 'stable', 'robust', 'fault tolerance', 'availability'],
            'compatibility': ['compatibility', 'integration', 'support', 'compatible', 'interoperability'],
            'maintenance': ['maintenance', 'update', 'upgrade', 'support', 'lifecycle']
        }
        
        # Conflict indicators
        self.conflict_indicators = [
            ('however', 'but', 'nevertheless', 'on the other hand'),
            ('while', 'whereas', 'although', 'despite'),
            ('contrary', 'opposite', 'different', 'disagree'),
            ('better', 'worse', 'superior', 'inferior'),
            ('more', 'less', 'higher', 'lower')
        ]
    
    def cross_validate_sources(self, 
                              query: str, 
                              source_contents: List[Dict[str, Any]]) -> CrossValidationResult:
        """
        Perform cross-validation analysis across multiple sources.
        
        Args:
            query: Research query for context
            source_contents: List of source data with content
            
        Returns:
            CrossValidationResult with detailed analysis
        """
        logger.info(f"Starting cross-validation for {len(source_contents)} sources")
        
        # Extract perspectives from each source
        all_perspectives = []
        for i, source in enumerate(source_contents, 1):
            perspectives = self._extract_perspectives(i, source)
            all_perspectives.extend(perspectives)
        
        logger.info(f"Extracted {len(all_perspectives)} perspectives")
        
        # Group perspectives by topic
        topic_groups = self._group_perspectives_by_topic(all_perspectives, query)
        
        # Analyze each topic
        topic_analyses = []
        for topic_name, perspectives in topic_groups.items():
            analysis = self._analyze_topic_perspectives(topic_name, perspectives)
            topic_analyses.append(analysis)
        
        # Calculate source agreement matrix
        agreement_matrix = self._calculate_source_agreement_matrix(source_contents, topic_analyses)
        
        # Calculate overall metrics
        consensus_topics = sum(1 for analysis in topic_analyses 
                             if analysis.consensus_level >= self.consensus_threshold)
        overall_consensus = consensus_topics / len(topic_analyses) if topic_analyses else 0.0
        
        # Generate validation summary
        validation_summary = self._generate_validation_summary(topic_analyses, agreement_matrix)
        
        result = CrossValidationResult(
            total_topics=len(topic_analyses),
            consensus_topics=consensus_topics,
            conflicted_topics=len(topic_analyses) - consensus_topics,
            topic_analyses=topic_analyses,
            overall_consensus_score=round(overall_consensus, 3),
            source_agreement_matrix=agreement_matrix,
            validation_summary=validation_summary
        )
        
        logger.info(f"Cross-validation completed: {overall_consensus:.1%} consensus")
        return result
    
    def _extract_perspectives(self, source_id: int, source_data: Dict[str, Any]) -> List[Perspective]:
        """Extract individual perspectives/claims from a source."""
        perspectives = []
        
        title = source_data.get('title', '')
        abstract = source_data.get('abstract', source_data.get('summary', ''))
        content = source_data.get('content', '')
        
        # Combine all text
        full_text = f"{title}. {abstract}. {content}".strip()
        
        if not full_text:
            return perspectives
        
        # Extract claims/statements
        claims = self._extract_claims_from_text(full_text)
        
        for claim in claims:
            # Identify topic keywords for this claim
            topic_keywords = self._identify_topic_keywords(claim)
            
            if topic_keywords:  # Only include claims with identifiable topics
                perspective = Perspective(
                    source_id=source_id,
                    source_title=title,
                    claim=claim,
                    context=self._extract_claim_context(claim, full_text),
                    confidence=self._calculate_claim_confidence(claim, source_data),
                    topic_keywords=topic_keywords
                )
                perspectives.append(perspective)
        
        return perspectives
    
    def _extract_claims_from_text(self, text: str) -> List[str]:
        """Extract individual claims/statements from text."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter out short or non-substantive sentences
            if len(sentence) < 20:
                continue
            
            # Look for substantive claims (avoid questions, lists, etc.)
            if self._is_substantive_claim(sentence):
                claims.append(sentence)
        
        return claims
    
    def _is_substantive_claim(self, sentence: str) -> bool:
        """Check if a sentence represents a substantive claim."""
        # Exclude questions
        if sentence.strip().endswith('?'):
            return False
        
        # Exclude very short sentences
        if len(sentence.split()) < 5:
            return False
        
        # Look for claim indicators
        claim_indicators = [
            'shows', 'demonstrates', 'indicates', 'suggests', 'proves', 'reveals',
            'found', 'discovered', 'observed', 'reported', 'concluded',
            'provides', 'offers', 'enables', 'allows', 'improves', 'reduces',
            'is', 'are', 'can', 'will', 'may', 'should', 'must'
        ]
        
        sentence_lower = sentence.lower()
        has_claim_indicator = any(indicator in sentence_lower for indicator in claim_indicators)
        
        # Look for comparative statements
        has_comparison = any(comp in sentence_lower for comp in ['better', 'worse', 'more', 'less', 'vs', 'compared'])
        
        return has_claim_indicator or has_comparison
    
    def _identify_topic_keywords(self, claim: str) -> List[str]:
        """Identify topic keywords present in a claim."""
        claim_lower = claim.lower()
        found_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in claim_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _extract_claim_context(self, claim: str, full_text: str) -> str:
        """Extract context around a specific claim."""
        claim_start = full_text.find(claim)
        if claim_start == -1:
            return claim
        
        # Get surrounding context
        context_start = max(0, claim_start - 100)
        context_end = min(len(full_text), claim_start + len(claim) + 100)
        
        context = full_text[context_start:context_end]
        return context.strip()
    
    def _calculate_claim_confidence(self, claim: str, source_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a claim based on source quality."""
        confidence = 0.5  # Base confidence
        
        # Source type bonus
        source_type = source_data.get('source_type', '')
        if source_type == 'arxiv':
            confidence += 0.3  # Academic papers
        elif source_type == 'pubmed':
            confidence += 0.25  # Medical research
        elif source_type == 'journal':
            confidence += 0.2  # Journal articles
        
        # Author information
        if source_data.get('authors'):
            confidence += 0.1
        
        # Publication date (more recent = slightly higher confidence)
        year = source_data.get('year')
        if year:
            try:
                year_int = int(year)
                if year_int >= 2020:
                    confidence += 0.1
                elif year_int >= 2015:
                    confidence += 0.05
            except ValueError:
                pass
        
        # Claim specificity (more specific = higher confidence)
        if len(claim.split()) > 15:  # Detailed claims
            confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _group_perspectives_by_topic(self, 
                                   perspectives: List[Perspective], 
                                   query: str) -> Dict[str, List[Perspective]]:
        """Group perspectives by topic for comparison."""
        topic_groups = defaultdict(list)
        
        # Extract query topics for context
        query_topics = self._identify_topic_keywords(query)
        
        for perspective in perspectives:
            # Assign to primary topic
            primary_topic = self._determine_primary_topic(perspective, query_topics)
            topic_groups[primary_topic].append(perspective)
        
        # Only return topics with multiple perspectives for comparison
        filtered_groups = {topic: persp_list for topic, persp_list in topic_groups.items() 
                          if len(persp_list) > 1}
        
        return filtered_groups
    
    def _determine_primary_topic(self, perspective: Perspective, query_topics: List[str]) -> str:
        """Determine the primary topic for a perspective."""
        # Prefer query-related topics
        for topic in query_topics:
            if topic in perspective.topic_keywords:
                return topic
        
        # Otherwise use the first topic keyword
        if perspective.topic_keywords:
            return perspective.topic_keywords[0]
        
        return "general"
    
    def _analyze_topic_perspectives(self, topic_name: str, perspectives: List[Perspective]) -> TopicAnalysis:
        """Analyze perspectives on a specific topic."""
        if len(perspectives) < 2:
            return TopicAnalysis(
                topic_name=topic_name,
                topic_keywords=[topic_name],
                perspectives=perspectives,
                consensus_level=1.0,
                conflicting_views=[],
                majority_view=perspectives[0].claim if perspectives else None,
                minority_views=[],
                confidence_score=perspectives[0].confidence if perspectives else 0.0
            )
        
        # Identify conflicts
        conflicting_views = self._identify_conflicts(perspectives)
        
        # Calculate consensus level
        consensus_level = self._calculate_consensus_level(perspectives, conflicting_views)
        
        # Identify majority and minority views
        majority_view, minority_views = self._identify_majority_minority_views(perspectives)
        
        # Calculate confidence score
        confidence_score = sum(p.confidence for p in perspectives) / len(perspectives)
        
        # Extract topic keywords
        all_keywords = []
        for p in perspectives:
            all_keywords.extend(p.topic_keywords)
        topic_keywords = list(set(all_keywords))
        
        return TopicAnalysis(
            topic_name=topic_name,
            topic_keywords=topic_keywords,
            perspectives=perspectives,
            consensus_level=round(consensus_level, 3),
            conflicting_views=conflicting_views,
            majority_view=majority_view,
            minority_views=minority_views,
            confidence_score=round(confidence_score, 3)
        )
    
    def _identify_conflicts(self, perspectives: List[Perspective]) -> List[Tuple[Perspective, Perspective]]:
        """Identify conflicting perspectives."""
        conflicts = []
        
        for i in range(len(perspectives)):
            for j in range(i + 1, len(perspectives)):
                perspective1 = perspectives[i]
                perspective2 = perspectives[j]
                
                if self._are_perspectives_conflicting(perspective1, perspective2):
                    conflicts.append((perspective1, perspective2))
        
        return conflicts
    
    def _are_perspectives_conflicting(self, p1: Perspective, p2: Perspective) -> bool:
        """Check if two perspectives are conflicting."""
        # Simple approach: look for opposing terms
        claim1_lower = p1.claim.lower()
        claim2_lower = p2.claim.lower()
        
        # Check for explicit contradictions
        opposing_pairs = [
            ('better', 'worse'), ('superior', 'inferior'), ('advantage', 'disadvantage'),
            ('increase', 'decrease'), ('higher', 'lower'), ('more', 'less'),
            ('effective', 'ineffective'), ('secure', 'insecure'), ('fast', 'slow'),
            ('yes', 'no'), ('true', 'false'), ('supports', 'contradicts')
        ]
        
        for term1, term2 in opposing_pairs:
            if (term1 in claim1_lower and term2 in claim2_lower) or \
               (term2 in claim1_lower and term1 in claim2_lower):
                return True
        
        # Check for negation patterns
        if ('not' in claim1_lower and 'not' not in claim2_lower) or \
           ('not' in claim2_lower and 'not' not in claim1_lower):
            # Look for common terms (excluding 'not')
            words1 = set(claim1_lower.replace('not', '').split())
            words2 = set(claim2_lower.replace('not', '').split())
            common_words = words1.intersection(words2)
            if len(common_words) >= 3:  # Significant overlap suggests contradiction
                return True
        
        return False
    
    def _calculate_consensus_level(self, 
                                 perspectives: List[Perspective], 
                                 conflicts: List[Tuple[Perspective, Perspective]]) -> float:
        """Calculate level of consensus among perspectives."""
        if len(perspectives) <= 1:
            return 1.0
        
        # Count non-conflicting pairs
        total_pairs = len(perspectives) * (len(perspectives) - 1) // 2
        conflicting_pairs = len(conflicts)
        non_conflicting_pairs = total_pairs - conflicting_pairs
        
        consensus_level = non_conflicting_pairs / total_pairs if total_pairs > 0 else 1.0
        return consensus_level
    
    def _identify_majority_minority_views(self, perspectives: List[Perspective]) -> Tuple[Optional[str], List[str]]:
        """Identify majority and minority viewpoints."""
        if len(perspectives) <= 2:
            return None, []
        
        # Group similar perspectives (simplified approach)
        view_groups = defaultdict(list)
        
        for perspective in perspectives:
            # Use first few words as a simple grouping key
            key_words = ' '.join(perspective.claim.split()[:5]).lower()
            view_groups[key_words].append(perspective)
        
        # Find majority view
        if view_groups:
            majority_key = max(view_groups.keys(), key=lambda k: len(view_groups[k]))
            majority_view = view_groups[majority_key][0].claim
            
            # Identify minority views
            minority_views = []
            for key, group in view_groups.items():
                if key != majority_key:
                    minority_views.append(group[0].claim)
            
            return majority_view, minority_views
        
        return None, []
    
    def _calculate_source_agreement_matrix(self, 
                                          source_contents: List[Dict[str, Any]], 
                                          topic_analyses: List[TopicAnalysis]) -> Dict[Tuple[int, int], float]:
        """Calculate agreement scores between source pairs."""
        agreement_matrix = {}
        
        for i in range(1, len(source_contents) + 1):
            for j in range(i + 1, len(source_contents) + 1):
                agreement_score = self._calculate_pairwise_agreement(i, j, topic_analyses)
                agreement_matrix[(i, j)] = round(agreement_score, 3)
        
        return agreement_matrix
    
    def _calculate_pairwise_agreement(self, 
                                    source_id1: int, 
                                    source_id2: int, 
                                    topic_analyses: List[TopicAnalysis]) -> float:
        """Calculate agreement score between two sources."""
        agreements = 0
        disagreements = 0
        
        for analysis in topic_analyses:
            perspectives1 = [p for p in analysis.perspectives if p.source_id == source_id1]
            perspectives2 = [p for p in analysis.perspectives if p.source_id == source_id2]
            
            if perspectives1 and perspectives2:
                # Check if sources agree or disagree on this topic
                for p1 in perspectives1:
                    for p2 in perspectives2:
                        if self._are_perspectives_conflicting(p1, p2):
                            disagreements += 1
                        else:
                            agreements += 1
        
        total_comparisons = agreements + disagreements
        if total_comparisons == 0:
            return 0.5  # Neutral when no common topics
        
        return agreements / total_comparisons
    
    def _generate_validation_summary(self, 
                                   topic_analyses: List[TopicAnalysis], 
                                   agreement_matrix: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        """Generate summary statistics for validation."""
        if not topic_analyses:
            return {}
        
        consensus_scores = [analysis.consensus_level for analysis in topic_analyses]
        confidence_scores = [analysis.confidence_score for analysis in topic_analyses]
        
        summary = {
            'average_consensus': round(sum(consensus_scores) / len(consensus_scores), 3),
            'average_confidence': round(sum(confidence_scores) / len(confidence_scores), 3),
            'topics_with_conflicts': sum(1 for analysis in topic_analyses if analysis.conflicting_views),
            'total_conflicts': sum(len(analysis.conflicting_views) for analysis in topic_analyses),
            'most_consensual_topic': max(topic_analyses, key=lambda x: x.consensus_level).topic_name,
            'most_conflicted_topic': min(topic_analyses, key=lambda x: x.consensus_level).topic_name,
        }
        
        if agreement_matrix:
            agreement_scores = list(agreement_matrix.values())
            summary['average_source_agreement'] = round(sum(agreement_scores) / len(agreement_scores), 3)
            summary['highest_source_agreement'] = max(agreement_scores)
            summary['lowest_source_agreement'] = min(agreement_scores)
        
        return summary
    
    def generate_cross_validation_report(self, result: CrossValidationResult) -> str:
        """Generate human-readable cross-validation report."""
        report_lines = [
            "Cross-Source Validation Report",
            "=" * 40,
            f"Overall Consensus Score: {result.overall_consensus_score:.1%}",
            f"Total Topics Analyzed: {result.total_topics}",
            f"Consensus Topics: {result.consensus_topics}",
            f"Conflicted Topics: {result.conflicted_topics}",
            ""
        ]
        
        # Summary statistics
        if result.validation_summary:
            summary = result.validation_summary
            report_lines.extend([
                "Summary Statistics:",
                f"  Average Consensus: {summary.get('average_consensus', 0):.3f}",
                f"  Average Confidence: {summary.get('average_confidence', 0):.3f}",
                f"  Total Conflicts: {summary.get('total_conflicts', 0)}",
            ])
            
            if 'average_source_agreement' in summary:
                report_lines.append(f"  Average Source Agreement: {summary['average_source_agreement']:.3f}")
            
            report_lines.append("")
        
        # Most conflicted topics
        conflicted_topics = [analysis for analysis in result.topic_analyses 
                           if analysis.conflicting_views]
        if conflicted_topics:
            report_lines.append("Most Conflicted Topics:")
            for analysis in sorted(conflicted_topics, key=lambda x: len(x.conflicting_views), reverse=True)[:3]:
                report_lines.append(f"  • {analysis.topic_name}: {len(analysis.conflicting_views)} conflicts")
                for p1, p2 in analysis.conflicting_views[:2]:  # Show first 2 conflicts
                    report_lines.append(f"    - Source {p1.source_id}: {p1.claim[:60]}...")
                    report_lines.append(f"    - Source {p2.source_id}: {p2.claim[:60]}...")
            report_lines.append("")
        
        # High consensus topics
        consensus_topics = [analysis for analysis in result.topic_analyses 
                          if analysis.consensus_level >= self.consensus_threshold]
        if consensus_topics:
            report_lines.append("High Consensus Topics:")
            for analysis in sorted(consensus_topics, key=lambda x: x.consensus_level, reverse=True)[:3]:
                report_lines.append(f"  • {analysis.topic_name}: {analysis.consensus_level:.1%} consensus")
                if analysis.majority_view:
                    report_lines.append(f"    Majority view: {analysis.majority_view[:80]}...")
            report_lines.append("")
        
        # Source agreement matrix
        if result.source_agreement_matrix:
            report_lines.append("Source Agreement Matrix:")
            for (source1, source2), agreement in sorted(result.source_agreement_matrix.items()):
                report_lines.append(f"  Source {source1} ↔ Source {source2}: {agreement:.3f}")
        
        return "\n".join(report_lines)


# Example usage and testing
if __name__ == "__main__":
    # Initialize cross validator
    validator = CrossValidator(consensus_threshold=0.7)
    
    # Sample query
    query = "Docker performance compared to virtual machines"
    
    # Sample sources with different perspectives
    sources = [
        {
            'title': 'Docker Performance Benefits Study',
            'content': 'Docker containers demonstrate superior performance compared to virtual machines. Containers show 30% better resource utilization and significantly faster startup times. The lightweight nature of containers provides clear advantages.',
            'source_type': 'arxiv',
            'year': '2023'
        },
        {
            'title': 'Virtual Machine vs Container Analysis',
            'content': 'While containers offer some performance benefits, virtual machines provide better isolation and security. VMs are more suitable for production environments where security is paramount. However, containers do start faster.',
            'source_type': 'journal',
            'year': '2023'
        },
        {
            'title': 'Container Security Concerns',
            'content': 'Containers pose significant security risks compared to virtual machines. The shared kernel architecture introduces vulnerabilities. Virtual machines offer superior isolation and should be preferred for sensitive workloads.',
            'source_type': 'conference',
            'year': '2024'
        }
    ]
    
    print("🔄 Testing Cross Validator")
    print("=" * 50)
    
    # Perform cross-validation
    result = validator.cross_validate_sources(query, sources)
    
    # Generate report
    report = validator.generate_cross_validation_report(result)
    print(report)
    
    print(f"\n📊 Detailed Analysis:")
    for analysis in result.topic_analyses:
        print(f"\nTopic: {analysis.topic_name}")
        print(f"Consensus Level: {analysis.consensus_level:.3f}")
        print(f"Perspectives: {len(analysis.perspectives)}")
        print(f"Conflicts: {len(analysis.conflicting_views)}")
        if analysis.majority_view:
            print(f"Majority View: {analysis.majority_view[:100]}...")
        if analysis.minority_views:
            print(f"Minority Views: {len(analysis.minority_views)}")
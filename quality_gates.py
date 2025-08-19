"""
Quality Gates for Research Agent
Comprehensive validation system that orchestrates all quality checks.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Import validation components
try:
    from relevance_validator import RelevanceValidator, RelevanceScore
    from content_linker import ContentLinker, ContentValidationResult
    from cross_validator import CrossValidator, CrossValidationResult
except ImportError as e:
    logging.warning(f"Failed to import validation components: {e}")

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    details: Dict[str, Any]


@dataclass
class QualityAssessment:
    """Complete quality assessment result."""
    overall_quality: QualityLevel
    overall_score: float
    gate_results: List[QualityGateResult]
    validation_summary: Dict[str, Any]
    critical_issues: List[str]
    recommendations: List[str]
    source_quality_report: Dict[str, Any]
    content_quality_report: Dict[str, Any]


class QualityGates:
    """Orchestrates comprehensive quality validation for research process."""
    
    def __init__(self, 
                 relevance_threshold: float = 0.6,
                 content_validation_threshold: float = 0.8,
                 consensus_threshold: float = 0.7):
        
        # Initialize validation components
        self.relevance_validator = RelevanceValidator(relevance_threshold=relevance_threshold)
        self.content_linker = ContentLinker(similarity_threshold=0.6)
        self.cross_validator = CrossValidator(consensus_threshold=consensus_threshold)
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.8,
            QualityLevel.ACCEPTABLE: 0.6,
            QualityLevel.POOR: 0.4,
            QualityLevel.FAILED: 0.0
        }
        
        # Gate configurations
        self.gate_configs = {
            'source_relevance': {'weight': 0.25, 'critical': True},
            'source_coverage': {'weight': 0.15, 'critical': False},
            'content_validation': {'weight': 0.30, 'critical': True},
            'cross_validation': {'weight': 0.20, 'critical': False},
            'research_completeness': {'weight': 0.10, 'critical': False}
        }
    
    def run_pre_synthesis_gates(self, 
                                query: str, 
                                sources: List[Dict[str, Any]]) -> Tuple[bool, List[QualityGateResult]]:
        """
        Run pre-synthesis quality gates.
        
        Args:
            query: Research query
            sources: List of source data
            
        Returns:
            Tuple of (passed, gate_results)
        """
        logger.info("Running pre-synthesis quality gates")
        
        gate_results = []
        
        # Gate 1: Source Relevance Validation
        relevance_result = self._gate_source_relevance(query, sources)
        gate_results.append(relevance_result)
        
        # Gate 2: Source Coverage Assessment
        coverage_result = self._gate_source_coverage(query, sources)
        gate_results.append(coverage_result)
        
        # Gate 3: Source Quality Assessment
        quality_result = self._gate_source_quality(sources)
        gate_results.append(quality_result)
        
        # Check if critical gates passed
        critical_gates_passed = all(
            result.passed for result in gate_results 
            if self.gate_configs.get(result.gate_name, {}).get('critical', False)
        )
        
        overall_passed = critical_gates_passed and all(result.passed for result in gate_results)
        
        logger.info(f"Pre-synthesis gates: {'PASSED' if overall_passed else 'FAILED'}")
        return overall_passed, gate_results
    
    def run_post_synthesis_gates(self, 
                                 query: str,
                                 sources: List[Dict[str, Any]], 
                                 synthesis: str) -> Tuple[bool, List[QualityGateResult]]:
        """
        Run post-synthesis quality gates.
        
        Args:
            query: Research query
            sources: List of source data
            synthesis: Generated synthesis content
            
        Returns:
            Tuple of (passed, gate_results)
        """
        logger.info("Running post-synthesis quality gates")
        
        gate_results = []
        
        # Gate 1: Content-Citation Validation
        content_result = self._gate_content_validation(synthesis, sources)
        gate_results.append(content_result)
        
        # Gate 2: Cross-Source Validation
        cross_result = self._gate_cross_validation(query, sources)
        gate_results.append(cross_result)
        
        # Gate 3: Research Completeness
        completeness_result = self._gate_research_completeness(query, synthesis, sources)
        gate_results.append(completeness_result)
        
        # Check if critical gates passed
        critical_gates_passed = all(
            result.passed for result in gate_results 
            if self.gate_configs.get(result.gate_name, {}).get('critical', False)
        )
        
        overall_passed = critical_gates_passed
        
        logger.info(f"Post-synthesis gates: {'PASSED' if overall_passed else 'FAILED'}")
        return overall_passed, gate_results
    
    def assess_overall_quality(self, 
                              query: str,
                              sources: List[Dict[str, Any]], 
                              synthesis: str) -> QualityAssessment:
        """
        Perform comprehensive quality assessment.
        
        Args:
            query: Research query
            sources: List of source data
            synthesis: Generated synthesis content
            
        Returns:
            QualityAssessment with detailed analysis
        """
        logger.info("Starting comprehensive quality assessment")
        
        # Run all quality gates
        pre_passed, pre_results = self.run_pre_synthesis_gates(query, sources)
        post_passed, post_results = self.run_post_synthesis_gates(query, sources, synthesis)
        
        all_gate_results = pre_results + post_results
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(all_gate_results)
        
        # Determine quality level
        overall_quality = self._determine_quality_level(overall_score)
        
        # Generate validation summary
        validation_summary = self._generate_validation_summary(all_gate_results)
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(all_gate_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_gate_results)
        
        # Generate detailed reports
        source_quality_report = self._generate_source_quality_report(sources, pre_results)
        content_quality_report = self._generate_content_quality_report(synthesis, post_results)
        
        assessment = QualityAssessment(
            overall_quality=overall_quality,
            overall_score=round(overall_score, 3),
            gate_results=all_gate_results,
            validation_summary=validation_summary,
            critical_issues=critical_issues,
            recommendations=recommendations,
            source_quality_report=source_quality_report,
            content_quality_report=content_quality_report
        )
        
        logger.info(f"Quality assessment completed: {overall_quality.value} ({overall_score:.3f})")
        return assessment
    
    def _gate_source_relevance(self, query: str, sources: List[Dict[str, Any]]) -> QualityGateResult:
        """Validate source relevance using RelevanceValidator."""
        try:
            validated_sources = self.relevance_validator.batch_validate_sources(query, sources)
            stats = self.relevance_validator.get_validation_stats(query, sources)
            
            # Calculate score based on validation results
            pass_rate = 1.0 - stats['rejection_rate']
            avg_score = stats['average_score']
            
            # Combined score (70% pass rate, 30% average score)
            score = 0.7 * pass_rate + 0.3 * avg_score
            
            # Determine if gate passed
            passed = score >= 0.6 and stats['passed_sources'] > 0
            
            issues = []
            recommendations = []
            
            if stats['rejection_rate'] > 0.5:
                issues.append(f"High source rejection rate: {stats['rejection_rate']:.1%}")
                recommendations.append("Refine search terms to find more relevant sources")
            
            if stats['passed_sources'] < 3:
                issues.append(f"Insufficient sources: only {stats['passed_sources']} passed validation")
                recommendations.append("Expand search to find more relevant sources")
            
            return QualityGateResult(
                gate_name="source_relevance",
                passed=passed,
                score=round(score, 3),
                issues=issues,
                recommendations=recommendations,
                details=stats
            )
            
        except Exception as e:
            logger.error(f"Source relevance gate failed: {e}")
            return QualityGateResult(
                gate_name="source_relevance",
                passed=False,
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Check source data format and validation configuration"],
                details={}
            )
    
    def _gate_source_coverage(self, query: str, sources: List[Dict[str, Any]]) -> QualityGateResult:
        """Assess source coverage and diversity."""
        try:
            # Analyze source diversity
            source_types = set(source.get('source_type', 'unknown') for source in sources)
            publication_years = [source.get('year', '2020') for source in sources if source.get('year')]
            
            # Calculate coverage metrics
            type_diversity = len(source_types) / max(len(sources), 1)
            
            # Year spread (prefer recent but some variety)
            if publication_years:
                try:
                    years = [int(year) for year in publication_years if year.isdigit()]
                    if years:
                        year_range = max(years) - min(years)
                        year_diversity = min(year_range / 10.0, 1.0)  # Normalize to 0-1
                    else:
                        year_diversity = 0.5
                except:
                    year_diversity = 0.5
            else:
                year_diversity = 0.5
            
            # Source count adequacy
            count_score = min(len(sources) / 5.0, 1.0)  # Ideal: 5+ sources
            
            # Combined coverage score
            score = 0.4 * type_diversity + 0.3 * year_diversity + 0.3 * count_score
            
            passed = score >= 0.6
            
            issues = []
            recommendations = []
            
            if len(source_types) <= 1:
                issues.append("Limited source type diversity")
                recommendations.append("Include sources from different types (academic, industry, blogs)")
            
            if len(sources) < 3:
                issues.append("Insufficient number of sources")
                recommendations.append("Increase source count to at least 3-5 for better coverage")
            
            return QualityGateResult(
                gate_name="source_coverage",
                passed=passed,
                score=round(score, 3),
                issues=issues,
                recommendations=recommendations,
                details={
                    'source_types': list(source_types),
                    'source_count': len(sources),
                    'type_diversity': round(type_diversity, 3),
                    'year_diversity': round(year_diversity, 3)
                }
            )
            
        except Exception as e:
            logger.error(f"Source coverage gate failed: {e}")
            return QualityGateResult(
                gate_name="source_coverage",
                passed=False,
                score=0.0,
                issues=[f"Coverage assessment error: {str(e)}"],
                recommendations=["Check source metadata format"],
                details={}
            )
    
    def _gate_source_quality(self, sources: List[Dict[str, Any]]) -> QualityGateResult:
        """Assess overall source quality."""
        try:
            quality_scores = []
            
            for source in sources:
                source_score = 0.5  # Base score
                
                # Author information
                if source.get('authors'):
                    source_score += 0.1
                
                # Publication information
                if source.get('journal') or source.get('conference'):
                    source_score += 0.1
                
                # Content availability
                if source.get('abstract') or source.get('content'):
                    source_score += 0.1
                
                # DOI or official URL
                if source.get('doi') or (source.get('url', '').startswith('https://doi.org')):
                    source_score += 0.1
                
                # Source type quality
                source_type = source.get('source_type', '')
                if source_type in ['arxiv', 'pubmed', 'journal']:
                    source_score += 0.1
                
                quality_scores.append(min(source_score, 1.0))
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            passed = avg_quality >= 0.6
            
            issues = []
            recommendations = []
            
            low_quality_sources = sum(1 for score in quality_scores if score < 0.6)
            if low_quality_sources > 0:
                issues.append(f"{low_quality_sources} sources have low quality metadata")
                recommendations.append("Prioritize sources with complete metadata and authoritative publishers")
            
            return QualityGateResult(
                gate_name="source_quality",
                passed=passed,
                score=round(avg_quality, 3),
                issues=issues,
                recommendations=recommendations,
                details={
                    'average_quality': round(avg_quality, 3),
                    'quality_scores': [round(score, 3) for score in quality_scores],
                    'low_quality_count': low_quality_sources
                }
            )
            
        except Exception as e:
            logger.error(f"Source quality gate failed: {e}")
            return QualityGateResult(
                gate_name="source_quality",
                passed=False,
                score=0.0,
                issues=[f"Quality assessment error: {str(e)}"],
                recommendations=["Check source data structure"],
                details={}
            )
    
    def _gate_content_validation(self, synthesis: str, sources: List[Dict[str, Any]]) -> QualityGateResult:
        """Validate content-citation alignment using ContentLinker."""
        try:
            validation_result = self.content_linker.validate_content_citations(synthesis, sources)
            
            score = validation_result.validation_score
            passed = score >= 0.8  # High threshold for content validation
            
            issues = validation_result.overall_issues.copy()
            recommendations = []
            
            if validation_result.unsupported_claims > 0:
                recommendations.append("Remove or find evidence for unsupported claims")
            
            if len(validation_result.source_utilization) < len(sources) * 0.7:
                recommendations.append("Utilize more sources in synthesis")
            
            return QualityGateResult(
                gate_name="content_validation",
                passed=passed,
                score=round(score, 3),
                issues=issues,
                recommendations=recommendations,
                details={
                    'supported_claims': validation_result.supported_claims,
                    'unsupported_claims': validation_result.unsupported_claims,
                    'source_utilization': validation_result.source_utilization
                }
            )
            
        except Exception as e:
            logger.error(f"Content validation gate failed: {e}")
            return QualityGateResult(
                gate_name="content_validation",
                passed=False,
                score=0.0,
                issues=[f"Content validation error: {str(e)}"],
                recommendations=["Check synthesis content and source data format"],
                details={}
            )
    
    def _gate_cross_validation(self, query: str, sources: List[Dict[str, Any]]) -> QualityGateResult:
        """Validate cross-source perspectives using CrossValidator."""
        try:
            if len(sources) < 2:
                return QualityGateResult(
                    gate_name="cross_validation",
                    passed=True,
                    score=0.7,  # Neutral score for single source
                    issues=["Insufficient sources for cross-validation"],
                    recommendations=["Add more sources for perspective comparison"],
                    details={'source_count': len(sources)}
                )
            
            cross_result = self.cross_validator.cross_validate_sources(query, sources)
            
            score = cross_result.overall_consensus_score
            passed = score >= 0.5  # Moderate threshold for consensus
            
            issues = []
            recommendations = []
            
            if cross_result.conflicted_topics > cross_result.consensus_topics:
                issues.append("More conflicted topics than consensus topics")
                recommendations.append("Address conflicting perspectives in synthesis")
            
            if cross_result.validation_summary.get('total_conflicts', 0) > 5:
                issues.append("High number of conflicts between sources")
                recommendations.append("Carefully analyze and reconcile conflicting viewpoints")
            
            return QualityGateResult(
                gate_name="cross_validation",
                passed=passed,
                score=round(score, 3),
                issues=issues,
                recommendations=recommendations,
                details={
                    'consensus_topics': cross_result.consensus_topics,
                    'conflicted_topics': cross_result.conflicted_topics,
                    'validation_summary': cross_result.validation_summary
                }
            )
            
        except Exception as e:
            logger.error(f"Cross validation gate failed: {e}")
            return QualityGateResult(
                gate_name="cross_validation",
                passed=False,
                score=0.0,
                issues=[f"Cross validation error: {str(e)}"],
                recommendations=["Check source content format"],
                details={}
            )
    
    def _gate_research_completeness(self, query: str, synthesis: str, sources: List[Dict[str, Any]]) -> QualityGateResult:
        """Assess research completeness and gap identification."""
        try:
            # Analyze query coverage
            query_terms = query.lower().split()
            synthesis_lower = synthesis.lower()
            
            # Check if key query terms are addressed
            addressed_terms = sum(1 for term in query_terms if term in synthesis_lower)
            coverage_score = addressed_terms / len(query_terms) if query_terms else 0.0
            
            # Check synthesis length and structure
            synthesis_length = len(synthesis.split())
            length_adequacy = min(synthesis_length / 200.0, 1.0)  # Target: 200+ words
            
            # Check for research gaps identification
            gap_indicators = ['gap', 'limitation', 'future research', 'further study', 'unexplored']
            has_gaps = any(indicator in synthesis_lower for indicator in gap_indicators)
            gap_score = 1.0 if has_gaps else 0.5
            
            # Combined completeness score
            score = 0.5 * coverage_score + 0.3 * length_adequacy + 0.2 * gap_score
            
            passed = score >= 0.6
            
            issues = []
            recommendations = []
            
            if coverage_score < 0.7:
                issues.append("Synthesis doesn't address all aspects of the query")
                recommendations.append("Ensure synthesis covers all key terms from the research query")
            
            if synthesis_length < 150:
                issues.append("Synthesis is too brief")
                recommendations.append("Expand synthesis with more detailed analysis")
            
            if not has_gaps:
                recommendations.append("Consider identifying research gaps or limitations")
            
            return QualityGateResult(
                gate_name="research_completeness",
                passed=passed,
                score=round(score, 3),
                issues=issues,
                recommendations=recommendations,
                details={
                    'query_coverage': round(coverage_score, 3),
                    'synthesis_length': synthesis_length,
                    'addressed_terms': addressed_terms,
                    'total_terms': len(query_terms),
                    'has_gaps': has_gaps
                }
            )
            
        except Exception as e:
            logger.error(f"Research completeness gate failed: {e}")
            return QualityGateResult(
                gate_name="research_completeness",
                passed=False,
                score=0.0,
                issues=[f"Completeness assessment error: {str(e)}"],
                recommendations=["Check synthesis and query format"],
                details={}
            )
    
    def _calculate_overall_score(self, gate_results: List[QualityGateResult]) -> float:
        """Calculate weighted overall quality score."""
        total_score = 0.0
        total_weight = 0.0
        
        for result in gate_results:
            weight = self.gate_configs.get(result.gate_name, {}).get('weight', 0.1)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return QualityLevel.FAILED
    
    def _generate_validation_summary(self, gate_results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate summary of all validation results."""
        passed_gates = sum(1 for result in gate_results if result.passed)
        critical_gates = [result for result in gate_results 
                         if self.gate_configs.get(result.gate_name, {}).get('critical', False)]
        critical_passed = sum(1 for result in critical_gates if result.passed)
        
        return {
            'total_gates': len(gate_results),
            'passed_gates': passed_gates,
            'critical_gates': len(critical_gates),
            'critical_passed': critical_passed,
            'pass_rate': passed_gates / len(gate_results) if gate_results else 0.0,
            'critical_pass_rate': critical_passed / len(critical_gates) if critical_gates else 1.0,
            'gate_scores': {result.gate_name: result.score for result in gate_results}
        }
    
    def _identify_critical_issues(self, gate_results: List[QualityGateResult]) -> List[str]:
        """Identify critical issues that must be addressed."""
        critical_issues = []
        
        for result in gate_results:
            if not result.passed and self.gate_configs.get(result.gate_name, {}).get('critical', False):
                critical_issues.extend(result.issues)
        
        return critical_issues
    
    def _generate_recommendations(self, gate_results: List[QualityGateResult]) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Critical issues first
        for result in gate_results:
            if not result.passed and self.gate_configs.get(result.gate_name, {}).get('critical', False):
                recommendations.extend(result.recommendations)
        
        # Then other recommendations
        for result in gate_results:
            if not self.gate_configs.get(result.gate_name, {}).get('critical', False):
                recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _generate_source_quality_report(self, sources: List[Dict[str, Any]], pre_results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate detailed source quality report."""
        return {
            'total_sources': len(sources),
            'source_types': list(set(source.get('source_type', 'unknown') for source in sources)),
            'gate_results': {result.gate_name: result.details for result in pre_results}
        }
    
    def _generate_content_quality_report(self, synthesis: str, post_results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate detailed content quality report."""
        return {
            'synthesis_length': len(synthesis.split()),
            'synthesis_chars': len(synthesis),
            'gate_results': {result.gate_name: result.details for result in post_results}
        }
    
    def generate_quality_report(self, assessment: QualityAssessment) -> str:
        """Generate human-readable quality report."""
        report_lines = [
            "Research Quality Assessment Report",
            "=" * 50,
            f"Overall Quality: {assessment.overall_quality.value.upper()}",
            f"Overall Score: {assessment.overall_score:.3f}",
            ""
        ]
        
        # Validation summary
        summary = assessment.validation_summary
        report_lines.extend([
            "Validation Summary:",
            f"  Gates Passed: {summary['passed_gates']}/{summary['total_gates']}",
            f"  Critical Gates: {summary['critical_passed']}/{summary['critical_gates']}",
            f"  Pass Rate: {summary['pass_rate']:.1%}",
            ""
        ])
        
        # Critical issues
        if assessment.critical_issues:
            report_lines.append("Critical Issues:")
            for issue in assessment.critical_issues:
                report_lines.append(f"  ❌ {issue}")
            report_lines.append("")
        
        # Gate results
        report_lines.append("Quality Gate Results:")
        for result in assessment.gate_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            critical = " (CRITICAL)" if self.gate_configs.get(result.gate_name, {}).get('critical', False) else ""
            report_lines.append(f"  {result.gate_name}: {status} ({result.score:.3f}){critical}")
            
            if result.issues:
                for issue in result.issues:
                    report_lines.append(f"    - {issue}")
        
        report_lines.append("")
        
        # Recommendations
        if assessment.recommendations:
            report_lines.append("Recommendations:")
            for i, rec in enumerate(assessment.recommendations, 1):
                report_lines.append(f"  {i}. {rec}")
        
        return "\n".join(report_lines)


# Example usage and testing
if __name__ == "__main__":
    # Initialize quality gates
    gates = QualityGates()
    
    # Sample data
    query = "Docker containerization performance and security"
    synthesis = """
    Docker containers provide significant performance advantages over traditional virtual machines [Source 1]. 
    Research shows that containers achieve 30% better resource utilization and faster startup times. 
    However, security considerations must be carefully addressed when deploying containers in production [Source 2].
    Container security involves proper image scanning, runtime protection, and network isolation.
    Future research should focus on enhanced security frameworks for containerized environments.
    """
    
    sources = [
        {
            'title': 'Docker Performance Analysis',
            'content': 'Docker containers achieve 30% better resource utilization compared to VMs. Startup times are significantly faster.',
            'authors': ['John Doe'],
            'source_type': 'arxiv',
            'year': '2023'
        },
        {
            'title': 'Container Security Best Practices',
            'content': 'Container security requires careful attention to image vulnerabilities, runtime protection, and network policies.',
            'authors': ['Jane Smith'],
            'source_type': 'journal',
            'year': '2024'
        }
    ]
    
    print("🛡️ Testing Quality Gates")
    print("=" * 50)
    
    # Run comprehensive assessment
    assessment = gates.assess_overall_quality(query, sources, synthesis)
    
    # Generate report
    report = gates.generate_quality_report(assessment)
    print(report)
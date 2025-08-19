"""
Content Linker for Research Agent
Validates content-citation mapping to prevent source-content mismatch.
"""

import re
import difflib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """Represents a piece of evidence from a source."""
    source_id: int
    source_title: str
    evidence_text: str
    confidence: float
    location: str  # Where in source this was found (title, abstract, content)
    context: str   # Surrounding context


@dataclass
class ClaimValidation:
    """Validation result for a specific claim."""
    claim_text: str
    is_supported: bool
    evidence_items: List[EvidenceItem]
    confidence_score: float
    validation_method: str
    issues: List[str]


@dataclass
class ContentValidationResult:
    """Complete validation result for synthesized content."""
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    validation_score: float
    claim_validations: List[ClaimValidation]
    overall_issues: List[str]
    source_utilization: Dict[int, float]  # How much each source was used


class ContentLinker:
    """Links synthesized content to source evidence and validates citations."""
    
    def __init__(self, similarity_threshold: float = 0.6, min_evidence_length: int = 20):
        self.similarity_threshold = similarity_threshold
        self.min_evidence_length = min_evidence_length
        
        # Patterns for extracting claims
        self.claim_patterns = [
            r'[A-Z][^.!?]*[.!?]',  # Basic sentences
            r'(?:According to|Research shows|Studies indicate|Evidence suggests)[^.!?]*[.!?]',  # Research claims
            r'(?:However|Moreover|Furthermore|Additionally|In contrast)[^.!?]*[.!?]',  # Connecting claims
        ]
        
        # Citation patterns
        self.citation_patterns = [
            r'\[Source\s+(\d+)\]',  # [Source 1]
            r'\(Source\s+(\d+)\)',  # (Source 1)
            r'\[(\d+)\]',           # [1]
            r'\((\d+)\)',           # (1)
        ]
    
    def validate_content_citations(self, 
                                   synthesis_content: str, 
                                   source_contents: List[Dict[str, Any]]) -> ContentValidationResult:
        """
        Validate that synthesized content is properly supported by sources.
        
        Args:
            synthesis_content: The synthesized research content
            source_contents: List of source data with content
            
        Returns:
            ContentValidationResult with detailed validation analysis
        """
        logger.info("Starting content-citation validation")
        
        # Extract claims from synthesis
        claims = self._extract_claims(synthesis_content)
        logger.info(f"Extracted {len(claims)} claims for validation")
        
        # Create source evidence database
        source_evidence = self._build_evidence_database(source_contents)
        
        # Validate each claim
        claim_validations = []
        source_utilization = defaultdict(float)
        
        for claim in claims:
            validation = self._validate_claim(claim, source_evidence, synthesis_content)
            claim_validations.append(validation)
            
            # Track source utilization
            for evidence in validation.evidence_items:
                source_utilization[evidence.source_id] += evidence.confidence
        
        # Calculate overall metrics
        supported_claims = sum(1 for v in claim_validations if v.is_supported)
        validation_score = supported_claims / len(claims) if claims else 0.0
        
        # Identify overall issues
        overall_issues = self._identify_overall_issues(claim_validations, source_utilization, len(source_contents))
        
        # Normalize source utilization
        for source_id in source_utilization:
            source_utilization[source_id] = min(source_utilization[source_id], 1.0)
        
        result = ContentValidationResult(
            total_claims=len(claims),
            supported_claims=supported_claims,
            unsupported_claims=len(claims) - supported_claims,
            validation_score=round(validation_score, 3),
            claim_validations=claim_validations,
            overall_issues=overall_issues,
            source_utilization=dict(source_utilization)
        )
        
        logger.info(f"Content validation completed: {validation_score:.1%} claims supported")
        return result
    
    def _extract_claims(self, content: str) -> List[str]:
        """Extract individual claims from synthesized content."""
        claims = []
        
        # Remove citations for claim extraction
        clean_content = self._remove_citations(content)
        
        # Split into sentences using multiple patterns
        sentences = []
        
        # Basic sentence splitting
        basic_sentences = re.split(r'[.!?]+', clean_content)
        for sentence in basic_sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length filter
                sentences.append(sentence)
        
        # Further split complex sentences
        for sentence in sentences:
            # Split on common conjunctions that might separate claims
            sub_claims = re.split(r'\s+(?:however|moreover|furthermore|additionally|in contrast|nevertheless)\s+', 
                                sentence, flags=re.IGNORECASE)
            
            for sub_claim in sub_claims:
                sub_claim = sub_claim.strip()
                if len(sub_claim) > 15:  # Minimum meaningful claim length
                    claims.append(sub_claim)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            claim_normalized = claim.lower().strip()
            if claim_normalized not in seen:
                seen.add(claim_normalized)
                unique_claims.append(claim)
        
        return unique_claims
    
    def _remove_citations(self, text: str) -> str:
        """Remove citation markers from text."""
        clean_text = text
        for pattern in self.citation_patterns:
            clean_text = re.sub(pattern, '', clean_text)
        return clean_text
    
    def _build_evidence_database(self, source_contents: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Build searchable evidence database from sources."""
        evidence_db = {}
        
        for i, source in enumerate(source_contents, 1):
            # Extract all available text
            title = source.get('title', '')
            abstract = source.get('abstract', source.get('summary', ''))
            content = source.get('content', '')
            
            # Break into searchable segments
            segments = []
            
            if title:
                segments.append(('title', title))
            
            if abstract:
                # Split abstract into sentences
                abstract_sentences = re.split(r'[.!?]+', abstract)
                for sentence in abstract_sentences:
                    sentence = sentence.strip()
                    if len(sentence) >= self.min_evidence_length:
                        segments.append(('abstract', sentence))
            
            if content:
                # Split content into meaningful chunks
                content_chunks = self._split_content_into_chunks(content)
                for chunk in content_chunks:
                    segments.append(('content', chunk))
            
            evidence_db[i] = {
                'title': title,
                'segments': segments,
                'metadata': source
            }
        
        return evidence_db
    
    def _split_content_into_chunks(self, content: str, chunk_size: int = 200) -> List[str]:
        """Split content into meaningful chunks for evidence matching."""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                if len(current_chunk.strip()) >= self.min_evidence_length:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_evidence_length:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _validate_claim(self, 
                       claim: str, 
                       evidence_db: Dict[int, Dict[str, Any]], 
                       full_synthesis: str) -> ClaimValidation:
        """Validate a single claim against the evidence database."""
        evidence_items = []
        
        # Check if claim has explicit citations
        cited_sources = self._extract_citations_from_claim_context(claim, full_synthesis)
        
        # Search for supporting evidence
        for source_id, source_data in evidence_db.items():
            for location, segment in source_data['segments']:
                similarity = self._calculate_text_similarity(claim, segment)
                
                if similarity >= self.similarity_threshold:
                    evidence = EvidenceItem(
                        source_id=source_id,
                        source_title=source_data['title'],
                        evidence_text=segment,
                        confidence=similarity,
                        location=location,
                        context=self._extract_context(segment, source_data['segments'])
                    )
                    evidence_items.append(evidence)
        
        # Sort evidence by confidence
        evidence_items.sort(key=lambda x: x.confidence, reverse=True)
        
        # Determine if claim is supported
        is_supported = len(evidence_items) > 0
        confidence_score = max((e.confidence for e in evidence_items), default=0.0)
        
        # Identify issues
        issues = []
        if not is_supported:
            issues.append("No supporting evidence found in sources")
        
        if cited_sources and not evidence_items:
            issues.append("Claim cites sources but no evidence found in those sources")
        
        if not cited_sources and evidence_items:
            issues.append("Evidence exists but claim lacks proper citations")
        
        # Determine validation method
        if cited_sources and evidence_items:
            validation_method = "citation_and_evidence"
        elif evidence_items:
            validation_method = "evidence_only"
        elif cited_sources:
            validation_method = "citation_only"
        else:
            validation_method = "no_validation"
        
        return ClaimValidation(
            claim_text=claim,
            is_supported=is_supported,
            evidence_items=evidence_items[:3],  # Top 3 evidence items
            confidence_score=round(confidence_score, 3),
            validation_method=validation_method,
            issues=issues
        )
    
    def _extract_citations_from_claim_context(self, claim: str, full_synthesis: str) -> List[int]:
        """Extract citation numbers near a specific claim."""
        # Find the claim in the full synthesis
        claim_start = full_synthesis.find(claim)
        if claim_start == -1:
            return []
        
        # Look for citations in a window around the claim
        window_start = max(0, claim_start - 100)
        window_end = min(len(full_synthesis), claim_start + len(claim) + 100)
        context_window = full_synthesis[window_start:window_end]
        
        # Extract citation numbers
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, context_window)
            for match in matches:
                try:
                    citations.append(int(match))
                except ValueError:
                    continue
        
        return list(set(citations))  # Remove duplicates
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text snippets."""
        # Normalize texts
        text1_norm = self._normalize_for_comparison(text1)
        text2_norm = self._normalize_for_comparison(text2)
        
        if not text1_norm or not text2_norm:
            return 0.0
        
        # Use difflib for sequence matching
        similarity = difflib.SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        # Also check for key term overlap
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            # Combine similarity scores
            similarity = 0.7 * similarity + 0.3 * word_overlap
        
        return similarity
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'what'
        }
        
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    
    def _extract_context(self, segment: str, all_segments: List[Tuple[str, str]]) -> str:
        """Extract context around a specific segment."""
        # Find the segment in the list
        for i, (location, text) in enumerate(all_segments):
            if text == segment:
                # Get surrounding segments for context
                context_parts = []
                
                # Previous segment
                if i > 0:
                    context_parts.append(all_segments[i-1][1][:50] + "...")
                
                # Current segment
                context_parts.append(f"[{text}]")
                
                # Next segment
                if i < len(all_segments) - 1:
                    context_parts.append("..." + all_segments[i+1][1][:50])
                
                return " ".join(context_parts)
        
        return segment  # Fallback
    
    def _identify_overall_issues(self, 
                                claim_validations: List[ClaimValidation], 
                                source_utilization: Dict[int, float], 
                                total_sources: int) -> List[str]:
        """Identify systematic issues with the content-citation mapping."""
        issues = []
        
        # Check for high number of unsupported claims
        unsupported_rate = sum(1 for v in claim_validations if not v.is_supported) / len(claim_validations)
        if unsupported_rate > 0.3:
            issues.append(f"High rate of unsupported claims ({unsupported_rate:.1%})")
        
        # Check for unused sources
        unused_sources = total_sources - len(source_utilization)
        if unused_sources > 0:
            issues.append(f"{unused_sources} sources not used in synthesis")
        
        # Check for over-reliance on single source
        if source_utilization:
            max_utilization = max(source_utilization.values())
            if max_utilization > 0.7 and len(source_utilization) > 1:
                issues.append("Over-reliance on single source")
        
        # Check for claims without citations
        claims_without_citations = sum(1 for v in claim_validations 
                                     if v.evidence_items and v.validation_method == "evidence_only")
        if claims_without_citations > 0:
            issues.append(f"{claims_without_citations} claims lack proper citations despite having evidence")
        
        # Check for citations without evidence
        citations_without_evidence = sum(1 for v in claim_validations 
                                       if not v.evidence_items and v.validation_method == "citation_only")
        if citations_without_evidence > 0:
            issues.append(f"{citations_without_evidence} citations lack supporting evidence")
        
        return issues
    
    def generate_validation_report(self, validation_result: ContentValidationResult) -> str:
        """Generate a human-readable validation report."""
        report_lines = [
            "Content-Citation Validation Report",
            "=" * 40,
            f"Overall Validation Score: {validation_result.validation_score:.1%}",
            f"Total Claims: {validation_result.total_claims}",
            f"Supported Claims: {validation_result.supported_claims}",
            f"Unsupported Claims: {validation_result.unsupported_claims}",
            ""
        ]
        
        # Source utilization
        if validation_result.source_utilization:
            report_lines.append("Source Utilization:")
            for source_id, utilization in validation_result.source_utilization.items():
                report_lines.append(f"  Source {source_id}: {utilization:.1%}")
            report_lines.append("")
        
        # Overall issues
        if validation_result.overall_issues:
            report_lines.append("Issues Identified:")
            for issue in validation_result.overall_issues:
                report_lines.append(f"  • {issue}")
            report_lines.append("")
        
        # Unsupported claims (first 5)
        unsupported_claims = [v for v in validation_result.claim_validations if not v.is_supported]
        if unsupported_claims:
            report_lines.append("Unsupported Claims (sample):")
            for i, claim in enumerate(unsupported_claims[:5], 1):
                report_lines.append(f"  {i}. {claim.claim_text[:100]}...")
                if claim.issues:
                    report_lines.append(f"     Issues: {', '.join(claim.issues)}")
            report_lines.append("")
        
        # Well-supported claims (first 3)
        well_supported = [v for v in validation_result.claim_validations 
                         if v.is_supported and v.confidence_score > 0.8]
        if well_supported:
            report_lines.append("Well-Supported Claims (sample):")
            for i, claim in enumerate(well_supported[:3], 1):
                report_lines.append(f"  {i}. {claim.claim_text[:100]}...")
                report_lines.append(f"     Confidence: {claim.confidence_score:.3f}")
                if claim.evidence_items:
                    report_lines.append(f"     Evidence: {claim.evidence_items[0].evidence_text[:80]}...")
        
        return "\n".join(report_lines)
    
    def get_evidence_for_claim(self, claim: str, validation_result: ContentValidationResult) -> List[EvidenceItem]:
        """Get evidence items for a specific claim."""
        for claim_validation in validation_result.claim_validations:
            if claim_validation.claim_text == claim:
                return claim_validation.evidence_items
        return []


# Example usage and testing
if __name__ == "__main__":
    # Initialize content linker
    linker = ContentLinker(similarity_threshold=0.6)
    
    # Sample synthesis content
    synthesis = """
    Recent research shows that Docker containers provide significant advantages over traditional virtualization [Source 1]. 
    Container technology offers better resource utilization and faster startup times compared to virtual machines. 
    However, security considerations must be carefully addressed when deploying containers in production environments [Source 2].
    The orchestration of containers using Kubernetes has become the industry standard for managing containerized applications.
    """
    
    # Sample source contents
    sources = [
        {
            'title': 'Docker vs Virtual Machines: Performance Comparison',
            'content': 'Our benchmarks demonstrate that Docker containers provide better resource utilization and significantly faster startup times compared to traditional virtual machines. Container technology shows 30% better memory efficiency.',
            'abstract': 'This study compares Docker containers with virtual machines across various performance metrics.'
        },
        {
            'title': 'Container Security Best Practices',
            'content': 'Security considerations are paramount when deploying containers in production. Common vulnerabilities include privilege escalation and insecure container images. Proper security policies must be implemented.',
            'abstract': 'A comprehensive guide to securing containerized applications in production environments.'
        },
        {
            'title': 'Machine Learning in Healthcare',
            'content': 'Machine learning algorithms show promise in medical diagnosis and treatment recommendation systems.',
            'abstract': 'Applications of AI in healthcare settings.'
        }
    ]
    
    print("🔗 Testing Content Linker")
    print("=" * 50)
    
    # Validate content
    result = linker.validate_content_citations(synthesis, sources)
    
    # Generate report
    report = linker.generate_validation_report(result)
    print(report)
    
    print(f"\n📊 Detailed Results:")
    for i, claim_validation in enumerate(result.claim_validations, 1):
        print(f"\nClaim {i}: {claim_validation.claim_text[:80]}...")
        print(f"Supported: {claim_validation.is_supported}")
        print(f"Confidence: {claim_validation.confidence_score}")
        print(f"Evidence Items: {len(claim_validation.evidence_items)}")
        if claim_validation.issues:
            print(f"Issues: {', '.join(claim_validation.issues)}")
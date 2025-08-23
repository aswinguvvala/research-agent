"""
AI Honest Synthesizer
Evidence-based synthesis with honest assessment of limitations and evidence quality.
"""

import openai
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class EvidenceAssessment:
    """Assessment of evidence quality for a specific claim or finding."""
    claim: str
    evidence_level: str  # 'strong', 'moderate', 'limited', 'insufficient'
    supporting_sources: int
    confidence_score: float
    limitations: List[str]
    quality_factors: Dict[str, float]


@dataclass
class HonestSynthesis:
    """Honest synthesis with transparent evidence assessment."""
    summary: str
    key_findings: List[str]
    evidence_assessments: List[EvidenceAssessment]
    methodology_critique: str
    limitations_discussion: str
    confidence_intervals: Dict[str, Tuple[float, float]]
    bias_assessment: str
    reliability_score: float
    recommendations: List[str]
    honest_caveats: List[str]
    synthesis_metadata: Dict[str, Any]


class AIHonestSynthesizer:
    """
    AI synthesizer that provides honest, evidence-based synthesis with transparent
    assessment of limitations, biases, and evidence quality.
    """
    
    def __init__(self, openai_api_key: str, debug_mode: bool = False):
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.debug_mode = debug_mode
        
        # Evidence quality thresholds
        self.evidence_thresholds = {
            'strong': {'min_sources': 5, 'min_citations': 100, 'min_recency': 0.4},
            'moderate': {'min_sources': 3, 'min_citations': 20, 'min_recency': 0.2},
            'limited': {'min_sources': 2, 'min_citations': 5, 'min_recency': 0.1},
            'insufficient': {'min_sources': 1, 'min_citations': 0, 'min_recency': 0.0}
        }
        
        logger.info("🔍 AI Honest Synthesizer initialized")
    
    async def create_honest_synthesis(self, 
                                    query: str,
                                    sources: List[Dict[str, Any]],
                                    existing_synthesis: Optional[str] = None) -> HonestSynthesis:
        """
        Create honest synthesis with transparent evidence assessment.
        
        Args:
            query: Original research query
            sources: List of source documents
            existing_synthesis: Optional existing synthesis to evaluate
            
        Returns:
            HonestSynthesis with evidence-based assessment
        """
        logger.info(f"🔍 Creating honest synthesis for: {query}")
        logger.info(f"📊 Analyzing {len(sources)} sources for evidence quality")
        
        try:
            # Step 1: Extract key claims from existing synthesis or sources
            key_claims = await self._extract_key_claims(query, sources, existing_synthesis)
            
            # Step 2: Assess evidence for each claim
            evidence_assessments = await self._assess_evidence_for_claims(key_claims, sources)
            
            # Step 3: Generate honest summary
            honest_summary = await self._generate_honest_summary(query, sources, evidence_assessments)
            
            # Step 4: Extract validated key findings
            key_findings = await self._extract_validated_findings(evidence_assessments)
            
            # Step 5: Critique methodology
            methodology_critique = await self._critique_methodology(sources)
            
            # Step 6: Discuss limitations
            limitations_discussion = await self._discuss_limitations(sources, evidence_assessments)
            
            # Step 7: Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(evidence_assessments)
            
            # Step 8: Assess bias
            bias_assessment = await self._assess_bias(sources, query)
            
            # Step 9: Calculate reliability score
            reliability_score = self._calculate_reliability_score(sources, evidence_assessments)
            
            # Step 10: Generate recommendations
            recommendations = await self._generate_honest_recommendations(query, evidence_assessments)
            
            # Step 11: Identify honest caveats
            honest_caveats = await self._identify_honest_caveats(sources, evidence_assessments)
            
            # Create synthesis metadata
            synthesis_metadata = {
                'total_sources': len(sources),
                'high_quality_sources': sum(1 for s in sources if s.get('citations', 0) > 50),
                'recent_sources': sum(1 for s in sources if s.get('year', 0) >= datetime.now().year - 3),
                'evidence_assessments_count': len(evidence_assessments),
                'synthesis_timestamp': datetime.now().isoformat(),
                'avg_source_citations': sum(s.get('citations', 0) for s in sources) / len(sources) if sources else 0
            }
            
            synthesis = HonestSynthesis(
                summary=honest_summary,
                key_findings=key_findings,
                evidence_assessments=evidence_assessments,
                methodology_critique=methodology_critique,
                limitations_discussion=limitations_discussion,
                confidence_intervals=confidence_intervals,
                bias_assessment=bias_assessment,
                reliability_score=reliability_score,
                recommendations=recommendations,
                honest_caveats=honest_caveats,
                synthesis_metadata=synthesis_metadata
            )
            
            logger.info(f"✅ Honest synthesis completed with {len(evidence_assessments)} evidence assessments")
            logger.info(f"📈 Reliability score: {reliability_score:.2f}")
            
            return synthesis
            
        except Exception as e:
            logger.error(f"❌ Honest synthesis failed: {e}")
            return self._create_fallback_honest_synthesis(query, sources)
    
    async def _extract_key_claims(self, 
                                query: str, 
                                sources: List[Dict[str, Any]], 
                                existing_synthesis: Optional[str]) -> List[str]:
        """Extract key claims that need evidence assessment."""
        try:
            if existing_synthesis:
                # Extract claims from existing synthesis
                prompt = f"""
                Extract 5-8 key factual claims from this research synthesis:
                
                Query: "{query}"
                Synthesis: {existing_synthesis[:1000]}
                
                Extract specific, testable claims that can be evaluated against evidence.
                Avoid vague or subjective statements.
                
                Return as numbered list of specific claims.
                """
            else:
                # Extract claims from source abstracts
                source_abstracts = []
                for source in sources[:15]:
                    title = source.get('title', 'Unknown')
                    abstract = source.get('abstract', 'No abstract')[:200]
                    source_abstracts.append(f"- {title}: {abstract}")
                
                abstracts_text = "\n".join(source_abstracts)
                
                prompt = f"""
                Extract 5-8 key factual claims about "{query}" from these research sources:
                
                {abstracts_text}
                
                Extract specific, testable claims that can be evaluated against evidence.
                Focus on main findings, conclusions, or assertions.
                
                Return as numbered list of specific claims.
                """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2
            )
            
            claims_text = response.choices[0].message.content.strip()
            claims = self._parse_numbered_list(claims_text)
            
            return claims[:8]  # Limit to 8 claims
            
        except Exception as e:
            logger.error(f"Key claims extraction failed: {e}")
            return [f"Research on {query} shows various approaches and findings"]
    
    async def _assess_evidence_for_claims(self, 
                                        claims: List[str], 
                                        sources: List[Dict[str, Any]]) -> List[EvidenceAssessment]:
        """Assess evidence quality for each claim."""
        assessments = []
        
        for claim in claims:
            try:
                # Find sources that support this claim
                supporting_sources = await self._find_supporting_sources(claim, sources)
                
                # Assess evidence level
                evidence_level = self._determine_evidence_level(supporting_sources)
                
                # Calculate confidence score
                confidence_score = self._calculate_claim_confidence(supporting_sources)
                
                # Identify limitations
                limitations = await self._identify_claim_limitations(claim, supporting_sources)
                
                # Calculate quality factors
                quality_factors = self._calculate_quality_factors(supporting_sources)
                
                assessment = EvidenceAssessment(
                    claim=claim,
                    evidence_level=evidence_level,
                    supporting_sources=len(supporting_sources),
                    confidence_score=confidence_score,
                    limitations=limitations,
                    quality_factors=quality_factors
                )
                
                assessments.append(assessment)
                
            except Exception as e:
                logger.warning(f"Evidence assessment failed for claim: {claim[:50]}... Error: {e}")
                continue
        
        return assessments
    
    async def _find_supporting_sources(self, 
                                     claim: str, 
                                     sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find sources that support a specific claim."""
        try:
            # Extract key terms from claim
            claim_keywords = self._extract_keywords(claim)
            
            supporting_sources = []
            
            for source in sources:
                title = source.get('title', '').lower()
                abstract = source.get('abstract', '').lower()
                
                # Check for keyword overlap
                keyword_matches = sum(1 for keyword in claim_keywords 
                                    if keyword.lower() in title or keyword.lower() in abstract)
                
                # Simple heuristic: if >50% of keywords match, consider it supporting
                if len(claim_keywords) > 0 and keyword_matches / len(claim_keywords) > 0.5:
                    supporting_sources.append(source)
            
            # Also use AI to validate relevance for top candidates
            if len(supporting_sources) > 10:
                supporting_sources = await self._ai_validate_support(claim, supporting_sources[:10])
            
            return supporting_sources
            
        except Exception as e:
            logger.warning(f"Finding supporting sources failed: {e}")
            return sources[:3]  # Fallback to first few sources
    
    async def _ai_validate_support(self, 
                                 claim: str, 
                                 candidate_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use AI to validate which sources actually support the claim."""
        try:
            source_summaries = []
            for i, source in enumerate(candidate_sources):
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', 'No abstract')[:200]
                source_summaries.append(f"{i+1}. {title}: {abstract}")
            
            sources_text = "\n".join(source_summaries)
            
            prompt = f"""
            Determine which sources provide evidence for this claim: "{claim}"
            
            Candidate sources:
            {sources_text}
            
            Return the numbers of sources that provide evidence for this claim.
            Only include sources that directly support or relate to the claim.
            
            Format: [1, 3, 5] (example)
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract numbers
            numbers = re.findall(r'\d+', response_text)
            supporting_indices = [int(num) - 1 for num in numbers if int(num) <= len(candidate_sources)]
            
            return [candidate_sources[i] for i in supporting_indices if 0 <= i < len(candidate_sources)]
            
        except Exception as e:
            logger.warning(f"AI validation failed: {e}")
            return candidate_sources  # Return all candidates on failure
    
    def _determine_evidence_level(self, supporting_sources: List[Dict[str, Any]]) -> str:
        """Determine evidence level based on supporting sources."""
        if not supporting_sources:
            return 'insufficient'
        
        source_count = len(supporting_sources)
        avg_citations = sum(s.get('citations', 0) for s in supporting_sources) / len(supporting_sources)
        recent_ratio = sum(1 for s in supporting_sources 
                          if s.get('year', 0) >= datetime.now().year - 5) / len(supporting_sources)
        
        # Check against thresholds
        for level in ['strong', 'moderate', 'limited']:
            thresholds = self.evidence_thresholds[level]
            if (source_count >= thresholds['min_sources'] and 
                avg_citations >= thresholds['min_citations'] and 
                recent_ratio >= thresholds['min_recency']):
                return level
        
        return 'insufficient'
    
    def _calculate_claim_confidence(self, supporting_sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a claim based on supporting sources."""
        if not supporting_sources:
            return 0.0
        
        confidence = 0.0
        
        # Source count factor (up to 0.4)
        source_count = len(supporting_sources)
        confidence += min(0.4, source_count * 0.08)
        
        # Citation quality factor (up to 0.3)
        avg_citations = sum(s.get('citations', 0) for s in supporting_sources) / len(supporting_sources)
        if avg_citations > 200:
            confidence += 0.3
        elif avg_citations > 50:
            confidence += 0.2
        elif avg_citations > 10:
            confidence += 0.1
        
        # Recency factor (up to 0.2)
        recent_sources = sum(1 for s in supporting_sources 
                           if s.get('year', 0) >= datetime.now().year - 3)
        recency_ratio = recent_sources / len(supporting_sources)
        confidence += recency_ratio * 0.2
        
        # Venue quality factor (up to 0.1)
        high_quality_venues = sum(1 for s in supporting_sources
                                if any(venue in s.get('venue', '').lower() 
                                     for venue in ['nature', 'science', 'cell', 'pnas']))
        if high_quality_venues > 0:
            confidence += min(0.1, high_quality_venues * 0.05)
        
        return min(1.0, confidence)
    
    async def _identify_claim_limitations(self, 
                                        claim: str, 
                                        supporting_sources: List[Dict[str, Any]]) -> List[str]:
        """Identify limitations in the evidence for a claim."""
        limitations = []
        
        # Sample size limitation
        if len(supporting_sources) < 3:
            limitations.append(f"Limited evidence base: only {len(supporting_sources)} supporting sources")
        
        # Recency limitation
        current_year = datetime.now().year
        old_sources = sum(1 for s in supporting_sources if s.get('year', 0) < current_year - 10)
        if old_sources > len(supporting_sources) * 0.5:
            limitations.append("Evidence may be outdated: significant portion from >10 years ago")
        
        # Citation limitation
        low_citation_sources = sum(1 for s in supporting_sources if s.get('citations', 0) < 10)
        if low_citation_sources > len(supporting_sources) * 0.5:
            limitations.append("Evidence quality concerns: many sources have low citation counts")
        
        # Venue diversity limitation
        venues = [s.get('venue', 'unknown') for s in supporting_sources]
        unique_venues = len(set(venues))
        if unique_venues < len(supporting_sources) * 0.5:
            limitations.append("Limited venue diversity: evidence concentrated in few publications")
        
        return limitations[:4]  # Limit to 4 most important limitations
    
    def _calculate_quality_factors(self, supporting_sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality factors for evidence assessment."""
        if not supporting_sources:
            return {}
        
        # Citation quality
        avg_citations = sum(s.get('citations', 0) for s in supporting_sources) / len(supporting_sources)
        citation_score = min(1.0, avg_citations / 100)  # Normalize to 100 citations
        
        # Recency score
        current_year = datetime.now().year
        avg_age = sum(current_year - s.get('year', current_year) for s in supporting_sources) / len(supporting_sources)
        recency_score = max(0.0, 1.0 - (avg_age / 20))  # Decreases over 20 years
        
        # Venue quality
        high_quality_count = sum(1 for s in supporting_sources
                               if any(venue in s.get('venue', '').lower() 
                                    for venue in ['nature', 'science', 'cell', 'pnas']))
        venue_score = high_quality_count / len(supporting_sources)
        
        # Source diversity
        venues = [s.get('venue', 'unknown') for s in supporting_sources]
        diversity_score = len(set(venues)) / len(venues) if venues else 0
        
        return {
            'citation_quality': citation_score,
            'recency': recency_score,
            'venue_quality': venue_score,
            'source_diversity': diversity_score
        }
    
    async def _generate_honest_summary(self, 
                                     query: str, 
                                     sources: List[Dict[str, Any]], 
                                     evidence_assessments: List[EvidenceAssessment]) -> str:
        """Generate honest summary acknowledging evidence limitations."""
        try:
            # Prepare evidence overview
            evidence_overview = []
            for assessment in evidence_assessments[:5]:  # Top 5 assessments
                evidence_overview.append(
                    f"- {assessment.claim[:50]}... (Evidence: {assessment.evidence_level}, "
                    f"Confidence: {assessment.confidence_score:.2f})"
                )
            
            evidence_text = "\n".join(evidence_overview)
            
            # Calculate overall evidence strength
            strong_evidence = sum(1 for a in evidence_assessments if a.evidence_level == 'strong')
            total_assessments = len(evidence_assessments)
            
            prompt = f"""
            Create an honest research summary for: "{query}"
            
            Evidence Assessment Overview:
            {evidence_text}
            
            Research Context:
            - Total sources: {len(sources)}
            - Strong evidence claims: {strong_evidence}/{total_assessments}
            - Average source age: {self._calculate_avg_source_age(sources):.1f} years
            
            Create a summary that:
            1. Presents findings honestly with evidence levels
            2. Acknowledges limitations and uncertainties
            3. Distinguishes between strong and weak evidence
            4. Avoids overstating conclusions
            5. Uses qualifying language where appropriate
            
            Be scholarly but honest about evidence quality.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Honest summary generation failed: {e}")
            return f"Research on '{query}' shows mixed evidence quality with some well-supported findings and areas requiring further investigation."
    
    async def _extract_validated_findings(self, evidence_assessments: List[EvidenceAssessment]) -> List[str]:
        """Extract only well-validated findings."""
        validated_findings = []
        
        for assessment in evidence_assessments:
            if assessment.evidence_level in ['strong', 'moderate'] and assessment.confidence_score > 0.6:
                # Add qualifier based on evidence strength
                if assessment.evidence_level == 'strong':
                    qualifier = "Strong evidence suggests"
                else:
                    qualifier = "Moderate evidence indicates"
                
                validated_finding = f"{qualifier}: {assessment.claim}"
                validated_findings.append(validated_finding)
        
        return validated_findings
    
    async def _critique_methodology(self, sources: List[Dict[str, Any]]) -> str:
        """Critique research methodology across sources."""
        try:
            # Analyze methodological diversity
            method_indicators = {}
            for source in sources:
                abstract = source.get('abstract', '').lower()
                
                # Look for methodology keywords
                if 'experiment' in abstract or 'trial' in abstract:
                    method_indicators['experimental'] = method_indicators.get('experimental', 0) + 1
                if 'survey' in abstract or 'questionnaire' in abstract:
                    method_indicators['survey'] = method_indicators.get('survey', 0) + 1
                if 'analysis' in abstract or 'statistical' in abstract:
                    method_indicators['analytical'] = method_indicators.get('analytical', 0) + 1
                if 'model' in abstract or 'simulation' in abstract:
                    method_indicators['modeling'] = method_indicators.get('modeling', 0) + 1
            
            methods_summary = ", ".join([f"{method}: {count}" for method, count in method_indicators.items()])
            
            prompt = f"""
            Critique the research methodology across these sources:
            
            Total sources: {len(sources)}
            Methodological approaches identified: {methods_summary}
            
            Provide a methodological critique that addresses:
            1. Methodological diversity and limitations
            2. Potential biases in research approaches
            3. Generalizability concerns
            4. Methodological gaps or blind spots
            5. Suggestions for methodological improvements
            
            Be constructive but honest about methodological limitations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Methodology critique failed: {e}")
            return "Methodological diversity varies across sources, with potential for both strengths and limitations in research approaches."
    
    async def _discuss_limitations(self, 
                                 sources: List[Dict[str, Any]], 
                                 evidence_assessments: List[EvidenceAssessment]) -> str:
        """Discuss limitations in the research and evidence."""
        try:
            # Collect all limitations
            all_limitations = []
            for assessment in evidence_assessments:
                all_limitations.extend(assessment.limitations)
            
            # Count common limitations
            limitation_counts = Counter(all_limitations)
            common_limitations = [lim for lim, count in limitation_counts.most_common(5)]
            
            # Analyze source limitations
            source_age_dist = self._analyze_source_age_distribution(sources)
            citation_dist = self._analyze_citation_distribution(sources)
            
            limitations_text = "\n".join(f"- {lim}" for lim in common_limitations)
            
            prompt = f"""
            Discuss limitations in this research synthesis:
            
            Common evidence limitations:
            {limitations_text}
            
            Source characteristics:
            - Age distribution: {source_age_dist}
            - Citation distribution: {citation_dist}
            - Total sources: {len(sources)}
            
            Provide an honest discussion of:
            1. Evidence quality limitations
            2. Coverage gaps
            3. Temporal limitations
            4. Methodological constraints
            5. Generalizability concerns
            
            Be specific and constructive about limitations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Limitations discussion failed: {e}")
            return "This synthesis has several limitations including evidence quality variations and potential coverage gaps."
    
    def _calculate_confidence_intervals(self, evidence_assessments: List[EvidenceAssessment]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for different aspects."""
        if not evidence_assessments:
            return {}
        
        confidence_scores = [a.confidence_score for a in evidence_assessments]
        
        # Simple confidence intervals (mean ± std)
        import statistics
        
        mean_confidence = statistics.mean(confidence_scores)
        std_confidence = statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
        
        # Evidence level distribution
        evidence_levels = [a.evidence_level for a in evidence_assessments]
        strong_ratio = evidence_levels.count('strong') / len(evidence_levels)
        moderate_ratio = evidence_levels.count('moderate') / len(evidence_levels)
        
        return {
            'overall_confidence': (max(0, mean_confidence - std_confidence), 
                                 min(1, mean_confidence + std_confidence)),
            'strong_evidence_ratio': (max(0, strong_ratio - 0.1), min(1, strong_ratio + 0.1)),
            'moderate_evidence_ratio': (max(0, moderate_ratio - 0.1), min(1, moderate_ratio + 0.1))
        }
    
    async def _assess_bias(self, sources: List[Dict[str, Any]], query: str) -> str:
        """Assess potential biases in the research collection."""
        try:
            # Analyze publication bias indicators
            venue_distribution = Counter(s.get('venue', 'unknown') for s in sources)
            year_distribution = Counter(s.get('year', 0) for s in sources)
            
            # Check for geographic bias
            author_affiliations = []
            for source in sources:
                authors = source.get('authors', [])
                if authors:
                    author_affiliations.extend(authors[:2])  # First 2 authors
            
            prompt = f"""
            Assess potential biases in this research collection on: "{query}"
            
            Collection characteristics:
            - Top venues: {dict(venue_distribution.most_common(3))}
            - Year range: {min(year_distribution.keys()) if year_distribution else 'unknown'} - {max(year_distribution.keys()) if year_distribution else 'unknown'}
            - Total sources: {len(sources)}
            
            Assess potential biases:
            1. Publication bias (venue concentration)
            2. Temporal bias (publication years)
            3. Geographic/institutional bias
            4. Methodological bias
            5. Language/accessibility bias
            
            Provide honest assessment of bias risks.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Bias assessment failed: {e}")
            return "Potential biases include publication venue concentration and temporal distribution limitations."
    
    def _calculate_reliability_score(self, 
                                   sources: List[Dict[str, Any]], 
                                   evidence_assessments: List[EvidenceAssessment]) -> float:
        """Calculate overall reliability score for the synthesis."""
        if not sources or not evidence_assessments:
            return 0.0
        
        score = 0.0
        
        # Source quality factor (0-0.3)
        avg_citations = sum(s.get('citations', 0) for s in sources) / len(sources)
        citation_score = min(0.3, avg_citations / 200)  # Normalize to 200 citations
        score += citation_score
        
        # Evidence strength factor (0-0.4)
        strong_evidence = sum(1 for a in evidence_assessments if a.evidence_level == 'strong')
        evidence_score = (strong_evidence / len(evidence_assessments)) * 0.4
        score += evidence_score
        
        # Source diversity factor (0-0.2)
        venues = [s.get('venue', 'unknown') for s in sources]
        diversity_score = (len(set(venues)) / len(venues)) * 0.2 if venues else 0
        score += diversity_score
        
        # Recency factor (0-0.1)
        current_year = datetime.now().year
        recent_sources = sum(1 for s in sources if s.get('year', 0) >= current_year - 5)
        recency_score = (recent_sources / len(sources)) * 0.1
        score += recency_score
        
        return min(1.0, score)
    
    async def _generate_honest_recommendations(self, 
                                             query: str, 
                                             evidence_assessments: List[EvidenceAssessment]) -> List[str]:
        """Generate honest recommendations based on evidence quality."""
        try:
            # Analyze evidence patterns
            strong_claims = [a.claim for a in evidence_assessments if a.evidence_level == 'strong']
            weak_claims = [a.claim for a in evidence_assessments if a.evidence_level in ['limited', 'insufficient']]
            
            strong_text = "; ".join(strong_claims[:3])
            weak_text = "; ".join(weak_claims[:3])
            
            prompt = f"""
            Generate honest recommendations for: "{query}"
            
            Well-supported findings: {strong_text}
            Weakly-supported claims: {weak_text}
            
            Provide 3-5 honest recommendations that:
            1. Are based on well-supported evidence
            2. Acknowledge evidence limitations
            3. Suggest areas needing more research
            4. Are actionable and specific
            5. Include appropriate caveats
            
            Use qualifying language where evidence is limited.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            recommendations_text = response.choices[0].message.content.strip()
            recommendations = self._parse_numbered_list(recommendations_text)
            
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Further research needed to strengthen evidence base"]
    
    async def _identify_honest_caveats(self, 
                                     sources: List[Dict[str, Any]], 
                                     evidence_assessments: List[EvidenceAssessment]) -> List[str]:
        """Identify important caveats for the synthesis."""
        caveats = []
        
        # Evidence quality caveats
        insufficient_evidence = sum(1 for a in evidence_assessments if a.evidence_level == 'insufficient')
        if insufficient_evidence > 0:
            caveats.append(f"Some findings based on insufficient evidence ({insufficient_evidence} claims)")
        
        # Source size caveats
        if len(sources) < 10:
            caveats.append(f"Limited source base: only {len(sources)} sources analyzed")
        
        # Recency caveats
        current_year = datetime.now().year
        old_sources = sum(1 for s in sources if s.get('year', 0) < current_year - 10)
        if old_sources > len(sources) * 0.3:
            caveats.append("Significant portion of evidence from older research (>10 years)")
        
        # Citation quality caveats
        low_citation_sources = sum(1 for s in sources if s.get('citations', 0) < 10)
        if low_citation_sources > len(sources) * 0.4:
            caveats.append("Many sources have limited citation validation")
        
        # Venue diversity caveats
        venues = [s.get('venue', 'unknown') for s in sources]
        unique_venues = len(set(venues))
        if unique_venues < len(sources) * 0.5:
            caveats.append("Evidence concentrated in limited publication venues")
        
        return caveats[:5]  # Limit to most important caveats
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'use', 'way', 'she', 'many'}
        
        return [word for word in words if word not in stop_words and len(word) > 3]
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered list from text."""
        items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match numbered items
            match = re.match(r'^\d+\.?\s*(.+)', line)
            if match:
                items.append(match.group(1).strip())
            elif line and not line.startswith('#') and len(line) > 10:
                items.append(line)
        
        return items
    
    def _calculate_avg_source_age(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate average age of sources in years."""
        current_year = datetime.now().year
        ages = [current_year - s.get('year', current_year) for s in sources if s.get('year', 0) > 0]
        return sum(ages) / len(ages) if ages else 0
    
    def _analyze_source_age_distribution(self, sources: List[Dict[str, Any]]) -> str:
        """Analyze source age distribution."""
        current_year = datetime.now().year
        recent = sum(1 for s in sources if s.get('year', 0) >= current_year - 3)
        medium = sum(1 for s in sources if current_year - 10 < s.get('year', 0) < current_year - 3)
        old = sum(1 for s in sources if s.get('year', 0) <= current_year - 10)
        
        return f"Recent(<3y): {recent}, Medium(3-10y): {medium}, Old(>10y): {old}"
    
    def _analyze_citation_distribution(self, sources: List[Dict[str, Any]]) -> str:
        """Analyze citation distribution."""
        high = sum(1 for s in sources if s.get('citations', 0) > 100)
        medium = sum(1 for s in sources if 10 < s.get('citations', 0) <= 100)
        low = sum(1 for s in sources if s.get('citations', 0) <= 10)
        
        return f"High(>100): {high}, Medium(10-100): {medium}, Low(≤10): {low}"
    
    def _create_fallback_honest_synthesis(self, query: str, sources: List[Dict[str, Any]]) -> HonestSynthesis:
        """Create fallback synthesis when main synthesis fails."""
        return HonestSynthesis(
            summary=f"Analysis of {len(sources)} sources on '{query}' reveals mixed evidence quality requiring careful interpretation.",
            key_findings=[f"Research on {query} shows diverse approaches with varying evidence strength"],
            evidence_assessments=[],
            methodology_critique="Methodological analysis limited due to processing constraints.",
            limitations_discussion="This synthesis has significant limitations due to processing constraints.",
            confidence_intervals={},
            bias_assessment="Potential biases cannot be fully assessed in this analysis.",
            reliability_score=0.3,
            recommendations=["Further comprehensive analysis recommended"],
            honest_caveats=["This analysis has significant limitations", "Evidence assessment is incomplete"],
            synthesis_metadata={'total_sources': len(sources), 'fallback_mode': True}
        )


async def test_honest_synthesizer():
    """Test function for the honest synthesizer."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    synthesizer = AIHonestSynthesizer(api_key, debug_mode=True)
    
    query = "machine learning transformer architectures"
    
    # Create test sources
    test_sources = [
        {
            'title': 'Attention Is All You Need',
            'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...',
            'year': 2017,
            'venue': 'NIPS',
            'citations': 45000,
            'authors': ['Vaswani et al.']
        },
        {
            'title': 'Limited Study on Transformers',
            'abstract': 'A small-scale study examining transformer performance...',
            'year': 2022,
            'venue': 'Minor Conference',
            'citations': 5,
            'authors': ['Smith et al.']
        }
    ]
    
    print(f"🔍 Testing honest synthesis for: {query}")
    synthesis = await synthesizer.create_honest_synthesis(query, test_sources)
    
    print(f"\nReliability Score: {synthesis.reliability_score:.2f}")
    print(f"Evidence Assessments: {len(synthesis.evidence_assessments)}")
    print(f"Honest Caveats: {len(synthesis.honest_caveats)}")
    print(f"Key Findings: {len(synthesis.key_findings)}")


if __name__ == "__main__":
    asyncio.run(test_honest_synthesizer())
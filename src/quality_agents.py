"""
Quality Assurance Agents
Specialized agents for fact-checking, synthesis, and quality control
in the multi-agent research system.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from collections import Counter, defaultdict
import re

from .multi_agent_research_system import BaseAgent, ResearchTask, AgentMessage, MessageType

logger = logging.getLogger(__name__)


class FactChecker(BaseAgent):
    """Specialized agent for fact-checking and validation of research findings"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("fact_checker", "Fact Verification Specialist", openai_api_key)
        self.specialization = "fact_checking"
        self.capabilities.update([
            "source_verification", "claim_validation", "consistency_checking",
            "bias_detection", "reliability_assessment", "cross_referencing"
        ])
        
        # Fact-checking specific configurations
        self.verified_sources = set()
        self.flagged_sources = set()
        self.fact_database = {}
        self.validation_history = []
        
        # Credibility scoring weights
        self.source_weights = {
            "arxiv": 0.9,
            "pubmed": 0.9,
            "academic_journal": 0.9,
            "government": 0.8,
            "wikipedia": 0.6,
            "news": 0.5,
            "blog": 0.3,
            "social_media": 0.2,
            "unknown": 0.1
        }
    
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process fact-checking tasks by validating research findings"""
        logger.info(f"Fact checker processing task: {task.description}")
        
        try:
            # Extract claims and findings to verify
            findings = task.metadata.get("findings_to_verify", [])
            sources = task.metadata.get("sources", [])
            
            if not findings:
                return {"task_id": task.id, "agent_type": "fact_checker", "error": "No findings provided for verification"}
            
            # Perform fact-checking analysis
            verification_results = []
            
            for finding in findings[:10]:  # Limit to prevent overwhelm
                result = await self._verify_finding(finding, sources)
                verification_results.append(result)
            
            # Assess overall reliability
            reliability_score = self._calculate_reliability_score(verification_results)
            
            # Identify potential biases and issues
            bias_assessment = await self._assess_bias(findings, sources)
            
            # Check for internal consistency
            consistency_check = await self._check_consistency(findings)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(verification_results, bias_assessment)
            
            return {
                "task_id": task.id,
                "agent_type": "fact_checker",
                "verification_results": verification_results,
                "reliability_score": reliability_score,
                "bias_assessment": bias_assessment,
                "consistency_check": consistency_check,
                "recommendations": recommendations,
                "findings": [f"Verified {len(verification_results)} claims", f"Overall reliability: {reliability_score:.2f}"],
                "confidence": 0.95,
                "sources": ["fact_verification", "source_analysis"],
                "metadata": {
                    "verification_timestamp": datetime.now().isoformat(),
                    "claims_verified": len(verification_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fact-checking task: {e}")
            return {
                "task_id": task.id,
                "agent_type": "fact_checker",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _verify_finding(self, finding: str, sources: List[str]) -> Dict:
        """Verify a specific finding or claim"""
        prompt = f"""
        Fact-check this research finding: "{finding}"
        
        Available sources: {', '.join(sources[:5])}
        
        Provide verification assessment as JSON:
        {{
            "claim": "{finding}",
            "verification_status": "verified|partially_verified|disputed|unverified",
            "confidence_level": 0.0-1.0,
            "supporting_evidence": ["evidence1", "evidence2"],
            "contradicting_evidence": ["contradiction1", "contradiction2"],
            "source_quality": "high|medium|low",
            "potential_issues": ["issue1", "issue2"],
            "verification_notes": "detailed assessment"
        }}
        
        Be objective and highlight any uncertainties or limitations.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert fact-checker. Be thorough and objective. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=600)
            result = json.loads(response)
            
            # Add source credibility assessment
            result["source_credibility"] = self._assess_source_credibility(sources)
            
            return result
        except Exception as e:
            logger.warning(f"Verification failed for finding: {e}")
            return {
                "claim": finding,
                "verification_status": "unverified",
                "confidence_level": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "source_quality": "unknown",
                "potential_issues": ["verification_failed"],
                "verification_notes": f"Automatic verification failed: {str(e)}",
                "source_credibility": 0.0
            }
    
    def _assess_source_credibility(self, sources: List[str]) -> float:
        """Assess the credibility of sources"""
        if not sources:
            return 0.0
        
        total_weight = 0.0
        for source in sources:
            source_lower = source.lower()
            weight = 0.1  # Default for unknown
            
            for source_type, source_weight in self.source_weights.items():
                if source_type in source_lower:
                    weight = source_weight
                    break
            
            total_weight += weight
        
        return min(total_weight / len(sources), 1.0)
    
    def _calculate_reliability_score(self, verification_results: List[Dict]) -> float:
        """Calculate overall reliability score from verification results"""
        if not verification_results:
            return 0.0
        
        total_score = 0.0
        for result in verification_results:
            status = result.get("verification_status", "unverified")
            confidence = result.get("confidence_level", 0.0)
            source_credibility = result.get("source_credibility", 0.0)
            
            # Weight by status
            status_weight = {
                "verified": 1.0,
                "partially_verified": 0.7,
                "disputed": 0.3,
                "unverified": 0.0
            }.get(status, 0.0)
            
            # Combine factors
            claim_score = (status_weight * 0.5 + confidence * 0.3 + source_credibility * 0.2)
            total_score += claim_score
        
        return total_score / len(verification_results)
    
    async def _assess_bias(self, findings: List[str], sources: List[str]) -> Dict:
        """Assess potential biases in findings and sources"""
        findings_text = " ".join(findings[:5])  # Sample of findings
        sources_text = ", ".join(sources[:10])
        
        prompt = f"""
        Assess potential biases in this research:
        
        Sample findings: {findings_text[:800]}
        Sources: {sources_text}
        
        Analyze bias as JSON:
        {{
            "source_bias": "low|medium|high",
            "selection_bias": "low|medium|high",
            "confirmation_bias": "low|medium|high",
            "language_bias": "present|not_detected",
            "temporal_bias": "current|outdated|mixed",
            "geographic_bias": "global|regional|local",
            "bias_indicators": ["indicator1", "indicator2"],
            "mitigation_suggestions": ["suggestion1", "suggestion2"],
            "overall_bias_risk": "low|medium|high"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a bias detection expert. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=500)
            return json.loads(response)
        except:
            return {
                "source_bias": "medium",
                "selection_bias": "medium",
                "confirmation_bias": "medium",
                "language_bias": "not_detected",
                "temporal_bias": "current",
                "geographic_bias": "regional",
                "bias_indicators": ["limited_source_diversity"],
                "mitigation_suggestions": ["diversify_sources", "cross_validate_claims"],
                "overall_bias_risk": "medium"
            }
    
    async def _check_consistency(self, findings: List[str]) -> Dict:
        """Check for internal consistency among findings"""
        if len(findings) < 2:
            return {"consistency_score": 1.0, "conflicts": [], "notes": "Insufficient data for consistency check"}
        
        findings_text = "\n".join([f"{i+1}. {finding}" for i, finding in enumerate(findings[:10])])
        
        prompt = f"""
        Check for logical consistency among these findings:
        
        {findings_text}
        
        Analyze consistency as JSON:
        {{
            "consistency_score": 0.0-1.0,
            "conflicts": [
                {{"finding_1": 1, "finding_2": 3, "conflict": "description of conflict"}},
                {{"finding_1": 2, "finding_2": 4, "conflict": "description of conflict"}}
            ],
            "supporting_relationships": [
                {{"finding_1": 1, "finding_2": 2, "relationship": "how they support each other"}}
            ],
            "logical_gaps": ["gap1", "gap2"],
            "consistency_notes": "overall assessment"
        }}
        
        A score of 1.0 means fully consistent, 0.0 means highly contradictory.
        """
        
        messages = [
            {"role": "system", "content": "You are a logical consistency analyst. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=600)
            return json.loads(response)
        except:
            return {
                "consistency_score": 0.7,
                "conflicts": [],
                "supporting_relationships": [],
                "logical_gaps": ["analysis_incomplete"],
                "consistency_notes": "Automatic consistency check failed"
            }
    
    async def _generate_recommendations(self, verification_results: List[Dict], bias_assessment: Dict) -> List[str]:
        """Generate recommendations based on fact-checking results"""
        verified_count = sum(1 for r in verification_results if r.get("verification_status") == "verified")
        total_count = len(verification_results)
        reliability_ratio = verified_count / total_count if total_count > 0 else 0
        
        recommendations = []
        
        # Reliability recommendations
        if reliability_ratio < 0.5:
            recommendations.append("Consider additional verification from independent sources")
        elif reliability_ratio < 0.7:
            recommendations.append("Some claims need stronger supporting evidence")
        
        # Bias recommendations
        overall_bias_risk = bias_assessment.get("overall_bias_risk", "medium")
        if overall_bias_risk == "high":
            recommendations.append("High bias risk detected - diversify sources and perspectives")
        elif overall_bias_risk == "medium":
            recommendations.append("Consider including alternative viewpoints to reduce bias")
        
        # Source quality recommendations
        low_quality_sources = sum(1 for r in verification_results if r.get("source_quality") == "low")
        if low_quality_sources > total_count * 0.3:
            recommendations.append("Improve source quality by using more authoritative references")
        
        # Specific issue recommendations
        common_issues = []
        for result in verification_results:
            common_issues.extend(result.get("potential_issues", []))
        
        issue_counts = Counter(common_issues)
        for issue, count in issue_counts.most_common(3):
            if count > 1:
                recommendations.append(f"Address recurring issue: {issue}")
        
        return recommendations[:5]  # Limit to most important recommendations


class SynthesisAgent(BaseAgent):
    """Specialized agent for synthesizing research findings from multiple sources and agents"""
    
    def __init__(self, openai_api_key: str):
        super().__init__("synthesis_agent", "Research Synthesis Specialist", openai_api_key)
        self.specialization = "synthesis"
        self.capabilities.update([
            "cross_domain_synthesis", "pattern_identification", "knowledge_integration",
            "gap_analysis", "trend_detection", "comprehensive_reporting"
        ])
        
        # Synthesis-specific configurations
        self.synthesis_templates = {}
        self.domain_connections = {}
        self.synthesis_history = []
    
    async def process_research_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process synthesis tasks by combining findings from multiple agents"""
        logger.info(f"Synthesis agent processing task: {task.description}")
        
        try:
            # Get multi-agent findings from task metadata
            agent_findings = task.metadata.get("agent_findings", {})
            research_goal = task.metadata.get("research_goal", "Unknown goal")
            
            if not agent_findings:
                return {"task_id": task.id, "agent_type": "synthesis", "error": "No agent findings provided for synthesis"}
            
            # Organize findings by domain
            organized_findings = self._organize_findings_by_domain(agent_findings)
            
            # Identify cross-domain patterns and connections
            cross_domain_patterns = await self._identify_cross_domain_patterns(organized_findings)
            
            # Create comprehensive synthesis
            comprehensive_synthesis = await self._create_comprehensive_synthesis(
                organized_findings, cross_domain_patterns, research_goal
            )
            
            # Identify research gaps and opportunities
            research_gaps = await self._identify_research_gaps(organized_findings, comprehensive_synthesis)
            
            # Generate actionable insights
            actionable_insights = await self._generate_actionable_insights(
                comprehensive_synthesis, cross_domain_patterns
            )
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(
                comprehensive_synthesis, research_goal
            )
            
            # Assess synthesis quality and confidence
            quality_assessment = self._assess_synthesis_quality(organized_findings, comprehensive_synthesis)
            
            return {
                "task_id": task.id,
                "agent_type": "synthesis",
                "executive_summary": executive_summary,
                "comprehensive_synthesis": comprehensive_synthesis,
                "cross_domain_patterns": cross_domain_patterns,
                "research_gaps": research_gaps,
                "actionable_insights": actionable_insights,
                "quality_assessment": quality_assessment,
                "domains_synthesized": list(organized_findings.keys()),
                "findings": [executive_summary, f"Synthesized {len(organized_findings)} domains"],
                "confidence": quality_assessment.get("overall_confidence", 0.8),
                "sources": ["multi_agent_synthesis"],
                "metadata": {
                    "synthesis_timestamp": datetime.now().isoformat(),
                    "agents_involved": list(agent_findings.keys()),
                    "research_goal": research_goal
                }
            }
            
        except Exception as e:
            logger.error(f"Error in synthesis task: {e}")
            return {
                "task_id": task.id,
                "agent_type": "synthesis",
                "error": str(e),
                "confidence": 0.0
            }
    
    def _organize_findings_by_domain(self, agent_findings: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Organize findings by research domain"""
        organized = defaultdict(list)
        
        for agent_id, findings in agent_findings.items():
            if isinstance(findings, dict):
                agent_type = findings.get("agent_type", "unknown")
                domain_findings = findings.get("findings", [])
                
                # Extract structured findings
                for finding in domain_findings:
                    organized[agent_type].append({
                        "agent": agent_id,
                        "finding": finding,
                        "confidence": findings.get("confidence", 0.5),
                        "sources": findings.get("sources", []),
                        "metadata": findings.get("metadata", {})
                    })
        
        return dict(organized)
    
    async def _identify_cross_domain_patterns(self, organized_findings: Dict[str, List[Dict]]) -> List[Dict]:
        """Identify patterns and connections across domains"""
        domains = list(organized_findings.keys())
        
        if len(domains) < 2:
            return []
        
        # Create summary of each domain's findings
        domain_summaries = {}
        for domain, findings in organized_findings.items():
            summary = " ".join([str(f.get("finding", "")) for f in findings[:5]])
            domain_summaries[domain] = summary[:500]  # Limit length
        
        prompt = f"""
        Identify cross-domain patterns and connections between these research domains:
        
        {chr(10).join([f"{domain}: {summary}" for domain, summary in domain_summaries.items()])}
        
        Find patterns as JSON array:
        [
            {{
                "pattern_name": "descriptive name",
                "domains_involved": ["domain1", "domain2"],
                "connection_type": "reinforcing|contradicting|complementary|causal",
                "description": "detailed description of the pattern",
                "significance": "why this pattern matters",
                "confidence": 0.0-1.0
            }}
        ]
        
        Limit to 5 most significant patterns.
        """
        
        messages = [
            {"role": "system", "content": "You are a cross-domain pattern analyst. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=800)
            return json.loads(response)
        except:
            return [{
                "pattern_name": "Multi-domain consensus",
                "domains_involved": domains[:2],
                "connection_type": "reinforcing",
                "description": "Multiple domains support similar conclusions",
                "significance": "Increases confidence in findings",
                "confidence": 0.7
            }]
    
    async def _create_comprehensive_synthesis(
        self, organized_findings: Dict[str, List[Dict]], 
        cross_domain_patterns: List[Dict], 
        research_goal: str
    ) -> str:
        """Create comprehensive synthesis of all findings"""
        
        # Prepare domain summaries
        domain_summaries = []
        for domain, findings in organized_findings.items():
            finding_texts = [str(f.get("finding", "")) for f in findings[:3]]
            domain_summaries.append(f"{domain.title()}: {'; '.join(finding_texts)}")
        
        # Prepare pattern summaries
        pattern_summaries = []
        for pattern in cross_domain_patterns[:3]:
            pattern_summaries.append(f"{pattern.get('pattern_name', '')}: {pattern.get('description', '')}")
        
        synthesis_prompt = f"""
        Create a comprehensive research synthesis for: "{research_goal}"
        
        Domain Findings:
        {chr(10).join(domain_summaries)}
        
        Cross-Domain Patterns:
        {chr(10).join(pattern_summaries)}
        
        Create a synthesis that:
        1. Integrates findings from all domains
        2. Highlights key insights and discoveries
        3. Addresses the original research goal
        4. Notes areas of consensus and disagreement
        5. Discusses implications and significance
        6. Identifies limitations and uncertainties
        
        Write in a clear, professional style suitable for a research report.
        Aim for 300-400 words.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert research synthesizer creating comprehensive analyses."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        try:
            return await self._llm_request(messages, max_tokens=600)
        except Exception as e:
            logger.error(f"Synthesis creation failed: {e}")
            return f"Synthesis of research findings for '{research_goal}' across {len(organized_findings)} domains. Key patterns identified include {len(cross_domain_patterns)} cross-domain connections. Further analysis needed."
    
    async def _identify_research_gaps(
        self, organized_findings: Dict[str, List[Dict]], 
        synthesis: str
    ) -> List[Dict]:
        """Identify gaps in the research that need further investigation"""
        
        domains_covered = list(organized_findings.keys())
        
        prompt = f"""
        Identify research gaps based on this synthesis and domain coverage:
        
        Synthesis: {synthesis[:600]}
        Domains covered: {', '.join(domains_covered)}
        
        Identify gaps as JSON array:
        [
            {{
                "gap_type": "methodological|domain|temporal|geographic",
                "description": "specific description of the gap",
                "priority": "high|medium|low",
                "suggested_approach": "how to address this gap",
                "required_expertise": ["expertise1", "expertise2"],
                "estimated_effort": "low|medium|high"
            }}
        ]
        
        Focus on significant gaps that would improve understanding.
        Limit to 4 most important gaps.
        """
        
        messages = [
            {"role": "system", "content": "You are a research gap analyst. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=500)
            return json.loads(response)
        except:
            return [{
                "gap_type": "methodological",
                "description": "Need for longitudinal validation studies",
                "priority": "medium",
                "suggested_approach": "Design long-term follow-up research",
                "required_expertise": ["research_methodology"],
                "estimated_effort": "high"
            }]
    
    async def _generate_actionable_insights(
        self, synthesis: str, cross_domain_patterns: List[Dict]
    ) -> List[Dict]:
        """Generate actionable insights from the synthesis"""
        
        patterns_summary = "; ".join([p.get("description", "") for p in cross_domain_patterns[:3]])
        
        prompt = f"""
        Generate actionable insights from this research synthesis:
        
        Synthesis: {synthesis[:600]}
        Key patterns: {patterns_summary}
        
        Generate insights as JSON array:
        [
            {{
                "insight": "clear, actionable insight",
                "action_category": "research|business|policy|technical|social",
                "stakeholders": ["who should act on this"],
                "timeframe": "immediate|short_term|long_term",
                "impact_potential": "low|medium|high",
                "implementation_steps": ["step1", "step2", "step3"]
            }}
        ]
        
        Focus on insights that can drive meaningful action.
        Limit to 4 most valuable insights.
        """
        
        messages = [
            {"role": "system", "content": "You are an actionable insights generator. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._llm_request(messages, max_tokens=600)
            return json.loads(response)
        except:
            return [{
                "insight": "Multi-domain research reveals significant opportunities",
                "action_category": "research",
                "stakeholders": ["researchers", "practitioners"],
                "timeframe": "short_term",
                "impact_potential": "medium",
                "implementation_steps": ["validate_findings", "plan_next_phase", "engage_stakeholders"]
            }]
    
    async def _create_executive_summary(self, synthesis: str, research_goal: str) -> str:
        """Create an executive summary of the synthesis"""
        
        prompt = f"""
        Create an executive summary for this research synthesis:
        
        Research Goal: {research_goal}
        Full Synthesis: {synthesis[:800]}
        
        Create a concise executive summary (100-150 words) that:
        1. States the research objective clearly
        2. Highlights the most important findings
        3. Notes key implications
        4. Mentions any critical limitations
        5. Provides a clear conclusion
        
        Write for an executive audience who needs the key points quickly.
        """
        
        messages = [
            {"role": "system", "content": "You are an executive summary writer. Be concise and impactful."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            return await self._llm_request(messages, max_tokens=250)
        except:
            return f"Research on '{research_goal}' reveals multi-domain insights with significant implications for stakeholders. Key findings demonstrate both opportunities and challenges requiring further investigation."
    
    def _assess_synthesis_quality(self, organized_findings: Dict[str, List[Dict]], synthesis: str) -> Dict:
        """Assess the quality and completeness of the synthesis"""
        
        total_findings = sum(len(findings) for findings in organized_findings.values())
        domains_count = len(organized_findings)
        
        # Calculate confidence based on various factors
        domain_coverage_score = min(domains_count / 4.0, 1.0)  # Assume 4 domains is ideal
        findings_volume_score = min(total_findings / 20.0, 1.0)  # Assume 20 findings is good
        synthesis_length_score = min(len(synthesis) / 1000.0, 1.0)  # Reasonable synthesis length
        
        # Average confidence of individual findings
        all_confidences = []
        for findings in organized_findings.values():
            for finding in findings:
                all_confidences.append(finding.get("confidence", 0.5))
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.5
        
        # Overall confidence calculation
        overall_confidence = (
            domain_coverage_score * 0.25 +
            findings_volume_score * 0.25 +
            synthesis_length_score * 0.2 +
            avg_confidence * 0.3
        )
        
        return {
            "overall_confidence": min(overall_confidence, 1.0),
            "domain_coverage": domains_count,
            "total_findings": total_findings,
            "average_finding_confidence": avg_confidence,
            "synthesis_completeness": synthesis_length_score,
            "quality_notes": f"Synthesized {domains_count} domains with {total_findings} findings"
        }


# Export quality agents
__all__ = [
    'FactChecker',
    'SynthesisAgent'
]
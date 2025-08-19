"""
Enhanced AI Synthesizer
Advanced synthesis engine with structured output templates and comprehensive analysis.
"""

import openai
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from advanced_search_engine import EnhancedSource

logger = logging.getLogger(__name__)


@dataclass
class StructuredSynthesis:
    """Structured synthesis result with categorized information."""
    # Core content
    executive_summary: str
    detailed_analysis: str
    structured_content: Dict[str, Any]
    
    # Categorization
    main_categories: List[Dict[str, Any]]
    comparative_analysis: Optional[str] = None
    practical_applications: List[str] = None
    
    # Quality assessment
    synthesis_type: str = "comprehensive"
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    
    # Research insights
    key_findings: List[str] = None
    knowledge_gaps: List[str] = None
    future_directions: List[str] = None
    recommendations: List[str] = None
    
    # Evidence and sources
    source_quality_distribution: Dict[str, int] = None
    evidence_strength: str = "moderate"
    citation_coverage: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.practical_applications is None:
            self.practical_applications = []
        if self.key_findings is None:
            self.key_findings = []
        if self.knowledge_gaps is None:
            self.knowledge_gaps = []
        if self.future_directions is None:
            self.future_directions = []
        if self.recommendations is None:
            self.recommendations = []
        if self.source_quality_distribution is None:
            self.source_quality_distribution = {}
        if self.citation_coverage is None:
            self.citation_coverage = {}


class SynthesisTemplate:
    """Templates for different types of research synthesis."""
    
    @staticmethod
    def get_template(research_type: str, domain: str) -> Dict[str, Any]:
        """Get synthesis template based on research type and domain."""
        
        base_template = {
            "structure": {
                "executive_summary": "Concise overview of key findings",
                "main_sections": [],
                "comparative_analysis": "Compare and contrast different approaches",
                "practical_implications": "Real-world applications and implementations",
                "conclusion": "Synthesis of findings and future directions"
            },
            "content_guidelines": {
                "use_evidence": True,
                "include_citations": True,
                "structured_format": True,
                "practical_focus": True
            }
        }
        
        # Customize based on research type
        if research_type == "survey":
            base_template["structure"]["main_sections"] = [
                "Historical Development",
                "Current State of the Art",
                "Categorization and Taxonomy",
                "Comparative Analysis",
                "Applications and Use Cases",
                "Future Trends"
            ]
        elif research_type == "comparison":
            base_template["structure"]["main_sections"] = [
                "Compared Approaches",
                "Evaluation Criteria", 
                "Performance Analysis",
                "Strengths and Weaknesses",
                "Use Case Recommendations"
            ]
        elif research_type == "explanation":
            base_template["structure"]["main_sections"] = [
                "Fundamental Concepts",
                "Technical Details",
                "Examples and Applications",
                "Common Misconceptions",
                "Practical Guidance"
            ]
        elif research_type == "implementation":
            base_template["structure"]["main_sections"] = [
                "Implementation Approaches",
                "Technical Requirements",
                "Step-by-Step Guidance",
                "Common Challenges",
                "Best Practices"
            ]
        
        # Domain-specific customizations
        if domain in ["machine learning", "artificial intelligence", "deep learning"]:
            base_template["ml_specific"] = {
                "include_algorithms": True,
                "performance_metrics": True,
                "code_examples": True,
                "dataset_references": True
            }
        
        return base_template


class EnhancedAISynthesizer:
    """
    Enhanced AI synthesizer with structured output templates and comprehensive analysis.
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.template_engine = SynthesisTemplate()
        
        logger.info("🧬 Enhanced AI Synthesizer initialized with structured templates")
    
    async def synthesize_comprehensive_findings(self, 
                                              query: str,
                                              sources: List[EnhancedSource],
                                              query_analysis: Dict[str, Any],
                                              search_metadata: Dict[str, Any]) -> StructuredSynthesis:
        """
        Create comprehensive structured synthesis from sources.
        
        Args:
            query: Original research question
            sources: List of enhanced sources with quality metrics
            query_analysis: Analysis of the research query
            search_metadata: Metadata about the search process
            
        Returns:
            StructuredSynthesis with comprehensive analysis
        """
        logger.info(f"🧬 Starting comprehensive synthesis for: {query[:50]}...")
        
        if not sources:
            return self._create_no_sources_synthesis(query, query_analysis)
        
        # Get synthesis template
        research_type = query_analysis.get("research_type", "explanation")
        domain = query_analysis.get("domain_detected", "general")
        template = self.template_engine.get_template(research_type, domain)
        
        # Analyze source quality and distribution
        source_analysis = self._analyze_source_collection(sources)
        
        # Generate structured content sections
        structured_content = await self._generate_structured_content(
            query, sources, template, research_type, domain
        )
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            query, sources, structured_content, source_analysis
        )
        
        # Generate detailed analysis
        detailed_analysis = await self._generate_detailed_analysis(
            query, sources, structured_content, template
        )
        
        # Extract insights and gaps
        insights = await self._extract_research_insights(query, sources, structured_content)
        
        # Calculate quality scores
        confidence_score = self._calculate_confidence_score(sources, source_analysis)
        completeness_score = self._calculate_completeness_score(sources, insights)
        
        # Determine synthesis type
        synthesis_type = self._determine_synthesis_type(sources, source_analysis, research_type)
        
        result = StructuredSynthesis(
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            structured_content=structured_content,
            main_categories=structured_content.get("categories", []),
            comparative_analysis=structured_content.get("comparative_analysis"),
            practical_applications=structured_content.get("practical_applications", []),
            synthesis_type=synthesis_type,
            confidence_score=confidence_score,
            completeness_score=completeness_score,
            key_findings=insights.get("key_findings", []),
            knowledge_gaps=insights.get("knowledge_gaps", []),
            future_directions=insights.get("future_directions", []),
            recommendations=insights.get("recommendations", []),
            source_quality_distribution=source_analysis["quality_distribution"],
            evidence_strength=source_analysis["evidence_strength"],
            citation_coverage=source_analysis["citation_metrics"]
        )
        
        logger.info(f"🧬 Synthesis completed: {synthesis_type} (confidence: {confidence_score:.3f})")
        return result
    
    async def _generate_structured_content(self, 
                                         query: str,
                                         sources: List[EnhancedSource],
                                         template: Dict[str, Any],
                                         research_type: str,
                                         domain: str) -> Dict[str, Any]:
        """Generate structured content based on template."""
        
        # Prepare source information for analysis
        source_summaries = []
        for i, source in enumerate(sources[:15], 1):  # Limit for token management
            summary = {
                "id": i,
                "title": source.title,
                "authors": source.authors[:3],  # Limit authors
                "year": source.publication_year,
                "abstract": source.abstract[:300] if source.abstract else "",  # Limit abstract
                "quality_score": source.quality_score,
                "citations": source.metrics.citation_count if source.metrics else 0,
                "source_type": source.source_type
            }
            source_summaries.append(summary)
        
        structure_prompt = f"""Create a comprehensive structured analysis of the research findings:

RESEARCH QUERY: "{query}"
RESEARCH TYPE: {research_type}
DOMAIN: {domain}

AVAILABLE SOURCES ({len(sources)}):
{json.dumps(source_summaries[:10], indent=2)}

SYNTHESIS TEMPLATE:
{json.dumps(template["structure"], indent=2)}

Create a structured response in this JSON format:
{{
    "categories": [
        {{
            "name": "Category Name",
            "description": "What this category covers",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "supporting_sources": [1, 2, 3],
            "confidence": 0.8
        }}
    ],
    "comparative_analysis": "Detailed comparison of different approaches/methods/theories",
    "practical_applications": [
        {{
            "application": "Specific use case",
            "description": "How it's implemented",
            "benefits": ["Benefit 1", "Benefit 2"],
            "limitations": ["Limitation 1", "Limitation 2"]
        }}
    ],
    "technical_details": {{
        "algorithms": ["Algorithm 1", "Algorithm 2"],
        "methodologies": ["Method 1", "Method 2"],
        "performance_metrics": ["Metric 1", "Metric 2"]
    }},
    "evidence_quality": {{
        "strong_evidence": ["Finding with strong support"],
        "moderate_evidence": ["Finding with moderate support"],
        "limited_evidence": ["Finding with limited support"]
    }}
}}

GUIDELINES:
1. Base all content on the provided sources
2. Organize information into clear, logical categories
3. Include specific technical details when available
4. Compare and contrast different approaches
5. Highlight practical applications and real-world usage
6. Assess evidence quality honestly
7. Reference source IDs for traceability"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are an expert research synthesizer specializing in {domain}. Create comprehensive, structured analyses based on academic sources. Always respond with valid JSON."},
                    {"role": "user", "content": structure_prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"❌ Structured content generation failed: {e}")
            return self._create_fallback_structured_content(query, sources)
    
    async def _generate_executive_summary(self, 
                                        query: str,
                                        sources: List[EnhancedSource],
                                        structured_content: Dict[str, Any],
                                        source_analysis: Dict[str, Any]) -> str:
        """Generate executive summary of findings."""
        
        summary_prompt = f"""Create a concise executive summary of the research findings:

RESEARCH QUERY: "{query}"
TOTAL SOURCES: {len(sources)}
HIGH-QUALITY SOURCES: {source_analysis['high_quality_count']}

KEY CATEGORIES:
{chr(10).join([f"- {cat.get('name', 'Unknown')}: {cat.get('description', '')}" for cat in structured_content.get('categories', [])])}

EVIDENCE QUALITY:
- Strong evidence: {len(structured_content.get('evidence_quality', {}).get('strong_evidence', []))} findings
- Moderate evidence: {len(structured_content.get('evidence_quality', {}).get('moderate_evidence', []))} findings
- Limited evidence: {len(structured_content.get('evidence_quality', {}).get('limited_evidence', []))} findings

Create a 2-3 paragraph executive summary that:
1. Directly answers the research question
2. Highlights the most important findings
3. Summarizes the evidence quality and scope
4. Mentions any significant limitations or gaps

Write in clear, professional language suitable for researchers and practitioners."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, informative executive summaries of research findings."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Executive summary generation failed: {e}")
            return f"Research on '{query}' found {len(sources)} sources with varying levels of evidence quality. Detailed analysis follows."
    
    async def _generate_detailed_analysis(self, 
                                        query: str,
                                        sources: List[EnhancedSource],
                                        structured_content: Dict[str, Any],
                                        template: Dict[str, Any]) -> str:
        """Generate detailed analysis with full synthesis."""
        
        sections = template["structure"]["main_sections"]
        categories = structured_content.get("categories", [])
        
        analysis_prompt = f"""Create a comprehensive detailed analysis following this structure:

RESEARCH QUERY: "{query}"
REQUIRED SECTIONS: {sections}

AVAILABLE CONTENT:
{json.dumps({
    'categories': categories,
    'comparative_analysis': structured_content.get('comparative_analysis', ''),
    'practical_applications': structured_content.get('practical_applications', []),
    'technical_details': structured_content.get('technical_details', {})
}, indent=2)}

Create a detailed analysis with these sections:
{chr(10).join([f"{i+1}. {section}" for i, section in enumerate(sections)])}

For each section:
1. Provide comprehensive coverage based on the available sources
2. Include specific examples and technical details
3. Reference the evidence quality (strong/moderate/limited)
4. Compare different approaches where applicable
5. Maintain academic rigor while being accessible

Format as markdown with clear headings and subheadings."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst who creates comprehensive, well-structured analyses. Write in clear, academic style with proper organization."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Detailed analysis generation failed: {e}")
            return f"Detailed analysis of '{query}' based on {len(sources)} sources. Analysis could not be generated due to processing error."
    
    async def _extract_research_insights(self, 
                                       query: str,
                                       sources: List[EnhancedSource],
                                       structured_content: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract key insights, gaps, and future directions."""
        
        insights_prompt = f"""Analyze the research findings and extract key insights:

RESEARCH QUERY: "{query}"
STRUCTURED FINDINGS:
{json.dumps(structured_content, indent=2)[:3000]}...

Extract insights in this JSON format:
{{
    "key_findings": [
        "Most significant discovery or conclusion",
        "Important trend or pattern identified",
        "Notable consensus or disagreement in literature"
    ],
    "knowledge_gaps": [
        "Area lacking sufficient research",
        "Methodological limitation identified",
        "Unexplored aspect of the topic"
    ],
    "future_directions": [
        "Promising research direction",
        "Emerging trend to watch",
        "Technical advancement needed"
    ],
    "recommendations": [
        "Practical recommendation for practitioners",
        "Methodological recommendation for researchers",
        "Implementation guideline"
    ]
}}

Focus on:
1. What are the most important takeaways?
2. Where are the gaps in current knowledge?
3. What should be studied next?
4. What actions should practitioners take?"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting strategic insights from research findings. Always respond with valid JSON."},
                    {"role": "user", "content": insights_prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"❌ Insights extraction failed: {e}")
            return {
                "key_findings": [f"Analysis of {len(sources)} sources on {query}"],
                "knowledge_gaps": ["Unable to identify gaps due to processing error"],
                "future_directions": ["Further research needed"],
                "recommendations": ["Review individual sources for specific guidance"]
            }
    
    def _analyze_source_collection(self, sources: List[EnhancedSource]) -> Dict[str, Any]:
        """Analyze the quality and characteristics of the source collection."""
        if not sources:
            return {
                "high_quality_count": 0,
                "quality_distribution": {},
                "evidence_strength": "none",
                "citation_metrics": {},
                "temporal_coverage": {},
                "source_types": {}
            }
        
        # Quality distribution
        quality_ranges = {"high": 0, "medium": 0, "low": 0}
        citation_counts = []
        years = []
        source_types = {}
        
        for source in sources:
            # Quality categorization
            if source.quality_score >= 0.7:
                quality_ranges["high"] += 1
            elif source.quality_score >= 0.4:
                quality_ranges["medium"] += 1
            else:
                quality_ranges["low"] += 1
            
            # Citation data
            if source.metrics:
                citation_counts.append(source.metrics.citation_count)
            
            # Temporal data
            if source.publication_year:
                years.append(source.publication_year)
            
            # Source types
            source_types[source.source_type] = source_types.get(source.source_type, 0) + 1
        
        # Evidence strength assessment
        high_quality_ratio = quality_ranges["high"] / len(sources)
        if high_quality_ratio >= 0.6:
            evidence_strength = "strong"
        elif high_quality_ratio >= 0.3:
            evidence_strength = "moderate"
        else:
            evidence_strength = "limited"
        
        # Citation metrics
        citation_metrics = {}
        if citation_counts:
            citation_metrics = {
                "total_citations": sum(citation_counts),
                "average_citations": sum(citation_counts) / len(citation_counts),
                "highly_cited_papers": sum(1 for c in citation_counts if c > 100)
            }
        
        # Temporal coverage
        temporal_coverage = {}
        if years:
            temporal_coverage = {
                "earliest_year": min(years),
                "latest_year": max(years),
                "span_years": max(years) - min(years),
                "recent_papers": sum(1 for y in years if y >= 2020)
            }
        
        return {
            "high_quality_count": quality_ranges["high"],
            "quality_distribution": quality_ranges,
            "evidence_strength": evidence_strength,
            "citation_metrics": citation_metrics,
            "temporal_coverage": temporal_coverage,
            "source_types": source_types
        }
    
    def _calculate_confidence_score(self, 
                                  sources: List[EnhancedSource],
                                  source_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the synthesis."""
        if not sources:
            return 0.0
        
        # Factors contributing to confidence
        source_count_score = min(len(sources) / 20, 1.0)  # Normalize to 20 sources
        quality_score = source_analysis["high_quality_count"] / len(sources)
        
        evidence_strength_scores = {"strong": 1.0, "moderate": 0.7, "limited": 0.3, "none": 0.0}
        evidence_score = evidence_strength_scores.get(source_analysis["evidence_strength"], 0.0)
        
        # Citation factor
        citation_score = 0.5  # Default
        if source_analysis["citation_metrics"]:
            avg_citations = source_analysis["citation_metrics"].get("average_citations", 0)
            citation_score = min(avg_citations / 100, 1.0)  # Normalize to 100 citations
        
        # Weighted confidence calculation
        confidence = (
            source_count_score * 0.2 +
            quality_score * 0.3 +
            evidence_score * 0.3 +
            citation_score * 0.2
        )
        
        return round(confidence, 3)
    
    def _calculate_completeness_score(self, 
                                    sources: List[EnhancedSource],
                                    insights: Dict[str, List[str]]) -> float:
        """Calculate completeness score based on coverage."""
        if not sources:
            return 0.0
        
        # Check for different aspects of completeness
        source_diversity = len(set(source.source_type for source in sources)) / 3.0  # Max 3 types
        temporal_span = 1.0  # Default
        
        years = [s.publication_year for s in sources if s.publication_year]
        if years and len(set(years)) > 1:
            span = max(years) - min(years)
            temporal_span = min(span / 10, 1.0)  # Normalize to 10 years
        
        insight_coverage = len(insights.get("key_findings", [])) / 5.0  # Normalize to 5 findings
        
        completeness = (source_diversity * 0.4 + temporal_span * 0.3 + insight_coverage * 0.3)
        return round(min(completeness, 1.0), 3)
    
    def _determine_synthesis_type(self, 
                                sources: List[EnhancedSource],
                                source_analysis: Dict[str, Any],
                                research_type: str) -> str:
        """Determine the type of synthesis achieved."""
        if not sources:
            return "no_sources"
        
        high_quality_ratio = source_analysis["high_quality_count"] / len(sources)
        evidence_strength = source_analysis["evidence_strength"]
        
        if len(sources) >= 15 and high_quality_ratio >= 0.6 and evidence_strength == "strong":
            return "comprehensive"
        elif len(sources) >= 10 and high_quality_ratio >= 0.4 and evidence_strength in ["strong", "moderate"]:
            return "substantial"
        elif len(sources) >= 5 and evidence_strength in ["moderate", "strong"]:
            return "partial"
        else:
            return "limited"
    
    def _create_no_sources_synthesis(self, query: str, query_analysis: Dict[str, Any]) -> StructuredSynthesis:
        """Create synthesis when no sources are available."""
        return StructuredSynthesis(
            executive_summary=f"No relevant sources were found for the research query: '{query}'. This may indicate that the topic is very new, highly specialized, or requires different search strategies.",
            detailed_analysis=f"# Research Analysis: {query}\n\nUnfortunately, the comprehensive search did not yield relevant academic sources for this query. This could be due to several factors:\n\n1. The topic may be too new for academic publication\n2. Different terminology might be used in the literature\n3. The research area may be highly specialized\n4. Additional databases or search strategies may be needed\n\n## Recommendations\n\n- Try alternative search terms or broader queries\n- Consult specialized databases in the relevant domain\n- Look for industry reports or technical documentation\n- Consider reaching out to experts in the field",
            structured_content={
                "categories": [],
                "evidence_quality": {"strong_evidence": [], "moderate_evidence": [], "limited_evidence": []},
                "practical_applications": []
            },
            synthesis_type="no_sources",
            confidence_score=0.0,
            completeness_score=0.0,
            key_findings=["No sources found for analysis"],
            knowledge_gaps=["Entire research area needs exploration"],
            recommendations=["Expand search strategy", "Try alternative databases", "Consult domain experts"]
        )
    
    def _create_fallback_structured_content(self, query: str, sources: List[EnhancedSource]) -> Dict[str, Any]:
        """Create fallback structured content when AI generation fails."""
        return {
            "categories": [
                {
                    "name": "General Findings",
                    "description": f"Analysis of {len(sources)} sources related to: {query}",
                    "key_points": [f"Found {len(sources)} relevant sources"],
                    "supporting_sources": list(range(1, min(len(sources) + 1, 6))),
                    "confidence": 0.5
                }
            ],
            "comparative_analysis": "Unable to generate comparative analysis due to processing error.",
            "practical_applications": [
                {
                    "application": "General application",
                    "description": "Applications identified in the source material",
                    "benefits": ["Refer to individual sources"],
                    "limitations": ["Analysis limited due to processing error"]
                }
            ],
            "evidence_quality": {
                "strong_evidence": [],
                "moderate_evidence": [f"Analysis of {len(sources)} sources"],
                "limited_evidence": ["Processing limitations affected analysis"]
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_enhanced_synthesizer():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Please set OPENAI_API_KEY for testing")
            return
        
        synthesizer = EnhancedAISynthesizer(api_key)
        
        # Mock sources for testing
        mock_sources = [
            EnhancedSource(
                title="ReLU: A Simple and Effective Activation Function",
                authors=["Author A", "Author B"],
                abstract="ReLU activation function has become widely used...",
                url="https://example.com/1",
                publication_year=2020,
                source_type="semantic",
                quality_score=0.8
            ),
            EnhancedSource(
                title="Comparative Study of Activation Functions",
                authors=["Author C", "Author D"],
                abstract="We compare various activation functions including sigmoid, tanh, and ReLU...",
                url="https://example.com/2",
                publication_year=2021,
                source_type="arxiv",
                quality_score=0.7
            )
        ]
        
        # Mock query analysis
        query = "types of activation functions in neural networks"
        mock_analysis = {
            "research_type": "survey",
            "domain_detected": "machine learning"
        }
        
        print(f"🧪 Testing Enhanced Synthesizer")
        print("=" * 70)
        print(f"Query: {query}")
        print(f"Sources: {len(mock_sources)}")
        print("=" * 70)
        
        result = await synthesizer.synthesize_comprehensive_findings(
            query=query,
            sources=mock_sources,
            query_analysis=mock_analysis,
            search_metadata={}
        )
        
        print(f"\n📊 Synthesis Results:")
        print(f"Synthesis Type: {result.synthesis_type}")
        print(f"Confidence Score: {result.confidence_score}")
        print(f"Completeness Score: {result.completeness_score}")
        print(f"Categories: {len(result.main_categories)}")
        print(f"Key Findings: {len(result.key_findings)}")
        print(f"Knowledge Gaps: {len(result.knowledge_gaps)}")
        
        print(f"\n📝 Executive Summary:")
        print(result.executive_summary[:300] + "...")
    
    # Run test
    asyncio.run(test_enhanced_synthesizer())
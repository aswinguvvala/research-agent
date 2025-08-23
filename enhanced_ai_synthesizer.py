"""
Enhanced AI Synthesizer
Structured synthesis engine that creates comprehensive research analysis with categorized findings.
"""

import openai
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class SynthesisSection:
    """A section of the research synthesis."""
    title: str
    content: str
    evidence_strength: str  # 'strong', 'moderate', 'limited'
    source_count: int
    confidence_score: float
    key_findings: List[str]
    supporting_sources: List[str]


@dataclass
class StructuredSynthesis:
    """Complete structured synthesis of research findings."""
    executive_summary: str
    key_findings: List[str]
    thematic_sections: List[SynthesisSection]
    comparative_analysis: str
    methodological_insights: str
    practical_applications: str
    limitations_and_gaps: str
    future_directions: str
    confidence_assessment: Dict[str, float]
    synthesis_metadata: Dict[str, Any]


@dataclass
class ThematicCluster:
    """A cluster of sources around a common theme."""
    theme: str
    sources: List[Dict[str, Any]]
    relevance_scores: List[float]
    theme_strength: float
    key_concepts: List[str]


class EnhancedAISynthesizer:
    """
    Enhanced AI synthesizer that creates structured research synthesis
    with categorized findings, evidence assessment, and comprehensive analysis.
    """
    
    def __init__(self, openai_api_key: str, debug_mode: bool = False):
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.debug_mode = debug_mode
        
        # Synthesis configuration
        self.max_sources_per_section = 8
        self.min_sources_for_strong_evidence = 5
        self.min_sources_for_moderate_evidence = 3
        
        logger.info("📝 Enhanced AI Synthesizer initialized")
    
    async def create_structured_synthesis(self, 
                                        query: str,
                                        sources: List[Dict[str, Any]],
                                        relevance_assessments: Optional[List[Dict[str, Any]]] = None) -> StructuredSynthesis:
        """
        Create comprehensive structured synthesis from research sources.
        
        Args:
            query: Original research query
            sources: List of source documents
            relevance_assessments: Optional relevance assessments for sources
            
        Returns:
            StructuredSynthesis with comprehensive analysis
        """
        logger.info(f"📝 Creating structured synthesis for: {query}")
        logger.info(f"📊 Processing {len(sources)} sources")
        
        try:
            # Step 1: Create source index for citations
            source_index = self._create_source_index(sources)
            
            # Step 2: Identify thematic clusters
            thematic_clusters = await self._identify_thematic_clusters(sources)
            
            # Step 3: Generate executive summary with inline citations
            executive_summary = await self._generate_executive_summary_with_citations(query, sources, thematic_clusters, source_index)
            
            # Step 4: Extract key findings with citations
            key_findings = await self._extract_key_findings_with_citations(query, sources, thematic_clusters, source_index)
            
            # Step 5: Create thematic sections with inline citations
            thematic_sections = await self._create_thematic_sections_with_citations(thematic_clusters, source_index)
            
            # Step 6: Generate comparative analysis with citations
            comparative_analysis = await self._generate_comparative_analysis_with_citations(query, sources, thematic_clusters, source_index)
            
            # Step 7: Extract methodological insights with citations
            methodological_insights = await self._extract_methodological_insights_with_citations(sources, source_index)
            
            # Step 8: Identify practical applications with citations
            practical_applications = await self._identify_practical_applications_with_citations(query, sources, source_index)
            
            # Step 9: Analyze limitations and gaps with citations
            limitations_and_gaps = await self._analyze_limitations_and_gaps_with_citations(sources, thematic_clusters, source_index)
            
            # Step 10: Suggest future directions with citations
            future_directions = await self._suggest_future_directions_with_citations(query, sources, limitations_and_gaps, source_index)
            
            # Step 11: Assess confidence
            confidence_assessment = self._assess_synthesis_confidence(sources, thematic_clusters, thematic_sections)
            
            # Create synthesis metadata
            synthesis_metadata = {
                'total_sources': len(sources),
                'thematic_clusters': len(thematic_clusters),
                'synthesis_timestamp': datetime.now().isoformat(),
                'query_analyzed': query,
                'avg_source_year': self._calculate_avg_year(sources),
                'source_types': self._analyze_source_types(sources),
                'citation_style': 'inline_numbered'
            }
            
            synthesis = StructuredSynthesis(
                executive_summary=executive_summary,
                key_findings=key_findings,
                thematic_sections=thematic_sections,
                comparative_analysis=comparative_analysis,
                methodological_insights=methodological_insights,
                practical_applications=practical_applications,
                limitations_and_gaps=limitations_and_gaps,
                future_directions=future_directions,
                confidence_assessment=confidence_assessment,
                synthesis_metadata=synthesis_metadata
            )
            
            logger.info(f"✅ Structured synthesis completed with {len(thematic_sections)} thematic sections and inline citations")
            return synthesis
            
        except Exception as e:
            logger.error(f"❌ Synthesis creation failed: {e}")
            return self._create_fallback_synthesis(query, sources)
    
    async def _identify_thematic_clusters(self, sources: List[Dict[str, Any]]) -> List[ThematicCluster]:
        """Identify thematic clusters in the research sources."""
        try:
            # Extract key information from sources
            source_summaries = []
            for i, source in enumerate(sources[:20]):  # Limit for efficiency
                title = source.get('title', 'Unknown Title')
                abstract = source.get('abstract', 'No abstract')[:300]
                source_summaries.append(f"{i+1}. {title}\nAbstract: {abstract}")
            
            sources_text = "\n\n".join(source_summaries)
            
            prompt = f"""
            Analyze these research sources and identify 3-6 major thematic clusters.
            
            Sources:
            {sources_text}
            
            For each cluster, provide:
            1. Theme name (concise and descriptive)
            2. Source numbers that belong to this cluster
            3. Key concepts associated with this theme
            
            Format as:
            THEME: [Theme Name]
            SOURCES: [comma-separated source numbers]
            CONCEPTS: [key concept 1, key concept 2, etc.]
            
            ---
            
            Focus on grouping sources that share similar research focus, methodology, or findings.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            clusters_text = response.choices[0].message.content.strip()
            clusters = self._parse_thematic_clusters(clusters_text, sources)
            
            logger.info(f"🔍 Identified {len(clusters)} thematic clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Thematic clustering failed: {e}")
            # Fallback: create simple clusters based on years
            return self._create_fallback_clusters(sources)
    
    def _create_source_index(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Create an index mapping source titles to citation numbers."""
        source_index = {}
        for i, source in enumerate(sources):
            source_title = source.get('title', f'Source {i+1}')
            source_index[source_title] = i + 1
        return source_index
    
    def _get_citation_number(self, source: Dict[str, Any], source_index: Dict[str, int]) -> int:
        """Get citation number for a source."""
        source_title = source.get('title', 'Unknown')
        return source_index.get(source_title, 1)
    
    def _format_inline_citation(self, source: Dict[str, Any], source_index: Dict[str, int]) -> str:
        """Format an inline citation for a source."""
        citation_num = self._get_citation_number(source, source_index)
        return f"[{citation_num}]"
    
    def _format_multiple_citations(self, sources: List[Dict[str, Any]], source_index: Dict[str, int]) -> str:
        """Format multiple inline citations."""
        citation_nums = [self._get_citation_number(source, source_index) for source in sources]
        citation_nums = sorted(list(set(citation_nums)))  # Remove duplicates and sort
        
        if len(citation_nums) == 1:
            return f"[{citation_nums[0]}]"
        elif len(citation_nums) == 2:
            return f"[{citation_nums[0]}, {citation_nums[1]}]"
        elif len(citation_nums) > 2:
            return f"[{citation_nums[0]}-{citation_nums[-1]}]"
        return ""
    
    async def _generate_executive_summary_with_citations(self, 
                                        query: str, 
                                        sources: List[Dict[str, Any]], 
                                        clusters: List[ThematicCluster],
                                        source_index: Dict[str, int]) -> str:
        """Generate executive summary with inline citations."""
        try:
            # Get cluster summaries with citations
            cluster_summaries = []
            for cluster in clusters:
                cluster_citations = self._format_multiple_citations(cluster.sources[:3], source_index)
                cluster_summaries.append(f"- {cluster.theme}: {len(cluster.sources)} sources covering {', '.join(cluster.key_concepts[:3])} {cluster_citations}")
            
            clusters_overview = "\n".join(cluster_summaries)
            
            # Get top sources for context with citations
            top_sources = sorted(sources, key=lambda x: x.get('citations', 0), reverse=True)[:5]
            top_sources_text = "\n".join([f"- {s.get('title', 'Unknown')} {self._format_inline_citation(s, source_index)}" for s in top_sources])
            
            prompt = f"""
            Create a comprehensive executive summary for research on: "{query}"
            
            Research Overview:
            - Total sources analyzed: {len(sources)}
            - Thematic clusters identified:
            {clusters_overview}
            
            Key high-impact sources:
            {top_sources_text}
            
            IMPORTANT: Include inline citations using [number] format throughout your summary. Use the citation numbers provided above.
            
            Create an executive summary that includes:
            1. Overall research landscape overview with citations
            2. Key themes and patterns identified with supporting citations
            3. Most significant findings or conclusions with citations
            4. State of the field assessment with citations
            5. Notable trends or developments with citations
            
            Example format: "Research shows significant advances in this field [1, 3]. Multiple studies indicate that... [2, 4]."
            
            Keep it comprehensive but concise (3-4 paragraphs) with proper inline citations.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Executive summary with citations generation failed: {e}")
            return f"Executive summary for '{query}': Analysis of {len(sources)} sources reveals multiple research themes and approaches in this domain [1-{len(sources)}]."

    async def _generate_executive_summary(self, 
                                        query: str, 
                                        sources: List[Dict[str, Any]], 
                                        clusters: List[ThematicCluster]) -> str:
        """Generate executive summary of the research findings."""
        try:
            # Get cluster summaries
            cluster_summaries = []
            for cluster in clusters:
                cluster_summaries.append(f"- {cluster.theme}: {len(cluster.sources)} sources covering {', '.join(cluster.key_concepts[:3])}")
            
            clusters_overview = "\n".join(cluster_summaries)
            
            # Get top sources for context
            top_sources = sorted(sources, key=lambda x: x.get('citations', 0), reverse=True)[:5]
            top_sources_text = "\n".join([f"- {s.get('title', 'Unknown')}" for s in top_sources])
            
            prompt = f"""
            Create a comprehensive executive summary for research on: "{query}"
            
            Research Overview:
            - Total sources analyzed: {len(sources)}
            - Thematic clusters identified:
            {clusters_overview}
            
            Key high-impact sources:
            {top_sources_text}
            
            Create an executive summary that includes:
            1. Overall research landscape overview
            2. Key themes and patterns identified
            3. Most significant findings or conclusions
            4. State of the field assessment
            5. Notable trends or developments
            
            Keep it comprehensive but concise (3-4 paragraphs).
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return f"Executive summary for '{query}': Analysis of {len(sources)} sources reveals multiple research themes and approaches in this domain."
    
    async def _extract_key_findings_with_citations(self, 
                                  query: str, 
                                  sources: List[Dict[str, Any]], 
                                  clusters: List[ThematicCluster],
                                  source_index: Dict[str, int]) -> List[str]:
        """Extract key findings with inline citations."""
        try:
            # Select representative sources from each cluster with citations
            representative_sources = []
            for cluster in clusters:
                # Get highest cited source from each cluster
                cluster_sources = sorted(cluster.sources, key=lambda x: x.get('citations', 0), reverse=True)
                if cluster_sources:
                    source = cluster_sources[0]
                    title = source.get('title', 'Unknown')
                    abstract = source.get('abstract', 'No abstract')[:200]
                    citation = self._format_inline_citation(source, source_index)
                    representative_sources.append(f"[{cluster.theme}] {title} {citation}: {abstract}")
            
            sources_context = "\n\n".join(representative_sources)
            
            prompt = f"""
            Extract 5-8 key findings from this research on: "{query}"
            
            Representative sources by theme:
            {sources_context}
            
            IMPORTANT: Include inline citations using [number] format for each finding. Use the citation numbers provided above.
            
            Extract key findings that are:
            1. Significant and well-supported with citations
            2. Relevant to the research query with evidence
            3. Represent different aspects/themes with citations
            4. Based on multiple sources when possible
            
            Format as numbered list of concise, specific findings WITH inline citations.
            Example: "1. Research shows significant improvement in performance [1, 3]."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            findings_text = response.choices[0].message.content.strip()
            findings = self._parse_numbered_list(findings_text)
            
            return findings[:8]  # Limit to 8 findings
            
        except Exception as e:
            logger.error(f"Key findings with citations extraction failed: {e}")
            return [f"Research on {query} shows multiple approaches and methodologies in the field [1-{len(sources)}]."]

    async def _extract_key_findings(self, 
                                  query: str, 
                                  sources: List[Dict[str, Any]], 
                                  clusters: List[ThematicCluster]) -> List[str]:
        """Extract key findings from the research sources."""
        try:
            # Select representative sources from each cluster
            representative_sources = []
            for cluster in clusters:
                # Get highest cited source from each cluster
                cluster_sources = sorted(cluster.sources, key=lambda x: x.get('citations', 0), reverse=True)
                if cluster_sources:
                    source = cluster_sources[0]
                    title = source.get('title', 'Unknown')
                    abstract = source.get('abstract', 'No abstract')[:200]
                    representative_sources.append(f"[{cluster.theme}] {title}: {abstract}")
            
            sources_context = "\n\n".join(representative_sources)
            
            prompt = f"""
            Extract 5-8 key findings from this research on: "{query}"
            
            Representative sources by theme:
            {sources_context}
            
            Extract key findings that are:
            1. Significant and well-supported
            2. Relevant to the research query
            3. Represent different aspects/themes
            4. Based on multiple sources when possible
            
            Format as numbered list of concise, specific findings.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            findings_text = response.choices[0].message.content.strip()
            findings = self._parse_numbered_list(findings_text)
            
            return findings[:8]  # Limit to 8 findings
            
        except Exception as e:
            logger.error(f"Key findings extraction failed: {e}")
            return [f"Research on {query} shows multiple approaches and methodologies in the field."]
    
    async def _create_thematic_sections_with_citations(self, clusters: List[ThematicCluster], source_index: Dict[str, int]) -> List[SynthesisSection]:
        """Create detailed sections for each thematic cluster with inline citations."""
        sections = []
        
        for cluster in clusters:
            try:
                # Prepare source information with citations
                source_details = []
                for source in cluster.sources[:self.max_sources_per_section]:
                    title = source.get('title', 'Unknown')
                    abstract = source.get('abstract', 'No abstract')[:250]
                    year = source.get('year', 'Unknown')
                    citations = source.get('citations', 0)
                    citation_num = self._format_inline_citation(source, source_index)
                    source_details.append(f"- {title} {citation_num} ({year}, {citations} citations): {abstract}")
                
                sources_text = "\n".join(source_details)
                
                prompt = f"""
                Create a comprehensive analysis for the theme: "{cluster.theme}"
                
                Sources in this cluster:
                {sources_text}
                
                IMPORTANT: Include inline citations using [number] format throughout your analysis. Use the citation numbers provided above.
                
                Create a detailed section that includes:
                1. Overview of this research theme with citations
                2. Key approaches and methodologies with citations
                3. Main findings and insights with citations
                4. Notable patterns or trends with citations
                5. Specific evidence from the sources with citations
                
                Focus on synthesis rather than just summarizing individual papers.
                Make it scholarly and evidence-based with proper inline citations.
                Example: "Studies demonstrate significant improvements [1, 2]. Research by Smith et al. shows... [3]."
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=900,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                
                # Assess evidence strength
                evidence_strength = self._assess_evidence_strength(cluster)
                
                # Extract key findings for this section with citations
                section_findings = await self._extract_section_findings_with_citations(cluster, content, source_index)
                
                # Get supporting source titles with citations
                supporting_sources = [f"{s.get('title', 'Unknown')[:50]} {self._format_inline_citation(s, source_index)}" for s in cluster.sources[:5]]
                
                # Calculate confidence
                confidence_score = self._calculate_section_confidence(cluster)
                
                section = SynthesisSection(
                    title=cluster.theme,
                    content=content,
                    evidence_strength=evidence_strength,
                    source_count=len(cluster.sources),
                    confidence_score=confidence_score,
                    key_findings=section_findings,
                    supporting_sources=supporting_sources
                )
                
                sections.append(section)
                
            except Exception as e:
                logger.error(f"Section creation with citations failed for {cluster.theme}: {e}")
                continue
        
        return sections

    async def _create_thematic_sections(self, clusters: List[ThematicCluster]) -> List[SynthesisSection]:
        """Create detailed sections for each thematic cluster."""
        sections = []
        
        for cluster in clusters:
            try:
                # Prepare source information
                source_details = []
                for source in cluster.sources[:self.max_sources_per_section]:
                    title = source.get('title', 'Unknown')
                    abstract = source.get('abstract', 'No abstract')[:250]
                    year = source.get('year', 'Unknown')
                    citations = source.get('citations', 0)
                    source_details.append(f"- {title} ({year}, {citations} citations): {abstract}")
                
                sources_text = "\n".join(source_details)
                
                prompt = f"""
                Create a comprehensive analysis for the theme: "{cluster.theme}"
                
                Sources in this cluster:
                {sources_text}
                
                Create a detailed section that includes:
                1. Overview of this research theme
                2. Key approaches and methodologies
                3. Main findings and insights
                4. Notable patterns or trends
                5. Specific evidence from the sources
                
                Focus on synthesis rather than just summarizing individual papers.
                Make it scholarly and evidence-based.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                
                # Assess evidence strength
                evidence_strength = self._assess_evidence_strength(cluster)
                
                # Extract key findings for this section
                section_findings = await self._extract_section_findings(cluster, content)
                
                # Get supporting source titles
                supporting_sources = [s.get('title', 'Unknown')[:50] for s in cluster.sources[:5]]
                
                # Calculate confidence
                confidence_score = self._calculate_section_confidence(cluster)
                
                section = SynthesisSection(
                    title=cluster.theme,
                    content=content,
                    evidence_strength=evidence_strength,
                    source_count=len(cluster.sources),
                    confidence_score=confidence_score,
                    key_findings=section_findings,
                    supporting_sources=supporting_sources
                )
                
                sections.append(section)
                
            except Exception as e:
                logger.error(f"Section creation failed for {cluster.theme}: {e}")
                continue
        
        return sections
    
    async def _extract_section_findings_with_citations(self, cluster: ThematicCluster, content: str, source_index: Dict[str, int]) -> List[str]:
        """Extract key findings from a section's content with citations."""
        try:
            # Get cluster source citations
            cluster_citations = self._format_multiple_citations(cluster.sources[:3], source_index)
            
            prompt = f"""
            Extract 2-4 key findings from this research section about "{cluster.theme}":
            
            {content[:800]}
            
            IMPORTANT: Include inline citations using [number] format for findings. 
            Available sources for this cluster: {cluster_citations}
            
            Return as a numbered list of specific, evidence-based findings WITH citations.
            Example: "1. Studies show significant correlation between X and Y {cluster_citations}."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.3
            )
            
            findings_text = response.choices[0].message.content.strip()
            return self._parse_numbered_list(findings_text)[:4]
            
        except Exception as e:
            logger.warning(f"Section findings with citations extraction failed: {e}")
            cluster_citations = self._format_multiple_citations(cluster.sources[:2], source_index)
            return [f"Key findings related to {cluster.theme} {cluster_citations}"]
    
    async def _generate_comparative_analysis_with_citations(self, 
                                           query: str, 
                                           sources: List[Dict[str, Any]], 
                                           clusters: List[ThematicCluster],
                                           source_index: Dict[str, int]) -> str:
        """Generate comparative analysis with inline citations."""
        try:
            # Get cluster overviews with citations
            cluster_overviews = []
            for cluster in clusters:
                key_concepts = ", ".join(cluster.key_concepts[:3])
                cluster_citations = self._format_multiple_citations(cluster.sources[:3], source_index)
                cluster_overviews.append(f"- {cluster.theme} ({len(cluster.sources)} sources): {key_concepts} {cluster_citations}")
            
            clusters_text = "\n".join(cluster_overviews)
            
            prompt = f"""
            Create a comparative analysis for research on: "{query}"
            
            Research themes identified:
            {clusters_text}
            
            IMPORTANT: Include inline citations using [number] format throughout your analysis. Use the citation numbers provided above.
            
            Provide a comparative analysis that:
            1. Compares different approaches across themes with citations
            2. Identifies convergent vs. divergent findings with citations
            3. Discusses methodological differences with citations
            4. Highlights complementary insights with citations
            5. Notes any conflicting results or perspectives with citations
            
            Focus on synthesizing across themes rather than within themes.
            Example: "While Theme A studies show X [1, 2], Theme B research indicates Y [3, 4]."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Comparative analysis with citations generation failed: {e}")
            return "Comparative analysis reveals multiple approaches and perspectives in the research domain [1-" + str(len(sources)) + "]."

    async def _generate_comparative_analysis(self, 
                                           query: str, 
                                           sources: List[Dict[str, Any]], 
                                           clusters: List[ThematicCluster]) -> str:
        """Generate comparative analysis across different approaches."""
        try:
            # Get cluster overviews
            cluster_overviews = []
            for cluster in clusters:
                key_concepts = ", ".join(cluster.key_concepts[:3])
                cluster_overviews.append(f"- {cluster.theme} ({len(cluster.sources)} sources): {key_concepts}")
            
            clusters_text = "\n".join(cluster_overviews)
            
            prompt = f"""
            Create a comparative analysis for research on: "{query}"
            
            Research themes identified:
            {clusters_text}
            
            Provide a comparative analysis that:
            1. Compares different approaches across themes
            2. Identifies convergent vs. divergent findings
            3. Discusses methodological differences
            4. Highlights complementary insights
            5. Notes any conflicting results or perspectives
            
            Focus on synthesizing across themes rather than within themes.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Comparative analysis generation failed: {e}")
            return "Comparative analysis reveals multiple approaches and perspectives in the research domain."
    
    async def _extract_methodological_insights_with_citations(self, sources: List[Dict[str, Any]], source_index: Dict[str, int]) -> str:
        """Extract insights about research methodologies with citations."""
        try:
            # Extract methodology information from abstracts with citations
            methodology_info = []
            for source in sources[:15]:  # Limit for efficiency
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', '')
                citation = self._format_inline_citation(source, source_index)
                
                # Look for methodology keywords in abstracts
                method_keywords = ['method', 'approach', 'technique', 'algorithm', 'framework', 'model', 'analysis']
                if any(keyword in abstract.lower() for keyword in method_keywords):
                    methodology_info.append(f"- {title} {citation}: {abstract[:150]}...")
            
            methods_text = "\n".join(methodology_info[:10])
            
            prompt = f"""
            Analyze the methodological approaches used in these research sources:
            
            {methods_text}
            
            IMPORTANT: Include inline citations using [number] format throughout your analysis. Use the citation numbers provided above.
            
            Provide insights about:
            1. Common methodological approaches with citations
            2. Innovative or novel methods with citations
            3. Methodological trends with citations
            4. Strengths and limitations of approaches with citations
            5. Methodological gaps or opportunities with citations
            
            Focus on the research methods and approaches rather than findings.
            Example: "Common approaches include machine learning methods [1, 3] and statistical analysis [2, 5]."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Methodological insights with citations extraction failed: {e}")
            return f"Various methodological approaches are employed across the research domain [1-{len(sources)}]."
    
    async def _identify_practical_applications_with_citations(self, query: str, sources: List[Dict[str, Any]], source_index: Dict[str, int]) -> str:
        """Identify practical applications with citations."""
        try:
            # Extract application-related content with citations
            application_content = []
            for source in sources[:12]:
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', '')
                citation = self._format_inline_citation(source, source_index)
                
                # Look for application keywords
                app_keywords = ['application', 'practical', 'implementation', 'deployment', 'real-world', 'industry']
                if any(keyword in abstract.lower() for keyword in app_keywords):
                    application_content.append(f"- {title} {citation}: {abstract[:150]}...")
            
            if not application_content:
                # Use general sources if no specific application content found
                application_content = [f"- {s.get('title', 'Unknown')} {self._format_inline_citation(s, source_index)}: {s.get('abstract', '')[:150]}..." 
                                     for s in sources[:8]]
            
            apps_text = "\n".join(application_content[:8])
            
            prompt = f"""
            Identify practical applications and implications for: "{query}"
            
            Based on these sources:
            {apps_text}
            
            IMPORTANT: Include inline citations using [number] format throughout your analysis. Use the citation numbers provided above.
            
            Discuss:
            1. Current practical applications with citations
            2. Potential future applications with citations
            3. Implementation considerations with citations
            4. Real-world impact and benefits with citations
            5. Challenges for practical deployment with citations
            
            Focus on actionable insights and practical value.
            Example: "Current applications include healthcare systems [1] and financial modeling [3, 5]."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Practical applications with citations identification failed: {e}")
            return f"The research has various potential applications in both academic and industry contexts [1-{len(sources)}]."
    
    async def _analyze_limitations_and_gaps_with_citations(self, 
                                          sources: List[Dict[str, Any]], 
                                          clusters: List[ThematicCluster],
                                          source_index: Dict[str, int]) -> str:
        """Analyze limitations and research gaps with citations."""
        try:
            # Analyze coverage gaps with citations
            covered_themes = [cluster.theme for cluster in clusters]
            
            # Get recent vs older research distribution
            current_year = datetime.now().year
            recent_sources = [s for s in sources if s.get('year', 0) >= current_year - 3]
            older_sources = [s for s in sources if s.get('year', 0) < current_year - 3]
            
            # Sample sources for limitation analysis with citations
            sample_sources = []
            for source in sources[:10]:
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', '')[:200]
                year = source.get('year', 'Unknown')
                citation = self._format_inline_citation(source, source_index)
                sample_sources.append(f"- {title} {citation} ({year}): {abstract}")
            
            sources_sample = "\n".join(sample_sources)
            
            prompt = f"""
            Analyze limitations and research gaps based on this research analysis:
            
            Covered themes: {', '.join(covered_themes)}
            Recent sources: {len(recent_sources)}/{len(sources)}
            
            Sample sources:
            {sources_sample}
            
            IMPORTANT: Include inline citations using [number] format throughout your analysis. Use the citation numbers provided above.
            
            Identify:
            1. Methodological limitations with citations
            2. Coverage gaps or blind spots with citations
            3. Temporal limitations (recency of research) with citations
            4. Scope limitations with citations
            5. Areas needing more research with citations
            
            Be specific and constructive in identifying gaps.
            Example: "Limited sample sizes in current studies [2, 4] suggest need for larger scale research."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Limitations analysis with citations failed: {e}")
            return f"Further research is needed to address methodological and coverage limitations [1-{len(sources)}]."
    
    async def _suggest_future_directions_with_citations(self, 
                                       query: str, 
                                       sources: List[Dict[str, Any]], 
                                       limitations: str,
                                       source_index: Dict[str, int]) -> str:
        """Suggest future research directions with citations."""
        try:
            # Get recent high-impact sources for trend analysis with citations
            recent_sources = [s for s in sources if s.get('year', 0) >= datetime.now().year - 2]
            high_impact = sorted(sources, key=lambda x: x.get('citations', 0), reverse=True)[:5]
            
            recent_titles = [f"{s.get('title', 'Unknown')} {self._format_inline_citation(s, source_index)}" for s in recent_sources[:5]]
            impact_titles = [f"{s.get('title', 'Unknown')} {self._format_inline_citation(s, source_index)}" for s in high_impact]
            
            prompt = f"""
            Suggest future research directions for: "{query}"
            
            Based on current research state and identified limitations:
            {limitations[:500]}
            
            Recent research trends: {', '.join(recent_titles)}
            High-impact work: {', '.join(impact_titles)}
            
            IMPORTANT: Include inline citations using [number] format throughout your suggestions. Use the citation numbers provided above.
            
            Suggest future directions that:
            1. Address identified gaps and limitations with citations
            2. Build on current strengths with citations
            3. Explore emerging opportunities with citations
            4. Consider interdisciplinary approaches with citations
            5. Have practical significance with citations
            
            Provide specific, actionable research directions.
            Example: "Building on recent advances [1, 3], future research should explore scalability challenges."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Future directions with citations suggestion failed: {e}")
            return f"Future research should address current limitations and explore emerging opportunities in the field [1-{len(sources)}]."

    async def _extract_methodological_insights(self, sources: List[Dict[str, Any]]) -> str:
        """Extract insights about research methodologies used."""
        try:
            # Extract methodology information from abstracts
            methodology_info = []
            for source in sources[:15]:  # Limit for efficiency
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', '')
                
                # Look for methodology keywords in abstracts
                method_keywords = ['method', 'approach', 'technique', 'algorithm', 'framework', 'model', 'analysis']
                if any(keyword in abstract.lower() for keyword in method_keywords):
                    methodology_info.append(f"- {title}: {abstract[:150]}...")
            
            methods_text = "\n".join(methodology_info[:10])
            
            prompt = f"""
            Analyze the methodological approaches used in these research sources:
            
            {methods_text}
            
            Provide insights about:
            1. Common methodological approaches
            2. Innovative or novel methods
            3. Methodological trends
            4. Strengths and limitations of approaches
            5. Methodological gaps or opportunities
            
            Focus on the research methods and approaches rather than findings.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Methodological insights extraction failed: {e}")
            return "Various methodological approaches are employed across the research domain."
    
    async def _identify_practical_applications(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Identify practical applications and implications."""
        try:
            # Extract application-related content
            application_content = []
            for source in sources[:12]:
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', '')
                
                # Look for application keywords
                app_keywords = ['application', 'practical', 'implementation', 'deployment', 'real-world', 'industry']
                if any(keyword in abstract.lower() for keyword in app_keywords):
                    application_content.append(f"- {title}: {abstract[:150]}...")
            
            if not application_content:
                # Use general sources if no specific application content found
                application_content = [f"- {s.get('title', 'Unknown')}: {s.get('abstract', '')[:150]}..." 
                                     for s in sources[:8]]
            
            apps_text = "\n".join(application_content[:8])
            
            prompt = f"""
            Identify practical applications and implications for: "{query}"
            
            Based on these sources:
            {apps_text}
            
            Discuss:
            1. Current practical applications
            2. Potential future applications
            3. Implementation considerations
            4. Real-world impact and benefits
            5. Challenges for practical deployment
            
            Focus on actionable insights and practical value.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Practical applications identification failed: {e}")
            return "The research has various potential applications in both academic and industry contexts."
    
    async def _analyze_limitations_and_gaps(self, 
                                          sources: List[Dict[str, Any]], 
                                          clusters: List[ThematicCluster]) -> str:
        """Analyze limitations and research gaps."""
        try:
            # Analyze coverage gaps
            covered_themes = [cluster.theme for cluster in clusters]
            
            # Get recent vs older research distribution
            current_year = datetime.now().year
            recent_sources = [s for s in sources if s.get('year', 0) >= current_year - 3]
            older_sources = [s for s in sources if s.get('year', 0) < current_year - 3]
            
            # Sample sources for limitation analysis
            sample_sources = []
            for source in sources[:10]:
                title = source.get('title', 'Unknown')
                abstract = source.get('abstract', '')[:200]
                year = source.get('year', 'Unknown')
                sample_sources.append(f"- {title} ({year}): {abstract}")
            
            sources_sample = "\n".join(sample_sources)
            
            prompt = f"""
            Analyze limitations and research gaps based on this research analysis:
            
            Covered themes: {', '.join(covered_themes)}
            Recent sources: {len(recent_sources)}/{len(sources)}
            
            Sample sources:
            {sources_sample}
            
            Identify:
            1. Methodological limitations
            2. Coverage gaps or blind spots
            3. Temporal limitations (recency of research)
            4. Scope limitations
            5. Areas needing more research
            
            Be specific and constructive in identifying gaps.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Limitations analysis failed: {e}")
            return "Further research is needed to address methodological and coverage limitations."
    
    async def _suggest_future_directions(self, 
                                       query: str, 
                                       sources: List[Dict[str, Any]], 
                                       limitations: str) -> str:
        """Suggest future research directions."""
        try:
            # Get recent high-impact sources for trend analysis
            recent_sources = [s for s in sources if s.get('year', 0) >= datetime.now().year - 2]
            high_impact = sorted(sources, key=lambda x: x.get('citations', 0), reverse=True)[:5]
            
            recent_titles = [s.get('title', 'Unknown') for s in recent_sources[:5]]
            impact_titles = [s.get('title', 'Unknown') for s in high_impact]
            
            prompt = f"""
            Suggest future research directions for: "{query}"
            
            Based on current research state and identified limitations:
            {limitations[:500]}
            
            Recent research trends: {', '.join(recent_titles)}
            High-impact work: {', '.join(impact_titles)}
            
            Suggest future directions that:
            1. Address identified gaps and limitations
            2. Build on current strengths
            3. Explore emerging opportunities
            4. Consider interdisciplinary approaches
            5. Have practical significance
            
            Provide specific, actionable research directions.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Future directions suggestion failed: {e}")
            return "Future research should address current limitations and explore emerging opportunities in the field."
    
    async def _extract_section_findings(self, cluster: ThematicCluster, content: str) -> List[str]:
        """Extract key findings from a section's content."""
        try:
            prompt = f"""
            Extract 2-4 key findings from this research section about "{cluster.theme}":
            
            {content[:800]}
            
            Return as a numbered list of specific, evidence-based findings.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            findings_text = response.choices[0].message.content.strip()
            return self._parse_numbered_list(findings_text)[:4]
            
        except Exception as e:
            logger.warning(f"Section findings extraction failed: {e}")
            return []
    
    def _parse_thematic_clusters(self, clusters_text: str, sources: List[Dict[str, Any]]) -> List[ThematicCluster]:
        """Parse thematic clusters from AI response."""
        clusters = []
        
        # Split by theme sections
        sections = re.split(r'---|\n\n(?=THEME:)', clusters_text)
        
        for section in sections:
            if 'THEME:' not in section:
                continue
                
            try:
                # Extract theme name
                theme_match = re.search(r'THEME:\s*(.+)', section)
                if not theme_match:
                    continue
                theme = theme_match.group(1).strip()
                
                # Extract source numbers
                sources_match = re.search(r'SOURCES:\s*(.+)', section)
                source_indices = []
                if sources_match:
                    source_nums_text = sources_match.group(1).strip()
                    source_nums = re.findall(r'\d+', source_nums_text)
                    source_indices = [int(num) - 1 for num in source_nums if int(num) <= len(sources)]
                
                # Extract concepts
                concepts_match = re.search(r'CONCEPTS:\s*(.+)', section)
                concepts = []
                if concepts_match:
                    concepts_text = concepts_match.group(1).strip()
                    concepts = [c.strip() for c in concepts_text.split(',') if c.strip()]
                
                # Get cluster sources
                cluster_sources = []
                relevance_scores = []
                for idx in source_indices:
                    if 0 <= idx < len(sources):
                        cluster_sources.append(sources[idx])
                        relevance_scores.append(0.8)  # Default relevance
                
                if cluster_sources:
                    cluster = ThematicCluster(
                        theme=theme,
                        sources=cluster_sources,
                        relevance_scores=relevance_scores,
                        theme_strength=len(cluster_sources) / len(sources),
                        key_concepts=concepts[:5]
                    )
                    clusters.append(cluster)
                    
            except Exception as e:
                logger.warning(f"Failed to parse cluster section: {e}")
                continue
        
        return clusters
    
    def _create_fallback_clusters(self, sources: List[Dict[str, Any]]) -> List[ThematicCluster]:
        """Create simple fallback clusters when AI clustering fails."""
        clusters = []
        
        # Group by publication year ranges
        current_year = datetime.now().year
        
        recent_sources = [s for s in sources if s.get('year', 0) >= current_year - 3]
        older_sources = [s for s in sources if s.get('year', 0) < current_year - 3]
        
        if recent_sources:
            clusters.append(ThematicCluster(
                theme="Recent Developments",
                sources=recent_sources,
                relevance_scores=[0.8] * len(recent_sources),
                theme_strength=len(recent_sources) / len(sources),
                key_concepts=["recent", "current", "latest"]
            ))
        
        if older_sources:
            clusters.append(ThematicCluster(
                theme="Foundational Research",
                sources=older_sources,
                relevance_scores=[0.7] * len(older_sources),
                theme_strength=len(older_sources) / len(sources),
                key_concepts=["foundational", "established", "classical"]
            ))
        
        return clusters
    
    def _assess_evidence_strength(self, cluster: ThematicCluster) -> str:
        """Assess the strength of evidence in a cluster."""
        source_count = len(cluster.sources)
        
        if source_count >= self.min_sources_for_strong_evidence:
            return "strong"
        elif source_count >= self.min_sources_for_moderate_evidence:
            return "moderate"
        else:
            return "limited"
    
    def _calculate_section_confidence(self, cluster: ThematicCluster) -> float:
        """Calculate confidence score for a section."""
        confidence = 0.5  # Base confidence
        
        # Source count factor
        if len(cluster.sources) >= 5:
            confidence += 0.2
        elif len(cluster.sources) >= 3:
            confidence += 0.1
        
        # Theme strength factor
        confidence += cluster.theme_strength * 0.2
        
        # Quality indicators
        avg_citations = sum(s.get('citations', 0) for s in cluster.sources) / len(cluster.sources)
        if avg_citations > 50:
            confidence += 0.1
        elif avg_citations > 10:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _assess_synthesis_confidence(self, 
                                   sources: List[Dict[str, Any]], 
                                   clusters: List[ThematicCluster], 
                                   sections: List[SynthesisSection]) -> Dict[str, float]:
        """Assess confidence in different aspects of the synthesis."""
        total_sources = len(sources)
        
        return {
            'overall_confidence': min(1.0, 0.5 + (total_sources / 40)),  # Increases with source count
            'thematic_clustering': min(1.0, len(clusters) / 5),  # Optimal around 5 clusters
            'evidence_strength': sum(1 for s in sections if s.evidence_strength == 'strong') / len(sections) if sections else 0,
            'coverage_completeness': min(1.0, len(clusters) * 0.2),
            'source_quality': self._assess_overall_source_quality(sources)
        }
    
    def _assess_overall_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """Assess overall quality of the source collection."""
        if not sources:
            return 0.0
        
        quality_scores = []
        for source in sources:
            score = 0.0
            
            # Citation factor
            citations = source.get('citations', 0)
            if citations > 100:
                score += 0.4
            elif citations > 10:
                score += 0.2
            elif citations > 0:
                score += 0.1
            
            # Recency factor
            year = source.get('year', 0)
            current_year = datetime.now().year
            if year >= current_year - 3:
                score += 0.3
            elif year >= current_year - 7:
                score += 0.2
            
            # Abstract availability
            if source.get('abstract') and source['abstract'] != 'No abstract available':
                score += 0.3
            
            quality_scores.append(min(1.0, score))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered list from text."""
        items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match numbered items (1., 2., etc.)
            match = re.match(r'^\d+\.?\s*(.+)', line)
            if match:
                items.append(match.group(1).strip())
            elif line and not line.startswith('#') and len(line) > 10:
                # Include non-numbered substantial lines
                items.append(line)
        
        return items
    
    def _calculate_avg_year(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate average publication year of sources."""
        years = [s.get('year', 0) for s in sources if s.get('year', 0) > 0]
        return sum(years) / len(years) if years else 0
    
    def _analyze_source_types(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze the types of sources in the collection."""
        source_types = Counter()
        
        for source in sources:
            source_type = source.get('source_type', 'unknown')
            source_types[source_type] += 1
        
        return dict(source_types)
    
    def _create_fallback_synthesis(self, query: str, sources: List[Dict[str, Any]]) -> StructuredSynthesis:
        """Create fallback synthesis when main synthesis fails."""
        return StructuredSynthesis(
            executive_summary=f"Analysis of {len(sources)} sources related to '{query}' reveals multiple research approaches and findings.",
            key_findings=[f"Research on {query} shows diverse methodological approaches"],
            thematic_sections=[],
            comparative_analysis="Limited comparative analysis available due to processing constraints.",
            methodological_insights="Various research methodologies are employed in the field.",
            practical_applications="The research has potential applications in academic and practical contexts.",
            limitations_and_gaps="Further analysis needed to identify specific limitations and gaps.",
            future_directions="Future research should address current knowledge gaps.",
            confidence_assessment={'overall_confidence': 0.3},
            synthesis_metadata={'total_sources': len(sources), 'fallback_mode': True}
        )


async def test_synthesizer():
    """Test function for the synthesizer."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    synthesizer = EnhancedAISynthesizer(api_key, debug_mode=True)
    
    query = "machine learning transformer architectures"
    
    # Create test sources
    test_sources = [
        {
            'title': 'Attention Is All You Need',
            'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...',
            'year': 2017,
            'venue': 'NIPS',
            'citations': 45000,
            'source_type': 'conference'
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'abstract': 'We introduce a new language representation model called BERT...',
            'year': 2019,
            'venue': 'NAACL',
            'citations': 35000,
            'source_type': 'conference'
        }
    ]
    
    print(f"📝 Testing synthesis for: {query}")
    synthesis = await synthesizer.create_structured_synthesis(query, test_sources)
    
    print(f"\nExecutive Summary: {synthesis.executive_summary[:200]}...")
    print(f"Key Findings: {len(synthesis.key_findings)}")
    print(f"Thematic Sections: {len(synthesis.thematic_sections)}")
    print(f"Overall Confidence: {synthesis.confidence_assessment.get('overall_confidence', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(test_synthesizer())
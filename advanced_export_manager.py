"""
Advanced Export Manager
Comprehensive export functionality for research results with proper bibliography generation.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import re
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ExportMetadata:
    """Metadata for exported research."""
    export_timestamp: str
    export_format: str
    research_id: str
    query: str
    source_count: int
    export_version: str = "2.0"
    generated_by: str = "Advanced Autonomous Research Agent"

@dataclass
class CitationData:
    """Structured citation data."""
    authors: List[str]
    title: str
    year: str
    venue: Optional[str]
    publisher: Optional[str]
    url: Optional[str]
    doi: Optional[str]
    pages: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    citation_number: int
    source_type: str = "journal"

class AdvancedExportManager:
    """
    Advanced export manager with support for multiple formats and proper bibliography.
    """
    
    def __init__(self, export_base_path: str = "./exports"):
        self.export_base_path = Path(export_base_path)
        self.export_base_path.mkdir(exist_ok=True)
        
        # Citation style formatters
        self.citation_formatters = {
            'apa': self._format_apa_citation,
            'mla': self._format_mla_citation,
            'ieee': self._format_ieee_citation,
            'chicago': self._format_chicago_citation
        }
        
        logger.info(f"📁 Advanced Export Manager initialized with base path: {self.export_base_path}")
    
    async def export_research_result(self, 
                                   research_result: Dict[str, Any], 
                                   format_type: str = "comprehensive",
                                   citation_style: str = "apa",
                                   include_metadata: bool = True,
                                   custom_filename: Optional[str] = None) -> Dict[str, str]:
        """
        Export research result in multiple formats.
        
        Args:
            research_result: Complete research result data
            format_type: Export format type (comprehensive, academic, summary, web)
            citation_style: Citation style (apa, mla, ieee, chicago)
            include_metadata: Whether to include export metadata
            custom_filename: Custom filename prefix
            
        Returns:
            Dictionary with file paths for different export formats
        """
        logger.info(f"📤 Exporting research result in {format_type} format with {citation_style} citations")
        
        try:
            # Generate base filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            research_id = research_result.get('research_id', 'unknown')
            base_filename = custom_filename or f"research_{research_id}_{timestamp}"
            
            # Create citations data
            citations = self._extract_citations_data(research_result)
            
            # Create export metadata
            metadata = ExportMetadata(
                export_timestamp=datetime.now().isoformat(),
                export_format=format_type,
                research_id=research_id,
                query=research_result.get('query', 'Unknown Query'),
                source_count=len(research_result.get('sources', [])),
                export_version="2.0",
                generated_by="Advanced Autonomous Research Agent"
            )
            
            export_files = {}
            
            # Export in different formats based on format_type
            if format_type == "comprehensive":
                # Export all formats for comprehensive export
                export_files['markdown'] = await self._export_markdown_comprehensive(
                    research_result, citations, citation_style, metadata, base_filename
                )
                export_files['json'] = await self._export_json_comprehensive(
                    research_result, citations, metadata, base_filename
                )
                export_files['txt'] = await self._export_text_comprehensive(
                    research_result, citations, citation_style, metadata, base_filename
                )
                export_files['html'] = await self._export_html_comprehensive(
                    research_result, citations, citation_style, metadata, base_filename
                )
                export_files['bibliography'] = await self._export_bibliography_only(
                    citations, citation_style, base_filename
                )
                
            elif format_type == "academic":
                # Academic paper format
                export_files['markdown'] = await self._export_academic_markdown(
                    research_result, citations, citation_style, metadata, base_filename
                )
                export_files['bibliography'] = await self._export_bibliography_only(
                    citations, citation_style, base_filename
                )
                
            elif format_type == "summary":
                # Summary format
                export_files['markdown'] = await self._export_summary_markdown(
                    research_result, citations, citation_style, metadata, base_filename
                )
                
            elif format_type == "web":
                # Web-optimized format
                export_files['html'] = await self._export_html_web(
                    research_result, citations, citation_style, metadata, base_filename
                )
                export_files['json'] = await self._export_json_web(
                    research_result, citations, metadata, base_filename
                )
            
            # Always include raw JSON export
            export_files['raw_json'] = await self._export_raw_json(
                research_result, metadata, base_filename
            )
            
            logger.info(f"✅ Export completed successfully with {len(export_files)} files")
            return export_files
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            raise e
    
    def _extract_citations_data(self, research_result: Dict[str, Any]) -> List[CitationData]:
        """Extract and structure citation data from research result."""
        citations = []
        sources = research_result.get('sources', [])
        
        for i, source in enumerate(sources, 1):
            citation = CitationData(
                authors=source.get('authors', []),
                title=source.get('title', 'Unknown Title'),
                year=str(source.get('year', 'n.d.')),
                venue=source.get('journal') or source.get('venue'),
                publisher=source.get('publisher'),
                url=source.get('url'),
                doi=source.get('doi'),
                pages=source.get('pages'),
                volume=source.get('volume'),
                issue=source.get('issue'),
                citation_number=i,
                source_type=source.get('source_type', 'journal')
            )
            citations.append(citation)
        
        return citations
    
    # Citation Formatters
    
    def _format_apa_citation(self, citation: CitationData) -> str:
        """Format citation in APA style."""
        authors_str = self._format_authors_apa(citation.authors)
        
        citation_parts = [authors_str]
        citation_parts.append(f"({citation.year}).")
        citation_parts.append(f"{citation.title}.")
        
        if citation.venue:
            if citation.source_type == 'journal':
                venue_part = f"*{citation.venue}*"
                if citation.volume:
                    venue_part += f", {citation.volume}"
                if citation.issue:
                    venue_part += f"({citation.issue})"
                if citation.pages:
                    venue_part += f", {citation.pages}"
                citation_parts.append(venue_part + ".")
            else:
                citation_parts.append(f"*{citation.venue}*.")
        
        if citation.doi:
            citation_parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            citation_parts.append(citation.url)
        
        return " ".join(citation_parts)
    
    def _format_mla_citation(self, citation: CitationData) -> str:
        """Format citation in MLA style."""
        authors_str = self._format_authors_mla(citation.authors)
        
        citation_parts = [f"{authors_str}."]
        citation_parts.append(f'"{citation.title}."')
        
        if citation.venue:
            citation_parts.append(f"*{citation.venue}*,")
        
        if citation.volume:
            citation_parts.append(f"vol. {citation.volume},")
        if citation.issue:
            citation_parts.append(f"no. {citation.issue},")
        
        citation_parts.append(f"{citation.year},")
        
        if citation.pages:
            citation_parts.append(f"pp. {citation.pages}.")
        
        if citation.url:
            citation_parts.append(citation.url + ".")
        
        return " ".join(citation_parts)
    
    def _format_ieee_citation(self, citation: CitationData) -> str:
        """Format citation in IEEE style."""
        authors_str = self._format_authors_ieee(citation.authors)
        
        citation_parts = [f"{authors_str},"]
        citation_parts.append(f'"{citation.title},"')
        
        if citation.venue:
            if citation.source_type == 'journal':
                citation_parts.append(f"*{citation.venue}*,")
            else:
                citation_parts.append(f"in *{citation.venue}*,")
        
        if citation.volume:
            citation_parts.append(f"vol. {citation.volume},")
        if citation.issue:
            citation_parts.append(f"no. {citation.issue},")
        if citation.pages:
            citation_parts.append(f"pp. {citation.pages},")
        
        citation_parts.append(f"{citation.year}.")
        
        if citation.doi:
            citation_parts.append(f"doi: {citation.doi}")
        
        return " ".join(citation_parts)
    
    def _format_chicago_citation(self, citation: CitationData) -> str:
        """Format citation in Chicago style."""
        authors_str = self._format_authors_chicago(citation.authors)
        
        citation_parts = [f"{authors_str}."]
        citation_parts.append(f'"{citation.title}."')
        
        if citation.venue:
            citation_parts.append(f"*{citation.venue}*")
        
        if citation.volume:
            citation_parts.append(f"{citation.volume},")
        if citation.issue:
            citation_parts.append(f"no. {citation.issue}")
        
        citation_parts.append(f"({citation.year}):")
        
        if citation.pages:
            citation_parts.append(f"{citation.pages}.")
        
        if citation.url:
            citation_parts.append(f"Accessed {datetime.now().strftime('%B %d, %Y')}. {citation.url}.")
        
        return " ".join(citation_parts)
    
    # Author formatting helpers
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format authors for APA style."""
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]}, & {authors[1]}"
        else:
            return f"{', '.join(authors[:-1])}, & {authors[-1]}"
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """Format authors for MLA style."""
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]}, and {authors[1]}"
        else:
            return f"{authors[0]}, et al"
    
    def _format_authors_ieee(self, authors: List[str]) -> str:
        """Format authors for IEEE style."""
        if not authors:
            return "Unknown Author"
        if len(authors) <= 3:
            return ", ".join(authors)
        else:
            return f"{authors[0]}, et al."
    
    def _format_authors_chicago(self, authors: List[str]) -> str:
        """Format authors for Chicago style."""
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]}, and {authors[1]}"
        else:
            return f"{authors[0]}, et al."
    
    # Export format implementations
    
    async def _export_markdown_comprehensive(self, research_result: Dict[str, Any], 
                                           citations: List[CitationData], 
                                           citation_style: str,
                                           metadata: ExportMetadata,
                                           base_filename: str) -> str:
        """Export comprehensive markdown format."""
        formatter = self.citation_formatters[citation_style]
        
        content = []
        
        # Title and metadata
        content.append(f"# Research Report: {research_result.get('query', 'Unknown Query')}")
        content.append("")
        content.append(f"**Research ID:** {metadata.research_id}")
        content.append(f"**Export Date:** {metadata.export_timestamp}")
        content.append(f"**Sources:** {metadata.source_count}")
        content.append(f"**Citation Style:** {citation_style.upper()}")
        content.append("")
        
        # Executive Summary
        if research_result.get('executive_summary'):
            content.append("## Executive Summary")
            content.append("")
            content.append(research_result['executive_summary'])
            content.append("")
        
        # Main Synthesis
        content.append("## Research Synthesis")
        content.append("")
        content.append(research_result.get('synthesis', 'No synthesis available.'))
        content.append("")
        
        # Key Findings
        key_findings = research_result.get('key_findings', [])
        if key_findings:
            content.append("## Key Findings")
            content.append("")
            for i, finding in enumerate(key_findings, 1):
                content.append(f"{i}. {finding}")
            content.append("")
        
        # Detailed Sections
        detailed_sections = research_result.get('detailed_sections', [])
        if detailed_sections:
            content.append("## Detailed Analysis")
            content.append("")
            for section in detailed_sections:
                content.append(f"### {section.get('title', 'Section')}")
                content.append("")
                content.append(section.get('content', ''))
                content.append("")
        
        # Quality Assessment
        quality_assessment = research_result.get('quality_assessment')
        if quality_assessment:
            content.append("## Quality Assessment")
            content.append("")
            content.append(f"**Overall Quality:** {quality_assessment.get('overall_quality', 'Unknown')}")
            content.append(f"**Confidence Score:** {quality_assessment.get('confidence_score', 'N/A')}")
            content.append("")
        
        # Bibliography
        content.append("## Bibliography")
        content.append("")
        for citation in citations:
            formatted_citation = formatter(citation)
            content.append(f"[{citation.citation_number}] {formatted_citation}")
        content.append("")
        
        # Appendices
        content.append("## Appendices")
        content.append("")
        content.append("### Source Summary")
        content.append("")
        sources = research_result.get('sources', [])
        for i, source in enumerate(sources, 1):
            content.append(f"**[{i}] {source.get('title', 'Unknown Title')}**")
            if source.get('abstract'):
                content.append("")
                content.append("*Abstract:*")
                content.append(source['abstract'])
            content.append("")
        
        # Write file
        filepath = self.export_base_path / f"{base_filename}_comprehensive.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.info(f"📝 Comprehensive Markdown export saved: {filepath}")
        return str(filepath)
    
    async def _export_json_comprehensive(self, research_result: Dict[str, Any],
                                       citations: List[CitationData],
                                       metadata: ExportMetadata,
                                       base_filename: str) -> str:
        """Export comprehensive JSON format."""
        export_data = {
            "metadata": asdict(metadata),
            "research_result": research_result,
            "formatted_citations": [asdict(citation) for citation in citations],
            "export_info": {
                "format": "comprehensive_json",
                "version": "2.0",
                "features": [
                    "inline_citations",
                    "structured_synthesis",
                    "quality_assessment",
                    "bibliography",
                    "metadata"
                ]
            }
        }
        
        filepath = self.export_base_path / f"{base_filename}_comprehensive.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"🔗 Comprehensive JSON export saved: {filepath}")
        return str(filepath)
    
    async def _export_text_comprehensive(self, research_result: Dict[str, Any],
                                       citations: List[CitationData],
                                       citation_style: str,
                                       metadata: ExportMetadata,
                                       base_filename: str) -> str:
        """Export comprehensive plain text format."""
        formatter = self.citation_formatters[citation_style]
        
        content = []
        content.append("=" * 80)
        content.append(f"RESEARCH REPORT: {research_result.get('query', 'Unknown Query').upper()}")
        content.append("=" * 80)
        content.append("")
        content.append(f"Research ID: {metadata.research_id}")
        content.append(f"Export Date: {metadata.export_timestamp}")
        content.append(f"Sources: {metadata.source_count}")
        content.append(f"Citation Style: {citation_style.upper()}")
        content.append("")
        
        # Main content
        content.append("RESEARCH SYNTHESIS")
        content.append("-" * 40)
        content.append(research_result.get('synthesis', 'No synthesis available.'))
        content.append("")
        
        # Key findings
        key_findings = research_result.get('key_findings', [])
        if key_findings:
            content.append("KEY FINDINGS")
            content.append("-" * 40)
            for i, finding in enumerate(key_findings, 1):
                content.append(f"{i}. {finding}")
            content.append("")
        
        # Bibliography
        content.append("BIBLIOGRAPHY")
        content.append("-" * 40)
        for citation in citations:
            formatted_citation = formatter(citation)
            content.append(f"[{citation.citation_number}] {formatted_citation}")
        content.append("")
        
        content.append("=" * 80)
        content.append(f"Generated by: {metadata.generated_by}")
        content.append("=" * 80)
        
        filepath = self.export_base_path / f"{base_filename}_comprehensive.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.info(f"📄 Comprehensive text export saved: {filepath}")
        return str(filepath)
    
    async def _export_html_comprehensive(self, research_result: Dict[str, Any],
                                       citations: List[CitationData],
                                       citation_style: str,
                                       metadata: ExportMetadata,
                                       base_filename: str) -> str:
        """Export comprehensive HTML format."""
        formatter = self.citation_formatters[citation_style]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {research_result.get('query', 'Unknown Query')}</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metadata {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
        }}
        .section {{
            background-color: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .citation {{
            margin-bottom: 10px;
            padding: 10px;
            background-color: #f8f8f8;
            border-left: 4px solid #667eea;
        }}
        .inline-citation {{
            background-color: #e3f2fd;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            color: #1976d2;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .quality-indicator {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .quality-excellent {{ background-color: #4caf50; color: white; }}
        .quality-good {{ background-color: #8bc34a; color: white; }}
        .quality-acceptable {{ background-color: #ffc107; color: black; }}
        .quality-poor {{ background-color: #ff9800; color: white; }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Research Report</h1>
        <h2>{research_result.get('query', 'Unknown Query')}</h2>
    </div>
    
    <div class="metadata">
        <strong>Research ID:</strong> {metadata.research_id}<br>
        <strong>Export Date:</strong> {metadata.export_timestamp}<br>
        <strong>Sources:</strong> {metadata.source_count}<br>
        <strong>Citation Style:</strong> {citation_style.upper()}
    </div>
"""
        
        # Executive Summary
        if research_result.get('executive_summary'):
            html_content += f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <p>{research_result['executive_summary']}</p>
    </div>
"""
        
        # Main Synthesis with inline citations highlighted
        synthesis = research_result.get('synthesis', 'No synthesis available.')
        # Highlight inline citations
        synthesis = re.sub(r'\[(\d+(?:[-,]\d+)*)\]', r'<span class="inline-citation">[\1]</span>', synthesis)
        
        html_content += f"""
    <div class="section">
        <h2>Research Synthesis</h2>
        <p>{synthesis}</p>
    </div>
"""
        
        # Key Findings
        key_findings = research_result.get('key_findings', [])
        if key_findings:
            findings_html = ""
            for i, finding in enumerate(key_findings, 1):
                # Highlight inline citations in findings
                finding = re.sub(r'\[(\d+(?:[-,]\d+)*)\]', r'<span class="inline-citation">[\1]</span>', finding)
                findings_html += f"<li>{finding}</li>"
            
            html_content += f"""
    <div class="section">
        <h2>Key Findings</h2>
        <ol>{findings_html}</ol>
    </div>
"""
        
        # Quality Assessment
        quality_assessment = research_result.get('quality_assessment')
        if quality_assessment:
            quality_class = f"quality-{quality_assessment.get('overall_quality', 'acceptable').lower()}"
            html_content += f"""
    <div class="section">
        <h2>Quality Assessment</h2>
        <p><strong>Overall Quality:</strong> <span class="quality-indicator {quality_class}">{quality_assessment.get('overall_quality', 'Unknown').title()}</span></p>
        <p><strong>Confidence Score:</strong> {quality_assessment.get('confidence_score', 'N/A')}</p>
    </div>
"""
        
        # Bibliography
        citations_html = ""
        for citation in citations:
            formatted_citation = formatter(citation)
            citations_html += f'<div class="citation">[{citation.citation_number}] {formatted_citation}</div>'
        
        html_content += f"""
    <div class="section">
        <h2>Bibliography</h2>
        {citations_html}
    </div>
"""
        
        html_content += f"""
    <div class="footer">
        <p>Generated by {metadata.generated_by}</p>
        <p>Export Version: {metadata.export_version}</p>
    </div>
</body>
</html>
"""
        
        filepath = self.export_base_path / f"{base_filename}_comprehensive.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"🌐 Comprehensive HTML export saved: {filepath}")
        return str(filepath)
    
    async def _export_bibliography_only(self, citations: List[CitationData],
                                      citation_style: str,
                                      base_filename: str) -> str:
        """Export bibliography only."""
        formatter = self.citation_formatters[citation_style]
        
        content = []
        content.append(f"# Bibliography ({citation_style.upper()} Style)")
        content.append("")
        content.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        content.append(f"Total Sources: {len(citations)}")
        content.append("")
        
        for citation in citations:
            formatted_citation = formatter(citation)
            content.append(f"[{citation.citation_number}] {formatted_citation}")
            content.append("")
        
        filepath = self.export_base_path / f"{base_filename}_bibliography_{citation_style}.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.info(f"📚 Bibliography export saved: {filepath}")
        return str(filepath)
    
    async def _export_academic_markdown(self, research_result: Dict[str, Any],
                                      citations: List[CitationData],
                                      citation_style: str,
                                      metadata: ExportMetadata,
                                      base_filename: str) -> str:
        """Export academic paper format."""
        content = []
        
        # Academic paper structure
        content.append(f"# {research_result.get('query', 'Research Query')}")
        content.append("")
        content.append("## Abstract")
        content.append("")
        
        # Use executive summary as abstract if available
        abstract = research_result.get('executive_summary') or research_result.get('synthesis', '')[:500] + "..."
        content.append(abstract)
        content.append("")
        
        content.append("## Introduction")
        content.append("")
        content.append("This research synthesis examines current literature and findings related to the research question.")
        content.append("")
        
        content.append("## Literature Review and Findings")
        content.append("")
        content.append(research_result.get('synthesis', 'No synthesis available.'))
        content.append("")
        
        # Key findings as separate section
        key_findings = research_result.get('key_findings', [])
        if key_findings:
            content.append("## Key Findings")
            content.append("")
            for finding in key_findings:
                content.append(f"- {finding}")
            content.append("")
        
        content.append("## Conclusion")
        content.append("")
        content.append("This synthesis provides a comprehensive overview of current research in the field and identifies key patterns and insights from the literature.")
        content.append("")
        
        # References
        formatter = self.citation_formatters[citation_style]
        content.append("## References")
        content.append("")
        for citation in citations:
            formatted_citation = formatter(citation)
            content.append(formatted_citation)
            content.append("")
        
        filepath = self.export_base_path / f"{base_filename}_academic.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.info(f"🎓 Academic format export saved: {filepath}")
        return str(filepath)
    
    async def _export_summary_markdown(self, research_result: Dict[str, Any],
                                     citations: List[CitationData],
                                     citation_style: str,
                                     metadata: ExportMetadata,
                                     base_filename: str) -> str:
        """Export summary format."""
        content = []
        
        content.append(f"# Research Summary: {research_result.get('query', 'Unknown Query')}")
        content.append("")
        content.append(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
        content.append(f"**Sources:** {len(citations)}")
        content.append("")
        
        # Executive summary or first part of synthesis
        summary_text = research_result.get('executive_summary') or research_result.get('synthesis', '')[:1000] + "..."
        content.append("## Summary")
        content.append("")
        content.append(summary_text)
        content.append("")
        
        # Top key findings
        key_findings = research_result.get('key_findings', [])
        if key_findings:
            content.append("## Key Points")
            content.append("")
            for i, finding in enumerate(key_findings[:5], 1):  # Limit to top 5
                content.append(f"{i}. {finding}")
            content.append("")
        
        # Sources count by type
        source_types = {}
        for source in research_result.get('sources', []):
            source_type = source.get('source_type', 'unknown')
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        if source_types:
            content.append("## Source Types")
            content.append("")
            for source_type, count in source_types.items():
                content.append(f"- {source_type.title()}: {count}")
            content.append("")
        
        filepath = self.export_base_path / f"{base_filename}_summary.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        logger.info(f"📋 Summary export saved: {filepath}")
        return str(filepath)
    
    async def _export_html_web(self, research_result: Dict[str, Any],
                             citations: List[CitationData],
                             citation_style: str,
                             metadata: ExportMetadata,
                             base_filename: str) -> str:
        """Export web-optimized HTML format."""
        # Simplified HTML for web use
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{research_result.get('query', 'Research Results')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; margin: 20px; }}
        .research-result {{ max-width: 800px; margin: 0 auto; }}
        .synthesis {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .citation {{ font-size: 0.9em; color: #666; }}
        .inline-citation {{ background: #e3f2fd; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="research-result">
        <h1>{research_result.get('query', 'Research Results')}</h1>
        <div class="synthesis">{research_result.get('synthesis', 'No synthesis available.')}</div>
        <div class="sources">
            <h2>Sources ({len(citations)})</h2>
"""
        
        for citation in citations:
            formatted_citation = self.citation_formatters[citation_style](citation)
            html_content += f'<div class="citation">[{citation.citation_number}] {formatted_citation}</div>'
        
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        
        filepath = self.export_base_path / f"{base_filename}_web.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"🌍 Web HTML export saved: {filepath}")
        return str(filepath)
    
    async def _export_json_web(self, research_result: Dict[str, Any],
                             citations: List[CitationData],
                             metadata: ExportMetadata,
                             base_filename: str) -> str:
        """Export web-optimized JSON format."""
        web_data = {
            "query": research_result.get('query'),
            "synthesis": research_result.get('synthesis'),
            "key_findings": research_result.get('key_findings', []),
            "source_count": len(citations),
            "sources": [
                {
                    "id": citation.citation_number,
                    "title": citation.title,
                    "authors": citation.authors,
                    "year": citation.year,
                    "url": citation.url
                }
                for citation in citations
            ],
            "export_timestamp": metadata.export_timestamp
        }
        
        filepath = self.export_base_path / f"{base_filename}_web.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(web_data, f, indent=2, default=str)
        
        logger.info(f"🔗 Web JSON export saved: {filepath}")
        return str(filepath)
    
    async def _export_raw_json(self, research_result: Dict[str, Any],
                             metadata: ExportMetadata,
                             base_filename: str) -> str:
        """Export raw JSON data."""
        raw_data = {
            "metadata": asdict(metadata),
            "research_result": research_result
        }
        
        filepath = self.export_base_path / f"{base_filename}_raw.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        logger.info(f"💾 Raw JSON export saved: {filepath}")
        return str(filepath)

# Export utility functions

async def export_research_comprehensive(research_result: Dict[str, Any], 
                                      export_path: str = "./exports",
                                      citation_style: str = "apa") -> Dict[str, str]:
    """
    Convenience function to export research result in comprehensive format.
    """
    export_manager = AdvancedExportManager(export_path)
    return await export_manager.export_research_result(
        research_result, 
        format_type="comprehensive",
        citation_style=citation_style
    )

async def export_bibliography(research_result: Dict[str, Any],
                            export_path: str = "./exports", 
                            citation_style: str = "apa") -> str:
    """
    Convenience function to export bibliography only.
    """
    export_manager = AdvancedExportManager(export_path)
    citations = export_manager._extract_citations_data(research_result)
    return await export_manager._export_bibliography_only(citations, citation_style, "bibliography")
"""
Simple CLI Interface for Research Agent
Interactive command-line interface for conducting research.
"""

import asyncio
import os
import sys
import argparse
from datetime import datetime
import json
from pathlib import Path

from research_agent import ResearchAgent
try:
    from enhanced_research_agent import EnhancedResearchAgent
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


class ResearchCLI:
    """Simple command-line interface for the research agent."""
    
    def __init__(self, api_key: str, enhanced_mode: bool = False, debug_mode: bool = False):
        self.enhanced_mode = enhanced_mode and ENHANCED_AVAILABLE
        self.debug_mode = debug_mode
        
        if self.enhanced_mode:
            debug_info = " (debug mode)" if debug_mode else ""
            print(f"🔬 Enhanced Research Agent (with validation layers){debug_info}")
            self.agent = EnhancedResearchAgent(api_key, debug_mode=debug_mode)
        else:
            if enhanced_mode and not ENHANCED_AVAILABLE:
                print("⚠️  Enhanced mode requested but dependencies not available. Using basic mode.")
            debug_info = " (debug mode)" if debug_mode else ""
            print(f"🔬 Basic Research Agent{debug_info}")
            self.agent = ResearchAgent(api_key)
            
        self.current_result = None
        
    def display_banner(self):
        """Display the application banner."""
        mode_info = "Enhanced Mode with Validation Layers" if self.enhanced_mode else "Basic Mode"
        enhanced_features = """
🛡️ Source relevance validation
🔗 Content-citation verification
🔄 Cross-source perspective analysis  
📊 Quality assessment and recommendations""" if self.enhanced_mode else ""
        
        banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    Autonomous Research Agent                     ║
║                     Real Research, Real Results                  ║
║                         {mode_info:^28}                         ║
╚══════════════════════════════════════════════════════════════════╝

🔍 Searches ArXiv, PubMed, and web sources
📄 Extracts real content from papers and articles  
📚 Provides proper citations and synthesis{enhanced_features}
⚡ Simple, functional interface

Type 'help' for commands or 'quit' to exit.
"""
        print(banner)
    
    def display_help(self):
        """Display help information."""
        enhanced_commands = """
validation            Show validation report from last research
quality               Show quality assessment from last research""" if self.enhanced_mode else ""
        
        debug_commands = """
debug <query>         Run debug analysis on a query""" if self.enhanced_mode and self.debug_mode else ""
        
        enhanced_info = f"""
Enhanced Mode Features (Available: {'✅' if self.enhanced_mode else '❌'}):
─────────────────────────────────────────────────────────────
🛡️ Source relevance validation and scoring
🔗 Content-citation verification to prevent hallucination  
🔄 Cross-source perspective analysis and conflict detection
📊 Quality assessment with actionable recommendations{enhanced_commands}
""" if ENHANCED_AVAILABLE else ""
        
        help_text = f"""
Available Commands:
─────────────────────────────────────────────────────────────

research <question>     Conduct research on a question
                       Example: research "latest advances in AI reasoning"

export <format>        Export last research results
                       Formats: txt, json, md
                       Example: export txt

citation <style>       Change citation style  
                       Styles: apa, mla, ieee
                       Example: citation mla

sources               Show sources from last research{enhanced_commands}{debug_commands}

help                  Show this help message
quit / exit           Exit the application{enhanced_info}

Examples:
─────────
research "What are the benefits of transformer architecture?"
research "COVID-19 treatment effectiveness studies"
research "quantum computing recent developments"
export md
citation ieee
"""
        print(help_text)
    
    async def run_interactive(self):
        """Run the interactive CLI."""
        self.display_banner()
        
        citation_style = "apa"
        
        while True:
            try:
                # Get user input
                user_input = input("🔬 Research Agent > ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command in ['quit', 'exit']:
                    print("👋 Thank you for using Research Agent!")
                    break
                    
                elif command == 'help':
                    self.display_help()
                    
                elif command == 'research':
                    if not args:
                        print("❌ Please provide a research question.")
                        print("   Example: research \"latest advances in AI reasoning\"")
                        continue
                    
                    # Clean up the query (remove quotes if present)
                    query = args.strip('"\'')
                    await self.conduct_research(query, citation_style)
                    
                elif command == 'export':
                    format_type = args.lower() if args else "txt"
                    self.export_results(format_type)
                    
                elif command == 'citation':
                    if args.lower() in ['apa', 'mla', 'ieee']:
                        citation_style = args.lower()
                        print(f"✅ Citation style set to {citation_style.upper()}")
                    else:
                        print("❌ Invalid citation style. Use: apa, mla, or ieee")
                        
                elif command == 'sources':
                    self.show_sources()
                    
                elif command == 'validation' and self.enhanced_mode:
                    self.show_validation_report()
                    
                elif command == 'quality' and self.enhanced_mode:
                    self.show_quality_assessment()
                    
                elif command == 'debug' and self.enhanced_mode and self.debug_mode:
                    if not args:
                        print("❌ Please provide a query to debug.")
                        print("   Example: debug \"machine learning transformers\"")
                        continue
                    query = args.strip('"\'')
                    self.show_debug_analysis(query)
                    
                else:
                    # Treat unknown commands as research queries
                    query = user_input.strip('"\'')
                    print(f"🔍 Interpreting as research query: \"{query}\"")
                    await self.conduct_research(query, citation_style)
                    
            except KeyboardInterrupt:
                print("\n👋 Thank you for using Research Agent!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("   Type 'help' for available commands.")
    
    async def conduct_research(self, query: str, citation_style: str = "apa"):
        """Conduct research and display results."""
        print(f"\n{'='*60}")
        print(f"🔬 Research Query: {query}")
        print(f"📖 Citation Style: {citation_style.upper()}")
        print(f"🔬 Mode: {'Enhanced (with validation)' if self.enhanced_mode else 'Basic'}")
        print(f"{'='*60}")
        
        try:
            # Start research
            if self.enhanced_mode:
                self.current_result = await self.agent.conduct_enhanced_research(query, citation_style)
            else:
                self.current_result = await self.agent.conduct_research(query, citation_style)
            
            # Display results
            self.display_results()
            
        except Exception as e:
            print(f"❌ Research failed: {e}")
            print("   Please check your internet connection and API keys.")
    
    def display_results(self):
        """Display research results in a formatted way."""
        if not self.current_result:
            print("❌ No research results to display.")
            return
        
        result = self.current_result
        
        if self.enhanced_mode:
            # Enhanced mode result display
            print(f"\n📊 Research Summary")
            print(f"   Query: {result.query}")
            print(f"   Sources found: {len(result.validated_sources)}")
            print(f"   Research time: {result.research_time} seconds")
            print(f"   Quality: {result.quality_assessment.overall_quality.value.upper() if result.quality_assessment else 'Unknown'}")
            print(f"   Confidence: {result.confidence_score:.1%}")
            print(f"   Completed: {result.timestamp[:19]}")
            
            print(f"\n🧠 Research Synthesis")
            print("-" * 40)
            print(result.synthesis)
            
            # Show validation summary
            if result.quality_assessment:
                print(f"\n🛡️ Validation Summary")
                print(f"   Overall Quality: {result.quality_assessment.overall_quality.value.upper()}")
                print(f"   Quality Score: {result.quality_assessment.overall_score:.1%}")
                print(f"   Validation Gates: {result.quality_assessment.validation_summary.get('passed_gates', 0)}/{result.quality_assessment.validation_summary.get('total_gates', 0)} passed")
            
            print(f"\n📚 Validated Sources ({len(result.validated_sources)} found)")
            print("-" * 40)
            for i, (source, relevance) in enumerate(result.validated_sources, 1):
                title = source.get('title', 'Unknown Title')
                authors = source.get('authors', [])
                year = source.get('year', 'Unknown Year')
                score = relevance.overall_score if hasattr(relevance, 'overall_score') else 'Unknown'
                
                # Format authors
                if authors:
                    if len(authors) == 1:
                        author_str = authors[0]
                    elif len(authors) <= 3:
                        author_str = ', '.join(authors)
                    else:
                        author_str = f"{authors[0]} et al."
                else:
                    author_str = "Unknown Author"
                
                print(f"{i}. {author_str} ({year}). {title}")
                if isinstance(score, (int, float)):
                    print(f"   Relevance: {score:.1%}")
            
            # Show recommendations and quality-specific guidance
            quality_level = result.quality_assessment.overall_quality.value.upper() if result.quality_assessment else 'UNKNOWN'
            
            if result.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in result.recommendations[:3]:  # Show top 3
                    print(f"   • {rec}")
                if len(result.recommendations) > 3:
                    print(f"   ... and {len(result.recommendations) - 3} more (use 'quality' command)")
            
            # Add quality-specific guidance
            if quality_level in ['POOR', 'LOW']:
                print(f"\n⚠️  Quality Improvement Tips:")
                print(f"   • Try more specific queries (e.g., 'agile vs waterfall SDLC' instead of 'SDLC')")
                print(f"   • Include domain keywords (e.g., 'software development methodology')")
                print(f"   • Specify what you want to know (e.g., 'benefits', 'comparison', 'best practices')")
                print(f"   • Break complex questions into smaller, focused queries")
            elif quality_level == 'MEDIUM':
                print(f"\n✨ To improve results further:")
                print(f"   • Add more specific context or constraints to your query")
                print(f"   • Try related queries to get additional perspectives")
            
            print(f"\n📋 Next Steps:")
            if quality_level in ['POOR', 'LOW']:
                print(f"   • Try a refined query with the tips above")
                print(f"   • Type 'validation' to understand why quality was low")
            print(f"   • Type 'quality' to see detailed quality assessment")
            print(f"   • Type 'export txt' to save results")
            print(f"   • Type 'export json' for detailed data")
            print(f"   • Ask another research question")
            
        else:
            # Basic mode result display
            print(f"\n📊 Research Summary")
            print(f"   Sources found: {result['num_sources']}")
            print(f"   Research time: {result['research_time']} seconds")
            print(f"   Completed: {result['timestamp'][:19]}")
            
            print(f"\n🧠 Research Synthesis")
            print("-" * 40)
            print(result['synthesis'])
            
            print(f"\n📚 Sources ({result['num_sources']} found)")
            print("-" * 40)
            for i, source in enumerate(result['sources'], 1):
                title = source.get('title', 'Unknown Title')
                authors = source.get('authors', [])
                year = source.get('year', 'Unknown Year')
                
                # Format authors
                if authors:
                    if len(authors) == 1:
                        author_str = authors[0]
                    elif len(authors) <= 3:
                        author_str = ', '.join(authors)
                    else:
                        author_str = f"{authors[0]} et al."
                else:
                    author_str = "Unknown Author"
                
                print(f"{i}. {author_str} ({year}). {title}")
            
            print(f"\n💡 Next Steps:")
            print(f"   • Type 'export txt' to save results")
            print(f"   • Type 'export json' for detailed data")
            print(f"   • Type 'sources' to see full bibliography")
            print(f"   • Ask another research question")
    
    def export_results(self, format_type: str = "txt"):
        """Export research results to file."""
        if not self.current_result:
            print("❌ No research results to export.")
            return
        
        # Create exports directory
        exports_dir = Path("exports")
        exports_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.current_result['query'][:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        query_safe = query_safe.replace(' ', '_')
        
        try:
            if format_type == "txt":
                filename = f"{query_safe}_{timestamp}.txt"
                filepath = exports_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.current_result['report'])
                
                print(f"✅ Results exported to: {filepath}")
                
            elif format_type == "json":
                filename = f"{query_safe}_{timestamp}.json"
                filepath = exports_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Data exported to: {filepath}")
                
            elif format_type == "md":
                filename = f"{query_safe}_{timestamp}.md"
                filepath = exports_dir / filename
                
                markdown_content = self.format_markdown_report()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                print(f"✅ Markdown report exported to: {filepath}")
                
            else:
                print("❌ Invalid format. Use: txt, json, or md")
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def format_markdown_report(self) -> str:
        """Format results as markdown."""
        if not self.current_result:
            return ""
        
        result = self.current_result
        
        markdown = f"""# Research Report: {result['query']}

**Generated:** {result['timestamp'][:19]}  
**Sources:** {result['num_sources']}  
**Research Time:** {result['research_time']} seconds

## Synthesis

{result['synthesis']}

## Sources

"""
        
        for i, source in enumerate(result['sources'], 1):
            title = source.get('title', 'Unknown Title')
            authors = source.get('authors', [])
            year = source.get('year', 'Unknown Year')
            url = source.get('url', '')
            
            author_str = ', '.join(authors) if authors else "Unknown Author"
            
            markdown += f"{i}. **{title}**  \n"
            markdown += f"   *{author_str}* ({year})  \n"
            if url:
                markdown += f"   [Link]({url})  \n"
            markdown += "\n"
        
        markdown += "\n---\n*Generated by Autonomous Research Agent*"
        
        return markdown
    
    def show_validation_report(self):
        """Display detailed validation report for enhanced mode."""
        if not self.enhanced_mode:
            print("❌ Validation reports are only available in enhanced mode.")
            return
            
        if not self.current_result:
            print("❌ No research results available.")
            return
        
        result = self.current_result
        
        print(f"\n🛡️ Detailed Validation Report")
        print("=" * 60)
        
        # Domain Analysis
        if result.domain_profile:
            print(f"\n📊 Domain Analysis:")
            print(f"   Detected Domain: {result.domain_profile.domain.value.replace('_', ' ').title()}")
            print(f"   Confidence: {result.domain_profile.confidence:.1%}")
            print(f"   Keywords: {', '.join(result.domain_profile.keywords[:5])}")
        
        # Cross-Validation Results
        if result.cross_validation_result:
            cv = result.cross_validation_result
            print(f"\n🔄 Cross-Source Analysis:")
            print(f"   Total Topics: {cv.total_topics}")
            print(f"   Consensus Topics: {cv.consensus_topics}")
            print(f"   Conflicted Topics: {cv.conflicted_topics}")
            print(f"   Overall Consensus: {cv.overall_consensus_score:.1%}")
        
        # Content Validation
        if result.content_validation_result:
            cv = result.content_validation_result
            print(f"\n🔗 Content-Citation Validation:")
            print(f"   Validation Score: {cv.validation_score:.1%}")
            print(f"   Supported Claims: {cv.supported_claims}")
            print(f"   Unsupported Claims: {cv.unsupported_claims}")
            if cv.overall_issues:
                print(f"   Issues Found: {len(cv.overall_issues)}")
                for issue in cv.overall_issues[:3]:
                    print(f"     • {issue}")
        
        # Search Summary
        if result.search_summary:
            print(f"\n🔍 Search Summary:")
            for key, value in result.search_summary.items():
                if isinstance(value, (int, float)):
                    print(f"   {key.replace('_', ' ').title()}: {value}")
                elif isinstance(value, str) and len(value) < 100:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
    
    def show_quality_assessment(self):
        """Display detailed quality assessment for enhanced mode."""
        if not self.enhanced_mode:
            print("❌ Quality assessment is only available in enhanced mode.")
            return
            
        if not self.current_result:
            print("❌ No research results available.")
            return
        
        result = self.current_result
        
        if not result.quality_assessment:
            print("❌ No quality assessment available.")
            return
        
        qa = result.quality_assessment
        
        print(f"\n📊 Quality Assessment Report")
        print("=" * 60)
        
        print(f"\nOverall Assessment:")
        print(f"   Quality Level: {qa.overall_quality.value.upper()}")
        print(f"   Quality Score: {qa.overall_score:.1%}")
        print(f"   Confidence Score: {result.confidence_score:.1%}")
        
        print(f"\nValidation Summary:")
        vs = qa.validation_summary
        print(f"   Total Gates: {vs.get('total_gates', 0)}")
        print(f"   Passed Gates: {vs.get('passed_gates', 0)}")
        print(f"   Critical Gates: {vs.get('critical_gates', 0)}")
        print(f"   Critical Passed: {vs.get('critical_passed', 0)}")
        print(f"   Pass Rate: {vs.get('pass_rate', 0):.1%}")
        
        # Quality Gate Results
        print(f"\nQuality Gate Results:")
        for gate_result in qa.gate_results:
            status = "✅ PASS" if gate_result.passed else "❌ FAIL"
            print(f"   {gate_result.gate_name}: {status} ({gate_result.score:.1%})")
            
            if gate_result.issues:
                for issue in gate_result.issues:
                    print(f"     ⚠️  {issue}")
        
        # Critical Issues
        if qa.critical_issues:
            print(f"\n🚨 Critical Issues:")
            for issue in qa.critical_issues:
                print(f"   • {issue}")
        
        # Recommendations
        if qa.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(qa.recommendations, 1):
                print(f"   {i}. {rec}")
    
    def show_sources(self):
        """Display detailed source information."""
        if not self.current_result:
            print("❌ No research results available.")
            return
        
        print(f"\n📚 Detailed Bibliography")
        print("=" * 50)
        
        for i, source in enumerate(self.current_result['sources'], 1):
            print(f"\n{i}. {source.get('title', 'Unknown Title')}")
            print(f"   Authors: {', '.join(source.get('authors', ['Unknown']))}")
            print(f"   Year: {source.get('year', 'Unknown')}")
            print(f"   Type: {source.get('source_type', 'Unknown')}")
            if source.get('url'):
                print(f"   URL: {source['url']}")
            if source.get('doi'):
                print(f"   DOI: {source['doi']}")
            if source.get('journal'):
                print(f"   Journal: {source['journal']}")
    
    def show_debug_analysis(self, query: str):
        """Display debug analysis for a query."""
        if not self.enhanced_mode or not self.debug_mode:
            print("❌ Debug analysis is only available in enhanced debug mode.")
            return
            
        print(f"\n🐛 Debug Analysis for Query: '{query}'")
        print("=" * 60)
        
        try:
            debug_info = self.agent.debug_validation_pipeline(query)
            
            print(f"\n📊 Query Analysis:")
            qa = debug_info['query_analysis']
            print(f"   Original: {qa['original']}")
            print(f"   Expanded: {qa['expanded']}")
            print(f"   Terms: {qa['terms']}")
            print(f"   Domain: {qa['domain_detected']}")
            
            print(f"\n🔧 Validation Settings:")
            va = debug_info['validation_analysis']
            print(f"   Relevance Threshold: {va['relevance_threshold']}")
            print(f"   Fallback Threshold: {va['fallback_threshold']}")
            print(f"   Semantic Available: {va['semantic_available']}")
            print(f"   Content Threshold: {va['content_threshold']}")
            print(f"   Consensus Threshold: {va['consensus_threshold']}")
            
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(debug_info['recommendations'], 1):
                print(f"   {i}. {rec}")
                
            print(f"\n🔍 To see detailed validation logs, run research with this query.")
            
        except Exception as e:
            print(f"❌ Debug analysis failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Autonomous Research Agent CLI")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--query", help="Research query to run non-interactively")
    parser.add_argument("--citation", choices=['apa', 'mla', 'ieee'], 
                       default='apa', help="Citation style")
    parser.add_argument("--export", choices=['txt', 'json', 'md'], 
                       help="Export format (for non-interactive mode)")
    parser.add_argument("--enhanced", action='store_true', 
                       help="Use enhanced mode with validation layers")
    parser.add_argument("--debug", action='store_true',
                       help="Enable debug mode for detailed validation logging")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OpenAI API key required.")
        print("   Set OPENAI_API_KEY environment variable or use --api-key flag.")
        print("   Get your API key at: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    # Initialize CLI
    cli = ResearchCLI(api_key, enhanced_mode=args.enhanced, debug_mode=args.debug)
    
    # Check if running in non-interactive mode
    if args.query:
        async def run_single_query():
            print(f"🔬 Research Agent - Single Query Mode")
            print(f"Query: {args.query}")
            print(f"Citation: {args.citation}")
            print("-" * 50)
            
            await cli.conduct_research(args.query, args.citation)
            
            if args.export and cli.current_result:
                cli.export_results(args.export)
        
        asyncio.run(run_single_query())
    else:
        # Run interactive mode
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
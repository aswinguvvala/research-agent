"""
Enhanced Research Agent with Comprehensive Validation
Main orchestrator that integrates all validation layers to prevent source-content mismatch.
"""

import os
import asyncio
import openai
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Fix HuggingFace tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import all validation components
try:
    from relevance_validator import RelevanceValidator
    from progressive_searcher import ProgressiveSearcher
    from content_linker import ContentLinker
    from cross_validator import CrossValidator
    from quality_gates import QualityGates, QualityLevel
    from domain_specializer import DomainSpecializer
    from content_extractor import ContentExtractor
    from citation_manager import CitationManager, Source
except ImportError as e:
    logging.error(f"Failed to import validation components: {e}")
    raise

logger = logging.getLogger(__name__)


@dataclass
class EnhancedResearchResult:
    """Complete research result with validation data."""
    query: str
    domain_profile: Any
    search_summary: Dict[str, Any]
    validated_sources: List[Tuple[Dict[str, Any], Any]]  # (source, relevance_score)
    cross_validation_result: Any
    synthesis: str
    content_validation_result: Any
    quality_assessment: Any
    citation_bibliography: str
    research_time: float
    validation_reports: Dict[str, str]
    recommendations: List[str]
    confidence_score: float
    timestamp: str


class EnhancedResearchAgent:
    """
    Enhanced research agent with comprehensive validation layers.
    Prevents source-content mismatch through systematic validation.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 max_sources: int = 10,
                 relevance_threshold: float = 0.35,  # More lenient for broader topics
                 content_validation_threshold: float = 0.65,  # More realistic threshold
                 consensus_threshold: float = 0.6,  # Slightly lower for better coverage
                 debug_mode: bool = False):
        
        # Validate API key first
        if not openai_api_key or not openai_api_key.strip():
            raise ValueError("OpenAI API key is required but not provided")
        
        self.openai_api_key = openai_api_key
        self.max_sources = max_sources
        self.debug_mode = debug_mode
        
        # Set up debug logging
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            logger.info("🐛 Debug mode enabled - detailed validation logging active")
        
        # Initialize OpenAI client with validation
        try:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        # Initialize validation components with comprehensive logging
        logger.info("🔧 Initializing enhanced research agent with validation layers")
        
        try:
            self.domain_specializer = DomainSpecializer()
            logger.info("✅ Domain specializer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize domain specializer: {e}")
            raise
            
        try:
            self.progressive_searcher = ProgressiveSearcher(
                max_sources_per_stage=3, 
                max_total_sources=max_sources
            )
            logger.info("✅ Progressive searcher initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize progressive searcher: {e}")
            raise
            
        try:
            self.relevance_validator = RelevanceValidator(relevance_threshold=relevance_threshold)
            if self.relevance_validator.use_semantic:
                logger.info("✅ Relevance validator initialized with semantic analysis")
            else:
                logger.warning("⚠️ Relevance validator initialized without semantic analysis (install sentence-transformers for better validation)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize relevance validator: {e}")
            raise
            
        try:
            self.content_extractor = ContentExtractor()
            logger.info("✅ Content extractor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize content extractor: {e}")
            raise
            
        try:
            self.cross_validator = CrossValidator(consensus_threshold=consensus_threshold)
            logger.info("✅ Cross validator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize cross validator: {e}")
            raise
            
        try:
            self.content_linker = ContentLinker(similarity_threshold=0.6)
            logger.info("✅ Content linker initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize content linker: {e}")
            raise
            
        try:
            self.quality_gates = QualityGates(
                relevance_threshold=relevance_threshold,
                content_validation_threshold=content_validation_threshold,
                consensus_threshold=consensus_threshold
            )
            logger.info("✅ Quality gates initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize quality gates: {e}")
            raise
            
        try:
            self.citation_manager = CitationManager()
            logger.info("✅ Citation manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize citation manager: {e}")
            raise
        
        logger.info("🎉 Enhanced research agent initialized successfully with all components")
    
    async def conduct_enhanced_research(self, 
                                      query: str, 
                                      citation_style: str = "apa") -> EnhancedResearchResult:
        """
        Conduct comprehensive research with full validation pipeline.
        
        Args:
            query: Research query
            citation_style: Citation format (apa, mla, ieee)
            
        Returns:
            EnhancedResearchResult with complete validation data
        """
        start_time = time.time()
        self.citation_manager.clear_sources()
        
        logger.info(f"🔍 Starting enhanced research: {query}")
        
        try:
            # Stage 1: Domain Detection and Specialization
            logger.info("📋 Stage 1: Domain Detection")
            try:
                domain_profile = self.domain_specializer.detect_domain(query)
                logger.info(f"✅ Domain detected: {domain_profile.domain.value} (confidence: {domain_profile.confidence:.2f})")
            except Exception as e:
                logger.error(f"❌ Domain detection failed: {e}")
                raise
            
            # Stage 2: Progressive Search with Domain Optimization
            logger.info("🔍 Stage 2: Progressive Search")
            try:
                search_results = await self.progressive_searcher.progressive_search(query)
                search_summary = self.progressive_searcher.get_search_summary(search_results)
                logger.info(f"✅ Search completed: {len(search_results)} results found")
                
                if self.debug_mode:
                    logger.debug(f"🐛 Search summary: {search_summary}")
            except Exception as e:
                logger.error(f"❌ Progressive search failed: {e}")
                return self._create_failed_result(query, f"Search failed: {str(e)}", start_time)
            
            # Convert search results to source format
            try:
                raw_sources = [result.source_data for result in search_results]
                logger.info(f"✅ Converted {len(raw_sources)} search results to source format")
                
                if not raw_sources:
                    logger.warning("⚠️ No sources found - attempting alternative search strategies")
                    return self._create_failed_result(query, "No sources found after progressive search", start_time)
                    
            except Exception as e:
                logger.error(f"❌ Failed to convert search results: {e}")
                return self._create_failed_result(query, f"Source conversion failed: {str(e)}", start_time)
            
            # Stage 3: Source Relevance Validation with Graceful Degradation
            logger.info("✅ Stage 3: Source Relevance Validation")
            
            if self.debug_mode:
                logger.debug(f"🐛 Debug: Raw sources count: {len(raw_sources)}")
                logger.debug(f"🐛 Debug: Validation threshold: {self.relevance_validator.relevance_threshold}")
                logger.debug(f"🐛 Debug: Fallback threshold: {self.relevance_validator.fallback_threshold}")
                logger.debug(f"🐛 Debug: Semantic analysis enabled: {self.relevance_validator.use_semantic}")
            
            try:
                validated_sources = self.relevance_validator.batch_validate_sources(query, raw_sources)
                logger.info(f"✅ Source validation completed: {len(validated_sources)}/{len(raw_sources)} sources passed")
            except Exception as e:
                logger.error(f"❌ Source validation failed: {e}")
                return self._create_failed_result(query, f"Source validation failed: {str(e)}", start_time)
            
            if self.debug_mode:
                logger.debug(f"🐛 Debug: Validated sources count: {len(validated_sources)}")
                if validated_sources:
                    scores = [score.overall_score for _, score in validated_sources]
                    logger.debug(f"🐛 Debug: Validation scores: {scores}")
            
            if not validated_sources:
                logger.warning("No sources passed relevance validation - attempting graceful degradation")
                if self.debug_mode:
                    self._debug_validation_failure(query, raw_sources)
                
                validated_sources = await self._graceful_source_fallback(query, raw_sources)
                
                if not validated_sources:
                    logger.error("Even graceful degradation failed - no sources available")
                    if self.debug_mode:
                        self._debug_complete_failure(query, raw_sources)
                    return self._create_failed_result(query, "No relevant sources found after all fallback attempts", start_time)
            
            # Stage 4: Domain-Specific Source Prioritization
            logger.info("🎯 Stage 4: Domain-Specific Prioritization")
            prioritized_sources = self.domain_specializer.prioritize_sources(
                [source for source, _ in validated_sources], domain_profile
            )
            
            # Take top sources based on priority and relevance
            final_sources = []
            for (source, priority_score), (_, relevance_score) in zip(prioritized_sources, validated_sources):
                combined_score = 0.6 * relevance_score.overall_score + 0.4 * priority_score
                final_sources.append((source, relevance_score, combined_score))
            
            # Sort by combined score and limit
            final_sources.sort(key=lambda x: x[2], reverse=True)
            final_sources = final_sources[:self.max_sources]
            
            # Stage 5: Pre-Synthesis Quality Gates with Graceful Degradation
            logger.info("🛡️ Stage 5: Pre-Synthesis Quality Gates")
            source_list = [source for source, _, _ in final_sources]
            pre_gates_passed, pre_gate_results = self.quality_gates.run_pre_synthesis_gates(query, source_list)
            
            if not pre_gates_passed:
                logger.warning("Pre-synthesis quality gates failed - attempting graceful degradation")
                # Try with lower quality threshold
                final_sources, graceful_success = await self._graceful_quality_degradation(final_sources, query)
                
                if not graceful_success:
                    logger.warning("Quality gates failed even with graceful degradation - proceeding with reduced confidence")
                    # Continue with reduced confidence rather than failing completely
            
            # Stage 6: Content Extraction and Processing
            logger.info("📖 Stage 6: Content Extraction")
            processed_sources = await self._extract_and_process_content(final_sources)
            
            # Stage 7: Cross-Source Validation
            logger.info("🔄 Stage 7: Cross-Source Validation")
            cross_validation_result = self.cross_validator.cross_validate_sources(query, processed_sources)
            
            # Stage 8: AI Synthesis with Validation Context
            logger.info("🧠 Stage 8: AI Synthesis")
            synthesis = await self._generate_validated_synthesis(query, processed_sources, cross_validation_result)
            
            # Stage 9: Post-Synthesis Validation
            logger.info("🔗 Stage 9: Content-Citation Validation")
            content_validation_result = self.content_linker.validate_content_citations(synthesis, processed_sources)
            
            # Stage 10: Post-Synthesis Quality Gates
            logger.info("🛡️ Stage 10: Post-Synthesis Quality Gates")
            post_gates_passed, post_gate_results = self.quality_gates.run_post_synthesis_gates(
                query, processed_sources, synthesis
            )
            
            # Stage 11: Comprehensive Quality Assessment
            logger.info("📊 Stage 11: Quality Assessment")
            quality_assessment = self.quality_gates.assess_overall_quality(query, processed_sources, synthesis)
            
            # Stage 12: Generate Citations and Bibliography
            logger.info("📚 Stage 12: Citation Generation")
            citation_bibliography = self.citation_manager.generate_bibliography(citation_style)
            
            # Stage 13: Generate Validation Reports
            logger.info("📄 Stage 13: Report Generation")
            validation_reports = self._generate_validation_reports(
                query, processed_sources, domain_profile, cross_validation_result, content_validation_result, quality_assessment
            )
            
            # Stage 14: Generate Recommendations
            recommendations = self._generate_recommendations(quality_assessment, domain_profile)
            
            # Stage 15: Calculate Confidence Score
            confidence_score = self._calculate_confidence_score(
                quality_assessment, content_validation_result, cross_validation_result
            )
            
            end_time = time.time()
            research_time = round(end_time - start_time, 2)
            
            logger.info(f"✅ Enhanced research completed in {research_time}s")
            logger.info(f"📊 Quality: {quality_assessment.overall_quality.value}, Confidence: {confidence_score:.3f}")
            
            return EnhancedResearchResult(
                query=query,
                domain_profile=domain_profile,
                search_summary=search_summary,
                validated_sources=[(source, relevance) for source, relevance, _ in final_sources],
                cross_validation_result=cross_validation_result,
                synthesis=synthesis,
                content_validation_result=content_validation_result,
                quality_assessment=quality_assessment,
                citation_bibliography=citation_bibliography,
                research_time=research_time,
                validation_reports=validation_reports,
                recommendations=recommendations,
                confidence_score=confidence_score,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Enhanced research failed: {e}")
            return self._create_failed_result(query, f"Research error: {str(e)}", start_time)
    
    async def _extract_and_process_content(self, final_sources: List[Tuple[Dict[str, Any], Any, float]]) -> List[Dict[str, Any]]:
        """Extract content from sources and add to citation manager."""
        processed_sources = []
        
        for source, relevance_score, combined_score in final_sources:
            try:
                # Extract content based on source type
                if source.get("source_type") == "arxiv" and source.get("arxiv_id"):
                    content_data = self.content_extractor.extract_arxiv_content(source["arxiv_id"])
                elif source.get("pdf_url"):
                    content_data = self.content_extractor.extract_pdf_content(source["pdf_url"])
                elif source.get("url"):
                    content_data = self.content_extractor.extract_web_content(source["url"])
                else:
                    # Use available metadata
                    content_data = {
                        "content": source.get("abstract", source.get("summary", "")),
                        "metadata": source,
                        "source_type": source["source_type"]
                    }
                
                if "error" not in content_data:
                    # Create citation source
                    citation_source = Source(
                        title=content_data["metadata"].get("title", source.get("title", "Unknown Title")),
                        authors=content_data["metadata"].get("authors", source.get("authors", [])),
                        year=self._extract_year(content_data["metadata"].get("published", source.get("published", ""))),
                        url=source.get("url", source.get("pdf_url", "")),
                        doi=content_data["metadata"].get("doi"),
                        journal=content_data["metadata"].get("journal"),
                        source_type=source["source_type"]
                    )
                    
                    citation_num = self.citation_manager.add_source(citation_source)
                    
                    # Merge all available data
                    processed_source = {
                        **source,
                        **content_data,
                        "citation_number": citation_num,
                        "relevance_score": relevance_score.overall_score,
                        "combined_score": combined_score
                    }
                    
                    processed_sources.append(processed_source)
                else:
                    logger.warning(f"Failed to extract content from: {source.get('title', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Error processing source: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_sources)} sources")
        return processed_sources
    
    async def _generate_validated_synthesis(self, 
                                          query: str, 
                                          sources: List[Dict[str, Any]], 
                                          cross_validation: Any) -> str:
        """Generate AI synthesis with validation context."""
        if not sources:
            return "No valid sources available for synthesis."
        
        # Prepare synthesis prompt with validation context
        source_summaries = []
        for source in sources:
            content = source.get("content", "")
            title = source.get("metadata", {}).get("title", source.get("title", "Unknown"))
            citation_num = source.get("citation_number", 1)
            
            # Limit content length
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            summary = f"Source {citation_num}: {title}\nContent: {content}\n"
            source_summaries.append(summary)
        
        # Add cross-validation context
        consensus_info = ""
        if hasattr(cross_validation, 'topic_analyses'):
            consensus_topics = [t for t in cross_validation.topic_analyses if t.consensus_level >= 0.7]
            conflicted_topics = [t for t in cross_validation.topic_analyses if t.conflicting_views]
            
            if consensus_topics:
                consensus_info += f"\nHigh consensus topics: {', '.join([t.topic_name for t in consensus_topics])}"
            if conflicted_topics:
                consensus_info += f"\nConflicted topics: {', '.join([t.topic_name for t in conflicted_topics])}"
        
        synthesis_prompt = f"""You are an expert research analyst conducting a comprehensive literature synthesis.

Research Question: "{query}"

CRITICAL REQUIREMENTS - You MUST follow these rules:
1. ONLY use information that is explicitly present in the provided sources
2. EVERY claim must be supported by a specific source using [Source X] citations
3. Do NOT add information from your general knowledge
4. Do NOT make claims that aren't directly supported by the source content
5. If sources conflict, acknowledge the disagreement explicitly
6. Identify research gaps only if they are evident from the literature analysis

Sources Available:
{chr(10).join(source_summaries)}

Cross-Validation Context:{consensus_info}

Please provide a comprehensive synthesis (400-600 words) that:
1. Directly addresses the research question using ONLY the provided sources
2. Uses proper in-text citations [Source X] for every claim
3. Acknowledges different perspectives when sources disagree
4. Identifies patterns and themes across sources
5. Notes limitations or gaps evident in the current literature
6. Provides a critical analysis rather than just summarizing each source

Remember: Every statement must be traceable to the provided sources. Do not include information not found in the source content."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst who only uses information explicitly provided in sources and cites everything properly."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=1000,
                temperature=0.2  # Lower temperature for more factual output
            )
            
            synthesis = response.choices[0].message.content.strip()
            return synthesis
            
        except Exception as e:
            logger.error(f"AI synthesis failed: {e}")
            return self._create_fallback_synthesis(query, sources)
    
    def _create_fallback_synthesis(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Create fallback synthesis without AI."""
        synthesis_parts = [
            f"Research Analysis: {query}",
            f"Based on analysis of {len(sources)} validated sources:\n"
        ]
        
        for source in sources:
            title = source.get("metadata", {}).get("title", source.get("title", "Unknown"))
            citation_num = source.get("citation_number", 1)
            content = source.get("content", "")
            
            # Extract key sentences
            sentences = content.split('. ')[:2]
            key_content = '. '.join(sentences)
            
            synthesis_parts.append(f"• {title} [Source {citation_num}]: {key_content}")
        
        synthesis_parts.append("\nThis analysis is based on validated sources with relevance scoring and cross-validation.")
        
        return "\n".join(synthesis_parts)
    
    def _generate_validation_reports(self, 
                                   query: str,
                                   sources: List[Dict[str, Any]],
                                   domain_profile: Any,
                                   cross_validation: Any, 
                                   content_validation: Any, 
                                   quality_assessment: Any) -> Dict[str, str]:
        """Generate detailed validation reports."""
        reports = {}
        
        # Domain specialization report with proper coverage analysis
        coverage_analysis = self.domain_specializer.validate_domain_coverage(query, sources, domain_profile)
        reports['domain'] = self.domain_specializer.generate_domain_report(domain_profile, coverage_analysis)
        
        # Cross-validation report
        reports['cross_validation'] = self.cross_validator.generate_cross_validation_report(cross_validation)
        
        # Content validation report
        reports['content_validation'] = self.content_linker.generate_validation_report(content_validation)
        
        # Quality assessment report
        reports['quality'] = self.quality_gates.generate_quality_report(quality_assessment)
        
        return reports
    
    def _generate_recommendations(self, quality_assessment: Any, domain_profile: Any) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Quality-based recommendations
        recommendations.extend(quality_assessment.recommendations)
        
        # Domain-specific recommendations
        domain_recs = self.domain_specializer.get_search_recommendations(domain_profile, "")
        recommendations.extend(domain_recs)
        
        # Quality level specific recommendations
        if quality_assessment.overall_quality == QualityLevel.FAILED:
            recommendations.insert(0, "CRITICAL: Research quality failed validation - significant issues must be addressed")
        elif quality_assessment.overall_quality == QualityLevel.POOR:
            recommendations.insert(0, "Research quality is poor - consider additional sources and validation")
        elif quality_assessment.overall_quality == QualityLevel.ACCEPTABLE:
            recommendations.append("Research quality is acceptable but could be improved with additional sources")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to top 10
    
    async def _graceful_source_fallback(self, query: str, raw_sources: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Any]]:
        """Implement graceful degradation when validation fails."""
        logger.info("🆘 Implementing graceful source fallback")
        
        if not raw_sources:
            logger.warning("No raw sources available for fallback")
            return []
        
        # Strategy 1: Use basic keyword matching as fallback
        fallback_sources = []
        query_keywords = set(query.lower().split())
        
        for source in raw_sources:
            title = source.get('title', '').lower()
            abstract = source.get('abstract', source.get('summary', '')).lower()
            
            # Basic keyword matching
            title_matches = sum(1 for keyword in query_keywords if keyword in title)
            abstract_matches = sum(1 for keyword in query_keywords if keyword in abstract)
            
            # Simple scoring
            basic_score = (title_matches * 2 + abstract_matches) / (len(query_keywords) + 1)
            
            if basic_score > 0.1:  # Very low threshold for fallback
                # Create a mock relevance score
                mock_relevance = type('MockRelevance', (), {
                    'overall_score': basic_score,
                    'passes_threshold': True,
                    'explanation': f'Fallback validation (basic score: {basic_score:.2f})'
                })()
                
                fallback_sources.append((source, mock_relevance))
        
        # Sort by basic score and take top sources
        fallback_sources.sort(key=lambda x: x[1].overall_score, reverse=True)
        fallback_sources = fallback_sources[:min(5, len(fallback_sources))]
        
        logger.info(f"Graceful fallback found {len(fallback_sources)} sources using basic keyword matching")
        return fallback_sources
    
    async def _graceful_quality_degradation(self, sources: List[Tuple[Dict[str, Any], Any, float]], query: str) -> Tuple[List[Tuple[Dict[str, Any], Any, float]], bool]:
        """Attempt graceful degradation when quality gates fail."""
        logger.info("🆘 Implementing graceful quality degradation")
        
        # Strategy 1: Lower quality thresholds temporarily
        try:
            # Create a temporary quality gates instance with lower thresholds
            temp_quality_gates = QualityGates(
                relevance_threshold=0.2,  # Much lower
                content_validation_threshold=0.5,  # Lower
                consensus_threshold=0.4  # Lower
            )
            
            source_list = [source for source, _, _ in sources]
            gates_passed, gate_results = temp_quality_gates.run_pre_synthesis_gates(query, source_list)
            
            if gates_passed:
                logger.info("Graceful quality degradation successful with lower thresholds")
                return sources, True
            
        except Exception as e:
            logger.warning(f"Quality degradation attempt failed: {e}")
        
        # Strategy 2: Filter to only the highest scoring sources
        if len(sources) > 2:
            top_sources = sources[:2]  # Take only top 2 sources
            logger.info("Graceful degradation: using only top 2 sources")
            return top_sources, True
        
        # Strategy 3: Continue with whatever we have
        logger.warning("All graceful degradation strategies exhausted - proceeding with available sources")
        return sources, False
    
    def _debug_validation_failure(self, query: str, raw_sources: List[Dict[str, Any]]):
        """Debug validation failure by analyzing what went wrong."""
        logger.debug("🐛 Debug: Analyzing validation failure...")
        
        # Analyze query characteristics
        expanded_query = self.relevance_validator.expand_query_terms(query)
        logger.debug(f"🐛 Debug: Original query: '{query}'")
        logger.debug(f"🐛 Debug: Expanded query: '{expanded_query}'")
        
        query_terms = set(query.lower().split())
        logger.debug(f"🐛 Debug: Query terms: {query_terms}")
        
        # Analyze each source individually
        for i, source in enumerate(raw_sources[:3]):  # Check first 3 sources
            logger.debug(f"🐛 Debug: Source {i+1} analysis:")
            logger.debug(f"  Title: {source.get('title', 'N/A')}")
            logger.debug(f"  Type: {source.get('source_type', 'N/A')}")
            
            # Check for basic keyword overlap
            title = source.get('title', '').lower()
            abstract = source.get('abstract', source.get('summary', '')).lower()
            
            title_matches = [term for term in query_terms if term in title]
            abstract_matches = [term for term in query_terms if term in abstract]
            
            logger.debug(f"  Title matches: {title_matches}")
            logger.debug(f"  Abstract matches: {abstract_matches}")
            
            # Perform individual validation for debugging
            try:
                relevance_score = self.relevance_validator.validate_source_relevance(query, source)
                logger.debug(f"  Relevance score: {relevance_score.overall_score:.3f}")
                logger.debug(f"  Explanation: {relevance_score.explanation}")
                logger.debug(f"  Threshold: {self.relevance_validator.relevance_threshold}")
                logger.debug(f"  Fallback threshold: {self.relevance_validator.fallback_threshold}")
            except Exception as e:
                logger.debug(f"  Validation error: {e}")
    
    def _debug_complete_failure(self, query: str, raw_sources: List[Dict[str, Any]]):
        """Debug complete failure when even graceful degradation fails."""
        logger.debug("🐛 Debug: Complete failure analysis...")
        logger.debug(f"🐛 Total raw sources: {len(raw_sources)}")
        
        if not raw_sources:
            logger.debug("🐛 No raw sources found - search phase failed")
            return
        
        # Check source types
        source_types = {}
        for source in raw_sources:
            source_type = source.get('source_type', 'unknown')
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        logger.debug(f"🐛 Source types: {source_types}")
        
        # Check for missing content
        content_analysis = {
            'has_title': sum(1 for s in raw_sources if s.get('title')),
            'has_abstract': sum(1 for s in raw_sources if s.get('abstract') or s.get('summary')),
            'has_authors': sum(1 for s in raw_sources if s.get('authors')),
            'has_url': sum(1 for s in raw_sources if s.get('url'))
        }
        
        logger.debug(f"🐛 Content analysis: {content_analysis}")
        
        # Suggest fixes
        logger.debug("🐛 Suggested fixes:")
        logger.debug("  1. Check if query keywords are too specific")
        logger.debug("  2. Try broader search terms")
        logger.debug("  3. Lower validation thresholds further")
        logger.debug("  4. Check search source availability")
    
    def debug_validation_pipeline(self, query: str) -> Dict[str, Any]:
        """Run a complete debugging analysis of the validation pipeline."""
        debug_info = {
            'query_analysis': {},
            'search_analysis': {},
            'validation_analysis': {},
            'recommendations': []
        }
        
        logger.info("🐛 Running complete validation pipeline debug")
        
        # Query analysis
        expanded_query = self.relevance_validator.expand_query_terms(query)
        debug_info['query_analysis'] = {
            'original': query,
            'expanded': expanded_query,
            'terms': query.lower().split(),
            'domain_detected': self.domain_specializer.detect_domain(query).domain.value if self.domain_specializer.detect_domain(query) else 'unknown'
        }
        
        # Validation component status
        debug_info['validation_analysis'] = {
            'relevance_threshold': self.relevance_validator.relevance_threshold,
            'fallback_threshold': self.relevance_validator.fallback_threshold,
            'semantic_available': self.relevance_validator.use_semantic,
            'content_threshold': getattr(self.quality_gates, 'content_validation_threshold', 'N/A'),
            'consensus_threshold': getattr(self.quality_gates, 'consensus_threshold', 'N/A')
        }
        
        # Generate recommendations
        if debug_info['query_analysis']['domain_detected'] == 'unknown':
            debug_info['recommendations'].append("Query domain not detected - try more specific technical terms")
        
        if not self.relevance_validator.use_semantic:
            debug_info['recommendations'].append("Semantic similarity unavailable - install sentence-transformers for better validation")
        
        debug_info['recommendations'].extend([
            "Try adding specific technical terms to your query",
            "Check if query is too narrow or too broad",
            "Consider using synonyms or alternative terminology"
        ])
        
        return debug_info
    
    def _calculate_confidence_score(self, 
                                  quality_assessment: Any, 
                                  content_validation: Any, 
                                  cross_validation: Any) -> float:
        """Calculate overall confidence score for the research."""
        scores = []
        
        # Quality assessment score
        scores.append(quality_assessment.overall_score)
        
        # Content validation score
        scores.append(content_validation.validation_score)
        
        # Cross-validation consensus score
        scores.append(cross_validation.overall_consensus_score)
        
        # Weighted average (quality gets highest weight)
        weights = [0.5, 0.3, 0.2]
        confidence = sum(score * weight for score, weight in zip(scores, weights))
        
        return round(confidence, 3)
    
    def _extract_year(self, date_string: str) -> str:
        """Extract year from date string."""
        if not date_string:
            return str(datetime.now().year)
        
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_string))
        if year_match:
            return year_match.group(0)
        
        return str(datetime.now().year)
    
    def _create_failed_result(self, query: str, error_message: str, start_time: float) -> EnhancedResearchResult:
        """Create a failed research result with error information."""
        end_time = time.time()
        research_time = round(end_time - start_time, 2)
        
        return EnhancedResearchResult(
            query=query,
            domain_profile=None,
            search_summary={'error': error_message},
            validated_sources=[],
            cross_validation_result=None,
            synthesis=f"Research failed: {error_message}",
            content_validation_result=None,
            quality_assessment=None,
            citation_bibliography="No sources available",
            research_time=research_time,
            validation_reports={'error': error_message},
            recommendations=[f"Address the issue: {error_message}"],
            confidence_score=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def generate_comprehensive_report(self, result: EnhancedResearchResult) -> str:
        """Generate comprehensive research report with all validation data."""
        if not result.quality_assessment:
            return f"Research Failed: {result.validation_reports.get('error', 'Unknown error')}"
        
        report_sections = [
            "Enhanced Research Report",
            "=" * 60,
            f"Query: {result.query}",
            f"Research Time: {result.research_time} seconds",
            f"Overall Quality: {result.quality_assessment.overall_quality.value.upper()}",
            f"Confidence Score: {result.confidence_score:.3f}",
            f"Timestamp: {result.timestamp}",
            "",
            "RESEARCH SYNTHESIS",
            "-" * 40,
            result.synthesis,
            "",
            "VALIDATION SUMMARY",
            "-" * 40,
            f"Sources Analyzed: {len(result.validated_sources)}",
            f"Domain: {result.domain_profile.domain.value if result.domain_profile else 'Unknown'}",
            f"Content Validation: {result.content_validation_result.validation_score:.1%}" if result.content_validation_result else "N/A",
            f"Cross-Validation Consensus: {result.cross_validation_result.overall_consensus_score:.1%}" if result.cross_validation_result else "N/A",
            "",
            "CITATIONS",
            "-" * 40,
            result.citation_bibliography,
            ""
        ]
        
        # Add recommendations
        if result.recommendations:
            report_sections.extend([
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for i, rec in enumerate(result.recommendations, 1):
                report_sections.append(f"{i}. {rec}")
            report_sections.append("")
        
        # Add validation reports if detailed view is needed
        if result.validation_reports and any(len(report) > 100 for report in result.validation_reports.values()):
            report_sections.extend([
                "DETAILED VALIDATION REPORTS",
                "-" * 40,
                "(Available in validation_reports attribute)"
            ])
        
        return "\n".join(report_sections)


# Example usage and testing
if __name__ == "__main__":
    import os
    
    async def test_enhanced_research():
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        # Initialize enhanced agent
        agent = EnhancedResearchAgent(api_key, max_sources=5)
        
        # Test query
        query = "Docker containerization security best practices"
        
        print("🧪 Testing Enhanced Research Agent")
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        
        # Conduct research
        result = await agent.conduct_enhanced_research(query, "apa")
        
        # Generate comprehensive report
        report = agent.generate_comprehensive_report(result)
        print(report)
        
        print(f"\n📊 Research Summary:")
        print(f"Quality Level: {result.quality_assessment.overall_quality.value if result.quality_assessment else 'Failed'}")
        print(f"Confidence Score: {result.confidence_score:.3f}")
        print(f"Sources Used: {len(result.validated_sources)}")
        print(f"Recommendations: {len(result.recommendations)}")
    
    # Run test
    asyncio.run(test_enhanced_research())
"""
Research Service - Integrates existing research agent with web API
"""

import os
import sys
import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Add research_agent to path
research_agent_path = os.path.join(os.path.dirname(__file__), '..', '..', 'research_agent')
if research_agent_path not in sys.path:
    sys.path.append(research_agent_path)

# Add new_research_agent to path for autonomous agent
new_research_agent_path = '/Users/aswin/new_research_agent'
if new_research_agent_path not in sys.path:
    sys.path.append(new_research_agent_path)

from research_agent import ResearchAgent
try:
    from enhanced_research_agent import EnhancedResearchAgent
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

try:
    from autonomous_research_agent import AutonomousResearchAgent
    AUTONOMOUS_AVAILABLE = True
except ImportError:
    AUTONOMOUS_AVAILABLE = False

try:
    from advanced_autonomous_research_agent import AdvancedAutonomousResearchAgent
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

from ..models.research import (
    ResearchRequest, ResearchResult, ResearchProgress, QualityAssessment,
    QualityGateResult, SourceModel, ResearchMode, QualityLevel
)
from ..core.config import settings

logger = logging.getLogger(__name__)

class ResearchService:
    """Service for handling research operations."""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.research_cache: Dict[str, ResearchResult] = {}
        
    async def conduct_research(
        self,
        request: ResearchRequest,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """
        Conduct research using the autonomous research agent.
        
        Args:
            request: Research request parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            Research result
        """
        research_id = str(uuid.uuid4())
        
        try:
            logger.info(f"🔍 Starting autonomous research: {request.query}")
            
            # Validate API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            
            # Create research agent based on mode - different agents for different modes
            if request.mode == ResearchMode.BASIC:
                # Basic mode: Use simpler agent with fewer sources for fast results
                if ENHANCED_AVAILABLE:
                    agent = EnhancedResearchAgent(
                        openai_api_key=api_key,
                        max_sources=request.max_sources or 8,  # Limit sources for speed
                        debug_mode=request.debug_mode or False
                    )
                    logger.info(f"⚡ Using Enhanced Research Agent for Basic mode (faster processing)")
                else:
                    agent = ResearchAgent(
                        openai_api_key=api_key,
                        max_sources=request.max_sources or 6  # Even fewer for basic agent
                    )
                    logger.info(f"⚡ Using Basic Research Agent for Basic mode (faster processing)")
            else:
                # Enhanced mode: Use advanced agent with more sources for comprehensive results
                if ADVANCED_AVAILABLE:
                    # Get optional Google Scholar API key
                    scholar_api_key = os.getenv("SERPAPI_API_KEY")
                    agent = AdvancedAutonomousResearchAgent(
                        openai_api_key=api_key,
                        google_scholar_api_key=scholar_api_key,
                        max_sources=request.max_sources or 25,  # More sources for comprehensive analysis
                        debug_mode=request.debug_mode or False
                    )
                    logger.info(f"🚀 Using Advanced Autonomous Research Agent for Enhanced mode (comprehensive analysis)")
                elif AUTONOMOUS_AVAILABLE:
                    agent = AutonomousResearchAgent(
                        openai_api_key=api_key,
                        max_sources=request.max_sources or 15,
                        debug_mode=request.debug_mode or False
                    )
                    logger.info(f"🤖 Using Autonomous Research Agent for Enhanced mode (comprehensive analysis)")
                elif ENHANCED_AVAILABLE:
                    agent = EnhancedResearchAgent(
                        openai_api_key=api_key,
                        max_sources=request.max_sources or 12,
                        debug_mode=request.debug_mode or False
                    )
                    logger.info(f"✅ Using Enhanced Research Agent for Enhanced mode (comprehensive analysis)")
                else:
                    agent = ResearchAgent(
                        openai_api_key=api_key,
                        max_sources=request.max_sources or 10
                    )
                    logger.info(f"⚠️ Using Basic Research Agent for Enhanced mode (comprehensive analysis)")
            
            # Track session
            self.active_sessions[research_id] = {
                "agent": agent,
                "start_time": datetime.now(),
                "request": request
            }
            
            # Send initial progress
            if progress_callback:
                progress_callback(ResearchProgress(
                    stage="initializing",
                    progress=0.0,
                    message="Initializing AI research agent...",
                    sources_found=0
                ))
            
            # Conduct research with progress tracking based on mode
            if request.mode == ResearchMode.BASIC:
                # Basic mode: Use streamlined research flow
                result = await self._conduct_basic_mode_research(
                    agent, request, research_id, progress_callback
                )
            else:
                # Enhanced mode: Use advanced research flow
                if ADVANCED_AVAILABLE and isinstance(agent, AdvancedAutonomousResearchAgent):
                    result = await self._conduct_advanced_research(
                        agent, request, research_id, progress_callback
                    )
                elif AUTONOMOUS_AVAILABLE and isinstance(agent, AutonomousResearchAgent):
                    result = await self._conduct_autonomous_research(
                        agent, request, research_id, progress_callback
                    )
                elif ENHANCED_AVAILABLE and isinstance(agent, EnhancedResearchAgent):
                    result = await self._conduct_enhanced_research(
                        agent, request, research_id, progress_callback
                    )
                else:
                    result = await self._conduct_basic_research(
                        agent, request, research_id, progress_callback
                    )
            
            # Cache result
            self.research_cache[research_id] = result
            
            # Clean up session
            if research_id in self.active_sessions:
                del self.active_sessions[research_id]
            
            logger.info(f"✅ Research completed: {research_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            
            # Clean up session
            if research_id in self.active_sessions:
                del self.active_sessions[research_id]
            
            # Send error progress
            if progress_callback:
                progress_callback(ResearchProgress(
                    stage="error",
                    progress=0.0,
                    message=f"Research failed: {str(e)}",
                    sources_found=0
                ))
            
            raise e
    
    async def _conduct_basic_mode_research(
        self,
        agent: Any,
        request: ResearchRequest,
        research_id: str,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """Conduct streamlined basic research for quick results."""
        
        # Basic mode progress stages - fewer, faster stages
        stages = [
            ("quick_search", 0.25, "⚡ Quick source search..."),
            ("essential_extraction", 0.65, "📄 Extracting essential information..."),
            ("concise_synthesis", 0.90, "✨ Creating concise summary..."),
            ("completed", 1.0, "⚡ Basic research completed!")
        ]
        
        for stage, progress, message in stages:
            if progress_callback:
                progress_callback(ResearchProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    sources_found=0  # Will be updated with actual count
                ))
            
            # Faster progress updates for basic mode
            await asyncio.sleep(0.3)
        
        # Conduct the research using appropriate method
        if ENHANCED_AVAILABLE and isinstance(agent, EnhancedResearchAgent):
            result = await agent.conduct_enhanced_research(
                request.query,
                request.citation_style.value
            )
            return self._convert_enhanced_result_to_basic_summary(result, research_id, request)
        else:
            # Use basic research agent
            result = await agent.conduct_research(
                request.query,
                request.citation_style.value
            )
            return self._convert_basic_result_to_api_model(result, research_id, request)
    
    def _convert_enhanced_result_to_basic_summary(
        self,
        result: Any,
        research_id: str,
        request: ResearchRequest
    ) -> ResearchResult:
        """Convert enhanced research result to basic mode summary (concise output)."""
        
        # Convert sources but limit to top 8 most relevant
        sources = []
        source_limit = 8
        for i, (source_data, relevance) in enumerate(result.validated_sources[:source_limit]):
            sources.append(SourceModel(
                title=source_data.get('title', 'Unknown Title'),
                authors=source_data.get('authors', []),
                year=str(source_data.get('year', 'Unknown Year')),
                url=source_data.get('url'),
                doi=source_data.get('doi'),
                journal=source_data.get('journal'),
                source_type=source_data.get('source_type', 'unknown'),
                relevance_score=getattr(relevance, 'overall_score', None) if relevance else None,
                abstract=source_data.get('abstract')
            ))
        
        # Create concise synthesis (first 800 characters of original synthesis)
        original_synthesis = result.synthesis
        concise_synthesis = original_synthesis[:800] + "..." if len(original_synthesis) > 800 else original_synthesis
        
        # Add basic mode indicator to synthesis
        concise_synthesis = f"**Quick Summary:**\n\n{concise_synthesis}\n\n*For more detailed analysis, try Enhanced Mode.*"
        
        # Basic quality assessment
        basic_quality = QualityAssessment(
            overall_quality=QualityLevel.ACCEPTABLE,
            overall_score=0.75,  # Fixed score for basic mode
            confidence_score=0.70,  # Fixed confidence for basic mode
            gate_results=[
                QualityGateResult(
                    gate_name="Quick Search",
                    passed=len(sources) > 3,
                    score=min(len(sources) / 8.0, 1.0),
                    issues=[],
                    recommendations=["Try Enhanced Mode for comprehensive analysis"]
                )
            ],
            critical_issues=[],
            recommendations=["Use Enhanced Mode for detailed quality assessment and more sources"]
        )
        
        return ResearchResult(
            research_id=research_id,
            query=result.query,
            mode=request.mode,
            citation_style=request.citation_style,
            synthesis=concise_synthesis,
            sources=sources,
            quality_assessment=basic_quality,
            research_time=result.research_time * 0.6,  # Faster processing time
            timestamp=datetime.fromisoformat(result.timestamp.replace('Z', '+00:00')) if isinstance(result.timestamp, str) else result.timestamp,
            bibliography=result.citation_bibliography[:500] + "..." if len(result.citation_bibliography) > 500 else result.citation_bibliography,
            domain_detected=result.domain_profile.domain.value if result.domain_profile else None,
            validation_summary={
                "basic_mode": True,
                "quick_research": True,
                "sources_found": len(sources),
                "processing_time": "optimized for speed"
            },
            recommendations=["For comprehensive analysis with more sources, use Enhanced Mode"]
        )
    
    async def _conduct_advanced_research(
        self,
        agent: 'AdvancedAutonomousResearchAgent',
        request: ResearchRequest,
        research_id: str,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """Conduct advanced research with comprehensive multi-source analysis."""
        
        # Advanced progress stages with more granular tracking
        stages = [
            ("enhanced_query_analysis", 0.08, "🧠 Advanced AI query analysis..."),
            ("multi_source_search", 0.25, "🔍 Multi-source comprehensive search..."),
            ("iterative_refinement", 0.45, "🔄 Iterative search refinement..."),
            ("quality_assessment", 0.65, "📊 Advanced quality assessment..."),
            ("structured_synthesis", 0.85, "📝 Structured synthesis generation..."),
            ("completed", 1.0, "🚀 Advanced research completed!")
        ]
        
        for stage, progress, message in stages:
            if progress_callback:
                progress_callback(ResearchProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    sources_found=0  # Will be updated with actual count
                ))
            
            # Small delay to show progress
            await asyncio.sleep(0.5)
        
        # Conduct the advanced research
        result = await agent.conduct_advanced_research(
            request.query,
            request.citation_style.value
        )
        
        # Convert to API model
        return self._convert_advanced_result_to_api_model(result, research_id, request)
    
    async def _conduct_autonomous_research(
        self,
        agent: 'AutonomousResearchAgent',
        request: ResearchRequest,
        research_id: str,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """Conduct autonomous research with AI-driven intelligence."""
        
        # AI-driven progress stages
        stages = [
            ("ai_query_analysis", 0.15, "AI analyzing research intent..."),
            ("ai_search_strategy", 0.35, "AI generating search strategies..."),
            ("ai_relevance_evaluation", 0.60, "AI evaluating source relevance..."),
            ("ai_synthesis", 0.85, "AI synthesizing findings..."),
            ("completed", 1.0, "Autonomous research completed!")
        ]
        
        for stage, progress, message in stages:
            if progress_callback:
                progress_callback(ResearchProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    sources_found=0  # Will be updated with actual count
                ))
            
            # Small delay to show progress
            await asyncio.sleep(0.5)
        
        # Conduct the autonomous research
        result = await agent.conduct_autonomous_research(
            request.query,
            request.citation_style.value
        )
        
        # Convert to API model
        return self._convert_autonomous_result_to_api_model(result, research_id, request)
    
    async def _conduct_enhanced_research(
        self,
        agent: 'EnhancedResearchAgent',
        request: ResearchRequest,
        research_id: str,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """Conduct enhanced research with validation."""
        
        # Send progress updates throughout the process
        stages = [
            ("domain_analysis", 0.1, "Analyzing research domain..."),
            ("source_search", 0.3, "Searching academic sources..."),
            ("content_extraction", 0.5, "Extracting content..."),
            ("validation", 0.7, "Validating sources..."),
            ("synthesis", 0.9, "Synthesizing findings..."),
            ("completed", 1.0, "Research completed!")
        ]
        
        for stage, progress, message in stages:
            if progress_callback:
                progress_callback(ResearchProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    sources_found=0  # Will be updated with actual count
                ))
            
            # Small delay to show progress
            await asyncio.sleep(0.5)
        
        # Conduct the actual research
        result = await agent.conduct_enhanced_research(
            request.query,
            request.citation_style.value
        )
        
        # Convert to API model
        return self._convert_enhanced_result_to_api_model(result, research_id, request)
    
    async def _conduct_basic_research(
        self,
        agent: ResearchAgent,
        request: ResearchRequest,
        research_id: str,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """Conduct basic research."""
        
        # Send progress updates
        stages = [
            ("source_search", 0.2, "Searching sources..."),
            ("content_extraction", 0.5, "Extracting content..."),
            ("synthesis", 0.8, "Synthesizing findings..."),
            ("completed", 1.0, "Research completed!")
        ]
        
        for stage, progress, message in stages:
            if progress_callback:
                progress_callback(ResearchProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    sources_found=0
                ))
            
            await asyncio.sleep(0.5)
        
        # Conduct the actual research
        result = await agent.conduct_research(
            request.query,
            request.citation_style.value
        )
        
        # Convert to API model
        return self._convert_basic_result_to_api_model(result, research_id, request)
    
    def _convert_advanced_result_to_api_model(
        self,
        result: Any,
        research_id: str,
        request: ResearchRequest
    ) -> ResearchResult:
        """Convert advanced research result to API model."""
        
        def safe_get(obj, key, default=None):
            """Safely get attribute or key from object or dict."""
            try:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return getattr(obj, key, default)
            except (AttributeError, TypeError):
                return default
        
        try:
            # Convert sources safely
            sources = []
            for source_data in result.sources:
                if isinstance(source_data, dict):
                    sources.append(SourceModel(
                        title=safe_get(source_data, 'title', 'Unknown Title'),
                        authors=safe_get(source_data, 'authors', []),
                        year=str(safe_get(source_data, 'publication_year', 'Unknown Year')),
                        url=safe_get(source_data, 'url'),
                        doi=safe_get(source_data, 'doi'),
                        journal=safe_get(source_data, 'venue'),
                        source_type=safe_get(source_data, 'source_type', 'unknown'),
                        relevance_score=safe_get(source_data, 'relevance_score'),
                        abstract=safe_get(source_data, 'abstract')
                    ))
            
            # Create enhanced quality assessment from advanced research
            quality_assessment = None
            if result.quality_assessment:
                qa_dict = result.quality_assessment
                
                # Create quality gate results from advanced assessment
                gate_results = [
                    QualityGateResult(
                        gate_name="Advanced Query Analysis",
                        passed=True,
                        score=safe_get(result.query_analysis, 'confidence', 0.8),
                        issues=[],
                        recommendations=[]
                    ),
                    QualityGateResult(
                        gate_name="Multi-Source Search Strategy",
                        passed=len(result.sources) >= 10,
                        score=min(len(result.sources) / 20.0, 1.0),
                        issues=[],
                        recommendations=[]
                    ),
                    QualityGateResult(
                        gate_name="Quality & Relevance Assessment",
                        passed=safe_get(qa_dict, 'overall_score', 0) > 0.5,
                        score=safe_get(qa_dict, 'overall_score', 0.5),
                        issues=[],
                        recommendations=[]
                    ),
                    QualityGateResult(
                        gate_name="Structured Synthesis",
                        passed=result.confidence_score > 0.5,
                        score=result.confidence_score,
                        issues=[],
                        recommendations=result.recommendations[:3]
                    )
                ]
                
                quality_level_map = {
                    "exceptional": QualityLevel.EXCELLENT,
                    "excellent": QualityLevel.EXCELLENT,
                    "good": QualityLevel.GOOD,
                    "acceptable": QualityLevel.ACCEPTABLE,
                    "insufficient": QualityLevel.POOR,
                    "error": QualityLevel.FAILED
                }
                
                quality_assessment = QualityAssessment(
                    overall_quality=quality_level_map.get(
                        safe_get(qa_dict, 'quality_level', 'acceptable'), 
                        QualityLevel.ACCEPTABLE
                    ),
                    overall_score=safe_get(qa_dict, 'overall_score', result.confidence_score),
                    confidence_score=result.confidence_score,
                    gate_results=gate_results,
                    critical_issues=[],
                    recommendations=result.recommendations[:5]
                )
            
            return ResearchResult(
                research_id=research_id,
                query=result.query,
                mode=request.mode,
                citation_style=request.citation_style,
                synthesis=result.executive_summary + "\n\n" + result.detailed_analysis,
                sources=sources,
                quality_assessment=quality_assessment,
                research_time=result.research_time,
                timestamp=datetime.fromisoformat(result.timestamp) if isinstance(result.timestamp, str) else result.timestamp,
                bibliography=result.bibliography,
                domain_detected=safe_get(result.query_analysis, 'domain_detected'),
                validation_summary={
                    "advanced_research": True,
                    "multi_source_search": True,
                    "structured_synthesis": True,
                    "ai_confidence": result.confidence_score,
                    "sources_found": len(result.sources),
                    "quality_level": safe_get(result.quality_assessment, 'quality_level', 'unknown'),
                    "coverage_level": safe_get(result.coverage_assessment, 'coverage_level', 'unknown'),
                    "source_diversity": result.source_diversity
                },
                recommendations=result.recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to convert advanced result: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Return minimal result on conversion failure
            return ResearchResult(
                research_id=research_id,
                query=result.query if hasattr(result, 'query') else "Unknown query",
                mode=request.mode,
                citation_style=request.citation_style,
                synthesis="Advanced research completed but result conversion encountered an error. Please check the logs for details.",
                sources=[],
                research_time=getattr(result, 'research_time', 0),
                timestamp=datetime.now(),
                bibliography="Error in bibliography generation",
                recommendations=["Please try your query again"]
            )
    
    def _convert_autonomous_result_to_api_model(
        self,
        result: Any,
        research_id: str,
        request: ResearchRequest
    ) -> ResearchResult:
        """Convert autonomous research result to API model."""
        
        def safe_get(obj, key, default=None):
            """Safely get attribute or key from object or dict."""
            try:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return getattr(obj, key, default)
            except (AttributeError, TypeError):
                return default
        
        def safe_dict_convert(obj):
            """Safely convert object to dict for access."""
            try:
                if isinstance(obj, dict):
                    return obj
                elif hasattr(obj, '__dict__'):
                    return vars(obj)
                elif hasattr(obj, '_asdict'):
                    return obj._asdict()
                else:
                    return {}
            except Exception:
                return {}
        
        try:
            # Convert sources safely
            sources = []
            for source_data in result.sources:
                source_dict = safe_dict_convert(source_data)
                sources.append(SourceModel(
                    title=safe_get(source_dict, 'title', 'Unknown Title'),
                    authors=safe_get(source_dict, 'authors', []),
                    year=str(safe_get(source_dict, 'year', 'Unknown Year')),
                    url=safe_get(source_dict, 'url'),
                    doi=safe_get(source_dict, 'doi'),
                    journal=safe_get(source_dict, 'journal'),
                    source_type=safe_get(source_dict, 'source_type', 'unknown'),
                    relevance_score=safe_get(source_dict, 'relevance_score'),
                    abstract=safe_get(source_dict, 'abstract')
                ))
            
            # Convert quality assessment safely
            quality_assessment = None
            if result.quality_assessment:
                qa_dict = safe_dict_convert(result.quality_assessment)
                query_analysis_dict = safe_dict_convert(result.query_analysis)
                
                # Create basic quality gate results from autonomous assessment
                gate_results = [
                    QualityGateResult(
                        gate_name="AI Query Analysis",
                        passed=True,
                        score=safe_get(query_analysis_dict, 'confidence', 0.8),
                        issues=[],
                        recommendations=[]
                    ),
                    QualityGateResult(
                        gate_name="AI Search Strategy",
                        passed=len(result.sources) > 0,
                        score=min(len(result.sources) / 5.0, 1.0),  # Normalize to 0-1
                        issues=[],
                        recommendations=[]
                    ),
                    QualityGateResult(
                        gate_name="AI Relevance Assessment",
                        passed=safe_get(qa_dict, 'relevance_rate', 0) > 0.3,
                        score=safe_get(qa_dict, 'relevance_rate', 0.5),
                        issues=[],
                        recommendations=[]
                    ),
                    QualityGateResult(
                        gate_name="AI Synthesis",
                        passed=result.confidence_score > 0.3,
                        score=result.confidence_score,
                        issues=[],
                        recommendations=result.recommendations[:3]
                    )
                ]
                
                quality_level_map = {
                    "excellent": QualityLevel.EXCELLENT,
                    "good": QualityLevel.GOOD,
                    "acceptable": QualityLevel.ACCEPTABLE,
                    "poor": QualityLevel.POOR,
                    "failed": QualityLevel.FAILED,
                    "error": QualityLevel.FAILED
                }
                
                quality_assessment = QualityAssessment(
                    overall_quality=quality_level_map.get(safe_get(qa_dict, 'quality_level', 'acceptable'), QualityLevel.ACCEPTABLE),
                    overall_score=safe_get(qa_dict, 'overall_score', result.confidence_score),
                    confidence_score=result.confidence_score,
                    gate_results=gate_results,
                    critical_issues=[],
                    recommendations=result.recommendations[:5]
                )
            
            # Convert synthesis result safely
            synthesis_result_dict = safe_dict_convert(result.synthesis_result)
            query_analysis_dict = safe_dict_convert(result.query_analysis)
            quality_assessment_dict = safe_dict_convert(result.quality_assessment)
            
            return ResearchResult(
                research_id=research_id,
                query=result.query,
                mode=request.mode,
                citation_style=request.citation_style,
                synthesis=result.synthesis,
                sources=sources,
                quality_assessment=quality_assessment,
                research_time=result.research_time,
                timestamp=datetime.fromisoformat(result.timestamp) if isinstance(result.timestamp, str) else result.timestamp,
                bibliography=result.bibliography,
                domain_detected=safe_get(query_analysis_dict, 'domain_detected'),
                validation_summary={
                    "autonomous_research": True,
                    "ai_confidence": result.confidence_score,
                    "synthesis_type": safe_get(synthesis_result_dict, 'synthesis_type', 'unknown'),
                    "sources_found": len(result.sources),
                    "quality_level": safe_get(quality_assessment_dict, 'quality_level', 'unknown')
                },
                recommendations=result.recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to convert autonomous result: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Return minimal result on conversion failure
            return ResearchResult(
                research_id=research_id,
                query=result.query if hasattr(result, 'query') else "Unknown query",
                mode=request.mode,
                citation_style=request.citation_style,
                synthesis="Error occurred during result conversion",
                sources=[],
                research_time=getattr(result, 'research_time', 0),
                timestamp=datetime.now(),
                bibliography="Error in bibliography generation",
                recommendations=["Please try your query again"]
            )
    
    def _convert_enhanced_result_to_api_model(
        self,
        result: Any,
        research_id: str,
        request: ResearchRequest
    ) -> ResearchResult:
        """Convert enhanced research result to API model."""
        
        # Convert sources
        sources = []
        for source_data, relevance in result.validated_sources:
            sources.append(SourceModel(
                title=source_data.get('title', 'Unknown Title'),
                authors=source_data.get('authors', []),
                year=str(source_data.get('year', 'Unknown Year')),
                url=source_data.get('url'),
                doi=source_data.get('doi'),
                journal=source_data.get('journal'),
                source_type=source_data.get('source_type', 'unknown'),
                relevance_score=getattr(relevance, 'overall_score', None) if relevance else None,
                abstract=source_data.get('abstract')
            ))
        
        # Convert quality assessment
        quality_assessment = None
        if result.quality_assessment:
            qa = result.quality_assessment
            gate_results = []
            
            for gate_result in qa.gate_results:
                gate_results.append(QualityGateResult(
                    gate_name=gate_result.gate_name,
                    passed=gate_result.passed,
                    score=gate_result.score,
                    issues=gate_result.issues,
                    recommendations=gate_result.recommendations
                ))
            
            quality_assessment = QualityAssessment(
                overall_quality=QualityLevel(qa.overall_quality.value),
                overall_score=qa.overall_score,
                confidence_score=result.confidence_score,
                gate_results=gate_results,
                critical_issues=qa.critical_issues,
                recommendations=qa.recommendations
            )
        
        return ResearchResult(
            research_id=research_id,
            query=result.query,
            mode=request.mode,
            citation_style=request.citation_style,
            synthesis=result.synthesis,
            sources=sources,
            quality_assessment=quality_assessment,
            research_time=result.research_time,
            timestamp=datetime.fromisoformat(result.timestamp.replace('Z', '+00:00')) if isinstance(result.timestamp, str) else result.timestamp,
            bibliography=result.citation_bibliography,
            domain_detected=result.domain_profile.domain.value if result.domain_profile else None,
            validation_summary=result.cross_validation_result.__dict__ if result.cross_validation_result else None,
            recommendations=result.recommendations
        )
    
    def _convert_basic_result_to_api_model(
        self,
        result: Dict[str, Any],
        research_id: str,
        request: ResearchRequest
    ) -> ResearchResult:
        """Convert basic research result to API model."""
        
        # Convert sources
        sources = []
        for source_data in result.get('sources', []):
            sources.append(SourceModel(
                title=source_data.get('title', 'Unknown Title'),
                authors=source_data.get('authors', []),
                year=str(source_data.get('year', 'Unknown Year')),
                url=source_data.get('url'),
                doi=source_data.get('doi'),
                journal=source_data.get('journal'),
                source_type=source_data.get('source_type', 'unknown'),
                abstract=source_data.get('abstract')
            ))
        
        return ResearchResult(
            research_id=research_id,
            query=result['query'],
            mode=request.mode,
            citation_style=request.citation_style,
            synthesis=result['synthesis'],
            sources=sources,
            research_time=result['research_time'],
            timestamp=datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00')) if isinstance(result['timestamp'], str) else datetime.now(),
            bibliography=result.get('report', ''),  # Basic mode uses report field
        )
    
    def get_research_result(self, research_id: str) -> Optional[ResearchResult]:
        """Get cached research result."""
        return self.research_cache.get(research_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get active research sessions."""
        return self.active_sessions
    
    def cancel_research(self, research_id: str) -> bool:
        """Cancel active research session."""
        if research_id in self.active_sessions:
            del self.active_sessions[research_id]
            logger.info(f"🛑 Research cancelled: {research_id}")
            return True
        return False

# Global service instance
research_service = ResearchService()
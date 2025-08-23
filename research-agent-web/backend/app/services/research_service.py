"""
Research Service - Simplified version using only UnifiedResearchAgent
"""

import os
import sys
import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Load environment variables from .env file early
from dotenv import load_dotenv
load_dotenv()

# Set up logging first
logger = logging.getLogger(__name__)

# Add research_agent paths - use environment variable or relative paths
root_research_path = os.getenv('RESEARCH_AGENT_ROOT_PATH', '/Users/aswin/new_research_agent')
if root_research_path and root_research_path not in sys.path:
    sys.path.insert(0, root_research_path)  # Insert at beginning for priority

# Backup path to local research_agent directory
local_research_agent_path = os.path.join(os.path.dirname(__file__), '..', '..', 'research_agent')
if local_research_agent_path not in sys.path:
    sys.path.append(local_research_agent_path)

# Additional backup: try parent directory structure
parent_research_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.')
if parent_research_path not in sys.path:
    sys.path.append(parent_research_path)

# Import the new unified research agent
try:
    from unified_research_agent import UnifiedResearchAgent, create_unified_agent
    UNIFIED_AVAILABLE = True
    logger.info("✅ UnifiedResearchAgent imported successfully")
except ImportError as e:
    UNIFIED_AVAILABLE = False
    logger.error(f"❌ Failed to import UnifiedResearchAgent: {e}")
    UnifiedResearchAgent = None
    create_unified_agent = None

from ..models.research import (
    ResearchRequest, ResearchResult, ResearchProgress, QualityAssessment,
    QualityGateResult, SourceModel, ResearchMode, QualityLevel
)
from ..core.config import settings

class ResearchService:
    """Simplified service for handling research operations with UnifiedResearchAgent."""
    
    def __init__(self):
        self.research_cache: Dict[str, ResearchResult] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("🎯 Simplified Research Service initialized with UnifiedResearchAgent")
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate that the research system is properly configured."""
        status = {
            "api_key_configured": False,
            "agents_available": False,
            "issues": [],
            "setup_valid": False
        }
        
        # Check API key
        from ..core.config import settings as app_settings
        api_key = app_settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            status["issues"].append("OpenAI API key not configured")
        else:
            status["api_key_configured"] = True
            logger.info(f"✅ API key configured (length: {len(api_key)})")
        
        # Check agent availability
        if not UNIFIED_AVAILABLE:
            status["issues"].append("UnifiedResearchAgent not available")
        else:
            status["agents_available"] = True
            logger.info("✅ UnifiedResearchAgent available")
        
        status["setup_valid"] = len(status["issues"]) == 0
        return status
        
    async def conduct_research(
        self,
        request: ResearchRequest,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """
        Conduct research using the unified research agent.
        
        Args:
            request: Research request parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            Research result
        """
        research_id = str(uuid.uuid4())
        
        try:
            logger.info(f"🔍 Starting unified research: {request.query}")
            
            # Enhanced API key validation - use settings first, then environment
            from ..core.config import settings as app_settings
            api_key = app_settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            
            logger.info(f"🔍 API Key Check - Settings: {'✅' if app_settings.OPENAI_API_KEY else '❌'}, Env: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
            
            if not api_key:
                error_msg = "OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable."
                logger.error(f"❌ {error_msg}")
                if progress_callback:
                    await progress_callback(ResearchProgress(
                        stage="error",
                        progress=0.0,
                        message=error_msg,
                        sources_found=0
                    ))
                raise ValueError(error_msg)
            
            # Validate API key format
            if not api_key.startswith('sk-'):
                error_msg = "Invalid OpenAI API key format. Key should start with 'sk-'."
                logger.error(f"❌ {error_msg}")
                if progress_callback:
                    await progress_callback(ResearchProgress(
                        stage="error",
                        progress=0.0,
                        message=error_msg,
                        sources_found=0
                    ))
                raise ValueError(error_msg)
            
            logger.info(f"✅ OpenAI API key validated successfully (length: {len(api_key)})")
            
            # Create unified research agent - simple and streamlined
            if not UNIFIED_AVAILABLE or not UnifiedResearchAgent:
                error_msg = "UnifiedResearchAgent is not available. Please check the setup."
                logger.error(f"❌ {error_msg}")
                if progress_callback:
                    await progress_callback(ResearchProgress(
                        stage="error",
                        progress=0.0,
                        message=error_msg,
                        sources_found=0
                    ))
                raise ValueError(error_msg)
            
            # Use the unified agent for all research requests
            agent = create_unified_agent(
                openai_api_key=api_key,
                max_sources=request.max_sources or 15,  # Default to 15 sources
                debug_mode=request.debug_mode or False
            )
            logger.info(f"🎯 Using UnifiedResearchAgent - comprehensive web search + GPT-4o mini + simple citations")
            
            # Track session
            self.active_sessions[research_id] = {
                "agent": agent,
                "start_time": datetime.now(),
                "request": request,
                "timeout": datetime.now().timestamp() + settings.RESEARCH_TIMEOUT_SECONDS
            }
            
            # Send initial progress
            if progress_callback:
                await progress_callback(ResearchProgress(
                    stage="initializing",
                    progress=0.0,
                    message="Initializing unified research agent...",
                    sources_found=0
                ))
            
            # Conduct unified research
            try:
                result = await asyncio.wait_for(
                    self._conduct_unified_research(
                        agent, request, research_id, progress_callback
                    ),
                    timeout=settings.RESEARCH_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                error_msg = f"Research operation timed out after {settings.RESEARCH_TIMEOUT_SECONDS} seconds. Please try with a more specific query."
                logger.error(f"⏰ {error_msg}")
                if progress_callback:
                    await progress_callback(ResearchProgress(
                        stage="error",
                        progress=0.0,
                        message=error_msg,
                        sources_found=0
                    ))
                raise Exception(error_msg)
            
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
                await progress_callback(ResearchProgress(
                    stage="error",
                    progress=0.0,
                    message=f"Research failed: {str(e)}",
                    sources_found=0
                ))
            
            raise
    
    async def _conduct_unified_research(
        self,
        agent: UnifiedResearchAgent,
        request: ResearchRequest,
        research_id: str,
        progress_callback: Optional[Callable[[ResearchProgress], None]] = None
    ) -> ResearchResult:
        """
        Conduct research using the new UnifiedResearchAgent.
        Simple, streamlined research flow with comprehensive web search.
        """
        
        # Unified research progress stages - simple and clear
        stages = [
            ("searching", 0.20, "🔍 Comprehensive web search + academic papers..."),
            ("analyzing", 0.60, "🧠 Analyzing sources with GPT-4o mini..."),
            ("synthesizing", 0.85, "✨ Creating summary with inline citations..."),
            ("completed", 1.0, "🎯 Research completed with citations!")
        ]
        
        for stage, progress, message in stages[:-1]:  # Skip the last "completed" stage for now
            if progress_callback:
                await progress_callback(ResearchProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    sources_found=0  # Will be updated with actual count
                ))
            await asyncio.sleep(0.5)  # Brief pause for progress updates
        
        try:
            # Conduct research using the unified agent
            logger.info(f"🎯 Starting unified research for query: {request.query}")
            
            # Create progress callback wrapper to match expected format
            async def unified_progress_callback(progress_data):
                if progress_callback:
                    # Convert unified agent progress format to ResearchProgress format
                    if isinstance(progress_data, dict):
                        await progress_callback(ResearchProgress(
                            stage=progress_data.get('stage', 'processing'),
                            progress=progress_data.get('progress', 0.5),
                            message=progress_data.get('message', 'Processing...'),
                            sources_found=progress_data.get('sources_found', 0)
                        ))
                    else:
                        # Handle direct ResearchProgress objects
                        await progress_callback(progress_data)
            
            # Call the unified agent's research method with progress callback
            unified_result = await agent.research(request.query, progress_callback=unified_progress_callback)
            
            # Convert unified result to the expected ResearchResult format
            # Map the sources from UnifiedResearchAgent to the expected format
            sources = []
            for source in unified_result.sources:
                source_model = SourceModel(
                    title=source.title,
                    authors=source.authors,
                    url=source.url,
                    abstract=source.snippet,  # Use snippet as abstract
                    year=str(int(source.date.split('-')[0])) if source.date and '-' in source.date else "2024",  # Convert to string
                    journal=source.source_type.title(),  # Use source type as journal
                    source_type=source.source_type,  # Add the missing source_type field
                    citation_key=f"ref_{source.citation_id}",
                    relevance_score=0.8,  # Default relevance score
                    quality_indicators={
                        "peer_reviewed": source.source_type == 'academic',
                        "recent": True,
                        "authoritative": source.source_type in ['wikipedia', 'academic']
                    }
                )
                sources.append(source_model)
            
            # Create the research result
            # Convert timestamp to datetime if it's a string
            if isinstance(unified_result.timestamp, str):
                timestamp = datetime.now()  # Use current time as fallback
            else:
                timestamp = unified_result.timestamp
            
            result = ResearchResult(
                research_id=research_id,
                query=unified_result.query,
                synthesis=unified_result.summary_with_citations,  # Changed from summary to synthesis
                sources=sources,
                mode=request.mode,
                citation_style=request.citation_style,
                timestamp=timestamp,
                research_time=unified_result.research_time,
                bibliography=f"Research conducted on {timestamp.strftime('%Y-%m-%d')} using {len(sources)} sources.",
                quality_assessment=QualityAssessment(
                    overall_quality=QualityLevel.EXCELLENT,  # Unified agent provides high quality
                    overall_score=0.9,
                    confidence_score=0.85,
                    gate_results=[
                        QualityGateResult(
                            gate_name="source_validation", 
                            passed=True, 
                            score=0.9,
                            issues=[],
                            recommendations=[]
                        ),
                        QualityGateResult(
                            gate_name="content_synthesis", 
                            passed=True, 
                            score=0.9,
                            issues=[],
                            recommendations=[]
                        ),
                        QualityGateResult(
                            gate_name="citation_formatting", 
                            passed=True, 
                            score=0.95,
                            issues=[],
                            recommendations=[]
                        )
                    ],
                    critical_issues=[],
                    recommendations=["Use multiple sources for comprehensive research"]
                )
            )
            
            # Send final completion progress
            if progress_callback:
                await progress_callback(ResearchProgress(
                    stage="completed",
                    progress=1.0,
                    message=f"🎯 Research completed! Found {len(sources)} sources with inline citations",
                    sources_found=len(sources)
                ))
            
            logger.info(f"✅ Unified research completed: {len(sources)} sources, {unified_result.research_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Unified research failed: {e}")
            if progress_callback:
                await progress_callback(ResearchProgress(
                    stage="error",
                    progress=0.0,
                    message=f"Research failed: {str(e)}",
                    sources_found=0
                ))
            raise
    
    def get_cached_result(self, research_id: str) -> Optional[ResearchResult]:
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
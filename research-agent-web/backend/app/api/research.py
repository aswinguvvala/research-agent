"""
Research API endpoints
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
import tempfile
import json
import os

from ..models.research import (
    ResearchRequest, ResearchResult, ResearchProgress, ExportRequest,
    ResearchHistoryItem, ResearchHistory, ErrorResponse, ExportFormat
)
from ..services.research_service import research_service
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for research history (in production, use proper database)
research_history: List[ResearchHistoryItem] = []

@router.post("/conduct", response_model=ResearchResult)
async def conduct_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
) -> ResearchResult:
    """
    Conduct research on a query.
    
    This endpoint initiates a research session and returns the complete results.
    For real-time progress updates, use the WebSocket endpoint.
    """
    try:
        logger.info(f"🔍 Research request: {request.query} (mode: {request.mode.value})")
        
        # Validate request
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 3 characters long"
            )
        
        # Conduct research
        result = await research_service.conduct_research(request)
        
        # Add to history
        history_item = ResearchHistoryItem(
            research_id=result.research_id,
            query=result.query,
            mode=result.mode,
            quality_level=result.quality_assessment.overall_quality if result.quality_assessment else None,
            sources_count=len(result.sources),
            research_time=result.research_time,
            timestamp=result.timestamp
        )
        research_history.append(history_item)
        
        # Keep only last 50 items
        if len(research_history) > 50:
            research_history.pop(0)
        
        logger.info(f"✅ Research completed: {result.research_id}")
        return result
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Research error: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

@router.get("/result/{research_id}", response_model=ResearchResult)
async def get_research_result(research_id: str) -> ResearchResult:
    """Get research result by ID."""
    
    result = research_service.get_research_result(research_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Research result not found: {research_id}"
        )
    
    return result

@router.get("/history", response_model=ResearchHistory)
async def get_research_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page")
) -> ResearchHistory:
    """Get research history with pagination."""
    
    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    # Get items in reverse chronological order
    items = list(reversed(research_history[start_idx:end_idx]))
    
    return ResearchHistory(
        items=items,
        total_count=len(research_history),
        page=page,
        page_size=page_size
    )

@router.post("/export", response_class=FileResponse)
async def export_research(request: ExportRequest):
    """Export research results to file."""
    
    # Get research result
    result = research_service.get_research_result(request.research_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Research result not found: {request.research_id}"
        )
    
    try:
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        
        # Generate filename
        safe_query = "".join(c for c in result.query[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_query = safe_query.replace(' ', '_')
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        
        if request.format == ExportFormat.JSON:
            filename = f"{safe_query}_{timestamp}.json"
            filepath = os.path.join(temp_dir, filename)
            
            # Convert to dict and save
            result_dict = result.dict()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            return FileResponse(
                filepath,
                media_type="application/json",
                filename=filename
            )
        
        elif request.format == ExportFormat.MD:
            filename = f"{safe_query}_{timestamp}.md"
            filepath = os.path.join(temp_dir, filename)
            
            # Generate markdown
            markdown_content = _generate_markdown_report(result, request.include_metadata)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return FileResponse(
                filepath,
                media_type="text/markdown",
                filename=filename
            )
        
        elif request.format == ExportFormat.TXT:
            filename = f"{safe_query}_{timestamp}.txt"
            filepath = os.path.join(temp_dir, filename)
            
            # Generate text report
            text_content = _generate_text_report(result, request.include_metadata)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            return FileResponse(
                filepath,
                media_type="text/plain",
                filename=filename
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Export format not supported: {request.format}"
            )
    
    except Exception as e:
        logger.error(f"❌ Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/active-sessions")
async def get_active_sessions() -> Dict[str, Any]:
    """Get active research sessions."""
    
    sessions = research_service.get_active_sessions()
    
    # Convert to serializable format
    serializable_sessions = {}
    for session_id, session_data in sessions.items():
        serializable_sessions[session_id] = {
            "start_time": session_data["start_time"].isoformat(),
            "query": session_data["request"].query,
            "mode": session_data["request"].mode.value
        }
    
    return {
        "active_sessions": serializable_sessions,
        "total_active": len(sessions)
    }

@router.delete("/session/{research_id}")
async def cancel_research(research_id: str) -> Dict[str, str]:
    """Cancel active research session."""
    
    success = research_service.cancel_research(research_id)
    
    if success:
        return {"message": f"Research session cancelled: {research_id}"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Active research session not found: {research_id}"
        )

@router.get("/settings")
async def get_research_settings() -> Dict[str, Any]:
    """Get current research settings."""
    return {
        "max_sources": settings.MAX_SOURCES,
        "relevance_threshold": settings.RELEVANCE_THRESHOLD,
        "content_validation_threshold": settings.CONTENT_VALIDATION_THRESHOLD,
        "consensus_threshold": settings.CONSENSUS_THRESHOLD,
        "enhanced_mode_available": True,  # Will be determined by imports
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@router.post("/recover-session")
async def recover_research_session(request: ResearchRequest) -> Dict[str, Any]:
    """
    Attempt to recover a research session based on the request.
    This checks if there's a cached result for a similar query.
    """
    try:
        # Check if there's a recent result for this exact query
        for research_id, cached_result in research_service.research_cache.items():
            if (cached_result.query.lower().strip() == request.query.lower().strip() and
                cached_result.mode == request.mode):
                
                # Check if the result is recent (within last 24 hours)
                time_diff = datetime.now() - cached_result.timestamp
                if time_diff.total_seconds() < 24 * 60 * 60:  # 24 hours
                    logger.info(f"🔄 Recovered cached research result: {research_id}")
                    return {
                        "recovered": True,
                        "research_id": research_id,
                        "result": cached_result,
                        "message": "Found recent research results for this query"
                    }
        
        # No recovery possible
        return {
            "recovered": False,
            "message": "No recent research results found for this query"
        }
        
    except Exception as e:
        logger.error(f"❌ Session recovery error: {e}")
        return {
            "recovered": False,
            "message": f"Session recovery failed: {str(e)}"
        }

# Helper functions

def _generate_markdown_report(result: ResearchResult, include_metadata: bool = True) -> str:
    """Generate markdown report from research result."""
    
    lines = [
        f"# Research Report: {result.query}",
        "",
        f"**Generated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Mode:** {result.mode.value.title()}",
        f"**Sources Found:** {len(result.sources)}",
        f"**Research Time:** {result.research_time:.1f} seconds",
    ]
    
    if result.quality_assessment:
        lines.extend([
            f"**Quality:** {result.quality_assessment.overall_quality.value.title()}",
            f"**Confidence:** {result.quality_assessment.confidence_score:.1%}"
        ])
    
    lines.extend([
        "",
        "## Research Synthesis",
        "",
        result.synthesis,
        "",
        "## Sources",
        ""
    ])
    
    # Add sources
    for i, source in enumerate(result.sources, 1):
        author_str = ", ".join(source.authors) if source.authors else "Unknown Author"
        lines.append(f"{i}. **{source.title}**")
        lines.append(f"   *{author_str}* ({source.year})")
        
        if source.url:
            lines.append(f"   [Link]({source.url})")
        
        if source.relevance_score is not None:
            lines.append(f"   Relevance: {source.relevance_score:.1%}")
        
        lines.append("")
    
    if include_metadata and result.quality_assessment:
        lines.extend([
            "## Quality Assessment",
            "",
            f"**Overall Quality:** {result.quality_assessment.overall_quality.value.title()}",
            f"**Quality Score:** {result.quality_assessment.overall_score:.1%}",
            f"**Confidence Score:** {result.quality_assessment.confidence_score:.1%}",
            ""
        ])
        
        if result.quality_assessment.recommendations:
            lines.extend([
                "### Recommendations",
                ""
            ])
            for rec in result.quality_assessment.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
    
    lines.extend([
        "---",
        "*Generated by Research Agent Web Application*"
    ])
    
    return "\n".join(lines)

def _generate_text_report(result: ResearchResult, include_metadata: bool = True) -> str:
    """Generate text report from research result."""
    
    lines = [
        f"RESEARCH REPORT: {result.query}",
        "=" * 60,
        f"Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {result.mode.value.title()}",
        f"Sources Found: {len(result.sources)}",
        f"Research Time: {result.research_time:.1f} seconds",
    ]
    
    if result.quality_assessment:
        lines.extend([
            f"Quality: {result.quality_assessment.overall_quality.value.title()}",
            f"Confidence: {result.quality_assessment.confidence_score:.1%}"
        ])
    
    lines.extend([
        "",
        "RESEARCH SYNTHESIS",
        "-" * 20,
        result.synthesis,
        "",
        "SOURCES",
        "-" * 20
    ])
    
    # Add sources
    for i, source in enumerate(result.sources, 1):
        author_str = ", ".join(source.authors) if source.authors else "Unknown Author"
        lines.append(f"{i}. {source.title}")
        lines.append(f"   {author_str} ({source.year})")
        
        if source.url:
            lines.append(f"   {source.url}")
        
        if source.relevance_score is not None:
            lines.append(f"   Relevance: {source.relevance_score:.1%}")
        
        lines.append("")
    
    if include_metadata and result.quality_assessment:
        lines.extend([
            "QUALITY ASSESSMENT",
            "-" * 20,
            f"Overall Quality: {result.quality_assessment.overall_quality.value.title()}",
            f"Quality Score: {result.quality_assessment.overall_score:.1%}",
            f"Confidence Score: {result.quality_assessment.confidence_score:.1%}",
            ""
        ])
        
        if result.quality_assessment.recommendations:
            lines.extend([
                "Recommendations:",
                ""
            ])
            for rec in result.quality_assessment.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
    
    lines.extend([
        "-" * 60,
        "Generated by Research Agent Web Application"
    ])
    
    return "\n".join(lines)
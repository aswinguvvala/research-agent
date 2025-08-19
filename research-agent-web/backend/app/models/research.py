"""
Pydantic models for research API requests and responses
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ResearchMode(str, Enum):
    """Research mode options."""
    BASIC = "basic"
    ENHANCED = "enhanced"

class CitationStyle(str, Enum):
    """Citation style options."""
    APA = "apa"
    MLA = "mla"
    IEEE = "ieee"

class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

class ExportFormat(str, Enum):
    """Export format options."""
    TXT = "txt"
    JSON = "json"
    MD = "md"
    PDF = "pdf"

# Request Models

class ResearchRequest(BaseModel):
    """Research request model."""
    query: str = Field(..., min_length=3, max_length=500, description="Research query")
    mode: ResearchMode = Field(default=ResearchMode.ENHANCED, description="Research mode")
    citation_style: CitationStyle = Field(default=CitationStyle.APA, description="Citation style")
    max_sources: Optional[int] = Field(default=10, ge=1, le=50, description="Maximum number of sources")
    debug_mode: Optional[bool] = Field(default=False, description="Enable debug mode")

class ExportRequest(BaseModel):
    """Export request model."""
    research_id: str = Field(..., description="Research session ID")
    format: ExportFormat = Field(..., description="Export format")
    include_metadata: Optional[bool] = Field(default=True, description="Include metadata")

# Response Models

class SourceModel(BaseModel):
    """Source information model."""
    title: str
    authors: List[str]
    year: str
    url: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    source_type: str
    relevance_score: Optional[float] = None
    abstract: Optional[str] = None

class QualityGateResult(BaseModel):
    """Quality gate result model."""
    gate_name: str
    passed: bool
    score: float
    issues: List[str]
    recommendations: List[str]

class QualityAssessment(BaseModel):
    """Quality assessment model."""
    overall_quality: QualityLevel
    overall_score: float
    confidence_score: float
    gate_results: List[QualityGateResult]
    critical_issues: List[str]
    recommendations: List[str]

class ResearchProgress(BaseModel):
    """Research progress model for real-time updates."""
    stage: str
    progress: float  # 0.0 to 1.0
    message: str
    sources_found: int
    estimated_completion: Optional[datetime] = None

class ResearchResult(BaseModel):
    """Complete research result model."""
    research_id: str
    query: str
    mode: ResearchMode
    citation_style: CitationStyle
    synthesis: str
    sources: List[SourceModel]
    quality_assessment: Optional[QualityAssessment] = None
    research_time: float
    timestamp: datetime
    bibliography: str
    
    # Enhanced mode specific fields
    domain_detected: Optional[str] = None
    validation_summary: Optional[Dict[str, Any]] = None
    cross_validation_result: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

class ResearchSession(BaseModel):
    """Research session information."""
    session_id: str
    created_at: datetime
    queries: List[str]
    total_sources_found: int
    avg_quality_score: Optional[float] = None

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: str
    version: str
    features: Dict[str, bool]
    timestamp: datetime

# WebSocket Models

class WSMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime

class WSResearchUpdate(BaseModel):
    """WebSocket research update model."""
    research_id: str
    progress: ResearchProgress

class WSError(BaseModel):
    """WebSocket error model."""
    error: str
    message: str
    timestamp: datetime

# Configuration Models

class ResearchSettings(BaseModel):
    """Research settings model."""
    relevance_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    content_validation_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_research_time: int = Field(default=300, ge=60, le=600)  # seconds
    enable_debug_logs: bool = Field(default=False)

# History Models

class ResearchHistoryItem(BaseModel):
    """Research history item model."""
    research_id: str
    query: str
    mode: ResearchMode
    quality_level: Optional[QualityLevel] = None
    sources_count: int
    research_time: float
    timestamp: datetime
    
class ResearchHistory(BaseModel):
    """Research history collection model."""
    items: List[ResearchHistoryItem]
    total_count: int
    page: int
    page_size: int
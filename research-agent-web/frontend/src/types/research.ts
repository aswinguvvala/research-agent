// TypeScript types for Research Agent Web Application
// These match the backend Pydantic models

export enum ResearchMode {
  BASIC = 'basic',
  ENHANCED = 'enhanced'
}

export enum CitationStyle {
  APA = 'apa',
  MLA = 'mla',
  IEEE = 'ieee'
}

export enum QualityLevel {
  EXCELLENT = 'excellent',
  GOOD = 'good',
  ACCEPTABLE = 'acceptable',
  POOR = 'poor',
  FAILED = 'failed'
}

export enum ExportFormat {
  TXT = 'txt',
  JSON = 'json',
  MD = 'md',
  PDF = 'pdf'
}

// Request types
export interface ResearchRequest {
  query: string;
  mode: ResearchMode;
  citation_style: CitationStyle;
  max_sources?: number;
  debug_mode?: boolean;
}

export interface ExportRequest {
  research_id: string;
  format: ExportFormat;
  include_metadata?: boolean;
}

// Response types
export interface SourceModel {
  title: string;
  authors: string[];
  year: string;
  url?: string;
  doi?: string;
  journal?: string;
  source_type: string;
  relevance_score?: number;
  abstract?: string;
  
  // Enhanced metadata for Google Gemini-style research
  credibility_score?: number;
  citation_count?: number;
  publication_date?: string;
  publisher?: string;
  volume?: string;
  issue?: string;
  pages?: string;
  isbn?: string;
  language?: string;
  country?: string;
  
  // Content analysis
  key_excerpts?: string[];
  main_topics?: string[];
  methodology?: string;
  sample_size?: number;
  study_type?: string;
  
  // Quality indicators
  peer_reviewed?: boolean;
  open_access?: boolean;
  quality_indicators?: Record<string, any>;
  
  // Citation formatting
  apa_citation?: string;
  mla_citation?: string;
  ieee_citation?: string;
}

export interface QualityGateResult {
  gate_name: string;
  passed: boolean;
  score: number;
  issues: string[];
  recommendations: string[];
}

export interface QualityAssessment {
  overall_quality: QualityLevel;
  overall_score: number;
  confidence_score: number;
  gate_results: QualityGateResult[];
  critical_issues: string[];
  recommendations: string[];
}

export interface ResearchProgress {
  stage: string;
  progress: number; // 0.0 to 1.0
  message: string;
  sources_found: number;
  estimated_completion?: string;
}

export interface ResearchResult {
  research_id: string;
  query: string;
  mode: ResearchMode;
  citation_style: CitationStyle;
  synthesis: string;
  sources: SourceModel[];
  quality_assessment?: QualityAssessment;
  research_time: number;
  timestamp: string;
  bibliography: string;
  
  // Enhanced synthesis features (Gemini Deep Research style)
  executive_summary?: string;
  detailed_sections?: Array<{[key: string]: string}>;
  key_findings?: string[];
  conflicting_information?: Array<Record<string, any>>;
  confidence_levels?: Record<string, number>;
  
  // Source organization
  primary_sources?: string[];
  supporting_sources?: string[];
  contradictory_sources?: string[];
  
  // Enhanced mode specific fields
  domain_detected?: string;
  validation_summary?: Record<string, any>;
  cross_validation_result?: Record<string, any>;
  recommendations?: string[];
  
  // Research insights
  research_gaps?: string[];
  future_research_directions?: string[];
  practical_implications?: string[];
}

export interface ResearchSession {
  session_id: string;
  created_at: string;
  queries: string[];
  total_sources_found: number;
  avg_quality_score?: number;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  features: Record<string, boolean>;
  timestamp: string;
}

// WebSocket types
export interface WSMessage {
  type: string;
  data: Record<string, any>;
  timestamp: string;
}

export interface WSResearchUpdate {
  research_id: string;
  progress: ResearchProgress;
}

export interface WSError {
  error: string;
  message: string;
  timestamp: string;
}

// History types
export interface ResearchHistoryItem {
  research_id: string;
  query: string;
  mode: ResearchMode;
  quality_level?: QualityLevel;
  sources_count: number;
  research_time: number;
  timestamp: string;
}

export interface ResearchHistory {
  items: ResearchHistoryItem[];
  total_count: number;
  page: number;
  page_size: number;
}

// Configuration types
export interface ResearchSettings {
  relevance_threshold: number;
  content_validation_threshold: number;
  consensus_threshold: number;
  max_research_time: number;
  enable_debug_logs: boolean;
}

// UI state types
export interface UIState {
  isLoading: boolean;
  error?: string;
  darkMode: boolean;
  sidebarOpen: boolean;
  currentPage: string;
}

export interface ResearchState {
  currentRequest?: ResearchRequest;
  currentResult?: ResearchResult;
  progress?: ResearchProgress;
  history: ResearchHistoryItem[];
  isResearching: boolean;
  error?: string;
}

// Utility types
export type ResearchStage = 
  | 'initializing'
  | 'domain_analysis'
  | 'source_search'
  | 'content_extraction'
  | 'validation'
  | 'synthesis'
  | 'completed'
  | 'error';

export interface QualityBadgeProps {
  level: QualityLevel;
  score?: number;
  showScore?: boolean;
}

export interface SourceCardProps {
  source: SourceModel;
  index: number;
  citationStyle: CitationStyle;
  onCitationCopy?: () => void;
}

export interface ProgressIndicatorProps {
  progress: ResearchProgress;
  animated?: boolean;
}
// Custom hooks for research functionality
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import type { RootState, AppDispatch } from '../store';
import {
  conductResearch,
  loadHistory,
  setCurrentRequest,
  setProgress,
  startResearch,
  stopResearch,
  clearError,
  clearCurrentResult,
  setCurrentResult,
  updateCitationStyle,
  updateResearchMode,
} from '../store/slices/researchSlice';
import { WebSocketService } from '../services/api';
import {
  ResearchRequest,
  ResearchProgress,
  ResearchResult,
  CitationStyle,
  ResearchMode,
} from '../types/research';

// Hook for health checking
export function useHealthCheck() {
  const [healthStatus, setHealthStatus] = useState<{
    status: 'healthy' | 'degraded' | 'error' | 'loading';
    openai_configured: boolean;
    openai_valid_format: boolean;
    issues: string[];
    recommendations: string[];
  }>({
    status: 'loading',
    openai_configured: false,
    openai_valid_format: false,
    issues: [],
    recommendations: []
  });

  const checkHealth = useCallback(async () => {
    try {
      const { apiService } = await import('../services/api');
      const health = await apiService.healthCheck();
      setHealthStatus({
        status: health.status as 'healthy' | 'degraded' | 'error',
        openai_configured: health.features?.openai_configured || false,
        openai_valid_format: health.features?.openai_valid_format || false,
        issues: health.research_service?.issues || [],
        recommendations: health.research_service?.recommendations || []
      });
    } catch (error) {
      setHealthStatus({
        status: 'error',
        openai_configured: false,
        openai_valid_format: false,
        issues: ['Failed to connect to research service'],
        recommendations: ['Check that the backend server is running']
      });
    }
  }, []);

  useEffect(() => {
    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return {
    healthStatus,
    checkHealth,
    isHealthy: healthStatus.status === 'healthy',
    hasApiKey: healthStatus.openai_configured && healthStatus.openai_valid_format
  };
}

// Hook for research state and actions
export function useResearch() {
  const dispatch = useDispatch<AppDispatch>();
  const research = useSelector((state: RootState) => state.research);
  
  const actions = {
    conductResearch: useCallback((request: ResearchRequest) => {
      dispatch(conductResearch(request));
    }, [dispatch]),
    
    loadHistory: useCallback((page?: number, pageSize?: number) => {
      dispatch(loadHistory({ page, pageSize }));
    }, [dispatch]),
    
    setCurrentRequest: useCallback((request: ResearchRequest) => {
      dispatch(setCurrentRequest(request));
    }, [dispatch]),
    
    startResearch: useCallback(() => {
      dispatch(startResearch());
    }, [dispatch]),
    
    stopResearch: useCallback(() => {
      dispatch(stopResearch());
    }, [dispatch]),
    
    clearError: useCallback(() => {
      dispatch(clearError());
    }, [dispatch]),
    
    clearCurrentResult: useCallback(() => {
      dispatch(clearCurrentResult());
    }, [dispatch]),
    
    setCurrentResult: useCallback((result: ResearchResult) => {
      dispatch(setCurrentResult(result));
    }, [dispatch]),
    
    updateCitationStyle: useCallback((style: CitationStyle) => {
      dispatch(updateCitationStyle(style));
    }, [dispatch]),
    
    updateResearchMode: useCallback((mode: ResearchMode) => {
      dispatch(updateResearchMode(mode));
    }, [dispatch]),
  };
  
  return {
    ...research,
    ...actions,
  };
}

// Hook for WebSocket connection
export function useWebSocket(clientId: string) {
  const wsRef = useRef<WebSocketService | null>(null);
  const dispatch = useDispatch<AppDispatch>();
  const researchTimeoutRef = useRef<number | null>(null);
  
  const connect = useCallback(async () => {
    if (!wsRef.current) {
      wsRef.current = new WebSocketService(clientId);
      
      // Set up message handlers
      wsRef.current.onMessage('research_progress', (data) => {
        try {
          const progress: ResearchProgress = data.progress;
          dispatch(setProgress(progress));
          
          // Reset timeout when we receive progress updates
          if (researchTimeoutRef.current) {
            clearTimeout(researchTimeoutRef.current);
          }
          
          // Set a new timeout (15 minutes + 1 minute buffer)
          researchTimeoutRef.current = setTimeout(() => {
            dispatch(setProgress({
              stage: 'error',
              progress: 0,
              message: 'Research operation timed out. The server may be overloaded. Please try again with a more specific query or use Basic mode.',
              sources_found: 0
            }));
          }, 16 * 60 * 1000); // 16 minutes
        } catch (error) {
          console.error('Error handling research progress:', error);
          dispatch(setProgress({
            stage: 'error',
            progress: 0,
            message: 'Error processing research progress. Please try again.',
            sources_found: 0
          }));
        }
      });
      
      wsRef.current.onMessage('research_complete', (data: ResearchResult) => {
        try {
          // Clear timeout when research completes
          if (researchTimeoutRef.current) {
            clearTimeout(researchTimeoutRef.current);
            researchTimeoutRef.current = null;
          }
          
          // Signal WebSocket that research is complete (re-enable heartbeat)
          wsRef.current?.setResearchComplete();
          
          // Clear session on completion
          localStorage.removeItem(`research-session-${clientId}`);
          
          // Set the actual research result data - this was missing!
          dispatch(setCurrentResult(data));
          
          dispatch(setProgress({
            stage: 'completed',
            progress: 1.0,
            message: 'Research completed successfully',
            sources_found: data?.sources?.length || 0
          }));
        } catch (error) {
          console.error('Error handling research completion:', error);
          dispatch(setProgress({
            stage: 'error',
            progress: 0,
            message: 'Error processing research results. Please try again.',
            sources_found: 0
          }));
        }
      });
      
      wsRef.current.onMessage('error', (data) => {
        // Clear timeout on error
        if (researchTimeoutRef.current) {
          clearTimeout(researchTimeoutRef.current);
          researchTimeoutRef.current = null;
        }
        
        // Signal WebSocket that research is complete (re-enable heartbeat)
        wsRef.current?.setResearchComplete();
        
        // Clear session on error
        localStorage.removeItem(`research-session-${clientId}`);
        
        dispatch(setProgress({
          stage: 'error',
          progress: 0,
          message: data.message || 'Research failed due to an unexpected error',
          sources_found: 0
        }));
      });
      
      wsRef.current.onMessage('connection_failed', (data) => {
        dispatch(setProgress({
          stage: 'error',
          progress: 0,
          message: 'Connection to research service failed. Please check your internet connection and try again.',
          sources_found: 0
        }));
      });
      
      wsRef.current.onMessage('connected', () => {
        console.log('✅ WebSocket connected successfully');
      });
      
      try {
        const apiService = await import('../services/api');
        await wsRef.current.connect(apiService.apiService.getWebSocketURL(clientId));
      } catch (error) {
        console.error('❌ Failed to connect WebSocket:', error);
        dispatch(setProgress({
          stage: 'error',
          progress: 0,
          message: 'Failed to establish connection to research service. Please refresh the page and try again.',
          sources_found: 0
        }));
      }
    }
  }, [clientId, dispatch]);
  
  const disconnect = useCallback(() => {
    // Clear research timeout
    if (researchTimeoutRef.current) {
      clearTimeout(researchTimeoutRef.current);
      researchTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }
  }, []);
  
  const startResearchWithWebSocket = useCallback((request: ResearchRequest) => {
    if (wsRef.current) {
      // Clear any existing timeout
      if (researchTimeoutRef.current) {
        clearTimeout(researchTimeoutRef.current);
      }
      
      // Save session for recovery
      localStorage.setItem(`research-session-${clientId}`, JSON.stringify({
        request,
        timestamp: Date.now(),
        clientId
      }));
      
      dispatch(startResearch());
      wsRef.current.startResearch(request);
      
      // Set initial timeout (15 minutes + 1 minute buffer)
      researchTimeoutRef.current = setTimeout(() => {
        dispatch(setProgress({
          stage: 'error',
          progress: 0,
          message: 'Research operation timed out. The server may be overloaded. Please try again with a more specific query or use Basic mode.',
          sources_found: 0
        }));
        // Clear session on timeout
        localStorage.removeItem(`research-session-${clientId}`);
      }, 16 * 60 * 1000); // 16 minutes
    }
  }, [dispatch, clientId]);
  
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);
  
  // Add session recovery effect
  useEffect(() => {
    // Check if there's a research session in progress from localStorage
    const savedSession = localStorage.getItem(`research-session-${clientId}`);
    if (savedSession) {
      try {
        const sessionData = JSON.parse(savedSession);
        const sessionAge = Date.now() - sessionData.timestamp;
        
        // If session is less than 30 minutes old, try to recover
        if (sessionAge < 30 * 60 * 1000) {
          console.log('🔄 Recovering research session:', sessionData);
          dispatch(startResearch());
          dispatch(setCurrentRequest(sessionData.request));
          
          // Set a recovery timeout
          setTimeout(() => {
            dispatch(setProgress({
              stage: 'error',
              progress: 0,
              message: 'Previous research session could not be recovered. Please start a new search.',
              sources_found: 0
            }));
            localStorage.removeItem(`research-session-${clientId}`);
          }, 5000); // 5 second timeout for recovery
        } else {
          // Clean up old session
          localStorage.removeItem(`research-session-${clientId}`);
        }
      } catch (error) {
        console.error('Failed to recover research session:', error);
        localStorage.removeItem(`research-session-${clientId}`);
      }
    }
  }, [clientId, dispatch]);
  
  return {
    isConnected: wsRef.current?.isConnected || false,
    startResearch: startResearchWithWebSocket,
    connect,
    disconnect,
  };
}

// Hook for research progress tracking
export function useResearchProgress() {
  const progress = useSelector((state: RootState) => state.research.progress);
  const isResearching = useSelector((state: RootState) => state.research.isResearching);
  
  const getProgressPercentage = useCallback(() => {
    return progress ? Math.round(progress.progress * 100) : 0;
  }, [progress]);
  
  const getProgressColor = useCallback(() => {
    if (!progress) return 'primary';
    
    switch (progress.stage) {
      case 'error':
        return 'error';
      case 'completed':
        return 'success';
      default:
        return 'primary';
    }
  }, [progress]);
  
  const getEstimatedTimeRemaining = useCallback(() => {
    if (!progress?.estimated_completion) return null;
    
    const completion = new Date(progress.estimated_completion);
    const now = new Date();
    const diff = completion.getTime() - now.getTime();
    
    if (diff <= 0) return null;
    
    const seconds = Math.ceil(diff / 1000);
    if (seconds < 60) return `${seconds}s`;
    
    const minutes = Math.ceil(seconds / 60);
    if (minutes < 60) return `${minutes}m`;
    
    const hours = Math.ceil(minutes / 60);
    return `${hours}h`;
  }, [progress]);
  
  return {
    progress,
    isResearching,
    progressPercentage: getProgressPercentage(),
    progressColor: getProgressColor(),
    estimatedTimeRemaining: getEstimatedTimeRemaining(),
  };
}

// Hook for research history management
export function useResearchHistory() {
  const history = useSelector((state: RootState) => state.research.history);
  const dispatch = useDispatch<AppDispatch>();
  
  const loadHistoryData = useCallback((page?: number, pageSize?: number) => {
    dispatch(loadHistory({ page, pageSize }));
  }, [dispatch]);
  
  const getRecentQueries = useCallback((limit: number = 5) => {
    return history.slice(0, limit).map(item => item.query);
  }, [history]);
  
  const getAverageQuality = useCallback(() => {
    const validItems = history.filter(item => item.quality_level);
    if (validItems.length === 0) return null;
    
    const qualityScores = validItems.map(item => {
      switch (item.quality_level) {
        case 'excellent': return 5;
        case 'good': return 4;
        case 'acceptable': return 3;
        case 'poor': return 2;
        case 'failed': return 1;
        default: return 3;
      }
    });
    
    const average = qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length;
    return Math.round(average * 10) / 10;
  }, [history]);
  
  const getTotalResearchTime = useCallback(() => {
    return history.reduce((total, item) => total + item.research_time, 0);
  }, [history]);
  
  return {
    history,
    loadHistory: loadHistoryData,
    recentQueries: getRecentQueries(),
    averageQuality: getAverageQuality(),
    totalResearchTime: getTotalResearchTime(),
    totalResearches: history.length,
  };
}
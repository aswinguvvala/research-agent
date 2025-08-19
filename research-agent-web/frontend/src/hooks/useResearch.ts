// Custom hooks for research functionality
import { useCallback, useEffect, useRef } from 'react';
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
  
  const connect = useCallback(async () => {
    if (!wsRef.current) {
      wsRef.current = new WebSocketService(clientId);
      
      // Set up message handlers
      wsRef.current.onMessage('research_progress', (data) => {
        const progress: ResearchProgress = data.progress;
        dispatch(setProgress(progress));
      });
      
      wsRef.current.onMessage('research_complete', (data: ResearchResult) => {
        dispatch(setProgress({
          stage: 'completed',
          progress: 1.0,
          message: 'Research completed successfully',
          sources_found: data.sources.length
        }));
      });
      
      wsRef.current.onMessage('error', (data) => {
        dispatch(setProgress({
          stage: 'error',
          progress: 0,
          message: data.message,
          sources_found: 0
        }));
      });
      
      try {
        const apiService = await import('../services/api');
        await wsRef.current.connect(apiService.apiService.getWebSocketURL(clientId));
        console.log('✅ WebSocket connected successfully');
      } catch (error) {
        console.error('❌ Failed to connect WebSocket:', error);
      }
    }
  }, [clientId, dispatch]);
  
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }
  }, []);
  
  const startResearchWithWebSocket = useCallback((request: ResearchRequest) => {
    if (wsRef.current) {
      dispatch(startResearch());
      wsRef.current.startResearch(request);
    }
  }, [dispatch]);
  
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);
  
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
  
  const loadHistory = useCallback((page?: number, pageSize?: number) => {
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
    loadHistory,
    recentQueries: getRecentQueries(),
    averageQuality: getAverageQuality(),
    totalResearchTime: getTotalResearchTime(),
    totalResearches: history.length,
  };
}
// Research state management slice
import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import {
  ResearchRequest,
  ResearchResult,
  ResearchProgress,
  ResearchHistoryItem,
  ResearchState,
  ResearchMode,
  CitationStyle
} from '../../types/research';
import { apiService } from '../../services/api';

// Initial state
const initialState: ResearchState = {
  currentRequest: undefined,
  currentResult: undefined,
  progress: undefined,
  history: [],
  isResearching: false,
  error: undefined,
};

// Async thunks
export const conductResearch = createAsyncThunk(
  'research/conduct',
  async (request: ResearchRequest, { rejectWithValue }) => {
    try {
      const result = await apiService.conductResearch(request);
      return { request, result };
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Research failed');
    }
  }
);

export const loadHistory = createAsyncThunk(
  'research/loadHistory',
  async ({ page = 1, pageSize = 10 }: { page?: number; pageSize?: number } = {}, { rejectWithValue }) => {
    try {
      const history = await apiService.getResearchHistory(page, pageSize);
      return history.items;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to load history');
    }
  }
);

export const getResearchResult = createAsyncThunk(
  'research/getResult',
  async (researchId: string, { rejectWithValue }) => {
    try {
      const result = await apiService.getResearchResult(researchId);
      return result;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Failed to get research result');
    }
  }
);

// Research slice
const researchSlice = createSlice({
  name: 'research',
  initialState,
  reducers: {
    setCurrentRequest: (state, action: PayloadAction<ResearchRequest>) => {
      state.currentRequest = action.payload;
      state.error = undefined;
    },
    
    setProgress: (state, action: PayloadAction<ResearchProgress>) => {
      state.progress = action.payload;
      state.isResearching = action.payload.stage !== 'completed' && action.payload.stage !== 'error';
      
      if (action.payload.stage === 'error') {
        state.error = action.payload.message;
        state.isResearching = false;
      }
    },
    
    startResearch: (state) => {
      state.isResearching = true;
      state.error = undefined;
      state.progress = undefined;
      state.currentResult = undefined;
    },
    
    stopResearch: (state) => {
      state.isResearching = false;
      state.progress = undefined;
    },
    
    clearError: (state) => {
      state.error = undefined;
    },
    
    clearCurrentResult: (state) => {
      state.currentResult = undefined;
      state.progress = undefined;
      state.isResearching = false;
    },
    
    addToHistory: (state, action: PayloadAction<ResearchHistoryItem>) => {
      // Add to beginning of history and limit to 50 items
      state.history.unshift(action.payload);
      if (state.history.length > 50) {
        state.history = state.history.slice(0, 50);
      }
    },
    
    updateCitationStyle: (state, action: PayloadAction<CitationStyle>) => {
      if (state.currentRequest) {
        state.currentRequest.citation_style = action.payload;
      }
    },
    
    updateResearchMode: (state, action: PayloadAction<ResearchMode>) => {
      if (state.currentRequest) {
        state.currentRequest.mode = action.payload;
      }
    },
  },
  
  extraReducers: (builder) => {
    builder
      // Conduct research
      .addCase(conductResearch.pending, (state) => {
        state.isResearching = true;
        state.error = undefined;
      })
      .addCase(conductResearch.fulfilled, (state, action) => {
        state.isResearching = false;
        state.currentRequest = action.payload.request;
        state.currentResult = action.payload.result;
        state.progress = {
          stage: 'completed',
          progress: 1.0,
          message: 'Research completed successfully',
          sources_found: action.payload.result.sources.length
        };
        
        // Add to history
        const historyItem: ResearchHistoryItem = {
          research_id: action.payload.result.research_id,
          query: action.payload.result.query,
          mode: action.payload.result.mode,
          quality_level: action.payload.result.quality_assessment?.overall_quality,
          sources_count: action.payload.result.sources.length,
          research_time: action.payload.result.research_time,
          timestamp: action.payload.result.timestamp
        };
        state.history.unshift(historyItem);
        if (state.history.length > 50) {
          state.history = state.history.slice(0, 50);
        }
      })
      .addCase(conductResearch.rejected, (state, action) => {
        state.isResearching = false;
        state.error = action.payload as string;
        state.progress = {
          stage: 'error',
          progress: 0,
          message: action.payload as string,
          sources_found: 0
        };
      })
      
      // Load history
      .addCase(loadHistory.fulfilled, (state, action) => {
        state.history = action.payload;
      })
      .addCase(loadHistory.rejected, (state, action) => {
        state.error = action.payload as string;
      })
      
      // Get research result
      .addCase(getResearchResult.fulfilled, (state, action) => {
        state.currentResult = action.payload;
      })
      .addCase(getResearchResult.rejected, (state, action) => {
        state.error = action.payload as string;
      });
  },
});

export const {
  setCurrentRequest,
  setProgress,
  startResearch,
  stopResearch,
  clearError,
  clearCurrentResult,
  addToHistory,
  updateCitationStyle,
  updateResearchMode,
} = researchSlice.actions;

export default researchSlice.reducer;
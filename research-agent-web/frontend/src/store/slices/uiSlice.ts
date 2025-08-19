// UI state management slice
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { UIState } from '../../types/research';

// Initial state
const initialState: UIState = {
  isLoading: false,
  error: undefined,
  darkMode: false,
  sidebarOpen: true,
  currentPage: 'research',
};

// UI slice
const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    
    setError: (state, action: PayloadAction<string | undefined>) => {
      state.error = action.payload;
    },
    
    clearError: (state) => {
      state.error = undefined;
    },
    
    toggleDarkMode: (state) => {
      state.darkMode = !state.darkMode;
      // Persist to localStorage
      localStorage.setItem('darkMode', JSON.stringify(state.darkMode));
    },
    
    setDarkMode: (state, action: PayloadAction<boolean>) => {
      state.darkMode = action.payload;
      localStorage.setItem('darkMode', JSON.stringify(state.darkMode));
    },
    
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
      localStorage.setItem('sidebarOpen', JSON.stringify(state.sidebarOpen));
    },
    
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
      localStorage.setItem('sidebarOpen', JSON.stringify(state.sidebarOpen));
    },
    
    setCurrentPage: (state, action: PayloadAction<string>) => {
      state.currentPage = action.payload;
    },
    
    initializeFromStorage: (state) => {
      // Load dark mode preference
      const savedDarkMode = localStorage.getItem('darkMode');
      if (savedDarkMode !== null) {
        state.darkMode = JSON.parse(savedDarkMode);
      } else {
        // Default to system preference
        state.darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
      }
      
      // Load sidebar preference
      const savedSidebarOpen = localStorage.getItem('sidebarOpen');
      if (savedSidebarOpen !== null) {
        state.sidebarOpen = JSON.parse(savedSidebarOpen);
      }
      
      // Apply dark mode to document
      if (state.darkMode) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    },
  },
});

export const {
  setLoading,
  setError,
  clearError,
  toggleDarkMode,
  setDarkMode,
  toggleSidebar,
  setSidebarOpen,
  setCurrentPage,
  initializeFromStorage,
} = uiSlice.actions;

export default uiSlice.reducer;
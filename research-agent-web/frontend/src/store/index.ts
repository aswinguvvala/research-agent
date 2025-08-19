// Redux store configuration
import { configureStore } from '@reduxjs/toolkit';
import researchSlice from './slices/researchSlice';
import uiSlice from './slices/uiSlice';

export const store = configureStore({
  reducer: {
    research: researchSlice,
    ui: uiSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['research/setProgress'],
        ignoredPaths: ['research.progress.estimated_completion'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Provider } from 'react-redux'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './index.css'
import { store } from './store'
import { ThemeProvider } from './contexts/ThemeContext'
import Layout from './components/Layout/Layout'
import ErrorBoundary from './components/ErrorBoundary'
import ResearchPage from './pages/ResearchPage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <ThemeProvider>
        <Provider store={store}>
          <Router>
            <Layout>
              <Routes>
                <Route path="/" element={<ResearchPage />} />
                <Route path="/research" element={<ResearchPage />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </Layout>
          </Router>
        </Provider>
      </ThemeProvider>
    </ErrorBoundary>
  </StrictMode>,
)

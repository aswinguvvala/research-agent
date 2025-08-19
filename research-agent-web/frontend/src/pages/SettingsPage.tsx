// Settings page - application configuration
import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { toggleDarkMode } from '../store/slices/uiSlice';
import { apiService } from '../services/api';
import { Cog6ToothIcon, CheckCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface BackendSettings {
  max_sources: number;
  relevance_threshold: number;
  content_validation_threshold: number;
  consensus_threshold: number;
  enhanced_mode_available: boolean;
  openai_configured: boolean;
}

const SettingsPage: React.FC = () => {
  const dispatch = useDispatch();
  const { darkMode } = useSelector((state: RootState) => state.ui);
  const [backendSettings, setBackendSettings] = useState<BackendSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setIsLoading(true);
      const settings = await apiService.getResearchSettings();
      setBackendSettings(settings);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load settings');
    } finally {
      setIsLoading(false);
    }
  };

  const checkBackendHealth = async () => {
    try {
      const health = await apiService.healthCheck();
      alert(`Backend is healthy!\nVersion: ${health.version}\nStatus: ${health.status}`);
    } catch (err) {
      alert(`Backend health check failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-3 mb-4">
          <Cog6ToothIcon className="h-8 w-8 text-primary-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Settings
          </h1>
        </div>
        <p className="text-gray-600 dark:text-gray-300">
          Configure your research agent preferences and view system status
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-error-50 border border-error-200 rounded-lg p-4">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-5 w-5 text-error-600 mr-2" />
            <p className="text-error-800">{error}</p>
          </div>
        </div>
      )}

      {/* UI Settings */}
      <div className="card p-6">
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          User Interface
        </h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                Dark Mode
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Switch between light and dark themes
              </p>
            </div>
            <button
              onClick={() => dispatch(toggleDarkMode())}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                darkMode ? 'bg-primary-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  darkMode ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Backend Settings */}
      {backendSettings && (
        <div className="card p-6">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Research Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Maximum Sources
              </h3>
              <p className="text-2xl font-bold text-primary-600">
                {backendSettings.max_sources}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Maximum number of sources per research
              </p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Relevance Threshold
              </h3>
              <p className="text-2xl font-bold text-primary-600">
                {Math.round(backendSettings.relevance_threshold * 100)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Minimum relevance score for sources
              </p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Content Validation
              </h3>
              <p className="text-2xl font-bold text-primary-600">
                {Math.round(backendSettings.content_validation_threshold * 100)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Content validation threshold
              </p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Consensus Threshold
              </h3>
              <p className="text-2xl font-bold text-primary-600">
                {Math.round(backendSettings.consensus_threshold * 100)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Cross-source consensus threshold
              </p>
            </div>
          </div>
        </div>
      )}

      {/* System Status */}
      {backendSettings && (
        <div className="card p-6">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            System Status
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                  Enhanced Mode
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Advanced validation and quality assessment
                </p>
              </div>
              <div className="flex items-center space-x-2">
                {backendSettings.enhanced_mode_available ? (
                  <>
                    <CheckCircleIcon className="h-5 w-5 text-success-600" />
                    <span className="text-sm text-success-600 font-medium">Available</span>
                  </>
                ) : (
                  <>
                    <ExclamationTriangleIcon className="h-5 w-5 text-warning-600" />
                    <span className="text-sm text-warning-600 font-medium">Unavailable</span>
                  </>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                  OpenAI Integration
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  AI-powered synthesis and analysis
                </p>
              </div>
              <div className="flex items-center space-x-2">
                {backendSettings.openai_configured ? (
                  <>
                    <CheckCircleIcon className="h-5 w-5 text-success-600" />
                    <span className="text-sm text-success-600 font-medium">Configured</span>
                  </>
                ) : (
                  <>
                    <ExclamationTriangleIcon className="h-5 w-5 text-warning-600" />
                    <span className="text-sm text-warning-600 font-medium">Not Configured</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="card p-6">
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          System Actions
        </h2>
        <div className="space-y-3">
          <button
            onClick={checkBackendHealth}
            className="btn-secondary"
          >
            Check Backend Health
          </button>
          <button
            onClick={loadSettings}
            className="btn-secondary"
          >
            Refresh Settings
          </button>
        </div>
      </div>

      {/* Version Info */}
      <div className="text-center text-sm text-gray-500 dark:text-gray-400">
        <p>Research Agent Web Application v1.0.0</p>
        <p>Built with React, TypeScript, and Tailwind CSS</p>
      </div>
    </div>
  );
};

export default SettingsPage;
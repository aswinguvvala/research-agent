// Research page - main research interface
import React, { useState } from 'react';
import { useResearch, useWebSocket } from '../hooks/useResearch';
import { ResearchMode, CitationStyle } from '../types/research';
import { MagnifyingGlassIcon, SparklesIcon } from '@heroicons/react/24/outline';

const ResearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<ResearchMode>(ResearchMode.ENHANCED);
  const [citationStyle, setCitationStyle] = useState<CitationStyle>(CitationStyle.APA);
  
  const {
    currentResult,
    isResearching,
    error,
    conductResearch,
    clearError,
  } = useResearch();

  const clientId = React.useMemo(() => `client-${Date.now()}-${Math.random()}`, []);
  const { startResearch: startWebSocketResearch } = useWebSocket(clientId);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isResearching) return;

    const request = {
      query: query.trim(),
      mode,
      citation_style: citationStyle,
    };

    // Use WebSocket for real-time updates
    startWebSocketResearch(request);
  };

  const handleExampleQuery = (exampleQuery: string) => {
    setQuery(exampleQuery);
  };

  const exampleQueries = [
    "What are the latest advances in large language model reasoning?",
    "Compare agile vs waterfall software development methodologies",
    "How effective are transformer architectures for natural language processing?",
    "What are the benefits and drawbacks of microservices architecture?",
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center space-x-3 mb-4">
          <MagnifyingGlassIcon className="h-10 w-10 text-primary-600" />
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            AI Research Agent
          </h1>
          <SparklesIcon className="h-10 w-10 text-primary-600" />
        </div>
        <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
          Conduct comprehensive research with AI-powered source discovery, validation, and synthesis
        </p>
      </div>

      {/* Research Form */}
      <div className="max-w-4xl mx-auto">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Query Input */}
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Research Question
            </label>
            <div className="relative">
              <input
                id="query"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your research question..."
                className="input-field-large pr-12"
                disabled={isResearching}
              />
              <button
                type="submit"
                disabled={!query.trim() || isResearching}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <MagnifyingGlassIcon className="h-5 w-5" />
              </button>
            </div>
          </div>

          {/* Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Research Mode */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Research Mode
              </label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as ResearchMode)}
                className="input-field"
                disabled={isResearching}
              >
                <option value={ResearchMode.BASIC}>Basic Mode</option>
                <option value={ResearchMode.ENHANCED}>Enhanced Mode (Recommended)</option>
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {mode === ResearchMode.ENHANCED 
                  ? 'Includes validation layers and quality assessment'
                  : 'Faster research with basic validation'
                }
              </p>
            </div>

            {/* Citation Style */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Citation Style
              </label>
              <select
                value={citationStyle}
                onChange={(e) => setCitationStyle(e.target.value as CitationStyle)}
                className="input-field"
                disabled={isResearching}
              >
                <option value={CitationStyle.APA}>APA Style</option>
                <option value={CitationStyle.MLA}>MLA Style</option>
                <option value={CitationStyle.IEEE}>IEEE Style</option>
              </select>
            </div>
          </div>

          {/* Submit Button */}
          <div className="text-center">
            <button
              type="submit"
              disabled={!query.trim() || isResearching}
              className="btn-primary px-8 py-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isResearching ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2 inline-block"></div>
                  Researching...
                </>
              ) : (
                'Start Research'
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Example Queries */}
      {!isResearching && !currentResult && (
        <div className="max-w-4xl mx-auto">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Example Research Questions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {exampleQueries.map((example, index) => (
              <button
                key={index}
                onClick={() => handleExampleQuery(example)}
                className="text-left p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-300 hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-colors"
              >
                <p className="text-sm text-gray-700 dark:text-gray-300">{example}</p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="max-w-4xl mx-auto">
          <div className="bg-error-50 border border-error-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <p className="text-error-800">{error}</p>
              <button
                onClick={clearError}
                className="text-error-600 hover:text-error-800"
              >
                ×
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Research Progress */}
      {isResearching && (
        <div className="max-w-4xl mx-auto">
          <div className="card p-6">
            <div className="text-center space-y-4">
              <div className="animate-pulse">
                <MagnifyingGlassIcon className="h-12 w-12 text-primary-600 mx-auto" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Conducting Research
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Searching sources and analyzing content...
              </p>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-primary-600 h-2 rounded-full animate-pulse" style={{ width: '45%' }}></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Research Results */}
      {currentResult && !isResearching && (
        <div className="max-w-4xl mx-auto">
          <div className="card p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Research Results
            </h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">Query:</h3>
                <p className="text-gray-700 dark:text-gray-300">{currentResult.query}</p>
              </div>
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">Sources Found:</h3>
                <p className="text-gray-700 dark:text-gray-300">{currentResult.sources.length}</p>
              </div>
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">Synthesis:</h3>
                <div className="prose max-w-none text-gray-700 dark:text-gray-300">
                  <p>{currentResult.synthesis}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResearchPage;
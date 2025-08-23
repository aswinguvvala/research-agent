import React, { useState } from 'react';
import { useResearch, useWebSocket, useResearchProgress, useHealthCheck } from '../hooks/useResearch';
import { ResearchMode, CitationStyle } from '../types/research';
import { 
  MagnifyingGlassIcon, 
  SparklesIcon, 
  DocumentTextIcon, 
  LightBulbIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  KeyIcon 
} from '@heroicons/react/24/outline';

const ResearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  
  const {
    currentResult,
    isResearching,
    error,
    conductResearch,
    clearError,
  } = useResearch();

  const clientId = React.useMemo(() => `client-${Date.now()}-${Math.random()}`, []);
  const { startResearch: startWebSocketResearch, disconnect: disconnectWebSocket } = useWebSocket(clientId);
  const { progress, progressPercentage, estimatedTimeRemaining } = useResearchProgress();
  const { healthStatus, isHealthy, hasApiKey } = useHealthCheck();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isResearching || !hasApiKey) return;

    const request = {
      query: query.trim(),
      mode: ResearchMode.ENHANCED, // Always use enhanced mode with unified agent
      citation_style: CitationStyle.APA, // Always use simple inline citations
    };

    startWebSocketResearch(request);
  };

  const handleExampleQuery = (exampleQuery: string) => {
    setQuery(exampleQuery);
  };

  const exampleQueries = [
    "What are the latest advances in large language model reasoning?",
    "History of neural networks from perceptron to transformers", 
    "Recent breakthroughs in quantum computing 2023-2024",
    "Compare agile vs waterfall software development methodologies",
  ];

  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8 animate-fade-in">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-block p-4 bg-dark-blue-800 rounded-full mb-4 shadow-hard-glow">
            <SparklesIcon className="h-10 w-10 text-accent-purple" />
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-dark-blue-300 to-accent-purple">
            AI Research Agent
          </h1>
          <p className="text-lg text-dark-blue-200 mt-4 max-w-2xl mx-auto">
            Comprehensive web search + academic papers + GPT-4o mini synthesis + simple inline citations.
          </p>
        </div>

        {/* API Status Alert */}
        {!hasApiKey && (
          <div className="glass card border-orange-500/50 mb-8 animate-fade-in">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <KeyIcon className="h-8 w-8 text-orange-400" />
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-orange-300 mb-3">
                  OpenAI API Key Required
                </h3>
                <p className="text-dark-blue-200 mb-4">
                  To use the research agent, you need to configure an OpenAI API key on the backend server.
                </p>
                
                <div className="bg-dark-blue-800 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold text-dark-blue-100 mb-2">Setup Instructions:</h4>
                  <ol className="list-decimal list-inside space-y-2 text-dark-blue-200 text-sm">
                    <li>Get an API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-accent-teal hover:underline">OpenAI Platform</a></li>
                    <li>Set the environment variable: <code className="bg-dark-blue-700 px-2 py-1 rounded text-accent-purple">OPENAI_API_KEY=your-key-here</code></li>
                    <li>Restart the backend server</li>
                    <li>Refresh this page</li>
                  </ol>
                </div>

                {healthStatus.issues.length > 0 && (
                  <div className="text-sm text-orange-300">
                    <strong>Current Issues:</strong>
                    <ul className="list-disc list-inside mt-1 space-y-1">
                      {healthStatus.issues.map((issue, index) => (
                        <li key={index}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Backend Status Indicator */}
        {healthStatus.status === 'error' && (
          <div className="glass card border-red-500/50 mb-8 animate-fade-in">
            <div className="flex items-center space-x-3">
              <ExclamationTriangleIcon className="h-6 w-6 text-red-400" />
              <div>
                <h3 className="text-lg font-semibold text-red-300">Backend Connection Error</h3>
                <p className="text-dark-blue-200">
                  Cannot connect to the research service. Make sure the backend server is running on port 8000.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Success Status */}
        {isHealthy && hasApiKey && (
          <div className="glass card border-green-500/50 mb-8 animate-fade-in">
            <div className="flex items-center space-x-3">
              <CheckCircleIcon className="h-6 w-6 text-green-400" />
              <div>
                <h3 className="text-lg font-semibold text-green-300">Unified Research Agent Ready</h3>
                <p className="text-dark-blue-200">
                  Comprehensive web search + GPT-4o mini + inline citations ready.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Research Form */}
        <div className="glass card card-hover mb-12 animate-slide-up">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute top-1/2 left-5 transform -translate-y-1/2 h-6 w-6 text-dark-blue-300" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your research question, topic, or keyword..."
                className="input-field pl-14 pr-40 py-4 text-lg bg-dark-blue-800 border-dark-blue-700 focus:ring-accent-purple"
                disabled={isResearching || !hasApiKey}
              />
              <button
                type="submit"
                disabled={!query.trim() || isResearching || !hasApiKey}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 btn-primary py-2 px-6 text-base disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {!hasApiKey ? 'API Key Required' : isResearching ? 'Researching...' : 'Start Research'}
              </button>
            </div>

          </form>
        </div>

        {/* Information Section */}
        {!isResearching && !currentResult && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            <div className="glass card card-hover p-6">
              <div className="flex items-center mb-4">
                <LightBulbIcon className="h-8 w-8 text-accent-teal mr-3" />
                <h3 className="text-xl font-semibold text-dark-blue-100">How it Works</h3>
              </div>
              <ul className="space-y-3 text-dark-blue-200">
                <li className="flex items-start"><span className="mr-3 mt-1">1.</span><span><span className="font-semibold">Enter Query:</span> Start with a clear research question.</span></li>
                <li className="flex items-start"><span className="mr-3 mt-1">2.</span><span><span className="font-semibold">Comprehensive Search:</span> Web search + academic papers with dynamic time filtering.</span></li>
                <li className="flex items-start"><span className="mr-3 mt-1">3.</span><span><span className="font-semibold">GPT-4o Mini Analysis:</span> AI analyzes sources and creates comprehensive summary.</span></li>
                <li className="flex items-start"><span className="mr-3 mt-1">4.</span><span><span className="font-semibold">Inline Citations:</span> Get summary with [1], [2], [3] citations and source links.</span></li>
              </ul>
            </div>
            <div className="glass card card-hover p-6">
              <div className="flex items-center mb-4">
                <DocumentTextIcon className="h-8 w-8 text-accent-pink mr-3" />
                <h3 className="text-xl font-semibold text-dark-blue-100">Example Queries</h3>
              </div>
              <div className="space-y-3">
                {exampleQueries.slice(0, 3).map((example, index) => (
                  <button
                    key={index}
                    onClick={() => handleExampleQuery(example)}
                    className="w-full text-left p-3 rounded-lg bg-dark-blue-800 hover:bg-dark-blue-700 transition-colors duration-200 text-dark-blue-200"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Results/Loading Section */}
        {isResearching && (
          <div className="text-center p-8 glass card">
            <div className="animate-subtle-pulse mb-6">
              <div className="inline-block p-4 bg-dark-blue-800 rounded-full shadow-hard-glow">
                <MagnifyingGlassIcon className="h-12 w-12 text-accent-purple" />
              </div>
            </div>
            <h2 className="text-2xl font-semibold text-dark-blue-100 mb-2">Research in Progress...</h2>
            <p className="text-dark-blue-200 mb-4">The AI agent is analyzing sources and synthesizing the report. Please wait.</p>
            
            {/* Enhanced Progress Indicator */}
            <div className="max-w-md mx-auto">
              <div className="flex justify-between text-sm text-dark-blue-300 mb-2">
                <span>Progress ({progressPercentage}%)</span>
                <span>
                  🚀 Enhanced Mode
                </span>
              </div>
              <div className="w-full bg-dark-blue-800 rounded-full h-3 mb-4">
                <div 
                  className="bg-gradient-to-r from-accent-purple to-accent-teal h-3 rounded-full transition-all duration-500 ease-out" 
                  style={{ width: `${Math.max(progressPercentage, 5)}%` }}
                ></div>
              </div>
              
              {/* Status and Time Estimate */}
              <div className="space-y-2 text-sm text-dark-blue-300">
                <p className="flex items-center justify-center gap-2">
                  <span className="w-2 h-2 bg-accent-purple rounded-full animate-pulse"></span>
                  {progress?.message || 'AI is processing your query...'}
                </p>
                <div className="flex justify-between text-xs">
                  <span>
                    {progress?.sources_found ? `${progress.sources_found} sources found` : 'Searching for sources...'}
                  </span>
                  <span>
                    {estimatedTimeRemaining ? `~${estimatedTimeRemaining} remaining` : 
                     'Estimated: 3-8 min'
                    }
                  </span>
                </div>
              </div>
            </div>
            
            {/* Cancel Option */}
            <button 
              onClick={() => {
                // Properly stop research and reset state without page reload
                disconnectWebSocket();
                // Clear any saved session
                localStorage.removeItem(`research-session-${clientId}`);
              }}
              className="mt-6 text-sm text-dark-blue-400 hover:text-dark-blue-200 transition-colors underline"
            >
              Cancel Research
            </button>
          </div>
        )}

        {error && (
          <div className="p-6 glass card border-red-500/50 animate-fade-in">
            <h3 className="text-xl font-semibold text-red-400 mb-3">Research Failed</h3>
            <p className="text-dark-blue-200 mb-4">{error}</p>
            
            <div className="space-y-3">
              <div className="text-sm text-dark-blue-300">
                <strong>What you can try:</strong>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>Try a more specific research query</li>
                  <li>Switch to Basic mode for faster results</li>
                  <li>Check your internet connection</li>
                  <li>Refresh the page if the issue persists</li>
                </ul>
              </div>
              
              <div className="flex gap-3">
                <button 
                  onClick={clearError} 
                  className="btn-primary flex-1"
                >
                  Try Again
                </button>
                <button 
                  onClick={() => {
                    clearError();
                    // Mode is always enhanced in the unified agent
                  }} 
                  className="btn-secondary flex-1"
                >
                  Try Enhanced Mode
                </button>
              </div>
            </div>
          </div>
        )}

        {currentResult && !isResearching && (
          <div className="glass card animate-fade-in">
            <h2 className="text-3xl font-bold text-dark-blue-100 mb-6">Research Report</h2>
            <div className="space-y-6">
              <div className="p-4 rounded-lg bg-dark-blue-800">
                <h4 className="font-semibold text-lg text-dark-blue-200 mb-2">Query:</h4>
                <p className="text-dark-blue-100 italic">{currentResult.query}</p>
              </div>
              <div className="prose prose-lg max-w-none text-dark-blue-200 prose-headings:text-dark-blue-100 prose-strong:text-dark-blue-100">
                <h3 className="text-2xl font-semibold">Synthesis</h3>
                <p>{currentResult.synthesis}</p>
              </div>
              <div>
                <h3 className="text-2xl font-semibold mb-4">Cited Sources ({currentResult.sources.length})</h3>
                <div className="space-y-6">
                  {currentResult.sources.map((source, index) => (
                    <div key={index} className="p-6 rounded-lg bg-dark-blue-800 border border-dark-blue-700">
                      {/* Source header */}
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <h4 className="text-lg font-semibold text-dark-blue-100 mb-2">
                            <a href={source.url} target="_blank" rel="noopener noreferrer" className="text-accent-teal hover:underline">
                              [{index + 1}] {source.title}
                            </a>
                          </h4>
                          <p className="text-dark-blue-300 text-sm">
                            {source.authors.join(', ')} ({source.year})
                            {source.journal && ` • ${source.journal}`}
                            {source.publisher && ` • ${source.publisher}`}
                          </p>
                        </div>
                        
                        {/* Quality indicators */}
                        <div className="flex flex-col items-end space-y-2">
                          {source.relevance_score && (
                            <div className="flex items-center space-x-2">
                              <span className="text-xs text-dark-blue-400">Relevance:</span>
                              <div className="w-16 bg-dark-blue-700 rounded-full h-2">
                                <div 
                                  className="bg-accent-purple h-2 rounded-full" 
                                  style={{ width: `${source.relevance_score * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-xs text-dark-blue-300">{Math.round(source.relevance_score * 100)}%</span>
                            </div>
                          )}
                          
                          {source.credibility_score && (
                            <div className="flex items-center space-x-2">
                              <span className="text-xs text-dark-blue-400">Credibility:</span>
                              <div className="w-16 bg-dark-blue-700 rounded-full h-2">
                                <div 
                                  className="bg-accent-teal h-2 rounded-full" 
                                  style={{ width: `${source.credibility_score * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-xs text-dark-blue-300">{Math.round(source.credibility_score * 100)}%</span>
                            </div>
                          )}
                          
                          {/* Quality badges */}
                          <div className="flex space-x-2">
                            {source.peer_reviewed && (
                              <span className="px-2 py-1 text-xs bg-green-900 text-green-300 rounded">Peer Reviewed</span>
                            )}
                            {source.open_access && (
                              <span className="px-2 py-1 text-xs bg-blue-900 text-blue-300 rounded">Open Access</span>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      {/* Abstract */}
                      {source.abstract && (
                        <div className="mb-4">
                          <h5 className="text-sm font-semibold text-dark-blue-200 mb-2">Abstract:</h5>
                          <p className="text-dark-blue-300 text-sm leading-relaxed">{source.abstract}</p>
                        </div>
                      )}
                      
                      {/* Key excerpts */}
                      {source.key_excerpts && source.key_excerpts.length > 0 && (
                        <div className="mb-4">
                          <h5 className="text-sm font-semibold text-dark-blue-200 mb-2">Key Excerpts:</h5>
                          <div className="space-y-2">
                            {source.key_excerpts.map((excerpt, excerptIndex) => (
                              <blockquote key={excerptIndex} className="border-l-4 border-accent-purple pl-4 text-dark-blue-300 text-sm italic">
                                "{excerpt}"
                              </blockquote>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Main topics */}
                      {source.main_topics && source.main_topics.length > 0 && (
                        <div className="mb-4">
                          <h5 className="text-sm font-semibold text-dark-blue-200 mb-2">Main Topics:</h5>
                          <div className="flex flex-wrap gap-2">
                            {source.main_topics.map((topic, topicIndex) => (
                              <span key={topicIndex} className="px-2 py-1 text-xs bg-dark-blue-700 text-dark-blue-200 rounded">
                                {topic}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Citation info */}
                      <div className="flex items-center justify-between pt-4 border-t border-dark-blue-700">
                        <div className="text-xs text-dark-blue-400">
                          {source.citation_count && `${source.citation_count} citations`}
                          {source.doi && ` • DOI: ${source.doi}`}
                        </div>
                        <div className="text-xs text-dark-blue-400">
                          {source.url && (
                            <a href={source.url} target="_blank" rel="noopener noreferrer" className="hover:text-accent-teal">
                              View Source →
                            </a>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResearchPage;

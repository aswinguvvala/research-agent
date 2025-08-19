import React, { useState } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('basic');
  const [citationStyle, setCitationStyle] = useState('apa');
  const [isResearching, setIsResearching] = useState(false);
  const [results, setResults] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsResearching(true);
    setResults(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/research/conduct', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          mode: mode,
          citation_style: citationStyle,
          max_sources: 5
        })
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        console.error('Research failed:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsResearching(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <h1 style={{ 
            color: 'white', 
            fontSize: '3rem', 
            fontWeight: 'bold',
            margin: '0 0 10px 0'
          }}>
            🔬 Research Agent
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: '1.2rem' }}>
            AI-Powered Research Assistant
          </p>
        </div>

        {/* Research Form */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '30px',
          boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
          marginBottom: '30px'
        }}>
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ 
                display: 'block', 
                marginBottom: '8px', 
                fontWeight: '600',
                color: '#374151'
              }}>
                Research Query
              </label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your research question..."
                style={{
                  width: '100%',
                  padding: '12px 16px',
                  border: '2px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '16px',
                  outline: 'none',
                  transition: 'border-color 0.2s'
                }}
                onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
                onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
              />
            </div>

            <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
              <div style={{ flex: 1 }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '8px', 
                  fontWeight: '600',
                  color: '#374151'
                }}>
                  Research Mode
                </label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '2px solid #e5e7eb',
                    borderRadius: '8px',
                    fontSize: '16px',
                    background: 'white'
                  }}
                >
                  <option value="basic">Basic Research</option>
                  <option value="enhanced">Enhanced Research</option>
                </select>
              </div>

              <div style={{ flex: 1 }}>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '8px', 
                  fontWeight: '600',
                  color: '#374151'
                }}>
                  Citation Style
                </label>
                <select
                  value={citationStyle}
                  onChange={(e) => setCitationStyle(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '2px solid #e5e7eb',
                    borderRadius: '8px',
                    fontSize: '16px',
                    background: 'white'
                  }}
                >
                  <option value="apa">APA Style</option>
                  <option value="mla">MLA Style</option>
                  <option value="ieee">IEEE Style</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={!query.trim() || isResearching}
              style={{
                width: '100%',
                padding: '16px',
                background: isResearching ? '#9ca3af' : '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '18px',
                fontWeight: '600',
                cursor: isResearching ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.2s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px'
              }}
            >
              {isResearching ? (
                <>
                  <div style={{
                    width: '20px',
                    height: '20px',
                    border: '3px solid rgba(255,255,255,0.3)',
                    borderRadius: '50%',
                    borderTopColor: 'white',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  Researching...
                </>
              ) : (
                '🚀 Start Research'
              )}
            </button>
          </form>
        </div>

        {/* Results Section */}
        {results && (
          <div style={{
            background: 'white',
            borderRadius: '16px',
            padding: '30px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ color: '#1f2937', marginBottom: '20px' }}>
              📊 Research Results
            </h2>
            
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ color: '#374151', marginBottom: '10px' }}>Query:</h3>
              <p style={{ color: '#6b7280', fontStyle: 'italic' }}>{results.query}</p>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ color: '#374151', marginBottom: '10px' }}>Synthesis:</h3>
              <div style={{ 
                background: '#f9fafb', 
                padding: '15px', 
                borderRadius: '8px',
                whiteSpace: 'pre-wrap'
              }}>
                {results.synthesis}
              </div>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ color: '#374151', marginBottom: '10px' }}>
                Sources ({results.sources?.length || 0}):
              </h3>
              {results.sources?.map((source, index) => (
                <div key={index} style={{
                  background: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  padding: '15px',
                  marginBottom: '10px'
                }}>
                  <h4 style={{ color: '#1f2937', marginBottom: '5px' }}>
                    {source.title}
                  </h4>
                  <p style={{ color: '#6b7280', fontSize: '14px' }}>
                    {source.authors?.join(', ')} ({source.year})
                  </p>
                  {source.url && (
                    <a 
                      href={source.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{ color: '#3b82f6', textDecoration: 'none' }}
                    >
                      📎 View Source
                    </a>
                  )}
                </div>
              ))}
            </div>

            <div style={{ 
              fontSize: '14px', 
              color: '#6b7280',
              textAlign: 'center',
              paddingTop: '20px',
              borderTop: '1px solid #e5e7eb'
            }}>
              ⏱️ Research completed in {results.research_time?.toFixed(2)} seconds
            </div>
          </div>
        )}
      </div>

      <style>
        {`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}

export default App;
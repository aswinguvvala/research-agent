import React, { useState } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('basic');
  const [citationStyle, setCitationStyle] = useState('apa');
  const [isResearching, setIsResearching] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsResearching(true);
    setResults(null);
    setError(null);

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
        const errorData = await response.json();
        setError(errorData.detail || 'Research failed. Please check the backend logs for more details.');
        console.error('Research failed:', response.statusText);
      }
    } catch (error) {
      setError('An unexpected error occurred. Please ensure the backend is running and accessible.');
      console.error('Error:', error);
    } finally {
      setIsResearching(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'url(https://images.unsplash.com/photo-1517976487-14210383765b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      padding: '20px'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <h1 style={{
            color: 'white',
            fontSize: '3.5rem',
            fontWeight: 'bold',
            margin: '0 0 10px 0',
            textShadow: '0 2px 4px rgba(0,0,0,0.5)'
          }}>
            🔬 Advanced Research Agent
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: '1.3rem', textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>
            Your AI-Powered Research Partner
          </p>
        </div>

        {/* Research Form */}
        <div style={{
          background: 'rgba(255,255,255,0.9)',
          backdropFilter: 'blur(10px)',
          borderRadius: '16px',
          padding: '40px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
          marginBottom: '30px'
        }}>
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '25px' }}>
              <label style={{
                display: 'block',
                marginBottom: '10px',
                fontWeight: '700',
                color: '#1f2937',
                fontSize: '1.1rem'
              }}>
                What would you like to research?
              </label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., 'The future of AI in healthcare'"
                style={{
                  width: '100%',
                  padding: '14px 18px',
                  border: '2px solid #d1d5db',
                  borderRadius: '10px',
                  fontSize: '16px',
                  outline: 'none',
                  transition: 'all 0.3s ease',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = '#3b82f6';
                  e.target.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.3)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = '#d1d5db';
                  e.target.style.boxShadow = 'none';
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '25px', marginBottom: '25px' }}>
              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '10px',
                  fontWeight: '700',
                  color: '#1f2937',
                  fontSize: '1.1rem'
                }}>
                  Research Depth
                </label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '14px 18px',
                    border: '2px solid #d1d5db',
                    borderRadius: '10px',
                    fontSize: '16px',
                    background: 'white',
                    appearance: 'none',
                    cursor: 'pointer'
                  }}
                >
                  <option value="basic">Quick Summary</option>
                  <option value="enhanced">In-depth Analysis</option>
                </select>
              </div>

              <div>
                <label style={{
                  display: 'block',
                  marginBottom: '10px',
                  fontWeight: '700',
                  color: '#1f2937',
                  fontSize: '1.1rem'
                }}>
                  Citation Format
                </label>
                <select
                  value={citationStyle}
                  onChange={(e) => setCitationStyle(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '14px 18px',
                    border: '2px solid #d1d5db',
                    borderRadius: '10px',
                    fontSize: '16px',
                    background: 'white',
                    appearance: 'none',
                    cursor: 'pointer'
                  }}
                >
                  <option value="apa">APA (American Psychological Association)</option>
                  <option value="mla">MLA (Modern Language Association)</option>
                  <option value="ieee">IEEE (Institute of Electrical and Electronics Engineers)</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={!query.trim() || isResearching}
              style={{
                width: '100%',
                padding: '18px',
                background: isResearching ? '#6b7280' : 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                color: 'white',
                border: 'none',
                borderRadius: '10px',
                fontSize: '18px',
                fontWeight: 'bold',
                cursor: isResearching ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '12px',
                boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = isResearching ? 'none' : 'translateY(-2px)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'none'}
            >
              {isResearching ? (
                <>
                  <div style={{
                    width: '20px',
                    height: '20px',
                    border: '3px solid rgba(255,255,255,0.4)',
                    borderRadius: '50%',
                    borderTopColor: 'white',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  Analyzing...
                </>
              ) : (
                '🚀 Begin Research'
              )}
            </button>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div style={{
            background: 'rgba(255, 107, 107, 0.9)',
            color: 'white',
            borderRadius: '16px',
            padding: '20px',
            boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
            marginBottom: '30px',
            textAlign: 'center'
          }}>
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}

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
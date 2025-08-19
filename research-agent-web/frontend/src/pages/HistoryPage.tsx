// History page - research history and management
import React, { useEffect } from 'react';
import { useResearchHistory } from '../hooks/useResearch';
import { ClockIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';

const HistoryPage: React.FC = () => {
  const { history, loadHistory, averageQuality, totalResearchTime, totalResearches } = useResearchHistory();

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getQualityBadgeColor = (level?: string) => {
    switch (level) {
      case 'excellent': return 'badge-success';
      case 'good': return 'badge-primary';
      case 'acceptable': return 'badge-warning';
      case 'poor': return 'badge-error';
      default: return 'badge-gray';
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-3 mb-4">
          <ClockIcon className="h-8 w-8 text-primary-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Research History
          </h1>
        </div>
        <p className="text-gray-600 dark:text-gray-300">
          View and manage your past research sessions
        </p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <MagnifyingGlassIcon className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Total Researches
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {totalResearches}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ClockIcon className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Total Time
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatDuration(totalResearchTime)}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="h-8 w-8 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-primary-600 font-bold text-sm">★</span>
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Average Quality
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {averageQuality ? `${averageQuality}/5` : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* History List */}
      <div className="card">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            Recent Research Sessions
          </h2>
        </div>

        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {history.length === 0 ? (
            <div className="px-6 py-12 text-center">
              <MagnifyingGlassIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                No research history
              </h3>
              <p className="text-gray-500 dark:text-gray-400">
                Start your first research to see it appear here.
              </p>
            </div>
          ) : (
            history.map((item) => (
              <div key={item.research_id} className="px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {item.query}
                    </h3>
                    <div className="mt-1 flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                      <span>
                        {new Date(item.timestamp).toLocaleDateString()}
                      </span>
                      <span>
                        {item.sources_count} sources
                      </span>
                      <span>
                        {formatDuration(item.research_time)}
                      </span>
                      <span className="capitalize">
                        {item.mode} mode
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    {item.quality_level && (
                      <span className={`badge ${getQualityBadgeColor(item.quality_level)}`}>
                        {item.quality_level}
                      </span>
                    )}
                    <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                      View
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default HistoryPage;
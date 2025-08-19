// Sidebar component
import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useLocation, useNavigate } from 'react-router-dom';
import { RootState } from '../../store';
import { setCurrentPage } from '../../store/slices/uiSlice';
import {
  MagnifyingGlassIcon,
  ClockIcon,
  Cog6ToothIcon,
  AcademicCapIcon,
} from '@heroicons/react/24/outline';

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ElementType;
  current: boolean;
}

const Sidebar: React.FC = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const location = useLocation();
  const { sidebarOpen } = useSelector((state: RootState) => state.ui);

  const navigationItems: NavigationItem[] = [
    {
      name: 'Research',
      href: '/research',
      icon: MagnifyingGlassIcon,
      current: location.pathname === '/' || location.pathname === '/research',
    },
    {
      name: 'History',
      href: '/history',
      icon: ClockIcon,
      current: location.pathname === '/history',
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Cog6ToothIcon,
      current: location.pathname === '/settings',
    },
  ];

  const handleNavigation = (item: NavigationItem) => {
    navigate(item.href);
    dispatch(setCurrentPage(item.name.toLowerCase()));
  };

  return (
    <div className={`
      fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 
      transform transition-transform duration-300 ease-in-out
      ${sidebarOpen ? 'translate-x-0' : '-translate-x-48'}
    `}>
      <div className="flex flex-col h-full">
        {/* Logo */}
        <div className="flex items-center h-16 px-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <AcademicCapIcon className="h-8 w-8 text-primary-600 dark:text-primary-400" />
            {sidebarOpen && (
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                  Research Agent
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  AI-Powered Research
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.name}
                onClick={() => handleNavigation(item)}
                className={`
                  w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors
                  ${item.current
                    ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-200'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700'
                  }
                `}
              >
                <Icon className="h-5 w-5 mr-3 flex-shrink-0" />
                {sidebarOpen && (
                  <span className="truncate">{item.name}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        {sidebarOpen && (
          <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <p>Research Agent v1.0</p>
              <p>Real research, real results</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
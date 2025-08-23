import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useLocation, useNavigate } from 'react-router-dom';
import { RootState } from '../../store';
import { setCurrentPage, toggleSidebar } from '../../store/slices/uiSlice';
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
    },
    {
      name: 'History',
      href: '/history',
      icon: ClockIcon,
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Cog6ToothIcon,
    },
  ];

  const handleNavigation = (item: NavigationItem) => {
    navigate(item.href);
    dispatch(setCurrentPage(item.name.toLowerCase()));
  };

  return (
    <>
      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/60 z-30 md:hidden" onClick={() => dispatch(toggleSidebar())} />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-40 transition-transform duration-300 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        md:relative md:translate-x-0
        w-64 bg-dark-blue-900 border-r border-dark-blue-800 flex flex-col
      `}>
        {/* Logo */}
        <div className="flex items-center h-16 px-6 border-b border-dark-blue-800 flex-shrink-0">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-accent-purple rounded-lg shadow-hard-glow">
              <AcademicCapIcon className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-lg font-semibold text-dark-blue-100">Research Agent</h1>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-6 py-8 space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const current = location.pathname.startsWith(item.href);
            return (
              <button
                key={item.name}
                onClick={() => handleNavigation(item)}
                className={`
                  w-full flex items-center px-4 py-3 text-base font-medium rounded-lg transition-colors duration-200
                  ${current
                    ? 'bg-dark-blue-800 text-white'
                    : 'text-dark-blue-300 hover:text-white hover:bg-dark-blue-800'
                  }
                `}
              >
                <Icon className="h-6 w-6 mr-4" />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-dark-blue-800">
          <div className="text-xs text-dark-blue-400">
            <p>v1.0.0 - Enhanced UI</p>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;

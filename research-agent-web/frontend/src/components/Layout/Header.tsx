import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { toggleSidebar } from '../../store/slices/uiSlice';
import { useTheme } from '../../contexts/ThemeContext';
import { Bars3Icon, MoonIcon, SunIcon, BellIcon } from '@heroicons/react/24/outline';

const Header: React.FC = () => {
  const dispatch = useDispatch();
  const { isDark, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-30 w-full bg-opacity-50 backdrop-filter backdrop-blur-lg bg-dark-blue-900 border-b border-dark-blue-700">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 -mb-px">
          {/* Header: Left side */}
          <div className="flex items-center space-x-4">
            <button
              onClick={() => dispatch(toggleSidebar())}
              className="text-dark-blue-300 hover:text-white"
            >
              <Bars3Icon className="w-6 h-6" />
            </button>
            <h1 className="text-lg font-semibold text-dark-blue-100">AI Research Agent</h1>
          </div>

          {/* Header: Right side */}
          <div className="flex items-center space-x-4">
            <button className="p-2 rounded-full text-dark-blue-300 hover:text-white hover:bg-dark-blue-800">
              <BellIcon className="w-6 h-6" />
            </button>
            <button
              onClick={toggleTheme}
              className="p-2 rounded-full text-dark-blue-300 hover:text-white hover:bg-dark-blue-800"
            >
              {isDark ? <SunIcon className="w-6 h-6" /> : <MoonIcon className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;

// Header component
import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store';
import { toggleSidebar, toggleDarkMode } from '../../store/slices/uiSlice';
import { Bars3Icon, MoonIcon, SunIcon } from '@heroicons/react/24/outline';

const Header: React.FC = () => {
  const dispatch = useDispatch();
  const { darkMode, currentPage } = useSelector((state: RootState) => state.ui);

  return (
    <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          <button
            onClick={() => dispatch(toggleSidebar())}
            className="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700 transition-colors"
          >
            <Bars3Icon className="h-5 w-5" />
          </button>
          
          <h1 className="text-xl font-semibold text-gray-900 dark:text-white capitalize">
            {currentPage}
          </h1>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Dark mode toggle */}
          <button
            onClick={() => dispatch(toggleDarkMode())}
            className="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700 transition-colors"
          >
            {darkMode ? (
              <SunIcon className="h-5 w-5" />
            ) : (
              <MoonIcon className="h-5 w-5" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
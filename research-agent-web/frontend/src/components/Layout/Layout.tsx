// Main layout component
import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';
import Sidebar from './Sidebar';
import Header from './Header';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { sidebarOpen, darkMode } = useSelector((state: RootState) => state.ui);

  return (
    <div className={`min-h-screen bg-gray-50 ${darkMode ? 'dark' : ''}`}>
      <div className="flex">
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main content */}
        <div className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ${
          sidebarOpen ? 'ml-64' : 'ml-16'
        }`}>
          {/* Header */}
          <Header />
          
          {/* Page content */}
          <main className="flex-1 p-6">
            <div className="max-w-7xl mx-auto">
              {children}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default Layout;
import React from 'react';
import { Home, TrendingUp, TrendingDown, Search } from 'lucide-react';

const Navigation = ({ currentPage, setCurrentPage }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'volatile', label: 'Volatile Stocks', icon: TrendingUp },
    { id: 'stable', label: 'Stable Stocks', icon: TrendingDown },
    { id: 'lookup', label: 'Stock Lookup', icon: Search },
  ];

  return (
    <nav className="bg-black border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-lg flex items-center justify-center">
              <span className="text-black font-bold text-xl">R</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">RAMP</h1>
              <p className="text-xs text-gray-400">Risk-Aware Market Predictor</p>
            </div>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentPage(item.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    isActive
                      ? 'bg-cyan-400 text-black'
                      : 'text-gray-300 hover:bg-gray-800'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {item.label}
                </button>
              );
            })}
          </div>

          {/* Mobile Navigation */}
          <div className="md:hidden flex items-center space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentPage(item.id)}
                  className={`p-2 rounded-lg ${
                    isActive
                      ? 'bg-cyan-400 text-black'
                      : 'text-gray-300 hover:bg-gray-800'
                  }`}
                  title={item.label}
                >
                  <Icon className="w-5 h-5" />
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
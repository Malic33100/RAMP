import React, { useState } from 'react';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import VolatileStocks from './components/VolatileStocks';
import StableStocks from './components/StableStocks';
import StockLookup from './components/StockLookup';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'volatile':
        return <VolatileStocks />;
      case 'stable':
        return <StableStocks />;
      case 'lookup':
        return <StockLookup />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="App">
      <Navigation currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <main>
        {renderPage()}
      </main>
    </div>
  );
}

export default App;

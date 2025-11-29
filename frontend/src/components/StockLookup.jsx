import React, { useState } from 'react';
import axios from 'axios';
import { Search, TrendingUp, AlertCircle } from 'lucide-react';

const StockLookup = () => {
  const [symbol, setSymbol] = useState('');
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!symbol.trim()) {
      setError('Please enter a stock symbol');
      return;
    }

    setLoading(true);
    setError(null);
    setStockData(null);

    try {
      const response = await axios.get(`http://localhost:5000/api/stocks/${symbol.toUpperCase()}`);
      setStockData(response.data);
    } catch (err) {
      if (err.response?.status === 404) {
        setError(`Stock '${symbol.toUpperCase()}' not found in RAMP database`);
      } else {
        setError('Failed to fetch stock data');
      }
      console.error('Error fetching stock:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (category) => {
    switch (category) {
      case 'HIGH':
        return 'bg-orange-500 bg-opacity-10 border-orange-500 text-orange-300';
      case 'MODERATE':
        return 'bg-yellow-500 bg-opacity-10 border-yellow-500 text-yellow-300';
      case 'LOW':
        return 'bg-cyan-400 bg-opacity-10 border-cyan-400 text-cyan-300';
      default:
        return 'bg-gray-800 border-gray-700 text-gray-300';
    }
  };

  const getRiskEmoji = (category) => {
    switch (category) {
      case 'HIGH':
        return '🔴';
      case 'MODERATE':
        return '🟡';
      case 'LOW':
        return '🟢';
      default:
        return '⚪';
    }
  };

  return (
    <div className="min-h-screen bg-black py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            📊 Stock Lookup
          </h1>
          <p className="text-gray-400">
            Search for detailed RAMP analysis of any stock
          </p>
        </div>

        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 mb-8">
          <form onSubmit={handleSearch}>
            <div className="flex gap-4">
              <div className="flex-1">
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  placeholder="Enter stock symbol (e.g., AAPL, MSFT, TSLA)"
                  className="w-full px-4 py-3 bg-black border border-gray-700 text-white rounded-lg focus:ring-2 focus:ring-cyan-400 focus:border-cyan-400 text-lg"
                  disabled={loading}
                />
              </div>
              <button
                type="submit"
                disabled={loading}
                className="px-8 py-3 bg-cyan-400 text-black rounded-lg hover:bg-cyan-500 disabled:bg-gray-700 disabled:cursor-not-allowed flex items-center gap-2 font-semibold transition-colors"
              >
                <Search className="w-5 h-5" />
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </form>
        </div>

        {error && (
          <div className="bg-orange-500 bg-opacity-10 border border-orange-500 rounded-lg p-4 mb-8 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-orange-500 font-semibold">Error</p>
              <p className="text-orange-300 text-sm">{error}</p>
            </div>
          </div>
        )}

        {stockData && (
          <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
            <div className="bg-gradient-to-r from-cyan-600 to-cyan-500 px-6 py-8 text-black">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-4xl font-bold mb-2">{stockData.symbol}</h2>
                  <p className="text-cyan-900 text-lg">{stockData.industry}</p>
                </div>
                <div className="text-right">
                  <p className="text-cyan-900 text-sm mb-1">RAMP Score</p>
                  <p className="text-5xl font-bold">{stockData.ramp_score}</p>
                </div>
              </div>
            </div>

            <div className="p-6 space-y-6">
              <div className={`border-2 rounded-lg p-6 ${getRiskColor(stockData.risk_category)}`}>
                <div className="flex items-center gap-3">
                  <span className="text-4xl">{getRiskEmoji(stockData.risk_category)}</span>
                  <div>
                    <p className="text-sm font-medium opacity-75">Risk Category</p>
                    <p className="text-2xl font-bold">{stockData.risk_category} VOLATILITY</p>
                    <p className="text-sm mt-1">{stockData.risk_description}</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-black rounded-lg p-4 border border-gray-800">
                  <p className="text-sm text-gray-400 mb-1">Predicted Volatility</p>
                  <p className="text-3xl font-bold text-white">
                    {stockData.predicted_volatility.toFixed(2)}%
                  </p>
                </div>

                <div className="bg-black rounded-lg p-4 border border-gray-800">
                  <p className="text-sm text-gray-400 mb-1">Percentile Rank</p>
                  <p className="text-3xl font-bold text-white">
                    {Math.round(stockData.percentile)}th
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    More volatile than {Math.round(stockData.percentile)}% of stocks
                  </p>
                </div>

                <div className="bg-black rounded-lg p-4 border border-gray-800">
                  <p className="text-sm text-gray-400 mb-1">Industry</p>
                  <p className="text-xl font-bold text-white">
                    {stockData.industry}
                  </p>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-400">RAMP Score Scale</p>
                  <p className="text-sm text-gray-500">0 (Stable) → 100 (Volatile)</p>
                </div>
                <div className="relative w-full h-8 bg-gradient-to-r from-cyan-400 via-yellow-500 to-orange-500 rounded-full">
                  <div 
                    className="absolute top-1/2 -translate-y-1/2 w-1 h-10 bg-white rounded"
                    style={{ left: `${stockData.ramp_score}%` }}
                  />
                  <div 
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 bg-white text-black px-2 py-1 rounded text-xs font-bold"
                    style={{ left: `${stockData.ramp_score}%`, top: '-30px' }}
                  >
                    {stockData.ramp_score}
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 border border-cyan-500 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Interpretation
                </h3>
                <p className="text-sm text-gray-300">
                  {stockData.risk_category === 'HIGH' && 
                    'This stock shows high predicted volatility. Suitable for risk-tolerant investors seeking potential high returns with significant price swings.'}
                  {stockData.risk_category === 'MODERATE' && 
                    'This stock shows moderate volatility. Offers a balance between growth potential and stability for diversified portfolios.'}
                  {stockData.risk_category === 'LOW' && 
                    'This stock shows low predicted volatility. Ideal for conservative investors prioritizing capital preservation and steady returns.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {!stockData && !error && !loading && (
          <div className="text-center py-16">
            <Search className="w-16 h-16 text-gray-700 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">
              Enter a stock symbol to view detailed RAMP analysis
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockLookup;
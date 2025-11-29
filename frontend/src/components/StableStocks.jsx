import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TrendingDown, Activity } from 'lucide-react';

const StableStocks = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStableStocks();
  }, []);

  const fetchStableStocks = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5000/api/stocks/stable');
      setData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load stable stocks');
      console.error('Error fetching stable stocks:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-cyan-400 mx-auto mb-4" />
          <p className="text-gray-400">Loading stable stocks...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="bg-gray-900 border border-cyan-400 rounded-lg p-6 max-w-md">
          <p className="text-cyan-400 font-semibold">Error</p>
          <p className="text-gray-300">{error}</p>
          <button 
            onClick={fetchStableStocks}
            className="mt-4 px-4 py-2 bg-cyan-400 text-black rounded hover:bg-cyan-500 font-medium"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <TrendingDown className="w-10 h-10 text-cyan-400" />
            <h1 className="text-4xl font-bold text-white">
              Most Stable Stocks
            </h1>
          </div>
          <p className="text-gray-400">
            Top 10 stocks with lowest predicted volatility
          </p>
        </div>

        {/* Least Volatile Industry Badge */}
        {data?.least_volatile_industry && (
          <div className="bg-gray-900 border-2 border-cyan-400 rounded-lg p-4 mb-6">
            <p className="text-sm text-gray-300 font-medium">
              🟢 Most Stable Industry: <span className="font-bold text-cyan-400">{data.least_volatile_industry}</span>
            </p>
          </div>
        )}

        {/* Stocks Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data?.stocks?.map((stock, index) => (
            <div 
              key={stock.symbol}
              className="bg-gray-900 rounded-lg hover:bg-gray-800 transition-all overflow-hidden border border-gray-800"
            >
              {/* Header with Rank */}
              <div className="bg-gradient-to-r from-cyan-500 to-cyan-400 px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="bg-black text-cyan-400 w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg">
                    {index + 1}
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-black">{stock.symbol}</h3>
                    <p className="text-cyan-900 text-sm">{stock.industry}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-cyan-900 text-xs">RAMP Score</p>
                  <p className="text-3xl font-bold text-black">{stock.ramp_score}</p>
                </div>
              </div>

              {/* Stats */}
              <div className="p-6">
                <div className="grid grid-cols-2 gap-4">
                  {/* Predicted Volatility */}
                  <div className="bg-black rounded-lg p-4 border border-gray-800">
                    <p className="text-xs text-gray-400 mb-1">Predicted Volatility</p>
                    <p className="text-2xl font-bold text-white">
                      {stock.predicted_volatility.toFixed(2)}%
                    </p>
                  </div>

                  {/* RAMP Score Gauge */}
                  <div className="bg-black rounded-lg p-4 border border-gray-800">
                    <p className="text-xs text-gray-400 mb-1">Risk Level</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-800 rounded-full h-2">
                        <div 
                          className="bg-cyan-400 h-2 rounded-full"
                          style={{ width: `${stock.ramp_score}%` }}
                        />
                      </div>
                      <span className="text-sm font-bold text-white">
                        {stock.ramp_score}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Info Badge */}
                <div className="mt-4 bg-cyan-400 bg-opacity-10 border border-cyan-400 rounded-lg p-3">
                  <p className="text-xs text-cyan-300">
                    <span className="font-semibold">🟢 Low Risk:</span> Minimal price fluctuations. 
                    Ideal for conservative investors and capital preservation.
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Info Box */}
        <div className="mt-8 bg-gray-900 border border-cyan-500 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-2">
            About Low Volatility Stocks
          </h3>
          <p className="text-gray-300 text-sm mb-2">
            These stocks have RAMP scores below 30, indicating low predicted volatility based on:
          </p>
          <ul className="text-gray-300 text-sm space-y-1 ml-5 list-disc">
            <li>Consistent historical price stability</li>
            <li>Steady volume patterns without major spikes</li>
            <li>Strong technical support levels</li>
            <li>Stable industry characteristics</li>
          </ul>
          <p className="text-gray-300 text-sm mt-3">
            <span className="font-semibold text-cyan-400">Investment Strategy:</span> Low volatility stocks are excellent 
            for portfolio stability, retirement accounts, and risk-averse investors. They typically offer 
            steady dividends and capital preservation, though with lower growth potential.
          </p>
        </div>
      </div>
    </div>
  );
};

export default StableStocks;
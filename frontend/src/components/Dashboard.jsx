import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, Activity, Database } from 'lucide-react';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [modelAccuracy, setModelAccuracy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      const [statsResponse, accuracyResponse] = await Promise.all([
        axios.get('http://localhost:5000/api/stats/overview'),
        axios.get('http://localhost:5000/api/model-accuracy')
      ]);

      setStats(statsResponse.data);
      setModelAccuracy(accuracyResponse.data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Error fetching dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin text-cyan-400 mx-auto mb-4" />
          <p className="text-gray-400">Loading RAMP Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="bg-gray-900 border border-orange-500 rounded-lg p-6 max-w-md">
          <p className="text-orange-500 font-semibold">Error</p>
          <p className="text-gray-300">{error}</p>
          <button 
            onClick={fetchDashboardData}
            className="mt-4 px-4 py-2 bg-orange-500 text-black rounded hover:bg-orange-600 font-medium"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🎯 RAMP Dashboard
          </h1>
          <p className="text-gray-400">
            Risk-Aware Market Predictor - Real-time volatility analysis
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 hover:border-cyan-500 transition-colors">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm font-medium">Total Stocks</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {stats?.total_stocks || 0}
                </p>
              </div>
              <Database className="w-12 h-12 text-cyan-400" />
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 hover:border-purple-500 transition-colors">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm font-medium">Industries</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {stats?.total_industries || 0}
                </p>
              </div>
              <Activity className="w-12 h-12 text-purple-500" />
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 hover:border-cyan-400 transition-colors">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm font-medium">Avg RAMP Score</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {stats?.average_ramp_score?.toFixed(1) || '0.0'}
                </p>
              </div>
              <TrendingUp className="w-12 h-12 text-cyan-400" />
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 hover:border-orange-500 transition-colors">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm font-medium">Avg Volatility</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {stats?.average_volatility?.toFixed(2)}%
                </p>
              </div>
              <TrendingDown className="w-12 h-12 text-orange-500" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">
              Risk Distribution
            </h2>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">
                    🔴 High Risk (≥70)
                  </span>
                  <span className="text-sm font-bold text-white">
                    {stats?.risk_distribution?.high || 0} stocks
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div 
                    className="bg-orange-500 h-2 rounded-full"
                    style={{ 
                      width: `${(stats?.risk_distribution?.high / stats?.total_stocks * 100) || 0}%` 
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">
                    🟡 Moderate Risk (30-70)
                  </span>
                  <span className="text-sm font-bold text-white">
                    {stats?.risk_distribution?.moderate || 0} stocks
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div 
                    className="bg-yellow-500 h-2 rounded-full"
                    style={{ 
                      width: `${(stats?.risk_distribution?.moderate / stats?.total_stocks * 100) || 0}%` 
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">
                    🟢 Low Risk (&lt;30)
                  </span>
                  <span className="text-sm font-bold text-white">
                    {stats?.risk_distribution?.low || 0} stocks
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div 
                    className="bg-cyan-400 h-2 rounded-full"
                    style={{ 
                      width: `${(stats?.risk_distribution?.low / stats?.total_stocks * 100) || 0}%` 
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">
              Model Accuracy
            </h2>
            <div className="space-y-4">
              <div className="bg-cyan-500 bg-opacity-10 border border-cyan-500 rounded-lg p-4">
                <p className="text-sm text-cyan-400 mb-1">Overall Accuracy</p>
                <p className="text-4xl font-bold text-cyan-400">
                  {modelAccuracy?.overall_accuracy}%
                </p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-black rounded-lg p-3 border border-gray-800">
                  <p className="text-xs text-gray-400 mb-1">High Volatility</p>
                  <p className="text-2xl font-bold text-white">
                    {modelAccuracy?.high_volatility_accuracy}%
                  </p>
                </div>
                <div className="bg-black rounded-lg p-3 border border-gray-800">
                  <p className="text-xs text-gray-400 mb-1">Low Volatility</p>
                  <p className="text-2xl font-bold text-white">
                    {modelAccuracy?.low_volatility_accuracy}%
                  </p>
                </div>
              </div>

              <p className="text-xs text-gray-400 mt-2">
                {modelAccuracy?.description}
              </p>
              <p className="text-xs text-gray-400">
                Sample size: {modelAccuracy?.sample_size?.toLocaleString()}+ predictions
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-cyan-500 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-2">
            About RAMP Scores
          </h3>
          <p className="text-gray-300 text-sm">
            RAMP (Risk-Aware Market Predictor) scores range from 0-100, where higher scores 
            indicate higher predicted volatility. Scores are calculated using machine learning 
            analysis of historical price movements, volume patterns, and technical indicators.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
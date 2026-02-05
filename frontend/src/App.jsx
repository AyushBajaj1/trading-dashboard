import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Zap, AlertCircle } from 'lucide-react';

const TradingDashboard = () => {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState('sma_crossover');
  const [capital, setCapital] = useState(10000);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const API_URL = 'http://localhost:5000/api';

  // Fetch available strategies on mount
  useEffect(() => {
    fetchStrategies();
  }, []);

  const fetchStrategies = async () => {
    try {
      const response = await fetch(`${API_URL}/strategies`);
      const data = await response.json();
      setStrategies(data);
    } catch (err) {
      console.error('Failed to fetch strategies:', err);
      setError('Failed to connect to API. Make sure Flask backend is running on port 5000.');
    }
  };

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy: selectedStrategy,
          capital: capital,
          days: 500
        })
      });

      if (!response.ok) {
        throw new Error('Backtest failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to run backtest. Check console for details.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const MetricCard = ({ title, value, icon: Icon, trend }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border border-slate-200">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-slate-600">{title}</span>
        <Icon className={trend > 0 ? 'text-green-500' : trend < 0 ? 'text-red-500' : 'text-blue-500'} size={20} />
      </div>
      <p className={`text-3xl font-bold ${
        trend > 0 ? 'text-green-600' : trend < 0 ? 'text-red-600' : 'text-slate-800'
      }`}>
        {value}
      </p>
    </div>
  );

  const StrategyCard = ({ strategy, isSelected, onClick }) => (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
        isSelected 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-slate-200 bg-white hover:border-blue-300'
      }`}
    >
      <h3 className="font-semibold text-lg text-slate-800">{strategy.name}</h3>
      <p className="text-sm text-slate-600 mt-1">{strategy.description}</p>
      <span className={`inline-block mt-2 px-2 py-1 rounded text-xs font-medium ${
        strategy.type === 'Machine Learning' 
          ? 'bg-purple-100 text-purple-700' 
          : 'bg-blue-100 text-blue-700'
      }`}>
        {strategy.type}
      </span>
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-slate-800 flex items-center gap-3">
            <Activity className="text-blue-500" size={40} />
            Algorithmic Trading Platform
          </h1>
          <p className="text-slate-600 mt-2">
            Professional React + Flask trading system with ML strategies
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
            <AlertCircle className="text-red-500 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <p className="text-red-800 font-medium">Connection Error</p>
              <p className="text-red-600 text-sm mt-1">{error}</p>
              <p className="text-red-600 text-sm mt-2">
                Make sure to run: <code className="bg-red-100 px-2 py-1 rounded">python backend/app.py</code>
              </p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sidebar - Strategy Selection */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6 border border-slate-200">
              <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <Zap className="text-blue-500" size={24} />
                Select Strategy
              </h2>
              
              <div className="space-y-3 mb-6">
                {strategies.map((strategy) => (
                  <StrategyCard
                    key={strategy.id}
                    strategy={strategy}
                    isSelected={selectedStrategy === strategy.id}
                    onClick={() => setSelectedStrategy(strategy.id)}
                  />
                ))}
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Initial Capital
                </label>
                <div className="relative">
                  <span className="absolute left-3 top-2.5 text-slate-500">$</span>
                  <input
                    type="number"
                    value={capital}
                    onChange={(e) => setCapital(Number(e.target.value))}
                    className="w-full pl-7 pr-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    min="1000"
                    step="1000"
                  />
                </div>
              </div>

              <button
                onClick={runBacktest}
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Running Backtest...
                  </>
                ) : (
                  <>
                    <BarChart3 size={20} />
                    Run Backtest
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Main Content - Results */}
          <div className="lg:col-span-2">
            {results ? (
              <>
                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <MetricCard
                    title="Total Return"
                    value={`${results.metrics.total_return}%`}
                    icon={results.metrics.total_return >= 0 ? TrendingUp : TrendingDown}
                    trend={results.metrics.total_return >= 0 ? 1 : -1}
                  />
                  <MetricCard
                    title="Final Value"
                    value={`$${results.metrics.final_value.toLocaleString()}`}
                    icon={DollarSign}
                    trend={0}
                  />
                  <MetricCard
                    title="Sharpe Ratio"
                    value={results.metrics.sharpe_ratio}
                    icon={Activity}
                    trend={results.metrics.sharpe_ratio > 1 ? 1 : -1}
                  />
                  <MetricCard
                    title="Win Rate"
                    value={`${results.metrics.win_rate}%`}
                    icon={BarChart3}
                    trend={results.metrics.win_rate > 50 ? 1 : -1}
                  />
                </div>

                {results.accuracy && (
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-6">
                    <p className="text-purple-800 font-medium">
                      🤖 ML Model Accuracy: <span className="text-2xl">{results.accuracy}%</span>
                    </p>
                  </div>
                )}

                {/* Equity Curve Chart */}
                <div className="bg-white rounded-lg shadow-md p-6 border border-slate-200 mb-6">
                  <h2 className="text-xl font-bold text-slate-800 mb-4">Portfolio Performance</h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={results.equity_curve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis 
                        dataKey="date" 
                        stroke="#64748b"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => {
                          const date = new Date(value);
                          return `${date.getMonth() + 1}/${date.getDate()}`;
                        }}
                      />
                      <YAxis 
                        stroke="#64748b"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1e293b', 
                          border: 'none',
                          borderRadius: '8px',
                          color: 'white'
                        }}
                        formatter={(value) => [`$${value.toFixed(2)}`, 'Portfolio Value']}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#3b82f6" 
                        strokeWidth={3}
                        name="Portfolio Value"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Trades Table */}
                <div className="bg-white rounded-lg shadow-md p-6 border border-slate-200">
                  <h2 className="text-xl font-bold text-slate-800 mb-4">Recent Trades</h2>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b-2 border-slate-200">
                          <th className="text-left py-3 px-4 text-sm font-semibold text-slate-600">Date</th>
                          <th className="text-left py-3 px-4 text-sm font-semibold text-slate-600">Type</th>
                          <th className="text-right py-3 px-4 text-sm font-semibold text-slate-600">Price</th>
                          <th className="text-right py-3 px-4 text-sm font-semibold text-slate-600">Shares</th>
                          <th className="text-right py-3 px-4 text-sm font-semibold text-slate-600">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.trades.reverse().map((trade, idx) => (
                          <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                            <td className="py-3 px-4 text-sm text-slate-700">{trade.date}</td>
                            <td className="py-3 px-4">
                              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                                trade.type === 'BUY' 
                                  ? 'bg-green-100 text-green-700' 
                                  : 'bg-red-100 text-red-700'
                              }`}>
                                {trade.type}
                              </span>
                            </td>
                            <td className="text-right py-3 px-4 text-sm text-slate-700">
                              ${trade.price.toFixed(2)}
                            </td>
                            <td className="text-right py-3 px-4 text-sm text-slate-700">
                              {trade.shares}
                            </td>
                            <td className={`text-right py-3 px-4 text-sm font-semibold ${
                              trade.profit > 0 
                                ? 'text-green-600' 
                                : trade.profit < 0 
                                ? 'text-red-600' 
                                : 'text-slate-600'
                            }`}>
                              {trade.profit ? `$${trade.profit.toFixed(2)}` : '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white rounded-lg shadow-md p-12 text-center border border-slate-200">
                <Activity className="mx-auto mb-4 text-slate-300" size={64} />
                <h3 className="text-xl font-semibold text-slate-600 mb-2">
                  Ready to Backtest
                </h3>
                <p className="text-slate-500">
                  Select a strategy and click "Run Backtest" to see results
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;


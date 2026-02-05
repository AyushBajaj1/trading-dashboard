# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ==================== DATA GENERATION ====================
class MarketDataGenerator:
    @staticmethod
    def generate_ohlcv(days=500, start_price=100):
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.normal(0.0005, 0.02, days)
        prices = start_price * np.exp(np.cumsum(returns))
        
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            open_price = close * (1 + np.random.uniform(-0.01, 0.01))
            high = max(open_price, close) * (1 + abs(np.random.uniform(0, 0.02)))
            low = min(open_price, close) * (1 - abs(np.random.uniform(0, 0.02)))
            volume = int(np.random.uniform(1e6, 1e7))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        return pd.DataFrame(data)

# ==================== TECHNICAL INDICATORS ====================
class TechnicalIndicators:
    @staticmethod
    def sma(data, period):
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        return data['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data, period=14):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

# ==================== TRADING STRATEGIES ====================
class TradingStrategy:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
    
    def calculate_metrics(self):
        if not self.equity_curve:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
        
        # Total return
        final_value = self.equity_curve[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Sharpe ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if len(returns) > 0 and returns.std() != 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Max drawdown
        equity = pd.Series(self.equity_curve)
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_dd = drawdowns.min() * 100
        
        # Win rate
        completed_trades = [t for t in self.trades if 'profit' in t]
        winning_trades = [t for t in completed_trades if t.get('profit', 0) > 0]
        win_rate = (len(winning_trades) / len(completed_trades) * 100) if completed_trades else 0
        
        return {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'win_rate': round(win_rate, 2),
            'final_value': round(final_value, 2),
            'total_trades': len(completed_trades)
        }

class SMACrossoverStrategy(TradingStrategy):
    def backtest(self, data):
        data['sma_short'] = TechnicalIndicators.sma(data, 20)
        data['sma_long'] = TechnicalIndicators.sma(data, 50)
        
        for i in range(50, len(data)):
            price = data.iloc[i]['close']
            date = data.iloc[i]['date']
            
            if (data.iloc[i]['sma_short'] > data.iloc[i]['sma_long'] and 
                data.iloc[i-1]['sma_short'] <= data.iloc[i-1]['sma_long'] and 
                self.position == 0):
                
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                
                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'value': cost
                })
            
            elif (data.iloc[i]['sma_short'] < data.iloc[i]['sma_long'] and 
                  data.iloc[i-1]['sma_short'] >= data.iloc[i-1]['sma_long'] and 
                  self.position > 0):
                
                revenue = self.position * price
                profit = revenue - self.trades[-1]['value']
                self.capital += revenue
                
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': self.position,
                    'value': revenue,
                    'profit': round(profit, 2)
                })
                
                self.position = 0
            
            portfolio_value = self.capital + (self.position * price)
            self.equity_curve.append(portfolio_value)
        
        return self

class RSIStrategy(TradingStrategy):
    def backtest(self, data):
        data['rsi'] = TechnicalIndicators.rsi(data, 14)
        
        for i in range(15, len(data)):
            price = data.iloc[i]['close']
            date = data.iloc[i]['date']
            rsi_value = data.iloc[i]['rsi']
            
            if rsi_value < 30 and self.position == 0:
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                
                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'value': cost,
                    'indicator': f'RSI: {round(rsi_value, 2)}'
                })
            
            elif rsi_value > 70 and self.position > 0:
                revenue = self.position * price
                profit = revenue - self.trades[-1]['value']
                self.capital += revenue
                
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': self.position,
                    'value': revenue,
                    'profit': round(profit, 2),
                    'indicator': f'RSI: {round(rsi_value, 2)}'
                })
                
                self.position = 0
            
            portfolio_value = self.capital + (self.position * price)
            self.equity_curve.append(portfolio_value)
        
        return self

class MeanReversionStrategy(TradingStrategy):
    def backtest(self, data):
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data, 20, 2)
        data['bb_upper'] = upper
        data['bb_middle'] = middle
        data['bb_lower'] = lower
        
        for i in range(20, len(data)):
            price = data.iloc[i]['close']
            date = data.iloc[i]['date']
            
            if price <= data.iloc[i]['bb_lower'] and self.position == 0:
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                
                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'value': cost
                })
            
            elif price >= data.iloc[i]['bb_middle'] and self.position > 0:
                revenue = self.position * price
                profit = revenue - self.trades[-1]['value']
                self.capital += revenue
                
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': self.position,
                    'value': revenue,
                    'profit': round(profit, 2)
                })
                
                self.position = 0
            
            portfolio_value = self.capital + (self.position * price)
            self.equity_curve.append(portfolio_value)
        
        return self

class MLStrategy(TradingStrategy):
    def prepare_features(self, data):
        data['sma_5'] = TechnicalIndicators.sma(data, 5)
        data['sma_20'] = TechnicalIndicators.sma(data, 20)
        data['sma_50'] = TechnicalIndicators.sma(data, 50)
        data['rsi'] = TechnicalIndicators.rsi(data, 14)
        data['returns_1d'] = data['close'].pct_change(1)
        data['returns_5d'] = data['close'].pct_change(5)
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        return data.dropna()
    
    def backtest(self, data):
        data = self.prepare_features(data.copy())
        
        feature_cols = ['sma_5', 'sma_20', 'sma_50', 'rsi', 'returns_1d', 'returns_5d', 'volume_ratio']
        X = data[feature_cols]
        y = data['target']
        
        split_idx = int(len(data) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        test_data = data.iloc[split_idx:].reset_index(drop=True)
        
        for i in range(len(predictions)):
            price = test_data.iloc[i]['close']
            date = test_data.iloc[i]['date']
            prediction = predictions[i]
            
            if prediction == 1 and self.position == 0:
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                
                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'value': cost
                })
            
            elif prediction == 0 and self.position > 0:
                revenue = self.position * price
                profit = revenue - self.trades[-1]['value']
                self.capital += revenue
                
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': self.position,
                    'value': revenue,
                    'profit': round(profit, 2)
                })
                
                self.position = 0
            
            portfolio_value = self.capital + (self.position * price)
            self.equity_curve.append(portfolio_value)
        
        self.accuracy = round(model.score(X_test_scaled, y_test) * 100, 2)
        return self

# ==================== API ROUTES ====================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Trading API is running'})

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    strategies = [
        {
            'id': 'sma_crossover',
            'name': 'SMA Crossover',
            'description': 'Buys when 20-day SMA crosses above 50-day SMA',
            'type': 'Technical'
        },
        {
            'id': 'rsi',
            'name': 'RSI Strategy',
            'description': 'Mean reversion using RSI oversold/overbought levels',
            'type': 'Technical'
        },
        {
            'id': 'mean_reversion',
            'name': 'Bollinger Bands',
            'description': 'Buys at lower band, sells at middle band',
            'type': 'Technical'
        },
        {
            'id': 'ml_random_forest',
            'name': 'Random Forest ML',
            'description': 'Machine learning classifier predicting price direction',
            'type': 'Machine Learning'
        }
    ]
    return jsonify(strategies)

@app.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        params = request.json
        strategy_type = params.get('strategy', 'sma_crossover')
        capital = params.get('capital', 10000)
        days = params.get('days', 500)
        
        # Generate market data
        data = MarketDataGenerator.generate_ohlcv(days=days)
        
        # Run selected strategy
        if strategy_type == 'sma_crossover':
            strategy = SMACrossoverStrategy(capital)
            strategy_name = "SMA Crossover (20/50)"
        elif strategy_type == 'rsi':
            strategy = RSIStrategy(capital)
            strategy_name = "RSI Strategy"
        elif strategy_type == 'mean_reversion':
            strategy = MeanReversionStrategy(capital)
            strategy_name = "Bollinger Bands Mean Reversion"
        elif strategy_type == 'ml_random_forest':
            strategy = MLStrategy(capital)
            strategy_name = "Random Forest ML"
        else:
            return jsonify({'error': 'Invalid strategy'}), 400
        
        strategy.backtest(data)
        metrics = strategy.calculate_metrics()
        
        # Prepare equity curve for charting
        equity_data = []
        start_date = pd.to_datetime(data.iloc[0]['date'])
        for i, value in enumerate(strategy.equity_curve):
            equity_data.append({
                'date': (start_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'value': round(value, 2)
            })
        
        result = {
            'strategy_name': strategy_name,
            'metrics': metrics,
            'trades': strategy.trades[-20:],  # Last 20 trades
            'equity_curve': equity_data,
            'accuracy': getattr(strategy, 'accuracy', None)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Flask Trading API Starting...")
    print("=" * 60)
    print("📡 API running on: http://localhost:5000")
    print("🔗 CORS enabled for React frontend")
    print("=" * 60)
    app.run(debug=True, host='127.0.0.1', port=5000)
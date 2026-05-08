from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Popular tickers with cosmetic metadata ────────────────────────────────────
TICKER_META = {
    "AAPL":  {"name": "Apple Inc.",             "sector": "Technology",    "start_price": 150},
    "MSFT":  {"name": "Microsoft Corp.",         "sector": "Technology",    "start_price": 320},
    "GOOGL": {"name": "Alphabet Inc.",           "sector": "Technology",    "start_price": 140},
    "AMZN":  {"name": "Amazon.com Inc.",         "sector": "Consumer",      "start_price": 185},
    "NVDA":  {"name": "NVIDIA Corp.",            "sector": "Technology",    "start_price": 450},
    "META":  {"name": "Meta Platforms Inc.",     "sector": "Technology",    "start_price": 510},
    "TSLA":  {"name": "Tesla Inc.",              "sector": "Automotive",    "start_price": 250},
    "BRK.B": {"name": "Berkshire Hathaway B",   "sector": "Financials",    "start_price": 410},
    "JPM":   {"name": "JPMorgan Chase & Co.",    "sector": "Financials",    "start_price": 200},
    "V":     {"name": "Visa Inc.",               "sector": "Financials",    "start_price": 275},
    "JNJ":   {"name": "Johnson & Johnson",       "sector": "Healthcare",    "start_price": 155},
    "SPY":   {"name": "S&P 500 ETF",             "sector": "ETF",           "start_price": 480},
    "QQQ":   {"name": "Nasdaq-100 ETF",          "sector": "ETF",           "start_price": 420},
    "BTC":   {"name": "Bitcoin (synthetic)",     "sector": "Crypto",        "start_price": 40000},
}

# ── Data generation ───────────────────────────────────────────────────────────
class MarketDataGenerator:
    @staticmethod
    def generate_ohlcv(days: int, ticker: str = "AAPL"):
        meta = TICKER_META.get(ticker.upper(), {"start_price": 100})
        start_price = meta["start_price"]

        # Deterministic seed per ticker so results are reproducible
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(ticker.upper())) % (2**31)
        rng = np.random.default_rng(seed)

        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')  # business days
        # Slightly higher drift for "growth" tickers
        drift = 0.0006 if ticker.upper() in ("NVDA", "TSLA", "META", "BTC") else 0.0003
        vol = 0.03 if ticker.upper() in ("TSLA", "BTC") else 0.015
        returns = rng.normal(drift, vol, days)
        prices = start_price * np.exp(np.cumsum(returns))

        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            open_price = close * (1 + rng.uniform(-0.008, 0.008))
            high = max(open_price, close) * (1 + abs(rng.uniform(0, 0.015)))
            low  = min(open_price, close) * (1 - abs(rng.uniform(0, 0.015)))
            volume = int(rng.uniform(5e6, 5e7))
            data.append({
                'date':   date.strftime('%Y-%m-%d'),
                'open':   round(open_price, 2),
                'high':   round(high, 2),
                'low':    round(low, 2),
                'close':  round(close, 2),
                'volume': volume,
            })
        return pd.DataFrame(data)


# ── Technical indicators ──────────────────────────────────────────────────────
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
        return sma + (std * std_dev), sma, sma - (std * std_dev)

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line


# ── Base strategy ─────────────────────────────────────────────────────────────
class TradingStrategy:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []

    def calculate_metrics(self):
        if not self.equity_curve:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}

        final_value = self.equity_curve[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = (np.sqrt(252) * returns.mean() / returns.std()
                  if len(returns) > 0 and returns.std() != 0 else 0)

        equity = pd.Series(self.equity_curve)
        rolling_max = equity.expanding().max()
        max_dd = ((equity - rolling_max) / rolling_max).min() * 100

        completed = [t for t in self.trades if 'profit' in t]
        winning   = [t for t in completed if t.get('profit', 0) > 0]
        win_rate  = (len(winning) / len(completed) * 100) if completed else 0

        return {
            'total_return':  round(total_return, 2),
            'sharpe_ratio':  round(sharpe, 2),
            'max_drawdown':  round(max_dd, 2),
            'win_rate':      round(win_rate, 2),
            'final_value':   round(final_value, 2),
            'total_trades':  len(completed),
        }


# ── Strategies ────────────────────────────────────────────────────────────────
class SMACrossoverStrategy(TradingStrategy):
    def backtest(self, data):
        data = data.copy()
        data['sma_short'] = TechnicalIndicators.sma(data, 20)
        data['sma_long']  = TechnicalIndicators.sma(data, 50)
        for i in range(50, len(data)):
            price = data.iloc[i]['close']
            date  = data.iloc[i]['date']
            if (data.iloc[i]['sma_short'] > data.iloc[i]['sma_long'] and
                    data.iloc[i-1]['sma_short'] <= data.iloc[i-1]['sma_long'] and
                    self.position == 0):
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                self.trades.append({'date': date, 'type': 'BUY', 'price': price, 'shares': shares, 'value': cost})
            elif (data.iloc[i]['sma_short'] < data.iloc[i]['sma_long'] and
                  data.iloc[i-1]['sma_short'] >= data.iloc[i-1]['sma_long'] and
                  self.position > 0):
                revenue = self.position * price
                profit  = revenue - self.trades[-1]['value']
                self.capital += revenue
                self.trades.append({'date': date, 'type': 'SELL', 'price': price, 'shares': self.position,
                                    'value': revenue, 'profit': round(profit, 2)})
                self.position = 0
            self.equity_curve.append(self.capital + self.position * price)
        return self


class RSIStrategy(TradingStrategy):
    def backtest(self, data):
        data = data.copy()
        data['rsi'] = TechnicalIndicators.rsi(data, 14)
        for i in range(15, len(data)):
            price = data.iloc[i]['close']
            date  = data.iloc[i]['date']
            rsi_v = data.iloc[i]['rsi']
            if rsi_v < 30 and self.position == 0:
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                self.trades.append({'date': date, 'type': 'BUY', 'price': price, 'shares': shares,
                                    'value': cost, 'indicator': f'RSI {round(rsi_v, 1)}'})
            elif rsi_v > 70 and self.position > 0:
                revenue = self.position * price
                profit  = revenue - self.trades[-1]['value']
                self.capital += revenue
                self.trades.append({'date': date, 'type': 'SELL', 'price': price, 'shares': self.position,
                                    'value': revenue, 'profit': round(profit, 2),
                                    'indicator': f'RSI {round(rsi_v, 1)}'})
                self.position = 0
            self.equity_curve.append(self.capital + self.position * price)
        return self


class MeanReversionStrategy(TradingStrategy):
    def backtest(self, data):
        data = data.copy()
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data, 20, 2)
        data['bb_upper']  = upper
        data['bb_middle'] = middle
        data['bb_lower']  = lower
        for i in range(20, len(data)):
            price = data.iloc[i]['close']
            date  = data.iloc[i]['date']
            if price <= data.iloc[i]['bb_lower'] and self.position == 0:
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                self.trades.append({'date': date, 'type': 'BUY', 'price': price, 'shares': shares,
                                    'value': cost, 'indicator': 'BB Lower'})
            elif price >= data.iloc[i]['bb_middle'] and self.position > 0:
                revenue = self.position * price
                profit  = revenue - self.trades[-1]['value']
                self.capital += revenue
                self.trades.append({'date': date, 'type': 'SELL', 'price': price, 'shares': self.position,
                                    'value': revenue, 'profit': round(profit, 2),
                                    'indicator': 'BB Mid'})
                self.position = 0
            self.equity_curve.append(self.capital + self.position * price)
        return self


class MLStrategy(TradingStrategy):
    def prepare_features(self, data):
        data['sma_5']        = TechnicalIndicators.sma(data, 5)
        data['sma_20']       = TechnicalIndicators.sma(data, 20)
        data['sma_50']       = TechnicalIndicators.sma(data, 50)
        data['rsi']          = TechnicalIndicators.rsi(data, 14)
        data['returns_1d']   = data['close'].pct_change(1)
        data['returns_5d']   = data['close'].pct_change(5)
        data['volume_sma']   = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['target']       = (data['close'].shift(-1) > data['close']).astype(int)
        return data.dropna()

    def backtest(self, data):
        data = self.prepare_features(data.copy())
        feature_cols = ['sma_5', 'sma_20', 'sma_50', 'rsi', 'returns_1d', 'returns_5d', 'volume_ratio']
        X = data[feature_cols]; y = data['target']
        split_idx = int(len(data) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_s, y_train)
        predictions = model.predict(X_test_s)
        test_data   = data.iloc[split_idx:].reset_index(drop=True)
        for i, pred in enumerate(predictions):
            price = test_data.iloc[i]['close']
            date  = test_data.iloc[i]['date']
            if pred == 1 and self.position == 0:
                shares = int(self.capital / price)
                cost = shares * price
                self.capital -= cost
                self.position = shares
                self.trades.append({'date': date, 'type': 'BUY', 'price': price, 'shares': shares,
                                    'value': cost, 'indicator': 'ML ↑'})
            elif pred == 0 and self.position > 0:
                revenue = self.position * price
                profit  = revenue - self.trades[-1]['value']
                self.capital += revenue
                self.trades.append({'date': date, 'type': 'SELL', 'price': price, 'shares': self.position,
                                    'value': revenue, 'profit': round(profit, 2), 'indicator': 'ML ↓'})
                self.position = 0
            self.equity_curve.append(self.capital + self.position * price)
        self.accuracy = round(model.score(X_test_s, y_test) * 100, 2)
        return self


# ── Schemas ───────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    strategy: str   = 'sma_crossover'
    ticker:   str   = 'AAPL'
    capital:  float = 10000
    days:     int   = 500


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/tickers")
def get_tickers():
    return [
        {"symbol": k, "name": v["name"], "sector": v["sector"]}
        for k, v in TICKER_META.items()
    ]

@app.get("/api/strategies")
def get_strategies():
    return [
        {"id": "sma_crossover",   "name": "SMA Crossover",   "type": "Trend",      "description": "20/50-day moving average crossover"},
        {"id": "rsi",             "name": "RSI Mean Rev.",    "type": "Mean Rev.",  "description": "Buy oversold (RSI<30), sell overbought (RSI>70)"},
        {"id": "mean_reversion",  "name": "Bollinger Bands", "type": "Mean Rev.",  "description": "Buy at lower band, sell at middle band"},
        {"id": "ml_random_forest","name": "Random Forest",   "type": "ML",         "description": "Random Forest classifier on price/volume features"},
    ]

@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    ticker = req.ticker.upper()
    data   = MarketDataGenerator.generate_ohlcv(days=req.days, ticker=ticker)

    strategy_map = {
        'sma_crossover':    (SMACrossoverStrategy,  "SMA Crossover (20/50)"),
        'rsi':              (RSIStrategy,           "RSI Strategy"),
        'mean_reversion':   (MeanReversionStrategy, "Bollinger Bands"),
        'ml_random_forest': (MLStrategy,            "Random Forest ML"),
    }

    if req.strategy not in strategy_map:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    StrategyClass, strategy_name = strategy_map[req.strategy]
    strategy = StrategyClass(req.capital)
    strategy.backtest(data)
    metrics = strategy.calculate_metrics()

    # Price chart data — subsample to keep payload small
    step = max(1, len(data) // 300)
    price_data = [
        {'date': row['date'], 'price': row['close'],
         'high': row['high'], 'low': row['low'], 'volume': row['volume']}
        for row in data.iloc[::step].to_dict('records')
    ]

    # Build equity curve aligned to price data dates
    start_date = pd.to_datetime(data.iloc[0]['date'])
    equity_data = [
        {'date': (start_date + timedelta(days=i)).strftime('%Y-%m-%d'), 'value': round(v, 2)}
        for i, v in enumerate(strategy.equity_curve)
    ]

    # Trade signal set for chart overlay (date → type)
    signal_map = {t['date']: t['type'] for t in strategy.trades}

    meta = TICKER_META.get(ticker, {"name": ticker, "sector": "—"})

    return {
        'ticker':        ticker,
        'ticker_name':   meta['name'],
        'strategy_name': strategy_name,
        'metrics':       metrics,
        'trades':        strategy.trades,
        'equity_curve':  equity_data,
        'price_data':    price_data,
        'signal_map':    signal_map,
        'accuracy':      getattr(strategy, 'accuracy', None),
    }

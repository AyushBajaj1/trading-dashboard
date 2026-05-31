from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
import time
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

# ── Ticker registry ───────────────────────────────────────────────────────────
TICKER_META = {
    "AAPL":  {"name": "Apple Inc.",            "sector": "Technology",  "yf": "AAPL"},
    "MSFT":  {"name": "Microsoft Corp.",        "sector": "Technology",  "yf": "MSFT"},
    "GOOGL": {"name": "Alphabet Inc.",          "sector": "Technology",  "yf": "GOOGL"},
    "AMZN":  {"name": "Amazon.com Inc.",        "sector": "Consumer",    "yf": "AMZN"},
    "NVDA":  {"name": "NVIDIA Corp.",           "sector": "Technology",  "yf": "NVDA"},
    "META":  {"name": "Meta Platforms Inc.",    "sector": "Technology",  "yf": "META"},
    "TSLA":  {"name": "Tesla Inc.",             "sector": "Automotive",  "yf": "TSLA"},
    "JPM":   {"name": "JPMorgan Chase & Co.",   "sector": "Financials",  "yf": "JPM"},
    "V":     {"name": "Visa Inc.",              "sector": "Financials",  "yf": "V"},
    "JNJ":   {"name": "Johnson & Johnson",      "sector": "Healthcare",  "yf": "JNJ"},
    "SPY":   {"name": "S&P 500 ETF",            "sector": "ETF",         "yf": "SPY"},
    "QQQ":   {"name": "Nasdaq-100 ETF",         "sector": "ETF",         "yf": "QQQ"},
    "BTC":   {"name": "Bitcoin",                "sector": "Crypto",      "yf": "BTC-USD"},
    "ETH":   {"name": "Ethereum",               "sector": "Crypto",      "yf": "ETH-USD"},
}

# ── Data fetching with in-memory cache ────────────────────────────────────────
_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 3600  # 1 hour — prices don't change during a session

def fetch_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    meta = TICKER_META.get(ticker.upper())
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")

    yf_symbol = meta["yf"]
    cache_key  = (yf_symbol, days)
    now        = time.time()

    if cache_key in _cache:
        ts, df = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return df.copy()

    # Fetch extra calendar days to account for weekends + holidays
    end   = datetime.now()
    start = end - timedelta(days=int(days * 1.6))

    raw = yf.download(yf_symbol, start=start, end=end,
                      progress=False, auto_adjust=True)

    if raw.empty:
        raise HTTPException(status_code=502,
                            detail=f"Yahoo Finance returned no data for {yf_symbol}")

    # yf.download returns a MultiIndex (Price, Ticker) — drop the ticker level
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw.reset_index().rename(columns=str.lower)
    df = df.rename(columns={"date": "date"})   # already lowercase
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
    df = df.round({"open": 2, "high": 2, "low": 2, "close": 2})

    # Trim to requested number of trading days
    df = df.tail(days).reset_index(drop=True)

    _cache[cache_key] = (now, df)
    return df.copy()


# ── Technical indicators ──────────────────────────────────────────────────────
class TechnicalIndicators:
    @staticmethod
    def sma(data, period):
        return data["close"].rolling(window=period).mean()

    @staticmethod
    def ema(data, period):
        return data["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data, period=14):
        delta = data["close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs    = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        sma = data["close"].rolling(window=period).mean()
        std = data["close"].rolling(window=period).std()
        return sma + std * std_dev, sma, sma - std * std_dev


# ── Base strategy ─────────────────────────────────────────────────────────────
class TradingStrategy:
    def __init__(self, initial_capital: float = 10_000):
        self.initial_capital = initial_capital
        self.capital   = initial_capital
        self.position  = 0
        self.trades    = []
        self.equity_curve = []
        self.equity_dates = []

    def calculate_metrics(self) -> dict:
        if not self.equity_curve:
            return {"total_return": 0, "sharpe_ratio": 0,
                    "max_drawdown": 0, "win_rate": 0}

        final  = self.equity_curve[-1]
        ret    = (final - self.initial_capital) / self.initial_capital * 100

        series = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = (np.sqrt(252) * series.mean() / series.std()
                  if len(series) > 1 and series.std() != 0 else 0)

        equity      = pd.Series(self.equity_curve)
        rolling_max = equity.expanding().max()
        max_dd      = ((equity - rolling_max) / rolling_max).min() * 100

        completed = [t for t in self.trades if "profit" in t]
        winning   = [t for t in completed if t["profit"] > 0]
        win_rate  = len(winning) / len(completed) * 100 if completed else 0

        return {
            "total_return": round(ret, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "win_rate":     round(win_rate, 2),
            "final_value":  round(final, 2),
            "total_trades": len(completed),
        }


# ── Strategies ────────────────────────────────────────────────────────────────
class SMACrossoverStrategy(TradingStrategy):
    def backtest(self, data: pd.DataFrame, short: int = 20, long: int = 50):
        data = data.copy()
        data["sma_short"] = TechnicalIndicators.sma(data, short)
        data["sma_long"]  = TechnicalIndicators.sma(data, long)

        for i in range(long, len(data)):
            price = data.iloc[i]["close"]
            date  = data.iloc[i]["date"]
            prev, curr = data.iloc[i - 1], data.iloc[i]

            if curr["sma_short"] > curr["sma_long"] and \
               prev["sma_short"] <= prev["sma_long"] and self.position == 0:
                shares = int(self.capital / price)
                cost   = shares * price
                self.capital  -= cost
                self.position  = shares
                self.trades.append({"date": date, "type": "BUY",
                                    "price": price, "shares": shares, "value": cost,
                                    "indicator": f"SMA{short}/SMA{long}"})

            elif curr["sma_short"] < curr["sma_long"] and \
                 prev["sma_short"] >= prev["sma_long"] and self.position > 0:
                revenue = self.position * price
                profit  = revenue - self.trades[-1]["value"]
                self.capital  += revenue
                self.trades.append({"date": date, "type": "SELL",
                                    "price": price, "shares": self.position,
                                    "value": revenue, "profit": round(profit, 2),
                                    "indicator": f"SMA{short}/SMA{long}"})
                self.position = 0

            self.equity_curve.append(self.capital + self.position * price)
            self.equity_dates.append(date)
        return self


class RSIStrategy(TradingStrategy):
    def backtest(self, data: pd.DataFrame,
                 period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        data = data.copy()
        data["rsi"] = TechnicalIndicators.rsi(data, period)

        for i in range(period + 1, len(data)):
            price = data.iloc[i]["close"]
            date  = data.iloc[i]["date"]
            rsi_v = data.iloc[i]["rsi"]

            if rsi_v < oversold and self.position == 0:
                shares = int(self.capital / price)
                cost   = shares * price
                self.capital  -= cost
                self.position  = shares
                self.trades.append({"date": date, "type": "BUY",
                                    "price": price, "shares": shares,
                                    "value": cost, "indicator": f"RSI {round(rsi_v, 1)}"})

            elif rsi_v > overbought and self.position > 0:
                revenue = self.position * price
                profit  = revenue - self.trades[-1]["value"]
                self.capital  += revenue
                self.trades.append({"date": date, "type": "SELL",
                                    "price": price, "shares": self.position,
                                    "value": revenue, "profit": round(profit, 2),
                                    "indicator": f"RSI {round(rsi_v, 1)}"})
                self.position = 0

            self.equity_curve.append(self.capital + self.position * price)
            self.equity_dates.append(date)
        return self


class MeanReversionStrategy(TradingStrategy):
    def backtest(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
        data = data.copy()
        upper, middle, lower = TechnicalIndicators.bollinger_bands(data, period, std_dev)
        data["bb_upper"]  = upper
        data["bb_middle"] = middle
        data["bb_lower"]  = lower

        for i in range(period, len(data)):
            price = data.iloc[i]["close"]
            date  = data.iloc[i]["date"]

            if price <= data.iloc[i]["bb_lower"] and self.position == 0:
                shares = int(self.capital / price)
                cost   = shares * price
                self.capital  -= cost
                self.position  = shares
                self.trades.append({"date": date, "type": "BUY",
                                    "price": price, "shares": shares,
                                    "value": cost, "indicator": f"BB({period},{std_dev})"})

            elif price >= data.iloc[i]["bb_middle"] and self.position > 0:
                revenue = self.position * price
                profit  = revenue - self.trades[-1]["value"]
                self.capital  += revenue
                self.trades.append({"date": date, "type": "SELL",
                                    "price": price, "shares": self.position,
                                    "value": revenue, "profit": round(profit, 2),
                                    "indicator": "BB Mid"})
                self.position = 0

            self.equity_curve.append(self.capital + self.position * price)
            self.equity_dates.append(date)
        return self


class MLStrategy(TradingStrategy):
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data["sma_5"]        = TechnicalIndicators.sma(data, 5)
        data["sma_20"]       = TechnicalIndicators.sma(data, 20)
        data["sma_50"]       = TechnicalIndicators.sma(data, 50)
        data["rsi"]          = TechnicalIndicators.rsi(data, 14)
        data["returns_1d"]   = data["close"].pct_change(1)
        data["returns_5d"]   = data["close"].pct_change(5)
        data["volume_sma"]   = data["volume"].rolling(window=20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma"]
        data["target"]       = (data["close"].shift(-1) > data["close"]).astype(int)
        return data.dropna()

    def backtest(self, data: pd.DataFrame,
                 n_estimators: int = 100, max_depth: int = 10, train_split: float = 70.0):
        data = self.prepare_features(data.copy())
        FEATURES = ["sma_5", "sma_20", "sma_50", "rsi",
                    "returns_1d", "returns_5d", "volume_ratio"]
        X, y     = data[FEATURES], data["target"]
        split    = int(len(data) * (train_split / 100))

        scaler       = StandardScaler()
        X_train_s    = scaler.fit_transform(X[:split])
        X_test_s     = scaler.transform(X[split:])

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train_s, y[:split])
        predictions = model.predict(X_test_s)
        test_data   = data.iloc[split:].reset_index(drop=True)

        for i, pred in enumerate(predictions):
            price = test_data.iloc[i]["close"]
            date  = test_data.iloc[i]["date"]

            if pred == 1 and self.position == 0:
                shares = int(self.capital / price)
                cost   = shares * price
                self.capital  -= cost
                self.position  = shares
                self.trades.append({"date": date, "type": "BUY",
                                    "price": price, "shares": shares,
                                    "value": cost, "indicator": "ML ↑"})

            elif pred == 0 and self.position > 0:
                revenue = self.position * price
                profit  = revenue - self.trades[-1]["value"]
                self.capital  += revenue
                self.trades.append({"date": date, "type": "SELL",
                                    "price": price, "shares": self.position,
                                    "value": revenue, "profit": round(profit, 2),
                                    "indicator": "ML ↓"})
                self.position = 0

            self.equity_curve.append(self.capital + self.position * price)
            self.equity_dates.append(date)

        self.accuracy = round(model.score(X_test_s, y[split:]) * 100, 2)
        return self


# ── Request schema ────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    strategy: str   = "sma_crossover"
    ticker:   str   = "AAPL"
    capital:  float = 10_000
    days:     int   = 500
    # SMA Crossover
    sma_short:      int   = 20
    sma_long:       int   = 50
    # RSI
    rsi_period:     int   = 14
    rsi_oversold:   float = 30.0
    rsi_overbought: float = 70.0
    # Bollinger Bands
    bb_period:      int   = 20
    bb_std:         float = 2.0
    # Random Forest
    n_estimators:   int   = 100
    max_depth:      int   = 10
    train_split:    float = 70.0   # percentage, e.g. 70 → 70%


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
        {"id": "sma_crossover",    "name": "SMA Crossover",   "type": "Trend",     "description": "20/50-day moving average crossover"},
        {"id": "rsi",              "name": "RSI Mean Rev.",    "type": "Mean Rev.", "description": "Buy oversold (RSI<30), sell overbought (RSI>70)"},
        {"id": "mean_reversion",   "name": "Bollinger Bands", "type": "Mean Rev.", "description": "Buy at lower band, sell at middle band"},
        {"id": "ml_random_forest", "name": "Random Forest",   "type": "ML",        "description": "Random Forest classifier on price/volume features"},
    ]


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    ticker = req.ticker.upper()
    data   = fetch_ohlcv(ticker, req.days)       # real data from Yahoo Finance

    strategy_map = {
        "sma_crossover":    (SMACrossoverStrategy,  "SMA Crossover (20/50)"),
        "rsi":              (RSIStrategy,           "RSI Strategy"),
        "mean_reversion":   (MeanReversionStrategy, "Bollinger Bands"),
        "ml_random_forest": (MLStrategy,            "Random Forest ML"),
    }

    if req.strategy not in strategy_map:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    StrategyClass, strategy_name = strategy_map[req.strategy]
    strategy = StrategyClass(req.capital)

    if req.strategy == "sma_crossover":
        strategy.backtest(data, short=req.sma_short, long=req.sma_long)
    elif req.strategy == "rsi":
        strategy.backtest(data, period=req.rsi_period,
                          oversold=req.rsi_oversold, overbought=req.rsi_overbought)
    elif req.strategy == "mean_reversion":
        strategy.backtest(data, period=req.bb_period, std_dev=req.bb_std)
    elif req.strategy == "ml_random_forest":
        strategy.backtest(data, n_estimators=req.n_estimators,
                          max_depth=req.max_depth, train_split=req.train_split)

    metrics = strategy.calculate_metrics()

    # Subsample price data to ~300 points so the payload stays small
    step       = max(1, len(data) // 300)
    price_data = [
        {"date": r["date"], "price": r["close"],
         "high": r["high"],  "low":  r["low"], "volume": r["volume"]}
        for r in data.iloc[::step].to_dict("records")
    ]

    equity_data = [
        {"date": d, "value": round(v, 2)}
        for d, v in zip(strategy.equity_dates, strategy.equity_curve)
    ]

    signal_map = {t["date"]: t["type"] for t in strategy.trades}
    meta       = TICKER_META[ticker]

    return {
        "ticker":        ticker,
        "ticker_name":   meta["name"],
        "strategy_name": strategy_name,
        "metrics":       metrics,
        "trades":        strategy.trades,
        "equity_curve":  equity_data,
        "price_data":    price_data,
        "signal_map":    signal_map,
        "accuracy":      getattr(strategy, "accuracy", None),
    }

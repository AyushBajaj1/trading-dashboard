# Algo Trading Backtest

A dark-themed backtesting dashboard. Pick a stock, choose a strategy, configure parameters, and see the results — equity curve, trade markers on the price chart, and a full trade log.

![screenshot](docs/screenshot.png)

## Stack

| Layer | Tech |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS v4, Recharts |
| Backend | FastAPI, pandas, NumPy, scikit-learn |

## Getting started

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

## Features

**Stocks** — 14 tickers (AAPL, NVDA, TSLA, BTC, SPY, …). Each ticker seeds a deterministic synthetic OHLCV series, so results are reproducible. Volatile tickers (TSLA, BTC, NVDA, META) get higher drift and volatility.

**Strategies**

| Strategy | Type | Signal |
|---|---|---|
| SMA Crossover | Trend | Buy when 20-day SMA crosses above 50-day SMA, sell on cross-below |
| RSI Mean Reversion | Mean Rev. | Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought) |
| Bollinger Bands | Mean Rev. | Buy at lower band, sell at middle band |
| Random Forest | ML | Random Forest classifier trained on price/volume features; predicts next-day direction |

**Metrics** — Total return, final portfolio value, Sharpe ratio, max drawdown, win rate, trade count. ML strategies also show test-set accuracy.

**Charts** — Price chart with ▲/▼ markers at exact buy/sell dates, plus a separate equity curve with a reference line at starting capital.

**Trade log** — Every trade with date, side, price, shares, total value, P&L, and indicator value.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/tickers` | List of supported tickers with name and sector |
| `GET` | `/api/strategies` | List of available strategies |
| `POST` | `/api/backtest` | Run a backtest (see below) |

**POST /api/backtest**

```json
{
  "ticker":   "AAPL",
  "strategy": "sma_crossover",
  "capital":  10000,
  "days":     500
}
```

`strategy` options: `sma_crossover`, `rsi`, `mean_reversion`, `ml_random_forest`

## Project structure

```
algo-trading/
├── backend/
│   ├── main.py          # FastAPI app — data gen, indicators, strategies, routes
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx      # Entire UI — sidebar, charts, stats, trade log
    │   └── index.css    # Tailwind v4 import + @theme custom tokens
    ├── vite.config.js
    └── package.json
```

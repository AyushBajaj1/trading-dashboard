# Trading Dashboard for Backtesting

A web app for backtesting algorithmic trading strategies against real historical stock and crypto price data.

## Stack
#### Frontend: React 19, Vite, Tailwind CSS, Recharts 
#### Backend: FastAPI, pandas, NumPy, scikit-learn, yfinance

## Installation

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
**Stocks** — 14 tickers across equities, ETFs, and crypto (AAPL, NVDA, TSLA, BTC, SPY, …). Historical data for these equities is then fetched from Yahoo Finance and cached for the session.

**Strategies**
| Strategy | Type | Signal |
|---|---|---|
| SMA Crossover | Trend | Buy when 20-day SMA crosses above 50-day SMA, sell on cross-below |
| RSI Mean Reversion | Mean Rev. | Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought) |
| Bollinger Bands | Mean Rev. | Buy at lower band, sell at middle band |
| Random Forest | ML | Random Forest classifier trained on price/volume features; predicts next-day direction |

**Hyperparameters** — Each strategy exposes tunable parameters (SMA windows, RSI thresholds, Bollinger std dev, RF estimators/depth/train split) configurable from the sidebar.

**Metrics** — Total return, final portfolio value, Sharpe ratio, max drawdown, win rate, trade count. ML strategy also shows test-set accuracy.

**Charts** — Price chart with markers at exact buy/sell dates, plus a separate equity curve with a reference line at starting capital.

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

Optional hyperparameter fields: `sma_short`, `sma_long`, `rsi_period`, `rsi_oversold`, `rsi_overbought`, `bb_period`, `bb_std`, `n_estimators`, `max_depth`, `train_split`

## Project structure

```
algo-trading/
├── backend/
│   ├── main.py         
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx      
    │   └── index.css    
    ├── vite.config.js
    └── package.json
```

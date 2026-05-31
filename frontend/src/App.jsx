import { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  Dot,
} from 'recharts';

const API = 'http://localhost:8000/api';

// ── Formatters ────────────────────────────────────────────────────────────────
const fmt$ = (v) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(v);
const fmtK = (v) => {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
};
const fmtPct = (v) => `${v > 0 ? '+' : ''}${v}%`;
const fmtDate = (d) => {
  const dt = new Date(d);
  return dt.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
};

// ── Tiny helpers ──────────────────────────────────────────────────────────────
const clx = (...args) => args.filter(Boolean).join(' ');

function Pill({ children, color = 'default' }) {
  const colors = {
    default: 'bg-surface3 text-ink2',
    trend:   'bg-blue/10 text-blue',
    rev:     'bg-amber/10 text-amber',
    ml:      'bg-violet/10 text-violet',
    green:   'bg-green/10 text-green',
    red:     'bg-red/10 text-red',
  };
  return (
    <span className={clx('inline-block px-1.5 py-0.5 rounded text-[10px] font-medium', colors[color])}>
      {children}
    </span>
  );
}

// ── Stock picker ──────────────────────────────────────────────────────────────
function TickerSearch({ tickers, selected, onSelect }) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const filtered = query
    ? tickers.filter(t =>
        t.symbol.toLowerCase().includes(query.toLowerCase()) ||
        t.name.toLowerCase().includes(query.toLowerCase())
      )
    : tickers;

  const current = tickers.find(t => t.symbol === selected);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 bg-surface2 border border-border2 rounded-lg px-3 py-2.5 text-left hover:border-ink3 transition-colors"
      >
        <span className="font-mono font-bold text-ink text-sm">{selected}</span>
        <span className="text-ink3 text-xs truncate flex-1">{current?.name}</span>
        <svg className={clx('w-3.5 h-3.5 text-ink3 flex-shrink-0 transition-transform', open && 'rotate-180')} viewBox="0 0 16 16" fill="none">
          <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {open && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-surface2 border border-border2 rounded-lg shadow-xl z-50 overflow-hidden">
          <div className="p-2 border-b border-border">
            <input
              autoFocus
              placeholder="Search ticker or name…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full bg-surface3 border border-border2 rounded-md px-2.5 py-1.5 text-xs text-ink placeholder-ink3 outline-none focus:border-ink3"
            />
          </div>
          <div className="max-h-52 overflow-y-auto">
            {filtered.map(t => (
              <button
                key={t.symbol}
                onClick={() => { onSelect(t.symbol); setOpen(false); setQuery(''); }}
                className={clx(
                  'w-full flex items-center gap-2 px-3 py-2 hover:bg-surface3 text-left transition-colors',
                  t.symbol === selected && 'bg-surface3'
                )}
              >
                <span className="font-mono font-bold text-ink text-xs w-12 flex-shrink-0">{t.symbol}</span>
                <span className="text-ink2 text-xs truncate flex-1">{t.name}</span>
                <span className="text-ink3 text-[10px] flex-shrink-0">{t.sector}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Strategy card ─────────────────────────────────────────────────────────────
function StrategyCard({ s, selected, onSelect }) {
  const typeColor = { Trend: 'trend', 'Mean Rev.': 'rev', ML: 'ml' };
  const active = selected === s.id;
  return (
    <button
      onClick={() => onSelect(s.id)}
      className={clx(
        'w-full text-left rounded-lg px-3 py-2 border transition-all flex items-center justify-between gap-2',
        active
          ? 'border-ink3 bg-surface2'
          : 'border-border bg-surface hover:border-border2 hover:bg-surface2'
      )}
    >
      <span className={clx('text-xs font-medium truncate', active ? 'text-ink' : 'text-ink2')}>{s.name}</span>
      <Pill color={typeColor[s.type] || 'default'}>{s.type}</Pill>
    </button>
  );
}

// ── Info tooltip ──────────────────────────────────────────────────────────────
function InfoTip({ text }) {
  return (
    <span className="relative group inline-flex items-center ml-1">
      <span className="w-3 h-3 rounded-full border border-ink3/50 text-ink3 text-[8px] flex items-center justify-center cursor-default select-none font-bold leading-none">?</span>
      <span className="pointer-events-none absolute top-full left-0 mt-1.5 z-50 hidden group-hover:block w-44 bg-surface2 border border-border2 rounded-lg px-2.5 py-2 text-[11px] text-ink2 shadow-xl leading-relaxed whitespace-normal">
        {text}
      </span>
    </span>
  );
}

// ── Number inputs ─────────────────────────────────────────────────────────────
function NumberInput({ label, value, onChange, prefix, min, max, step, tip }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-ink3 text-[11px] uppercase tracking-wide flex items-center">
        {label}
        {tip && <InfoTip text={tip} />}
      </label>
      <div className="relative">
        {prefix && (
          <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink3 text-xs pointer-events-none">
            {prefix}
          </span>
        )}
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min} max={max} step={step}
          className={clx(
            'w-full bg-surface border border-border2 rounded-lg py-2 text-xs text-ink outline-none',
            'focus:border-ink3 transition-colors',
            prefix ? 'pl-6 pr-2' : 'px-2.5'
          )}
        />
      </div>
    </div>
  );
}

// ── Stat tile ─────────────────────────────────────────────────────────────────
function Stat({ label, value, sub, valueClass }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-ink3 text-[11px]">{label}</span>
      <span className={clx('font-mono font-bold text-sm', valueClass || 'text-ink')}>{value}</span>
      {sub && <span className="text-ink3 text-[10px]">{sub}</span>}
    </div>
  );
}

// ── Price chart custom dot (trade signals) ────────────────────────────────────
function SignalDot(props) {
  const { cx, cy, payload, signalMap } = props;
  const sig = signalMap?.[payload?.date];
  if (!sig) return null;
  const fill = sig === 'BUY' ? '#22c55e' : '#ef4444';
  return (
    <g>
      <circle cx={cx} cy={cy} r={4} fill={fill} stroke="#0a0a0a" strokeWidth={1.5} />
      <text x={cx} y={cy - 8} textAnchor="middle" fill={fill} fontSize={8} fontFamily="monospace">
        {sig === 'BUY' ? '▲' : '▼'}
      </text>
    </g>
  );
}

// ── Chart tooltip ─────────────────────────────────────────────────────────────
function PriceTooltip({ active, payload, label, signalMap }) {
  if (!active || !payload?.length) return null;
  const sig = signalMap?.[label];
  return (
    <div className="bg-surface2 border border-border2 rounded-lg p-2.5 text-xs shadow-xl min-w-[120px]">
      <div className="text-ink3 mb-1.5">{label}</div>
      <div className="font-mono font-bold text-ink">{fmt$(payload[0].value)}</div>
      {sig && (
        <div className={clx('mt-1 text-[10px] font-semibold', sig === 'BUY' ? 'text-green' : 'text-red')}>
          {sig === 'BUY' ? '▲ BUY signal' : '▼ SELL signal'}
        </div>
      )}
    </div>
  );
}

function EquityTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-surface2 border border-border2 rounded-lg p-2.5 text-xs shadow-xl">
      <div className="text-ink3 mb-1">{label}</div>
      <div className="font-mono font-bold text-ink">{fmt$(payload[0].value)}</div>
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyState({ loading }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center px-8">
      {loading ? (
        <>
          <div className="w-7 h-7 rounded-full border-2 border-ink3 border-t-ink2 animate-spin" />
          <p className="text-ink3 text-sm">Running backtest…</p>
        </>
      ) : (
        <>
          <div className="w-14 h-14 rounded-2xl bg-surface2 border border-border2 flex items-center justify-center">
            <svg className="w-6 h-6 text-ink3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          <div>
            <p className="text-ink2 text-sm font-semibold">No results yet</p>
            <p className="text-ink3 text-xs mt-1.5 leading-relaxed">
              Pick a stock &amp; strategy,<br/>then hit Run Backtest
            </p>
          </div>
        </>
      )}
    </div>
  );
}

// ── Trade log ─────────────────────────────────────────────────────────────────
function TradeLog({ trades }) {
  const rows = [...trades].reverse().slice(0, 30);
  return (
    <div className="border-t border-border">
      <div className="flex items-center gap-2 px-5 py-3 border-b border-border">
        <span className="text-ink3 text-[11px] uppercase tracking-wide">Trade Log</span>
        <span className="font-mono text-ink3 text-[10px]">({trades.length} total)</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              {['Date','Side','Price','Shares','Value','P&L','Signal'].map((h, i) => (
                <th key={h} className={clx(
                  'px-4 py-2.5 font-medium text-ink3 text-[11px] uppercase tracking-wide whitespace-nowrap',
                  i <= 1 ? 'text-left' : 'text-right'
                )}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((t, i) => (
              <tr key={i} className="border-b border-border hover:bg-surface transition-colors">
                <td className="px-4 py-2 font-mono text-ink3">{t.date}</td>
                <td className="px-4 py-2">
                  <Pill color={t.type === 'BUY' ? 'green' : 'red'}>{t.type}</Pill>
                </td>
                <td className="px-4 py-2 font-mono text-ink text-right">{fmt$(t.price)}</td>
                <td className="px-4 py-2 font-mono text-ink text-right">{t.shares}</td>
                <td className="px-4 py-2 font-mono text-ink text-right">{fmt$(t.value)}</td>
                <td className={clx(
                  'px-4 py-2 font-mono font-semibold text-right',
                  t.profit == null ? 'text-ink3' : t.profit > 0 ? 'text-green' : 'text-red'
                )}>
                  {t.profit != null ? fmt$(t.profit) : '—'}
                </td>
                <td className="px-4 py-2 font-mono text-ink3 text-right">{t.indicator ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Hyperparam config ─────────────────────────────────────────────────────────
const HYPERPARAM_DEFS = {
  sma_crossover: [
    { key: 'sma_short', label: 'Short window', min: 5,  max: 100, step: 1,   default: 20,
      tip: 'Faster moving average. A buy signal fires when this crosses above the long window.' },
    { key: 'sma_long',  label: 'Long window',  min: 10, max: 200, step: 5,   default: 50,
      tip: 'Slower moving average. A sell signal fires when the short window crosses back below this.' },
  ],
  rsi: [
    { key: 'rsi_period',     label: 'Period',     min: 5,  max: 50,  step: 1,  default: 14,
      tip: 'Number of days used to calculate RSI. Lower = more sensitive to recent price swings.' },
    { key: 'rsi_oversold',   label: 'Oversold',   min: 10, max: 45,  step: 5,  default: 30,
      tip: 'Buy when RSI drops below this level. Under 30 typically means the asset is oversold.' },
    { key: 'rsi_overbought', label: 'Overbought', min: 55, max: 90,  step: 5,  default: 70,
      tip: 'Sell when RSI rises above this level. Over 70 typically means the asset is overbought.' },
  ],
  mean_reversion: [
    { key: 'bb_period', label: 'Period',  min: 5,   max: 50,  step: 1,   default: 20,
      tip: 'Lookback period for the moving average and standard deviation used to build the bands.' },
    { key: 'bb_std',    label: 'Std dev', min: 0.5, max: 3.5, step: 0.5, default: 2.0,
      tip: 'How many standard deviations wide the bands are. Higher = wider bands, fewer but stronger signals.' },
  ],
  ml_random_forest: [
    { key: 'n_estimators', label: 'Estimators', min: 10, max: 500, step: 10, default: 100,
      tip: 'Number of decision trees in the forest. More trees improve accuracy but take longer to run.' },
    { key: 'max_depth',    label: 'Max depth',  min: 2,  max: 30,  step: 1,  default: 10,
      tip: 'How deep each decision tree can grow. Too high risks memorizing the training data (overfitting).' },
    { key: 'train_split',  label: 'Train %',    min: 50, max: 80,  step: 5,  default: 70,
      tip: 'Percentage of historical data used to train the model. The rest is used to run the backtest.' },
  ],
};

const DEFAULT_HYPERPARAMS = Object.values(HYPERPARAM_DEFS)
  .flat()
  .reduce((acc, p) => ({ ...acc, [p.key]: p.default }), {});

// ── Hyperparams panel ─────────────────────────────────────────────────────────
function HyperParams({ strategy, params, onChange }) {
  const defs = HYPERPARAM_DEFS[strategy] ?? [];
  if (!defs.length) return null;
  return (
    <section className="flex flex-col gap-2">
      <p className="text-ink3 text-[11px] uppercase tracking-wide">Hyperparameters</p>
      <div className="flex flex-col gap-2">
        {defs.map(d => (
          <div key={d.key} className="flex items-center justify-between gap-2">
            <label className="text-ink2 text-xs flex-1 flex items-center">
              {d.label}
              {d.tip && <InfoTip text={d.tip} />}
            </label>
            <input
              type="number"
              value={params[d.key] ?? d.default}
              onChange={e => onChange(d.key, Number(e.target.value))}
              min={d.min} max={d.max} step={d.step}
              className="w-20 bg-surface border border-border2 rounded-md px-2 py-1 text-xs text-ink text-right outline-none focus:border-ink3 transition-colors font-mono"
            />
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tickers, setTickers]     = useState([]);
  const [strategies, setStrategies] = useState([]);
  const [ticker, setTicker]       = useState('AAPL');
  const [strategy, setStrategy]   = useState('sma_crossover');
  const [capital, setCapital]     = useState(10000);
  const [days, setDays]           = useState(500);
  const [hyperparams, setHyperparams] = useState(DEFAULT_HYPERPARAMS);
  const [loading, setLoading]     = useState(false);
  const [results, setResults]     = useState(null);
  const [error, setError]         = useState(null);

  const setParam = (key, value) =>
    setHyperparams(prev => ({ ...prev, [key]: value }));

  useEffect(() => {
    Promise.all([
      fetch(`${API}/tickers`).then(r => r.json()),
      fetch(`${API}/strategies`).then(r => r.json()),
    ])
      .then(([t, s]) => { setTickers(t); setStrategies(s); })
      .catch(() => setError('Cannot reach API — run: uvicorn main:app --reload'));
  }, []);

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy, ticker, capital, days, ...hyperparams }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Backtest failed');
      setResults(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const m = results?.metrics;
  const retColor  = m ? (m.total_return > 0 ? 'text-green' : m.total_return < 0 ? 'text-red' : 'text-ink') : 'text-ink';
  const eqColor   = m ? (m.total_return >= 0 ? '#22c55e' : '#ef4444') : '#3b82f6';
  const gradId    = 'eqGrad';
  const tickStep  = results ? Math.max(1, Math.floor((results.price_data?.length ?? 0) / 6)) : 1;
  const eqStep    = results ? Math.max(1, Math.floor((results.equity_curve?.length ?? 0) / 6)) : 1;

  return (
    <div className="flex flex-col h-full bg-bg text-ink overflow-hidden">

      {/* ── Header ── */}
      <header className="h-11 flex items-center gap-4 px-5 border-b border-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded bg-ink/90 flex items-center justify-center">
            <svg className="w-3 h-3 text-bg" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="2 11 6 6 9 9 14 4"/>
            </svg>
          </div>
          <span className="font-semibold text-sm tracking-tight">Backtest</span>
        </div>

        <div className="h-4 w-px bg-border2" />

        {results && (
          <div className="flex items-center gap-1.5">
            <span className="font-mono font-bold text-sm">{results.ticker}</span>
            <span className="text-ink3 text-xs">{results.ticker_name}</span>
            <span className={clx('font-mono text-xs font-semibold ml-2', retColor)}>
              {fmtPct(m.total_return)}
            </span>
          </div>
        )}

        {error && (
          <div className="ml-auto px-3 py-1 bg-red/10 border border-red/20 rounded-md text-red text-xs">
            {error}
          </div>
        )}
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left sidebar ── */}
        <aside className="w-60 flex-shrink-0 border-r border-border bg-surface flex flex-col overflow-y-auto">
          <div className="flex flex-col gap-4 p-4">

            {/* Ticker */}
            <section className="flex flex-col gap-2">
              <p className="text-ink3 text-[11px] uppercase tracking-wide">Stock</p>
              {tickers.length > 0
                ? <TickerSearch tickers={tickers} selected={ticker} onSelect={setTicker} />
                : <div className="h-10 bg-surface2 rounded-lg animate-pulse" />
              }
            </section>

            {/* Strategies */}
            <section className="flex flex-col gap-2">
              <p className="text-ink3 text-[11px] uppercase tracking-wide">Strategy</p>
              <div className="flex flex-col gap-1">
                {strategies.map(s => (
                  <StrategyCard key={s.id} s={s} selected={strategy} onSelect={setStrategy} />
                ))}
              </div>
            </section>

            {/* Strategy hyperparams */}
            <HyperParams strategy={strategy} params={hyperparams} onChange={setParam} />

            {/* General params */}
            <section className="flex flex-col gap-3">
              <p className="text-ink3 text-[11px] uppercase tracking-wide">General</p>
              <NumberInput label="Capital" value={capital} onChange={setCapital} prefix="$" min={1000} step={1000}
                tip="Starting cash for the backtest. The strategy buys as many whole shares as this allows." />
              <NumberInput label="Days" value={days} onChange={setDays} min={100} max={2000} step={100}
                tip="How many trading days of historical data to use. ~252 = 1 year, ~500 = 2 years." />
            </section>
          </div>

          {/* Run button pinned to bottom */}
          <div className="p-4 mt-auto border-t border-border">
            <button
              onClick={run}
              disabled={loading}
              className={clx(
                'w-full py-2.5 rounded-lg text-sm font-semibold transition-all',
                loading
                  ? 'bg-surface2 text-ink3 cursor-not-allowed'
                  : 'bg-ink text-bg hover:bg-ink/90 active:scale-[0.98]'
              )}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-ink3 border-t-ink2 animate-spin" />
                  Running…
                </span>
              ) : 'Run Backtest'}
            </button>
          </div>
        </aside>

        {/* ── Main panel ── */}
        <main className="flex-1 overflow-y-auto flex flex-col">
          {!results && !loading ? (
            <EmptyState loading={false} />
          ) : loading && !results ? (
            <EmptyState loading={true} />
          ) : results && (
            <>
              {/* Stats row */}
              <div className="border-b border-border px-6 py-4 flex items-center gap-0 overflow-x-auto">
                {[
                  { label: 'Total Return', value: fmtPct(m.total_return), cls: retColor },
                  { label: 'Final Value',  value: fmt$(m.final_value) },
                  { label: 'Sharpe',       value: m.sharpe_ratio.toFixed(2),
                    cls: m.sharpe_ratio > 1 ? 'text-green' : m.sharpe_ratio < 0 ? 'text-red' : '' },
                  { label: 'Max Drawdown', value: fmtPct(m.max_drawdown),
                    cls: m.max_drawdown < -10 ? 'text-red' : '' },
                  { label: 'Win Rate',     value: `${m.win_rate}%`,
                    cls: m.win_rate > 55 ? 'text-green' : m.win_rate < 40 ? 'text-red' : '' },
                  { label: 'Closed Trades', value: m.total_trades,
                    tip: 'Completed buy+sell round-trips. Open positions at period end are not counted.' },
                  results.accuracy != null
                    ? { label: 'ML Accuracy', value: `${results.accuracy}%`, cls: 'text-violet' }
                    : { label: 'Strategy',    value: results.strategy_name.split(' (')[0] },
                ].map((s, i, arr) => (
                  <div key={i} className={clx('flex flex-col gap-1 px-6', i < arr.length - 1 && 'border-r border-border')}>
                    <span className="text-ink3 text-[11px] whitespace-nowrap flex items-center">
                      {s.label}{s.tip && <InfoTip text={s.tip} />}
                    </span>
                    <span className={clx('font-mono font-bold text-sm whitespace-nowrap', s.cls || 'text-ink')}>{s.value}</span>
                  </div>
                ))}
              </div>

              {/* Price chart */}
              <div className="px-6 pt-6 pb-3">
                <p className="text-ink3 text-[11px] uppercase tracking-wide mb-4">
                  Price  <span className="text-ink2 normal-case font-mono">· {results.ticker}</span>
                  <span className="ml-3 text-[10px]">
                    <span className="inline-block w-2 h-2 rounded-full bg-green mr-1 align-middle" />buy
                    <span className="inline-block w-2 h-2 rounded-full bg-red ml-2 mr-1 align-middle" />sell
                  </span>
                </p>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={results.price_data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" vertical={false} />
                    <XAxis dataKey="date" stroke="transparent" tick={{ fill: '#555', fontSize: 10 }}
                      tickLine={false} interval={tickStep}
                      tickFormatter={fmtDate} />
                    <YAxis stroke="transparent" tick={{ fill: '#555', fontSize: 10 }}
                      tickLine={false} tickFormatter={fmtK} width={48} />
                    <Tooltip content={<PriceTooltip signalMap={results.signal_map} />} wrapperStyle={{ transition: 'none' }} />
                    <Line type="monotone" dataKey="price" stroke="#555555" strokeWidth={1.5}
                      dot={(props) => <SignalDot {...props} signalMap={results.signal_map} />}
                      activeDot={{ r: 3, fill: '#e8e8e8', strokeWidth: 0 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Equity curve */}
              <div className="px-6 pt-2 pb-6">
                <p className="text-ink3 text-[11px] uppercase tracking-wide mb-4">
                  Portfolio Value
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={results.equity_curve} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={eqColor} stopOpacity={0.15} />
                        <stop offset="100%" stopColor={eqColor} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" vertical={false} />
                    <XAxis dataKey="date" stroke="transparent" tick={{ fill: '#555', fontSize: 10 }}
                      tickLine={false} interval={eqStep} tickFormatter={fmtDate} />
                    <YAxis stroke="transparent" tick={{ fill: '#555', fontSize: 10 }}
                      tickLine={false} tickFormatter={fmtK} width={48} />
                    <ReferenceLine y={capital} stroke="#2a2a2a" strokeDasharray="4 3" />
                    <Tooltip content={<EquityTooltip />} wrapperStyle={{ transition: 'none' }} />
                    <Area type="monotone" dataKey="value" stroke={eqColor} strokeWidth={1.5}
                      fill={`url(#${gradId})`} dot={false}
                      activeDot={{ r: 3, fill: eqColor, strokeWidth: 0 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Trade log */}
              <TradeLog trades={results.trades} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}

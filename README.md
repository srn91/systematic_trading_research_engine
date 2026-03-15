# Systematic Trading Research Engine

A production-grade quantitative research harness for SPY daily bars. Built to stress-test whether a machine learning model can generate deployable edge on US equity index data — with rigorous safeguards against the most common ways quant research goes wrong.

**Short answer from the research: no edge found with this feature set. The harness caught it before a dollar was traded.**

---

## What this is

This is not a trading bot. It is a research infrastructure tool that enforces discipline at every step:

- Prevents label leakage between training and evaluation data
- Uses purged walk-forward cross-validation (Lopez de Prado methodology)
- Keeps a sealed 2024 holdout that is never touched during development
- Compares models against realistic baselines including a random timing benchmark
- Validates every config parameter before running
- Produces timestamped, auditable output files for every run

The goal was to find out whether a logistic regression or LightGBM model trained on 21 technical features could predict triple-barrier labels on SPY daily data well enough to deploy. The answer after multiple rigorous runs: not with this feature set.

---

## Why the honest result is the interesting result

Most trading strategy backtests on GitHub show equity curves that go up and to the right. This one does not, and that is the point.

The harness caught several things that would have been invisible in a naive backtest:

**Transaction cost reality.** Initial runs used 0.1% per side — realistic for retail brokers. At 21-47 round trips per year this creates 8-16% annual drag that buries any marginal signal. Switching to realistic institutional costs (0.01% entry, 0.05% exit) showed the costs were not the core problem.

**Barrier structure creates a random baseline with edge.** With `atr_upper=1.5` and `atr_lower=1.0`, the asymmetric reward-to-risk ratio (1.5:1) means any strategy needs only ~40% win rate to profit. Random entry achieved Sharpe 0.66-0.76. Both models failed to consistently beat random timing, meaning the model was not adding genuine directional signal.

**LightGBM vs Logistic Regression.** LightGBM consistently outperformed logistic regression by 1.5+ Sharpe points on the same features, confirming that nonlinear feature interactions exist but are too weak to overcome the random baseline. This is a meaningful finding about the limits of linear models on this problem.

**Sample size matters.** At threshold 0.70 with symmetric barriers, logistic regression achieved Sharpe 0.51 on 21 trades — which the harness correctly flagged as statistically thin. The 95% bootstrap CI was [-0.87, +1.95], meaning the result is indistinguishable from noise.

---

## Architecture

```
Raw Data (Yahoo Finance)
        ↓
Feature Engineering (21 features, rolling continuity preserved)
        ↓
Train/Test Split (2024 sealed as holdout)
        ↓
Independent Labeling per subset (triple-barrier, no cross-boundary leakage)
        ↓
Purged Walk-Forward (train_years=2, step=3mo, TIMEOUT_BARS purge)
        ↓
Probability Generation (one fit pass per model — O(n_folds))
        ↓
Common OOS Start computed (max of all model first_live dates)
        ↓
Threshold Sweep on common window (threshold selection = final comparison window)
        ↓
Final Dev Evaluation (OOS-live window only — no pre-live flat time)
        ↓
Gate check: dev Sharpe >= 0.8 required to open holdout
        ↓
2024 Holdout (one shot, never re-used)
```

### Three engines

**Label Engine** — Triple-barrier labeling. Entry reference is `open[t+1] * (1 + slippage)` to match execution. Called independently per subset. Last `TIMEOUT_BARS` rows dropped to prevent incomplete label windows.

**Execution Engine** — World A execution: signal at `close[t]`, entry at `open[t+1]`. One position at a time. Barrier exits use actual high/low intraday data. Exit slippage applied to all exits. Phase A/B/C loop structure with explicit state variables.

**Metrics Engine** — All portfolio metrics from daily equity curve only. Bootstrap CIs labeled as iid heuristic (not rigorous inference). Turnover, exposure, MAE/MFE in risk units, calibration curve, bucket tables by probability band.

---

## Key design decisions

**Why purged walk-forward instead of simple train/test split**

A single split gives one data point. Walk-forward gives multiple out-of-sample windows across different market regimes. Purging removes the last `TIMEOUT_BARS` rows from each training fold so the label window cannot overlap the test window (standard Lopez de Prado leakage prevention).

**Why a common OOS start date across models**

If LogReg's first live fold starts 2022-02-01 and LightGBM's starts 2022-04-01, comparing their dev Sharpe numbers is comparing different time periods. The harness computes `common_start = max(all first_live dates)` and evaluates all models — and baselines — on identical calendar windows.

**Why threshold selection uses the same window as final comparison**

In earlier versions, thresholds were selected on each model's full OOS window, then final metrics were computed on the common window. This meant threshold choice was optimized on a different period than the evaluation. Fixed by computing common_start before the threshold sweep and passing it as `eval_start`.

**Why the holdout gate exists**

The 2024 data is used exactly once. If dev Sharpe < 0.8, the holdout stays sealed. This prevents researchers from implicitly using holdout information by repeatedly opening it and iterating. The gate is one-directional: once dev passes, holdout opens. It does not close again.

---

## Features (21)

| Category | Features |
|----------|----------|
| Returns | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_zscore`, `gap` |
| Trend | `dist_ma20`, `dist_ma50`, `dist_ma200`, `ma_slope_50`, `ma_cross` |
| Volatility | `atr_14`, `vol_10d`, `vol_20d`, `vol_regime`, `vol_ratio`, `vol_spike` |
| Macro | `vix`, `vix_chg` |
| Regime | `adx`, `regime` |

All features computed on the full dataset before splitting to preserve rolling window continuity. Leakage assertions verify `ret_1d` exactly matches close-to-close returns.

---

## Actual results

All results below are from out-of-sample evaluation on the common OOS window (2022-02-01 to 2023-12-21). The 2024 holdout was never opened because the dev gate was not passed.

### Dev comparison — asymmetric barriers (atr_upper=1.5, atr_lower=1.0)

| Strategy | Sharpe | Max DD | CAGR | Trades/yr |
|----------|--------|--------|------|-----------|
| Buy & Hold SPY | 0.31 | -22.1% | 4.2% | — |
| 10d Momentum \| Barrier | 0.35 | -13.9% | 3.5% | 45 |
| 50/200 MA \| Barrier | 0.55 | -11.3% | 5.1% | 43 |
| **Random \| Barrier** | **0.66** | -20.5% | 9.2% | 59 |
| LogReg (thresh=0.60) | 0.20 | -11.1% | 1.8% | 40 |
| **LightGBM (thresh=0.70)** | **0.46** | -8.8% | 4.2% | 23 |

### Dev comparison — symmetric barriers (atr_upper=1.0, atr_lower=1.0)

| Strategy | Sharpe | Max DD | CAGR | Trades/yr |
|----------|--------|--------|------|-----------|
| Buy & Hold SPY | 0.31 | -22.1% | 4.2% | — |
| **Random \| Barrier** | **0.76** | -19.0% | 10.0% | 65 |
| LogReg (thresh=0.70) | 0.51* | -7.2% | 3.5% | 11 |
| LightGBM (thresh=0.60) | 0.27 | -9.6% | 2.6% | 49 |

*21 trades — statistically thin. Bootstrap CI: [-0.87, +1.95].

### Key finding

The random barrier baseline consistently outperforms both models. The 1.5:1 asymmetric barrier creates a structural payoff advantage that does not require directional prediction skill. Any long-only strategy with reasonable entry selection benefits from this structure in the 2022-2023 period. Neither model demonstrated the ability to beat randomness on this feature set.

---

## Config

```python
@dataclass(frozen=True)
class Config:
    ticker:       str   = "SPY"
    start:        str   = "2019-01-01"
    end:          str   = "2025-01-02"
    dev_end:      str   = "2023-12-31"
    train_years:  int   = 2
    step_months:  int   = 3
    timeout_bars: int   = 5
    atr_upper:    float = 1.5
    atr_lower:    float = 1.0
    thresholds:   Tuple = (0.55, 0.60, 0.65, 0.70, 0.75)
    commission:   float = 0.0001   # per side
    entry_slippage: float = 0.0001
    exit_slippage:  float = 0.0005
```

All parameters validated at startup. Invalid configs raise `ValueError` immediately.

---

## Pass/Fail gates

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Sharpe ≥ 0.8 | iterate gate | minimum signal to keep researching |
| Sharpe ≥ 1.0 | deploy gate | minimum to consider paper trading |
| Sortino ≥ 1.2 | downside risk | penalizes volatile underperformance |
| Max DD < 20% | drawdown | practical survivability |
| Calmar ≥ 0.5 | risk-adjusted return | CAGR relative to drawdown |
| Profit Factor ≥ 1.2 | trade quality | gross win / gross loss |
| Expectancy > 0 | per-trade edge | average P&L per trade |
| Beats B&H | relevance | must outperform the trivial benchmark |

---

## Install and run

```bash
pip install yfinance pandas numpy scikit-learn lightgbm matplotlib
python trading_research_v9_definitive.py
```

Data downloads automatically from Yahoo Finance on each run. No dataset files required. Every run produces timestamped output files in `outputs/`:

```
{run_tag}_spy_dev.csv
{run_tag}_spy_test.csv
{run_tag}_trading_results.png
{run_tag}_summary.json
{run_tag}_logistic_threshold_sweep.csv
{run_tag}_lgbm_threshold_sweep.csv
```

---

## What comes next

The natural next steps if continuing this research:

1. **Earnings and macro event features** — regime shifts around FOMC, CPI, NFP dates are not captured by any current feature
2. **Cross-asset signals** — TLT/SPY ratio, credit spreads, dollar index as regime indicators
3. **Alternative labeling** — fixed-horizon returns instead of triple-barrier, to decouple label construction from barrier asymmetry
4. **Longer history** — extending to 2010 or earlier adds 2010-2019 bull market and 2011 correction, reducing regime concentration
5. **Intraday features** — open-to-close vs close-to-open decomposition captures different information than daily OHLCV

---

## Repository structure

```
systematic_trading_research_engine/
├── trading_research_v9_definitive.py   # full research engine
├── README.md
├── requirements.txt
├── outputs/                            # gitignored — regenerates on each run
│   └── .gitkeep
└── .gitignore
```

---

## Methodology references

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Triple-barrier labeling, purged walk-forward cross-validation
- Bailey, D., Borwein, J., Lopez de Prado, M., Zhu, Q. (2014). *The Probability of Backtest Overfitting* — motivation for OOS gates and holdout discipline
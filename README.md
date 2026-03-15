# Systematic Trading Research Engine

A production-grade quantitative research harness for SPY daily bars. Built to test whether machine learning can generate deployable edge on US equity index data while defending against the classic failure modes of quant research: leakage, bad baselines, sloppy evaluation windows, and fake confidence from thin samples.

**Bottom line: no deployable edge found with this feature set.**  
The engine did its job. It rejected weak signal before any real capital was put at risk.

---

## What this is

This is **not** a trading bot. It is a research engine designed to answer one hard question honestly:

> Can a model trained on daily technical and regime features predict short-horizon SPY direction well enough to justify deployment?

The engine enforces discipline through:

- independent labeling of dev and test subsets to prevent cross-boundary leakage
- purged walk-forward validation instead of a single lucky split
- a sealed 2024 holdout that stays closed unless the dev result is strong enough
- common out-of-sample evaluation windows across models for fair comparison
- realistic execution assumptions with entry slippage, exit slippage, and commissions
- auditable timestamped artifacts for every run

The point is not to manufacture an upward-sloping backtest. The point is to kill bad ideas fast.

---

## Latest canonical run

This README reflects the latest run below:

- **Run tag:** `20260315_130256`
- **Config:** symmetric barriers, `atr_upper=1.0`, `atr_lower=1.0`
- **Tie policy:** `worst_case`
- **Dev window:** 2020-01-30 to 2023-12-29 after feature warmup and labeling
- **Common OOS comparison window:** 2022-02-01 to 2023-12-21
- **2024 holdout:** remained sealed because dev Sharpe did not clear the gate

### Published result summary

| Strategy | Sharpe | Sortino | Max DD | CAGR | Trades/yr |
|----------|--------|---------|--------|------|-----------|
| Buy & Hold SPY | 0.31 | 0.46 | -22.1% | 4.2% | 0.5 |
| 10d Momentum \| Barrier | 0.54 | 0.72 | -12.5% | 5.5% | 52.4 |
| 50/200 MA \| Barrier | 0.55 | 0.67 | -11.2% | 4.9% | 51.9 |
| 10d Momentum \| Flip | 0.20 | 0.23 | -15.2% | 1.7% | 13.8 |
| **Random \| Barrier** | **0.76** | **1.01** | -19.0% | 10.0% | 64.6 |
| Random Timing (Matched Count) | -0.41 | -0.55 | -24.9% | -7.2% | 70.9 |
| **LogReg \| thresh=0.70** | **0.51** | **0.32** | **-7.2%** | **3.5%** | **11.1** |
| LGBM \| thresh=0.75 | 0.33 | 0.32 | -7.5% | 2.6% | 34.4 |

### Main finding

The best model was logistic regression at threshold `0.70`, with dev Sharpe `0.51`. That is still a fail under the engine’s research gate of `Sharpe >= 0.8`. LightGBM improved from earlier unstable runs, but still underperformed logistic regression and still failed the deployment bar.

The most important comparison is not “model versus buy-and-hold.” It is **model versus strong dumb baselines**. On that test, the feature set lost. The `Random | Barrier` baseline posted a Sharpe of `0.76`, beating both ML models.

That means the barrier structure and long-only market regime are doing a lot of the work. The model is not adding enough real edge.

---

## Why this result matters

Most GitHub strategy repos are theater. They give you a backtest curve and hope you don’t inspect the plumbing. This repo does the opposite.

It found several uncomfortable truths:

### 1. The model does not beat randomness convincingly
A random long-only barrier strategy reached **Sharpe 0.76** in the common OOS window.  
Logistic regression reached **0.51**.  
LightGBM reached **0.33**.

That is the whole knife fight. If random timing with the same execution structure can do better than your ML model, your “signal” is probably regime + payoff design, not predictive skill.

### 2. Logistic regression beat LightGBM
That is mildly surprising and useful. It suggests one of two things:

- the nonlinear interactions in these 21 features are too weak or unstable to monetize, or
- LightGBM is fitting noise that does not survive out-of-sample

Either way, extra complexity was not earned.

### 3. The best LogReg result is statistically thin
LogReg’s best setting took only **21 trades** over the common OOS window.

Its bootstrap Sharpe CI was:

- **[-0.87, 1.95]**

That interval is enormous. Translation: the observed Sharpe is nowhere near stable enough to trust.

### 4. Calibration is not strong
The probability calibration tables show visible misalignment at higher predicted probability buckets, especially for LightGBM. When a model says “high confidence” but realized hit rates do not track, thresholding becomes fragile and seductive nonsense creeps in.

### 5. The holdout stayed sealed
This is a feature, not a limitation.

The 2024 holdout was not opened because the best dev Sharpe was only `0.51`, below the required `0.8`. That prevents the all-too-common clown show where people keep peeking at test data until the “research” magically works.

---

## Research architecture

```text
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
Threshold Sweep on common window
        ↓
Final Dev Evaluation (OOS-live window only)
        ↓
Gate check: dev Sharpe >= 0.8 required to open holdout
        ↓
2024 Holdout (one shot, never re-used)
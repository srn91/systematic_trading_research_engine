#!/usr/bin/env python3
"""
TRADING RESEARCH ENGINE — V9 DEFINITIVE
=========================================
V9: one targeted fix from ChatGPT V8 review.

FIX:
  summary.json now includes final_model_winner_by_test_sharpe.
  - None if holdout never opened.
  - If exactly one model ran in holdout, that model name is stored.
  - If multiple models ran, the field stores the one with highest 2024 test Sharpe.
  - dev_model_winner (dev-only comparison) is still present and distinct.

All prior fixes from v8-definitive preserved.

Install:
  pip install yfinance pandas numpy scikit-learn lightgbm matplotlib

Run:
  python trading_research_v9_definitive.py
"""
from __future__ import annotations

import os
import sys
import json
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

try:
    import yfinance as yf
except Exception as e:
    raise ImportError("yfinance required: pip install yfinance") from e

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# A2: Print at startup — before any config or data
print(f"  [INFO] LightGBM available: {HAS_LGBM}")
if not HAS_LGBM:
    print("  [INFO] Install with: pip install lightgbm")
    print("  [INFO] LightGBM model will be skipped in this run.")


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    ticker:      str   = "SPY"
    vix_ticker:  str   = "^VIX"
    start:       str   = "2019-01-01"
    # T1-3: "2025-01-02" avoids yfinance exclusive-end off-by-one on Dec 31
    end:         str   = "2025-01-02"

    dev_end:     str   = "2023-12-31"   # 2024 sealed until run_final_test()

    train_years: int   = 2
    
    step_months: int   = 3
    timeout_bars: int  = 5
    atr_upper:   float = 1.0
    atr_lower:   float = 1.0    

    # T2-8: Tie policy when both barriers hit same bar.
    # "worst_case" = stop wins (conservative default).
    # "best_case"  = target wins (optimistic upper bound).
    # Note: "skip" removed — it was fake (silently treated as stop).
    intraday_tie_policy: str = "worst_case"

    thresholds: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)
    min_trades:  int   = 15
    min_pf:      float = 1.00
    min_exposure:float = 0.01

    commission:   float = 0.0001
    entry_slippage: float = 0.0001
    exit_slippage:  float = 0.0005

    # T3-11: Bootstrap parameters
    n_bootstrap: int   = 1000
    bootstrap_ci:float = 0.95

    random_seed: int   = 42
    # T1-4: Portable relative path — not hardcoded /mnt/...
    out_dir:     str   = "outputs"


CFG = Config()


# ============================================================
# UTILITIES
# ============================================================

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in c).lower().strip("_")
                      for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, keyword: str) -> pd.Series:
    kw = keyword.lower()
    for c in df.columns:
        if kw in c.lower():
            return df[c]
    raise KeyError(f"Column '{keyword}' not found. Available: {list(df.columns)}")


def _annualize_sharpe(daily_rets: pd.Series) -> float:
    r   = daily_rets.fillna(0.0)
    std = r.std()
    return float(r.mean() / (std + 1e-12) * np.sqrt(252)) if std > 0 else 0.0


def _safe_fmt(value: Any, fmt: str, na: str = "N/A") -> str:
    """No conditional inside format spec — no f-string crash possible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return na
    return fmt.format(value)


# ============================================================
# STEP 1: DATA
# T1-3: end="2025-01-02" avoids off-by-one on Dec 31 2024
# ============================================================

def download_data(cfg: Config) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1: DATA")
    print(f"  T1-3: Downloading through {cfg.end} (avoids Dec-31 off-by-one)")
    print("="*60)

    spy = yf.download(cfg.ticker, start=cfg.start, end=cfg.end,
                      auto_adjust=True, progress=False)
    vix = yf.download(cfg.vix_ticker, start=cfg.start, end=cfg.end,
                      auto_adjust=True, progress=False)

    spy, vix = _flatten_columns(spy), _flatten_columns(vix)

    df = pd.DataFrame({
        "open":   _pick_col(spy, "open").values,
        "high":   _pick_col(spy, "high").values,
        "low":    _pick_col(spy, "low").values,
        "close":  _pick_col(spy, "close").values,
        "volume": _pick_col(spy, "volume").values,
    }, index=spy.index)

    df["vix"] = _pick_col(vix, "close").reindex(df.index).ffill()
    df = df.sort_index().dropna()

    if df.empty:
        raise RuntimeError("Download returned empty dataframe.")
    if not (df["close"] > 0).all():
        raise ValueError("Non-positive prices detected.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index not sorted.")
    if df.index.duplicated().any():
        raise ValueError("Duplicate dates detected.")

    n_dev  = int((df.index <= pd.Timestamp(cfg.dev_end)).sum())
    n_test = int((df.index >  pd.Timestamp(cfg.dev_end)).sum())
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Dev rows:   {n_dev}  ({cfg.start} → {cfg.dev_end})")
    print(f"  Test rows:  {n_test} (2024 — sealed until run_final_test())")
    return df


# ============================================================
# STEP 2: FEATURES
# ============================================================

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l = df["high"], df["low"]
    up, dn = h - h.shift(), l.shift() - l
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = _atr(df, period)
    pdi = 100 * pdm.ewm(span=period).mean() / atr
    mdi = 100 * mdm.ewm(span=period).mean() / atr
    dx  = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    return dx.ewm(span=period).mean()


def engineer_features(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    print("\n" + "="*60)
    print("STEP 2: FEATURES (computed on full df — rolling continuity)")
    print("="*60)

    f = df_raw.copy()
    c = f["close"]

    f["ret_1d"]      = c.pct_change(1)
    f["ret_3d"]      = c.pct_change(3)
    f["ret_5d"]      = c.pct_change(5)
    f["ret_10d"]     = c.pct_change(10)
    mu, sig          = f["ret_1d"].rolling(20).mean(), f["ret_1d"].rolling(20).std()
    f["ret_zscore"]  = (f["ret_1d"] - mu) / sig
    f["gap"]         = (f["open"] - c.shift(1)) / c.shift(1)

    for w in (20, 50, 200):
        f[f"ma_{w}"]      = c.rolling(w).mean()
        f[f"dist_ma{w}"]  = (c - f[f"ma_{w}"]) / f[f"ma_{w}"]
    f["ma_slope_50"] = f["ma_50"].pct_change(5)
    f["ma_cross"]    = np.sign(f["ma_50"] - f["ma_200"])

    f["atr_14"]      = _atr(f, 14)
    f["vol_10d"]     = f["ret_1d"].rolling(10).std()
    f["vol_20d"]     = f["ret_1d"].rolling(20).std()
    f["vol_regime"]  = f["vol_20d"].rolling(252).rank(pct=True)
    f["vol_ratio"]   = f["volume"] / f["volume"].rolling(20).mean()
    f["vol_spike"]   = (f["vol_ratio"] > 2.0).astype(int)
    f["vix_chg"]     = f["vix"].pct_change(1)
    f["adx"]         = _adx(f, 14)
    f["regime"]      = np.where(f["adx"] > 25, 1,
                        np.where(f["vol_regime"] > 0.8, 2, 0))
    f = f.dropna().copy()

    FEAT = [
        "ret_1d","ret_3d","ret_5d","ret_10d","ret_zscore","gap",
        "dist_ma20","dist_ma50","dist_ma200","ma_slope_50","ma_cross",
        "atr_14","vol_10d","vol_20d","vol_regime",
        "vol_ratio","vol_spike","vix","vix_chg","adx","regime",
    ]

    expected = df_raw["close"].pct_change(1).reindex(f.index)
    if float((f["ret_1d"] - expected).abs().max()) > 1e-10:
        raise AssertionError("LEAKAGE: ret_1d mismatch.")
    if f[FEAT].isna().any().any():
        raise AssertionError("NaN in features after dropna.")

    print(f"  Features: {len(FEAT)} | Rows after warmup: {len(f)}")
    print("  ✓ Leakage assertions passed")
    return f, FEAT


# ============================================================
# ENGINE 1: LABELS
# Entry reference: open[t+1] * (1 + entry_slippage) — matches execution.
# T2-8: tie policy parameterized.
# Last TIMEOUT_BARS rows dropped (incomplete resolution window).
# ============================================================

def label_subset(df: pd.DataFrame, cfg: Config, subset_name: str) -> pd.DataFrame:
    x      = df.copy()
    opens  = x["open"].to_numpy()
    highs  = x["high"].to_numpy()
    lows   = x["low"].to_numpy()
    atrs   = x["atr_14"].to_numpy()
    n      = len(x)

    if n <= cfg.timeout_bars + 2:
        raise ValueError(f"{subset_name}: subset too small for labeling.")

    labels = np.zeros(n, dtype=int)
    n_ties = 0   # count of same-bar both-barrier hits (for diagnostic)

    for t in range(n - 1):
        entry_ref = opens[t + 1] * (1.0 + cfg.entry_slippage)
        atr       = atrs[t]
        upper     = entry_ref + cfg.atr_upper * atr
        lower     = entry_ref - cfg.atr_lower * atr

        last = min(t + cfg.timeout_bars, n - 1)
        for j in range(t + 1, last + 1):
            hit_up = highs[j] >= upper
            hit_dn = lows[j]  <= lower

            if hit_up and hit_dn:
                # T2-8: Parameterized tie policy (worst_case / best_case only)
                n_ties += 1
                if cfg.intraday_tie_policy == "best_case":
                    labels[t] = 1   # target wins
                else:
                    labels[t] = 0   # stop wins (worst_case default)
                break
            elif hit_up:
                labels[t] = 1; break
            elif hit_dn:
                labels[t] = 0; break

    x["label"] = labels
    x = x.iloc[:-cfg.timeout_bars].copy()

    rate    = float(x["label"].mean())
    by_year = x.groupby(x.index.year)["label"].agg(
        long_rate="mean", signals="sum", bars="count")
    min_sig = int(by_year["signals"].min()) if not by_year.empty else 0

    print(f"\n  Labels [{subset_name}]")
    print(f"    Tie policy:     {cfg.intraday_tie_policy} ({n_ties} ties resolved)")
    print(f"    Rows:           {len(x)}")
    print(f"    LONG rate:      {rate:.1%}")
    print(f"    Min sigs/year:  {min_sig}")
    print(by_year.to_string())

    if rate < 0.20:   print("  ⚠ Too few LONG labels — try ATR_UPPER=1.2")
    elif rate > 0.55: print("  ⚠ Too many LONG labels — try ATR_UPPER=1.8")
    else:             print("  ✓ LONG rate acceptable")
    if min_sig < 30:
        print(f"  ⚠ Min signals/year={min_sig} < 30 — overfit risk")

    return x


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(model_type: str, cfg: Config):
    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced",
            solver="lbfgs", random_state=cfg.random_seed)
    if model_type == "lgbm":
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed: pip install lightgbm")
        return lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=15, class_weight="balanced",
            random_state=cfg.random_seed, verbose=-1)
    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# ENGINE 2: EXECUTION ENGINE
# T2-6: exit_slippage applied to all exit prices (stops, targets, timeouts).
#        entry_slippage embedded in entry_price.
# T2-8: tie policy used consistently with labeling.
# World A: signal @ close[t] → entry @ open[t+1].
# ============================================================

def execution_engine(
    df: pd.DataFrame,
    signal_series: pd.Series,
    cfg: Config,
    exit_mode: str = "barrier",
    force_exit_last_bar: bool = True,
    prob_series: Optional[pd.Series] = None,
    fold_series: Optional[pd.Series] = None,
    label_series: Optional[pd.Series] = None,
    threshold: float = 0.60,
    strategy_name: str = "model",
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Cost model (T2-6):
      - Entry:  open[t+1] * (1 + entry_slippage)   → embedded in entry_price
      - Commission: cfg.commission per side
      - Exit:   exit_price * (1 - exit_slippage)   → applied to all exits
    """
    opens  = df["open"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs   = df["atr_14"].to_numpy()
    dates  = df.index
    n      = len(df)

    signal_series = signal_series.reindex(dates).fillna(0).astype(int)

    daily_pnl: pd.Series = pd.Series(0.0, index=dates)
    trades: List[Dict[str, Any]] = []

    in_pos          = False
    signal_idx:     Optional[int]   = None
    entry_idx:      Optional[int]   = None
    entry_sched_for:Optional[int]   = None
    flip_exit_for:  Optional[int]   = None
    entry_price:    Optional[float] = None
    upper:          Optional[float] = None
    lower:          Optional[float] = None
    timeout_bar:    Optional[int]   = None
    mae = mfe       = 0.0

    def _get(series: Optional[pd.Series], idx: Optional[int],
             default: Any = np.nan) -> Any:
        if series is None or idx is None:
            return default
        try:
            return float(series.iloc[idx])
        except Exception:
            return default

    def _apply_exit_slippage(price: float) -> float:
        """T2-6: exit slippage reduces realized exit price."""
        return price * (1.0 - cfg.exit_slippage)

    def _record(exit_i: int, exit_p: float, reason: str) -> None:
        if entry_idx is None or signal_idx is None or entry_price is None:
            return

        gross = (exit_p - entry_price) / entry_price
        net   = gross - cfg.commission * 2

        atr_sig = float(atrs[signal_idx])
        stop_d  = (cfg.atr_lower * atr_sig / entry_price) if entry_price > 0 else np.nan
        tgt_d   = (cfg.atr_upper * atr_sig / entry_price) if entry_price > 0 else np.nan
        mae_r   = (mae / stop_d) if (not pd.isna(stop_d) and stop_d != 0) else np.nan
        mfe_r   = (mfe / tgt_d)  if (not pd.isna(tgt_d)  and tgt_d  != 0) else np.nan

        gap = 0.0
        if entry_idx > 0 and closes[signal_idx] != 0:
            gap = (opens[entry_idx] - closes[signal_idx]) / closes[signal_idx]

        trades.append({
            "signal_date":   dates[signal_idx],
            "entry_date":    dates[entry_idx],
            "exit_date":     dates[exit_i],
            "entry_price":   float(entry_price),
            "exit_price":    float(exit_p),
            "exit_reason":   reason,
            "hold_days":     int(exit_i - entry_idx + 1),
            "gross_ret":     float(gross),
            "net_ret":       float(net),
            "mae":           float(mae),
            "mfe":           float(mfe),
            "mae_r":         float(mae_r) if not pd.isna(mae_r) else np.nan,
            "mfe_r":         float(mfe_r) if not pd.isna(mfe_r) else np.nan,
            "overnight_gap": float(gap),
            "signal_prob":   _get(prob_series,  signal_idx),
            "fold_id":       _get(fold_series,  signal_idx, -1),
            "atr_at_signal": float(atr_sig),
            "label_truth":   _get(label_series, signal_idx),
            "threshold":     float(threshold),
            "strategy":      strategy_name,
            "year":          int(dates[entry_idx].year),
            "regime_at_sig": int(df["regime"].iloc[signal_idx])
                             if "regime" in df.columns else -1,
        })

    def _reset() -> None:
        nonlocal in_pos, signal_idx, entry_idx, entry_sched_for, flip_exit_for
        nonlocal entry_price, upper, lower, timeout_bar, mae, mfe
        in_pos = False
        signal_idx = entry_idx = entry_sched_for = flip_exit_for = None
        entry_price = upper = lower = None
        timeout_bar = None
        mae = mfe = 0.0

    def _tie_exit(barrier_lower: float, barrier_upper: float) -> Tuple[float, str]:
        """T2-8: Tie-break — consistent with labeling. worst_case/best_case only."""
        if cfg.intraday_tie_policy == "best_case":
            return _apply_exit_slippage(barrier_upper), "target_tie"
        else:
            return _apply_exit_slippage(barrier_lower), "stop_tie"

    for i in range(n):
        phase_a_acted = False

        # ── PHASE A: OPEN ──

        if flip_exit_for == i and in_pos:
            # T2-6: apply exit slippage to flip exit
            exit_p  = _apply_exit_slippage(float(opens[i]))
            prev_c  = float(closes[i-1]) if i > 0 else float(entry_price or exit_p)
            day_ret = (exit_p - prev_c) / prev_c - cfg.commission
            daily_pnl.iloc[i] += day_ret
            _record(i, exit_p, "signal_flip")
            _reset()
            phase_a_acted = True

        elif entry_sched_for == i and not in_pos:
            # T2-6: entry_slippage embedded in entry_price
            ep          = float(opens[i]) * (1.0 + cfg.entry_slippage)
            entry_price = ep
            entry_idx   = i
            atr_sig     = float(atrs[signal_idx]) if signal_idx is not None \
                          else float(atrs[max(i-1, 0)])
            upper       = ep + cfg.atr_upper * atr_sig
            lower       = ep - cfg.atr_lower * atr_sig
            timeout_bar = ((signal_idx + cfg.timeout_bars) if signal_idx is not None
                           else (i + cfg.timeout_bars))
            mae = mfe   = 0.0
            entry_sched_for = None

            mfe = max(mfe, (float(highs[i]) - ep) / ep)
            mae = min(mae, (float(lows[i])  - ep) / ep)

            exit_p = reason = None
            is_last = force_exit_last_bar and (i == n - 1)

            if exit_mode == "barrier":
                hit_up = highs[i] >= upper
                hit_dn = lows[i]  <= lower
                if hit_up and hit_dn:
                    exit_p, reason = _tie_exit(lower, upper)
                elif hit_up:
                    exit_p = _apply_exit_slippage(float(upper)); reason = "tp_entry_day"
                elif hit_dn:
                    exit_p = _apply_exit_slippage(float(lower)); reason = "sl_entry_day"
                elif timeout_bar is not None and i >= timeout_bar:
                    exit_p = _apply_exit_slippage(float(closes[i])); reason = "timeout_entry_day"
                elif is_last:
                    exit_p = _apply_exit_slippage(float(closes[i])); reason = "forced_last_bar"

            if exit_p is not None:
                day_ret = (exit_p - ep) / ep - cfg.commission * 2
                daily_pnl.iloc[i] += day_ret
                _record(i, exit_p, reason or "exit_entry_day")
                _reset()
            else:
                daily_pnl.iloc[i] += (float(closes[i]) - ep) / ep - cfg.commission
                in_pos = True

            phase_a_acted = True

        # ── PHASE B: HOLD ──

        if in_pos and not phase_a_acted:
            mfe = max(mfe, (float(highs[i]) - entry_price) / entry_price)
            mae = min(mae, (float(lows[i])  - entry_price) / entry_price)

            prev_c  = float(closes[i-1]) if i > 0 else float(entry_price)
            exit_p  = reason = None
            is_last = force_exit_last_bar and (i == n - 1)

            if exit_mode == "barrier":
                hit_up = highs[i] >= upper
                hit_dn = lows[i]  <= lower
                if hit_up and hit_dn:
                    exit_p, reason = _tie_exit(lower, upper)
                elif hit_up:
                    exit_p = _apply_exit_slippage(float(upper)); reason = "take_profit"
                elif hit_dn:
                    exit_p = _apply_exit_slippage(float(lower)); reason = "stop_loss"
                elif timeout_bar is not None and i >= timeout_bar:
                    exit_p = _apply_exit_slippage(float(closes[i])); reason = "timeout"
                elif is_last:
                    exit_p = _apply_exit_slippage(float(closes[i])); reason = "forced_last_bar"
            elif exit_mode == "flip" and is_last:
                exit_p = _apply_exit_slippage(float(closes[i])); reason = "forced_last_bar"

            if exit_p is not None:
                daily_pnl.iloc[i] += (exit_p - prev_c) / prev_c - cfg.commission
                _record(i, exit_p, reason or "exit")
                _reset()
            else:
                daily_pnl.iloc[i] += (float(closes[i]) - prev_c) / prev_c

        # ── PHASE C: CLOSE ──

        if in_pos and exit_mode == "flip" and flip_exit_for is None:
            if i + 1 < n and signal_series.iloc[i] == 0:
                flip_exit_for = i + 1

        if not in_pos and entry_sched_for is None:
            if i + 1 < n and signal_series.iloc[i] == 1:
                entry_sched_for = i + 1
                signal_idx      = i

    return daily_pnl, trades


# ============================================================
# BUY & HOLD BASELINE
# ============================================================

def bah_baseline(df: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, List[Dict]]:
    opens  = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    dates  = df.index
    n      = len(df)

    pnl         = pd.Series(0.0, index=dates)
    entry_price = float(opens[0]) * (1.0 + cfg.entry_slippage)

    pnl.iloc[0] = (float(closes[0]) - entry_price) / entry_price - cfg.commission
    for i in range(1, n - 1):
        pnl.iloc[i] = (float(closes[i]) - float(closes[i-1])) / float(closes[i-1])
    if n > 1:
        exit_p = float(closes[n-1]) * (1.0 - cfg.exit_slippage)
        pnl.iloc[n-1] = (exit_p - float(closes[n-2])) / float(closes[n-2]) - cfg.commission

    trades = [{
        "signal_date": dates[0], "entry_date": dates[0], "exit_date": dates[n-1],
        "entry_price": entry_price, "exit_price": float(closes[n-1]) * (1 - cfg.exit_slippage),
        "exit_reason": "end_of_period", "hold_days": n,
        "gross_ret": (float(closes[n-1]) - entry_price) / entry_price,
        "net_ret":   (float(closes[n-1]) * (1-cfg.exit_slippage) - entry_price) / entry_price
                     - cfg.commission * 2,
        "mae": np.nan, "mfe": np.nan, "mae_r": np.nan, "mfe_r": np.nan,
        "overnight_gap": 0.0, "signal_prob": np.nan, "fold_id": -1,
        "atr_at_signal": np.nan, "label_truth": np.nan, "threshold": np.nan,
        "strategy": "buy_and_hold", "year": int(dates[0].year), "regime_at_sig": -1,
    }]
    return pnl, trades


# ============================================================
# ENGINE 3: METRICS ENGINE
# T3-9: Turnover metric added.
# T3-11: Bootstrap CI for Sharpe and Expectancy.
# ============================================================

def _bootstrap_ci(
    data: np.ndarray,
    stat_fn,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Tuple[float, float]:
    """
    Bootstrap CI — iid resampling. Rough heuristic only.
    Daily returns are autocorrelated; treat output as directional, not rigorous.
    """
    if len(data) == 0:
        return np.nan, np.nan   # Fix 8: guard against empty input
    rng      = np.random.default_rng(seed)
    boots    = [stat_fn(rng.choice(data, size=len(data), replace=True))
                for _ in range(n_bootstrap)]
    lo_pct   = (1 - ci) / 2 * 100
    hi_pct   = (1 + ci) / 2 * 100
    return float(np.percentile(boots, lo_pct)), float(np.percentile(boots, hi_pct))


def metrics_engine(
    daily_pnl: pd.Series,
    trades: List[Dict[str, Any]],
    label: str,
    cfg: Config,
    calib_probs: Optional[np.ndarray] = None,
    calib_labels: Optional[np.ndarray] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    r   = daily_pnl.fillna(0.0)
    ann = np.sqrt(252)

    std     = r.std()
    sharpe  = float(r.mean() / (std + 1e-12) * ann) if std > 0 else 0.0
    neg     = r[r < 0]
    dstd    = neg.std()
    sortino = float(r.mean() / (dstd + 1e-12) * ann) if dstd > 0 else 0.0
    cum     = (1 + r).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    max_dd  = float(dd.min())
    n_yrs   = max(len(r) / 252.0, 1e-9)
    cagr    = float(cum.iloc[-1] ** (1.0 / n_yrs) - 1.0)
    calmar  = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    td = pd.DataFrame(trades) if trades else pd.DataFrame()
    if not td.empty and "net_ret" in td.columns:
        rets      = td["net_ret"]
        wins      = rets[rets > 0]
        losses    = rets[rets < 0]
        pf        = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 999.0
        wr        = float((rets > 0).mean())
        exp       = float(rets.mean())
        avg_h     = float(td["hold_days"].mean())
        # Accurate calendar exposure
        exposure  = float(td["hold_days"].sum() / len(r))
        by_reason = td["exit_reason"].value_counts().to_dict()
        avg_mae_r = float(td["mae_r"].mean()) if "mae_r" in td.columns else np.nan
        avg_mfe_r = float(td["mfe_r"].mean()) if "mfe_r" in td.columns else np.nan
        avg_gap   = float(td["overnight_gap"].mean()) if "overnight_gap" in td.columns else np.nan
        # T3-9: Turnover = round-trips per year
        n_trades  = int(len(td))
        turnover  = float(n_trades / n_yrs)

        # T3-11: Bootstrap CIs — WARNING: iid bootstrap ignores time dependence
        # and volatility clustering. Treat as rough heuristic only, not
        # statistically rigorous inference. Use block bootstrap for tighter bounds.
        rets_arr = rets.to_numpy()
        if len(rets_arr) >= 10:
            sharpe_daily = r.to_numpy()
            sharpe_lo, sharpe_hi = _bootstrap_ci(
                sharpe_daily,
                lambda x: x.mean()/(x.std()+1e-12)*ann,
                cfg.n_bootstrap, cfg.bootstrap_ci, cfg.random_seed)
            exp_lo, exp_hi = _bootstrap_ci(
                rets_arr, np.mean,
                cfg.n_bootstrap, cfg.bootstrap_ci, cfg.random_seed)
        else:
            sharpe_lo = sharpe_hi = exp_lo = exp_hi = np.nan
    else:
        pf=wr=exp=avg_h=exposure=avg_mae_r=avg_mfe_r=avg_gap=turnover=0.0
        by_reason={}; n_trades=0
        sharpe_lo=sharpe_hi=exp_lo=exp_hi=np.nan

    brier = None
    if (calib_probs is not None and calib_labels is not None
            and len(calib_probs) >= 10
            and len(calib_probs) == len(calib_labels)):
        brier = float(brier_score_loss(calib_labels, calib_probs))

    return dict(
        label=label,
        daily_pnl=r, cum=cum, trade_df=td,
        sharpe=sharpe, sortino=sortino, max_dd=max_dd,
        calmar=calmar, cagr=cagr,
        profit_factor=pf, win_rate=wr, expectancy=exp,
        avg_hold=avg_h, exposure=exposure, n_trades=n_trades,
        turnover=turnover,   # T3-9
        by_reason=by_reason,
        avg_mae_r=avg_mae_r, avg_mfe_r=avg_mfe_r, avg_gap=avg_gap,
        brier=brier,
        # T3-11: bootstrap CIs
        sharpe_ci=(sharpe_lo, sharpe_hi),
        exp_ci=(exp_lo, exp_hi),
        calib_probs=calib_probs, calib_labels=calib_labels,
        pred_probs=pred_probs, pred_labels=pred_labels,
    )


# ============================================================
# BASELINES
# T1-2: baselines now accept an explicit df window.
#        Caller restricts to OOS-live window for fair comparison.
# ============================================================

def run_baselines(df: pd.DataFrame, cfg: Config, label_prefix: str) -> Dict[str, Dict]:
    print("\n" + "="*60)
    print(f"BASELINES [{label_prefix}]")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print("="*60)

    results: Dict[str, Dict] = {}

    bah_pnl, bah_tr = bah_baseline(df, cfg)
    results["Buy & Hold SPY (open[0])"] = metrics_engine(
        bah_pnl, bah_tr, "Buy & Hold SPY (open[0])", cfg)

    mom_sig = (df["ret_10d"].shift(1) > 0).astype(int).fillna(0)
    pnl, tr = execution_engine(df, mom_sig, cfg, exit_mode="barrier",
                               strategy_name="10d_mom|barrier")
    results["10d Momentum | Barrier"] = metrics_engine(pnl, tr, "10d Momentum | Barrier", cfg)

    ma_sig = (df["ma_cross"].shift(1) == 1).astype(int).fillna(0)
    pnl, tr = execution_engine(df, ma_sig, cfg, exit_mode="barrier",
                               strategy_name="50/200_ma|barrier")
    results["50/200 MA | Barrier"] = metrics_engine(pnl, tr, "50/200 MA | Barrier", cfg)

    pnl, tr = execution_engine(df, mom_sig, cfg, exit_mode="flip",
                               strategy_name="10d_mom|flip")
    results["10d Momentum | Flip"] = metrics_engine(pnl, tr, "10d Momentum | Flip", cfg)

    # Random barrier baseline
    rng      = np.random.default_rng(cfg.random_seed)
    rand_sig = pd.Series(rng.integers(0, 2, size=len(df)).astype(int), index=df.index)
    pnl, tr  = execution_engine(df, rand_sig, cfg, exit_mode="barrier",
                                strategy_name="random|barrier")
    results["Random | Barrier"] = metrics_engine(pnl, tr, "Random | Barrier", cfg)

    # T3-12: Random Timing baseline — same signal count as momentum,
    # randomly placed. Tests whether timing matters, not just signal frequency.
    # Note: matches RAW signal count, not realized-trade count (last bar cannot
    # generate an entry; some signals may be blocked by open position).
    # Interpret as: "same-frequency random long-only entries, random timing."
    real_sig_count = int(mom_sig.sum())
    null_arr   = np.zeros(len(df), dtype=int)
    null_idx   = rng.choice(len(df), size=real_sig_count, replace=False)
    null_arr[null_idx] = 1
    null_sig   = pd.Series(null_arr, index=df.index)
    pnl, tr    = execution_engine(df, null_sig, cfg, exit_mode="barrier",
                                  strategy_name="random_timing_matched")
    results["Random Timing (Matched Count)"] = metrics_engine(
        pnl, tr, "Random Timing (Matched Count)", cfg)

    print(f"\n  {'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>8} {'CAGR':>7} {'Trades':>7} {'Turn/yr':>8}")
    print("  " + "-"*80)
    for m in results.values():
        print(f"  {m['label']:<30} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']:>7.1%} {m['cagr']:>6.1%} {m['n_trades']:>7} "
              f"{m['turnover']:>7.1f}")

    return results


# ============================================================
# PURGED WALK-FORWARD — PROBABILITY GENERATION
# Probabilities computed ONCE. Threshold sweep is cheap (reuse stored probs).
# ============================================================

def purged_walk_forward_probabilities(
    df_dev: pd.DataFrame,
    feat_cols: List[str],
    cfg: Config,
    model_type: str,
) -> Dict[str, Any]:
    dates = df_dev.index
    X_all = df_dev[feat_cols].to_numpy()
    y_all = df_dev["label"].to_numpy()

    prob_series = pd.Series(np.nan, index=dates)
    fold_series = pd.Series(-1,     index=dates, dtype=int)
    fold_ranges: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []

    fold_starts = pd.date_range(
        start=dates[0] + pd.DateOffset(years=cfg.train_years),
        end=dates[-1],
        freq=f"{cfg.step_months}MS",
    )

    for fold_id, fold_start in enumerate(fold_starts):
        fold_end   = fold_start + pd.DateOffset(months=cfg.step_months)
        train_mask = ((dates >= fold_start - pd.DateOffset(years=cfg.train_years)) &
                      (dates <  fold_start))
        test_mask  = (dates >= fold_start) & (dates < fold_end)

        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]

        if len(train_idx) < 250 or len(test_idx) < 5:
            continue
        if len(train_idx) <= cfg.timeout_bars:
            continue

        # Purge last TIMEOUT_BARS from training
        train_idx = train_idx[:-cfg.timeout_bars]
        if len(train_idx) < 200:
            continue

        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te        = X_all[test_idx]

        if model_type == "logistic":
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        model = build_model(model_type, cfg)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]

        prob_series.iloc[test_idx] = probs
        fold_series.iloc[test_idx] = fold_id
        fold_ranges.append((fold_start, fold_end, fold_id))

    valid        = prob_series.notna().to_numpy()
    calib_probs  = prob_series.to_numpy()[valid]
    calib_labels = df_dev["label"].to_numpy()[valid]

    # T2-7: OOS coverage metrics
    first_live = prob_series.first_valid_index()
    last_live  = prob_series.last_valid_index()
    coverage   = float(prob_series.notna().mean())

    print(f"\n  [{model_type.upper()}] OOS Coverage (T2-7):")
    if first_live is not None:
        print(f"    First live date: {first_live.date()}")
        print(f"    Last live date:  {last_live.date()}")
    print(f"    Coverage:        {coverage:.1%} of dev period has predictions")
    if coverage < 0.50:
        print("  ⚠ Less than 50% of dev has OOS predictions — "
              "check train_years vs dev length")

    return {
        "prob_series":   prob_series,
        "fold_series":   fold_series,
        "fold_ranges":   fold_ranges,
        "calib_probs":   calib_probs,
        "calib_labels":  calib_labels,
        "model_type":    model_type,
        "first_live":    first_live,
        "last_live":     last_live,
        "coverage":      coverage,
    }


# ============================================================
# EVALUATE THRESHOLD
# T1-1: Evaluates on OOS-live window only (first valid prediction onward).
# T1-2: Caller restricts baselines to the same window.
# ============================================================

def evaluate_threshold(
    df_dev: pd.DataFrame,
    prob_pack: Dict[str, Any],
    cfg: Config,
    threshold: float,
    eval_start: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Evaluates on OOS-live window only.
    FIX-1: eval_start param forces a common start date across all dev models.
    If eval_start is later than the model's own first_live, eval_start wins.
    This ensures lr_m and lgbm_m metrics are on identical calendar windows
    so the dev comparison table is not mixing different effective periods.
    """
    prob_series = prob_pack["prob_series"]
    fold_series = prob_pack["fold_series"]
    model_type  = prob_pack["model_type"]

    first_live = prob_pack.get("first_live")
    if first_live is None:
        raise ValueError("No OOS predictions available. Check train_years vs dev length.")

    # Align all models to the same evaluation start so threshold selection
    # and model comparison use identical calendar windows.
    # When eval_start > first_live, earlier valid OOS predictions are intentionally
    # discarded in favour of cross-model comparability.
    start_ts = first_live
    if eval_start is not None:
        start_ts = max(first_live, eval_start)

    df_eval    = df_dev.loc[start_ts:].copy()
    prob_eval  = prob_series.loc[start_ts:]
    fold_eval  = fold_series.loc[start_ts:]
    label_eval = df_dev["label"].loc[start_ts:]

    signal_series = (prob_eval > threshold).fillna(False).astype(int)

    daily_pnl, trades = execution_engine(
        df_eval, signal_series, cfg,
        exit_mode="barrier", force_exit_last_bar=True,
        prob_series=prob_eval, fold_series=fold_eval, label_series=label_eval,
        threshold=threshold,
        strategy_name=f"{model_type}|thresh={threshold}",
    )

    # Fix 8: Calibration arrays restricted to eval window for explicit alignment.
    # calib_probs from prob_pack covers all OOS periods — same as eval window
    # since we start at first_live. Kept consistent here for clarity.
    eval_valid  = prob_eval.notna().to_numpy()
    eval_cp     = prob_eval.to_numpy()[eval_valid]
    eval_cl     = label_eval.to_numpy()[eval_valid]

    m = metrics_engine(
        daily_pnl, trades,
        label=f"{model_type.upper()} | thresh={threshold}",
        cfg=cfg,
        calib_probs=eval_cp,
        calib_labels=eval_cl,
        pred_probs=eval_cp,
        pred_labels=eval_cl,
    )

    # Fold diagnostics from stitched OOS equity curve
    # FIX-5: Skip folds with zero days in eval window (can happen when
    # common eval_start is later than some fold's end date).
    fold_rows = []
    for fold_start, fold_end, fold_id in prob_pack["fold_ranges"]:
        mask = (daily_pnl.index >= fold_start) & (daily_pnl.index < fold_end)
        if mask.sum() == 0:
            continue   # FIX-5: filter empty folds
        fr   = daily_pnl.loc[mask]
        fold_rows.append({
            "fold":    fold_start.strftime("%Y-%m"),
            "sharpe":  _annualize_sharpe(fr),
            "signals": int(signal_series.reindex(fr.index).fillna(0).sum()),
            "days":    int(mask.sum()),
        })

    fd = pd.DataFrame(fold_rows)
    m["fold_df"]    = fd
    m["pos_folds"]  = float((fd["sharpe"] > 0).mean()) if not fd.empty else 0.0
    m["oos_window"] = (start_ts, prob_pack.get("last_live"))
    return m


# ============================================================
# SWEEP THRESHOLDS FROM PROB PACK  (Fixes 1, 2, 3, 5)
# Top-level function — testable, reusable, not buried in main().
# Returns (best_thresh, sweep_df, prob_pack) for full audit trail.
# Fix 3: Gracefully skips if prob_pack has no first_live.
# ============================================================

def sweep_thresholds_from_prob_pack(
    df_dev: pd.DataFrame,
    prob_pack: Optional[Dict[str, Any]],
    cfg: Config,
    eval_start: Optional[pd.Timestamp] = None,
    out_dir: Optional[str] = None,
    run_tag: str = "",
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep all configured thresholds on a pre-computed prob_pack.
    Returns (best_thresh, sweep_df) — prob_pack NOT returned (caller already has it).

    Fix 3: Gracefully skips if prob_pack is None or has no first_live.
    Fix 2: Saves timestamped CSV audit trail if out_dir is provided.
    Fix 4: run_tag prefixes output filenames to avoid overwriting between runs.
    """
    if prob_pack is None:
        print("  [WARN] prob_pack is None — skipping threshold sweep.")
        return cfg.thresholds[0], pd.DataFrame()

    # Fix 3: Guard for model with no valid OOS predictions
    if prob_pack.get("first_live") is None:
        model_type_str = prob_pack.get("model_type", "unknown").upper()
        print(f"  [WARN] {model_type_str} has no valid OOS predictions. "
              f"Check train_years vs dev length. Skipping.")
        return cfg.thresholds[0], pd.DataFrame()

    model_type_str = prob_pack.get("model_type", "model").upper()
    # Fix 7: consistent string type — no mixed date/str
    window_str = (eval_start.strftime("%Y-%m-%d")
                  if eval_start is not None else "model-specific")

    print(f"\n  ── THRESHOLD SWEEP: {model_type_str} ──")
    print(f"  Evaluation window start: {window_str}")
    print(f"  Composite validity: trades>={cfg.min_trades}, "
          f"PF>{cfg.min_pf}, exposure>{cfg.min_exposure:.0%}")
    print(f"\n  {'Thresh':>8} {'Sharpe':>8} {'Trades':>8} "
          f"{'PF':>7} {'Expect':>9} {'Exposure':>9} {'Valid':>8}")
    print("  " + "-"*64)

    rows        = []
    best_thresh = cfg.thresholds[0]
    best_sharpe = -np.inf

    for t in cfg.thresholds:
        m = evaluate_threshold(df_dev, prob_pack, cfg, t, eval_start=eval_start)
        valid = (m["n_trades"]      >= cfg.min_trades and
                 m["profit_factor"] >  cfg.min_pf     and
                 m["exposure"]      >  cfg.min_exposure)
        flag = "✓" if valid else (
            f"✗ tr<{cfg.min_trades}" if m["n_trades"] < cfg.min_trades else
            f"✗ PF<{cfg.min_pf}"     if m["profit_factor"] <= cfg.min_pf else
            f"✗ exp<{cfg.min_exposure:.0%}"
        )
        rows.append({
            "threshold": t, "sharpe": m["sharpe"],
            "trades": m["n_trades"], "pf": m["profit_factor"],
            "expectancy": m["expectancy"], "exposure": m["exposure"],
            "valid": valid,
        })
        print(f"  {t:>8.2f} {m['sharpe']:>8.2f} {m['n_trades']:>8} "
              f"{m['profit_factor']:>7.2f} {m['expectancy']:>9.4f} "
              f"{m['exposure']:>8.1%} {flag:>8}")
        if valid and m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_thresh = t

    if best_sharpe == -np.inf:
        print(f"  ⚠ No threshold passed composite validity — "
              f"defaulting to {cfg.thresholds[0]}")
        best_thresh = cfg.thresholds[0]

    print(f"\n  Best: threshold={best_thresh}  Sharpe={best_sharpe:.2f}")

    sweep_df = pd.DataFrame(rows)

    # Fix 2+4: Save sweep CSV with timestamped run_tag prefix
    if out_dir is not None:
        model_name = prob_pack.get("model_type", "model")
        prefix     = f"{run_tag}_" if run_tag else ""
        csv_path   = os.path.join(out_dir, f"{prefix}{model_name}_threshold_sweep.csv")
        sweep_df.to_csv(csv_path, index=False)
        print(f"  Sweep table saved: {csv_path}")

    # Fix 5: prob_pack NOT returned — caller already holds the reference
    return best_thresh, sweep_df


# ============================================================
# REPORTING HELPERS
# ============================================================

def print_metrics(m: Dict[str, Any], baselines: Optional[Dict] = None) -> None:
    ows = m.get("oos_window")
    if ows and ows[0] is not None:
        print(f"\n  [OOS window: {ows[0].date()} → {ows[1].date()}]")

    print(f"\n  ── {m['label'].upper()} ──")
    print(f"  Portfolio (daily equity curve):")
    print(f"    Sharpe:        {m['sharpe']:.2f}  "
          f"95%CI [{_safe_fmt(m['sharpe_ci'][0], '{:.2f}')} – "
          f"{_safe_fmt(m['sharpe_ci'][1], '{:.2f}')}] (iid bootstrap — heuristic only)")
    print(f"    Sortino:       {m['sortino']:.2f}")
    print(f"    Max Drawdown:  {m['max_dd']:.1%}")
    print(f"    Calmar:        {m['calmar']:.2f}")
    print(f"    CAGR:          {m['cagr']:.1%}")
    print(f"  Trade stats:")
    print(f"    Trades:        {m['n_trades']}")
    print(f"    Turnover:      {m['turnover']:.1f} round-trips/year")
    print(f"    Win rate:      {m['win_rate']:.1%}  (side metric)")
    print(f"    Profit factor: {m['profit_factor']:.2f}")
    print(f"    Expectancy:    {m['expectancy']:.4f}  "
          f"95%CI [{_safe_fmt(m['exp_ci'][0], '{:.4f}')} – "
          f"{_safe_fmt(m['exp_ci'][1], '{:.4f}')}] (iid bootstrap — heuristic only)")
    print(f"    Avg hold:      {m['avg_hold']:.1f} days")
    print(f"    Exposure:      {m['exposure']:.1%}")
    print(f"    Exit reasons:  {m['by_reason']}")
    print(f"  Excursions (risk units):")
    print(f"    Avg MAE:       {_safe_fmt(m['avg_mae_r'], '{:.2f}R')}")
    print(f"    Avg MFE:       {_safe_fmt(m['avg_mfe_r'], '{:.2f}R')}")
    print(f"    Avg o/n gap:   {_safe_fmt(m['avg_gap'], '{:.3%}')}")
    print(f"  Brier score:     {_safe_fmt(m['brier'], '{:.4f}')}")

    print(f"\n  ── PASS / FAIL ──")
    checks = {
        "Sharpe ≥ 0.8  (iterate)": m["sharpe"] >= 0.8,
        "Sharpe ≥ 1.0  (deploy)":  m["sharpe"] >= 1.0,
        "Sortino ≥ 1.2":           m["sortino"] >= 1.2,
        "Max DD < 20%":            m["max_dd"] > -0.20,
        "Calmar ≥ 0.5":            m["calmar"] >= 0.5,
        "Profit Factor ≥ 1.2":     m["profit_factor"] >= 1.2,
        "Expectancy > 0":          m["expectancy"] > 0,
        # FIX-6: iid bootstrap CI is a rough heuristic — not rigorous inference.
        # Labeled explicitly to avoid overconfidence. Does not count toward
        # PASS/FAIL verdict (informational only via separate display below).
    }
    # T2-5: Explicit key lookup — no brittle dict-order assumption
    if baselines:
        bah = baselines.get("Buy & Hold SPY (open[0])")
        if bah:
            checks[f"Beats B&H ({bah['sharpe']:.2f})"] = m["sharpe"] > bah["sharpe"]

    for chk, passed in checks.items():
        print(f"  {'✓' if passed else '✗'} {chk}")

    # FIX-6: Bootstrap CI printed as informational — not a hard pass/fail gate.
    ci_lo = m["sharpe_ci"][0]
    ci_hi = m["sharpe_ci"][1]
    ci_str = (f"[{ci_lo:.2f} – {ci_hi:.2f}]"
              if not np.isnan(ci_lo) else "[N/A]")
    ci_flag = "positive lower bound" if (not np.isnan(ci_lo) and ci_lo > 0) \
              else "lower bound ≤ 0 — weak signal"
    print(f"  ℹ Sharpe 95% CI (iid heuristic): {ci_str} — {ci_flag}")

    n_pass = sum(checks.values())
    if m["sharpe"] >= 1.0:
        print(f"\n  🟢 STRONG PASS ({n_pass}/{len(checks)})")
    elif m["sharpe"] >= 0.8:
        print(f"\n  🟡 BORDERLINE ({n_pass}/{len(checks)})")
    else:
        print(f"\n  🔴 FAIL ({n_pass}/{len(checks)})")


def print_fold_stability(m: Dict[str, Any]) -> None:
    fd = m.get("fold_df", pd.DataFrame())
    if fd.empty:
        return
    print(f"\n  ── FOLD STABILITY ──")
    print(f"  Positive-Sharpe folds: {m.get('pos_folds', 0):.0%}  (need > 60%)")
    print(fd.to_string(index=False))

    cp = m.get("calib_probs")
    cl = m.get("calib_labels")
    if cp is not None and cl is not None and len(cp) > 10:
        print(f"\n  ── CALIBRATION ──")
        try:
            fp, mp_ = calibration_curve(cl, cp, n_bins=10)
            for f_, p_ in zip(fp, mp_):
                bar  = "█" * int(f_ * 20)
                flag = " ⚠" if abs(f_ - p_) > 0.15 else ""
                print(f"    Predicted {p_:.2f} → Actual {f_:.2f}  {bar}{flag}")
        except Exception:
            pass


def insufficient_sample_warnings(m: Dict[str, Any], context: str) -> None:
    n_trades = m.get("n_trades", 0)
    exposure = m.get("exposure", 0.0)
    threshold = 25 if "dev" in context.lower() else 10

    warned = False
    if n_trades < threshold:
        print(f"  ⚠ [{context}] Only {n_trades} trades — statistically thin.")
        warned = True
    if exposure < 0.02:
        print(f"  ⚠ [{context}] Exposure={exposure:.1%} < 2% — very sparse.")
        warned = True
    if not warned:
        print(f"  ✓ [{context}] Sample adequate ({n_trades} trades, {exposure:.1%} exposure)")


def trades_by_year_and_regime(
    m: Dict[str, Any],
    label: str,
    baseline_by_year: Optional[Dict[int, float]] = None,
) -> None:
    """
    T3-10 FIX: Year table now compares yearly compounded strategy return
    against yearly compounded B&H return. Both are annual compounded returns —
    apples to apples. Previous version compared mean trade return vs annual B&H,
    which was apples vs a forklift.
    """
    td       = m.get("trade_df", pd.DataFrame())
    pnl_ser  = m.get("daily_pnl")
    if td.empty or "net_ret" not in td.columns or pnl_ser is None:
        return

    # Build yearly compounded strategy return from daily_pnl
    strat_by_year = _build_yearly_return(pnl_ser)

    # FIX-4: Detect partial first year — OOS window may start mid-calendar-year.
    first_date       = pnl_ser.index[0]
    first_year       = first_date.year
    is_partial_first = first_date.month > 1 or first_date.day > 1

    print(f"\n  ── TRADES BY YEAR [{label}] ──")
    if is_partial_first:
        print(f"  * {first_year} is a partial year "
              f"(OOS window starts {first_date.date()})")
    has_bah = baseline_by_year is not None
    hdr = f"  {'StratRet':>9}  {'B&H Ret':>8}" if has_bah else f"  {'StratRet':>9}"
    print(f"  {'Year':>7} {'Trades':>8} {'WinRate':>9} "
          f"{'PF':>7}{hdr}")
    print("  " + "-"*(41 + (20 if has_bah else 10)))
    for yr, g in td.groupby("year"):
        rets = g["net_ret"]
        wins = rets[rets > 0]
        loss = rets[rets < 0]
        pf   = wins.sum()/abs(loss.sum()) if loss.sum() != 0 else 999.0
        strat_ret = strat_by_year.get(int(yr), float("nan"))
        strat_col = f"  {_safe_fmt(strat_ret, '{:>8.1%}')}"
        bah_col   = ""
        if has_bah:
            bah_ret = baseline_by_year.get(int(yr), float("nan"))
            bah_col = f"  {_safe_fmt(bah_ret, '{:>7.1%}')}"
        # FIX-4: Mark partial first year with asterisk
        yr_label  = f"{yr}*" if (int(yr) == first_year and is_partial_first) else f"{yr} "
        print(f"  {yr_label:>7} {len(g):>8} {(rets>0).mean():>9.1%} "
              f"{pf:>7.2f}{strat_col}{bah_col}")

    if "regime_at_sig" in td.columns:
        regime_names = {0: "neutral", 1: "trending", 2: "high-vol"}
        print(f"\n  ── TRADES BY REGIME [{label}] ──")
        print(f"  {'Regime':<12} {'Trades':>8} {'WinRate':>9} {'MeanRet':>9} {'PF':>7}")
        print("  " + "-"*48)
        for reg, g in td.groupby("regime_at_sig"):
            rets  = g["net_ret"]
            wins  = rets[rets > 0]
            loss  = rets[rets < 0]
            pf    = wins.sum()/abs(loss.sum()) if loss.sum() != 0 else 999.0
            rname = regime_names.get(int(reg), str(reg))
            print(f"  {rname:<12} {len(g):>8} {(rets>0).mean():>9.1%} "
                  f"{rets.mean():>9.4f} {pf:>7.2f}")


def _build_bah_by_year(bah_pnl: pd.Series) -> Dict[int, float]:
    """T3-10: Per-year compounded B&H return for comparison in year tables."""
    result = {}
    for yr, g in bah_pnl.groupby(bah_pnl.index.year):
        result[int(yr)] = float((1 + g).prod() - 1)
    return result


def _build_yearly_return(daily_pnl: pd.Series) -> Dict[int, float]:
    """T3-10 FIX: Per-year compounded strategy return from daily_pnl.
    Comparable against yearly B&H return (both are compounded annual returns).
    Previous version compared mean-trade-return vs annual B&H — apples vs forklift.
    """
    result = {}
    for yr, g in daily_pnl.groupby(daily_pnl.index.year):
        result[int(yr)] = float((1 + g.fillna(0.0)).prod() - 1)
    return result


def probability_bucket_tables(m: Dict[str, Any], threshold: float) -> str:
    buckets = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.01)]
    blabels = ["0.50-0.55","0.55-0.60","0.60-0.65","0.65-0.70","0.70+"]
    lines   = []

    pp = m.get("pred_probs")
    pl = m.get("pred_labels")
    td = m.get("trade_df", pd.DataFrame())

    lines.append("  Table A — ALL predictions")
    lines.append(f"  {'Bucket':<12} {'Count':>7} {'HitRate':>9} "
                 f"{'AvgProb':>9} {'TradeCt':>8}")
    for (lo, hi), lbl in zip(buckets, blabels):
        if pp is None or len(pp) == 0:
            lines.append(f"  {lbl:<12} {'0':>7} {'—':>9} {'—':>9} {'—':>8}")
            continue
        mask  = (pp >= lo) & (pp < hi)
        count = int(mask.sum())
        if count == 0:
            lines.append(f"  {lbl:<12} {'0':>7} {'—':>9} {'—':>9} {'—':>8}")
            continue
        hit  = float(np.mean(pl[mask])) if pl is not None else np.nan
        avg  = float(np.mean(pp[mask]))
        tct  = int(np.sum(pp[mask] > threshold))
        hit_str = f"{hit:9.1%}" if not np.isnan(hit) else "       —"
        lines.append(f"  {lbl:<12} {count:>7} {hit_str} {avg:>9.3f} {tct:>8}")

    lines.append("")
    lines.append("  Table B — EXECUTED trades")
    lines.append(f"  {'Bucket':<12} {'Count':>7} {'HitRate':>9} "
                 f"{'MeanRet':>9} {'PF':>7}")
    for (lo, hi), lbl in zip(buckets, blabels):
        if td.empty or "signal_prob" not in td.columns:
            lines.append(f"  {lbl:<12} {'0':>7} {'—':>9} {'—':>9} {'—':>7}")
            continue
        sub   = td[(td["signal_prob"] >= lo) & (td["signal_prob"] < hi)]
        count = len(sub)
        if count == 0:
            lines.append(f"  {lbl:<12} {'0':>7} {'—':>9} {'—':>9} {'—':>7}")
            continue
        rets  = sub["net_ret"]
        wins  = rets[rets > 0]
        loss  = rets[rets < 0]
        hit   = float(sub["label_truth"].mean()) \
                if "label_truth" in sub.columns else float((rets > 0).mean())
        pf    = wins.sum()/abs(loss.sum()) if loss.sum() != 0 else 999.0
        lines.append(f"  {lbl:<12} {count:>7} {hit:>9.1%} "
                     f"{rets.mean():>9.4f} {pf:>7.2f}")

    return "\n".join(lines)


def dev_vs_test_summary(
    lr_dev: Dict[str, Any],
    lgbm_dev: Optional[Dict[str, Any]],
    final_results: Optional[Dict[str, Any]],
) -> None:
    print("\n" + "="*60)
    print("DEV vs TEST SUMMARY")
    ows = lr_dev.get("oos_window")
    if ows and ows[0]:
        print(f"  Dev  = OOS-live window only: {ows[0].date()} → {ows[1].date()}")
    print(f"  Test = 2024 final holdout (one shot, never touched before)")
    print("="*60)

    lr_test   = (final_results or {}).get("logistic")
    lgbm_test = (final_results or {}).get("lgbm")

    print(f"\n  {'Metric':<24} {'LR Dev':>10} {'LR Test':>10} "
          f"{'LGBM Dev':>10} {'LGBM Test':>10}")
    print("  " + "-"*66)

    rows_def = [
        ("Sharpe",          "sharpe",        "{:>10.2f}"),
        ("Sharpe CI low",   None,            None),   # special
        ("Sortino",         "sortino",        "{:>10.2f}"),
        ("Max Drawdown",    "max_dd",         "{:>10.1%}"),
        ("Calmar",          "calmar",         "{:>10.2f}"),
        ("CAGR",            "cagr",           "{:>10.1%}"),
        ("Profit Factor",   "profit_factor",  "{:>10.2f}"),
        ("Expectancy",      "expectancy",     "{:>10.4f}"),
        ("Exp CI low",      None,            None),   # special
        ("Trades",          "n_trades",       "{:>10}"),
        ("Turnover/yr",     "turnover",       "{:>10.1f}"),
        ("Exposure",        "exposure",       "{:>10.1%}"),
        ("Brier Score",     "brier",          "{:>10.4f}"),
    ]

    def fv(d: Optional[Dict], key: str, fmt: str) -> str:
        if d is None: return "       N/A"
        v = d.get(key)
        return _safe_fmt(v, fmt, "       N/A")

    for name, key, fmt in rows_def:
        if key is None:
            # Bootstrap CI rows
            ci_key = "sharpe_ci" if "Sharpe" in name else "exp_ci"
            def ci_lo(d):
                if d is None: return "       N/A"
                ci = d.get(ci_key, (np.nan, np.nan))
                return _safe_fmt(ci[0], "{:>10.4f}" if "Exp" in name else "{:>10.2f}",
                                 "       N/A")
            print(f"  {name:<24}{ci_lo(lr_dev)}{ci_lo(lr_test)}"
                  f"{ci_lo(lgbm_dev)}{ci_lo(lgbm_test)}")
        else:
            print(f"  {name:<24}{fv(lr_dev,key,fmt)}{fv(lr_test,key,fmt)}"
                  f"{fv(lgbm_dev,key,fmt)}{fv(lgbm_test,key,fmt)}")

    if lr_test or lgbm_test:
        sharpes = [m["sharpe"] for m in [lr_test, lgbm_test] if m]
        best_sh = max(sharpes) if sharpes else float("nan")
        print("\n  ── TEST VERDICT ──")
        if best_sh >= 1.0:
            print(f"  🟢 STRONG PASS on 2024 holdout (best Sharpe={best_sh:.2f})")
        elif best_sh >= 0.8:
            print(f"  🟡 BORDERLINE on 2024 holdout (best Sharpe={best_sh:.2f})")
        else:
            print(f"  🔴 FAIL on 2024 holdout (best Sharpe={best_sh:.2f})")
        print("  2024 is the only number that matters. Dev was model selection only.")
        print("  Note: holdout opened conditionally (dev Sharpe >= 0.8).")
        print("  Stricter protocol: always evaluate holdout regardless of dev result.")


# ============================================================
# FINAL TEST — 2024 HOLDOUT
# ============================================================

def run_final_test(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    feat_cols: List[str],
    cfg: Config,
    best_thresh_lr: float,
    best_thresh_lgbm: Optional[float],   # Fix 1: None if LGBM had no valid dev sweep
    baselines_test: Dict[str, Dict],
) -> Dict[str, Any]:
    print("\n" + "█"*60)
    print("  FINAL TEST — 2024 HOLDOUT")
    print("  Trained on full df_dev. Evaluated on sealed 2024.")
    lgbm_in_test = best_thresh_lgbm is not None
    print(f"  LightGBM in test: {lgbm_in_test} "
          f"({'valid dev sweep' if lgbm_in_test else 'skipped — no valid dev sweep'})")
    print("█"*60)

    X_dev  = df_dev[feat_cols].to_numpy()
    y_dev  = df_dev["label"].to_numpy()
    X_test = df_test[feat_cols].to_numpy()
    y_test = df_test["label"].to_numpy()

    results: Dict[str, Any] = {}
    # Fix 1: Gate on whether LGBM had a valid dev sweep, not just HAS_LGBM.
    # best_thresh_lgbm is None when LGBM was skipped in dev — do not run
    # it on the holdout with a borrowed LogReg threshold.
    models_to_run = [("logistic", best_thresh_lr)]
    if best_thresh_lgbm is not None:
        models_to_run.append(("lgbm", best_thresh_lgbm))
        print(f"  LightGBM included — valid dev threshold: {best_thresh_lgbm}")
    else:
        print("  [INFO] LightGBM skipped in final test — no valid dev sweep.")

    # T2-5: get B&H for test comparison
    bah_test = baselines_test.get("Buy & Hold SPY (open[0])")
    bah_by_year_test = _build_bah_by_year(bah_test["daily_pnl"]) if bah_test else None

    for model_type, thresh in models_to_run:
        X_tr = X_dev.copy()
        X_te = X_test.copy()

        if model_type == "logistic":
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        model = build_model(model_type, cfg)
        model.fit(X_tr, y_dev)
        probs = model.predict_proba(X_te)[:, 1]

        sig  = pd.Series((probs > thresh).astype(int), index=df_test.index)
        pser = pd.Series(probs, index=df_test.index)

        pnl, trades = execution_engine(
            df_test, sig, cfg,
            exit_mode="barrier", force_exit_last_bar=True,
            prob_series=pser, label_series=df_test["label"],
            threshold=thresh,
            strategy_name=f"{model_type}_final_test",
        )

        m = metrics_engine(
            pnl, trades,
            label=f"{model_type.upper()} FINAL TEST (2024)",
            cfg=cfg,
            calib_probs=probs, calib_labels=y_test,
            pred_probs=probs,  pred_labels=y_test,
        )

        print_metrics(m, baselines=baselines_test)
        insufficient_sample_warnings(m, f"{model_type} TEST")
        trades_by_year_and_regime(m, f"{model_type} TEST",
                                  baseline_by_year=bah_by_year_test)
        print("\n" + probability_bucket_tables(m, thresh))
        results[model_type] = m

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_results(
    dev_models: List[Dict[str, Any]],
    baselines: Dict[str, Dict],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Trading Research Engine — SPY (OOS-live window)",
                 fontsize=12, fontweight="bold")
    COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    ax = axes[0, 0]
    # T2-5: explicit key lookup
    bah = baselines.get("Buy & Hold SPY (open[0])")
    if bah:
        ax.plot(bah["cum"].values, color="gray", lw=1.5, alpha=0.6, label="Buy&Hold")
    for i, m in enumerate(dev_models):
        ax.plot(m["cum"].values, color=COLORS[i % len(COLORS)],
                lw=1.5, label=m["label"][:24])
    ax.set_title("Equity Curve (OOS-live window)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i, m in enumerate(dev_models):
        rs = m["daily_pnl"].rolling(90).apply(
            lambda x: x.mean()/(x.std()+1e-12)*np.sqrt(252), raw=True)
        ax.plot(rs.values, color=COLORS[i % len(COLORS)], lw=1.2,
                label=m["label"][:24])
    ax.axhline(1.0, color="green", ls="--", alpha=0.5)
    ax.axhline(0.0, color="red",   ls="--", alpha=0.4)
    ax.set_title("Rolling 90d Sharpe"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Perfect calibration")
    for i, m in enumerate(dev_models):
        cp = m.get("calib_probs")
        cl = m.get("calib_labels")
        if cp is not None and cl is not None and len(cp) > 10:
            try:
                fp, mp_ = calibration_curve(cl, cp, n_bins=10)
                bs = _safe_fmt(m["brier"], "{:.3f}")
                ax.plot(mp_, fp, "s-", color=COLORS[i % len(COLORS)], lw=1.5,
                        label=f"{m['label'][:18]} B={bs}")
            except Exception:
                pass
    ax.set_title("Calibration"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i, m in enumerate(dev_models):
        td = m["trade_df"]
        if not td.empty and "net_ret" in td.columns:
            ax.hist(td["net_ret"], bins=30, alpha=0.55,
                    color=COLORS[i % len(COLORS)], edgecolor="white",
                    label=m["label"][:18])
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_title("Trade Return Distribution"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for i, m in enumerate(dev_models):
        td = m["trade_df"]
        if not td.empty and "mae_r" in td.columns:
            ax.scatter(td["mae_r"], td["mfe_r"], alpha=0.35, s=15,
                       color=COLORS[i % len(COLORS)], label=m["label"][:18])
    ax.axvline(-1, color="red",   ls="--", lw=0.8, label="Stop=-1R")
    ax.axhline(1,  color="green", ls="--", lw=0.8, label="Target=+1R")
    ax.set_title("MAE/MFE in Risk Units"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    for i, m in enumerate(dev_models):
        dd = (m["cum"] - m["cum"].cummax()) / m["cum"].cummax()
        ax.fill_between(range(len(dd)), dd.values, 0,
                        alpha=0.35, color=COLORS[i % len(COLORS)],
                        label=m["label"][:18])
    ax.set_title("Drawdown"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved: {out_path}")


# ============================================================
# CONFIG VALIDATION  (FIX-2, FIX-3)
# ============================================================

def validate_config(cfg: Config) -> None:
    """
    Reject nonsense configs before running anything.
    Catches: bad thresholds, negative costs, invalid ATR,
    unknown tie policy, etc.
    """
    # Thresholds
    if not cfg.thresholds:
        raise ValueError("Config: thresholds cannot be empty.")
    if any(not (0.0 < t < 1.0) for t in cfg.thresholds):
        raise ValueError(
            f"Config: all thresholds must be strictly in (0, 1). Got: {cfg.thresholds}")
    if tuple(sorted(cfg.thresholds)) != cfg.thresholds:
        raise ValueError(
            f"Config: thresholds must be sorted ascending. Got: {cfg.thresholds}")
    if len(set(cfg.thresholds)) != len(cfg.thresholds):
        raise ValueError(
            f"Config: thresholds must be unique. Got: {cfg.thresholds}")

    # Costs
    if cfg.entry_slippage < 0:
        raise ValueError(f"Config: entry_slippage must be >= 0. Got: {cfg.entry_slippage}")
    if cfg.exit_slippage < 0:
        raise ValueError(f"Config: exit_slippage must be >= 0. Got: {cfg.exit_slippage}")
    if cfg.commission < 0:
        raise ValueError(f"Config: commission must be >= 0. Got: {cfg.commission}")
    if cfg.entry_slippage > 0.10 or cfg.exit_slippage > 0.10:
        print("  [WARN] Slippage > 10% — that is unusually high. Verify config.")
    if cfg.commission > 0.05:
        print("  [WARN] Commission > 5% — that is unusually high. Verify config.")

    # ATR multipliers
    if cfg.atr_upper <= 0:
        raise ValueError(f"Config: atr_upper must be > 0. Got: {cfg.atr_upper}")
    if cfg.atr_lower <= 0:
        raise ValueError(f"Config: atr_lower must be > 0. Got: {cfg.atr_lower}")

    # FIX-3: Explicit tie policy validation — no silent fallback
    valid_tie_policies = {"worst_case", "best_case"}
    if cfg.intraday_tie_policy not in valid_tie_policies:
        raise ValueError(
            f"Config: intraday_tie_policy must be one of {valid_tie_policies}. "
            f"Got: '{cfg.intraday_tie_policy}'")

    # Threshold selection / validity params
    if cfg.min_pf <= 0:
        raise ValueError(f"Config: min_pf must be > 0. Got: {cfg.min_pf}")
    if not (0.0 <= cfg.min_exposure <= 1.0):
        raise ValueError(f"Config: min_exposure must be in [0, 1]. Got: {cfg.min_exposure}")

    # Bootstrap params
    if cfg.n_bootstrap < 1:
        raise ValueError(f"Config: n_bootstrap must be >= 1. Got: {cfg.n_bootstrap}")
    if not (0.0 < cfg.bootstrap_ci < 1.0):
        raise ValueError(f"Config: bootstrap_ci must be in (0, 1). Got: {cfg.bootstrap_ci}")

    # Horizon / training params
    if cfg.train_years < 1:
        raise ValueError(f"Config: train_years must be >= 1. Got: {cfg.train_years}")
    if cfg.step_months < 1:
        raise ValueError(f"Config: step_months must be >= 1. Got: {cfg.step_months}")
    if cfg.timeout_bars < 1:
        raise ValueError(f"Config: timeout_bars must be >= 1. Got: {cfg.timeout_bars}")
    if cfg.min_trades < 1:
        raise ValueError(f"Config: min_trades must be >= 1. Got: {cfg.min_trades}")

    # Date ordering
    try:
        ts_start   = pd.Timestamp(cfg.start)
        ts_dev_end = pd.Timestamp(cfg.dev_end)
        ts_end     = pd.Timestamp(cfg.end)
    except Exception as e:
        raise ValueError(f"Config: invalid date string — {e}")
    if not (ts_start < ts_dev_end < ts_end):
        raise ValueError(
            f"Config: require start < dev_end < end. "
            f"Got: start={cfg.start}, dev_end={cfg.dev_end}, end={cfg.end}")

    print("  ✓ Config validation passed.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ required.")

    cfg     = CFG
    out_dir = _ensure_dir(cfg.out_dir)

    # Fix A: run_tag defined here — before any file I/O — to avoid UnboundLocalError
    run_tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # A2: Print LightGBM status in run header
    print(f"\n  [CONFIG] out_dir: {os.path.abspath(out_dir)}")
    print(f"  [CONFIG] run_tag: {run_tag}")
    print(f"  [CONFIG] tie_policy: {cfg.intraday_tie_policy}")
    print(f"  [CONFIG] entry_slippage: {cfg.entry_slippage:.3%} | "
          f"exit_slippage: {cfg.exit_slippage:.3%}")
    print(f"  [CONFIG] LightGBM available: {HAS_LGBM}")

    # Validate config before any computation
    validate_config(cfg)

    # 1. Data (T1-3: end date gives buffer past Dec 31)
    df_raw = download_data(cfg)

    # 2. Features
    df_feat, feat_cols = engineer_features(df_raw)

    # 3. Split BEFORE labeling
    dev_end     = pd.Timestamp(cfg.dev_end)
    df_dev_raw  = df_feat[df_feat.index <= dev_end].copy()
    df_test_raw = df_feat[df_feat.index >  dev_end].copy()
    print(f"\n  Split before labeling:")
    print(f"  df_dev_raw:  {len(df_dev_raw)} rows  ({df_dev_raw.index[0].date()} → {df_dev_raw.index[-1].date()})")
    print(f"  df_test_raw: {len(df_test_raw)} rows (sealed)")

    # 4. Label independently per subset
    print("\n" + "="*60)
    print("ENGINE 1: LABELS (independent per subset — no cross-boundary leakage)")
    print("="*60)
    df_dev  = label_subset(df_dev_raw,  cfg, "DEV 2019-2023")
    df_test = label_subset(df_test_raw, cfg, "TEST 2024")
    df_dev.to_csv(os.path.join(out_dir, f"{run_tag}_spy_dev.csv"))
    df_test.to_csv(os.path.join(out_dir, f"{run_tag}_spy_test.csv"))

    # 5. Probability generation + threshold sweep.
    # Protocol:
    #   a) Generate raw probabilities for all models (one fit pass each).
    #   b) Compute common_start = max of all models' first_live dates.
    #   c) Sweep thresholds for all models using common_start.
    # Threshold selection and final comparison use the same window.
    print("\n" + "="*60)
    print("STEP 4: PROBABILITY GENERATION + THRESHOLD SWEEP")
    print("  common_start computed before sweep — consistent evaluation window.")
    print("="*60)

    # Step 4a: Generate raw probabilities
    lr_prob_pack   = purged_walk_forward_probabilities(df_dev, feat_cols, cfg, "logistic")
    lgbm_prob_pack = purged_walk_forward_probabilities(df_dev, feat_cols, cfg, "lgbm") \
                     if HAS_LGBM else None

    # Step 4b: Compute common_start from all available prob_packs
    candidate_starts = [p.get("first_live")
                        for p in [lr_prob_pack, lgbm_prob_pack] if p is not None]
    candidate_starts = [ts for ts in candidate_starts if ts is not None]
    # Fix 6: single name throughout — no pre_sweep_common_start alias
    common_start = max(candidate_starts) if candidate_starts else None

    # Fix 8: Snap common_start to nearest actual dev index if not exact
    if common_start is not None and common_start not in df_dev.index:
        idx_pos      = df_dev.index.searchsorted(common_start)
        idx_pos      = min(idx_pos, len(df_dev.index) - 1)
        common_start = df_dev.index[idx_pos]
        print(f"  [INFO] common_start snapped to nearest dev date: {common_start.date()}")

    print(f"\n  Common OOS start: {common_start.date() if common_start else 'N/A'}")

    # Step 4c: Sweep thresholds using top-level function.
    # Fix 9: Sweep DFs go straight to CSV — no unused locals stored.
    best_lr, _ = sweep_thresholds_from_prob_pack(
        df_dev, lr_prob_pack, cfg,
        eval_start=common_start, out_dir=out_dir, run_tag=run_tag)

    # Fix 3: Hard guard — LR is the base model; if it failed, stop loudly.
    if lr_prob_pack is None or lr_prob_pack.get("first_live") is None:
        raise RuntimeError(
            "LogReg produced no valid OOS predictions. "
            "Cannot continue — check train_years vs dev length.")

    best_lgbm = None   # Fix 2: None by default — only set after a valid dev sweep
    lgbm_pack = None   # Fix D: start as None; only set if sweep succeeds
    if HAS_LGBM and lgbm_prob_pack is not None:
        if lgbm_prob_pack.get("first_live") is None:
            print("  [WARN] LightGBM has no valid OOS predictions — skipping LGBM.")
        else:
            best_lgbm, _ = sweep_thresholds_from_prob_pack(
                df_dev, lgbm_prob_pack, cfg,
                eval_start=common_start, out_dir=out_dir, run_tag=run_tag)
            lgbm_pack = lgbm_prob_pack   # only assigned when sweep was valid

    # Convenience aliases for evaluate_threshold calls
    lr_pack = lr_prob_pack

    # 6. Final dev evaluation on common window — same as threshold sweep.
    print("\n" + "="*60)
    print("STEP 5: FINAL DEV EVALUATION (common OOS window)")
    print(f"  LogReg threshold:   {best_lr}")
    print(f"  LightGBM threshold: {best_lgbm if best_lgbm is not None else 'N/A (skipped)'}")
    print(f"  Common OOS start:   {common_start.date() if common_start else 'N/A'}")
    print("="*60)

    # Evaluate both models on the SAME common window
    lr_m   = evaluate_threshold(df_dev, lr_pack,   cfg, best_lr,   eval_start=common_start)
    lgbm_m = evaluate_threshold(df_dev, lgbm_pack, cfg, best_lgbm, eval_start=common_start) \
             if lgbm_pack else None

    # Dev baselines on same common window
    first_live_ts = common_start
    if first_live_ts is not None:
        df_dev_oos = df_dev.loc[first_live_ts:].copy()
        print(f"\n  T1-2: Dev baselines restricted to OOS window "
              f"({first_live_ts.date()} onward)")
    else:
        df_dev_oos = df_dev
        print("  T1-2: Could not determine OOS window — using full dev for baselines")

    baselines_dev = run_baselines(df_dev_oos, cfg, "DEV OOS-LIVE")

    # T3-10: per-year B&H return for comparison
    bah_dev      = baselines_dev.get("Buy & Hold SPY (open[0])")
    bah_by_yr_dev = _build_bah_by_year(bah_dev["daily_pnl"]) if bah_dev else None

    print_metrics(lr_m,   baselines_dev)
    if lgbm_m:
        print_metrics(lgbm_m, baselines_dev)
    print_fold_stability(lr_m)
    if lgbm_m:
        print_fold_stability(lgbm_m)

    print("\n  ── SAMPLE ADEQUACY ──")
    insufficient_sample_warnings(lr_m,   "LogReg DEV")
    if lgbm_m:
        insufficient_sample_warnings(lgbm_m, "LightGBM DEV")

    trades_by_year_and_regime(lr_m,   "LogReg DEV",   bah_by_yr_dev)
    if lgbm_m:
        trades_by_year_and_regime(lgbm_m, "LightGBM DEV", bah_by_yr_dev)

    print("\n=== LOGREG BUCKET TABLES ===")
    print(probability_bucket_tables(lr_m, best_lr))
    if lgbm_m:
        print("\n=== LIGHTGBM BUCKET TABLES ===")
        print(probability_bucket_tables(lgbm_m, best_lgbm))

    # Fix 4: Use run_tag prefix on all major output files
    dev_models = [lr_m] + ([lgbm_m] if lgbm_m else [])
    chart_path = os.path.join(out_dir, f"{run_tag}_trading_results.png")
    plot_results(dev_models, baselines_dev, chart_path)

    # Dev comparison
    print("\n" + "="*60)
    print("DEV COMPARISON (OOS-live window)")
    print("="*60)
    print(f"  {'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>8} {'CAGR':>7} {'Turn/yr':>8}")
    print("  " + "-"*74)
    for m in list(baselines_dev.values()) + dev_models:
        print(f"  {m['label']:<30} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd']:>7.1%} {m['cagr']:>6.1%} {m['turnover']:>7.1f}")

    if lgbm_m:
        sh_diff = lgbm_m["sharpe"] - lr_m["sharpe"]
        print("\n  ── LogReg vs LightGBM ──")
        if   sh_diff > 0.10: print(f"  LightGBM wins by {sh_diff:.2f} — use LightGBM.")
        elif sh_diff > 0:    print(f"  Marginal gain ({sh_diff:.2f}) — prefer LogReg.")
        else:                print(f"  LogReg wins/ties — complexity not earned.")

    # A2: Final report — explicitly note if LGBM was skipped
    if not HAS_LGBM:
        print("\n  [INFO] LightGBM was NOT included in this run — not installed.")
        print("  [INFO] Results are LogReg only. Install lightgbm for full comparison.")

    # Fix 1: Cleaner list comprehension — no -np.inf acrobatics
    final_results   = None
    dev_sharpes     = [lr_m["sharpe"]] + ([lgbm_m["sharpe"]] if lgbm_m else [])
    best_dev_sharpe = max(dev_sharpes)
    print("\n" + "="*60)
    if best_dev_sharpe >= 0.8:
        print(f"Dev Sharpe={best_dev_sharpe:.2f} ≥ 0.8 — opening 2024 holdout.")

        # T1-2: Test baselines on full test period (no OOS-window restriction for test)
        baselines_test = run_baselines(df_test, cfg, "TEST 2024")
        print("\n  ── TEST BASELINES ──")
        print(f"  {'Strategy':<30} {'Sharpe':>7} {'MaxDD':>8} {'CAGR':>7}")
        print("  " + "-"*56)
        for m in baselines_test.values():
            print(f"  {m['label']:<30} {m['sharpe']:>7.2f} "
                  f"{m['max_dd']:>7.1%} {m['cagr']:>6.1%}")

        final_results = run_final_test(
            df_dev, df_test, feat_cols, cfg,
            best_lr, best_lgbm, baselines_test)
    else:
        print(f"Dev Sharpe={best_dev_sharpe:.2f} < 0.8 — not opening 2024 holdout.")
        print("Fix the model first. Run again when dev results improve.")

    dev_vs_test_summary(lr_m, lgbm_m, final_results)

    # Fix B+C: summary.json built HERE — after final_results is assigned,
    # holdout decision is complete, and all test metrics are available.
    lr_t_s   = (final_results or {}).get("logistic", {})
    lgbm_t_s = (final_results or {}).get("lgbm",     {})
    summary  = {
        "run_tag":           run_tag,
        "common_start":      common_start.strftime("%Y-%m-%d") if common_start else None,
        "lr_coverage":       float(lr_pack["coverage"])   if lr_pack   else None,
        "lgbm_coverage":     float(lgbm_pack["coverage"]) if lgbm_pack else None,
        "best_lr_thresh":    float(best_lr),
        "best_lgbm_thresh":  float(best_lgbm) if best_lgbm is not None else None,
        "lr_dev_sharpe":     float(lr_m["sharpe"]),
        "lr_dev_max_dd":     float(lr_m["max_dd"]),
        "lr_dev_pf":         float(lr_m["profit_factor"]),
        "lr_dev_brier":      float(lr_m["brier"]) if lr_m["brier"] is not None else None,
        "lgbm_dev_sharpe":   float(lgbm_m["sharpe"]) if lgbm_m else None,
        "holdout_opened":    final_results is not None,
        "lr_test_sharpe":    float(lr_t_s["sharpe"])  if lr_t_s.get("sharpe")  is not None else None,
        "lr_test_pf":        float(lr_t_s["profit_factor"]) if lr_t_s.get("profit_factor") is not None else None,
        "lr_test_max_dd":    float(lr_t_s["max_dd"])  if lr_t_s.get("max_dd")  is not None else None,
        "lr_test_brier":     float(lr_t_s["brier"])   if lr_t_s.get("brier")   is not None else None,
        "lgbm_test_sharpe":  float(lgbm_t_s["sharpe"]) if lgbm_t_s.get("sharpe") is not None else None,
        # Fix 3: dev_model_winner, not model_winner — this is based on dev Sharpe only.
        # If holdout results exist, inspect lr_test_sharpe / lgbm_test_sharpe for the
        # true final winner.
        "dev_model_winner":  ("lgbm" if lgbm_m and lgbm_m["sharpe"] > lr_m["sharpe"] + 0.10
                              else "logistic"),
        # Explicit final winner — None only if holdout never opened.
        # If one model ran, stores that model name. If multiple ran, stores
        # the one with the highest 2024 test Sharpe. Distinct from dev_model_winner.
        "final_model_winner_by_test_sharpe": (
            None if not final_results else
            max(
                ((mt, final_results[mt]["sharpe"])
                 for mt in final_results if final_results[mt].get("sharpe") is not None),
                key=lambda x: x[1],
                default=(None, None)
            )[0]
        ),
    }
    summary_path = os.path.join(out_dir, f"{run_tag}_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    # Key numbers
    bah_d = baselines_dev.get("Buy & Hold SPY (open[0])")
    print("\n" + "="*60)
    print("KEY NUMBERS TO BRING BACK:")
    print("="*60)
    print(f"  1.  DEV LONG rate:               {df_dev['label'].mean():.1%}")
    print(f"  2.  DEV min signals/year:         "
          f"{df_dev.groupby(df_dev.index.year)['label'].sum().min()}")
    print(f"  3.  B&H Sharpe (OOS window):      "
          f"{_safe_fmt(bah_d['sharpe'] if bah_d else np.nan, '{:.2f}')}")
    print(f"  4.  Common OOS window start:      "
          f"{common_start.date() if common_start else 'N/A'}")
    # Fix 2: Guard lr_pack access
    if lr_pack:
        print(f"  5a. LogReg raw OOS coverage:      {lr_pack['coverage']:.1%}  "
              f"(model-specific, pre-alignment)")
    else:
        print("  5a. LogReg raw OOS coverage:      N/A (no valid OOS predictions)")
    if lgbm_pack:
        print(f"  5b. LightGBM raw OOS coverage:    {lgbm_pack['coverage']:.1%}  "
              f"(model-specific, pre-alignment)")
    if common_start is not None:
        common_cov = float((df_dev.index >= common_start).mean())
        print(f"  5c. Common comparison coverage:   {common_cov:.1%}  "
              f"(aligned window as share of full dev)")
    print(f"  6.  Best LogReg threshold:        {best_lr}")
    print(f"  7.  LogReg DEV Sharpe:            {lr_m['sharpe']:.2f}")
    print(f"  7b. LogReg Sharpe 95% CI:         "
          f"[{_safe_fmt(lr_m['sharpe_ci'][0], '{:.2f}')} – "
          f"{_safe_fmt(lr_m['sharpe_ci'][1], '{:.2f}')}]")
    print(f"  8.  LogReg DEV Max Drawdown:      {lr_m['max_dd']:.1%}")
    print(f"  9.  LogReg DEV Profit Factor:     {lr_m['profit_factor']:.2f}")
    print(f"  10. LogReg DEV Positive Folds:    {lr_m['pos_folds']:.0%}")
    print(f"  11. LogReg DEV Brier:             {_safe_fmt(lr_m['brier'], '{:.4f}')}")
    print(f"  12. LogReg DEV Turnover/yr:       {lr_m['turnover']:.1f}")
    if lgbm_m:
        print(f"  13. LightGBM DEV Sharpe:          {lgbm_m['sharpe']:.2f}")
    print(f"  14. LogReg DEV avg MAE (R):       {_safe_fmt(lr_m['avg_mae_r'], '{:.2f}')}")
    print(f"  15. LogReg DEV avg MFE (R):       {_safe_fmt(lr_m['avg_mfe_r'], '{:.2f}')}")
    if final_results:
        lr_t = final_results.get("logistic", {})
        print(f"  16. LogReg TEST Sharpe (2024):    {_safe_fmt(lr_t.get('sharpe'), '{:.2f}')}")
        print(f"  17. LogReg TEST Brier (2024):     {_safe_fmt(lr_t.get('brier'), '{:.4f}')}")
        print(f"  18. LogReg TEST CI Sharpe:        "
              f"[{_safe_fmt((lr_t.get('sharpe_ci') or (np.nan,))[0], '{:.2f}')} – "
              f"{_safe_fmt((lr_t.get('sharpe_ci') or (np.nan, np.nan))[1], '{:.2f}')}]")

    print(f"\n  Output files (run_tag={run_tag}):")
    print(f"    {run_tag}_spy_dev.csv")
    print(f"    {run_tag}_spy_test.csv")
    print(f"    {run_tag}_trading_results.png")
    print(f"    {run_tag}_summary.json")
    print(f"    {run_tag}_logistic_threshold_sweep.csv")
    if best_lgbm is not None:
        print(f"    {run_tag}_lgbm_threshold_sweep.csv")
    print(f"  Output dir: {os.path.abspath(out_dir)}")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
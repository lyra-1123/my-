"""Overfitting-aware factor screening: PBO (probability of backtest
overfitting, Bailey/Lopez de Prado-style CSCV) and walk-forward OOS Sharpe.

Every factor family here has up to 12 near-duplicate variants (4 windows x
{raw, pctrank500, pctrank2000}) — exactly the situation PBO is designed to
catch: if we just pick "whichever variant has the best in-sample Sharpe",
how likely is that pick to be noise rather than real signal?

To turn a regime-classification factor into something with a Sharpe ratio,
each variant is used to gate a single fixed, deliberately simple mean-
reversion rule (fade the last bar, held for `horizon` bars): the factor
itself is not a trading rule, so the gated rule's Sharpe measures whether
knowing the factor would have improved a bar-for-bar identical baseline
strategy, isolating the factor's contribution from strategy design (that's
Phase 3+4's job, not this screen's).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def build_proxy_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Fixed, factor-agnostic proxy rule: fade the last 1-bar move, held for
    `horizon` bars. This is deliberately the simplest possible mean-reversion
    rule (the same family of edge a martingale ladder tries to harvest) so
    that gating it with a factor isolates the factor's own contribution."""
    close = df["close"]
    r1 = close.pct_change()
    fwd = close.pct_change(horizon).shift(-horizon)
    return -np.sign(r1) * fwd


def decision_points(n_rows: int, horizon: int) -> np.ndarray:
    """Non-overlapping sample indices, spaced `horizon` bars apart, so the
    forward-return windows used for Sharpe/PBO don't overlap (overlapping
    windows would inflate apparent significance via autocorrelation)."""
    return np.arange(1, n_rows - horizon, horizon)


def safe_mask_from_threshold(factor: pd.Series, ic_sign: float, lo, hi) -> pd.Series:
    """`lo`/`hi` are the (20th, 80th) percentile thresholds to use — pass
    ones computed from whatever sample is allowed to see (full-sample for
    the CSCV/PBO test, in-sample-only for the walk-forward OOS test)."""
    return (factor >= hi) if ic_sign < 0 else (factor <= lo)


def conditional_returns(factor: pd.Series, ic_sign: float, base_return: pd.Series,
                         lo=None, hi=None) -> pd.Series:
    if lo is None or hi is None:
        lo, hi = factor.quantile(0.2), factor.quantile(0.8)
    mask = safe_mask_from_threshold(factor, ic_sign, lo, hi)
    return base_return.where(mask, 0.0)


def information_ratio(factor: pd.Series, label: pd.Series, times: pd.Series,
                       points: np.ndarray, freq: str = "M") -> dict:
    """ICIR: split the (non-overlapping) decision points into calendar
    periods (months by default), compute Spearman IC within each period
    separately, then IR = mean(IC)/std(IC) across periods — this measures
    whether a factor's predictive power is *consistent* period to period,
    not just strong on average (a factor with the same full-sample IC but
    wildly swinging sign period to period is far less trustworthy). The
    per-period IC series is oriented by the full-sample IC's sign so a
    reliably-negative factor scores a high positive IR, not a negative one.
    """
    sub = pd.DataFrame({
        "factor": factor.iloc[points].to_numpy(),
        "label": label.iloc[points].to_numpy(),
        "time": pd.to_datetime(pd.Series(times).iloc[points].to_numpy()),
    }).dropna()
    if len(sub) < 30:
        return {"ic": np.nan, "ir": np.nan, "n_periods": 0}

    full_ic = sub["factor"].corr(sub["label"], method="spearman")
    sign = np.sign(full_ic) if full_ic and not np.isnan(full_ic) else 1.0

    sub["period"] = sub["time"].dt.tz_localize(None).dt.to_period(freq)
    period_ic = sub.groupby("period").apply(
        lambda g: g["factor"].corr(g["label"], method="spearman") if len(g) >= 10 else np.nan,
        include_groups=False,
    ).dropna()

    if len(period_ic) < 3 or period_ic.std() == 0:
        return {"ic": full_ic, "ir": np.nan, "n_periods": len(period_ic)}

    oriented = period_ic * sign
    ir = oriented.mean() / oriented.std()
    return {"ic": full_ic, "ir": float(ir), "n_periods": len(period_ic)}


def _sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / x.std(ddof=1)


def pbo_for_family(variant_returns: dict, n_blocks: int = 10) -> tuple[float, dict]:
    """Combinatorially symmetric cross-validation (CSCV), simplified: split
    the (already non-overlapping) decision-point return series of every
    variant into n_blocks contiguous blocks, form all ways to bisect the
    blocks into an IS/OOS half, and check how often "best variant by IS
    Sharpe" ranks below the OOS median. Returns (PBO, extra diagnostics).
    """
    names = list(variant_returns.keys())
    n = len(next(iter(variant_returns.values())))
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    block_slices = [slice(edges[i], edges[i + 1]) for i in range(n_blocks)]

    # block_sharpe[b, v] = Sharpe of variant v's returns restricted to block b
    block_sharpe = np.full((n_blocks, len(names)), np.nan)
    for v, name in enumerate(names):
        arr = variant_returns[name]
        for b, sl in enumerate(block_slices):
            block_sharpe[b, v] = _sharpe(arr[sl])

    half = n_blocks // 2
    logits = []
    for train_blocks in combinations(range(n_blocks), half):
        test_blocks = [b for b in range(n_blocks) if b not in train_blocks]
        is_perf = np.nanmean(block_sharpe[list(train_blocks), :], axis=0)
        if np.all(np.isnan(is_perf)):
            continue
        best_v = int(np.nanargmax(is_perf))
        oos_perf = np.nanmean(block_sharpe[test_blocks, :], axis=0)
        if np.all(np.isnan(oos_perf)):
            continue
        rank = (pd.Series(oos_perf).rank(pct=True, na_option="bottom")).iloc[best_v]
        rank = min(max(rank, 1e-4), 1 - 1e-4)
        logits.append(np.log(rank / (1 - rank)))

    logits = np.array(logits)
    pbo = float((logits < 0).mean()) if len(logits) else float("nan")
    best_overall = names[int(np.nanargmax(np.nanmean(block_sharpe, axis=0)))]
    return pbo, {"n_splits": len(logits), "best_by_full_sample": best_overall}


def walk_forward_oos_sharpe(df_factor: pd.Series, base_return: pd.Series, ic_sign: float,
                             points: np.ndarray, is_frac: float = 0.7) -> dict:
    """Chronological walk-forward: fit the safe-zone threshold on the first
    is_frac of decision points only, apply that SAME fixed threshold to the
    held-out remainder — no look-ahead in either the threshold or the
    selection it's used for. A SINGLE split, so it's sensitive to whatever
    happens to land in that one held-out slice; see walk_forward_multi_fold
    for the more robust, multi-window version."""
    split = int(len(points) * is_frac)
    is_points, oos_points = points[:split], points[split:]

    is_factor = df_factor.iloc[is_points]
    lo, hi = is_factor.quantile(0.2), is_factor.quantile(0.8)

    is_ret = conditional_returns(df_factor.iloc[is_points], ic_sign, base_return.iloc[is_points], lo, hi)
    oos_ret = conditional_returns(df_factor.iloc[oos_points], ic_sign, base_return.iloc[oos_points], lo, hi)

    return {
        "is_sharpe": _sharpe(is_ret.to_numpy()),
        "oos_sharpe": _sharpe(oos_ret.to_numpy()),
        "oos_n_active": int((oos_ret != 0).sum()),
        "oos_n_total": len(oos_points),
    }


def fold_consistency_sharpe(returns: pd.Series, points: np.ndarray, n_folds: int = 5) -> dict:
    """For an ALREADY-DETERMINED return series (e.g. gated by a fixed voting
    rule with no per-fold parameter to refit) — NOT a genuine walk-forward
    test like walk_forward_multi_fold, since there's no threshold being
    fit-then-frozen here. Splits points into n_folds contiguous chunks,
    scores Sharpe within each (consistency check), and pools all of them
    for one overall Sharpe."""
    chunks = np.array_split(points, n_folds)
    fold_sharpes = np.array([_sharpe(returns.iloc[c].to_numpy()) for c in chunks])
    pooled = returns.iloc[points]
    return {
        "oos_sharpe_pooled": _sharpe(pooled.to_numpy()),
        "n_folds_positive": int(np.nansum(fold_sharpes > 0)),
        "n_folds_total": int(np.sum(~np.isnan(fold_sharpes))),
    }


def walk_forward_multi_fold(df_factor: pd.Series, base_return: pd.Series, ic_sign: float,
                             points: np.ndarray, n_folds: int = 5) -> dict:
    """Expanding-window walk-forward over n_folds+1 contiguous chunks: fold i
    trains on chunks [0..i) and tests on chunk i, each time refitting the
    safe-zone threshold on ONLY the training chunks and freezing it for that
    fold's test chunk (no look-ahead). All folds' held-out returns are then
    pooled into one track record and scored with a single Sharpe — this is
    much less sensitive to which specific sub-period a single 70/30 split
    happens to hold out (a regime shift landing entirely in one held-out
    slice can otherwise make a genuinely robust factor look broken, or vice
    versa) than the single-split walk_forward_oos_sharpe.
    """
    chunks = np.array_split(points, n_folds + 1)
    pooled_oos = []
    fold_sharpes = []
    for i in range(1, n_folds + 1):
        train_points = np.concatenate(chunks[:i])
        test_points = chunks[i]
        if len(test_points) == 0:
            continue
        train_factor = df_factor.iloc[train_points]
        lo, hi = train_factor.quantile(0.2), train_factor.quantile(0.8)
        test_ret = conditional_returns(df_factor.iloc[test_points], ic_sign, base_return.iloc[test_points], lo, hi)
        pooled_oos.append(test_ret)
        fold_sharpes.append(_sharpe(test_ret.to_numpy()))

    pooled = pd.concat(pooled_oos) if pooled_oos else pd.Series(dtype=float)
    fold_sharpes = np.array(fold_sharpes)
    return {
        "oos_sharpe_pooled": _sharpe(pooled.to_numpy()),
        "n_folds_positive": int(np.nansum(fold_sharpes > 0)),
        "n_folds_total": int(np.sum(~np.isnan(fold_sharpes))),
        "oos_n_active": int((pooled != 0).sum()),
        "oos_n_total": len(pooled),
    }

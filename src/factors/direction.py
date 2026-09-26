"""Directional (long/short) signals — distinct from library.py's regime-
classification factors (which only ever answer "is now safe to average
down", never "which way is price going"). Everything here is backward-
looking only (see library.py's look-ahead audit note in
reports/00_progress.md; the same rule applies here: rolling/shift(positive)
only, no future data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma_direction(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """+1 = fast MA above slow MA (long bias), -1 = below (short bias),
    0 = exactly equal (rare, treated as "no direction" downstream)."""
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    return np.sign(fast_ma - slow_ma)


def align_htf_direction(ltf_df: pd.DataFrame, htf_df: pd.DataFrame, htf_direction: pd.Series,
                         htf_bar_seconds: int) -> pd.Series:
    """Map a higher-timeframe direction series onto the lower-timeframe's
    bar index, WITHOUT look-ahead: an HTF bar starting at `time` only
    becomes known once it closes (time + htf_bar_seconds), so it's only
    valid for LTF bars at or after that close time — same convention as
    the H1->H4 alignment in Phase 2's build_htf_context."""
    ctx = pd.DataFrame({
        "valid_from": htf_df["time"] + pd.Timedelta(seconds=htf_bar_seconds),
        "htf_direction": htf_direction,
    }).dropna().sort_values("valid_from")
    merged = pd.merge_asof(
        ltf_df[["time"]].sort_values("time"), ctx,
        left_on="time", right_on="valid_from", direction="backward",
    )
    return merged["htf_direction"]


def cross_timeframe_signal(ltf_direction: pd.Series, htf_direction_aligned: pd.Series,
                            wait_bars: int = 4, reduced_size: float = 0.5) -> pd.DataFrame:
    """"Look at the big timeframe, trade the small one": full-size entry in
    the shared direction when LTF and HTF agree; if they disagree, stand
    down for `wait_bars` LTF bars, then defer to the HTF direction at
    `reduced_size` (rather than waiting indefinitely for the LTF to catch
    up — the HTF view is treated as authoritative once the grace period
    passes).
    """
    aligned = (ltf_direction == htf_direction_aligned) & (ltf_direction != 0)
    # length of the current run of consecutive misaligned bars, ending at
    # each row (resets to 0 the instant `aligned` is True)
    run_id = (aligned != aligned.shift()).cumsum()
    misalign_streak = (~aligned).groupby(run_id).cumcount() + 1
    misalign_streak = misalign_streak.where(~aligned, 0)

    deferred = (~aligned) & (misalign_streak >= wait_bars)

    direction = pd.Series(0.0, index=ltf_direction.index)
    size = pd.Series(0.0, index=ltf_direction.index)
    direction = direction.mask(aligned, ltf_direction)
    size = size.mask(aligned, 1.0)
    direction = direction.mask(deferred, htf_direction_aligned)
    size = size.mask(deferred, reduced_size)

    return pd.DataFrame({
        "aligned": aligned, "misalign_streak": misalign_streak, "deferred": deferred,
        "direction": direction, "size": size,
    })


def momentum_direction(factor: pd.Series, close: pd.Series, threshold: float) -> pd.Series:
    """Generalizes step-2's condition B: direction = sign of the last bar's
    move, but ONLY when `factor` is at/above `threshold` (i.e. the factor
    is read as "momentum/trend regime active", and price's own recent
    direction is trusted to continue). Fires both signs — no separate long/
    short variant needed."""
    active = factor >= threshold
    direction = np.sign(close.diff())
    return direction.where(active, 0.0)


def reversion_direction(factor: pd.Series, lo: float, hi: float) -> pd.Series:
    """Generalizes step-2's MFI condition to any factor: +1 when factor
    crosses UP through `lo` (recovering from an extreme low = long), -1
    when it crosses DOWN through `hi` (falling from an extreme high =
    short). A crossover EVENT, not a static level."""
    cross_up = (factor.shift(1) < lo) & (factor >= lo)
    cross_down = (factor.shift(1) > hi) & (factor <= hi)
    direction = pd.Series(0.0, index=factor.index)
    direction[cross_up] = 1.0
    direction[cross_down] = -1.0
    return direction


def _sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / x.std(ddof=1)


def combine_directions(direction_a: pd.Series, direction_b: pd.Series) -> pd.Series:
    """AND-consensus: only fires where both conditions are active AND agree
    on the same sign (mirrors signal_long/short's "A(long) & B(long)"
    template) — not just "both nonzero", genuine directional agreement."""
    agree = (direction_a == direction_b) & (direction_a != 0)
    return direction_a.where(agree, 0.0)


def _direction_for_mode(factor: pd.Series, close: pd.Series, mode: str,
                         quantile: float, lo=None, hi=None, thresh=None) -> pd.Series:
    if mode == "momentum":
        return momentum_direction(factor, close, thresh if thresh is not None else factor.quantile(quantile))
    lo = lo if lo is not None else factor.quantile(1 - quantile)
    hi = hi if hi is not None else factor.quantile(quantile)
    return reversion_direction(factor, lo, hi)


def _fold_chunks(n_rows: int, n_folds: int) -> list:
    """Contiguous BAR-INDEX chunks spanning the whole series (0..n_rows-1),
    NOT the sparse `decision_points` grid built for dense per-bar factor
    testing. Directional AND-signals fire rarely (a handful to a few
    hundred times a year) — subsetting to every Nth bar before looking for
    them silently discards most of the actual trigger bars (an earlier bug
    here made annual activation rates look ~10x lower than reality)."""
    return np.array_split(np.arange(n_rows), n_folds + 1)


def walk_forward_direction_pair(factor_a: pd.Series, mode_a: str, factor_b: pd.Series, mode_b: str,
                                 close: pd.Series, fwd_return: pd.Series, points: np.ndarray,
                                 n_folds: int = 5, quantile: float = 0.8) -> dict:
    """Same expanding-window walk-forward as walk_forward_direction, but for
    an AND-consensus pair: each factor's own threshold(s) are refit on the
    training range of each fold (no look-ahead), directions combined via
    combine_directions, then scored at every actual trigger bar in that
    fold's test range (not a sparse subsample — see _fold_chunks). `points`
    is accepted for signature compatibility but no longer used to define
    folds."""
    chunks = _fold_chunks(len(factor_a), n_folds)
    pooled_ret, pooled_dir = [], []
    fold_sharpes = []
    for i in range(1, n_folds + 1):
        train_idx, test_idx = np.concatenate(chunks[:i]), chunks[i]
        if len(test_idx) == 0:
            continue
        dir_a = _direction_for_mode(factor_a, close, mode_a, quantile,
                                     lo=factor_a.iloc[train_idx].quantile(1 - quantile),
                                     hi=factor_a.iloc[train_idx].quantile(quantile),
                                     thresh=factor_a.iloc[train_idx].quantile(quantile))
        dir_b = _direction_for_mode(factor_b, close, mode_b, quantile,
                                     lo=factor_b.iloc[train_idx].quantile(1 - quantile),
                                     hi=factor_b.iloc[train_idx].quantile(quantile),
                                     thresh=factor_b.iloc[train_idx].quantile(quantile))
        combined = combine_directions(dir_a, dir_b)
        test_slice = combined.iloc[test_idx]
        active = test_slice[test_slice != 0].index.intersection(fwd_return.dropna().index)
        test_ret = (combined.loc[active] * fwd_return.loc[active])
        pooled_ret.append(test_ret)
        pooled_dir.append(combined.loc[active])
        fold_sharpes.append(_sharpe(test_ret.to_numpy()))

    ret = pd.concat(pooled_ret) if pooled_ret else pd.Series(dtype=float)
    dirn = pd.concat(pooled_dir) if pooled_dir else pd.Series(dtype=float)
    fold_sharpes = np.array(fold_sharpes)
    return {
        "oos_sharpe_all": _sharpe(ret.to_numpy()),
        "oos_sharpe_long": _sharpe(ret[dirn > 0].to_numpy()),
        "oos_sharpe_short": _sharpe(ret[dirn < 0].to_numpy()),
        "n_long": int((dirn > 0).sum()), "n_short": int((dirn < 0).sum()),
        "n_folds_positive": int(np.nansum(fold_sharpes > 0)),
        "n_folds_total": int(np.sum(~np.isnan(fold_sharpes))),
    }


def walk_forward_direction(factor: pd.Series, close: pd.Series, fwd_return: pd.Series,
                            points: np.ndarray, mode: str, n_folds: int = 5,
                            quantile: float = 0.8) -> dict:
    """5-fold expanding-window walk-forward for momentum_direction/
    reversion_direction: threshold(s) fit on the training RANGE of each
    fold, frozen, applied to that fold's test range; scored at every
    actual trigger bar (see _fold_chunks — NOT the sparse decision_points
    grid). `points` kept for signature compatibility, unused for fold
    definition."""
    chunks = _fold_chunks(len(factor), n_folds)
    pooled_ret, pooled_dir = [], []
    fold_sharpes = []
    for i in range(1, n_folds + 1):
        train_idx, test_idx = np.concatenate(chunks[:i]), chunks[i]
        if len(test_idx) == 0:
            continue
        train_factor = factor.iloc[train_idx]
        if mode == "momentum":
            direction = momentum_direction(factor, close, train_factor.quantile(quantile))
        else:
            direction = reversion_direction(
                factor, train_factor.quantile(1 - quantile), train_factor.quantile(quantile)
            )
        test_slice = direction.iloc[test_idx]
        active = test_slice[test_slice != 0].index.intersection(fwd_return.dropna().index)
        test_ret = (direction.loc[active] * fwd_return.loc[active])
        pooled_ret.append(test_ret)
        pooled_dir.append(direction.loc[active])
        fold_sharpes.append(_sharpe(test_ret.to_numpy()))

    ret = pd.concat(pooled_ret) if pooled_ret else pd.Series(dtype=float)
    dirn = pd.concat(pooled_dir) if pooled_dir else pd.Series(dtype=float)
    fold_sharpes = np.array(fold_sharpes)
    return {
        "oos_sharpe_all": _sharpe(ret.to_numpy()),
        "oos_sharpe_long": _sharpe(ret[dirn > 0].to_numpy()),
        "oos_sharpe_short": _sharpe(ret[dirn < 0].to_numpy()),
        "n_long": int((dirn > 0).sum()), "n_short": int((dirn < 0).sum()),
        "n_folds_positive": int(np.nansum(fold_sharpes > 0)),
        "n_folds_total": int(np.sum(~np.isnan(fold_sharpes))),
    }

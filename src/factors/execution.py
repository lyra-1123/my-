"""Signal design step 3: turn a direction EVENT (direction.py) into an
actual simulated trade with real entry/exit rules, instead of scoring the
fixed-horizon `fwd_return` every walk-forward function up to this point has
used. Rules (per the user's spec):
  - entry: next bar's OPEN after the trigger bar (not the trigger bar's own
    close)
  - exit: whichever of stop-loss / take-profit / holding-to-N-bars-expiry
    happens first, checked against each held bar's actual high/low (path-
    dependent, not just the endpoint return)
  - a same-bar SL+TP double-touch is resolved in favor of the stop-loss
    (conservative assumption, standard backtest convention)
No look-ahead: ATR (for ATR-based stops) is read at the trigger bar (uses
only bars up to and including it), thresholds are still fit train-fold-only
exactly as in direction.py's walk-forward functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .direction import _fold_chunks, _direction_for_mode, combine_directions


def simulate_trades(direction: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, atr: pd.Series, n_hold: int,
                     sl_type: str | None = None, sl_level: float | None = None,
                     rr_ratio: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """sl_type: None (no stop, holding-to-expiry only), "atr" (stop distance =
    sl_level x ATR at the trigger bar), or "pct" (stop distance = sl_level x
    entry price). rr_ratio: take-profit distance = stop distance x rr_ratio
    (meaningless without a stop, so ignored when sl_type is None). n_hold
    counts the entry bar itself as the first held bar. Returns (per-trade
    returns, per-trade directions) — one entry per trigger with a full
    n_hold-bar window available in the data (edge-of-sample triggers near
    the very end are dropped, same as every fwd_return-based function so far).
    """
    d_all = direction.to_numpy()
    trig_idx = np.flatnonzero(d_all != 0)
    entry_idx = trig_idx + 1
    n_rows = len(d_all)
    valid = entry_idx + n_hold - 1 < n_rows
    trig_idx, entry_idx = trig_idx[valid], entry_idx[valid]
    if len(entry_idx) == 0:
        return np.array([]), np.array([])

    d = d_all[trig_idx]
    o, h, l, c = open_.to_numpy(), high.to_numpy(), low.to_numpy(), close.to_numpy()
    entry_price = o[entry_idx]

    if sl_type is None:
        exit_price = c[entry_idx + n_hold - 1]
        return d * (exit_price - entry_price) / entry_price, d

    if sl_type == "atr":
        sl_dist = sl_level * atr.to_numpy()[trig_idx]
    elif sl_type == "pct":
        sl_dist = sl_level * entry_price
    else:
        raise ValueError(f"unknown sl_type {sl_type!r}")
    tp_dist = sl_dist * rr_ratio

    offsets = np.arange(n_hold)
    window_idx = entry_idx[:, None] + offsets[None, :]
    highs = h[window_idx].astype(np.float32)
    lows = l[window_idx].astype(np.float32)

    is_long = d > 0
    sl_price = np.where(is_long, entry_price - sl_dist, entry_price + sl_dist)
    tp_price = np.where(is_long, entry_price + tp_dist, entry_price - tp_dist)

    sl_hit = np.where(is_long[:, None], lows <= sl_price[:, None], highs >= sl_price[:, None])
    tp_hit = np.where(is_long[:, None], highs >= tp_price[:, None], lows <= tp_price[:, None])

    any_sl, any_tp = sl_hit.any(axis=1), tp_hit.any(axis=1)
    first_sl = np.where(any_sl, sl_hit.argmax(axis=1), n_hold)
    first_tp = np.where(any_tp, tp_hit.argmax(axis=1), n_hold)
    sl_wins = first_sl <= first_tp  # same-bar tie -> stop-loss wins (conservative)
    hit_either = (first_sl < n_hold) | (first_tp < n_hold)

    exit_price = np.where(hit_either, np.where(sl_wins, sl_price, tp_price), c[entry_idx + n_hold - 1])
    return d * (exit_price - entry_price) / entry_price, d


def trade_windows(direction: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series,
                   close: pd.Series, atr: pd.Series, n_hold: int,
                   sl_type: str | None = None, sl_level: float | None = None,
                   rr_ratio: float | None = None) -> pd.DataFrame:
    """Same rules as simulate_trades, but returns each trade's (entry_idx,
    exit_idx, direction) bar-index window instead of its return -- for step
    5's overlap/interlock analysis (does a new trigger arrive while a
    previous trade from the same or another candidate is still open)."""
    d_all = direction.to_numpy()
    trig_idx = np.flatnonzero(d_all != 0)
    entry_idx = trig_idx + 1
    n_rows = len(d_all)
    valid = entry_idx + n_hold - 1 < n_rows
    trig_idx, entry_idx = trig_idx[valid], entry_idx[valid]
    if len(entry_idx) == 0:
        return pd.DataFrame(columns=["entry_idx", "exit_idx", "direction"])

    d = d_all[trig_idx]
    o, h, l, c = open_.to_numpy(), high.to_numpy(), low.to_numpy(), close.to_numpy()
    entry_price = o[entry_idx]

    if sl_type is None:
        exit_idx = entry_idx + n_hold - 1
        exit_price = c[exit_idx]
        return pd.DataFrame({"entry_idx": entry_idx, "exit_idx": exit_idx, "direction": d,
                              "entry_price": entry_price, "exit_price": exit_price})

    if sl_type == "atr":
        sl_dist = sl_level * atr.to_numpy()[trig_idx]
    elif sl_type == "pct":
        sl_dist = sl_level * entry_price
    else:
        raise ValueError(f"unknown sl_type {sl_type!r}")
    tp_dist = sl_dist * rr_ratio

    offsets = np.arange(n_hold)
    window_idx = entry_idx[:, None] + offsets[None, :]
    highs = h[window_idx].astype(np.float32)
    lows = l[window_idx].astype(np.float32)

    is_long = d > 0
    sl_price = np.where(is_long, entry_price - sl_dist, entry_price + sl_dist)
    tp_price = np.where(is_long, entry_price + tp_dist, entry_price - tp_dist)
    sl_hit = np.where(is_long[:, None], lows <= sl_price[:, None], highs >= sl_price[:, None])
    tp_hit = np.where(is_long[:, None], highs >= tp_price[:, None], lows <= tp_price[:, None])

    any_sl, any_tp = sl_hit.any(axis=1), tp_hit.any(axis=1)
    first_sl = np.where(any_sl, sl_hit.argmax(axis=1), n_hold)
    first_tp = np.where(any_tp, tp_hit.argmax(axis=1), n_hold)
    sl_wins = first_sl <= first_tp
    hit_either = (first_sl < n_hold) | (first_tp < n_hold)
    exit_offset = np.where(sl_wins, np.minimum(first_sl, n_hold - 1), np.minimum(first_tp, n_hold - 1))
    exit_idx = entry_idx + exit_offset
    exit_price = np.where(hit_either, np.where(sl_wins, sl_price, tp_price), c[entry_idx + n_hold - 1])
    return pd.DataFrame({"entry_idx": entry_idx, "exit_idx": exit_idx, "direction": d,
                          "entry_price": entry_price, "exit_price": exit_price})


def simulate_net_position(trades: pd.DataFrame) -> pd.DataFrame:
    """Reconciles overlapping trades from possibly many sources (candidates,
    or a single candidate's own long+short streams) into a SINGLE net
    position over time -- step 5's "opposing signal closes first" interlock
    policy. `trades` needs columns entry_idx, entry_price, direction,
    exit_idx, exit_price (its NATURAL exit if never interrupted; e.g. from
    `trade_windows`), and `source` (candidate name, for attribution).

    Rule: a new trigger in the position's existing direction is ignored
    (position already running, no size added); a new trigger in the
    OPPOSITE direction force-closes the current position immediately at the
    new trigger's own entry bar/price (this is a reversal), then opens the
    new one. A position that never meets a conflicting trigger before its
    own natural exit closes there instead. Simultaneous same-bar entries
    from different sources are resolved by input order (documented
    limitation, not a real tie-break rule). Returns one row per REALIZED
    trade, with `forced_close` marking ones cut short by a reversal."""
    trades = trades.sort_values("entry_idx", kind="stable").reset_index(drop=True)
    realized = []
    current = None
    for row in trades.itertuples(index=False):
        if current is not None:
            if current["exit_idx"] < row.entry_idx:
                current["actual_exit_idx"] = current["exit_idx"]
                current["actual_exit_price"] = current["exit_price"]
                current["forced_close"] = False
                realized.append(current)
                current = None
            elif np.sign(current["direction"]) == np.sign(row.direction):
                continue
            else:
                current["actual_exit_idx"] = row.entry_idx
                current["actual_exit_price"] = row.entry_price
                current["forced_close"] = True
                realized.append(current)
                current = None
        if current is None:
            current = {"entry_idx": row.entry_idx, "entry_price": row.entry_price,
                       "direction": row.direction, "exit_idx": row.exit_idx,
                       "exit_price": row.exit_price, "source": row.source}
    if current is not None:
        current["actual_exit_idx"] = current["exit_idx"]
        current["actual_exit_price"] = current["exit_price"]
        current["forced_close"] = False
        realized.append(current)
    out = pd.DataFrame(realized)
    out["ret"] = out["direction"] * (out["actual_exit_price"] - out["entry_price"]) / out["entry_price"]
    return out


def _sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / x.std(ddof=1)


def _pool_fold_results(pooled_ret, pooled_dir, fold_sharpes, return_raw: bool = False):
    ret_all = np.concatenate(pooled_ret) if pooled_ret else np.array([])
    dir_all = np.concatenate(pooled_dir) if pooled_dir else np.array([])
    fold_sharpes = np.array(fold_sharpes)
    result = {
        "oos_sharpe_all": _sharpe(ret_all),
        "oos_sharpe_long": _sharpe(ret_all[dir_all > 0]) if len(dir_all) else float("nan"),
        "oos_sharpe_short": _sharpe(ret_all[dir_all < 0]) if len(dir_all) else float("nan"),
        "n_long": int((dir_all > 0).sum()), "n_short": int((dir_all < 0).sum()),
        "n_folds_positive": int(np.nansum(fold_sharpes > 0)),
        "n_folds_total": int(np.sum(~np.isnan(fold_sharpes))),
    }
    if return_raw:
        result["ret_all"], result["dir_all"] = ret_all, dir_all
    return result


def _apply_direction_filter(direction: pd.Series, direction_filter: str | None) -> pd.Series:
    """direction_filter: None (keep both), "long" (zero out short triggers),
    or "short" (zero out long triggers) -- used by step 4 to search/score
    stop-loss/take-profit separately per side."""
    if direction_filter is None:
        return direction
    if direction_filter == "long":
        return direction.where(direction > 0, 0.0)
    if direction_filter == "short":
        return direction.where(direction < 0, 0.0)
    raise ValueError(f"unknown direction_filter {direction_filter!r}")


def walk_forward_execution_single(factor: pd.Series, close: pd.Series, open_: pd.Series,
                                   high: pd.Series, low: pd.Series, atr: pd.Series, mode: str,
                                   n_hold: int, sl_type=None, sl_level=None, rr_ratio=None,
                                   quantile: float = 0.8, n_folds: int = 5,
                                   direction_filter: str | None = None, return_raw: bool = False) -> dict:
    """Same expanding-window walk-forward as direction.walk_forward_direction
    (threshold fit on training range, frozen, applied to test range) but
    scored with simulate_trades' real entry/exit simulation instead of
    fwd_return. direction_filter restricts scoring to only long or only
    short triggers (see _apply_direction_filter) -- for step 4's per-side
    stop-loss/take-profit search."""
    chunks = _fold_chunks(len(factor), n_folds)
    pooled_ret, pooled_dir, fold_sharpes = [], [], []
    for i in range(1, n_folds + 1):
        train_idx, test_idx = np.concatenate(chunks[:i]), chunks[i]
        if len(test_idx) == 0:
            continue
        train_factor = factor.iloc[train_idx]
        direction_full = _direction_for_mode(
            factor, close, mode, quantile,
            lo=train_factor.quantile(1 - quantile), hi=train_factor.quantile(quantile),
            thresh=train_factor.quantile(quantile),
        )
        direction_full = _apply_direction_filter(direction_full, direction_filter)
        mask = np.zeros(len(factor), dtype=bool)
        mask[test_idx] = True
        direction_test = direction_full.where(pd.Series(mask, index=factor.index), 0.0)
        ret, d = simulate_trades(direction_test, open_, high, low, close, atr, n_hold,
                                  sl_type, sl_level, rr_ratio)
        pooled_ret.append(ret)
        pooled_dir.append(d)
        fold_sharpes.append(_sharpe(ret))
    return _pool_fold_results(pooled_ret, pooled_dir, fold_sharpes, return_raw)


def walk_forward_execution_pair(factor_a: pd.Series, mode_a: str, factor_b: pd.Series, mode_b: str,
                                 close: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series,
                                 atr: pd.Series, n_hold: int, sl_type=None, sl_level=None, rr_ratio=None,
                                 quantile: float = 0.8, n_folds: int = 5,
                                 direction_filter: str | None = None, return_raw: bool = False) -> dict:
    """Pair (AND-consensus) version of walk_forward_execution_single."""
    chunks = _fold_chunks(len(factor_a), n_folds)
    pooled_ret, pooled_dir, fold_sharpes = [], [], []
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
        combined = _apply_direction_filter(combined, direction_filter)
        mask = np.zeros(len(factor_a), dtype=bool)
        mask[test_idx] = True
        combined_test = combined.where(pd.Series(mask, index=factor_a.index), 0.0)
        ret, d = simulate_trades(combined_test, open_, high, low, close, atr, n_hold,
                                  sl_type, sl_level, rr_ratio)
        pooled_ret.append(ret)
        pooled_dir.append(d)
        fold_sharpes.append(_sharpe(ret))
    return _pool_fold_results(pooled_ret, pooled_dir, fold_sharpes, return_raw)


def _score_masked(direction_full, idx_range, open_, high, low, close, atr, n_hold,
                   sl_type=None, sl_level=None, rr_ratio=None):
    mask = np.zeros(len(direction_full), dtype=bool)
    mask[idx_range] = True
    restricted = direction_full.where(pd.Series(mask, index=direction_full.index), 0.0)
    return simulate_trades(restricted, open_, high, low, close, atr, n_hold, sl_type, sl_level, rr_ratio)


def _inner_select_n_hold(direction_full, train_idx, open_, high, low, close, atr, n_hold_grid,
                          min_trades: int = 10):
    """Step 6: pick the holding period using ONLY the training range's own
    Sharpe (mirrors 02n, but re-run inside every fold instead of once on the
    whole history) -- avoids freezing a single N chosen by peeking at all 5
    folds' pooled OOS performance at once."""
    best_n, best_sh, best_qualifies = n_hold_grid[0], float("-inf"), False
    for n in n_hold_grid:
        ret, _ = _score_masked(direction_full, train_idx, open_, high, low, close, atr, n)
        sh = _sharpe(ret)
        qualifies = len(ret) >= min_trades and not np.isnan(sh)
        sh_cmp = sh if not np.isnan(sh) else float("-inf")
        if (qualifies and not best_qualifies) or (qualifies == best_qualifies and sh_cmp > best_sh):
            best_n, best_sh, best_qualifies = n, sh_cmp, qualifies
    return best_n


def _inner_select_sl_tp(direction_side, train_idx, open_, high, low, close, atr, n_hold, sl_configs,
                         min_trades: int = 10):
    """Step 6: pick one side's (sl_type, sl_level, rr_ratio) using only the
    training range's Sharpe (mirrors 02o/02p, re-run inside every fold)."""
    best_cfg, best_sh, best_qualifies = sl_configs[0], float("-inf"), False
    for sl_type, sl_level, rr in sl_configs:
        real_sl_type = None if sl_type == "none" else sl_type
        ret, _ = _score_masked(direction_side, train_idx, open_, high, low, close, atr, n_hold,
                                real_sl_type, sl_level, rr)
        sh = _sharpe(ret)
        qualifies = len(ret) >= min_trades and not np.isnan(sh)
        sh_cmp = sh if not np.isnan(sh) else float("-inf")
        if (qualifies and not best_qualifies) or (qualifies == best_qualifies and sh_cmp > best_sh):
            best_cfg, best_sh, best_qualifies = (sl_type, sl_level, rr), sh_cmp, qualifies
    return best_cfg


def nested_walk_forward_single(factor: pd.Series, close: pd.Series, open_: pd.Series, high: pd.Series,
                                low: pd.Series, atr: pd.Series, mode: str, n_hold_grid, sl_configs,
                                quantile: float = 0.8, n_folds: int = 5) -> dict:
    """Full nested walk-forward (signal design step 6): everything 02n/02o/02p
    chose ONCE by looking at all 5 folds' pooled OOS performance together (N,
    then per-side stop-loss/take-profit) is instead re-chosen INSIDE every
    fold using ONLY that fold's training range, then frozen and applied to
    that fold's held-out test range -- the rigorous version of the same
    search, testing whether the earlier choices are stable across time or an
    artifact of picking on the full sample. Returns pooled OOS stats plus
    each fold's own chosen hyperparameters (for a stability diagnostic)."""
    chunks = _fold_chunks(len(factor), n_folds)
    pooled_ret, pooled_dir, fold_sharpes, fold_choices = [], [], [], []
    for i in range(1, n_folds + 1):
        train_idx, test_idx = np.concatenate(chunks[:i]), chunks[i]
        if len(test_idx) == 0:
            continue
        train_factor = factor.iloc[train_idx]
        direction_full = _direction_for_mode(
            factor, close, mode, quantile,
            lo=train_factor.quantile(1 - quantile), hi=train_factor.quantile(quantile),
            thresh=train_factor.quantile(quantile),
        )
        n_hold = _inner_select_n_hold(direction_full, train_idx, open_, high, low, close, atr, n_hold_grid)

        test_ret_parts, test_dir_parts = [], []
        choice = {"n_hold": n_hold}
        for side in ("long", "short"):
            side_dir = _apply_direction_filter(direction_full, side)
            sl_type, sl_level, rr = _inner_select_sl_tp(side_dir, train_idx, open_, high, low, close,
                                                         atr, n_hold, sl_configs)
            choice[f"{side}_sl_type"], choice[f"{side}_sl_level"], choice[f"{side}_rr"] = sl_type, sl_level, rr
            real_sl_type = None if sl_type == "none" else sl_type
            ret, d = _score_masked(side_dir, test_idx, open_, high, low, close, atr, n_hold,
                                    real_sl_type, sl_level, rr)
            test_ret_parts.append(ret)
            test_dir_parts.append(d)
        fold_ret = np.concatenate(test_ret_parts)
        fold_dir = np.concatenate(test_dir_parts)
        pooled_ret.append(fold_ret)
        pooled_dir.append(fold_dir)
        fold_sharpes.append(_sharpe(fold_ret))
        fold_choices.append(choice)
    result = _pool_fold_results(pooled_ret, pooled_dir, fold_sharpes)
    result["fold_choices"] = fold_choices
    return result


def nested_walk_forward_pair(factor_a: pd.Series, mode_a: str, factor_b: pd.Series, mode_b: str,
                              close: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series,
                              atr: pd.Series, n_hold_grid, sl_configs, quantile: float = 0.8,
                              n_folds: int = 5) -> dict:
    """Pair version of nested_walk_forward_single."""
    chunks = _fold_chunks(len(factor_a), n_folds)
    pooled_ret, pooled_dir, fold_sharpes, fold_choices = [], [], [], []
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
        n_hold = _inner_select_n_hold(combined, train_idx, open_, high, low, close, atr, n_hold_grid)

        test_ret_parts, test_dir_parts = [], []
        choice = {"n_hold": n_hold}
        for side in ("long", "short"):
            side_dir = _apply_direction_filter(combined, side)
            sl_type, sl_level, rr = _inner_select_sl_tp(side_dir, train_idx, open_, high, low, close,
                                                         atr, n_hold, sl_configs)
            choice[f"{side}_sl_type"], choice[f"{side}_sl_level"], choice[f"{side}_rr"] = sl_type, sl_level, rr
            real_sl_type = None if sl_type == "none" else sl_type
            ret, d = _score_masked(side_dir, test_idx, open_, high, low, close, atr, n_hold,
                                    real_sl_type, sl_level, rr)
            test_ret_parts.append(ret)
            test_dir_parts.append(d)
        fold_ret = np.concatenate(test_ret_parts)
        fold_dir = np.concatenate(test_dir_parts)
        pooled_ret.append(fold_ret)
        pooled_dir.append(fold_dir)
        fold_sharpes.append(_sharpe(fold_ret))
        fold_choices.append(choice)
    result = _pool_fold_results(pooled_ret, pooled_dir, fold_sharpes)
    result["fold_choices"] = fold_choices
    return result

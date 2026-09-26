"""Event-driven backtest engine for a long-only ATR-spaced martingale grid
on XAUUSD H1 bars.

Design (v1 baseline, no regime filter):
  - Always net-long bias (buy the dip): with no open layers, open an
    initial long layer. If price rises tp_atr_mult*ATR above the
    lot-weighted average entry, close the whole basket (take profit) and
    reset. If price falls grid_atr_mult*ATR below the last layer's entry
    and max_layers hasn't been reached, add a new layer at multiplier x
    the previous layer's size (classic martingale doubling).
  - Grid spacing and take-profit distance are both ATR(H1, grid_atr_window)
    multiples, not fixed dollar amounts — this matters because gold's price
    (and so ATR) moved roughly 5x over the sample (2009-2026), so a fixed
    grid spacing calibrated to one era would be wildly wrong in another.
  - Costs: spread (half paid on entry, half on exit, i.e. paid once
    round-trip), swap charged once per calendar day at rollover_hour_utc
    for every lot still open overnight.
  - Risk control: a broker-style stop-out — if margin_level (equity /
    used_margin) falls below stop_out_level, everything is force-liquidated
    at the current close, realizing whatever loss has accrued. This is
    what "blowing up" looks like in this simulation; blowups are logged
    separately from ordinary take-profit closes.

This is intentionally a simple, readable Python loop (not vectorized) —
108k H1 bars runs in a few seconds, and a martingale ladder's state
(open layers, running equity) is inherently path-dependent and awkward to
vectorize; clarity here matters more than speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MartingaleConfig:
    initial_lot: float = 0.01
    multiplier: float = 2.0
    max_layers: int = 8
    grid_atr_window: int = 14
    grid_atr_mult: float = 1.0
    tp_atr_mult: float = 1.0
    spread_dollars: float = 0.30     # round-trip cost per oz, split entry/exit
    swap_per_lot_per_day: float = -6.0  # USD, negative = cost, long-side financing
    contract_size: float = 100.0     # oz per standard lot
    leverage: int = 200
    initial_equity: float = 10_000.0
    stop_out_level: float = 0.2      # forced liquidation if equity/used_margin < this
    rollover_hour_utc: int = 21


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    blowups: pd.DataFrame
    max_layers_reached: int
    ruin_time: pd.Timestamp | None  # first time equity hit <= 0; trading stops for good after this


def run_backtest(df: pd.DataFrame, atr: pd.Series, cfg: MartingaleConfig,
                  allow_entry: pd.Series | None = None) -> BacktestResult:
    """allow_entry: optional boolean Series (same index as df) — when
    provided, a NEW LAYER (initial or additional) is only opened on bars
    where it's True; existing layers still manage TP/stop-out normally.
    This is the hook Phase 3b uses to gate entries with the Phase 2b regime
    factors, without touching the core state machine."""
    n = len(df)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    times = df["time"].to_numpy()
    atr_vals = atr.to_numpy()
    allow = allow_entry.to_numpy() if allow_entry is not None else np.ones(n, dtype=bool)

    equity = cfg.initial_equity
    layers: list[tuple[float, float]] = []  # (entry_price, lot)
    max_layers_reached = 0
    last_rollover_day = None
    ruined = False
    ruin_time = None

    equity_curve = np.empty(n)
    trade_records = []
    blowup_records = []

    for i in range(n):
        c, h, lo, a = close[i], high[i], low[i], atr_vals[i]
        if np.isnan(a):
            equity_curve[i] = equity
            continue
        t = pd.Timestamp(times[i])

        if ruined:
            # Account is dead: a real broker would have closed it once equity
            # hit zero. No more new positions, equity just sits flat.
            equity_curve[i] = equity
            continue

        if layers and t.hour == cfg.rollover_hour_utc:
            day = t.date()
            if day != last_rollover_day:
                total_lot = sum(l for _, l in layers)
                equity += cfg.swap_per_lot_per_day * total_lot
                last_rollover_day = day

        if not layers:
            if allow[i]:
                entry_price = c + cfg.spread_dollars / 2
                layers.append((entry_price, cfg.initial_lot))
                max_layers_reached = max(max_layers_reached, 1)
        else:
            total_lot = sum(l for _, l in layers)
            wavg = sum(p * l for p, l in layers) / total_lot
            tp_price = wavg + cfg.tp_atr_mult * a
            grid_price = layers[-1][0] - cfg.grid_atr_mult * a

            if h >= tp_price:
                exit_price = tp_price - cfg.spread_dollars / 2
                pnl = sum((exit_price - p) * l * cfg.contract_size for p, l in layers)
                equity += pnl
                trade_records.append({
                    "time": t, "type": "take_profit", "n_layers": len(layers), "pnl": pnl,
                })
                layers = []
            elif allow[i] and lo <= grid_price and len(layers) < cfg.max_layers:
                new_lot = cfg.initial_lot * (cfg.multiplier ** len(layers))
                layers.append((grid_price + cfg.spread_dollars / 2, new_lot))
                max_layers_reached = max(max_layers_reached, len(layers))

        if layers:
            total_lot = sum(l for _, l in layers)
            unrealized = sum((c - p) * l * cfg.contract_size for p, l in layers)
            used_margin = sum(p * l * cfg.contract_size for p, l in layers) / cfg.leverage
            margin_level = (equity + unrealized) / used_margin if used_margin > 0 else np.inf

            if margin_level < cfg.stop_out_level:
                equity += unrealized
                blowup_records.append({
                    "time": t, "n_layers": len(layers), "pnl": unrealized, "equity_after": equity,
                })
                trade_records.append({
                    "time": t, "type": "stop_out", "n_layers": len(layers), "pnl": unrealized,
                })
                layers = []
                unrealized = 0.0
                if equity <= 0:
                    ruined = True
                    ruin_time = t

            equity_curve[i] = equity + unrealized
        else:
            equity_curve[i] = equity

    return BacktestResult(
        equity_curve=pd.Series(equity_curve, index=df.index),
        trades=pd.DataFrame(trade_records),
        blowups=pd.DataFrame(blowup_records),
        max_layers_reached=max_layers_reached,
        ruin_time=ruin_time,
    )


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    return float(dd.min())

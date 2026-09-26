"""Event-driven backtest engine for a BIDIRECTIONAL ATR-spaced martingale
grid on XAUUSD M5 bars (phase 3b) -- extends martingale.py's long-only
design to run independent long and short grids, gated by direction (not
just permission) from the 12 validated signal-design candidates.

Design:
  - Two independent layer lists (long, short), each with the SAME grid
    logic as the v1 engine mirrored for its side: long adds on price
    falling grid_atr_mult*ATR below the last layer, takes profit on price
    rising tp_atr_mult*ATR above the lot-weighted average; short is the
    mirror (adds on price rising, takes profit on price falling).
  - `direction`: a signed Series (+1 = only long entries allowed, -1 = only
    short, 0 = neither) built from the 12 candidates' reconciled net
    position (see scripts/03b_build_direction_gate.py). This is NOT just a
    permission gate like v1's `allow_entry` -- a signal FLIP (long ->
    short or vice versa) force-closes whatever grid is currently open, at
    the current bar's close, before the new direction's grid is allowed to
    start (mirrors the "opposing signal closes first" interlock policy
    chosen in signal design step 5, now applied at the grid level). A
    transition to/from 0 (flat) does NOT force-close anything -- an existing
    grid still manages its own TP/stop-out normally; only NEW layers are
    gated off while direction is 0, exactly like v1's `allow_entry`.
  - Both directions share one account (equity, margin); a forced reversal
    close realizes whatever P&L the closed grid had at that moment.
  - Everything else (spread, swap, contract size, leverage, stop-out) is
    the same model as v1, with an added `swap_per_lot_per_day_short` since
    financing cost is not symmetric between long and short gold positions
    (v1's swap constant is long-side only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BidirectionalConfig:
    initial_lot: float = 0.01
    multiplier: float = 2.0
    max_layers: int = 8
    grid_atr_window: int = 14
    grid_atr_mult: float = 1.0
    tp_atr_mult: float = 1.0
    spread_dollars: float = 0.30
    swap_per_lot_per_day_long: float = -6.0
    swap_per_lot_per_day_short: float = 2.0  # ASSUMPTION: short gold financing credit on
    # a typical retail ECN account -- not calibrated from a specific broker, flagged for
    # confirmation before this is treated as realistic (see phase 3b report).
    contract_size: float = 100.0
    leverage: int = 200
    initial_equity: float = 10_000.0
    stop_out_level: float = 0.2
    rollover_hour_utc: int = 21


@dataclass
class BidirectionalResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    blowups: pd.DataFrame
    forced_reversals: pd.DataFrame
    max_layers_long: int
    max_layers_short: int
    ruin_time: pd.Timestamp | None


def _grid_pnl(layers: list, exit_price: float, side: int, contract_size: float) -> float:
    return sum((exit_price - p) * l * side * contract_size for p, l in layers)


def run_bidirectional_backtest(df: pd.DataFrame, atr: pd.Series, direction: pd.Series,
                                cfg: BidirectionalConfig) -> BidirectionalResult:
    n = len(df)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    times = df["time"].to_numpy()
    atr_vals = atr.to_numpy()
    dirn = direction.to_numpy()

    equity = cfg.initial_equity
    layers_long: list[tuple[float, float]] = []
    layers_short: list[tuple[float, float]] = []
    max_layers_long = max_layers_short = 0
    last_rollover_day = None
    ruined = False
    ruin_time = None

    equity_curve = np.empty(n)
    trade_records, blowup_records, reversal_records = [], [], []

    def close_grid(layers, side, exit_price, t, reason):
        pnl = _grid_pnl(layers, exit_price, side, cfg.contract_size)
        return pnl, {"time": t, "type": reason, "side": "long" if side > 0 else "short",
                     "n_layers": len(layers), "pnl": pnl}

    for i in range(n):
        c, h, lo, a = close[i], high[i], low[i], atr_vals[i]
        if np.isnan(a):
            equity_curve[i] = equity
            continue
        t = pd.Timestamp(times[i])
        d = dirn[i]

        if ruined:
            equity_curve[i] = equity
            continue

        if (layers_long or layers_short) and t.hour == cfg.rollover_hour_utc:
            day = t.date()
            if day != last_rollover_day:
                if layers_long:
                    equity += cfg.swap_per_lot_per_day_long * sum(l for _, l in layers_long)
                if layers_short:
                    equity += cfg.swap_per_lot_per_day_short * sum(l for _, l in layers_short)
                last_rollover_day = day

        # forced reversal: an opposing signal closes whatever grid is open right now
        if d > 0 and layers_short:
            pnl, rec = close_grid(layers_short, -1, c, t, "forced_reversal")
            equity += pnl
            rec["forced_by"] = "long_signal"
            reversal_records.append(rec)
            trade_records.append({**rec, "type": "forced_reversal"})
            layers_short = []
        elif d < 0 and layers_long:
            pnl, rec = close_grid(layers_long, 1, c, t, "forced_reversal")
            equity += pnl
            rec["forced_by"] = "short_signal"
            reversal_records.append(rec)
            trade_records.append({**rec, "type": "forced_reversal"})
            layers_long = []

        # long grid: open / add / take-profit
        if not layers_long:
            if d > 0:
                entry_price = c + cfg.spread_dollars / 2
                layers_long.append((entry_price, cfg.initial_lot))
                max_layers_long = max(max_layers_long, 1)
        else:
            total_lot = sum(l for _, l in layers_long)
            wavg = sum(p * l for p, l in layers_long) / total_lot
            tp_price = wavg + cfg.tp_atr_mult * a
            grid_price = layers_long[-1][0] - cfg.grid_atr_mult * a
            if h >= tp_price:
                exit_price = tp_price - cfg.spread_dollars / 2
                pnl = _grid_pnl(layers_long, exit_price, 1, cfg.contract_size)
                equity += pnl
                trade_records.append({"time": t, "type": "take_profit", "side": "long",
                                       "n_layers": len(layers_long), "pnl": pnl})
                layers_long = []
            elif d > 0 and lo <= grid_price and len(layers_long) < cfg.max_layers:
                new_lot = cfg.initial_lot * (cfg.multiplier ** len(layers_long))
                layers_long.append((grid_price + cfg.spread_dollars / 2, new_lot))
                max_layers_long = max(max_layers_long, len(layers_long))

        # short grid: open / add / take-profit (mirror of long)
        if not layers_short:
            if d < 0:
                entry_price = c - cfg.spread_dollars / 2
                layers_short.append((entry_price, cfg.initial_lot))
                max_layers_short = max(max_layers_short, 1)
        else:
            total_lot = sum(l for _, l in layers_short)
            wavg = sum(p * l for p, l in layers_short) / total_lot
            tp_price = wavg - cfg.tp_atr_mult * a
            grid_price = layers_short[-1][0] + cfg.grid_atr_mult * a
            if lo <= tp_price:
                exit_price = tp_price + cfg.spread_dollars / 2
                pnl = _grid_pnl(layers_short, exit_price, -1, cfg.contract_size)
                equity += pnl
                trade_records.append({"time": t, "type": "take_profit", "side": "short",
                                       "n_layers": len(layers_short), "pnl": pnl})
                layers_short = []
            elif d < 0 and h >= grid_price and len(layers_short) < cfg.max_layers:
                new_lot = cfg.initial_lot * (cfg.multiplier ** len(layers_short))
                layers_short.append((grid_price - cfg.spread_dollars / 2, new_lot))
                max_layers_short = max(max_layers_short, len(layers_short))

        if layers_long or layers_short:
            unrealized = (_grid_pnl(layers_long, c, 1, cfg.contract_size) +
                          _grid_pnl(layers_short, c, -1, cfg.contract_size))
            used_margin = (sum(p * l for p, l in layers_long) + sum(p * l for p, l in layers_short)) \
                * cfg.contract_size / cfg.leverage
            margin_level = (equity + unrealized) / used_margin if used_margin > 0 else np.inf

            if margin_level < cfg.stop_out_level:
                equity += unrealized
                blowup_records.append({"time": t, "n_layers_long": len(layers_long),
                                        "n_layers_short": len(layers_short), "pnl": unrealized,
                                        "equity_after": equity})
                trade_records.append({"time": t, "type": "stop_out", "side": "both",
                                       "n_layers": len(layers_long) + len(layers_short), "pnl": unrealized})
                layers_long, layers_short = [], []
                unrealized = 0.0
                if equity <= 0:
                    ruined = True
                    ruin_time = t

            equity_curve[i] = equity + unrealized
        else:
            equity_curve[i] = equity

    return BidirectionalResult(
        equity_curve=pd.Series(equity_curve, index=df.index),
        trades=pd.DataFrame(trade_records),
        blowups=pd.DataFrame(blowup_records),
        forced_reversals=pd.DataFrame(reversal_records),
        max_layers_long=max_layers_long,
        max_layers_short=max_layers_short,
        ruin_time=ruin_time,
    )

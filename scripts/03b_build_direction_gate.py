#!/usr/bin/env python3
"""Phase 3b, part 1: build the per-M5-bar direction gate for the
bidirectional martingale engine, from the 12 candidates validated by the
signal-design track (steps 1-8, reports/02v_final_signal_specification.md).

Reuses each candidate's trade_windows (full-sample-fitted thresholds, its
own step-7 final N/per-side stop-loss/take-profit -- these define how long
a trigger's directional view is considered "active", not what exits the
martingale grid) and the step-5 net-position interlock
(execution.py::simulate_net_position) to reconcile all 12 candidates' trade
windows into ONE net directional state per bar: +1 (long view active), -1
(short), 0 (flat -- no candidate currently has an open view). This is the
`direction` series martingale_bidirectional.run_bidirectional_backtest
consumes.

Usage:
    python scripts/03b_build_direction_gate.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, atr as atr_fn  # noqa: E402
from factors.direction import reversion_direction, combine_directions  # noqa: E402
from factors.execution import trade_windows, simulate_net_position  # noqa: E402

ATR_PERIOD = 14
QUANTILE = 0.8

SINGLES = [
    ("vol_of_vol", 100, None), ("adx", 50, 24000), ("bb_width", 100, None),
    ("bb_width", 300, 6000), ("garman_klass_vol", 20, 6000), ("garman_klass_vol", 240, None),
]
PAIRS = [
    ("adx_50_pctrank2000", "autocorr_returns_50_pctrank2000"),
    ("kurt_returns_100", "mean_reversion_speed_50_pctrank500"),
    ("realized_vol_100_pctrank2000", "variance_ratio_2_50_pctrank2000"),
    ("parkinson_vol_100_pctrank2000", "aroon_up_10_pctrank500"),
    ("avg_gap_50", "roc_10"),
    ("stochastic_d_100_pctrank2000", "keltner_width_20"),
]


def variant_name(family, n, pw):
    return f"{family}_{n}" if pw is None else f"{family}_{n}_pctrank{pw}"


def full_sample_direction(factor: pd.Series, quantile: float = QUANTILE) -> pd.Series:
    return reversion_direction(factor, factor.quantile(1 - quantile), factor.quantile(quantile))


def sl_args(row, side):
    sl_type = row[f"{side}_sl_type"]
    sl_type = None if sl_type == "none" else sl_type
    sl_level = None if pd.isna(row[f"{side}_sl_level"]) else float(row[f"{side}_sl_level"])
    rr = None if pd.isna(row[f"{side}_rr"]) else float(row[f"{side}_rr"])
    return sl_type, sl_level, rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/3] Loading M5 OHLC + factors_M5.parquet + step7 final parameters ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    assert df["close"].equals(factors["close"]), "row alignment mismatch"
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    atr14 = atr_fn(df, ATR_PERIOD)
    final = pd.read_csv(os.path.join(args.report_dir, "02t_final_parameters.csv")).set_index("name")

    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[variant_name(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    def candidate_trades(name, direction_full, n_hold, row):
        parts = []
        for side, sign in (("long", 1), ("short", -1)):
            side_dir = direction_full.where(np.sign(direction_full) == sign, 0.0)
            sl_type, sl_level, rr = sl_args(row, side)
            tw = trade_windows(side_dir, open_, high, low, close, atr14, n_hold, sl_type, sl_level, rr)
            parts.append(tw)
        tw = pd.concat(parts, ignore_index=True)
        tw["source"] = name
        return tw

    print("[2/3] Building trade windows for all 12 candidates ...")
    all_trades = []
    for family, n, pw in SINGLES:
        name = variant_name(family, n, pw)
        direction_full = full_sample_direction(single_factors[name])
        row = final.loc[name]
        tw = candidate_trades(name, direction_full, int(row["n_hold_median"]), row)
        all_trades.append(tw)
        print(f"      {name}: {len(tw)} trades")
    for var_a, var_b in PAIRS:
        name = f"{var_a}+{var_b}"
        dir_a, dir_b = full_sample_direction(factors[var_a]), full_sample_direction(factors[var_b])
        combined = combine_directions(dir_a, dir_b)
        row = final.loc[name]
        tw = candidate_trades(name, combined, int(row["n_hold_median"]), row)
        all_trades.append(tw)
        print(f"      {name}: {len(tw)} trades")

    pooled = pd.concat(all_trades, ignore_index=True)
    print(f"[3/3] Reconciling {len(pooled)} pooled trades into one net position (opposing signal closes first) ...")
    realized = simulate_net_position(pooled)
    print(f"      {len(realized)} realized net-position trades, "
          f"{int(realized['forced_close'].sum())} forced closes")

    n_rows = len(df)
    gate = np.zeros(n_rows, dtype=np.int8)
    for row in realized.itertuples(index=False):
        gate[int(row.entry_idx):int(row.actual_exit_idx) + 1] = int(np.sign(row.direction))

    frac_long = float((gate > 0).mean())
    frac_short = float((gate < 0).mean())
    print(f"      direction gate: {frac_long:.1%} of bars long, {frac_short:.1%} short, "
          f"{1 - frac_long - frac_short:.1%} flat")

    out = pd.DataFrame({"time": df["time"], "direction": gate})
    out_path = os.path.join(args.clean_dir, "direction_gate_M5.parquet")
    out.to_parquet(out_path, index=False)
    realized.to_csv(os.path.join(args.report_dir, "03b_direction_gate_trades.csv"), index=False)
    print(f"Wrote {out_path} ({frac_long:.1%} long / {frac_short:.1%} short / "
          f"{1 - frac_long - frac_short:.1%} flat)")


if __name__ == "__main__":
    main()

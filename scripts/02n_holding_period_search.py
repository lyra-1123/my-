#!/usr/bin/env python3
"""Signal design step 3, part 1: search the holding period N (in M5 bars)
for each of the 17 validated candidates (8 singles from 02k + 9 pairs from
02i), using a REAL entry/exit simulation for the first time in this track:
entry at next bar's open, exit at the close of the N-th held bar (no stop-
loss/take-profit yet -- that's 02o, once N is chosen here).

This replaces the placeholder HOLDING_BARS=12 used everywhere from 02f
through 02m with an actual per-candidate search over N in {6,12,24,48,96,192}
M5 bars (30min .. 16h), full grid, each candidate keeping its own best N
(per the user's choice: full grid search, not one shared N).

Usage:
    python scripts/02n_holding_period_search.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, atr as atr_fn  # noqa: E402
from factors.execution import walk_forward_execution_single, walk_forward_execution_pair  # noqa: E402

N_HOLD_GRID = (6, 12, 24, 48, 96, 192)
ATR_PERIOD = 14
MIN_ANNUAL_RATE = 20  # pairs
MIN_LONG_SHORT = 30  # singles (matches 02h's own bar)

# 8 singles: (family, n, pctrank_window_m5_or_None) -- 02k's winning windows
SINGLES = [
    ("vol_of_vol", 100, None), ("adx", 50, 24000), ("bb_width", 100, None),
    ("bb_width", 300, 6000), ("parkinson_vol", 100, 24000), ("garman_klass_vol", 20, 6000),
    ("parkinson_vol", 60, 6000), ("garman_klass_vol", 240, None),
]

# 9 pairs: original H1-bar-count-replica variant names (exist directly in factors_M5.parquet)
PAIRS = [
    ("adx_50_pctrank2000", "autocorr_returns_50_pctrank2000"),
    ("avg_gap_50", "garman_klass_vol_20_pctrank500"),
    ("kurt_returns_100", "mean_reversion_speed_50_pctrank500"),
    ("realized_vol_100_pctrank2000", "roc_10"),
    ("realized_vol_100_pctrank2000", "variance_ratio_2_50_pctrank2000"),
    ("parkinson_vol_100_pctrank2000", "aroon_up_10_pctrank500"),
    ("avg_gap_50", "zscore_vs_ma_100_pctrank500"),
    ("avg_gap_50", "roc_10"),
    ("stochastic_d_100_pctrank2000", "keltner_width_20"),
]


def variant_name(family, n, pw):
    return f"{family}_{n}" if pw is None else f"{family}_{n}_pctrank{pw}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading M5 OHLC + factors_M5.parquet ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    assert df["close"].equals(factors["close"]), "row alignment mismatch between raw M5 bars and factors_M5.parquet"
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    atr14 = atr_fn(df, ATR_PERIOD)
    years = (len(df) * 5 / 60 / 24) / 365.25

    print("[2/4] Building the 8 optimized single-factor variants ...")
    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    print(f"[3/4] Sweeping N in {N_HOLD_GRID} for 8 singles + 9 pairs ...")
    rows = []
    for family, n, pw in SINGLES:
        name = variant_name(family, n, pw)
        for n_hold in N_HOLD_GRID:
            wf = walk_forward_execution_single(single_factors[(family, n, pw)], close, open_, high, low,
                                                atr14, "reversion", n_hold)
            annual_rate = (wf["n_long"] + wf["n_short"]) / years
            ann_sharpe = wf["oos_sharpe_all"] * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")
            rows.append({"kind": "single", "name": name, "n_hold": n_hold, "hours": n_hold * 5 / 60,
                         "oos_sharpe_all": wf["oos_sharpe_all"], "ann_sharpe": ann_sharpe,
                         "annual_rate": annual_rate, "n_long": wf["n_long"], "n_short": wf["n_short"],
                         "n_folds_positive": wf["n_folds_positive"]})
        print(f"      {name} done")

    for var_a, var_b in PAIRS:
        name = f"{var_a}+{var_b}"
        fa, fb = factors[var_a], factors[var_b]
        for n_hold in N_HOLD_GRID:
            wf = walk_forward_execution_pair(fa, "reversion", fb, "reversion", close, open_, high, low,
                                              atr14, n_hold)
            annual_rate = (wf["n_long"] + wf["n_short"]) / years
            ann_sharpe = wf["oos_sharpe_all"] * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")
            rows.append({"kind": "pair", "name": name, "n_hold": n_hold, "hours": n_hold * 5 / 60,
                         "oos_sharpe_all": wf["oos_sharpe_all"], "ann_sharpe": ann_sharpe,
                         "annual_rate": annual_rate, "n_long": wf["n_long"], "n_short": wf["n_short"],
                         "n_folds_positive": wf["n_folds_positive"]})
        print(f"      {name} done")

    scan_df = pd.DataFrame(rows)
    scan_path = os.path.join(args.report_dir, "02n_holding_period_scan.csv")
    scan_df.to_csv(scan_path, index=False)

    print("[4/4] Picking best N per candidate ...")
    winners = []
    for name, grp in scan_df.groupby("name", sort=False):
        kind = grp["kind"].iloc[0]
        if kind == "single":
            usable = grp[(grp["n_folds_positive"] >= 3) & (grp["n_long"] >= MIN_LONG_SHORT) & (grp["n_short"] >= MIN_LONG_SHORT)]
        else:
            usable = grp[(grp["n_folds_positive"] >= 3) & (grp["annual_rate"] >= MIN_ANNUAL_RATE)]
        pool = usable if len(usable) else grp
        best = pool.loc[pool["ann_sharpe"].idxmax()]
        winners.append({"kind": kind, "name": name, "best_n_hold": int(best["n_hold"]),
                         "best_hours": best["hours"], "ann_sharpe": best["ann_sharpe"],
                         "annual_rate": best["annual_rate"], "n_folds_positive": best["n_folds_positive"],
                         "met_usability_bar": bool(len(usable))})
    winners_df = pd.DataFrame(winners)
    winners_path = os.path.join(args.report_dir, "02n_holding_period_winners.csv")
    winners_df.to_csv(winners_path, index=False)

    lines = [
        "# 持有期N搜索报告(信号设计步骤3-1)", "",
        "## 方法", "",
        "第一次用真实的入场出场规则打分(此前所有walk-forward都是用fwd_return在触发的那根"
        "bar直接算固定horizon收益，不是真实的交易模拟)：**下一根bar开盘价入场**，"
        "**持有到第N根bar收盘价出场**(暂不设止损止盈，止损止盈是下一步02o的工作)。"
        f"对8个单因子+9对组合各自在N∈{N_HOLD_GRID}(根M5，即30min~16h)网格上搜索，"
        "每个候选自己选自己的最优N(不强制共用一个N)。", "",
        "## 完整扫描结果", "",
        "| 候选 | N(根/小时) | OOS Sharpe(年化) | 年触发 | 折数为正 |",
        "|---|---|---|---|---|",
    ]
    for _, r in scan_df.iterrows():
        lines.append(f"| {r['name']} | {r['n_hold']:.0f}/{r['hours']:.1f}h | {r['ann_sharpe']:+.2f} | "
                      f"{r['annual_rate']:.0f} | {r['n_folds_positive']}/5 |")

    lines += ["", "## 每个候选选出的最优N", "",
              "| 候选 | 类型 | 最优N(根/小时) | OOS Sharpe(年化) | 年触发 | 达标 |",
              "|---|---|---|---|---|---|"]
    for _, r in winners_df.iterrows():
        lines.append(f"| {r['name']} | {r['kind']} | {r['best_n_hold']:.0f}/{r['best_hours']:.1f}h | "
                      f"{r['ann_sharpe']:+.2f} | {r['annual_rate']:.0f} | {'是' if r['met_usability_bar'] else '否(全组都不达标,仅取最优)'} |")
    lines += ["", f"完整数据：{scan_path}；最优N汇总：{winners_path}（02o止损止盈搜索会读取这份文件）。", ""]

    report_path = os.path.join(args.report_dir, "02n_holding_period_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

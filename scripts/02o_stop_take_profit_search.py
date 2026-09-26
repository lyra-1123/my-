#!/usr/bin/env python3
"""Signal design step 3, part 2: stop-loss / take-profit search, using each
candidate's own best N from 02n. Per the user's explicit ask: test BOTH a
fixed-ATR stop and a fixed-percentage stop, AND compare against no stop-loss
at all. Take-profit is symmetric: TP distance = SL distance x a fixed
risk:reward ratio (only meaningful once a stop defines the risk unit).

Grid: {no-SL baseline} + {ATR multiple in 0.5/1/1.5/2} x {RR in 1/1.5/2}
                        + {pct level in 0.2%/0.3%/0.5%/1%} x {RR in 1/1.5/2}
= 1 + 12 + 12 = 25 configurations per candidate.

Usage:
    python scripts/02o_stop_take_profit_search.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, atr as atr_fn  # noqa: E402
from factors.execution import walk_forward_execution_single, walk_forward_execution_pair  # noqa: E402

ATR_PERIOD = 14
ATR_MULTIPLES = (0.5, 1.0, 1.5, 2.0)
PCT_LEVELS = (0.002, 0.003, 0.005, 0.01)
RR_RATIOS = (1.0, 1.5, 2.0)
MIN_ANNUAL_RATE = 20
MIN_LONG_SHORT = 30

SINGLES = [
    ("vol_of_vol", 100, None), ("adx", 50, 24000), ("bb_width", 100, None),
    ("bb_width", 300, 6000), ("parkinson_vol", 100, 24000), ("garman_klass_vol", 20, 6000),
    ("parkinson_vol", 60, 6000), ("garman_klass_vol", 240, None),
]
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
    parser.add_argument("--n-hold-file", default="reports/02n_holding_period_winners.csv")
    args = parser.parse_args()

    print("[1/4] Loading M5 OHLC + factors_M5.parquet + chosen N per candidate ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    assert df["close"].equals(factors["close"]), "row alignment mismatch"
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    atr14 = atr_fn(df, ATR_PERIOD)
    years = (len(df) * 5 / 60 / 24) / 365.25
    n_hold_by_name = pd.read_csv(args.n_hold_file).set_index("name")["best_n_hold"].to_dict()

    print("[2/4] Building the 8 optimized single-factor variants ...")
    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[variant_name(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    configs = [("none", None, None)]
    for a in ATR_MULTIPLES:
        for rr in RR_RATIOS:
            configs.append(("atr", a, rr))
    for p in PCT_LEVELS:
        for rr in RR_RATIOS:
            configs.append(("pct", p, rr))

    print(f"[3/4] Sweeping {len(configs)} SL/TP configs x 17 candidates (each at its own best N) ...")
    rows = []

    def run_and_record(name, kind, wf_fn, *wf_args):
        n_hold = n_hold_by_name[name]
        for sl_type, sl_level, rr in configs:
            kwargs = {} if sl_type == "none" else {"sl_type": sl_type, "sl_level": sl_level, "rr_ratio": rr}
            wf = wf_fn(*wf_args, n_hold, **kwargs)
            annual_rate = (wf["n_long"] + wf["n_short"]) / years
            ann_sharpe = wf["oos_sharpe_all"] * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")
            rows.append({
                "kind": kind, "name": name, "n_hold": n_hold, "sl_type": sl_type, "sl_level": sl_level,
                "rr_ratio": rr, "oos_sharpe_all": wf["oos_sharpe_all"], "ann_sharpe": ann_sharpe,
                "annual_rate": annual_rate, "n_long": wf["n_long"], "n_short": wf["n_short"],
                "n_folds_positive": wf["n_folds_positive"],
            })

    for family, n, pw in SINGLES:
        name = variant_name(family, n, pw)
        run_and_record(name, "single", walk_forward_execution_single,
                        single_factors[name], close, open_, high, low, atr14, "reversion")
        print(f"      {name} done")

    for var_a, var_b in PAIRS:
        name = f"{var_a}+{var_b}"
        run_and_record(name, "pair", walk_forward_execution_pair,
                        factors[var_a], "reversion", factors[var_b], "reversion", close, open_, high, low, atr14)
        print(f"      {name} done")

    scan_df = pd.DataFrame(rows)
    scan_path = os.path.join(args.report_dir, "02o_stop_take_profit_scan.csv")
    scan_df.to_csv(scan_path, index=False)

    print("[4/4] Picking best config per candidate + no-SL vs with-SL comparison ...")
    winners = []
    for name, grp in scan_df.groupby("name", sort=False):
        kind = grp["kind"].iloc[0]
        baseline = grp[grp["sl_type"] == "none"].iloc[0]
        if kind == "single":
            usable = grp[(grp["n_folds_positive"] >= 3) & (grp["n_long"] >= MIN_LONG_SHORT) & (grp["n_short"] >= MIN_LONG_SHORT)]
        else:
            usable = grp[(grp["n_folds_positive"] >= 3) & (grp["annual_rate"] >= MIN_ANNUAL_RATE)]
        pool = usable if len(usable) else grp
        best = pool.loc[pool["ann_sharpe"].idxmax()]
        winners.append({
            "kind": kind, "name": name, "n_hold": baseline["n_hold"],
            "baseline_no_sl_ann_sharpe": baseline["ann_sharpe"],
            "best_sl_type": best["sl_type"], "best_sl_level": best["sl_level"], "best_rr": best["rr_ratio"],
            "best_ann_sharpe": best["ann_sharpe"], "best_annual_rate": best["annual_rate"],
            "best_folds_positive": best["n_folds_positive"],
            "sl_helps": bool(best["sl_type"] != "none" and best["ann_sharpe"] > baseline["ann_sharpe"]),
        })
    winners_df = pd.DataFrame(winners)
    winners_path = os.path.join(args.report_dir, "02o_stop_take_profit_winners.csv")
    winners_df.to_csv(winners_path, index=False)

    n_helped = winners_df["sl_helps"].sum()
    lines = [
        "# 止损止盈搜索报告(信号设计步骤3-2)", "",
        "## 方法", "",
        f"每个候选用02n选出的自己的最优N，搜索{len(configs)}种止损止盈组合：不设止损(基线，"
        "只有持有到期这一种出场) + ATR倍数止损(0.5x/1x/1.5x/2x的ATR14) x 盈亏比(1:1/1.5:1/2:1) "
        "+ 固定百分比止损(0.2%/0.3%/0.5%/1%) x 盈亏比(同上)。止盈距离=止损距离x盈亏比，"
        "同一根bar内先摸到止损位还是止盈位无法区分时按止损优先(保守假设)。入场=触发后下一根"
        "bar开盘价，出场=止损/止盈/持有到期三者中先发生的那个。", "",
        "## 每个候选：不设止损 vs 最优止损止盈配置", "",
        "| 候选 | N(根) | 无止损年化Sharpe | 最优止损类型 | 止损水平 | 盈亏比 | "
        "最优配置年化Sharpe | 年触发 | 折数为正 | 止损止盈是否有帮助 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in winners_df.sort_values("best_ann_sharpe", ascending=False).iterrows():
        lines.append(f"| {r['name']} | {r['n_hold']:.0f} | {r['baseline_no_sl_ann_sharpe']:+.2f} | "
                      f"{r['best_sl_type']} | {r['best_sl_level']} | {r['best_rr']} | "
                      f"{r['best_ann_sharpe']:+.2f} | {r['best_annual_rate']:.0f} | "
                      f"{r['best_folds_positive']}/5 | {'是' if r['sl_helps'] else '否'} |")

    lines += ["", f"**{n_helped}/{len(winners_df)}个候选加止损止盈后比不设止损更好**（按walk-forward "
              "年化OOS Sharpe比较，同一个N下）。", "",
              "## 结论与下一步", "",
              f"- 完整数据：{scan_path}；每候选最优配置：{winners_path}。",
              "- 下一步（步骤4）：多空分方向验证这批最终选定的止损止盈参数是否对多空都稳健。",
              ""]

    report_path = os.path.join(args.report_dir, "02o_stop_take_profit_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Signal design: M5-native window optimization for the factors already
validated by 02h (8 single factors) and 02i (9 pairs).

02g/02h/02i reused the H1-validated bar-count windows (e.g. "_100" = 100
bars) directly on M5, where the same bar COUNT is a very different absolute
time span (100 H1 bars = ~100h, 100 M5 bars = ~8.3h). Rather than rebuilding
the entire ~400-variant library at multiple M5-native windows (OOM'd at
15GB RAM — 32 families x 7 windows x 3 pctrank-variants is too much held in
memory at once), this script narrowly sweeps candidate windows ONLY for the
~16 factor families that actually appear in a validated candidate, testing
multipliers of the original H1 bar count: x1 (current, likely too short a
span on M5), x3, x6, x12 (same absolute time span as the original H1
window). Percentile-rank lookbacks (where used) are scaled x12 throughout
(same absolute-time-span fix, held fixed rather than swept, to keep the
search tractable).

Usage:
    python scripts/02j_window_optimization.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS  # noqa: E402
from factors.validation import decision_points  # noqa: E402
from factors.direction import walk_forward_direction  # noqa: E402

HOLDING_BARS = 12  # unchanged placeholder, same as 02h/02i, real value pending step 3
MULTIPLIERS = (1, 3, 6, 12)
PCTRANK_SCALE = 12  # scale the H1-validated pctrank lookback to the same absolute time span on M5
MIN_ANNUAL_RATE = 20
MIN_FOLDS_POSITIVE = 3

# (family, base_window_on_H1, pctrank_window_on_H1_or_None) for every factor
# appearing in a final-validated 02h single or 02i pair candidate.
SPECS = [
    ("vol_of_vol", 100, None),
    ("adx", 50, 2000),
    ("bb_width", 100, None),
    ("bb_width", 100, 500),
    ("parkinson_vol", 100, 2000),
    ("parkinson_vol", 20, 500),
    ("garman_klass_vol", 20, 500),
    ("garman_klass_vol", 20, None),
    ("autocorr_returns", 50, 2000),
    ("avg_gap", 50, None),
    ("kurt_returns", 100, None),
    ("mean_reversion_speed", 50, 500),
    ("realized_vol", 100, 2000),
    ("roc", 10, None),
    ("variance_ratio_2", 50, 2000),
    ("aroon_up", 10, 500),
    ("zscore_vs_ma", 100, 500),
    ("stochastic_d", 100, 2000),
    ("keltner_width", 20, None),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading M5 bars ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"])
    df = df.reset_index(drop=True)
    close = df["close"]
    years = (len(df) * 5 / 60 / 24) / 365.25
    print(f"      {len(df):,} M5 bars, {years:.1f} years")

    fwd_return = close.pct_change(HOLDING_BARS).shift(-HOLDING_BARS)
    points = decision_points(len(df), HOLDING_BARS)
    points = points[fwd_return.notna().to_numpy()[points]]

    print(f"[2/4] Sweeping {len(SPECS)} specs x {len(MULTIPLIERS)} window multipliers "
          f"({MULTIPLIERS}) ...")
    rows = []
    for family, n0, pw0 in SPECS:
        func = WINDOWED_FACTORS[family]
        for m in MULTIPLIERS:
            n = max(2, n0 * m)
            raw = func(df, n)
            if pw0 is not None:
                pw = pw0 * PCTRANK_SCALE
                factor = raw.rolling(pw).rank(pct=True)
                variant = f"{family}_{n}_pctrank{pw}"
            else:
                pw = None
                factor = raw
                variant = f"{family}_{n}"
            if factor.isna().all():
                continue
            wf = walk_forward_direction(factor, close, fwd_return, points, mode="reversion", n_folds=5)
            annual_rate = (wf["n_long"] + wf["n_short"]) / years
            ann_sharpe = wf["oos_sharpe_all"] * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")
            rows.append({
                "family": family, "pctrank_h1": pw0, "multiplier": m,
                "base_window_h1": n0, "window_m5": n, "window_hours": n * 5 / 60,
                "pctrank_window_m5": pw, "variant": variant,
                "oos_sharpe_all": wf["oos_sharpe_all"], "ann_sharpe": ann_sharpe,
                "annual_rate": annual_rate, "n_folds_positive": wf["n_folds_positive"],
            })
            print(f"      {variant}: ann_sharpe={ann_sharpe:+.2f} annual_rate={annual_rate:.0f} "
                  f"folds+={wf['n_folds_positive']}/5")

    scan_df = pd.DataFrame(rows)
    scan_path = os.path.join(args.report_dir, "02j_window_scan.csv")
    scan_df.to_csv(scan_path, index=False)

    print("[3/4] Picking winning window per spec ...")
    winners = []
    for (family, pw0), grp in scan_df.groupby(["family", "pctrank_h1"], dropna=False):
        original = grp[grp["multiplier"] == 1]
        usable = grp[(grp["n_folds_positive"] >= MIN_FOLDS_POSITIVE) & (grp["annual_rate"] >= MIN_ANNUAL_RATE)]
        pool = usable if len(usable) else grp
        winner = pool.loc[pool["ann_sharpe"].abs().idxmax()]
        orig_row = original.iloc[0] if len(original) else None
        winners.append({
            "family": family, "pctrank_h1": pw0,
            "original_variant": orig_row["variant"] if orig_row is not None else None,
            "original_ann_sharpe": orig_row["ann_sharpe"] if orig_row is not None else float("nan"),
            "winning_variant": winner["variant"], "winning_multiplier": winner["multiplier"],
            "winning_window_m5": winner["window_m5"], "winning_ann_sharpe": winner["ann_sharpe"],
            "winning_annual_rate": winner["annual_rate"], "winning_folds_positive": winner["n_folds_positive"],
            "met_usability_bar": bool(len(usable)),
        })
    winners_df = pd.DataFrame(winners).sort_values("family")
    winners_path = os.path.join(args.report_dir, "02j_window_winners.csv")
    winners_df.to_csv(winners_path, index=False)

    lines = [
        "# M5窗口参数优化报告", "",
        "## 方法", "",
        f"02h/02i直接搬用了H1上验证过的bar数窗口，但相同bar数在M5上代表的绝对时间跨度"
        f"缩小了12倍(M5是H1的1/12)。这里只对02h(8个单因子)+02i(9个组合)里**实际出现过"
        f"的{len(SPECS)}个因子规格**做窗口扫描（不是重建整个约400变体的因子库——那样会"
        f"在15GB内存限制下OOM），倍数扫描{MULTIPLIERS}(x1=当前用的原始bar数/偏短，"
        f"x12=与H1同等的绝对时间跨度)。有pctrank的因子，其历史分位回看窗口固定按x12"
        f"换算(与基础窗口的换算逻辑一致)，不在这次扫描的维度里。",
        "",
        f"持有期仍用占位值{HOLDING_BARS}根M5，与02h/02i保持一致以便直接比较。",
        "",
        "## 每个因子的窗口扫描结果", "",
    ]
    for (family, pw0), grp in scan_df.groupby(["family", "pctrank_h1"], dropna=False):
        pw_label = f"，pctrank({pw0}h1->{int(pw0*PCTRANK_SCALE)}m5)" if pw0 is not None else ""
        lines.append(f"**{family}**(H1原窗口{grp['base_window_h1'].iloc[0]}{pw_label})：")
        lines += ["", "| 倍数 | M5窗口(bar/小时) | OOS Sharpe(年化) | 年触发 | 折数为正 |",
                   "|---|---|---|---|---|"]
        for _, r in grp.sort_values("multiplier").iterrows():
            lines.append(f"| x{r['multiplier']} | {r['window_m5']:.0f}bar/{r['window_hours']:.1f}h | "
                          f"{r['ann_sharpe']:+.2f} | {r['annual_rate']:.0f} | {r['n_folds_positive']}/5 |")
        lines.append("")

    lines += ["## 优化结果汇总：每个因子的最终获胜窗口", "",
               "| 因子 | 原窗口(x1,当前用的) | 原ann Sharpe | 获胜窗口 | 获胜ann Sharpe | 年触发 | 达标(折数>=3且年触发>=20) |",
               "|---|---|---|---|---|---|---|"]
    for _, r in winners_df.iterrows():
        lines.append(f"| {r['family']} | {r['original_variant']} | {r['original_ann_sharpe']:+.2f} | "
                      f"{r['winning_variant']} | {r['winning_ann_sharpe']:+.2f} | "
                      f"{r['winning_annual_rate']:.0f} | {'是' if r['met_usability_bar'] else '否(全组都不达标，仅取最优)'} |")
    lines.append("")

    improved = (winners_df["winning_ann_sharpe"] > winners_df["original_ann_sharpe"]).sum()
    lines += [
        "## 结论与下一步", "",
        f"- {improved}/{len(winners_df)}个因子在换成M5原生窗口后年化OOS Sharpe有提升。",
        f"- 完整扫描数据：{scan_path}；获胜窗口汇总：{winners_path}。",
        "- 下一步：用每个因子的获胜窗口重建一个小规模(仅这些变体)的优化版M5因子表，"
        "重新跑02h式单因子验证和02i式两两组合验证，得到最终优化后的候选列表，"
        "再据此产出因子含义/参数汇总表。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02j_window_optimization_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[4/4] Wrote {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Signal design: re-validate 02h's 8 single-factor candidates and 02i's 9
pair candidates using each factor's WINNING window from 02j (M5-native
window optimization), instead of the H1-bar-count windows used originally.

Builds only the ~19 winning variants directly (not the full library), so
this stays well within memory, then reruns the same walk-forward validation
as 02h (single) / 02i (pair) on them.

Usage:
    python scripts/02k_optimized_validation.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS  # noqa: E402
from factors.validation import decision_points  # noqa: E402
from factors.direction import walk_forward_direction, walk_forward_direction_pair  # noqa: E402

HOLDING_BARS = 12

# (family, pctrank_h1_key) -> winning (n, pw) from reports/02j_window_winners.csv
WINNERS = {
    ("vol_of_vol", None): (100, None),
    ("adx", 2000): (50, 24000),
    ("bb_width", None): (100, None),
    ("bb_width", 500): (300, 6000),
    ("parkinson_vol", 2000): (100, 24000),
    ("garman_klass_vol", 500): (20, 6000),
    ("parkinson_vol", 500): (60, 6000),
    ("garman_klass_vol", None): (240, None),
    ("autocorr_returns", 2000): (50, 24000),
    ("avg_gap", None): (300, None),
    ("kurt_returns", None): (600, None),
    ("mean_reversion_speed", 500): (50, 6000),
    ("realized_vol", 2000): (100, 24000),
    ("roc", None): (10, None),
    ("variance_ratio_2", 2000): (300, 24000),
    ("aroon_up", 500): (10, 6000),
    ("zscore_vs_ma", 500): (600, 6000),
    ("stochastic_d", 2000): (300, 24000),
    ("keltner_width", None): (20, None),
}

# original 02h single candidates: (family, pctrank_h1_key)
SINGLES = [
    ("vol_of_vol", None), ("adx", 2000), ("bb_width", None), ("bb_width", 500),
    ("parkinson_vol", 2000), ("garman_klass_vol", 500), ("parkinson_vol", 500),
    ("garman_klass_vol", None),
]

# original 02i final validated pairs: ((family_a, pctrank_a), (family_b, pctrank_b))
PAIRS = [
    (("adx", 2000), ("autocorr_returns", 2000)),
    (("avg_gap", None), ("garman_klass_vol", 500)),
    (("kurt_returns", None), ("mean_reversion_speed", 500)),
    (("realized_vol", 2000), ("roc", None)),
    (("realized_vol", 2000), ("variance_ratio_2", 2000)),
    (("parkinson_vol", 2000), ("aroon_up", 500)),
    (("avg_gap", None), ("zscore_vs_ma", 500)),
    (("avg_gap", None), ("roc", None)),
    (("stochastic_d", 2000), ("keltner_width", None)),
]

MIN_ANNUAL_RATE = 20


def sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / x.std(ddof=1)


def variant_name(family: str, n: int, pw) -> str:
    return f"{family}_{n}" if pw is None else f"{family}_{n}_pctrank{pw}"


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

    fwd_return = close.pct_change(HOLDING_BARS).shift(-HOLDING_BARS)
    points = decision_points(len(df), HOLDING_BARS)
    points = points[fwd_return.notna().to_numpy()[points]]

    print(f"[2/4] Building {len(WINNERS)} winning-window factor variants ...")
    factors = {}
    for (family, pw0), (n, pw) in WINNERS.items():
        raw = WINDOWED_FACTORS[family](df, n)
        factors[(family, pw0)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw
        print(f"      {variant_name(family, n, pw)}")

    print("[3/4] Re-validating 8 single-factor candidates ...")
    single_rows = []
    for key in SINGLES:
        n, pw = WINNERS[key]
        wf = walk_forward_direction(factors[key], close, fwd_return, points, mode="reversion", n_folds=5)
        annual_rate = (wf["n_long"] + wf["n_short"]) / years
        ann_sharpe = wf["oos_sharpe_all"] * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")
        single_rows.append({
            "family": key[0], "old_pctrank_h1": key[1], "optimized_variant": variant_name(key[0], n, pw),
            "oos_sharpe_all": wf["oos_sharpe_all"], "ann_sharpe": ann_sharpe, "annual_rate": annual_rate,
            "n_folds_positive": wf["n_folds_positive"],
            "oos_sharpe_long": wf["oos_sharpe_long"], "oos_sharpe_short": wf["oos_sharpe_short"],
        })
    single_df = pd.DataFrame(single_rows)

    print("[4/4] Re-validating 9 pair candidates ...")
    pair_rows = []
    for key_a, key_b in PAIRS:
        n_a, pw_a = WINNERS[key_a]
        n_b, pw_b = WINNERS[key_b]
        wf = walk_forward_direction_pair(
            factors[key_a], "reversion", factors[key_b], "reversion", close, fwd_return, points, n_folds=5,
        )
        annual_rate = (wf["n_long"] + wf["n_short"]) / years
        ann_sharpe = wf["oos_sharpe_all"] * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")
        pair_rows.append({
            "variant_a": variant_name(*key_a, WINNERS[key_a][1]), "variant_b": variant_name(*key_b, WINNERS[key_b][1]),
            "oos_sharpe_all": wf["oos_sharpe_all"], "ann_sharpe": ann_sharpe, "annual_rate": annual_rate,
            "n_folds_positive": wf["n_folds_positive"],
            "oos_sharpe_long": wf["oos_sharpe_long"], "oos_sharpe_short": wf["oos_sharpe_short"],
        })
    pair_df = pd.DataFrame(pair_rows)

    single_path = os.path.join(args.report_dir, "02k_optimized_singles.csv")
    pair_path = os.path.join(args.report_dir, "02k_optimized_pairs.csv")
    single_df.to_csv(single_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    single_validated = single_df[
        (single_df["oos_sharpe_all"] > 0) & (single_df["n_folds_positive"] >= 3) & (single_df["annual_rate"] >= MIN_ANNUAL_RATE)
    ].sort_values("ann_sharpe", ascending=False)
    pair_validated = pair_df[
        (pair_df["oos_sharpe_all"] > 0) & (pair_df["n_folds_positive"] >= 3) & (pair_df["annual_rate"] >= MIN_ANNUAL_RATE)
    ].sort_values("ann_sharpe", ascending=False)

    lines = [
        "# 窗口优化后的最终候选验证报告", "",
        "## 方法", "",
        "用02j每个因子的获胜窗口替换02h/02i里对应的H1搬用窗口，重新跑一遍同样的"
        "5折walk-forward单因子/两两组合验证（持有期仍是占位值12根M5，标准不变：",
        f"OOS Sharpe>0、>=3/5折为正、年触发>={MIN_ANNUAL_RATE}次）。",
        "",
        "## 单因子（原8个候选，换用优化窗口后）", "",
        "| 因子(优化后) | OOS Sharpe(年化) | 年触发 | 折数为正 | 多/空(未年化) |",
        "|---|---|---|---|---|",
    ]
    for _, r in single_df.sort_values("ann_sharpe", ascending=False).iterrows():
        lines.append(f"| {r['optimized_variant']} | {r['ann_sharpe']:+.2f} | {r['annual_rate']:.0f} | "
                      f"{r['n_folds_positive']}/5 | {r['oos_sharpe_long']:+.3f}/{r['oos_sharpe_short']:+.3f} |")
    lines += ["", f"**通过筛选：{len(single_validated)}/8个**（原为8/8——因为8个候选是按当时"
              "窗口筛出来的，窗口一换，标准需要重新核实是否还成立）", ""]

    lines += ["## 两两组合（原9对候选，换用优化窗口后）", "",
              "| 因子A(优化后) | 因子B(优化后) | OOS Sharpe(年化) | 年触发 | 折数为正 |",
              "|---|---|---|---|---|"]
    for _, r in pair_df.sort_values("ann_sharpe", ascending=False).iterrows():
        lines.append(f"| {r['variant_a']} | {r['variant_b']} | {r['ann_sharpe']:+.2f} | "
                      f"{r['annual_rate']:.0f} | {r['n_folds_positive']}/5 |")
    lines += ["", f"**通过筛选：{len(pair_validated)}/9个**", ""]

    lines += [
        "## 结论与下一步", "",
        f"- 完整数据：{single_path}、{pair_path}。",
        "- 这是信号设计track目前的最终候选池（窗口已按M5原生尺度优化）。",
        "- 下一步：汇总每个通过因子的含义、参数、窗口来源，产出因子含义/参数汇总表。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02k_optimized_validation_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

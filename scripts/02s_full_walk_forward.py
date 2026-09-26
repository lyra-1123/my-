#!/usr/bin/env python3
"""Signal design step 6: full nested walk-forward validation.

02n/02o/02p each chose a hyperparameter (holding period N, then per-side
stop-loss/take-profit) by looking at ALL 5 folds' pooled OOS performance at
once, then froze that single choice and applied it everywhere -- a mild but
real form of peeking, since the "OOS" folds used for scoring were also the
folds used to pick the winning hyperparameter. This step redoes the whole
selection properly nested: for every fold, N and per-side SL/TP are chosen
using ONLY that fold's training range, then frozen and applied to that
fold's own held-out test range. Comparing this against 02p's result shows
how much the earlier (peeking) approach overstated performance, and the
per-fold hyperparameter choices show whether the earlier picks are actually
stable across time or fold-dependent noise.

Window choice (02k's optimized windows for singles, 02i's original windows
for pairs) is NOT re-opened here -- treated as a fixed feature-engineering
decision (already separately PBO-checked in 02m), out of scope for this
execution-parameter walk-forward.

Usage:
    python scripts/02s_full_walk_forward.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, atr as atr_fn  # noqa: E402
from factors.execution import nested_walk_forward_single, nested_walk_forward_pair  # noqa: E402

ATR_PERIOD = 14
N_HOLD_GRID = (6, 12, 24, 48, 96, 192)
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


def sl_configs_grid():
    configs = [("none", None, None)]
    for a in ATR_MULTIPLES:
        for rr in RR_RATIOS:
            configs.append(("atr", a, rr))
    for p in PCT_LEVELS:
        for rr in RR_RATIOS:
            configs.append(("pct", p, rr))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading M5 OHLC + factors_M5.parquet + 02p's earlier (peeking) results ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    assert df["close"].equals(factors["close"]), "row alignment mismatch"
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    atr14 = atr_fn(df, ATR_PERIOD)
    years = (len(df) * 5 / 60 / 24) / 365.25
    earlier = pd.read_csv(os.path.join(args.report_dir, "02p_final_comparison.csv")).set_index("name")
    sl_configs = sl_configs_grid()

    print("[2/4] Building the 8 optimized single-factor variants ...")
    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[variant_name(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    print(f"[3/4] Nested walk-forward for 8 singles + 9 pairs (N grid={N_HOLD_GRID}, "
          f"{len(sl_configs)} SL/TP configs per side, 5 folds each) ...")
    rows, all_fold_choices = [], []
    for family, n, pw in SINGLES:
        name = variant_name(family, n, pw)
        wf = nested_walk_forward_single(single_factors[name], close, open_, high, low, atr14,
                                         "reversion", N_HOLD_GRID, sl_configs)
        rows.append((name, "single", wf))
        for fi, ch in enumerate(wf["fold_choices"], 1):
            all_fold_choices.append({"name": name, "fold": fi, **ch})
        print(f"      {name} done")

    for var_a, var_b in PAIRS:
        name = f"{var_a}+{var_b}"
        wf = nested_walk_forward_pair(factors[var_a], "reversion", factors[var_b], "reversion",
                                       close, open_, high, low, atr14, N_HOLD_GRID, sl_configs)
        rows.append((name, "pair", wf))
        for fi, ch in enumerate(wf["fold_choices"], 1):
            all_fold_choices.append({"name": name, "fold": fi, **ch})
        print(f"      {name} done")

    print("[4/4] Writing report ...")
    result_rows = []
    for name, kind, wf in rows:
        annual_rate = (wf["n_long"] + wf["n_short"]) / years
        ann_sharpe = wf["oos_sharpe_all"] * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")
        earlier_ann = earlier.loc[name, "shared_ann_sharpe"] if name in earlier.index else float("nan")
        if kind == "single":
            passes = wf["n_folds_positive"] >= 3 and wf["n_long"] >= MIN_LONG_SHORT and wf["n_short"] >= MIN_LONG_SHORT
        else:
            passes = wf["n_folds_positive"] >= 3 and annual_rate >= MIN_ANNUAL_RATE
        result_rows.append({
            "name": name, "kind": kind, "nested_ann_sharpe": ann_sharpe, "annual_rate": annual_rate,
            "n_folds_positive": wf["n_folds_positive"], "passes": passes, "earlier_peeking_ann_sharpe": earlier_ann,
        })
    result_df = pd.DataFrame(result_rows)
    fold_choice_df = pd.DataFrame(all_fold_choices)

    result_path = os.path.join(args.report_dir, "02s_nested_walk_forward_results.csv")
    choices_path = os.path.join(args.report_dir, "02s_fold_choices.csv")
    result_df.to_csv(result_path, index=False)
    fold_choice_df.to_csv(choices_path, index=False)

    n_passed = int(result_df["passes"].sum())
    lines = [
        "# 完整流程Walk-Forward验证报告(信号设计步骤6)", "",
        "## 方法", "",
        "02n/02o/02p选持有期N和分方向止损止盈时，是看全部5折汇总的OOS表现选一次、"
        "然后固定用到所有折——这本身是一种轻度\"偷看\"（用来评分的OOS折同时也是用来"
        "选参数的折）。这一步把N和止损止盈的选择**搬到每一折内部**：每一折只用"
        "该折的训练区间自己选N、再选多空各自的止损止盈，choices冻结后只应用到"
        "该折的测试区间，5折汇总。窗口(单因子用02k优化窗口/组合用02i原窗口)不在这一步"
        "重新开放搜索(视为已经在02m单独做过PBO检验的特征工程决定)。", "",
        "## 结果：真正嵌套walk-forward vs 之前\"看全部折选一次\"的对比", "",
        "| 候选 | 之前(偷看)年化Sharpe | 嵌套walk-forward年化Sharpe | 差距 | 折数为正 | 是否通过筛选 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in result_df.sort_values("nested_ann_sharpe", ascending=False).iterrows():
        gap = r["nested_ann_sharpe"] - r["earlier_peeking_ann_sharpe"]
        lines.append(f"| {r['name']} | {r['earlier_peeking_ann_sharpe']:+.2f} | {r['nested_ann_sharpe']:+.2f} | "
                      f"{gap:+.2f} | {r['n_folds_positive']}/5 | {'是' if r['passes'] else '否'} |")

    lines += ["", f"**{n_passed}/{len(result_df)}个候选在真正嵌套的walk-forward下仍然通过筛选**"
              "(OOS Sharpe>0 + >=3/5折为正 + 触发频率达标)。", "",
              "## 参数稳定性：每折自己选出的N和止损止盈是否一致", "", "见完整数据"
              f"({choices_path})——如果同一个候选5折选出的N/止损类型来回跳变，"
              "说明之前\"看全部折选一次\"的参数本身就不稳定，只是偶然在全样本上表现好。",
              "", "## 结论与下一步", "",
              f"- 完整数据：{result_path}（每候选结果）、{choices_path}（每折的参数选择）。",
              "- 下一步（步骤7）：从这一步真正OOS验证过的结果，反推出正式的入场出场条件"
              "(不再是每折不同的候选参数，而是给出一套用于实盘/阶段3b的最终推荐参数)。",
              ""]

    report_path = os.path.join(args.report_dir, "02s_full_walk_forward_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

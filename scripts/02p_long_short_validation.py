#!/usr/bin/env python3
"""Signal design step 4: long/short direction-specific stop-loss/take-profit
validation. 02o picked ONE shared (stop-loss, take-profit) config per
candidate, scored on both directions pooled together -- but step 2 already
found sharp long/short asymmetry for at least one earlier signal (M5
autocorr+MFI: long ann Sharpe +0.10 vs short +0.02), so a shared config
could easily be masking a config that's great for one side and bad for the
other.

Two parts, per candidate, holding N fixed at 02o's chosen value:
  1. Diagnostic: score 02o's ALREADY-CHOSEN shared config separately on
     long-only and short-only trades.
  2. Search: independently re-run the same 25-config SL/TP grid restricted
     to long-only, and again restricted to short-only; pick each side's own
     best. Then merge that side-specific pair of configs' raw trade returns
     into one combined pooled Sharpe, to compare directly against the
     shared-config Sharpe from 02o.

Usage:
    python scripts/02p_long_short_validation.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, atr as atr_fn  # noqa: E402
from factors.execution import walk_forward_execution_single, walk_forward_execution_pair, _sharpe  # noqa: E402

ATR_PERIOD = 14
ATR_MULTIPLES = (0.5, 1.0, 1.5, 2.0)
PCT_LEVELS = (0.002, 0.003, 0.005, 0.01)
RR_RATIOS = (1.0, 1.5, 2.0)
MIN_FOLDS_POSITIVE = 3

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


def configs_grid():
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

    print("[1/4] Loading M5 OHLC + factors_M5.parquet + 02o's chosen configs ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    assert df["close"].equals(factors["close"]), "row alignment mismatch"
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    atr14 = atr_fn(df, ATR_PERIOD)
    years = (len(df) * 5 / 60 / 24) / 365.25
    shared = pd.read_csv(os.path.join(args.report_dir, "02o_stop_take_profit_winners.csv")).set_index("name")

    print("[2/4] Building the 8 optimized single-factor variants ...")
    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[variant_name(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    configs = configs_grid()

    def wf_fn_for(name):
        if name in single_factors:
            return (walk_forward_execution_single,
                    (single_factors[name], close, open_, high, low, atr14, "reversion"))
        var_a, var_b = name.split("+")
        return (walk_forward_execution_pair,
                (factors[var_a], "reversion", factors[var_b], "reversion", close, open_, high, low, atr14))

    print("[3/4] Diagnostic (shared config split by side) + per-side SL/TP search ...")
    diag_rows, search_rows, final_rows = [], [], []
    all_names = [variant_name(*s) for s in SINGLES] + [f"{a}+{b}" for a, b in PAIRS]
    for name in all_names:
        s = shared.loc[name]
        n_hold = int(s["n_hold"])
        sl_type = None if s["best_sl_type"] == "none" else s["best_sl_type"]
        sl_level = None if pd.isna(s["best_sl_level"]) else float(s["best_sl_level"])
        rr = None if pd.isna(s["best_rr"]) else float(s["best_rr"])
        fn, base_args = wf_fn_for(name)

        # 1. diagnostic: shared config, scored separately per side
        for side in ("long", "short"):
            wf = fn(*base_args, n_hold, sl_type=sl_type, sl_level=sl_level, rr_ratio=rr, direction_filter=side)
            diag_rows.append({"name": name, "side": side, "sharpe": wf["oos_sharpe_all"],
                               "n_trades": wf["n_long"] + wf["n_short"], "n_folds_positive": wf["n_folds_positive"]})

        # 2. per-side independent SL/TP search (N fixed at 02o's value)
        side_best = {}
        for side in ("long", "short"):
            best_wf, best_cfg = None, None
            for c_sl_type, c_sl_level, c_rr in configs:
                kwargs = {} if c_sl_type == "none" else {"sl_type": c_sl_type, "sl_level": c_sl_level, "rr_ratio": c_rr}
                wf = fn(*base_args, n_hold, direction_filter=side, return_raw=True, **kwargs)
                search_rows.append({"name": name, "side": side, "sl_type": c_sl_type, "sl_level": c_sl_level,
                                     "rr_ratio": c_rr, "sharpe": wf["oos_sharpe_all"],
                                     "n_trades": wf["n_long"] + wf["n_short"], "n_folds_positive": wf["n_folds_positive"]})
                qualifies = wf["n_folds_positive"] >= MIN_FOLDS_POSITIVE and (wf["n_long"] + wf["n_short"]) >= 10
                is_better = best_wf is None or (
                    (qualifies and not best_wf[1]) or
                    (qualifies == best_wf[1] and (np.nan_to_num(wf["oos_sharpe_all"], nan=-99) >
                                                   np.nan_to_num(best_wf[0]["oos_sharpe_all"], nan=-99)))
                )
                if is_better:
                    best_wf, best_cfg = (wf, qualifies), (c_sl_type, c_sl_level, c_rr)
            side_best[side] = (best_wf[0], best_cfg)

        long_wf, long_cfg = side_best["long"]
        short_wf, short_cfg = side_best["short"]
        combined_ret = np.concatenate([long_wf["ret_all"], short_wf["ret_all"]])
        split_sharpe = _sharpe(combined_ret)
        split_n_trades = long_wf["n_long"] + long_wf["n_short"] + short_wf["n_long"] + short_wf["n_short"]
        split_annual_rate = split_n_trades / years
        split_ann_sharpe = split_sharpe * (split_annual_rate ** 0.5) if split_annual_rate > 0 else float("nan")
        final_rows.append({
            "name": name, "n_hold": n_hold,
            "shared_sl_type": s["best_sl_type"], "shared_sl_level": s["best_sl_level"], "shared_rr": s["best_rr"],
            "shared_ann_sharpe": s["best_ann_sharpe"],
            "long_sl_type": long_cfg[0], "long_sl_level": long_cfg[1], "long_rr": long_cfg[2],
            "long_sharpe": long_wf["oos_sharpe_all"], "long_n_trades": long_wf["n_long"] + long_wf["n_short"],
            "short_sl_type": short_cfg[0], "short_sl_level": short_cfg[1], "short_rr": short_cfg[2],
            "short_sharpe": short_wf["oos_sharpe_all"], "short_n_trades": short_wf["n_long"] + short_wf["n_short"],
            "split_combined_sharpe": split_sharpe, "split_ann_sharpe": split_ann_sharpe,
        })
        print(f"      {name} done")

    diag_df, search_df, final_df = pd.DataFrame(diag_rows), pd.DataFrame(search_rows), pd.DataFrame(final_rows)
    diag_path = os.path.join(args.report_dir, "02p_diagnostic_shared_config_by_side.csv")
    search_path = os.path.join(args.report_dir, "02p_per_side_search.csv")
    final_path = os.path.join(args.report_dir, "02p_final_comparison.csv")
    diag_df.to_csv(diag_path, index=False)
    search_df.to_csv(search_path, index=False)
    final_df.to_csv(final_path, index=False)

    print("[4/4] Writing report ...")
    lines = [
        "# 多空分方向止盈止损验证报告(信号设计步骤4)", "",
        "## 方法", "",
        "1. **诊断**：02o选出的共享止损止盈配置，分别只用多头触发/只用空头触发重新算一遍Sharpe，"
        "看这个共享配置是不是被某一个方向的表现拉高/拉低了整体数字。",
        "2. **分方向搜索**：N固定用02o选出的值，止损止盈的25种配置分别只在多头/只在空头triggers上"
        "独立重新搜索一遍，各自选各自的最优配置；再把两个方向各自最优配置的真实交易收益合并，"
        "算一个\"分方向优化后\"的整体Sharpe，跟02o的\"共享配置\"整体Sharpe直接比较。",
        "",
        "## 诊断：共享配置分方向表现", "",
        "| 候选 | 多头Sharpe(未年化) | 多头折数为正 | 空头Sharpe(未年化) | 空头折数为正 |",
        "|---|---|---|---|---|",
    ]
    for name in all_names:
        long_row = diag_df[(diag_df["name"] == name) & (diag_df["side"] == "long")].iloc[0]
        short_row = diag_df[(diag_df["name"] == name) & (diag_df["side"] == "short")].iloc[0]
        lines.append(f"| {name} | {long_row['sharpe']:+.3f} | {long_row['n_folds_positive']}/5 | "
                      f"{short_row['sharpe']:+.3f} | {short_row['n_folds_positive']}/5 |")

    lines += ["", "## 分方向独立优化 vs 共享配置：整体年化Sharpe对比(口径一致，都已年化)", "",
              "| 候选 | 共享配置年化Sharpe | 分方向优化后年化Sharpe | 分方向是否更好 |",
              "|---|---|---|---|"]
    n_better = 0
    for _, r in final_df.iterrows():
        better = (not pd.isna(r["split_ann_sharpe"])) and (
            pd.isna(r["shared_ann_sharpe"]) or r["split_ann_sharpe"] > r["shared_ann_sharpe"]
        )
        n_better += int(better)
        lines.append(f"| {r['name']} | {r['shared_ann_sharpe']:+.2f} | {r['split_ann_sharpe']:+.2f} | "
                      f"{'是' if better else '否'} |")
    lines.append("")
    lines.append(f"**{n_better}/{len(final_df)}个候选分方向独立优化止损止盈后比共享一套配置更好**。")
    lines += ["",
              "## 每个候选：多空各自选出的止损止盈配置", "",
              "| 候选 | 多头止损 | 多头水平 | 多头盈亏比 | 多头Sharpe | 多头笔数 | "
              "空头止损 | 空头水平 | 空头盈亏比 | 空头Sharpe | 空头笔数 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in final_df.iterrows():
        lines.append(f"| {r['name']} | {r['long_sl_type']} | {r['long_sl_level']} | {r['long_rr']} | "
                      f"{r['long_sharpe']:+.3f} | {r['long_n_trades']:.0f} | "
                      f"{r['short_sl_type']} | {r['short_sl_level']} | {r['short_rr']} | "
                      f"{r['short_sharpe']:+.3f} | {r['short_n_trades']:.0f} |")

    lines += ["", "## 结论与下一步", "",
              f"- 完整数据：{diag_path}（诊断）、{search_path}（分方向搜索全量）、{final_path}（对比汇总）。",
              "- 下一步（步骤5）：多空对冲互锁验证——检查同一时刻多头信号和空头信号是否会互相冲突"
              "（比如两个组合候选同时给出反向信号），以及是否需要互锁规则。",
              ""]

    report_path = os.path.join(args.report_dir, "02p_long_short_validation_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

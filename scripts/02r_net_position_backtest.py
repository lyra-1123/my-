#!/usr/bin/env python3
"""Signal design step 5b: implement + validate the interlock policy the
user picked for step 5's measured overlap -- "net position, opposing
signal closes first" (a new opposite-direction trigger force-closes
whatever's open right now, at that trigger's own entry price, then opens
in the new direction; a same-direction trigger while already positioned
that way is ignored, not stacked). Applied consistently at both levels:
a single candidate's own long/short streams reconciled against each other,
and multiple candidates run together reconciled into one net position.

Uses execution.py's new `simulate_net_position` on each candidate's
full-sample-fitted trade_windows (same "as deployed" rule as 02q, entry_price/
exit_price now included). Compares:
  - each candidate's own reversal-corrected Sharpe vs its original pooled
    (uncorrected, overlaps ignored) Sharpe from 02o/02p
  - the 8-singles / 9-pairs / all-17 ensembles' single unified net-position
    equity Sharpe vs a naive "just pool everyone's independent trades"
    baseline for the same group

Usage:
    python scripts/02r_net_position_backtest.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, atr as atr_fn  # noqa: E402
from factors.direction import reversion_direction, combine_directions  # noqa: E402
from factors.execution import trade_windows, simulate_net_position, _sharpe  # noqa: E402

ATR_PERIOD = 14
QUANTILE = 0.8

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


def full_sample_direction(factor: pd.Series, quantile: float = QUANTILE) -> pd.Series:
    return reversion_direction(factor, factor.quantile(1 - quantile), factor.quantile(quantile))


def cfg(row, side):
    sl_type = row[f"{side}_sl_type"]
    sl_type = None if sl_type == "none" else sl_type
    sl_level = None if pd.isna(row[f"{side}_sl_level"]) else float(row[f"{side}_sl_level"])
    rr = None if pd.isna(row[f"{side}_rr"]) else float(row[f"{side}_rr"])
    return sl_type, sl_level, rr


def candidate_trades(name, direction_full, open_, high, low, close, atr14, n_hold, final_row):
    windows = []
    for side, sign in (("long", 1), ("short", -1)):
        side_dir = direction_full.where(np.sign(direction_full) == sign, 0.0)
        sl_type, sl_level, rr = cfg(final_row, side)
        tw = trade_windows(side_dir, open_, high, low, close, atr14, n_hold, sl_type, sl_level, rr)
        windows.append(tw)
    tw = pd.concat(windows, ignore_index=True)
    tw["source"] = name
    return tw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading M5 OHLC + factors_M5.parquet + step3/4 chosen params ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    assert df["close"].equals(factors["close"]), "row alignment mismatch"
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    atr14 = atr_fn(df, ATR_PERIOD)
    years = (len(df) * 5 / 60 / 24) / 365.25
    final = pd.read_csv(os.path.join(args.report_dir, "02p_final_comparison.csv")).set_index("name")

    print("[2/4] Building trade windows for all 17 candidates ...")
    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[variant_name(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    trades_by_name = {}
    for family, n, pw in SINGLES:
        name = variant_name(family, n, pw)
        direction_full = full_sample_direction(single_factors[name])
        n_hold = int(final.loc[name, "n_hold"])
        trades_by_name[name] = candidate_trades(name, direction_full, open_, high, low, close, atr14, n_hold, final.loc[name])

    for var_a, var_b in PAIRS:
        name = f"{var_a}+{var_b}"
        dir_a, dir_b = full_sample_direction(factors[var_a]), full_sample_direction(factors[var_b])
        combined = combine_directions(dir_a, dir_b)
        n_hold = int(final.loc[name, "n_hold"])
        trades_by_name[name] = candidate_trades(name, combined, open_, high, low, close, atr14, n_hold, final.loc[name])

    def ann_sharpe_of(ret: np.ndarray) -> tuple:
        annual_rate = len(ret) / years
        sh = _sharpe(ret)
        return sh, (sh * (annual_rate ** 0.5) if annual_rate > 0 else float("nan")), annual_rate

    print("[3/4] Per-candidate self-reversal-corrected Sharpe ...")
    self_rows = []
    for name, tw in trades_by_name.items():
        baseline_ret = tw["direction"] * (tw["exit_price"] - tw["entry_price"]) / tw["entry_price"]
        base_sh, base_ann, base_rate = ann_sharpe_of(baseline_ret.to_numpy())
        net = simulate_net_position(tw)
        net_sh, net_ann, net_rate = ann_sharpe_of(net["ret"].to_numpy())
        n_forced = int(net["forced_close"].sum())
        self_rows.append({"name": name, "baseline_ann_sharpe": base_ann, "baseline_n_trades": len(tw),
                           "net_ann_sharpe": net_ann, "net_n_trades": len(net), "n_forced_close": n_forced})
        print(f"      {name}: baseline={base_ann:+.2f} -> net-position={net_ann:+.2f} "
              f"({n_forced} forced closes / {len(net)} trades)")
    self_df = pd.DataFrame(self_rows)

    print("[4/4] Ensemble net-position backtest (8 singles / 9 pairs / all 17) ...")
    single_names = [variant_name(*s) for s in SINGLES]
    pair_names = [f"{a}+{b}" for a, b in PAIRS]
    ensemble_rows = []
    for label, names in [("8个单因子", single_names), ("9对组合", pair_names), ("全部17个", single_names + pair_names)]:
        pooled = pd.concat([trades_by_name[n] for n in names], ignore_index=True)
        naive_ret = pooled["direction"] * (pooled["exit_price"] - pooled["entry_price"]) / pooled["entry_price"]
        naive_sh, naive_ann, naive_rate = ann_sharpe_of(naive_ret.to_numpy())
        net = simulate_net_position(pooled)
        net_sh, net_ann, net_rate = ann_sharpe_of(net["ret"].to_numpy())
        n_forced = int(net["forced_close"].sum())
        ensemble_rows.append({"group": label, "naive_ann_sharpe": naive_ann, "naive_n_trades": len(pooled),
                               "net_ann_sharpe": net_ann, "net_n_trades": len(net), "n_forced_close": n_forced})
        print(f"      {label}: naive(pooled independent)={naive_ann:+.2f} ({len(pooled)} trades) "
              f"-> net-position={net_ann:+.2f} ({len(net)} trades, {n_forced} forced closes)")
    ensemble_df = pd.DataFrame(ensemble_rows)

    self_path = os.path.join(args.report_dir, "02r_self_net_position.csv")
    ens_path = os.path.join(args.report_dir, "02r_ensemble_net_position.csv")
    self_df.to_csv(self_path, index=False)
    ensemble_df.to_csv(ens_path, index=False)

    lines = [
        "# 净头寸互锁策略实现与验证报告(信号设计步骤5b)", "",
        "## 方法", "",
        "实现用户选定的互锁策略：**净头寸相抵，反向信号先平仓**——同方向的新触发在已有仓位"
        "存活期内被忽略(不加仓)；反方向的新触发立即把当前仓位强制平仓(按新触发那根bar的"
        "开盘价)，然后开新方向的仓位。单个候选自己的多空重叠、和多个候选一起跑的组合层面"
        "冲突，都用同一套`simulate_net_position`逻辑处理(用户要求两者policy一致)。", "",
        "对比口径：\"基线\"=假装每笔交易都能各自独立跑完全程、互不干扰(02o/02p用的口径)；"
        "\"净头寸\"=按上面的互锁规则实际会发生的结果(有的交易被反向信号提前砍仓)。都用"
        "全样本固定阈值(不是walk-forward逐折)，年化用交易笔数/12.2年折算。", "",
        "## 单个候选：自身反手修正前后对比", "",
        "| 候选 | 基线年化Sharpe | 净头寸年化Sharpe | 强制平仓笔数/总笔数 |",
        "|---|---|---|---|",
    ]
    for _, r in self_df.sort_values("net_ann_sharpe", ascending=False).iterrows():
        lines.append(f"| {r['name']} | {r['baseline_ann_sharpe']:+.2f} | {r['net_ann_sharpe']:+.2f} | "
                      f"{r['n_forced_close']:.0f}/{r['net_n_trades']:.0f} |")

    lines += ["", "## 组合层面：多个候选一起跑，净头寸互锁前后对比", "",
              "| 组合范围 | 朴素合并年化Sharpe(忽略冲突) | 净头寸年化Sharpe(实际会发生的) | "
              "强制平仓笔数/总笔数 |",
              "|---|---|---|---|"]
    for _, r in ensemble_df.iterrows():
        lines.append(f"| {r['group']} | {r['naive_ann_sharpe']:+.2f} | {r['net_ann_sharpe']:+.2f} | "
                      f"{r['n_forced_close']:.0f}/{r['net_n_trades']:.0f} |")

    lines += ["", "## 结论与下一步", "",
              f"- 完整数据：{self_path}（单候选）、{ens_path}（组合层面）。",
              "- 这套互锁逻辑(`src/factors/execution.py::simulate_net_position`)是可复用的基础"
              "设施，阶段3b把多个候选接入真实马丁引擎时可以直接复用同一个函数。",
              "- 下一步（步骤6）：Walk-Forward验证——把目前这套(窗口+N+分方向止损止盈+互锁"
              "策略)完整流程放进真正的多折walk-forward里再走一遍，而不是像这一步一样用"
              "全样本固定阈值。",
              ""]

    report_path = os.path.join(args.report_dir, "02r_net_position_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Signal design step 5: long/short hedge interlock validation.

Each of the 17 candidates (8 singles + 9 pairs) independently fires long
and short trades. If several are ever run together feeding one martingale
engine, two kinds of overlap matter:
  1. Self-overlap: a candidate's own new trigger (of either sign) arriving
     while its OWN previous trade is still open (within its holding window).
  2. Ensemble conflict: at the same bar, one candidate has a LONG trade open
     while another (or the same) has a SHORT trade open -- a single-ladder
     martingale engine can't hold both directions in one grid, so this is
     exactly the "hedge/interlock" question the user's roadmap step 5 names.

This script MEASURES both (using each candidate's final chosen parameters:
02k's optimized window for singles / 02i's window for pairs, 02n's N, 02p's
per-side stop-loss/take-profit) on the FULL-SAMPLE-fitted rule (the
"as-if-deployed-today" frozen thresholds, not a walk-forward refit -- this
step is about overlap mechanics, not another OOS validation), then reports
the frequencies. It does NOT pick an interlock policy -- that's a design
decision handed back to the user with these numbers.

Usage:
    python scripts/02q_hedge_interlock_validation.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, atr as atr_fn  # noqa: E402
from factors.direction import reversion_direction, combine_directions  # noqa: E402
from factors.execution import trade_windows  # noqa: E402

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


def build_trade_windows_for_candidate(name, direction_full, open_, high, low, close, atr14, n_hold, final_row):
    windows = []
    for side, sign in (("long", 1), ("short", -1)):
        side_dir = direction_full.where(np.sign(direction_full) == sign, 0.0)
        sl_type, sl_level, rr = cfg(final_row, side)
        tw = trade_windows(side_dir, open_, high, low, close, atr14, n_hold, sl_type, sl_level, rr)
        windows.append(tw)
    return pd.concat(windows, ignore_index=True).sort_values("entry_idx").reset_index(drop=True)


def self_overlap_stats(tw: pd.DataFrame) -> dict:
    if len(tw) < 2:
        return {"n_trades": len(tw), "n_overlaps": 0, "n_opposite_overlaps": 0, "overlap_rate": 0.0}
    tw = tw.sort_values("entry_idx").reset_index(drop=True)
    prev_exit = tw["exit_idx"].shift(1)
    prev_dir = tw["direction"].shift(1)
    overlap = tw["entry_idx"] <= prev_exit
    opposite_overlap = overlap & (np.sign(tw["direction"]) != np.sign(prev_dir))
    return {
        "n_trades": len(tw), "n_overlaps": int(overlap.sum()),
        "n_opposite_overlaps": int(opposite_overlap.sum()),
        "overlap_rate": float(overlap.mean()),
    }


def ensemble_conflict_bars(all_windows: list, n_rows: int) -> dict:
    """Build a bar-level long-open / short-open count timeline across every
    candidate's trades and measure how often both are simultaneously > 0."""
    long_open = np.zeros(n_rows, dtype=np.int32)
    short_open = np.zeros(n_rows, dtype=np.int32)
    for tw in all_windows:
        for _, r in tw.iterrows():
            lo, hi = int(r["entry_idx"]), int(r["exit_idx"])
            if r["direction"] > 0:
                long_open[lo:hi + 1] += 1
            else:
                short_open[lo:hi + 1] += 1
    both = (long_open > 0) & (short_open > 0)
    return {
        "bars_long_open": int((long_open > 0).sum()), "bars_short_open": int((short_open > 0).sum()),
        "bars_both_open": int(both.sum()), "conflict_rate_of_all_bars": float(both.mean()),
        "conflict_rate_of_any_open": float(both.sum() / max(1, ((long_open > 0) | (short_open > 0)).sum())),
    }


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
    final = pd.read_csv(os.path.join(args.report_dir, "02p_final_comparison.csv")).set_index("name")

    print("[2/4] Building the 8 optimized single-factor variants + trade windows for all 17 candidates ...")
    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[variant_name(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    trade_windows_by_name = {}
    for family, n, pw in SINGLES:
        name = variant_name(family, n, pw)
        direction_full = full_sample_direction(single_factors[name])
        n_hold = int(final.loc[name, "n_hold"])
        trade_windows_by_name[name] = build_trade_windows_for_candidate(
            name, direction_full, open_, high, low, close, atr14, n_hold, final.loc[name])
        print(f"      {name}: {len(trade_windows_by_name[name])} trades")

    for var_a, var_b in PAIRS:
        name = f"{var_a}+{var_b}"
        dir_a = full_sample_direction(factors[var_a])
        dir_b = full_sample_direction(factors[var_b])
        combined = combine_directions(dir_a, dir_b)
        n_hold = int(final.loc[name, "n_hold"])
        trade_windows_by_name[name] = build_trade_windows_for_candidate(
            name, combined, open_, high, low, close, atr14, n_hold, final.loc[name])
        print(f"      {name}: {len(trade_windows_by_name[name])} trades")

    print("[3/4] Self-overlap (per candidate) + ensemble conflict (8 singles / 9 pairs / all 17) ...")
    self_rows = []
    for name, tw in trade_windows_by_name.items():
        stats = self_overlap_stats(tw)
        stats["name"] = name
        self_rows.append(stats)
    self_df = pd.DataFrame(self_rows)[["name", "n_trades", "n_overlaps", "n_opposite_overlaps", "overlap_rate"]]

    n_rows = len(df)
    single_names = [variant_name(*s) for s in SINGLES]
    pair_names = [f"{a}+{b}" for a, b in PAIRS]
    ens_singles = ensemble_conflict_bars([trade_windows_by_name[n] for n in single_names], n_rows)
    ens_pairs = ensemble_conflict_bars([trade_windows_by_name[n] for n in pair_names], n_rows)
    ens_all = ensemble_conflict_bars(list(trade_windows_by_name.values()), n_rows)

    print("[4/4] Writing report ...")
    lines = [
        "# 多空对冲互锁验证报告(信号设计步骤5)", "",
        "## 方法", "",
        "用每个候选最终确定的参数(02k/02i的窗口、02n的N、02p的分方向止损止盈)、按**全样本"
        "固定阈值**(不是walk-forward逐折重新拟合——这一步测的是\"部署后信号会不会自相冲突\"，"
        "不是再做一次样本外验证)重建每个候选完整的交易时间窗口(entry_idx~exit_idx)，检查两类重叠：",
        "",
        "1. **自重叠**：同一个候选自己的新触发，在上一笔交易还没平仓时又来一笔——包括"
        "同方向重叠(信号还在场内又追加)和反方向重叠(还持多，又来一个空头触发，即\"对冲/反手\"场景)。",
        "2. **组合层面冲突**：任意时刻，是否同时存在\"某候选的多头仓位开着\"和\"某候选(或同一个)的"
        "空头仓位开着\"——这是如果多个候选一起接入同一个马丁引擎会遇到的真实问题(单向网格没法"
        "同时做多做空)。分别看8个单因子一起跑、9个组合一起跑、17个全部一起跑三种情形。",
        "",
        "**本脚本只测量频率，不替用户做互锁策略的决定**——策略选择(禁止同时持仓/允许对冲/"
        "净头寸相抵/参考步骤1的\"减仓等待\")留在下面结论里问。", "",
        "## 自重叠：每个候选自己的信号是否会追加/反手", "",
        "| 候选 | 交易笔数 | 重叠笔数 | 其中反方向重叠(对冲/反手) | 重叠率 |",
        "|---|---|---|---|---|",
    ]
    for _, r in self_df.sort_values("overlap_rate", ascending=False).iterrows():
        lines.append(f"| {r['name']} | {r['n_trades']:.0f} | {r['n_overlaps']:.0f} | "
                      f"{r['n_opposite_overlaps']:.0f} | {r['overlap_rate']:.1%} |")

    lines += ["", "## 组合层面：多头仓位和空头仓位同时开着的频率", "",
              "| 组合范围 | 多头曾开仓的bar数 | 空头曾开仓的bar数 | 多空同时开仓的bar数 | "
              "占全部bar比例 | 占\"有任意仓位开着\"bar的比例 |",
              "|---|---|---|---|---|---|"]
    for label, ens in [("8个单因子一起跑", ens_singles), ("9对组合一起跑", ens_pairs), ("全部17个一起跑", ens_all)]:
        lines.append(f"| {label} | {ens['bars_long_open']} | {ens['bars_short_open']} | "
                      f"{ens['bars_both_open']} | {ens['conflict_rate_of_all_bars']:.1%} | "
                      f"{ens['conflict_rate_of_any_open']:.1%} |")

    lines += ["", "## 结论与下一步", "",
              "- 完整自重叠明细见上表；组合层面冲突是不是需要一个显式的互锁规则，取决于"
              "多空同时开仓的频率有多高——频率如果很低，随便定一个简单规则(比如\"先到先得，"
              "冲突时新信号让路\")代价很小；频率如果很高，则需要认真设计(比如参考阶段2e"
              "跨周期验证里的\"减仓等待\"，或者干脆允许对冲，取决于经纪商账户是否支持双向持仓)。",
              "- 这一步只测量频率，具体互锁策略需要用户确认。", ""]

    report_path = os.path.join(args.report_dir, "02q_hedge_interlock_validation_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    self_df.to_csv(os.path.join(args.report_dir, "02q_self_overlap.csv"), index=False)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 3b, part 4: correct the actual bug behind 03c's failed recalibration.

03c swept grid_atr_window (the ATR AVERAGING window) assuming a wider
window would produce a larger ATR value, mimicking a coarser bar
timeframe -- but ATR is a MEAN of per-bar true ranges, so its window length
barely changes its dollar magnitude (only its smoothness/lag). What
actually differs between M5 and H1 is the BAR SIZE itself: an M5 bar's true
range is inherently smaller than an H1 bar's, regardless of how many bars
you average. Confirmed directly: M5 ATR(14) and M5 ATR(672) were both
~$1.19-1.21 in Jan 2009, while H1 ATR(14) was ~$4.33 in the same period --
a ~3.6x gap that no amount of M5 window-widening can close. That's why
every one of 03c's 22 configs failed at nearly the same date regardless of
window: the grid was always too narrow in dollar terms, no matter which
M5 window fed it.

This sweeps grid_atr_mult / tp_atr_mult over a much larger range (the
correct lever) at a fixed, M5-native grid_atr_window=14, to find one that
survives.

Usage:
    python scripts/03d_correct_grid_scale.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from factors.library import atr as atr_fn  # noqa: E402
from backtest.martingale_bidirectional import BidirectionalConfig, run_bidirectional_backtest  # noqa: E402

GRID_ATR_WINDOW = 14
MULT_GRID = (2, 4, 6, 8, 12, 16, 24, 32)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    return float(dd.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/3] Loading M5 OHLC + direction gate ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    gate = pd.read_parquet(os.path.join(args.clean_dir, "direction_gate_M5.parquet"))
    direction = gate["direction"]
    atr14 = atr_fn(df, GRID_ATR_WINDOW)

    print(f"[2/3] Sweeping grid_atr_mult x tp_atr_mult over {MULT_GRID} ...")
    rows = []
    for grid_mult in MULT_GRID:
        for tp_mult in MULT_GRID:
            cfg = BidirectionalConfig(grid_atr_window=GRID_ATR_WINDOW, grid_atr_mult=grid_mult, tp_atr_mult=tp_mult)
            result = run_bidirectional_backtest(df, atr14, direction, cfg)
            eq = result.equity_curve
            row = {
                "grid_atr_mult": grid_mult, "tp_atr_mult": tp_mult,
                "peak_equity": float(eq.max()), "final_equity": float(eq.iloc[-1]),
                "max_drawdown": max_drawdown(eq), "ruined": result.ruin_time is not None,
                "ruin_time": str(result.ruin_time) if result.ruin_time else "",
                "max_layers_long": result.max_layers_long, "max_layers_short": result.max_layers_short,
                "n_take_profit": int((result.trades["type"] == "take_profit").sum()) if len(result.trades) else 0,
                "n_stop_out": int((result.trades["type"] == "stop_out").sum()) if len(result.trades) else 0,
            }
            rows.append(row)
            print(f"      grid={grid_mult},tp={tp_mult}: peak=${row['peak_equity']:,.0f} "
                  f"final=${row['final_equity']:,.0f} ruined={row['ruined']}({row['ruin_time']}) "
                  f"mdd={row['max_drawdown']:.1%} stop_outs={row['n_stop_out']}")

    result_df = pd.DataFrame(rows)
    result_path = os.path.join(args.report_dir, "03d_scale_corrected_scan.csv")
    result_df.to_csv(result_path, index=False)

    survivors = result_df[~result_df["ruined"]]
    n_survived = len(survivors)
    pool = survivors if n_survived else result_df
    best = pool.loc[pool["final_equity"].idxmax()]

    print("[3/3] Writing report ...")
    lines = [
        "# 网格尺度修正后的重新扫描报告(阶段3b修正)", "",
        "## 背景：03c的搜索方向本身就搞错了", "",
        "03c扫的是ATR**平均窗口**的长度，隐含假设是\"窗口越长，ATR值越大，越接近H1的"
        "特征距离\"——但这个假设是错的。ATR是**每根bar真实波幅的平均值**，平均窗口"
        "拉长只会让这个平均数更平滑/滞后，不会让它的**数值量级**变大。M5上不管用"
        "ATR(14)还是ATR(672)，2009年1月这段时期算出来都是$1.19~1.21，几乎一样。"
        "真正的差异来自**bar本身的大小**：H1的一根bar本来就比M5的一根bar包含更多"
        "价格波动，H1 ATR(14)在同一时期是$4.33，是M5 ATR(14)的约3.6倍——这个差距"
        "不管M5上平均多少根bar都补不回来。所以03c那一版怎么调窗口，22组全部在"
        "同一时间附近爆仓，就是因为窗口从来不是正确的调节杆。", "",
        f"这里改成在M5原生的ATR(14)基础上，直接放大**倍数**(grid_atr_mult/tp_atr_mult)"
        f"来补偿这个约3.6倍的尺度差，扫描范围{MULT_GRID}。", "",
        "## 结果", "",
        "| 网格倍数 | 止盈倍数 | 峰值权益 | 最终权益 | 最大回撤 | 是否破产 | 破产时间 | 止损次数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in result_df.sort_values("final_equity", ascending=False).iterrows():
        lines.append(f"| {r['grid_atr_mult']} | {r['tp_atr_mult']} | ${r['peak_equity']:,.0f} | "
                      f"${r['final_equity']:,.0f} | {r['max_drawdown']:.1%} | "
                      f"{'是' if r['ruined'] else '否'} | {r['ruin_time']} | {r['n_stop_out']} |")

    lines += ["", f"**{n_survived}/{len(result_df)}个组合全程存活。**", "",
              "## 最终推荐配置", "",
              f"- grid_atr_window = {GRID_ATR_WINDOW}根M5(原生短窗口，已确认窗口长度不影响尺度)",
              f"- grid_atr_mult = {best['grid_atr_mult']}",
              f"- tp_atr_mult = {best['tp_atr_mult']}",
              f"- 峰值权益 ${best['peak_equity']:,.0f}，最终权益 ${best['final_equity']:,.0f}，"
              f"最大回撤 {best['max_drawdown']:.1%}，"
              f"{'破产于'+best['ruin_time'] if best['ruined'] else '全程存活'}",
              "", "## 结论与下一步", "",
              f"- 完整数据：{result_path}。",
              "- 如果这轮仍然全军覆没，说明问题不只是网格尺度，还要检查max_layers/"
              "multiplier本身对双向+做空的适配性(阶段3的多头基线能撑4年，很大程度上"
              "是靠金价长期上涨的顺风车；现在多空都做，空头是在逆着这个长期趋势下重注，"
              "可能需要更保守的层数/倍数，而不只是调间距)。",
              ""]

    report_path = os.path.join(args.report_dir, "03d_scale_corrected_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

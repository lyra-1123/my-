#!/usr/bin/env python3
"""Phase 3b, part 3: recalibrate the bidirectional grid's own ATR-based
parameters for M5's native timescale.

03b directly reused phase 3's H1-calibrated grid_atr_window=14/grid_atr_mult
=1.0/tp_atr_mult=1.0 on M5 bars -- the same "same bar count, very different
real time span" mistake the whole signal-design track spent phases 02j-02u
correcting for factor windows, now at the grid-spacing level: M5 ATR(14)
covers ~70 minutes vs H1 ATR(14)'s ~14 hours, so the grid was far too
tight and got run over in under a month.

Two-stage search, keeping the grid on M5 bars (per user's choice):
  Stage 1: fix grid_atr_mult=tp_atr_mult=1.0, sweep grid_atr_window over
    multiples of 14 (x1/x3/x6/x12/x24/x48 M5 bars) to find where the grid
    stops blowing up almost immediately.
  Stage 2: fix grid_atr_window at stage 1's best survivable value, sweep
    grid_atr_mult x tp_atr_mult over {0.5, 1.0, 1.5, 2.0} each (16 combos)
    to refine.

max_layers=8, multiplier=2.0, initial_lot=0.01 stay fixed at phase 3's
values for this pass (a separate question from the ATR-scale issue).

Usage:
    python scripts/03c_grid_parameter_search.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from factors.library import atr as atr_fn  # noqa: E402
from backtest.martingale_bidirectional import BidirectionalConfig, run_bidirectional_backtest  # noqa: E402

WINDOW_MULTIPLES = (1, 3, 6, 12, 24, 48)  # x14 M5 bars: 14,42,84,168,336,672
ATR_MULT_GRID = (0.5, 1.0, 1.5, 2.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    return float(dd.min())


def summarize(name, df, result, extra):
    eq = result.equity_curve
    row = {
        "config": name, "peak_equity": float(eq.max()), "final_equity": float(eq.iloc[-1]),
        "max_drawdown": max_drawdown(eq), "ruined": result.ruin_time is not None,
        "ruin_time": str(result.ruin_time) if result.ruin_time else "",
        "max_layers_long": result.max_layers_long, "max_layers_short": result.max_layers_short,
        "n_take_profit": int((result.trades["type"] == "take_profit").sum()) if len(result.trades) else 0,
        "n_stop_out": int((result.trades["type"] == "stop_out").sum()) if len(result.trades) else 0,
        "n_forced_reversal": len(result.forced_reversals),
    }
    row.update(extra)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading M5 OHLC + direction gate ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    gate = pd.read_parquet(os.path.join(args.clean_dir, "direction_gate_M5.parquet"))
    direction = gate["direction"]

    print(f"[2/4] Stage 1: sweeping grid_atr_window over x{WINDOW_MULTIPLES} of 14 M5 bars "
          f"(mult=tp=1.0 fixed) ...")
    stage1_rows = []
    atr_cache = {}
    for m in WINDOW_MULTIPLES:
        window = 14 * m
        atr_cache[window] = atr_fn(df, window)
        cfg = BidirectionalConfig(grid_atr_window=window, grid_atr_mult=1.0, tp_atr_mult=1.0)
        result = run_bidirectional_backtest(df, atr_cache[window], direction, cfg)
        row = summarize(f"window={window}(x{m})", df, result, {"grid_atr_window": window, "multiple": m})
        stage1_rows.append(row)
        print(f"      window={window}(x{m}): peak=${row['peak_equity']:,.0f} final=${row['final_equity']:,.0f} "
              f"ruined={row['ruined']}({row['ruin_time']}) mdd={row['max_drawdown']:.1%}")

    stage1_df = pd.DataFrame(stage1_rows)
    stage1_path = os.path.join(args.report_dir, "03c_stage1_window_scan.csv")
    stage1_df.to_csv(stage1_path, index=False)

    survivors = stage1_df[~stage1_df["ruined"]]
    if len(survivors):
        best_window_row = survivors.loc[survivors["final_equity"].idxmax()]
    else:
        best_window_row = stage1_df.loc[stage1_df["final_equity"].idxmax()]
    best_window = int(best_window_row["grid_atr_window"])
    print(f"      best survivable window: {best_window} "
          f"({'no survivors, picking least-bad' if not len(survivors) else ''})")

    print(f"[3/4] Stage 2: sweeping grid_atr_mult x tp_atr_mult over {ATR_MULT_GRID} "
          f"at window={best_window} ...")
    atr_best = atr_cache[best_window]
    stage2_rows = []
    for grid_mult in ATR_MULT_GRID:
        for tp_mult in ATR_MULT_GRID:
            cfg = BidirectionalConfig(grid_atr_window=best_window, grid_atr_mult=grid_mult, tp_atr_mult=tp_mult)
            result = run_bidirectional_backtest(df, atr_best, direction, cfg)
            row = summarize(f"grid={grid_mult},tp={tp_mult}", df, result,
                             {"grid_atr_mult": grid_mult, "tp_atr_mult": tp_mult})
            stage2_rows.append(row)
            print(f"      grid={grid_mult},tp={tp_mult}: peak=${row['peak_equity']:,.0f} "
                  f"final=${row['final_equity']:,.0f} ruined={row['ruined']}({row['ruin_time']}) "
                  f"mdd={row['max_drawdown']:.1%}")

    stage2_df = pd.DataFrame(stage2_rows)
    stage2_path = os.path.join(args.report_dir, "03c_stage2_mult_scan.csv")
    stage2_df.to_csv(stage2_path, index=False)

    survivors2 = stage2_df[~stage2_df["ruined"]]
    if len(survivors2):
        best_row = survivors2.loc[survivors2["final_equity"].idxmax()]
    else:
        best_row = stage2_df.loc[stage2_df["final_equity"].idxmax()]

    print("[4/4] Writing report ...")
    n_survived_1 = int((~stage1_df["ruined"]).sum())
    n_survived_2 = int((~stage2_df["ruined"]).sum())
    lines = [
        "# 网格参数M5重新校准报告(阶段3b延续)", "",
        "## 背景", "",
        "03b直接把阶段3的H1网格参数(grid_atr_window=14根H1≈14小时)原样用在M5 ATR上"
        "(14根M5≈70分钟)，网格间距缩小了十几倍，一个月内爆仓。这里保留M5(用户选择)，"
        "重新校准间距参数。", "",
        "## 阶段1：网格ATR窗口扫描(倍数×14根M5，mult=tp=1.0固定)", "",
        "| 窗口(根/倍数/小时) | 峰值权益 | 最终权益 | 最大回撤 | 是否破产 | 破产时间 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in stage1_df.iterrows():
        hours = r["grid_atr_window"] * 5 / 60
        lines.append(f"| {r['grid_atr_window']:.0f}/x{r['multiple']:.0f}/{hours:.1f}h | "
                      f"${r['peak_equity']:,.0f} | ${r['final_equity']:,.0f} | {r['max_drawdown']:.1%} | "
                      f"{'是' if r['ruined'] else '否'} | {r['ruin_time']} |")
    lines += ["", f"{n_survived_1}/{len(stage1_df)}个窗口全程存活。选定窗口={best_window}根"
              f"({best_window*5/60:.1f}小时)进入阶段2细化。", "",
              "## 阶段2：网格/止盈ATR倍数扫描", "",
              "| 网格倍数 | 止盈倍数 | 峰值权益 | 最终权益 | 最大回撤 | 是否破产 | 破产时间 |",
              "|---|---|---|---|---|---|---|"]
    for _, r in stage2_df.iterrows():
        lines.append(f"| {r['grid_atr_mult']} | {r['tp_atr_mult']} | ${r['peak_equity']:,.0f} | "
                      f"${r['final_equity']:,.0f} | {r['max_drawdown']:.1%} | "
                      f"{'是' if r['ruined'] else '否'} | {r['ruin_time']} |")
    lines += ["", f"{n_survived_2}/{len(stage2_df)}个组合全程存活。", "",
              "## 最终推荐配置", "",
              f"- grid_atr_window = {best_window}根M5({best_window*5/60:.1f}小时)",
              f"- grid_atr_mult = {best_row['grid_atr_mult']}",
              f"- tp_atr_mult = {best_row['tp_atr_mult']}",
              f"- 峰值权益 ${best_row['peak_equity']:,.0f}，最终权益 ${best_row['final_equity']:,.0f}，"
              f"最大回撤 {best_row['max_drawdown']:.1%}，{'破产于'+best_row['ruin_time'] if best_row['ruined'] else '全程存活'}",
              "", "## 结论与下一步", "",
              f"- 完整数据：{stage1_path}、{stage2_path}。",
              "- max_layers(8)/multiplier(2倍)/initial_lot(0.01)本次未重新搜索，"
              "仍是阶段3基线的默认值——如果这版网格参数校准后表现依然不理想，"
              "这三个是下一步要检查的对象。",
              ""]

    report_path = os.path.join(args.report_dir, "03c_grid_parameter_search_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

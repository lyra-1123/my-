#!/usr/bin/env python3
"""Phase 3b, part 2: run the bidirectional martingale engine on the full
M5 history, gated by the 12 candidates' reconciled direction (03b_build_
direction_gate.py's output), and compare against the phase 3 baseline
(long-only, H1, no filter, reports/03_baseline_backtest_report.md).

Usage:
    python scripts/03b_bidirectional_backtest.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from factors.library import atr as atr_fn  # noqa: E402
from backtest.martingale_bidirectional import BidirectionalConfig, run_bidirectional_backtest  # noqa: E402


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
    assert len(gate) == len(df), "direction gate / M5 bar count mismatch"
    atr14 = atr_fn(df, 14)

    print("[2/3] Running bidirectional backtest ...")
    cfg = BidirectionalConfig()
    result = run_bidirectional_backtest(df, atr14, gate["direction"], cfg)

    eq = result.equity_curve
    peak_equity = eq.max()
    peak_time = df["time"].iloc[eq.idxmax()]
    mdd = max_drawdown(eq)
    n_tp_long = int(((result.trades["type"] == "take_profit") & (result.trades["side"] == "long")).sum()) if len(result.trades) else 0
    n_tp_short = int(((result.trades["type"] == "take_profit") & (result.trades["side"] == "short")).sum()) if len(result.trades) else 0
    n_stop_out = int((result.trades["type"] == "stop_out").sum()) if len(result.trades) else 0
    final_equity = eq.iloc[-1]

    print(f"      peak equity ${peak_equity:,.0f} on {peak_time}, final equity ${final_equity:,.0f}")
    print(f"      max drawdown {mdd:.1%}, max layers long/short {result.max_layers_long}/{result.max_layers_short}")
    print(f"      take-profits long/short {n_tp_long}/{n_tp_short}, stop-outs {n_stop_out}")
    print(f"      ruined: {result.ruin_time}")

    print("[3/3] Writing report ...")
    lines = [
        "# 双向网格马丁回测报告(阶段3b)", "",
        "## 方法", "",
        "把信号设计track验证的12个候选(见`02v_final_signal_specification.md`)合并出的"
        "方向门(`03b_build_direction_gate.py`)接入新的双向网格引擎"
        "(`src/backtest/martingale_bidirectional.py`)：方向门允许做多时开多头网格，"
        "允许做空时开空头网格；**信号反手只冻结旧方向的新增层，不强平**(第一版曾"
        "尝试反手强平，结果在网格最深、浮亏最大的时刻剁仓，一个月就爆仓，已改为"
        "更保守的\"冻结不强平\"策略，两个方向的网格可以同时存在)。网格自身的加仓/"
        "止盈逻辑跟阶段3基线一致(ATR间距/ATR止盈/2倍加仓/最多8层)，只是现在M5"
        "原生运行、双向对称。参数(初始手数0.01/2倍加仓/最多8层/1xATR(14)间距和"
        "止盈/$10000初始资金/1:200杠杆)直接沿用阶段3基线的默认值，尚未针对M5+"
        "双向重新调优——这是第一次跑通，不是最终优化版本。", "",
        "**空头过夜利息(+2.0美元/手/天)是假设值**，不是从具体经纪商校准的，实盘前需要"
        "核实。", "",
        "## 结果", "",
        f"- 峰值权益：${peak_equity:,.0f}（{peak_time}）",
        f"- 最终权益：${final_equity:,.0f}",
        f"- 最大回撤：{mdd:.1%}",
        f"- 最多同时层数：多头{result.max_layers_long}层 / 空头{result.max_layers_short}层",
        f"- 止盈次数：多头{n_tp_long}次 / 空头{n_tp_short}次",
        f"- 强平(爆仓)次数：{n_stop_out}",
        f"- 是否破产：{'是，' + str(result.ruin_time) if result.ruin_time else '否，全程存活'}",
        "",
        "## 对照：阶段3基线(纯多头/H1/无过滤器)", "",
        "峰值权益$104,961(2012-10-09)，随后回撤95%，2013-04-15被黄金历史级暴跌"
        "一根H1 bar打出-$105,418强平，账户破产。", "",
        "## 结论与下一步", "",
        "- 完整交易记录：见`data/clean/direction_gate_M5.parquet`(方向门)、"
        "`reports/03b_direction_gate_trades.csv`(方向门实际拼接出的净头寸交易)。",
        "- 这是第一次把信号设计track接入真实马丁引擎的结果，网格自身参数"
        "(间距/止盈倍数/加仓层数/初始手数)还是阶段3基线的默认值，没有针对"
        "M5+双向+方向门重新调优——如果这版结果看起来有希望，下一步应该对这些"
        "网格参数做类似前面因子的网格搜索/walk-forward验证，而不是直接采用默认值。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "03b_bidirectional_backtest_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    result.trades.to_csv(os.path.join(args.report_dir, "03b_trades.csv"), index=False)
    if len(result.blowups):
        result.blowups.to_csv(os.path.join(args.report_dir, "03b_blowups.csv"), index=False)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

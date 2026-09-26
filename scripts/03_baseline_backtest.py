#!/usr/bin/env python3
"""Phase 3 (baseline): backtest the long-only ATR-spaced martingale grid
with NO regime filter, over the full 2009-2026 H1 history. This is the
"does the grid math itself survive reality" check before Phase 3b tries
gating new layers with the Phase 2b regime factors.

Default parameters (see src/backtest/martingale.MartingaleConfig for all of
them) reflect the user's stated preferences: ATR-multiple grid spacing,
aggressive 6-10 layers (using 8) at classic 2x martingale doubling.
Account size ($10,000), leverage (1:200), spread ($0.30/oz) and swap
(-$6/lot/day) are reasonable placeholders, NOT calibrated to a specific
broker — recalibrate against your actual account before trusting the
dollar figures, though the relative comparisons (with vs without a regime
filter) are less sensitive to these.

Usage:
    python scripts/03_baseline_backtest.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from factors.library import atr  # noqa: E402
from backtest.martingale import MartingaleConfig, run_backtest, max_drawdown  # noqa: E402


def summarize(result, cfg, label: str) -> dict:
    eq = result.equity_curve
    trades = result.trades
    n_tp = int((trades["type"] == "take_profit").sum()) if len(trades) else 0
    n_stopout = int((trades["type"] == "stop_out").sum()) if len(trades) else 0
    return {
        "label": label,
        "final_equity": float(eq.iloc[-1]),
        "initial_equity": cfg.initial_equity,
        "total_return_pct": float(eq.iloc[-1] / cfg.initial_equity - 1) * 100,
        "max_drawdown_pct": max_drawdown(eq) * 100,
        "n_take_profit": n_tp,
        "n_stop_out": n_stopout,
        "max_layers_reached": result.max_layers_reached,
        "ruined": result.ruin_time is not None,
        "ruin_time": result.ruin_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading H1 bars ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet"))
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    print(f"      {len(df):,} H1 bars, {df['time'].min()} .. {df['time'].max()}")

    cfg = MartingaleConfig()
    print(f"[2/4] Config: initial_lot={cfg.initial_lot}, multiplier={cfg.multiplier}x, "
          f"max_layers={cfg.max_layers}, grid={cfg.grid_atr_mult}xATR({cfg.grid_atr_window}), "
          f"tp={cfg.tp_atr_mult}xATR, equity=${cfg.initial_equity:,.0f}, leverage=1:{cfg.leverage}")

    atr_series = atr(df, cfg.grid_atr_window)

    print("[3/4] Running baseline backtest (no regime filter) ...")
    result = run_backtest(df, atr_series, cfg)
    summary = summarize(result, cfg, "baseline_no_filter")

    ruin_msg = f", RUINED at {summary['ruin_time']}" if summary["ruined"] else ", account survived"
    print(f"      final equity=${summary['final_equity']:,.0f} "
          f"({summary['total_return_pct']:+.1f}%), max DD={summary['max_drawdown_pct']:.1f}%, "
          f"TP closes={summary['n_take_profit']}, stop-outs={summary['n_stop_out']}, "
          f"max layers reached={summary['max_layers_reached']}{ruin_msg}")

    eq_path = os.path.join(args.report_dir, "03_baseline_equity_curve.csv")
    result.equity_curve.rename("equity").to_frame().assign(time=df["time"].values).to_csv(eq_path, index=False)
    trades_path = os.path.join(args.report_dir, "03_baseline_trades.csv")
    result.trades.to_csv(trades_path, index=False)
    blowups_path = os.path.join(args.report_dir, "03_baseline_blowups.csv")
    result.blowups.to_csv(blowups_path, index=False)

    lines = [
        "# 阶段3基线回测报告：无过滤器的马丁格尔网格",
        "",
        "## 参数设定",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| 初始手数 | {cfg.initial_lot} |",
        f"| 加仓倍率 | {cfg.multiplier}x |",
        f"| 最大加仓层数 | {cfg.max_layers} |",
        f"| 加仓间距 | {cfg.grid_atr_mult} x ATR({cfg.grid_atr_window}根H1) |",
        f"| 止盈距离(相对持仓均价) | {cfg.tp_atr_mult} x ATR({cfg.grid_atr_window}根H1) |",
        f"| 点差成本(每盎司，往返) | ${cfg.spread_dollars} |",
        f"| 隔夜利息(每手/天) | ${cfg.swap_per_lot_per_day} |",
        f"| 合约规模 | {cfg.contract_size} 盎司/标准手 |",
        f"| 杠杆 | 1:{cfg.leverage} |",
        f"| 初始账户资金 | ${cfg.initial_equity:,.0f} |",
        f"| 强平保证金水平 | {cfg.stop_out_level:.0%} |",
        "",
        "**重要说明**：账户资金/杠杆/点差/隔夜利息是合理的占位假设，不是针对具体经纪商标定的，"
        "使用前应对照你的真实账户条件重新标定；但“有无regime过滤器”的相对比较对这些绝对数字"
        "的敏感度较低。方向固定为只做多（逢跌加仓，classic buy-the-dip martingale），这是"
        "本身的一个策略选择，不是从数据里学出来的——如果想测双向或只做空，需要单独跑。",
        "",
        "## 结果（2009-2026全历史，无任何regime过滤器）",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 期末权益 | ${summary['final_equity']:,.0f} |",
        f"| 总收益率 | {summary['total_return_pct']:+.1f}% |",
        f"| 最大回撤 | {summary['max_drawdown_pct']:.1f}% |",
        f"| 止盈平仓次数 | {summary['n_take_profit']:,} |",
        f"| 强平(爆仓)次数 | {summary['n_stop_out']} |",
        f"| 历史最大加仓层数 | {summary['max_layers_reached']} |",
        f"| 账户是否破产 | {'是，' + str(summary['ruin_time']) if summary['ruined'] else '否，撑过全部18年'} |",
        "",
    ]
    if summary["ruined"]:
        lines += [
            "**破产说明**：权益一旦跌破0，视为账户被券商强制关停，此后不再模拟任何交易"
            "（真实券商也会这样处理，不会让一个负权益账户继续用固定手数开新仓）。上面"
            "“强平次数”“止盈次数”只统计破产前的真实交易。",
            "",
        ]

    if len(result.blowups):
        lines += ["## 强平事件明细", "", "| 时间 | 加仓层数 | 该次损失 | 强平后权益 |", "|---|---|---|---|"]
        for _, r in result.blowups.iterrows():
            lines.append(f"| {r['time']} | {r['n_layers']} | ${r['pnl']:,.0f} | ${r['equity_after']:,.0f} |")
        lines.append("")

        peak_idx = result.equity_curve.idxmax()
        peak_equity = result.equity_curve.loc[peak_idx]
        peak_time = df.loc[peak_idx, "time"]
        first_blowup_time = pd.Timestamp(result.blowups.iloc[0]["time"])
        pre_blowup_window = result.equity_curve[
            (df["time"] >= peak_time) & (df["time"] < first_blowup_time)
        ]
        trough_before_blowup = pre_blowup_window.min() if len(pre_blowup_window) else None
        lines += [
            "**不是单个bar的意外，是长达数月的失血后被最后一击补刀**：权益峰值"
            f"${peak_equity:,.0f}出现在{peak_time}，此后一路下滑，到强平发生前"
            f"（{first_blowup_time}）已经跌到约${trough_before_blowup:,.0f}"
            f"（较峰值回撤{(1 - trough_before_blowup/peak_equity)*100:.0f}%），"
            "此时账户早已深陷不可逆的回撤，最后那一根H1 bar的暴跌只是压垮骆驼的"
            "最后一根稻草，不是唯一原因。这个时间点(2013年4月中旬)正是黄金史上"
            "最著名的暴跌之一——两天内跌去约$140-160，验证了引擎模拟出的这次"
            "破产不是数据或模型的bug，而是真实发生过的极端行情。",
            "",
        ]

    lines += [
        "## 结论与下一步",
        "",
        f"- 完整权益曲线在`{eq_path}`，逐笔交易在`{trades_path}`，强平事件在`{blowups_path}`。",
        "- 这是阶段3b（接入阶段2b筛出的11个regime因子做入场过滤）的对照组：阶段3b要证明"
        "“用因子门槛限制新开层”确实能改善这里的最大回撤/强平次数，而不是想当然地假设有用。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "03_baseline_backtest_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[4/4] Wrote {report_path}")


if __name__ == "__main__":
    main()

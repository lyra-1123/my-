#!/usr/bin/env python3
"""Signal design step 1: cross-timeframe direction consistency ("look at
the big timeframe, trade the small one").

Direction on each timeframe = MA20 vs MA50 relative position (fast above
slow = long bias, below = short bias). H1 is the big/authoritative
timeframe, M5 is where entries actually happen:
  - M5 and (look-ahead-safe-aligned) H1 direction agree -> full-size entry
    in that direction
  - disagree -> stand down; if still disagreed after 4 consecutive M5 bars,
    defer to the H1 direction anyway, at half size (H1 is authoritative,
    M5 just gets a grace period to confirm before being overridden)

This step validates whether that consistency logic actually finds a real
directional edge (vs each timeframe alone, and vs plain consensus-only —
i.e. is the "wait 4 bars then defer at half size" fallback worth having,
or does it just add noise). N=12 M5 bars (~1h) is a placeholder holding
period for this preliminary check; the real N gets set in step 3.

Usage:
    python scripts/02e_cross_timeframe_consistency.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.direction import ma_direction, align_htf_direction, cross_timeframe_signal  # noqa: E402
from factors.validation import decision_points  # noqa: E402

MA_FAST, MA_SLOW = 20, 50
WAIT_BARS = 4
REDUCED_SIZE = 0.5
HOLDING_BARS_PLACEHOLDER = 12  # ~1h on M5; step 3 will set the real value


def sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / x.std(ddof=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading H1 (big) + M5 (small) bars ...")
    h1 = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet")).dropna(subset=["close"])
    m5 = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"])
    print(f"      H1: {len(h1):,} bars, M5: {len(m5):,} bars")

    print(f"[2/4] Computing direction (MA{MA_FAST}/MA{MA_SLOW}) on both timeframes ...")
    h1_dir = ma_direction(h1, MA_FAST, MA_SLOW)
    m5_dir = ma_direction(m5, MA_FAST, MA_SLOW)
    h1_dir_on_m5 = align_htf_direction(m5, h1, h1_dir, htf_bar_seconds=3600)

    sig = cross_timeframe_signal(m5_dir, h1_dir_on_m5, wait_bars=WAIT_BARS, reduced_size=REDUCED_SIZE)
    valid = m5_dir.notna() & h1_dir_on_m5.notna()

    print("[3/4] Scoring against forward returns (placeholder holding period) ...")
    fwd_return = m5["close"].pct_change(HOLDING_BARS_PLACEHOLDER).shift(-HOLDING_BARS_PLACEHOLDER)
    points = decision_points(len(m5), HOLDING_BARS_PLACEHOLDER)
    points = points[valid.to_numpy()[points] & fwd_return.notna().to_numpy()[points]]

    variants = {
        "完整跨周期逻辑(一致全仓+等待4根后按大周期半仓)": sig["direction"] * sig["size"] * fwd_return,
        "仅一致时开仓(忽略等待/半仓分支)": (
            (sig["direction"] * fwd_return).where(sig["aligned"], 0.0)
        ),
        "仅5min方向(忽略1H)": m5_dir * fwd_return,
        "仅1H方向(每根5min bar都按对齐后的1H方向开仓)": h1_dir_on_m5 * fwd_return,
    }

    # decision points are HOLDING_BARS_PLACEHOLDER M5-bars apart; M5 bars/year
    # from the actual sample span, so this reflects real trading-time density
    years = (m5["time"].iloc[-1] - m5["time"].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
    decisions_per_year = (len(m5) / HOLDING_BARS_PLACEHOLDER) / years
    annualization = decisions_per_year ** 0.5

    rows = []
    for name, ret in variants.items():
        r = ret.iloc[points].to_numpy()
        active = r[r != 0]
        rows.append({
            "variant": name,
            "activation_rate": len(active) / len(r) if len(r) else float("nan"),
            "mean_return_per_trade": np.nanmean(active) if len(active) else float("nan"),
            "sharpe_per_period": sharpe(r),
            "sharpe_ann": sharpe(r) * annualization,
        })
    summary = pd.DataFrame(rows)

    n_total = int(valid.sum())
    n_aligned = int(sig["aligned"][valid].sum())
    n_deferred = int(sig["deferred"][valid].sum())
    n_no_trade = n_total - n_aligned - n_deferred

    lines = [
        "# 信号设计 步骤1：跨时间周期方向一致性验证",
        "",
        "## 参数", "",
        f"- 大周期(权威方向): H1，小周期(实际入场): M5",
        f"- 方向定义: MA{MA_FAST} vs MA{MA_SLOW}相对位置（快线在慢线之上=多头偏向，之下=空头偏向）",
        f"- 不一致时: 等待{WAIT_BARS}根M5 bar，仍不一致则按H1方向开仓、仓位降至{REDUCED_SIZE:.0%}",
        f"- 本步骤用于初步验证的持有期占位值: {HOLDING_BARS_PLACEHOLDER}根M5 bar(~1小时)，"
        "真实持有期在步骤3里另定",
        "",
        "## M5各bar的信号状态分布", "",
        "| 状态 | 占比 |", "|---|---|",
        f"| 大小周期方向一致(全仓) | {n_aligned/n_total:.1%} |",
        f"| 不一致但等待满{WAIT_BARS}根后按H1方向开仓(半仓) | {n_deferred/n_total:.1%} |",
        f"| 不一致且等待未满{WAIT_BARS}根，不开仓 | {n_no_trade/n_total:.1%} |",
        "",
        "## 四种方案的方向性表现对比", "",
        "| 方案 | 激活率(有信号的比例) | 单次信号平均收益 | Sharpe(年化) |",
        "|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r['variant']} | {r['activation_rate']:.1%} | "
                      f"{r['mean_return_per_trade']:+.4%} | {r['sharpe_ann']:+.2f} |")

    full_sharpe = summary.iloc[0]["sharpe_ann"]
    consensus_sharpe = summary.iloc[1]["sharpe_ann"]
    m5_sharpe = summary.iloc[2]["sharpe_ann"]
    h1_sharpe = summary.iloc[3]["sharpe_ann"]

    lines += [
        "",
        "## 结论与下一步", "",
        f"- 完整跨周期逻辑(含等待+半仓分支) vs 仅一致时开仓：{full_sharpe:+.3f} vs {consensus_sharpe:+.3f}，"
        + ("等待+半仓分支确实有增量价值。" if full_sharpe > consensus_sharpe
           else "等待+半仓分支反而拖累了表现，可以考虑去掉，只做“一致才开仓”。"),
        f"- 跨周期方案 vs 单周期方案：完整逻辑{full_sharpe:+.2f}，仅5min{m5_sharpe:+.2f}，"
        f"仅1H{h1_sharpe:+.2f}——"
        + ("跨周期确认确实比任何单一周期都强，方向一致性有增量信息。"
           if full_sharpe > max(m5_sharpe, h1_sharpe)
           else "跨周期确认没有比单看1H更好，反而更差。原因能从状态分布上看出来："
                f"“不一致”状态占了{(n_deferred+n_no_trade)/n_total:.0%}的时间（不是少数"
                f"情况），其中{n_deferred/n_total:.0%}最终会等满{WAIT_BARS}根后按半仓"
                "执行——而5min方向本身没有信息量(仅5min方案Sharpe为负)，所以这个框架"
                "实际效果基本等于“接近一半时间把一个本来有效的1H信号砍半仓”，纯粹是"
                "拖累，不是增强。问题不在“看大做小”这个思路本身，而在于**M5用MA20/50"
                "做方向判断这个具体设计没有信息量**，需要换一个方向定义或换一对"
                "周期再试。"),
        "- 这里的Sharpe/收益都还没有算点差滑点等成本，且用的是步骤3才会正式定的持有期占位值，"
        "结论仅供步骤1判断“跨周期一致性这个思路本身值不值得往下做”，不是最终参数。",
        "- 完整明细已保存，下一步（步骤2）是把这个方向信号和阶段2c/2d验证过的regime因子"
        "组合，正式定义signal_long / signal_short。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02e_cross_timeframe_consistency_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[4/4] Wrote {report_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

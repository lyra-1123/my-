#!/usr/bin/env python3
"""Signal design step 2 (v2): define signal_long / signal_short on M5,
using 2 conditions ANDed (not 3) — the user's directive after v1 (3
conditions on H1) fired only 6/12 times in 18 years, too rare to be
usable.

v1 -> v2 changes:
  - Timeframe: H1 -> M5 (the actual intraday entry timeframe; H1-level
    factors were far too selective at M5 entry frequency anyway)
  - AND count: 3 -> 2 (dropped 因子条件A/mean_reversion_speed_20 — its
    lone Sharpe was ~0 in v1, i.e. it wasn't adding directional
    information, just cutting sample size)

Conditions (recomputed directly on M5 bars, same formulas/window-in-BARS
as Phase 2c's H1 picks — NOT re-validated at M5 granularity yet, since a
100-bar window is a very different time span on M5 (~8.3h) vs H1
(~100h); this is a first pass, flagged clearly, not a claim that these
exact parameters are M5-optimal):
  - 因子条件B: autocorr_returns_100 (M5) in its own top 20% (momentum/
    trending regime) + last M5 bar's return sign for direction
  - MFI条件: mfi_20 (M5) recovering from oversold (<20) for long, or
    falling from overbought (>80) for short

signal_long = B(long) & MFI(long); signal_short mirrors it.

Usage:
    python scripts/02f_signal_definition.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import autocorr_returns, mfi  # noqa: E402

AC_WINDOW = 100
MFI_WINDOW = 20
QUANTILE = 0.8
MFI_OVERSOLD, MFI_OVERBOUGHT = 20, 80
HOLDING_BARS_PLACEHOLDER = 12  # ~1h on M5; step 3 sets the real value


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

    print("[1/3] Loading M5 bars ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"])
    close = df["close"]
    print(f"      {len(df):,} M5 bars")

    print(f"[2/3] Computing autocorr_returns_{AC_WINDOW} and mfi_{MFI_WINDOW} on M5 ...")
    ac = autocorr_returns(df, AC_WINDOW)
    mfi_series = mfi(df, MFI_WINDOW)

    ac_high = ac >= ac.quantile(QUANTILE)
    last_ret = close.diff()
    cond_b_long, cond_b_short = ac_high & (last_ret > 0), ac_high & (last_ret < 0)
    mfi_long = (mfi_series.shift(1) < MFI_OVERSOLD) & (mfi_series >= MFI_OVERSOLD)
    mfi_short = (mfi_series.shift(1) > MFI_OVERBOUGHT) & (mfi_series <= MFI_OVERBOUGHT)

    signal_long = cond_b_long & mfi_long
    signal_short = cond_b_short & mfi_short

    valid = ac.notna() & mfi_series.notna()
    fwd_return = close.pct_change(HOLDING_BARS_PLACEHOLDER).shift(-HOLDING_BARS_PLACEHOLDER)
    valid_fwd = fwd_return.notna()

    direction = pd.Series(0.0, index=df.index)
    direction[signal_long] = 1.0
    direction[signal_short] = -1.0
    strategy_return = direction * fwd_return

    n_valid = int(valid.sum())
    n_long, n_short = int((signal_long & valid).sum()), int((signal_short & valid).sum())

    print("[3/3] Scoring against forward returns (placeholder holding period) ...")
    years = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
    r_long = strategy_return[signal_long & valid_fwd].to_numpy()
    r_short = strategy_return[signal_short & valid_fwd].to_numpy()
    r_all = np.concatenate([r_long, r_short])

    def ann_sharpe(r):
        trades_per_year = len(r) / years
        return sharpe(r) * (trades_per_year ** 0.5) if trades_per_year > 0 else float("nan")

    single_rows = []
    for name, mask, longmask, shortmask in [
        ("仅B(autocorr_returns+上根bar方向)", ac_high, cond_b_long, cond_b_short),
        ("仅MFI(超买超卖回归穿越)", mfi_series.notna(), mfi_long, mfi_short),
    ]:
        rows_dir = pd.Series(0.0, index=df.index)
        rows_dir[longmask] = 1.0
        rows_dir[shortmask] = -1.0
        active = longmask | shortmask
        ret = (rows_dir * fwd_return)[active & valid_fwd].to_numpy()
        single_rows.append({"condition": name, "n": int(active.sum()), "sharpe": sharpe(ret)})

    lines = [
        "# 信号设计 步骤2(v2)：M5级别 signal_long / signal_short（2条件AND）",
        "",
        "## v1->v2变化", "",
        "- 周期：H1改为M5（真正的日内入场周期）",
        "- AND数量：3改为2（去掉了v1里单独看Sharpe接近0的“条件A/mean_reversion_speed”，"
        "只保留B+MFI）",
        "- **注意**：B和MFI的窗口(100根/20根)沿用的是阶段2c在H1上验证过的bar数，直接"
        "搬到M5上是不同的绝对时间跨度(100根M5≈8.3小时，20根M5≈100分钟)，**还没有在M5"
        "粒度上重新验证过**，这是第一版尝试，不代表这组参数在M5上是最优的。",
        "",
        "## 信号定义", "",
        f"- 因子条件B：`autocorr_returns_{AC_WINDOW}`(M5) >= 自身历史80%分位 + 上一根M5 bar"
        "方向定多空",
        f"- MFI条件：`mfi_{MFI_WINDOW}`(M5) 从超卖区(<{MFI_OVERSOLD})回升=多头，"
        f"从超买区(>{MFI_OVERBOUGHT})回落=空头",
        "- signal_long = B(多) & MFI(多)；signal_short = B(空) & MFI(空)",
        f"- 验证用占位持有期: {HOLDING_BARS_PLACEHOLDER}根M5(~1小时)，真实值步骤3另定",
        "",
        "## 信号触发频率（18年M5数据）", "",
        "| | 次数 | 占全部有效bar的比例 |", "|---|---|---|",
        f"| signal_long | {n_long} | {n_long/n_valid:.3%} |",
        f"| signal_short | {n_short} | {n_short/n_valid:.3%} |",
        "",
        "## 组合信号 vs 单条件方向性表现（未计点差滑点，占位持有期）", "",
        "| | 触发次数 | Sharpe(未年化) | Sharpe(按每年触发次数年化) |", "|---|---|---|---|",
        f"| signal_long+signal_short组合 | {len(r_all)} | {sharpe(r_all):+.3f} | {ann_sharpe(r_all):+.2f} |",
        f"| 仅signal_long | {len(r_long)} | {sharpe(r_long):+.3f} | {ann_sharpe(r_long):+.2f} |",
        f"| 仅signal_short | {len(r_short)} | {sharpe(r_short):+.3f} | {ann_sharpe(r_short):+.2f} |",
    ]
    for r in single_rows:
        lines.append(f"| {r['condition']}(单独,不AND) | {r['n']} | {r['sharpe']:+.3f} | - |")

    lines += [
        "",
        "## 结论与下一步", "",
        f"- 2条件AND后，18年M5数据里signal_long触发{n_long}次、signal_short触发{n_short}次"
        f"（合计占比{(n_long+n_short)/n_valid:.3%}，约合每年{ (n_long+n_short)/years:.0f}次，"
        "触发频率对日内策略来说是可用的量级了）。",
        f"- 但edge本身偏弱且多空不对称：signal_long年化Sharpe约{ann_sharpe(r_long):+.2f}，"
        f"signal_short只有{ann_sharpe(r_short):+.2f}（接近0）——这组条件对多头方向更有效，"
        "对空头方向基本没有区分力，可能和金价样本期内长期偏多头的特征有关。这是占位"
        "持有期(1小时)下的初步结果，步骤3定好真实持有期/止损后需要重新算一遍，不代表"
        "最终结论，但空头这一侧值得留意，后面调参时可以重点看能不能把它调起来。",
        "- 完整数据在报告里；下一步（步骤3）把“不用简单阈值、用穿越检测”应用到B条件本身"
        "（目前B是分位阈值），并正式确定入场时机(下一根bar开盘价)、持有期N、止损方式。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02f_signal_definition_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")
    print(f"signal_long triggers: {n_long}, signal_short triggers: {n_short}")


if __name__ == "__main__":
    main()

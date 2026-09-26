#!/usr/bin/env python3
"""Signal design step 2: define signal_long / signal_short from the Phase
2c/2d validated factors (not new indicators — repurposing what Phase 2
already proved has regime-predictive power, into directional conditions).

Phase 2c's factors were built to answer "is now safe for a martingale
ladder", not "which way is price going" — so each one needs a direction
attached, not just a level:

  - 因子条件A = mean_reversion_speed_20 (validated mean_reversion pick):
    high value = price is currently reverting quickly toward its own MA20.
    Direction comes from WHICH SIDE of the MA it's reverting from:
    below MA -> long (bouncing up), above MA -> short (pulling back down).
  - 因子条件B = autocorr_returns_100 (validated momentum pick): high value
    = trending regime, recent moves tend to continue. Direction comes from
    the sign of the last bar's return: last bar up -> long, down -> short.
  - MFI条件 = mfi_20 recovering from oversold (<20, crossing back up) for
    long, or falling from overbought (>80, crossing back down) for short.

signal_long fires only when A, B and MFI all agree on "long" (AND, per the
template); signal_short is the mirror. Note this is a DIRECTION/ENTRY
signal — separate from Phase 2d's K-of-4 vote, which answers a different
question (once IN a martingale ladder, is it safe to add another layer).
Both will eventually feed the same live strategy, at different decision
points.

vol_of_vol and dist_from_high (Phase 2c's other two picks) aren't used
here — they don't have as natural a directional reading. They remain
candidates for the "safe to add a layer" gate instead.

Usage:
    python scripts/02f_signal_definition.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.validation import decision_points  # noqa: E402

MRS_COL = "mean_reversion_speed_20"
AC_COL = "autocorr_returns_100"
MFI_COL = "mfi_20"
MA_WINDOW = 20  # matches mean_reversion_speed_20's own internal MA
QUANTILE = 0.8  # "high" = top 20% of the factor's own history
MFI_OVERSOLD, MFI_OVERBOUGHT = 20, 80
HOLDING_BARS_PLACEHOLDER = 24  # ~1 day of H1; step 3 sets the real value


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

    print("[1/3] Loading H1 factor table ...")
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_H1.parquet"))
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet")).dropna(subset=["close"])
    close = df["close"]

    print("[2/3] Building signal_long / signal_short ...")
    mrs = factors[MRS_COL]
    ac = factors[AC_COL]
    mfi = factors[MFI_COL]
    ma = close.rolling(MA_WINDOW).mean()

    mrs_high = mrs >= mrs.quantile(QUANTILE)
    ac_high = ac >= ac.quantile(QUANTILE)
    last_ret = close.diff()

    cond_a_long, cond_a_short = mrs_high & (close < ma), mrs_high & (close > ma)
    cond_b_long, cond_b_short = ac_high & (last_ret > 0), ac_high & (last_ret < 0)
    mfi_long = (mfi.shift(1) < MFI_OVERSOLD) & (mfi >= MFI_OVERSOLD)
    mfi_short = (mfi.shift(1) > MFI_OVERBOUGHT) & (mfi <= MFI_OVERBOUGHT)

    signal_long = cond_a_long & cond_b_long & mfi_long
    signal_short = cond_a_short & cond_b_short & mfi_short

    valid = mrs.notna() & ac.notna() & mfi.notna() & ma.notna()
    fwd_return = close.pct_change(HOLDING_BARS_PLACEHOLDER).shift(-HOLDING_BARS_PLACEHOLDER)
    points = decision_points(len(df), HOLDING_BARS_PLACEHOLDER)
    points = points[valid.to_numpy()[points] & fwd_return.notna().to_numpy()[points]]

    direction = pd.Series(0.0, index=df.index)
    direction[signal_long] = 1.0
    direction[signal_short] = -1.0
    strategy_return = direction * fwd_return

    n_valid = int(valid.sum())
    n_long, n_short = int(signal_long.sum()), int(signal_short.sum())

    print("[3/3] Scoring against forward returns (placeholder holding period) ...")
    # signal_long/short only fire a handful of times in 18 years, so
    # sub-sampling at generic spaced-out "decision points" (built for dense
    # per-bar factor testing) misses almost all of them — score directly at
    # the actual trigger bars instead (naturally non-overlapping at this
    # rarity: consecutive triggers are essentially never <N bars apart).
    valid_fwd = fwd_return.notna()
    r_long = strategy_return[signal_long & valid_fwd].to_numpy()
    r_short = strategy_return[signal_short & valid_fwd].to_numpy()
    r_all = np.concatenate([r_long, r_short])

    # each individual condition alone, for comparison (does the AND-of-3
    # actually beat any single condition, or just cut sample size?)
    single_rows = []
    for name, mask in [
        ("仅A(mean_reversion_speed+MA位置)", cond_a_long | cond_a_short),
        ("仅B(autocorr_returns+上根bar方向)", cond_b_long | cond_b_short),
        ("仅MFI(超买超卖回归穿越)", mfi_long | mfi_short),
    ]:
        rows_dir = pd.Series(0.0, index=df.index)
        if "A(" in name:
            rows_dir[cond_a_long] = 1.0
            rows_dir[cond_a_short] = -1.0
        elif "B(" in name:
            rows_dir[cond_b_long] = 1.0
            rows_dir[cond_b_short] = -1.0
        else:
            rows_dir[mfi_long] = 1.0
            rows_dir[mfi_short] = -1.0
        ret = (rows_dir * fwd_return).where(mask).iloc[points].dropna().to_numpy()
        single_rows.append({"condition": name, "activation_rate": mask.iloc[points].mean(),
                             "sharpe": sharpe(ret)})

    lines = [
        "# 信号设计 步骤2：定义 signal_long / signal_short",
        "",
        "## 信号定义", "",
        f"- 因子条件A：`{MRS_COL}` >= 自身历史80%分位（均值回归速度快）"
        f"，且 close 在MA{MA_WINDOW}下方=多头方向，上方=空头方向",
        f"- 因子条件B：`{AC_COL}` >= 自身历史80%分位（动量/趋势持续regime），"
        "且上一根bar收阳=多头方向，收阴=空头方向",
        f"- MFI条件：`{MFI_COL}` 从超卖区(<{MFI_OVERSOLD})上穿回{MFI_OVERSOLD}=多头确认，"
        f"从超买区(>{MFI_OVERBOUGHT})下穿回{MFI_OVERBOUGHT}=空头确认",
        "- signal_long = 条件A(多) & 条件B(多) & MFI(多)；signal_short同理反向",
        f"- 本步骤验证用占位持有期: {HOLDING_BARS_PLACEHOLDER}根H1(~1天)，真实值步骤3另定",
        "",
        "## 信号触发频率（18年H1数据）", "",
        "| | 次数 | 占全部有效bar的比例 |", "|---|---|---|",
        f"| signal_long | {n_long} | {n_long/n_valid:.3%} |",
        f"| signal_short | {n_short} | {n_short/n_valid:.3%} |",
        "",
        "## 组合信号 vs 单条件方向性表现（未计点差滑点，占位持有期）", "",
        "| | 激活次数 | Sharpe(未年化) |", "|---|---|---|",
        f"| signal_long+signal_short组合 | {len(r_all)} | {sharpe(r_all):+.3f} |",
        f"| 仅signal_long | {len(r_long)} | {sharpe(r_long):+.3f} |",
        f"| 仅signal_short | {len(r_short)} | {sharpe(r_short):+.3f} |",
    ]
    for r in single_rows:
        lines.append(f"| {r['condition']}(单独,不AND) | {int(r['activation_rate']*len(points))} | {r['sharpe']:+.3f} |")

    lines += [
        "",
        "## 结论与下一步", "",
        f"- 三条件AND之后，18年里signal_long只触发{n_long}次、signal_short{n_short}次"
        f"（合计占比{(n_long+n_short)/n_valid:.2%}）——这是三个本来就不算宽松的条件"
        "同时成立的必然结果，样本量非常小，下面的Sharpe数字统计上不稳定，仅供方向性"
        "参考，步骤6(walk-forward)之前不能当结论用。",
        "- 完整数据在报告里；下一步（步骤3）把“不用简单阈值、用穿越检测”应用到条件A/B"
        "本身（目前A/B是分位阈值，不是穿越），并正式确定入场时机(下一根bar开盘价)、"
        "持有期N、止损方式。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02f_signal_definition_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")
    print(f"signal_long triggers: {n_long}, signal_short triggers: {n_short}")


if __name__ == "__main__":
    main()

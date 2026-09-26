#!/usr/bin/env python3
"""Signal design: systematic directional factor mining on M5.

Step 2 used exactly 2 hand-picked factors (autocorr_returns_100 + mfi_20)
converted into direction with 2 hand-picked modes (momentum-gate,
reversion-crossover). This step tests EVERY factor in the library (~400
variants on M5) against BOTH modes systematically, rather than relying on
manual picks — mirroring how Phase 2c/2d avoided "family representative"
shortcuts for the regime-classification factors.

Two-stage funnel (same shape as Phase 2b/2c):
  Stage 1 (cheap): full-sample threshold, aggregate Sharpe at the actual
    trigger bars, for every (factor, mode) combination. Keep only those
    with enough activations to be statistically meaningful.
  Stage 2 (expensive): 5-fold walk-forward (threshold refit per fold, no
    look-ahead) on the Stage-1 survivors, with long/short broken out
    separately since Phase 2f found sharp long/short asymmetry.

Usage:
    python scripts/02h_directional_factor_mining.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import family, FAMILY_CATEGORY  # noqa: E402
from factors.validation import decision_points  # noqa: E402
from factors.direction import momentum_direction, reversion_direction, walk_forward_direction  # noqa: E402

HOLDING_BARS = 12  # ~1h on M5, same placeholder as step 2
MIN_ACTIVATIONS = 200  # stage-1 statistical floor
STAGE2_TOP_N = 25
NON_FACTOR_COLS = {"time", "session", "day_of_week", "close", "open"}


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

    print("[1/5] Loading cached M5 factor table ...")
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    close = factors["close"]
    print(f"      {len(factors):,} M5 bars, {factors.shape[1]} columns")

    fwd_return = close.pct_change(HOLDING_BARS).shift(-HOLDING_BARS)
    points = decision_points(len(factors), HOLDING_BARS)
    points = points[fwd_return.notna().to_numpy()[points]]

    variant_cols = [c for c in factors.columns if c not in NON_FACTOR_COLS]
    print(f"[2/5] Stage 1: scanning {len(variant_cols)} variants x 2 modes = "
          f"{len(variant_cols)*2} combinations ...")

    stage1 = []
    for col in variant_cols:
        f = factors[col]
        if f.isna().all():
            continue
        for mode in ("momentum", "reversion"):
            if mode == "momentum":
                thresh = f.quantile(0.8)
                direction = momentum_direction(f, close, thresh)
            else:
                lo, hi = f.quantile(0.2), f.quantile(0.8)
                direction = reversion_direction(f, lo, hi)
            ret = (direction * fwd_return)
            active_idx = direction[direction != 0].index.intersection(fwd_return.dropna().index)
            r = ret.loc[active_idx].to_numpy()
            d = direction.loc[active_idx].to_numpy()
            if len(r) < MIN_ACTIVATIONS:
                continue
            stage1.append({
                "variant": col, "family": family(col),
                "category": FAMILY_CATEGORY.get(family(col), "other"),
                "mode": mode, "n_active": len(r),
                "n_long": int((d > 0).sum()), "n_short": int((d < 0).sum()),
                "sharpe": sharpe(r),
                "sharpe_long": sharpe(r[d > 0]), "sharpe_short": sharpe(r[d < 0]),
            })

    stage1_df = pd.DataFrame(stage1)
    stage1_path = os.path.join(args.report_dir, "02h_directional_stage1_scan.csv")
    stage1_df.to_csv(stage1_path, index=False)
    print(f"      {len(stage1_df)}/{len(variant_cols)*2} combinations clear "
          f"n_active>={MIN_ACTIVATIONS} -> {stage1_path}")

    print(f"[3/5] Stage 2: walk-forward validating top {STAGE2_TOP_N} by |Sharpe| ...")
    top = stage1_df.reindex(stage1_df["sharpe"].abs().sort_values(ascending=False).index).head(STAGE2_TOP_N)

    stage2 = []
    for _, row in top.iterrows():
        f = factors[row["variant"]]
        wf = walk_forward_direction(f, close, fwd_return, points, row["mode"], n_folds=5)
        stage2.append({**row.to_dict(), **{f"wf_{k}": v for k, v in wf.items()}})
    stage2_df = pd.DataFrame(stage2)
    stage2_path = os.path.join(args.report_dir, "02h_directional_stage2_walkforward.csv")
    stage2_df.to_csv(stage2_path, index=False)

    print("[4/5] Selecting final validated candidates (OOS Sharpe>0 both directions active, "
          ">=3/5 folds positive) ...")
    validated = stage2_df[
        (stage2_df["wf_oos_sharpe_all"] > 0)
        & (stage2_df["wf_n_folds_positive"] >= 3)
        & (stage2_df["wf_n_long"] >= 30) & (stage2_df["wf_n_short"] >= 30)
    ].sort_values("wf_oos_sharpe_all", ascending=False)

    lines = [
        "# 方向性因子系统性挖掘报告（信号设计track）",
        "",
        "## 方法", "",
        f"对M5上的{len(variant_cols)}个因子变体，各自测试两种方向化模式："
        "**动量模式**(因子处于自身历史80%分位以上时，方向=上一根bar涨跌方向延续)、"
        "**反转模式**(因子从自身历史20%分位以下上穿=多头，从80%分位以上下穿=空头，"
        "穿越事件而非静态阈值)。",
        "",
        f"两阶段筛选：阶段1(全样本阈值，粗筛)保留触发次数>={MIN_ACTIVATIONS}的组合；"
        f"阶段2对|Sharpe|最高的{STAGE2_TOP_N}个做5折walk-forward(阈值只在训练折定，"
        "冻结后用到测试折)，同时拆分多空分别验证——步骤2发现多空严重不对称，这里"
        "把这个检验制度化，不再是筛完就默认多空都行。",
        "",
        f"持有期占位值: {HOLDING_BARS}根M5(~1小时)，真实值待步骤3确定。",
        "",
        f"## 阶段2 walk-forward结果（{len(top)}个候选）", "",
        "| 因子 | 分类 | 模式 | OOS Sharpe(多空合计) | OOS Sharpe(多) | OOS Sharpe(空) | "
        "多空次数 | 折数为正 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in stage2_df.sort_values("wf_oos_sharpe_all", ascending=False).iterrows():
        lines.append(
            f"| {r['variant']} | {r['category']} | {r['mode']} | {r['wf_oos_sharpe_all']:+.3f} | "
            f"{r['wf_oos_sharpe_long']:+.3f} | {r['wf_oos_sharpe_short']:+.3f} | "
            f"{r['wf_n_long']}/{r['wf_n_short']} | {r['wf_n_folds_positive']}/5 |"
        )

    lines += ["", f"## 最终通过筛选的候选（OOS Sharpe>0 且 >=3/5折为正 且 多空触发都>=30次）："
               f"{len(validated)}个", ""]
    if len(validated):
        lines += ["| 因子 | 分类 | 模式 | OOS Sharpe | OOS Sharpe(多/空) |",
                   "|---|---|---|---|---|"]
        for _, r in validated.iterrows():
            lines.append(f"| {r['variant']} | {r['category']} | {r['mode']} | "
                          f"{r['wf_oos_sharpe_all']:+.3f} | "
                          f"{r['wf_oos_sharpe_long']:+.3f}/{r['wf_oos_sharpe_short']:+.3f} |")
    else:
        lines.append("无（说明多空都稳健的方向性因子，在这批候选里没有找到）")
    lines.append("")

    lines += [
        "## 结论与下一步", "",
        f"- 完整阶段1扫描({stage1_path})和阶段2明细({stage2_path})已保存。",
        "- 这一步只测了单因子，没有测试因子组合（AND）；步骤2的经验是AND会大幅降低"
        "触发频率，如果这里找到的最强单因子触发率不够高，可能需要OR逻辑或加权打分"
        "而不是继续AND。",
        "- 仍然是占位持有期，步骤3定真实持有期/止损后需要重新验证这里的候选。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02h_directional_factor_mining_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[5/5] Wrote {report_path}")


if __name__ == "__main__":
    main()

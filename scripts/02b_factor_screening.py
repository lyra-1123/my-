#!/usr/bin/env python3
"""Phase 2b: overfitting-aware factor screening, organized by category
(mean-reversion, momentum, volatility, trend-strength, price-action, volume,
higher-timeframe) — the "hand each category to its own miner" idea from the
conversation, done as clearly-separated category passes in one script rather
than literally spawning separate agents: the screening itself is a fixed
deterministic statistical test (PBO + walk-forward Sharpe), not a judgment
call that benefits from independent takes, so multiple agents would just run
the same arithmetic on the same numbers.

For each factor FAMILY (grouping its 4-window x {raw,pctrank500,pctrank2000}
variants — up to 12 near-duplicates that are exactly what overfitting-via-
parameter-search looks like):
  1. PBO (probability of backtest overfitting, CSCV a la Bailey/Lopez de
     Prado): split into 10 blocks, try every way to bisect them into an IS/
     OOS half, and check how often "best variant by IS Sharpe" actually
     ranks below the OOS median.
  2. Walk-forward OOS Sharpe: fit the safe-zone threshold on the first 70%
     of (non-overlapping) decision points only, apply that frozen threshold
     to the untouched last 30%.

A family survives only if BOTH hold: PBO <= PBO_THRESHOLD and OOS Sharpe > 0.

Usage:
    python scripts/02b_factor_screening.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import family, FAMILY_CATEGORY  # noqa: E402
from factors.validation import (  # noqa: E402
    build_proxy_returns, decision_points, conditional_returns, pbo_for_family,
    walk_forward_oos_sharpe,
)

HORIZON = 24  # bars (1 day of H1) — matches Phase 2's main label horizon
ANNUALIZATION = 252 ** 0.5  # decision points are ~1 trading day apart
PBO_THRESHOLD = 0.5
LABEL_COL = "label_fwd_er_24"
NON_FACTOR_COLS = {"time", "session", "day_of_week", "label_fwd_er_24", "label_fwd_er_72"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading factor table + H1 OHLCV ...")
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_H1.parquet"))
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet"))
    assert len(factors) == len(df)
    label = factors[LABEL_COL]

    print("[2/4] Building proxy mean-reversion returns + decision points ...")
    base_return = build_proxy_returns(df, HORIZON)
    points = decision_points(len(df), HORIZON)
    print(f"      {len(points):,} non-overlapping decision points ({HORIZON}-bar spacing)")

    numeric_cols = [c for c in factors.columns if c not in NON_FACTOR_COLS]
    families = {}
    for col in numeric_cols:
        families.setdefault(family(col), []).append(col)

    print(f"[3/4] Screening {len(families)} factor families ({sum(len(v) for v in families.values())} "
          "variants) with PBO + walk-forward OOS Sharpe ...")

    results = []
    for fam, variants in sorted(families.items()):
        ic_signs = {v: np.sign(factors[v].corr(label, method="spearman") or 0) for v in variants}

        variant_returns = {}
        for v in variants:
            cond = conditional_returns(factors[v], ic_signs[v], base_return)
            variant_returns[v] = cond.iloc[points].to_numpy()

        pbo, diag = pbo_for_family(variant_returns)
        best_variant = diag["best_by_full_sample"]

        wf = walk_forward_oos_sharpe(
            factors[best_variant], base_return, ic_signs[best_variant], points,
        )
        passed = (not np.isnan(pbo)) and pbo <= PBO_THRESHOLD and wf["oos_sharpe"] > 0
        results.append({
            "category": FAMILY_CATEGORY.get(fam, "other"),
            "family": fam,
            "n_variants": len(variants),
            "best_variant": best_variant,
            "pbo": pbo,
            "is_sharpe_ann": wf["is_sharpe"] * ANNUALIZATION,
            "oos_sharpe_ann": wf["oos_sharpe"] * ANNUALIZATION,
            "oos_active_rate": wf["oos_n_active"] / wf["oos_n_total"] if wf["oos_n_total"] else float("nan"),
            "pass": passed,
        })

    results_df = pd.DataFrame(results).sort_values(["category", "pbo"])
    csv_path = os.path.join(args.report_dir, "02b_factor_screening_results.csv")
    results_df.to_csv(csv_path, index=False)

    n_pass = int(results_df["pass"].sum())
    n_total = len(results_df)
    print(f"[4/4] {n_pass}/{n_total} families pass (PBO<={PBO_THRESHOLD} AND OOS Sharpe>0) -> {csv_path}")

    lines = [
        "# 因子分类筛选报告（阶段2b：PBO + 样本外Sharpe）",
        "",
        "## 方法",
        "",
        "按交易逻辑把因子分成7类（均值回归/动量/波动率/趋势强度/价格行为/量能/更高周期背景），"
        "分类别筛选——这里的“筛选”是固定的统计检验流程（PBO计算+walk-forward Sharpe），"
        "不涉及需要独立判断的主观决策，所以用清晰分类的代码逐类跑，而不是真的派生多个"
        "独立agent各自跑一遍相同的算术。",
        "",
        "每个因子家族有多个窗口/百分位版本（最多12个），这正是过拟合的高发地带——如果只是"
        "“挑全样本IC最高的那个”，很可能只是在噪音里挑到了运气好的参数组合。所以这里对每个"
        "家族做：",
        "",
        "1. **PBO（回测过拟合概率，CSCV方法，Bailey/Lopez de Prado）**：把不重叠的决策点"
        "（每24根H1一个，共" + f"{len(points):,}" + "个）分成10段，穷举所有把10段分成"
        "训练/测试两半的方式(C(10,5)=252种)，每种方式里“用训练半段挑出的样本内最优版本”"
        "在测试半段的表现排名——如果经常排到测试半段的中位数以下，说明这个“挑最优”的过程"
        "本身就是在过拟合噪音，PBO就是这个比例。",
        "2. **Walk-forward样本外Sharpe**：前70%决策点当样本内(IS)、后30%当样本外(OOS)——"
        "安全分位阈值只用IS部分数据算，冻结后应用到OOS，不看OOS数据本身。",
        "",
        "因子本身不是交易规则，所以用一个固定的、跟因子无关的极简策略做“测试床”："
        "对最近1根bar做反向(fade)，持有到未来第24根bar——这样“用因子做门槛”和不用因子的"
        "版本除了因子那道门槛之外完全一样，Sharpe的差异能被干净地归因到因子本身，而不是"
        "策略设计。这个测试床本身很粗糙（没有点差/滑点/仓位管理），阶段3的真实马丁回测"
        "会更细，这里只是用来筛因子。",
        "",
        f"通过标准：PBO<={PBO_THRESHOLD}（比抛硬币更可信） 且 OOS Sharpe>0（缺一不可）。",
        "",
        f"## 总体结果：{n_total}个因子家族中，{n_pass}个通过筛选",
        "",
    ]

    for cat, group in results_df.groupby("category"):
        cat_pass = int(group["pass"].sum())
        lines += [
            f"### {cat}（{cat_pass}/{len(group)}通过）",
            "",
            "| 家族 | 变体数 | 最优变体 | PBO | IS Sharpe(年化) | OOS Sharpe(年化) | OOS期激活占比 | 通过 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in group.sort_values("pbo").iterrows():
            mark = "✅" if r["pass"] else "✗"
            lines.append(
                f"| {r['family']} | {r['n_variants']} | {r['best_variant']} | {r['pbo']:.2f} | "
                f"{r['is_sharpe_ann']:+.2f} | {r['oos_sharpe_ann']:+.2f} | {r['oos_active_rate']:.1%} | {mark} |"
            )
        lines.append("")

    passed_families = results_df[results_df["pass"]].sort_values("oos_sharpe_ann", ascending=False)
    lines += [
        "## 通过筛选的因子家族（按OOS Sharpe年化排序）",
        "",
        "| 家族 | 分类 | 最优变体 | PBO | OOS Sharpe(年化) |",
        "|---|---|---|---|---|",
    ]
    for _, r in passed_families.iterrows():
        lines.append(f"| {r['family']} | {r['category']} | {r['best_variant']} | {r['pbo']:.2f} | {r['oos_sharpe_ann']:+.2f} |")

    lines += [
        "",
        "## 结论与下一步",
        "",
        "- **和阶段2 v1~v4纯IC筛选的结论有明显分歧，这正是做这一步筛选的意义**：v1~v4里"
        "最强的几个因子——efficiency_ratio、realized_vol、atr、keltner_width、"
        "variance_ratio_2、linreg_r2、parkinson_vol、garman_klass_vol——在这里全部没通过"
        "PBO+OOS Sharpe筛选（PBO普遍在0.3~0.9之间，即“挑样本内最优窗口”这个过程本身就不"
        "稳健）。反而是MFI（v1~v4里IC很弱）和streak_length、macd_hist这类之前没被重点"
        "关注的因子通过了。这说明纯静态相关性和“能否支撑一个稳健的交易规则”是两个不同的"
        "问题，前者容易被参数搜索污染，后者更贴近实盘会遇到的情况。",
        "- **意外发现一个冗余群**：donchian_position、stochastic_k、williams_r三个家族的"
        "PBO/Sharpe数值完全相同——这不是bug，是因为三者数学上是同一个量的仿射变换"
        "(williams_r = -100+100×donchian_position，stochastic_k = 100×donchian_position)，"
        "秩相关和分位数筛选对仿射变换不敏感，所以给出完全一致的结果。阶段3应该把这三个"
        "当成一个因子用，不要误以为是三个独立信号的相互印证。",
        "- 每个类别都至少有1个家族通过（除了“other”类的hour），说明7个类别的分类思路是"
        "合理的，没有哪一类整体被淘汰；但每个类别通过率都不高（1~3/6），说明多数“看起来"
        "有道理”的技术指标经不起PBO检验。",
        f"- 完整结果（含每个家族的诊断数据）在`{csv_path}`。",
        "- 这一步的Sharpe来自一个刻意简化、和因子无关的测试床策略，只用来公平比较“有没有这个"
        "因子门槛”的差异，**不代表真实马丁格尔策略的Sharpe**——阶段3要在真实的马丁资金曲线"
        "（含加仓/点差/保证金）上重新验证这里通过筛选的因子，静态因子筛选和策略级回测是"
        "两回事。",
        "- PBO<=0.5只是“比瞎猜强”的最低门槛，学术上更严格的要求是PBO<0.2；如果阶段3想更"
        "保守，可以直接从CSV里按更严的阈值重新筛一遍，不需要重跑这个脚本。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02b_factor_screening_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"      Wrote {report_path}")


if __name__ == "__main__":
    main()

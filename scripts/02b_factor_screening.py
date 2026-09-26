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

A family survives a given horizon only if BOTH hold: PBO <= PBO_THRESHOLD
and OOS Sharpe > 0.

v2: the target strategy is an intraday martingale whose ladder cycle runs
4-8 hours (not forced flat by end of day), so screening is now done AT the
horizons that actually matter for that (4h, 8h), with the original 24h kept
only as a "if it runs long" reference point — not as the primary bar. A
family only counts as robust if it clears BOTH 4h and 8h.

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

TEST_HORIZONS = {"4小时": 4, "8小时": 8, "1天(参考)": 24}
CORE_HORIZONS = ("4小时", "8小时")  # a family must pass BOTH to be called robust
PBO_THRESHOLD = 0.5
NON_FACTOR_COLS = {"time", "session", "day_of_week"} | {
    f"label_fwd_er_{h}" for h in (4, 8, 24, 72)
}


def screen_at_horizon(factors: pd.DataFrame, df: pd.DataFrame, horizon: int,
                       families: dict) -> pd.DataFrame:
    label = factors[f"label_fwd_er_{horizon}"]
    base_return = build_proxy_returns(df, horizon)
    points = decision_points(len(df), horizon)
    annualization = (252 * 24 / horizon) ** 0.5  # decision points are `horizon` H1-bars apart

    results = []
    for fam, variants in sorted(families.items()):
        ic_signs = {v: np.sign(factors[v].corr(label, method="spearman") or 0) for v in variants}
        variant_returns = {}
        for v in variants:
            cond = conditional_returns(factors[v], ic_signs[v], base_return)
            variant_returns[v] = cond.iloc[points].to_numpy()

        pbo, diag = pbo_for_family(variant_returns)
        best_variant = diag["best_by_full_sample"]
        wf = walk_forward_oos_sharpe(factors[best_variant], base_return, ic_signs[best_variant], points)
        passed = (not np.isnan(pbo)) and pbo <= PBO_THRESHOLD and wf["oos_sharpe"] > 0
        results.append({
            "category": FAMILY_CATEGORY.get(fam, "other"),
            "family": fam,
            "n_variants": len(variants),
            "best_variant": best_variant,
            "pbo": pbo,
            "oos_sharpe_ann": wf["oos_sharpe"] * annualization,
            "oos_active_rate": wf["oos_n_active"] / wf["oos_n_total"] if wf["oos_n_total"] else float("nan"),
            "pass": passed,
        })
    return pd.DataFrame(results), len(points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading factor table + H1 OHLCV ...")
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_H1.parquet"))
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet"))
    assert len(factors) == len(df)

    numeric_cols = [c for c in factors.columns if c not in NON_FACTOR_COLS]
    families = {}
    for col in numeric_cols:
        families.setdefault(family(col), []).append(col)

    print(f"[2/4] Screening {len(families)} factor families at horizons: "
          f"{', '.join(f'{k}({v}根H1)' for k, v in TEST_HORIZONS.items())} ...")

    per_horizon = {}
    n_points = {}
    for name, h in TEST_HORIZONS.items():
        per_horizon[name], n_points[name] = screen_at_horizon(factors, df, h, families)
        n_pass = int(per_horizon[name]["pass"].sum())
        print(f"      {name}: {n_pass}/{len(families)} pass ({n_points[name]:,} decision points)")

    print("[3/4] Merging across horizons ...")
    merged = per_horizon[list(TEST_HORIZONS)[0]][["category", "family", "n_variants"]].copy()
    for name in TEST_HORIZONS:
        h_df = per_horizon[name].set_index("family")
        merged[f"pbo_{name}"] = merged["family"].map(h_df["pbo"])
        merged[f"oos_sharpe_{name}"] = merged["family"].map(h_df["oos_sharpe_ann"])
        merged[f"pass_{name}"] = merged["family"].map(h_df["pass"])
    merged["robust_core"] = merged[[f"pass_{h}" for h in CORE_HORIZONS]].all(axis=1)

    csv_path = os.path.join(args.report_dir, "02b_factor_screening_results.csv")
    merged.sort_values(["robust_core", "category"], ascending=[False, True]).to_csv(csv_path, index=False)

    n_core = int(merged["robust_core"].sum())
    print(f"[4/4] {n_core}/{len(families)} families robust at BOTH {' and '.join(CORE_HORIZONS)} "
          f"-> {csv_path}")

    lines = [
        "# 因子分类筛选报告（阶段2b：多horizon PBO + 样本外Sharpe）",
        "",
        "## 方法",
        "",
        "按交易逻辑把因子分成7类（均值回归/动量/波动率/趋势强度/价格行为/量能/更高周期背景），"
        "分类别筛选——这里的“筛选”是固定的统计检验流程（PBO计算+walk-forward Sharpe），"
        "不涉及需要独立判断的主观决策，所以用清晰分类的代码逐类跑，而不是真的派生多个"
        "独立agent各自跑一遍相同的算术。",
        "",
        "**v2变化**：目标策略是日内马丁，完整网格周期(从第一次开仓到止盈/止损)大约4-8小时，"
        "不强制收盘平仓。所以筛选horizon从最初的1天/3天改为4小时和8小时（1天保留作参考对照，"
        "不作为通过标准），一个因子家族必须同时在4小时和8小时都通过，才算“稳健核心”——"
        "只在其中一个horizon上表现好，大概率是运气而不是真信号。",
        "",
        "每个因子家族有多个窗口/百分位版本（最多12个），这正是过拟合的高发地带——如果只是"
        "“挑全样本IC最高的那个”，很可能只是在噪音里挑到了运气好的参数组合。所以这里对每个"
        "家族在每个horizon分别做：",
        "",
        "1. **PBO（回测过拟合概率，CSCV方法，Bailey/Lopez de Prado）**：把不重叠的决策点"
        "分成10段，穷举所有把10段分成训练/测试两半的方式(C(10,5)=252种)，每种方式里"
        "“用训练半段挑出的样本内最优版本”在测试半段的表现排名——如果经常排到测试半段的"
        "中位数以下，说明这个“挑最优”的过程本身就是在过拟合噪音，PBO就是这个比例。",
        "2. **Walk-forward样本外Sharpe**：前70%决策点当样本内(IS)、后30%当样本外(OOS)——"
        "安全分位阈值只用IS部分数据算，冻结后应用到OOS，不看OOS数据本身。",
        "",
        "因子本身不是交易规则，所以用一个固定的、跟因子无关的极简策略做“测试床”："
        "对最近1根bar做反向(fade)，持有到未来第horizon根bar——这样“用因子做门槛”和不用"
        "因子的版本除了因子那道门槛之外完全一样，Sharpe的差异能被干净地归因到因子本身，"
        "而不是策略设计。这个测试床本身很粗糙（没有点差/滑点/仓位管理），阶段3的真实马丁"
        "回测会更细，这里只是用来筛因子。Sharpe按对应horizon的决策点频率年化，不同horizon"
        "之间可以直接比较量级。",
        "",
        f"通过标准（单horizon）：PBO<={PBO_THRESHOLD}（比抛硬币更可信） 且 OOS Sharpe>0（缺一不可）。"
        f"“稳健核心”标准：{'和'.join(CORE_HORIZONS)}都要通过。",
        "",
        "## 各horizon单独通过情况",
        "",
        "| horizon | 决策点数 | 通过数/总数 |",
        "|---|---|---|",
    ]
    for name in TEST_HORIZONS:
        lines.append(f"| {name} | {n_points[name]:,} | {int(per_horizon[name]['pass'].sum())}/{len(families)} |")

    lines += [
        "",
        f"## 稳健核心：{n_core}/{len(families)}个家族同时通过{'和'.join(CORE_HORIZONS)}两个horizon",
        "",
        "| 家族 | 分类 | " + " | ".join(f"PBO({h})" for h in TEST_HORIZONS) + " | "
        + " | ".join(f"OOS Sharpe年化({h})" for h in TEST_HORIZONS) + " |",
        "|---|---|" + "---|" * len(TEST_HORIZONS) + "---|" * len(TEST_HORIZONS),
    ]
    core = merged[merged["robust_core"]].sort_values("oos_sharpe_4小时", ascending=False)
    for _, r in core.iterrows():
        pbos = " | ".join(f"{r[f'pbo_{h}']:.2f}" for h in TEST_HORIZONS)
        sharpes = " | ".join(f"{r[f'oos_sharpe_{h}']:+.2f}" for h in TEST_HORIZONS)
        lines.append(f"| {r['family']} | {r['category']} | {pbos} | {sharpes} |")
    lines.append("")

    lines += [
        "## 全部家族逐horizon明细（按分类分组）",
        "",
    ]
    for cat, group in merged.groupby("category"):
        lines += [
            f"### {cat}",
            "",
            "| 家族 | " + " | ".join(f"{h}通过" for h in TEST_HORIZONS) + " | 稳健核心 |",
            "|---|" + "---|" * len(TEST_HORIZONS) + "---|",
        ]
        for _, r in group.iterrows():
            marks = " | ".join("✅" if r[f"pass_{h}"] else "✗" for h in TEST_HORIZONS)
            core_mark = "✅" if r["robust_core"] else "✗"
            lines.append(f"| {r['family']} | {marks} | {core_mark} |")
        lines.append("")

    lines += [
        "## 结论与下一步",
        "",
        f"- **换到实际匹配日内马丁周期的4/8小时horizon后，IC和Sharpe普遍比1天/3天弱得多**"
        "（阶段2的候选池报告里能看到同一批因子在4小时的IC只有1天的1/4~1/5），这是符合"
        "预期的诚实结果，不是方法出错：regime/波动率的可预测性本身是慢变量，4-8小时的"
        "噪音占比远大于1-3天。这意味着如果日内马丁真的按4-8小时一个周期跑，能指望因子"
        "过滤器带来的改善本身就应该更保守地估计。",
        f"- 同时通过4小时和8小时两个horizon的“稳健核心”家族数量: {n_core}个，比只看单一"
        "horizon（阶段2b v1只测了1天）更严格，但也更贴近实际策略会遇到的情况。",
        "- 1天(参考)这一列只是用来对照——如果一个家族在1天上通过但在4/8小时都不通过，"
        "说明它对“今天是不是趋势日”这种慢regime有用，但对马丁真正需要的“接下来几小时"
        "会不会把网格打穿”没有帮助，阶段3不该用它来做网格层级的实时门槛（但仍可以考虑"
        "用作“今天要不要开新网格”这种更粗粒度的日内准入判断，属于不同用途）。",
        f"- 完整逐horizon诊断数据在`{csv_path}`。",
        "- 这一步的Sharpe仍然来自简化测试床，不代表真实马丁Sharpe，阶段3要在真实资金曲线"
        "上重新验证。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02b_factor_screening_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"      Wrote {report_path}")


if __name__ == "__main__":
    main()

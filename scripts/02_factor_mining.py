#!/usr/bin/env python3
"""Phase 2: factor mining for regime classification (choppy vs trending).

For a martingale strategy the goal is NOT direction prediction — it's telling
apart the choppy/mean-reverting regime a martingale ladder survives in from a
persistent trend that blows it up. We evaluate every candidate factor against
forward Efficiency Ratio (Kaufman) at two horizons: a factor that reliably
predicts LOW forward ER is a candidate "safe to run martingale" filter; one
that predicts HIGH forward ER is an early-warning "stand down" signal.

Usage:
    python scripts/02_factor_mining.py \
        --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import build_factor_table  # noqa: E402
from factors.labels import forward_efficiency_ratio  # noqa: E402

HORIZONS = {"1天(24根H1)": 24, "3天(72根H1)": 72}


def spearman_ic_table(factors: pd.DataFrame, labels: dict) -> pd.DataFrame:
    numeric_cols = [c for c in factors.columns if c not in ("time", "session", "day_of_week")]
    rows = []
    for col in numeric_cols:
        row = {"factor": col}
        for label_name, label_series in labels.items():
            row[label_name] = factors[col].corr(label_series, method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_table(factor: pd.Series, label: pd.Series, q: int = 5) -> pd.DataFrame:
    valid = factor.notna() & label.notna()
    bucket = pd.qcut(factor[valid], q, labels=False, duplicates="drop")
    return label[valid].groupby(bucket).agg(["mean", "count"])


def categorical_table(factor: pd.Series, label: pd.Series) -> pd.Series:
    valid = factor.notna() & label.notna()
    return label[valid].groupby(factor[valid]).mean().sort_values()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading H1 bars ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet"))
    print(f"      {len(df):,} H1 bars")

    print("[2/4] Building factor table ...")
    factors = build_factor_table(df)

    print("[3/4] Building forward-ER labels and computing IC ...")
    labels = {name: forward_efficiency_ratio(df, h) for name, h in HORIZONS.items()}
    ic = spearman_ic_table(factors, labels)
    ic["max_abs_ic"] = ic[list(HORIZONS.keys())].abs().max(axis=1)
    ic = ic.sort_values("max_abs_ic", ascending=False).reset_index(drop=True)

    top_factors = ic["factor"].head(6).tolist()
    main_horizon_name, main_horizon = next(iter(HORIZONS.items()))

    lines = [
        "# 因子挖掘报告（阶段2）",
        "",
        "## 方法",
        "",
        "马丁格尔策略的核心需求不是预测涨跌方向，而是区分“震荡/均值回归”（马丁网格能存活的"
        "regime）和“单边趋势持续”（会把网格打爆的regime）。所以这里不做常规的方向性IC分析，"
        "而是用未来Kaufman效率系数（Efficiency Ratio, ER）作为“危险程度”标签：ER→1代表未来"
        "价格路径高效率地朝一个方向走（趋势持续，危险），ER→0代表未来路径来回震荡但净位移很小"
        "（安全）。所有候选因子只使用截至当前bar的历史数据计算（无未来函数），标签则专门使用"
        "未来窗口数据，仅用于评估、不作为任何模型输入。",
        "",
        f"评估基于H1（{len(df):,}根），标签窗口：" + "、".join(HORIZONS.keys()),
        "",
        "## 因子 vs 未来ER的Spearman秩相关（按|IC|排序，仅数值型因子）",
        "",
        "| 因子 | " + " | ".join(HORIZONS.keys()) + " | max|IC| |",
        "|---|" + "---|" * (len(HORIZONS) + 1),
    ]
    for _, row in ic.iterrows():
        vals = " | ".join(f"{row[h]:+.3f}" for h in HORIZONS.keys())
        lines.append(f"| {row['factor']} | {vals} | {row['max_abs_ic']:.3f} |")

    lines += [
        "",
        f"## TOP因子分层分析（{main_horizon_name}未来ER均值，按因子五分位）",
        "",
    ]
    for f in top_factors:
        qt = quantile_table(factors[f], labels[main_horizon_name])
        lines.append(f"### {f}")
        lines.append("")
        lines.append("| 五分位(0=最低) | 未来ER均值 | 样本数 |")
        lines.append("|---|---|---|")
        for idx, r in qt.iterrows():
            lines.append(f"| {int(idx)} | {r['mean']:.3f} | {int(r['count'])} |")
        lines.append("")

    lines += [f"## 时段(session) / 星期 与 {main_horizon_name}未来ER 的关系（非数值因子，"
               "用分组均值代替相关系数；session划分未经独立UTC核实，结论暂视为初步）", ""]
    for cat_col in ("session", "day_of_week"):
        cat_table = categorical_table(factors[cat_col], labels[main_horizon_name])
        lines.append(f"### {cat_col}")
        lines.append("")
        lines.append(f"| {cat_col} | 未来ER均值 |")
        lines.append("|---|---|")
        for idx, v in cat_table.items():
            lines.append(f"| {idx} | {v:.3f} |")
        lines.append("")

    # pick the best factor, plus the best factor from a *different* indicator
    # family (avoid pairing e.g. bb_width_24 with bb_width_48, which are
    # near-collinear windows of the same indicator and add no independent
    # information).
    def family(name: str) -> str:
        return name.rsplit("_", 1)[0]

    composite_a = ic.iloc[0]["factor"]
    composite_b = next(
        f for f in ic["factor"] if family(f) != family(composite_a)
    )

    def safe_side(factor_name: str) -> str:
        # "safe" = the side of the factor that historically saw LOWER
        # forward ER (choppier, not trending). Sign of the IC tells us
        # which tail that is; don't just assume "low factor = safe".
        return "high" if ic.set_index("factor").loc[factor_name, main_horizon_name] < 0 else "low"

    fa, fb = factors[composite_a], factors[composite_b]
    side_a, side_b = safe_side(composite_a), safe_side(composite_b)
    label_main = labels[main_horizon_name]
    valid = fa.notna() & fb.notna() & label_main.notna()

    def safe_mask(f, side, valid):
        return (f[valid] >= f[valid].quantile(0.8)) if side == "high" else (f[valid] <= f[valid].quantile(0.2))

    mask_a = safe_mask(fa, side_a, valid)
    mask_b = safe_mask(fb, side_b, valid)
    baseline_mean = label_main[valid].mean()
    both_safe_mean = label_main[valid][mask_a & mask_b].mean()
    both_safe_n = int((mask_a & mask_b).sum())
    lines += [
        f"## 组合过滤器验证：{composite_a}（取{side_a}20%分位）与 {composite_b}（取{side_b}20%分位）"
        "同时成立时（选两个不同指标家族里IC最强的因子，避免同族因子共线导致的虚假增益）",
        "",
        f"- 全样本未来ER均值（基准）: {baseline_mean:.3f}",
        f"- 两因子同时处于各自“安全”分位时未来ER均值: {both_safe_mean:.3f}（样本数 {both_safe_n:,}，"
        f"占比 {both_safe_n/valid.sum():.1%}）",
        f"- 相对基准降幅: {(baseline_mean - both_safe_mean) / baseline_mean:+.1%}"
        "（正且明显大于单因子分层降幅，说明组合两个弱因子确实能加强regime区分度，"
        "值得在阶段3回测里作为“允许开新一层马丁”的门槛条件之一；样本占比也要看，"
        "太小的窗口在实盘里可能常年不开单）",
        "",
    ]

    lines += [
        "## 结论与下一步",
        "",
        f"- IC最强的因子: {ic.iloc[0]['factor']}（max|IC|={ic.iloc[0]['max_abs_ic']:.3f}）。"
        "|IC|在0.05以下的因子对regime几乎没有区分力，不建议直接作为过滤器阈值依据。",
        "- 分层表若单调（五分位从0到4未来ER均值递增或递减），说明该因子可以直接拿来设阈值"
        "做regime过滤器；若不单调，只是整体统计上有点相关性，需要在阶段3回测里进一步验证"
        "而非直接信任。",
        "- 这里只用了价格衍生的技术类因子（波动率/趋势强度/超买超卖/时段），没有引入跨市场"
        "或宏观数据（本地目前只有XAUUSD自身行情）；如果后续要加美元指数/美债收益率/VIX等"
        "跨市场因子，需要额外的数据源。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02_factor_mining_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[4/4] Wrote {report_path}")

    out_path = os.path.join(args.clean_dir, "factors_H1.parquet")
    combined = factors.copy()
    for name, s in labels.items():
        combined[f"label_fwd_er_{HORIZONS[name]}"] = s
    combined.to_parquet(out_path, index=False)
    print(f"      Wrote {out_path}")


if __name__ == "__main__":
    main()

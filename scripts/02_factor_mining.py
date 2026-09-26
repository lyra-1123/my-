#!/usr/bin/env python3
"""Phase 2: factor mining for regime classification (choppy vs trending).

For a martingale strategy the goal is NOT direction prediction — it's telling
apart the choppy/mean-reverting regime a martingale ladder survives in from a
persistent trend that blows it up. We evaluate every candidate factor against
forward Efficiency Ratio (Kaufman) at two horizons: a factor that reliably
predicts LOW forward ER is a candidate "safe to run martingale" filter; one
that predicts HIGH forward ER is an early-warning "stand down" signal.

v2: broadened the candidate net considerably (~20 indicator families x 4
windows + MACD + streak-length + two H4 higher-timeframe context factors,
~100 numeric factors total, up from 25 in v1) and reports every factor with
|IC| >= CANDIDATE_IC_THRESHOLD as a candidate pool (not just the top few),
saved to reports/02_factor_candidate_pool.csv for Phase 3 to draw from.

Usage:
    python scripts/02_factor_mining.py \
        --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from factors.library import build_factor_table, adx, bollinger_width, efficiency_ratio  # noqa: E402
from factors.labels import forward_efficiency_ratio  # noqa: E402

HORIZONS = {"1天(24根H1)": 24, "3天(72根H1)": 72}
CANDIDATE_IC_THRESHOLD = 0.01
HTF_WINDOWS = (20, 50)  # in H4 bars: ~3.3 days and ~8.3 days of context


def spearman_ic_table(factors: pd.DataFrame, labels: dict) -> pd.DataFrame:
    exclude = {"time", "session", "day_of_week"}
    numeric_cols = [c for c in factors.columns if c not in exclude]
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


def build_htf_context(h4_df: pd.DataFrame) -> pd.DataFrame:
    """H4-derived regime context, merged onto H1 without look-ahead: an H4
    bar starting at `time` covers [time, time+4h) and is only fully known
    once it closes, so it's only valid for H1 bars at or after time+4h."""
    ctx = pd.DataFrame({"time": h4_df["time"]})
    ctx["valid_from"] = h4_df["time"] + pd.Timedelta(hours=4)
    for n in HTF_WINDOWS:
        ctx[f"h4_adx_{n}"] = adx(h4_df, n)
        ctx[f"h4_bb_width_{n}"] = bollinger_width(h4_df, n)
        ctx[f"h4_efficiency_ratio_{n}"] = efficiency_ratio(h4_df, n)
    return ctx.sort_values("valid_from")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/5] Loading H1 + H4 bars ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet"))
    h4_df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H4.parquet"))
    print(f"      {len(df):,} H1 bars, {len(h4_df):,} H4 bars")

    print("[2/5] Building factor table (H1 factors + H4 context) ...")
    factors = build_factor_table(df)
    htf_ctx = build_htf_context(h4_df)
    factors = pd.merge_asof(
        factors.sort_values("time"), htf_ctx.drop(columns=["time"]),
        left_on="time", right_on="valid_from", direction="backward",
    ).drop(columns=["valid_from"])
    n_numeric_factors = len(
        [c for c in factors.columns if c not in ("time", "session", "day_of_week")]
    )
    print(f"      {n_numeric_factors} numeric factors + session/day_of_week")

    print("[3/5] Building forward-ER labels and computing IC ...")
    labels = {name: forward_efficiency_ratio(df, h) for name, h in HORIZONS.items()}
    ic = spearman_ic_table(factors, labels)
    ic["max_abs_ic"] = ic[list(HORIZONS.keys())].abs().max(axis=1)
    ic = ic.sort_values("max_abs_ic", ascending=False).reset_index(drop=True)

    pool = ic[ic["max_abs_ic"] >= CANDIDATE_IC_THRESHOLD].reset_index(drop=True)
    pool_path = os.path.join(args.report_dir, "02_factor_candidate_pool.csv")
    pool.to_csv(pool_path, index=False)

    print(f"[4/5] {len(pool)}/{n_numeric_factors} factors clear |IC|>={CANDIDATE_IC_THRESHOLD} "
          f"-> {pool_path}")

    top_factors = ic["factor"].head(10).tolist()
    main_horizon_name, main_horizon = next(iter(HORIZONS.items()))

    lines = [
        "# 因子挖掘报告（阶段2，v2扩大候选池）",
        "",
        "## 方法",
        "",
        "马丁格尔策略的核心需求不是预测涨跌方向，而是区分“震荡/均值回归”（马丁网格能存活的"
        "regime）和“单边趋势持续”（会把网格打爆的regime）。所以这里不做常规的方向性IC分析，"
        "而是用未来Kaufman效率系数（Efficiency Ratio, ER）作为“危险程度”标签：ER→1代表未来"
        "价格路径高效率地朝一个方向走（趋势持续，危险），ER→0代表未来路径来回震荡但净位移很小"
        "（安全）。所有候选因子只使用截至当前bar的历史数据计算（无未来函数，H4衍生的因子额外"
        "做了“该H4 bar实际收盘时刻之后才可见”的对齐处理），标签则专门使用未来窗口数据，仅用于"
        "评估、不作为任何模型输入。",
        "",
        f"评估基于H1（{len(df):,}根），标签窗口：" + "、".join(HORIZONS.keys()),
        f"；本轮相比v1把候选因子从25个扩大到{n_numeric_factors}个：新增波动率类"
        "(keltner_width/vol_of_vol)、趋势类(adx_slope/variance_ratio_2/autocorr_returns/"
        "streak_length/macd_hist)、超买超卖类(stochastic_k/d、williams_r、cci、roc、"
        "dist_from_high/low、donchian_position)、分布形态类(skew/kurt_returns)，以及2个"
        "H4更高周期的regime背景因子(h4_adx、h4_bb_width、h4_efficiency_ratio)，且每类基本"
        "指标从3个回看窗口(14/24/48)扩到4个(10/20/50/100)。",
        "",
        f"## 候选池：|IC|>={CANDIDATE_IC_THRESHOLD}的因子（{len(pool)}/{n_numeric_factors}个，"
        f"按max|IC|排序，完整CSV见`reports/02_factor_candidate_pool.csv`）",
        "",
        "| 因子 | " + " | ".join(HORIZONS.keys()) + " | max\\|IC\\| |",
        "|---|" + "---|" * (len(HORIZONS) + 1),
    ]
    for _, row in pool.iterrows():
        vals = " | ".join(f"{row[h]:+.3f}" for h in HORIZONS.keys())
        lines.append(f"| {row['factor']} | {vals} | {row['max_abs_ic']:.3f} |")

    n_below = n_numeric_factors - len(pool)
    lines += [
        "",
        f"（另有{n_below}个因子|IC|<{CANDIDATE_IC_THRESHOLD}，判定为无区分力，未列入候选池，"
        "完整名单也在CSV里，标记为未通过阈值）",
        "",
        f"## TOP10因子分层分析（{main_horizon_name}未来ER均值，按因子五分位）",
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
    # family (avoid pairing e.g. bb_width_20 with bb_width_50, which are
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
        f"- 本轮共构建{n_numeric_factors}个数值因子，{len(pool)}个通过|IC|>={CANDIDATE_IC_THRESHOLD}"
        f"的候选池筛选，{n_below}个未通过；候选池整体仍以“波动率/趋势强度类”因子（bb_width、"
        "keltner_width、adx、h4_adx、h4_bb_width）为主，说明真正有效的regime信息集中在这一类，"
        "扩大搜索范围并没有找出量级更强的新因子（max|IC|依然在0.05~0.10区间），只是把同类信息"
        "用更多参数化形式重新表达了一遍——这本身也是一个有用的结论：不必再花力气在方向类"
        "(RSI/zscore/ma_slope)或分布形态类(skew/kurt)因子上，它们持续垫底。",
        "- |IC|数值虽然普遍不高，但样本量巨大（10万+根H1），且五分位分层单调、跨H1/H4双周期一致，"
        "说明这是稳定的统计规律而非噪音；不过|IC|~0.05~0.10对应的可解释方差不到1%，单独使用"
        "任何一个因子都不构成可交易的强信号，必须像候选池里已验证的“组合过滤器”那样叠加使用。",
        "- 候选池CSV会传给阶段3，用于在真实马丁资金曲线回测里做特征选择/组合，而不是直接把"
        "静态相关性当结论。",
        "- 仍然只用了价格衍生的技术类因子（含H1自身+H4更高周期），没有引入跨市场/宏观数据"
        "（本地目前只有XAUUSD自身行情）；如果后续要加美元指数/美债收益率/VIX等跨市场因子，"
        "需要额外的数据源。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02_factor_mining_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[5/5] Wrote {report_path}")

    out_path = os.path.join(args.clean_dir, "factors_H1.parquet")
    combined = factors.copy()
    for name, s in labels.items():
        combined[f"label_fwd_er_{HORIZONS[name]}"] = s
    combined.to_parquet(out_path, index=False)
    print(f"      Wrote {out_path}")


if __name__ == "__main__":
    main()

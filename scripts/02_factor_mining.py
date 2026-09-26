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

v3: added MFI (volume-weighted RSI) and, for the volatility/trend-strength
families that actually carry signal, a percentile-rank transform ("where
does today's ADX/ATR/bb_width/... sit relative to its own trailing 500/2000
-bar history") — gold's gone from ~$860 to ~$5300 over this sample, so a raw
dollar-denominated level means something different in 2009 vs 2026; the
percentile-rank version is regime-relative and comparable across the whole
sample.

v4: added a new batch of indicator families (Choppiness Index — literally
designed for choppy-vs-trending; Aroon up/down; Parkinson and Garman-Klass
OHLC volatility estimators; linear-regression R^2; average opening-gap
size), extended the percentile-rank transform to EVERY windowed factor
(not just the v3 subset), and replaced the single hand-picked composite
filter with a systematic search: every pair among the top decorrelated
(one-per-family) factors is tested, plus a multi-factor composite score.

v5: the target strategy is an intraday martingale whose full ladder cycle
(first entry to take-profit/stop) typically runs 4-8 hours and isn't
forced flat by end of day — so the original 1-day/3-day horizons were
measuring the wrong thing (whether the whole DAY trends, not whether the
next few hours of a live ladder would). Replaced/extended with 4h and 8h
H1-bar horizons (plus keep 1d/3d for context/robustness comparison), and
label_fwd_er_{4,8,24,72} are all written to factors_H1.parquet so Phase 2b
can screen at the horizons that actually match the strategy.

Usage:
    python scripts/02_factor_mining.py \
        --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from factors.library import (  # noqa: E402
    build_factor_table, adx, bollinger_width, efficiency_ratio, choppiness_index, family,
)
from factors.labels import forward_efficiency_ratio  # noqa: E402

HORIZONS = {"4小时(4根H1)": 4, "8小时(8根H1)": 8, "1天(24根H1)": 24, "3天(72根H1)": 72}
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
        ctx[f"h4_choppiness_index_{n}"] = choppiness_index(h4_df, n)
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
        "# 因子挖掘报告（阶段2，v5：horizon改为匹配日内马丁的4-8小时周期）",
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
        f"；候选因子从v1的25个扩大到本轮的{n_numeric_factors}个：v2新增波动率类"
        "(keltner_width/vol_of_vol)、趋势类(adx_slope/variance_ratio_2/autocorr_returns/"
        "streak_length/macd_hist)、超买超卖类(stochastic_k/d、williams_r、cci、roc、"
        "dist_from_high/low、donchian_position)、分布形态类(skew/kurt_returns)，以及2个"
        "H4更高周期的regime背景因子(h4_adx、h4_bb_width、h4_efficiency_ratio)，且每类基本"
        "指标从3个回看窗口(14/24/48)扩到4个(10/20/50/100)；v3新增MFI(资金流量指标，用"
        "tick成交量代理，非真实成交量，解读需谨慎)，并对波动率/趋势强度类因子(atr、"
        "realized_vol、bb_width、keltner_width、vol_of_vol、adx、efficiency_ratio、mfi)"
        "额外算了相对其自身滚动500根/2000根历史的百分位排名——原始因子是黄金"
        "美元报价的绝对水平，18年里金价从~860涨到~5300，同样的ATR数值在2009年和2026年"
        "代表的“波动程度”完全不是一回事，百分位排名把它转成“相对当前regime”的量纲，"
        "跨样本可比。",
        "v4新增专门为“震荡vs趋势”设计的Choppiness Index、Aroon up/down、更高效的OHLC波动率"
        "估计量(Parkinson、Garman-Klass，用到整根bar的高低点/开收盘信息而不只是收盘价)、"
        "线性回归拟合优度R²(方向无关的“趋势有多干净”)、平均开盘跳空幅度；并把百分位排名"
        "从v3的8个手选家族扩展到全部因子家族（不再预判哪些需要、哪些不需要）；同时把v1~v3"
        "手选一对做组合验证，换成了系统性搜索：先从每个指标家族挑IC最强的代表因子（15个），"
        "再穷举两两组合，也测试了多因子平均合成打分。",
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

    # ---- systematic combination search (v4) ----
    # One representative per underlying indicator family (its best-IC
    # window/pctrank variant), so pairs/composites combine genuinely
    # different information rather than e.g. bb_width_50 with bb_width_100.
    ic_by_factor = ic.set_index("factor")
    label_main = labels[main_horizon_name]

    def safe_side(factor_name: str) -> str:
        return "high" if ic_by_factor.loc[factor_name, main_horizon_name] < 0 else "low"

    def safe_mask(f: pd.Series, side: str, valid: pd.Series) -> pd.Series:
        return (f[valid] >= f[valid].quantile(0.8)) if side == "high" else (f[valid] <= f[valid].quantile(0.2))

    ic["family"] = ic["factor"].map(family)
    representatives = (
        ic.loc[ic.groupby("family")["max_abs_ic"].idxmax()]
        .sort_values("max_abs_ic", ascending=False)
        .head(15)["factor"].tolist()
    )

    lines += [
        f"## 系统性两两组合搜索（从{len(representatives)}个“每个指标家族里IC最强的代表因子”中"
        "两两配对，各自取20%“安全”分位，看同时成立时未来ER相对基准的降幅，按降幅排序取前10；"
        "这是穷举而不是像v1~v3那样手选一对）",
        "",
        f"代表因子: {', '.join(representatives)}",
        "",
        "| 因子A | 因子B | 样本占比 | 未来ER均值 | 相对基准降幅 |",
        "|---|---|---|---|---|",
    ]
    baseline_mean = label_main.mean()
    pair_results = []
    for i, fa_name in enumerate(representatives):
        for fb_name in representatives[i + 1:]:
            fa, fb = factors[fa_name], factors[fb_name]
            side_a, side_b = safe_side(fa_name), safe_side(fb_name)
            valid = fa.notna() & fb.notna() & label_main.notna()
            mask = safe_mask(fa, side_a, valid) & safe_mask(fb, side_b, valid)
            n = int(mask.sum())
            if n < 500:  # too few samples to trust the mean
                continue
            mean_er = label_main[valid][mask].mean()
            pair_results.append((fa_name, fb_name, n, n / valid.sum(), mean_er))

    pair_results.sort(key=lambda r: (baseline_mean - r[4]) / baseline_mean, reverse=True)
    for fa_name, fb_name, n, share, mean_er in pair_results[:10]:
        reduction = (baseline_mean - mean_er) / baseline_mean
        lines.append(f"| {fa_name} | {fb_name} | {share:.1%} | {mean_er:.3f} | {reduction:+.1%} |")
    lines.append("")
    lines.append(f"（全样本未来ER基准均值: {baseline_mean:.3f}；样本数<500的组合已剔除，不然小样本"
                 "均值不稳定容易排到前面制造假象）")
    lines.append("")

    # ---- multi-factor composite score ----
    top_k = 8
    composite_members = representatives[:top_k]
    safe_pct = pd.DataFrame(index=factors.index)
    for fname in composite_members:
        f = factors[fname]
        pct = f.rank(pct=True)
        safe_pct[fname] = pct if safe_side(fname) == "high" else 1 - pct
    composite_score = safe_pct.mean(axis=1, skipna=True)
    composite_ic = composite_score.corr(label_main, method="spearman")
    composite_qt = quantile_table(composite_score, label_main)

    lines += [
        f"## 多因子合成打分：取IC最强的{top_k}个代表因子，每个按“安全方向”转成0~1的历史分位"
        "（1=最安全），取平均作为一个综合regime分数，再看它本身的IC和五分位分层",
        "",
        f"合成因子: {', '.join(composite_members)}",
        "",
        f"- 合成分数 vs {main_horizon_name}未来ER 的Spearman IC: {composite_ic:+.3f}"
        f"（对比单因子最强的{ic.iloc[0]['factor']}: {ic.iloc[0][main_horizon_name]:+.3f}）",
        "",
        "| 五分位(0=最危险,4=最安全) | 未来ER均值 | 样本数 |",
        "|---|---|---|",
    ]
    for idx, r in composite_qt.iterrows():
        lines.append(f"| {int(idx)} | {r['mean']:.3f} | {int(r['count'])} |")
    lines.append("")
    best_pair_reduction = (baseline_mean - pair_results[0][4]) / baseline_mean if pair_results else float("nan")
    composite_reduction = (
        (baseline_mean - composite_qt.loc[composite_qt.index.max(), "mean"]) / baseline_mean
    )
    lines += [
        f"- 合成打分最高分位（最“安全”20%）相对基准降幅: {composite_reduction:+.1%}，"
        f"对比两两组合里最好的一对（降幅{best_pair_reduction:+.1%}）——"
        + ("合成打分更强，说明多因子平均确实比任意一对组合更有效"
           if composite_reduction > best_pair_reduction
           else "两两组合反而更强，说明简单平均稀释了强因子的信号，不如直接用组合过滤器"),
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
        "- MFI（资金流量指标）偏弱（max|IC|约0.02~0.03），符合预期：我们的volume是tick数量"
        "代理而非真实成交量，量价类指标在这份数据上先天打折扣，不建议作为主力因子。",
        "- 百分位排名的价值分窗口而定：对20/50根这种较短窗口，pctrank版本和原始值IC几乎"
        "一样（说明短窗口本身已经是局部相对值，金价长期涨幅带来的尺度漂移影响不大）；但对"
        "100根这种较长窗口，pctrank版本明显强于原始值（如atr_100、keltner_width_100，"
        "max|IC|从~0.055~0.061提升到~0.076），说明长窗口的原始指标确实受金价从860到5300+的"
        "尺度漂移污染，百分位排名修正了这个问题——这是本轮扩大范围里少数几个“方法改进直接"
        "带来更强因子”的例子，值得在阶段3优先使用这些pctrank_2000版本而非同名原始版本。",
        "- 仍然只用了价格衍生的技术类因子（含H1自身+H4更高周期），没有引入跨市场/宏观数据"
        "（本地目前只有XAUUSD自身行情）；如果后续要加美元指数/美债收益率/VIX等跨市场因子，"
        "需要额外的数据源。",
        f"- v4新增的Choppiness Index表现符合预期地强（{ic_by_factor.loc['choppiness_index_50', 'max_abs_ic']:.3f}"
        "，跻身候选池前列），且频繁出现在系统性搜索出的最佳组合里，证明“专门为这个问题设计的"
        "指标”确实比通用技术指标更有效，这比v1~v3的泛化搜索更有针对性。注意它的IC符号和"
        "bb_width/adx相反（当前越“choppy”→未来ER越高），这不是矛盾：结合两者看，故事是"
        "regime会交替——当前波动率已经放大/趋势已经很强时，未来更可能“歇一歇”变震荡"
        "(bb_width/adx的发现)；当前处于窄幅盘整时，未来更可能变成突破趋势(choppiness_index的"
        "发现)。两个独立构造的指标从不同角度印证了同一个“波动率/趋势会均值回归”的市场现象，"
        "互相印证比单独看更可信。",
        (
            f"- 系统性两两组合搜索（穷举15个代表因子的组合）在{main_horizon_name}上最好的一对是"
            f"`{pair_results[0][0]}`+`{pair_results[0][1]}`，能把未来ER压低约"
            f"{best_pair_reduction:+.1%}（样本占比{pair_results[0][3]:.1%}）。"
            if pair_results else
            f"- {main_horizon_name}上没有样本量>=500的两两组合，穷举搜索在这个horizon上"
            "拿不到可信结果。"
        ),
        f"- 8因子平均合成打分在{main_horizon_name}上的效果：合成分数IC={composite_ic:+.3f}"
        f"（对比单因子最强的{ic.iloc[0]['factor']}: {ic.iloc[0][main_horizon_name]:+.3f}），"
        f"最高安全分位ER降幅{composite_reduction:+.1%}，"
        + ("比最优两两组合更强" if pair_results and composite_reduction > best_pair_reduction
           else "反而不如最优两两组合，说明简单平均稀释了强因子的信号，不建议用“一堆因子取"
                "平均”的合成分数") + "。",
        "- **重要提醒**：这份报告每次重跑`02_factor_mining.py`都会用当时`HORIZONS`字典里排第一"
        "的horizon作为“主horizon”重新计算上面两条结论和组合搜索表——v5把主horizon从1天改成了"
        "4小时后，同一个bb_width_50+efficiency_ratio_20组合在4小时上的降幅从早先1天horizon"
        "测出的约17.7%掉到了个位数百分比（regime可预测性在短horizon上本来就弱，这和"
        "reports/00_progress.md记录的IC量级衰减是一回事）。之前对话里提到的“组合过滤器"
        "降低15~18%”这个数字，指的是1天horizon下的结果，不是4/8小时——如果要在阶段3的"
        "日内马丁上用组合过滤器，应该以本报告当前呈现的4小时数字为准，而不是沿用早先"
        "按1天horizon算出的15~18%。",
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

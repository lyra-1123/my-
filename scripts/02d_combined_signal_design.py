#!/usr/bin/env python3
"""Phase 2d: design and validate the COMBINED regime signal from the 4
category-representative factors picked in Phase 2c, before wiring it into
the real martingale engine (Phase 3b).

Earlier lesson (Phase 2 v4): averaging factor scores together diluted the
signal versus a two-factor AND filter. So instead of one continuous
average, this tests explicit VOTING rules: each of the 4 factors casts a
"safe" (1) or "danger" (0) vote (oriented by its own sign, safe side =
top/bottom 20% of its own history — same convention used throughout Phase
2), and a combined signal fires when at least K of 4 votes are "safe", for
every K from 1 (any factor) to 4 (all must agree). Each rule is scored with
the exact same methodology as every individual factor in Phase 2c: 5-fold
walk-forward pooled OOS Sharpe on the fixed fade-and-hold test-bed, at both
4h and 8h, plus the activation rate (a statistically strong rule that's
almost never "safe" is not usable in a live ladder).

Usage:
    python scripts/02d_combined_signal_design.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.validation import (  # noqa: E402
    build_proxy_returns, decision_points, walk_forward_multi_fold, fold_consistency_sharpe,
)

TEST_HORIZONS = {"4小时": 4, "8小时": 8}
CATEGORY_PICKS = {
    "均值回归": "mean_reversion_speed_20",
    "动量": "autocorr_returns_100",
    "波动率": "vol_of_vol_10_pctrank500",
    "价格行为": "dist_from_high_20_pctrank500",
}


def safe_votes(factors: pd.DataFrame, label: pd.Series, variants: dict) -> pd.DataFrame:
    """One 0/1 column per variant: 1 = this factor currently says "safe"
    (in its own historically safe 20% tail, oriented by its full-sample IC
    sign against THIS horizon's label)."""
    votes = pd.DataFrame(index=factors.index)
    for cat, v in variants.items():
        f = factors[v]
        ic = f.corr(label, method="spearman")
        sign = np.sign(ic) if ic and not np.isnan(ic) else 1.0
        if sign < 0:  # high factor value historically preceded LOW forward ER -> high = safe
            votes[cat] = (f >= f.quantile(0.8)).astype(int)
        else:
            votes[cat] = (f <= f.quantile(0.2)).astype(int)
    return votes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/3] Loading factor table + H1 OHLCV ...")
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_H1.parquet"))
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_H1.parquet"))

    rows = []
    vote_frames = {}
    for hname, h in TEST_HORIZONS.items():
        label = factors[f"label_fwd_er_{h}"]
        base_return = build_proxy_returns(df, h)
        points = decision_points(len(df), h)
        ann = (252 * 24 / h) ** 0.5

        votes = safe_votes(factors, label, CATEGORY_PICKS)
        vote_frames[hname] = votes
        n_safe = votes.sum(axis=1)

        # individual factors, for reference in the same table
        for cat, v in CATEGORY_PICKS.items():
            f = factors[v]
            ic = f.corr(label, method="spearman")
            sign = np.sign(ic) if ic and not np.isnan(ic) else 1.0
            wf = walk_forward_multi_fold(f, base_return, sign, points, n_folds=5)
            rows.append({
                "horizon": hname, "rule": f"单因子:{cat}", "k": "-",
                "oos_sharpe_ann": wf["oos_sharpe_pooled"] * ann,
                "activation_rate": (votes[cat] == 1).mean(),
                "folds_positive": f"{wf['n_folds_positive']}/{wf['n_folds_total']}",
            })

        # voting rules k=1..4 (votes already fixed from full-sample
        # thresholds, so this is fold-consistency scoring, not a genuine
        # walk-forward refit — see fold_consistency_sharpe's docstring)
        for k in range(1, 5):
            gate = (n_safe >= k)
            gated_return = base_return.where(gate, 0.0)
            fc = fold_consistency_sharpe(gated_return, points, n_folds=5)
            rows.append({
                "horizon": hname, "rule": f"至少{k}/4票安全", "k": k,
                "oos_sharpe_ann": fc["oos_sharpe_pooled"] * ann,
                "activation_rate": gate.iloc[points].mean(),
                "folds_positive": f"{fc['n_folds_positive']}/{fc['n_folds_total']}",
            })

    results = pd.DataFrame(rows)
    csv_path = os.path.join(args.report_dir, "02d_combined_signal_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"[2/3] Wrote {csv_path}")

    lines = ["# 组合信号设计报告（阶段2d）", "", "## 方法", "",
             "阶段2c选出的4个类别代表因子，各自按自身IC符号定“安全方向”（历史top/bottom 20%"
             "分位=安全，阈值用全样本算，和阶段2c/2b一贯的做法一致），每个因子对每个时刻投1票"
             "（安全=1，危险=0）。测试“至少K/4票安全”这4种投票规则（K=1到4）。",
             "",
             "**打分方式和阶段2c不完全一样，这里说清楚区别**：单因子那几行（表里“单因子:xx”）"
             "延用阶段2c的真正walk-forward——阈值只在训练折里定、冻结后用到测试折，5折都这样"
             "滚动。但投票规则本身没有需要按折重新拟合的参数（票怎么投在算votes时就已经用"
             "全样本定好了），所以对K票规则用的是“5折一致性打分”：把5折的收益分别算Sharpe"
             "看是否稳定，再把5折收益池化算一个总Sharpe——这不是严格意义上的样本外验证，"
             "而是稳健性检验，阶段3接入真实引擎前应该再补一次真正的样本外测试。",
             "另外算了“激活率”（这个规则历史上有多大比例时间判定为安全，激活率太低的规则"
             "统计上再强，实盘也几乎不开单，没有实际意义）。",
             "", "## 结果", "",
             "| horizon | 规则 | OOS Sharpe年化 | 激活率 | 样本折数为正 |",
             "|---|---|---|---|---|"]
    for _, r in results.iterrows():
        lines.append(f"| {r['horizon']} | {r['rule']} | {r['oos_sharpe_ann']:+.3f} | "
                      f"{r['activation_rate']:.1%} | {r['folds_positive']} |")
    lines.append("")

    best_rows = []
    for k in range(1, 5):
        sub = results[results["k"] == k]
        if len(sub) == 2 and (sub["oos_sharpe_ann"] > 0).all():
            best_rows.append(k)
    lines += [
        "## 结论与下一步", "",
        f"- 4h/8h同时OOS Sharpe为正的投票规则: {best_rows if best_rows else '无'}。",
        "- 完整数据（含单因子对照行）在上表和CSV里；选定规则后下一步是把它接入"
        "`src/backtest/martingale.py`的`allow_entry`参数，在真实马丁引擎（含加仓/点差/"
        "保证金）上重新验证，而不是继续用这个简化测试床的数字做最终结论。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02d_combined_signal_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[3/3] Wrote {report_path}")


if __name__ == "__main__":
    main()

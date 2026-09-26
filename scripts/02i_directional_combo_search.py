#!/usr/bin/env python3
"""Signal design: systematic pairwise combination search for directional
factors on M5 — same shape as Phase 2 v4's regime-factor combination
search, now applied to direction.

02h found NO single factor with a usable directional edge. This tests
every pair among family representatives (AND-consensus: both factors must
agree on the same direction, mirroring the signal_long/short template),
because Phase 2's regime work and step 2's autocorr+MFI combo both showed
edge lives in combinations, not single factors.

Explicit reminder driving the selection criteria: mining factors is only
useful if it produces a SIGNAL THAT CAN ACTUALLY BE TRADED — a pair with
a great Sharpe that fires 5 times in 18 years is not a signal, it's a
curiosity. Every candidate here is scored on OOS Sharpe AND annual
activation rate together; the final picks require both.

Usage:
    python scripts/02i_directional_combo_search.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.validation import decision_points  # noqa: E402
from factors.direction import (  # noqa: E402
    momentum_direction, reversion_direction, combine_directions, walk_forward_direction_pair,
)

HOLDING_BARS = 12
MIN_ACTIVATIONS = 200
MIN_ANNUAL_RATE = 20  # a combo firing less than this per year isn't a usable live signal
STAGE2_TOP_N = 30


def sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / x.std(ddof=1)


def build_direction(factor: pd.Series, close: pd.Series, mode: str, quantile: float = 0.8) -> pd.Series:
    if mode == "momentum":
        return momentum_direction(factor, close, factor.quantile(quantile))
    return reversion_direction(factor, factor.quantile(1 - quantile), factor.quantile(quantile))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--factors-file", default="factors_M5.parquet")
    parser.add_argument("--stage1-file", default="02h_directional_stage1_scan.csv")
    parser.add_argument("--out-prefix", default="02i")
    args = parser.parse_args()

    print("[1/6] Loading M5 factor table + stage-1 scan ...")
    factors = pd.read_parquet(os.path.join(args.clean_dir, args.factors_file))
    close = factors["close"]
    stage1 = pd.read_csv(os.path.join(args.report_dir, args.stage1_file))
    years = (factors.shape[0] * 5 / 60 / 24) / 365.25  # M5 bars -> years, rough

    fwd_return = close.pct_change(HOLDING_BARS).shift(-HOLDING_BARS)
    points = decision_points(len(factors), HOLDING_BARS)
    points = points[fwd_return.notna().to_numpy()[points]]

    print("[2/6] Picking one representative variant per family (best |Sharpe| in stage-1 scan) ...")
    reps = stage1.loc[stage1.groupby("family")["sharpe"].apply(lambda s: s.abs().idxmax())]
    reps = reps.sort_values("sharpe", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    print(f"      {len(reps)} family representatives")

    print(f"[3/6] Precomputing direction series for {len(reps)} representatives ...")
    directions = {}
    for _, r in reps.iterrows():
        directions[r["variant"]] = build_direction(factors[r["variant"]], close, r["mode"])

    print(f"[4/6] Stage 1: scanning {len(reps)*(len(reps)-1)//2} pairs ...")
    pair_rows = []
    rep_list = reps.to_dict("records")
    for a, b in combinations(rep_list, 2):
        combined = combine_directions(directions[a["variant"]], directions[b["variant"]])
        active_idx = combined[combined != 0].index.intersection(fwd_return.dropna().index)
        r = (combined * fwd_return).loc[active_idx].to_numpy()
        d = combined.loc[active_idx].to_numpy()
        if len(r) < MIN_ACTIVATIONS or (len(r) / years) < MIN_ANNUAL_RATE:
            continue
        pair_rows.append({
            "variant_a": a["variant"], "family_a": a["family"], "mode_a": a["mode"],
            "variant_b": b["variant"], "family_b": b["family"], "mode_b": b["mode"],
            "n_active": len(r), "annual_rate": len(r) / years,
            "n_long": int((d > 0).sum()), "n_short": int((d < 0).sum()),
            "sharpe": sharpe(r), "sharpe_long": sharpe(r[d > 0]), "sharpe_short": sharpe(r[d < 0]),
        })

    pair_df = pd.DataFrame(pair_rows)
    pair_path = os.path.join(args.report_dir, f"{args.out_prefix}_pair_stage1_scan.csv")
    pair_df.to_csv(pair_path, index=False)
    print(f"      {len(pair_df)} pairs clear n_active>={MIN_ACTIVATIONS} and "
          f">={MIN_ANNUAL_RATE}/year -> {pair_path}")

    print(f"[5/6] Stage 2: walk-forward validating top {STAGE2_TOP_N} by |Sharpe| ...")
    top = pair_df.reindex(pair_df["sharpe"].abs().sort_values(ascending=False).index).head(STAGE2_TOP_N)
    stage2 = []
    for _, row in top.iterrows():
        wf = walk_forward_direction_pair(
            factors[row["variant_a"]], row["mode_a"], factors[row["variant_b"]], row["mode_b"],
            close, fwd_return, points, n_folds=5,
        )
        stage2.append({**row.to_dict(), **{f"wf_{k}": v for k, v in wf.items()}})
    stage2_df = pd.DataFrame(stage2)
    stage2_path = os.path.join(args.report_dir, f"{args.out_prefix}_pair_stage2_walkforward.csv")
    stage2_df.to_csv(stage2_path, index=False)

    validated = stage2_df[
        (stage2_df["wf_oos_sharpe_all"] > 0) & (stage2_df["wf_n_folds_positive"] >= 3)
        & ((stage2_df["wf_n_long"] + stage2_df["wf_n_short"]) / years >= MIN_ANNUAL_RATE)
    ].sort_values("wf_oos_sharpe_all", ascending=False) if len(stage2_df) else stage2_df

    lines = [
        "# 方向性因子两两组合搜索报告", "",
        "## 方法", "",
        f"从02h的399个因子里，每个家族挑|Sharpe|最强的1个代表变体（共{len(reps)}个），"
        f"穷举两两配对（共{len(reps)*(len(reps)-1)//2}对），要求两个因子的方向AND一致"
        "（同时看多才算多头信号，同时看空才算空头信号，不是简单都非零）。",
        "",
        f"**筛选标准明确包含“能不能用”，不只是Sharpe**：触发次数>={MIN_ACTIVATIONS}"
        f"且年均触发>={MIN_ANNUAL_RATE}次才进入候选池——一个Sharpe很高但18年只响几次"
        "的组合不是能交易的信号，不纳入排名。",
        "",
        f"阶段2对Top{STAGE2_TOP_N}做5折walk-forward（两个因子的阈值都只在训练折定、"
        "冻结后用到测试折）。",
        "",
        f"## 阶段1候选池：{len(pair_df)}对满足触发频率要求", "",
    ]
    if len(pair_df):
        top_display = pair_df.reindex(pair_df["sharpe"].abs().sort_values(ascending=False).index).head(15)
        lines += ["| 因子A | 因子B | 年均触发 | Sharpe(合计/多/空) |", "|---|---|---|---|"]
        for _, r in top_display.iterrows():
            lines.append(f"| {r['variant_a']} | {r['variant_b']} | {r['annual_rate']:.0f} | "
                          f"{r['sharpe']:+.3f}/{r['sharpe_long']:+.3f}/{r['sharpe_short']:+.3f} |")
    lines.append("")

    lines += [f"## 阶段2 walk-forward结果（Top{len(stage2_df)}）", ""]
    if len(stage2_df):
        lines += ["| 因子A | 因子B | 年均触发 | OOS Sharpe(合计,未年化) | OOS Sharpe(年化) | "
                   "OOS Sharpe(多/空,未年化) | 折数为正 |",
                   "|---|---|---|---|---|---|---|"]
        for _, r in stage2_df.sort_values("wf_oos_sharpe_all", ascending=False).iterrows():
            annual = (r["wf_n_long"] + r["wf_n_short"]) / years
            ann_sharpe = r["wf_oos_sharpe_all"] * (annual ** 0.5) if annual > 0 else float("nan")
            lines.append(f"| {r['variant_a']} | {r['variant_b']} | {annual:.0f} | "
                          f"{r['wf_oos_sharpe_all']:+.3f} | {ann_sharpe:+.2f} | "
                          f"{r['wf_oos_sharpe_long']:+.3f}/{r['wf_oos_sharpe_short']:+.3f} | "
                          f"{r['wf_n_folds_positive']}/5 |")
    lines.append("")

    lines += [f"## 最终通过的候选（OOS Sharpe>0 且 >=3/5折为正 且 年均触发>={MIN_ANNUAL_RATE}次）："
               f"{len(validated)}个", ""]
    if len(validated):
        lines += ["| 因子A | 因子B | 年均触发 | OOS Sharpe(未年化) | OOS Sharpe(年化) |",
                   "|---|---|---|---|---|"]
        for _, r in validated.iterrows():
            annual = (r["wf_n_long"] + r["wf_n_short"]) / years
            ann_sharpe = r["wf_oos_sharpe_all"] * (annual ** 0.5) if annual > 0 else float("nan")
            lines.append(f"| {r['variant_a']} | {r['variant_b']} | {annual:.0f} | "
                          f"{r['wf_oos_sharpe_all']:+.3f} | {ann_sharpe:+.2f} |")
    else:
        lines.append("无——两两组合在“足够的触发频率”和“稳健的OOS Sharpe”之间没能同时满足")
    lines.append("")

    lines += [
        "## 结论与下一步", "",
        f"- 完整数据：{pair_path}（阶段1全量）、{stage2_path}（阶段2明细）。",
        "- 如果这一步仍然找不到可用信号，说明两两AND也不够，可能需要三因子组合、"
        "OR逻辑、或改变评分方式（比如不追求胜率/Sharpe，而是先看方向准确率）。",
        "- 仍是占位持有期，步骤3定真实参数后需要重新验证。",
        "",
    ]

    report_path = os.path.join(args.report_dir, f"{args.out_prefix}_directional_combo_search_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[6/6] Wrote {report_path}")


if __name__ == "__main__":
    main()

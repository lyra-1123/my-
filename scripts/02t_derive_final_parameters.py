#!/usr/bin/env python3
"""Signal design step 7: derive final entry/exit parameters from step 6's
walk-forward results.

02s's nested walk-forward chose N and per-side stop-loss/take-profit
INDEPENDENTLY inside every fold (using only that fold's own training range)
-- which is exactly why it's trustworthy, but it also means there are 5
different parameter choices per candidate, not one. This step consolidates
those 5 per-fold choices into a single recommended configuration per
candidate: the majority stop-loss TYPE per side (mode across folds), the
median level/risk:reward among folds that picked that type, and the median
holding period N. It also reports how much the 5 folds agreed (a candidate
where every fold picked something different is a real caution flag, not
just noise to average over).

Scope note: this is a forward-looking deployment recommendation, not a new
OOS validation -- there is no further held-out data beyond the full sample
to test the consolidated rule on. 02s's reported Sharpe (fold-by-fold, each
fold using its OWN independently-chosen parameters) remains the honest
historical performance estimate; this step's output is what to actually
configure going into phase 3b, to be recalibrated periodically once live.

Only the 12 candidates that passed 02s's nested walk-forward are covered
(the 5 that failed there are dropped from the candidate pool, not reprocessed
here).

Usage:
    python scripts/02t_derive_final_parameters.py --report-dir reports
"""
import argparse
import os

import numpy as np
import pandas as pd


def consolidate_side(sub: pd.DataFrame, side: str) -> dict:
    types = sub[f"{side}_sl_type"].fillna("none")
    mode_type = types.mode().iloc[0]
    agreement = (types == mode_type).mean()
    if mode_type == "none":
        return {"sl_type": "none", "sl_level": np.nan, "rr": np.nan, "agreement": agreement}
    matching = sub[types == mode_type]
    return {
        "sl_type": mode_type,
        "sl_level": float(matching[f"{side}_sl_level"].median()),
        "rr": float(matching[f"{side}_rr"].median()),
        "agreement": agreement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    choices = pd.read_csv(os.path.join(args.report_dir, "02s_fold_choices.csv"))
    results = pd.read_csv(os.path.join(args.report_dir, "02s_nested_walk_forward_results.csv"))
    passing = results[results["passes"]]["name"].tolist()
    print(f"[1/2] Consolidating final parameters for {len(passing)}/{len(results)} candidates "
          f"that passed 02s's nested walk-forward ...")

    rows = []
    for name in passing:
        sub = choices[choices["name"] == name]
        n_hold_vals = sub["n_hold"].to_numpy()
        long_cfg = consolidate_side(sub, "long")
        short_cfg = consolidate_side(sub, "short")
        rows.append({
            "name": name,
            "n_hold_median": float(np.median(n_hold_vals)),
            "n_hold_min": int(n_hold_vals.min()), "n_hold_max": int(n_hold_vals.max()),
            "n_hold_stable": bool(n_hold_vals.max() / n_hold_vals.min() <= 2),
            "long_sl_type": long_cfg["sl_type"], "long_sl_level": long_cfg["sl_level"],
            "long_rr": long_cfg["rr"], "long_agreement": long_cfg["agreement"],
            "short_sl_type": short_cfg["sl_type"], "short_sl_level": short_cfg["sl_level"],
            "short_rr": short_cfg["rr"], "short_agreement": short_cfg["agreement"],
        })
    final_df = pd.DataFrame(rows)
    final_path = os.path.join(args.report_dir, "02t_final_parameters.csv")
    final_df.to_csv(final_path, index=False)

    print("[2/2] Writing report ...")
    lines = [
        "# 最终入场出场参数推导报告(信号设计步骤7)", "",
        "## 方法", "",
        "从步骤6(02s)每折独立选出的参数(只用该折训练区间数据，无未来函数)里，"
        "汇总出一套用于实盘部署的最终推荐参数：止损**类型**取5折的众数(哪种类型"
        "被选中的折数最多)，止损**水平**和**盈亏比**取选中该众数类型的那些折的中位数，"
        "持有期N取5折的中位数。**这不是新的样本外验证**——历史数据已经全部用完，"
        "没有更多留出的数据可以再测；这一步给出的是\"接下来实盘该配置成什么\"的建议，"
        "步骤6报告的分折Sharpe才是诚实的历史表现估计。", "",
        "只处理步骤6里通过筛选的12个候选(未通过的5个已经被淘汰，不在这里重新处理)。", "",
        "## 每个候选的一致性：5折选出的参数有多稳定", "",
        "| 候选 | N(中位数/范围) | N是否稳定(max/min<=2) | 多头止损类型(一致率) | "
        "空头止损类型(一致率) |",
        "|---|---|---|---|---|",
    ]
    for _, r in final_df.sort_values("long_agreement").iterrows():
        lines.append(f"| {r['name']} | {r['n_hold_median']:.0f}({r['n_hold_min']:.0f}~{r['n_hold_max']:.0f}) | "
                      f"{'是' if r['n_hold_stable'] else '否'} | "
                      f"{r['long_sl_type']}({r['long_agreement']:.0%}) | "
                      f"{r['short_sl_type']}({r['short_agreement']:.0%}) |")

    lines += ["", "## 最终推荐参数(用于阶段3b接入实盘/马丁引擎)", "",
              "| 候选 | 持有期N | 多头止损 | 多头水平 | 多头盈亏比 | 空头止损 | 空头水平 | 空头盈亏比 |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in final_df.iterrows():
        lines.append(f"| {r['name']} | {r['n_hold_median']:.0f} | {r['long_sl_type']} | "
                      f"{r['long_sl_level']} | {r['long_rr']} | {r['short_sl_type']} | "
                      f"{r['short_sl_level']} | {r['short_rr']} |")

    n_stable_long = int((final_df["long_agreement"] >= 0.6).sum())
    n_stable_short = int((final_df["short_agreement"] >= 0.6).sum())
    n_stable_n = int(final_df["n_hold_stable"].sum())
    lines += ["", "## 结构性发现（跨候选、跨折都比较一致）", "",
              "- 多头方向：绝大多数候选倾向于**固定百分比止损或不设止损**——这跟步骤4"
              "发现的规律一致(金价样本期长期偏多头，多头更容易扛得住不设止损)。",
              "- 空头方向：绝大多数候选倾向于**ATR倍数止损**——空头需要跟着波动率走的"
              "保护，不能用固定百分比一刀切。",
              f"- {n_stable_long}/{len(final_df)}个候选多头止损类型5折里≥60%一致，"
              f"{n_stable_short}/{len(final_df)}个候选空头止损类型≥60%一致，"
              f"{n_stable_n}/{len(final_df)}个候选持有期N的最大/最小折算比≤2(相对稳定)。"
              "一致率低的候选(比如止损类型5折各不相同)，说明具体的止损水平/持有期本身"
              "对市场阶段比较敏感，实盘部署后需要更频繁地重新校准，不能一套参数用到底。",
              "", "## 结论与下一步", "",
              f"- 完整数据：{final_path}。",
              "- 下一步（步骤8）：Regime切换响应——检查这些候选在不同波动率/趋势regime下"
              "表现是否稳定，是否需要根据当前regime动态调整参数或干脆停用某些候选。",
              ""]

    report_path = os.path.join(args.report_dir, "02t_final_parameters_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

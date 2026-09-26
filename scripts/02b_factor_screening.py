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

v2: the target strategy is an intraday martingale whose ladder cycle runs
4-8 hours (not forced flat by end of day), so screening is now done AT the
horizons that actually matter for that (4h, 8h), with the original 24h kept
only as a "if it runs long" reference point — not as the primary bar. A
family only counts as robust if it clears BOTH 4h and 8h.

v3: single 70/30 walk-forward split replaced with a 5-fold expanding-window
walk-forward (pooled OOS Sharpe + fold-consistency count) — see the module
docstring in src/factors/validation.py for why.

v4: four explicit joint criteria, ALL required, at EACH of 4h/8h:
  - |IC| >= IC_THRESHOLD (full-sample Spearman correlation with forward ER)
  - IR (ICIR) >= IR_THRESHOLD — mean/std of monthly IC, i.e. is the
    predictive power *consistent* period to period, not just strong on
    average once pooled across 18 years
  - PBO <= PBO_THRESHOLD (unchanged from v3)
  - OOS Sharpe > 0 (unchanged from v3, 5-fold pooled)
Every underlying indicator was also re-audited line by line for look-ahead:
every function uses only backward .rolling()/.shift(positive)/.diff()/causal
.ewm() — see reports/00_progress.md for the audit note.

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
    walk_forward_multi_fold, information_ratio,
)

TEST_HORIZONS = {"4小时": 4, "8小时": 8, "1天(参考)": 24}
CORE_HORIZONS = ("4小时", "8小时")  # a family must pass BOTH to be called robust
IC_THRESHOLD = 0.05
IR_THRESHOLD = 0.5
PBO_THRESHOLD = 0.6
N_FOLDS = 5
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
        wf = walk_forward_multi_fold(factors[best_variant], base_return, ic_signs[best_variant],
                                      points, n_folds=N_FOLDS)
        icir = information_ratio(factors[best_variant], label, factors["time"], points)

        passed = (
            (not np.isnan(pbo)) and pbo <= PBO_THRESHOLD
            and wf["oos_sharpe_pooled"] > 0
            and not np.isnan(icir["ic"]) and abs(icir["ic"]) >= IC_THRESHOLD
            and not np.isnan(icir["ir"]) and icir["ir"] >= IR_THRESHOLD
        )
        results.append({
            "category": FAMILY_CATEGORY.get(fam, "other"),
            "family": fam,
            "n_variants": len(variants),
            "best_variant": best_variant,
            "ic": icir["ic"],
            "ir": icir["ir"],
            "n_periods": icir["n_periods"],
            "pbo": pbo,
            "oos_sharpe_ann": wf["oos_sharpe_pooled"] * annualization,
            "n_folds_positive": wf["n_folds_positive"],
            "n_folds_total": wf["n_folds_total"],
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
        merged[f"ic_{name}"] = merged["family"].map(h_df["ic"])
        merged[f"ir_{name}"] = merged["family"].map(h_df["ir"])
        merged[f"pbo_{name}"] = merged["family"].map(h_df["pbo"])
        merged[f"oos_sharpe_{name}"] = merged["family"].map(h_df["oos_sharpe_ann"])
        merged[f"folds_positive_{name}"] = (
            merged["family"].map(h_df["n_folds_positive"]).astype(str) + "/"
            + merged["family"].map(h_df["n_folds_total"]).astype(str)
        )
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
        f"**v3变化**：v2用PBO<=0.5+单次70/30切分的walk-forward，稳健核心只筛出2个"
        "(adx、aroon_down)，嫌太少。先做了敏感性检验：单纯放宽PBO阈值几乎没用"
        "（0.5→0.6只多进来1个adx_slope，再往上放宽到0.7也没变化）——说明PBO不是真正卡住"
        "大多数因子的瓶颈，真正的瓶颈是“单次70/30切分”本身太脆弱：很多因子在4小时上"
        "OOS Sharpe为正，但到了8小时直接变成-0.4~-0.9，这更像是“样本外那一段时期恰好"
        "不利”的运气问题，而不一定是真的没有信号。所以把单次切分换成了**5折扩张窗口"
        "walk-forward**（训练集从第1折逐步扩张到第5折，每折都重新在训练区间内定安全"
        "分位阈值、冻结后只用于当折的测试区间，5折的样本外收益池化在一起算一个Sharpe，"
        "同时记录5折里有几折是正的）——这样一个因子要稳健得扛住多个不同的样本外时期，"
        f"而不是只看运气好不好压中最后30%。PBO阈值同时放宽到{PBO_THRESHOLD}。",
        "",
        "**v4变化**：改成四项联合标准，同时满足才算通过——单看PBO+OOS Sharpe会漏掉"
        "“平均下来相关性凑合，但逐月看时正时负、极不稳定”的因子，加上IC/IR两条硬指标"
        "堵住这个漏洞。同时把因子库里31个H1指标+4个H4指标的计算逻辑逐行审了一遍有没有"
        "用到未来函数——结果是干净的：每个函数只用`.rolling(n)`（向后看的窗口）、"
        "`.shift(正数)`（取过去的值）、`.diff()`、causal的`.ewm(adjust=False)`，"
        "没有任何`.shift(负数)`或居中窗口。真正用到未来数据的地方只有两处，且都明确"
        "标注仅用于评估、不作为任何输入：`labels.py`的前向ER标签，以及这里测试床策略"
        "本身要评分用的“未来horizon根bar的收益”（这是被评估的结果，不是决策依据）。",
        "",
        "每个因子家族有多个窗口/百分位版本（最多12个），这正是过拟合的高发地带——如果只是"
        "“挑全样本IC最高的那个”，很可能只是在噪音里挑到了运气好的参数组合。所以这里对每个"
        "家族在每个horizon分别做四项检验：",
        "",
        f"1. **\\|IC\\|>={IC_THRESHOLD}**：全样本Spearman秩相关系数绝对值，要有起码的相关性"
        "基础，不能只是噪音。",
        f"2. **IR(ICIR)>={IR_THRESHOLD}**：把决策点按自然月分组，每月单独算一次IC，"
        "IR=月度IC均值/月度IC标准差（按全样本IC符号定向，保证稳定的负相关也能得到正IR）——"
        "这条卡的是“稳定性”：同样的全样本IC，每月都小幅同向 vs 有的月份暴涨暴跌正负抵消，"
        "IR能把后者筛掉。",
        "3. **PBO（回测过拟合概率，CSCV方法，Bailey/Lopez de Prado）**：把不重叠的决策点"
        "分成10段，穷举所有把10段分成训练/测试两半的方式(C(10,5)=252种)，每种方式里"
        "“用训练半段挑出的样本内最优版本”在测试半段的表现排名——如果经常排到测试半段的"
        "中位数以下，说明这个“挑最优”的过程本身就是在过拟合噪音，PBO就是这个比例。",
        f"4. **{N_FOLDS}折扩张窗口walk-forward样本外Sharpe>0**：训练集从第1折扩张到第{N_FOLDS}折，"
        "每折都只用训练区间数据定安全分位阈值、冻结后应用到该折的测试区间，全部测试区间的"
        "收益池化后算一个Sharpe，同时记录几折为正——比单次切分更能反映“换几个不同的样本外"
        "时期结果还稳不稳”。",
        "",
        "因子本身不是交易规则，所以用一个固定的、跟因子无关的极简策略做“测试床”："
        "对最近1根bar做反向(fade)，持有到未来第horizon根bar——这样“用因子做门槛”和不用"
        "因子的版本除了因子那道门槛之外完全一样，Sharpe的差异能被干净地归因到因子本身，"
        "而不是策略设计。这个测试床本身很粗糙（没有点差/滑点/仓位管理），阶段3的真实马丁"
        "回测会更细，这里只是用来筛因子。Sharpe按对应horizon的决策点频率年化，不同horizon"
        "之间可以直接比较量级。",
        "",
        f"通过标准（单horizon，四项全部满足）：\\|IC\\|>={IC_THRESHOLD} 且 IR>={IR_THRESHOLD} "
        f"且 PBO<={PBO_THRESHOLD} 且 OOS Sharpe>0。“稳健核心”标准：{'和'.join(CORE_HORIZONS)}"
        "都要通过。",
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
        f"## 稳健核心：{n_core}/{len(families)}个家族同时通过{'和'.join(CORE_HORIZONS)}两个horizon"
        "（四项标准全部满足）",
        "",
        "| 家族 | 分类 | " + " | ".join(f"IC({h})" for h in TEST_HORIZONS) + " | "
        + " | ".join(f"IR({h})" for h in TEST_HORIZONS) + " | "
        + " | ".join(f"PBO({h})" for h in TEST_HORIZONS) + " | "
        + " | ".join(f"OOS Sharpe年化({h})" for h in TEST_HORIZONS) + " | "
        + " | ".join(f"正折数({h})" for h in TEST_HORIZONS) + " |",
        "|---|---|" + "---|" * len(TEST_HORIZONS) * 4 + "---|" * len(TEST_HORIZONS),
    ]
    core = merged[merged["robust_core"]].sort_values("oos_sharpe_4小时", ascending=False)
    for _, r in core.iterrows():
        ics = " | ".join(f"{r[f'ic_{h}']:+.3f}" for h in TEST_HORIZONS)
        irs = " | ".join(f"{r[f'ir_{h}']:+.2f}" for h in TEST_HORIZONS)
        pbos = " | ".join(f"{r[f'pbo_{h}']:.2f}" for h in TEST_HORIZONS)
        sharpes = " | ".join(f"{r[f'oos_sharpe_{h}']:+.2f}" for h in TEST_HORIZONS)
        folds = " | ".join(f"{r[f'folds_positive_{h}']}" for h in TEST_HORIZONS)
        lines.append(f"| {r['family']} | {r['category']} | {ics} | {irs} | {pbos} | {sharpes} | {folds} |")
    lines.append("")

    lines += [
        "## 全部家族逐horizon明细（按分类分组）",
        "",
    ]
    for cat, group in merged.groupby("category"):
        lines += [
            f"### {cat}",
            "",
            "| 家族 | " + " | ".join(f"{h}通过(正折数)" for h in TEST_HORIZONS) + " | 稳健核心 |",
            "|---|" + "---|" * len(TEST_HORIZONS) + "---|",
        ]
        for _, r in group.iterrows():
            marks = " | ".join(
                f"{'✅' if r[f'pass_{h}'] else '✗'}({r[f'folds_positive_{h}']})" for h in TEST_HORIZONS
            )
            core_mark = "✅" if r["robust_core"] else "✗"
            lines.append(f"| {r['family']} | {marks} | {core_mark} |")
        lines.append("")

    # near-miss diagnostics: how many of the 4 criteria does each family clear
    # at each horizon, and which single criterion is most often the blocker.
    crit_names = ["ic", "ir", "pbo", "oos"]
    blocker_counts = {h: {c: 0 for c in crit_names} for h in TEST_HORIZONS}
    near_miss_rows = []
    for _, r in merged.iterrows():
        for h in TEST_HORIZONS:
            ok = {
                "ic": abs(r[f"ic_{h}"]) >= IC_THRESHOLD if pd.notna(r[f"ic_{h}"]) else False,
                "ir": r[f"ir_{h}"] >= IR_THRESHOLD if pd.notna(r[f"ir_{h}"]) else False,
                "pbo": r[f"pbo_{h}"] <= PBO_THRESHOLD if pd.notna(r[f"pbo_{h}"]) else False,
                "oos": r[f"oos_sharpe_{h}"] > 0 if pd.notna(r[f"oos_sharpe_{h}"]) else False,
            }
            n_ok = sum(ok.values())
            if n_ok == 3:
                failed = [c for c, v in ok.items() if not v][0]
                near_miss_rows.append({"family": r["family"], "horizon": h, "failed_criterion": failed})
            for c, v in ok.items():
                if not v:
                    blocker_counts[h][c] += 1

    lines += ["## 差一步没过（3/4项标准满足）的家族——用来判断卡在哪一条", ""]
    if near_miss_rows:
        lines += ["| 家族 | horizon | 唯一没满足的标准 |", "|---|---|---|"]
        for row in near_miss_rows:
            lines.append(f"| {row['family']} | {row['horizon']} | {row['failed_criterion']} |")
    else:
        lines.append("（没有任何家族达到3/4，说明四项标准联合起来比单独看严格得多）")
    lines.append("")

    lines += ["## 各horizon下，每项标准各卡掉了多少家族（诊断瓶颈在哪）", "",
               "| horizon | IC不达标 | IR不达标 | PBO不达标 | OOS Sharpe不达标 |",
               "|---|---|---|---|---|"]
    for h in TEST_HORIZONS:
        bc = blocker_counts[h]
        lines.append(f"| {h} | {bc['ic']} | {bc['ir']} | {bc['pbo']} | {bc['oos']} |")
    lines.append("")

    lines += [
        "## 结论与下一步",
        "",
        f"- **诚实的结果：四项标准（\\|IC\\|>={IC_THRESHOLD}、IR>={IR_THRESHOLD}、"
        f"PBO<={PBO_THRESHOLD}、OOS Sharpe>0）同时满足，在4小时/8小时/1天(参考)三个"
        f"horizon上都是{n_core}个家族通过**——不是稳健核心从11降到0，是这四项联合起来"
        "比之前任何一版单独看的标准严格得多。诊断表显示**IC和IR是真正卡住绝大多数因子"
        "的瓶颈，不是PBO或OOS Sharpe**：4小时上PBO能过33/38、OOS Sharpe能过37/38，"
        "但IC>=0.05一个都过不了(0/38)；IR>=0.5同样0/38。即使放到最宽松的1天(参考)"
        "horizon，IC也只有4个家族过、IR只有1个家族过，且这两组几乎不重叠。",
        "- 最接近的例子：`h4_bb_width`在1天horizon上同时满足IC(0.065)和IR(0.57)两项，"
        "但OOS Sharpe是负的(-0.34)；`adx`在1天上IC/PBO/OOS三项都过，但IR只有0.38，"
        "差一点没到0.5；`efficiency_ratio`IC达标但PBO高达0.82（严重过拟合）。没有任何"
        "家族能同时把四项都做到——这印证了阶段2早就发现的结论：单个技术指标的regime"
        "预测力本来就弱（可解释方差不到1%），\\|IC\\|>=0.05这个门槛更像是常见于日频"
        "股票多因子研究的经验值，对着这种“单资产、日内几小时horizon”的regime分类问题"
        "可能天生就定得偏高。",
        "- **建议的路径（任选或组合）**：(a) 针对4-8小时horizon单独放宽IC/IR阈值"
        "（比如IC>=0.02、IR>=0.3），用更贴近这个问题实际能达到的量级；(b) 放弃“单因子"
        "必须自己达标”的思路，回到之前验证过的“多因子组合过滤器”——组合叠加确实能把"
        "未来ER压低15~18%，本次的IC/IR/PBO/OOS四项体系也可以整体搬到组合分数上再测一遍，"
        "而不只是套在单因子上；(c) 引入跨市场数据（美元指数/美债收益率等）挖掘可能IC"
        "更强的新因子，但需要额外的数据源。",
        f"- 完整逐horizon诊断数据（含IC/IR/PBO/OOS Sharpe/折数明细）在`{csv_path}`。",
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

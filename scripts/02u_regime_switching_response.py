#!/usr/bin/env python3
"""Signal design step 8 (final step): regime-switching response.

Step 7 found holding period N drifting substantially across the 5
walk-forward folds for 10/12 candidates -- a sign the "right" execution
parameters may depend on market regime rather than being one fixed
constant. This step checks that directly: tags every trade (using each
candidate's step-7 final parameters) by the market regime active at its
entry bar, and reports per-regime Sharpe. A candidate whose edge is
concentrated in one regime and flat/negative in others needs a regime GATE
(only trade when the current regime matches); one that's consistently
positive across regimes doesn't.

Regime classifier: Kaufman efficiency_ratio(100) on M5 (already an audited,
backward-looking, no-look-ahead factor from the library, purpose-built for
"trending vs choppy"), percentile-ranked over a trailing 2000-bar window,
split into terciles: choppy (bottom third), mid, trending (top third).

Usage:
    python scripts/02u_regime_switching_response.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS, efficiency_ratio, atr as atr_fn  # noqa: E402
from factors.direction import reversion_direction, combine_directions  # noqa: E402
from factors.execution import trade_windows, _sharpe  # noqa: E402

ATR_PERIOD = 14
REGIME_ER_WINDOW = 100
REGIME_PCTRANK_WINDOW = 2000
QUANTILE = 0.8

SINGLES = [
    ("vol_of_vol", 100, None), ("adx", 50, 24000), ("bb_width", 100, None),
    ("bb_width", 300, 6000), ("garman_klass_vol", 20, 6000), ("garman_klass_vol", 240, None),
]
PAIRS = [
    ("adx_50_pctrank2000", "autocorr_returns_50_pctrank2000"),
    ("kurt_returns_100", "mean_reversion_speed_50_pctrank500"),
    ("realized_vol_100_pctrank2000", "variance_ratio_2_50_pctrank2000"),
    ("parkinson_vol_100_pctrank2000", "aroon_up_10_pctrank500"),
    ("avg_gap_50", "roc_10"),
    ("stochastic_d_100_pctrank2000", "keltner_width_20"),
]


def variant_name(family, n, pw):
    return f"{family}_{n}" if pw is None else f"{family}_{n}_pctrank{pw}"


def full_sample_direction(factor: pd.Series, quantile: float = QUANTILE) -> pd.Series:
    return reversion_direction(factor, factor.quantile(1 - quantile), factor.quantile(quantile))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/4] Loading M5 OHLC + factors_M5.parquet + step7 final parameters ...")
    df = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    assert df["close"].equals(factors["close"]), "row alignment mismatch"
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    atr14 = atr_fn(df, ATR_PERIOD)
    final = pd.read_csv(os.path.join(args.report_dir, "02t_final_parameters.csv")).set_index("name")

    print("[2/4] Building regime classifier (efficiency_ratio(100) pctrank(2000), terciles) ...")
    er = efficiency_ratio(df, REGIME_ER_WINDOW)
    er_pctrank = er.rolling(REGIME_PCTRANK_WINDOW).rank(pct=True)
    regime = pd.cut(er_pctrank, bins=[-0.01, 1 / 3, 2 / 3, 1.01], labels=["choppy", "mid", "trending"])
    print(f"      regime distribution: {regime.value_counts(normalize=True).to_dict()}")

    single_factors = {}
    for family, n, pw in SINGLES:
        raw = WINDOWED_FACTORS[family](df, n)
        single_factors[variant_name(family, n, pw)] = raw.rolling(pw).rank(pct=True) if pw is not None else raw

    def sl_args(row, side):
        sl_type = row[f"{side}_sl_type"]
        sl_type = None if sl_type == "none" else sl_type
        sl_level = None if pd.isna(row[f"{side}_sl_level"]) else float(row[f"{side}_sl_level"])
        rr = None if pd.isna(row[f"{side}_rr"]) else float(row[f"{side}_rr"])
        return sl_type, sl_level, rr

    def candidate_trades_tagged(name, direction_full, n_hold, row):
        parts = []
        for side, sign in (("long", 1), ("short", -1)):
            side_dir = direction_full.where(np.sign(direction_full) == sign, 0.0)
            sl_type, sl_level, rr = sl_args(row, side)
            tw = trade_windows(side_dir, open_, high, low, close, atr14, n_hold, sl_type, sl_level, rr)
            parts.append(tw)
        tw = pd.concat(parts, ignore_index=True)
        tw["regime"] = regime.to_numpy()[tw["entry_idx"].to_numpy()]
        tw["ret"] = tw["direction"] * (tw["exit_price"] - tw["entry_price"]) / tw["entry_price"]
        return tw

    print("[3/4] Tagging trades by regime for all 12 candidates ...")
    all_names = [variant_name(*s) for s in SINGLES] + [f"{a}+{b}" for a, b in PAIRS]
    per_candidate = {}
    for family, n, pw in SINGLES:
        name = variant_name(family, n, pw)
        direction_full = full_sample_direction(single_factors[name])
        row = final.loc[name]
        per_candidate[name] = candidate_trades_tagged(name, direction_full, int(row["n_hold_median"]), row)
        print(f"      {name} done")
    for var_a, var_b in PAIRS:
        name = f"{var_a}+{var_b}"
        dir_a, dir_b = full_sample_direction(factors[var_a]), full_sample_direction(factors[var_b])
        combined = combine_directions(dir_a, dir_b)
        row = final.loc[name]
        per_candidate[name] = candidate_trades_tagged(name, combined, int(row["n_hold_median"]), row)
        print(f"      {name} done")

    print("[4/4] Writing report ...")
    rows = []
    for name in all_names:
        tw = per_candidate[name]
        overall_sh = _sharpe(tw["ret"].to_numpy())
        regime_stats = {}
        for reg in ("choppy", "mid", "trending"):
            sub = tw[tw["regime"] == reg]
            regime_stats[reg] = {"sharpe": _sharpe(sub["ret"].to_numpy()), "n": len(sub)}
        sharpes = [regime_stats[r]["sharpe"] for r in ("choppy", "mid", "trending") if not np.isnan(regime_stats[r]["sharpe"])]
        regime_dependent = (len(sharpes) >= 2) and (max(sharpes) > 0 and min(sharpes) < 0)
        rows.append({
            "name": name, "overall_sharpe": overall_sh,
            "choppy_sharpe": regime_stats["choppy"]["sharpe"], "choppy_n": regime_stats["choppy"]["n"],
            "mid_sharpe": regime_stats["mid"]["sharpe"], "mid_n": regime_stats["mid"]["n"],
            "trending_sharpe": regime_stats["trending"]["sharpe"], "trending_n": regime_stats["trending"]["n"],
            "regime_dependent": regime_dependent,
        })
    result_df = pd.DataFrame(rows)
    result_path = os.path.join(args.report_dir, "02u_regime_breakdown.csv")
    result_df.to_csv(result_path, index=False)

    n_dependent = int(result_df["regime_dependent"].sum())
    lines = [
        "# Regime切换响应报告(信号设计步骤8，收官)", "",
        "## 方法", "",
        f"用`efficiency_ratio({REGIME_ER_WINDOW})`(已审计过的无未来函数因子，专门衡量"
        f"趋势/震荡)在M5上算，再滚动百分位排名(回看{REGIME_PCTRANK_WINDOW}根)分三档："
        "choppy(震荡，后1/3)、mid(中间)、trending(趋势，前1/3)。用步骤7的最终参数"
        "(每个候选自己的N+分方向止损止盈)重建全部交易，按每笔交易**入场那一刻**的regime"
        "打标签，分别计算三档下的Sharpe。", "",
        "## 每个候选分regime表现", "",
        "| 候选 | 整体Sharpe | 震荡(Sharpe/笔数) | 中间(Sharpe/笔数) | 趋势(Sharpe/笔数) | "
        "是否明显依赖某个regime |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in result_df.sort_values("regime_dependent", ascending=False).iterrows():
        lines.append(f"| {r['name']} | {r['overall_sharpe']:+.3f} | "
                      f"{r['choppy_sharpe']:+.3f}/{r['choppy_n']:.0f} | "
                      f"{r['mid_sharpe']:+.3f}/{r['mid_n']:.0f} | "
                      f"{r['trending_sharpe']:+.3f}/{r['trending_n']:.0f} | "
                      f"{'是' if r['regime_dependent'] else '否'} |")

    lines += ["", f"**{n_dependent}/{len(result_df)}个候选表现明显依赖regime**"
              "(至少一档为正、至少一档为负)——这些候选适合加一道regime门(只在对自己"
              "有利的regime开仓)；其余候选三档基本同号，说明本身就是跨regime稳健的，"
              "不需要额外的regime门。", "",
              "## 结论与下一步", "",
              f"- 完整数据：{result_path}。",
              "- 这是信号设计track的最后一步(步骤1~8全部完成)。信号设计track目前的正式"
              "产出：12个通过步骤6严格walk-forward验证的候选(6个单因子+6个组合)，"
              "每个都有步骤7给出的最终参数，以及这一步给出的regime依赖标记。",
              "- 后续工作：阶段3b——把这12个候选接入真实马丁引擎(`allow_entry`/方向"
              "gating)，regime依赖的候选加上对应的regime门，重新在真实引擎里跑一遍"
              "完整回测（含点差、滑点、保证金等真实约束），而不是这个信号设计track"
              "一直用的简化测试床。",
              ""]

    report_path = os.path.join(args.report_dir, "02u_regime_switching_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

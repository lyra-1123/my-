#!/usr/bin/env python3
"""Signal design: fill the PBO (probability of backtest overfitting) gap in
the directional-signal track. 02h/02i/02j only screened on OOS Sharpe +
fold consistency + activation-rate usability -- unlike the earlier
regime-classification track (02b/02c), which the user's original standard
(|IC|>0.05, IR>0.5, PBO low, OOS Sharpe>0) explicitly required PBO for.
This was a real gap: 02h searched 798 (variant, mode) combinations and 02i
searched 482 pairs before picking the best few, and neither checked whether
"best by in-sample Sharpe" was a robust pick or just noise from a big
multiple-comparison search.

Runs CSCV (Bailey/Lopez de Prado) PBO over three pools:
  1. 02h's single-factor search pool (798 candidates that cleared the
     activation floor)
  2. 02i's pairwise-combination search pool (482 pairs that cleared the
     frequency floor)
  3. 02j's per-family window-choice pool (19 specs x 4 window multipliers
     each -- was the "winning window" pick itself robust, or one of only
     4 candidates so less of a concern, but checked anyway)

Candidates' return arrays are computed one at a time and discarded after
their block-Sharpe row is recorded (never all held in memory at once --
798+482 full 1.28M-bar float64 arrays would be ~10GB).

Usage:
    python scripts/02m_pbo_check.py --clean-dir data/clean --report-dir reports
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors.library import WINDOWED_FACTORS  # noqa: E402
from factors.validation import _sharpe, pbo_from_block_sharpe  # noqa: E402
from factors.direction import momentum_direction, reversion_direction, combine_directions  # noqa: E402

HOLDING_BARS = 12
N_BLOCKS = 10

# same 19 (family, base_window_h1, pctrank_h1) specs as 02j_window_optimization.py
SPECS = [
    ("vol_of_vol", 100, None), ("adx", 50, 2000), ("bb_width", 100, None),
    ("bb_width", 100, 500), ("parkinson_vol", 100, 2000), ("parkinson_vol", 20, 500),
    ("garman_klass_vol", 20, 500), ("garman_klass_vol", 20, None),
    ("autocorr_returns", 50, 2000), ("avg_gap", 50, None), ("kurt_returns", 100, None),
    ("mean_reversion_speed", 50, 500), ("realized_vol", 100, 2000), ("roc", 10, None),
    ("variance_ratio_2", 50, 2000), ("aroon_up", 10, 500), ("zscore_vs_ma", 100, 500),
    ("stochastic_d", 100, 2000), ("keltner_width", 20, None),
]
MULTIPLIERS = (1, 3, 6, 12)
PCTRANK_SCALE = 12


def build_direction(factor: pd.Series, close: pd.Series, mode: str, quantile: float = 0.8) -> pd.Series:
    if mode == "momentum":
        return momentum_direction(factor, close, factor.quantile(quantile))
    return reversion_direction(factor, factor.quantile(1 - quantile), factor.quantile(quantile))


def block_sharpe_row(direction: pd.Series, fwd_return: pd.Series, eligible: np.ndarray, edges: np.ndarray) -> np.ndarray:
    active = direction[direction != 0].index.intersection(fwd_return.dropna().index)
    ret_full = pd.Series(0.0, index=direction.index)
    ret_full.loc[active] = direction.loc[active] * fwd_return.loc[active]
    arr = ret_full.to_numpy(dtype=np.float32)[eligible]
    return np.array([_sharpe(arr[edges[i]:edges[i + 1]]) for i in range(len(edges) - 1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    print("[1/5] Loading factors_M5.parquet (798+482 candidate pool, original H1-transplant windows) ...")
    factors = pd.read_parquet(os.path.join(args.clean_dir, "factors_M5.parquet"))
    close = factors["close"]
    fwd_return = close.pct_change(HOLDING_BARS).shift(-HOLDING_BARS)
    eligible = fwd_return.notna().to_numpy()
    n_eligible = int(eligible.sum())
    edges = np.linspace(0, n_eligible, N_BLOCKS + 1).astype(int)

    print("[2/5] PBO over 02h's 798 single-factor candidates ...")
    stage1_s = pd.read_csv(os.path.join(args.report_dir, "02h_directional_stage1_scan.csv"))
    names_s, block_sharpe_s = [], np.full((N_BLOCKS, len(stage1_s)), np.nan)
    for i, row in stage1_s.iterrows():
        direction = build_direction(factors[row["variant"]], close, row["mode"])
        block_sharpe_s[:, i] = block_sharpe_row(direction, fwd_return, eligible, edges)
        names_s.append(f"{row['variant']}_{row['mode']}")
        if i % 200 == 0:
            print(f"      {i}/{len(stage1_s)}")
    pbo_s, diag_s = pbo_from_block_sharpe(block_sharpe_s, names_s, N_BLOCKS)
    print(f"      PBO={pbo_s:.3f} ({diag_s['n_splits']} splits), full-sample best={diag_s['best_by_full_sample']}")

    print("[3/5] PBO over 02i's 482 pair candidates ...")
    stage1_p = pd.read_csv(os.path.join(args.report_dir, "02i_pair_stage1_scan.csv"))
    names_p, block_sharpe_p = [], np.full((N_BLOCKS, len(stage1_p)), np.nan)
    for i, row in stage1_p.iterrows():
        da = build_direction(factors[row["variant_a"]], close, row["mode_a"])
        db = build_direction(factors[row["variant_b"]], close, row["mode_b"])
        combined = combine_directions(da, db)
        block_sharpe_p[:, i] = block_sharpe_row(combined, fwd_return, eligible, edges)
        names_p.append(f"{row['variant_a']}+{row['variant_b']}")
        if i % 100 == 0:
            print(f"      {i}/{len(stage1_p)}")
    pbo_p, diag_p = pbo_from_block_sharpe(block_sharpe_p, names_p, N_BLOCKS)
    print(f"      PBO={pbo_p:.3f} ({diag_p['n_splits']} splits), full-sample best={diag_p['best_by_full_sample']}")

    print("[4/5] PBO per family for 02j's window-choice pool (19 specs x 4 windows) ...")
    df_m5 = pd.read_parquet(os.path.join(args.clean_dir, "XAUUSD_M5.parquet")).dropna(subset=["close"]).reset_index(drop=True)
    close_m5 = df_m5["close"]
    fwd_return_m5 = close_m5.pct_change(HOLDING_BARS).shift(-HOLDING_BARS)
    eligible_m5 = fwd_return_m5.notna().to_numpy()
    n_eligible_m5 = int(eligible_m5.sum())
    edges_m5 = np.linspace(0, n_eligible_m5, N_BLOCKS + 1).astype(int)

    spec_rows = []
    for family, n0, pw0 in SPECS:
        func = WINDOWED_FACTORS[family]
        names_w, block_sharpe_w = [], np.full((N_BLOCKS, len(MULTIPLIERS)), np.nan)
        for j, m in enumerate(MULTIPLIERS):
            n = max(2, n0 * m)
            raw = func(df_m5, n)
            if pw0 is not None:
                pw = pw0 * PCTRANK_SCALE
                factor = raw.rolling(pw).rank(pct=True)
                variant = f"{family}_{n}_pctrank{pw}"
            else:
                factor = raw
                variant = f"{family}_{n}"
            direction = reversion_direction(factor, factor.quantile(0.2), factor.quantile(0.8))
            block_sharpe_w[:, j] = block_sharpe_row(direction, fwd_return_m5, eligible_m5, edges_m5)
            names_w.append(variant)
        pbo_w, diag_w = pbo_from_block_sharpe(block_sharpe_w, names_w, N_BLOCKS)
        spec_rows.append({
            "family": family, "pctrank_h1": pw0, "pbo": pbo_w,
            "n_splits": diag_w["n_splits"], "best_by_full_sample": diag_w["best_by_full_sample"],
        })
        print(f"      {family}(pctrank_h1={pw0}): PBO={pbo_w:.3f}, best={diag_w['best_by_full_sample']}")

    spec_df = pd.DataFrame(spec_rows)
    spec_path = os.path.join(args.report_dir, "02m_pbo_window_specs.csv")
    spec_df.to_csv(spec_path, index=False)

    print("[5/5] Writing report ...")
    lines = [
        "# PBO(过拟合概率)补充检验报告", "",
        "## 背景", "",
        "02h(798个单因子候选)、02i(482对组合候选)、02j(19个家族x4个窗口=76个候选)"
        "三次搜索都只用OOS Sharpe+折一致性+触发频率筛选，没有做PBO——这是相对于"
        "阶段2c/2d\"IC+IR+PBO+OOS Sharpe\"标准的一个缺口。这里用CSCV(Bailey/"
        "Lopez de Prado方法，`src/factors/validation.py`里已有的`pbo_from_block_sharpe`"
        f"复用)补上：{N_BLOCKS}个连续区块，穷举所有对半分法({N_BLOCKS}选{N_BLOCKS//2})"
        "，每次用一半区块选\"样本内最优候选\"，检查它在另一半区块里的排名是否掉到中位数"
        "以下。PBO=这种情况发生的比例，越接近0.5说明样本内最优≈随机噪音(过拟合)，"
        "越接近0说明样本内最优在样本外也大概率靠前(搜索结果可信)。",
        "",
        "## 结果", "",
        "| 候选池 | 候选数 | PBO | 全样本最优候选 | 是否与实际筛选出的最终候选一致 |",
        "|---|---|---|---|---|",
        f"| 02h单因子(798) | {len(stage1_s)} | {pbo_s:.3f} | {diag_s['best_by_full_sample']} | "
        f"{'是' if 'vol_of_vol_100' in diag_s['best_by_full_sample'] else '否——见下方说明'} |",
        f"| 02i两两组合(482) | {len(stage1_p)} | {pbo_p:.3f} | {diag_p['best_by_full_sample']} | "
        f"{'是' if 'kurt_returns_100' in diag_p['best_by_full_sample'] or 'mean_reversion_speed' in diag_p['best_by_full_sample'] else '否——见下方说明'} |",
        "",
        "## 02j窗口选择池(19个家族，每个家族4个候选窗口)逐家族PBO", "",
        "| 家族 | pctrank(H1) | PBO | 全样本最优窗口 |",
        "|---|---|---|---|",
    ]
    for _, r in spec_df.iterrows():
        lines.append(f"| {r['family']} | {r['pctrank_h1']} | {r['pbo']:.3f} | {r['best_by_full_sample']} |")

    lines += [
        "", "## 解读", "",
        f"- PBO<0.5：样本内最优在样本外排名中位数以上的次数更多，说明这次搜索\"挑出来的"
        "最优\"不是纯噪音。PBO接近甚至超过0.5则说明这批候选里选出来的\"最优\"很可能是"
        "过拟合——样本内最优和样本外最优基本没关系。",
        "- 这里的PBO用的是全样本固定阈值(不是walk-forward逐折重新拟合的阈值)，衡量的是"
        "\"搜索/挑选过程本身\"有多可信，跟02h/02i/02k里walk-forward验证的\"这个具体因子"
        "OOS Sharpe是否为正\"是两个不同但互补的问题——一个测搜索过程，一个测选中的候选"
        "本身。",
        f"- 完整明细：{spec_path}（02j窗口池逐家族）；02h/02i的候选级明细未展开保存"
        "(798+482行的block Sharpe矩阵体积较大，只保留了本报告的汇总数字)。",
        "",
    ]

    report_path = os.path.join(args.report_dir, "02m_pbo_check_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

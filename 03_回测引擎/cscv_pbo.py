"""
cscv_pbo.py
============
指南 5.3 节"三大过拟合度量"：PBO（CSCV方法）+ DSR。基本照抄指南给的算法，
只做了必要的健壮性处理（除零、n_strategies=1时的退化情况）。

PBO 的前提是"N个候选策略"——单一策略没有意义谈"过拟合概率"。本项目已经在同一份
数据上调了6轮假设/变体，PBO 应该用于检验："在我们实际尝试过的这些变体里，
表现最好的那个（ATR动量v2），是不是只是矮子里的将军"。见
03_回测引擎/validate_ch5_atr_momentum_v2.py 里怎么构造这个"变体家族"。
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as kurt
from scipy.stats import norm, skew


def build_returns_matrix(trade_dfs: list[pd.DataFrame], n_bars: int) -> np.ndarray:
    """
    把多个策略变体的交易记录，投影到统一的bar时间线上，做成 CSCV 需要的
    returns_matrix（shape=(n_strategies, n_bars)）。
    每笔交易的收益率（return_pct）记在它的 exit_idx 那根bar上，其余bar记0——
    这是低频、持仓期短（本项目最长6根bar）信号的合理简化，不适合高频/长持仓策略。
    """
    matrix = np.zeros((len(trade_dfs), n_bars))
    for i, trades in enumerate(trade_dfs):
        if trades.empty:
            continue
        np.add.at(matrix[i], trades["exit_idx"].to_numpy(dtype=int), trades["return_pct"].to_numpy())
    return matrix


def calc_pbo(returns_matrix: np.ndarray, n_splits: int = 16) -> dict:
    """
    计算 PBO（Combinatorially Symmetric Cross-Validation）。
    returns_matrix: shape=(n_strategies, n_bars)，每个策略的逐bar收益率。
    """
    n_strategies, n_bars = returns_matrix.shape
    if n_strategies < 2:
        return {"pbo": float("nan"), "note": "PBO需要至少2个候选策略才有意义"}

    fold_size = n_bars // n_splits
    if fold_size == 0:
        raise ValueError(f"n_bars={n_bars} 太少，撑不起 n_splits={n_splits} 折")

    # 与指南逐组合拼接数组再求和数学上完全等价：log收益可加，先按块求和，
    # 每个组合只需把8个块的和加起来——否则128万根bar×12870个组合要跑几个小时
    log_returns = np.log1p(returns_matrix[:, :fold_size * n_splits])
    block_log = log_returns.reshape(n_strategies, n_splits, fold_size).sum(axis=2)
    all_blocks = set(range(n_splits))

    is_oos_records = []
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        oos_blocks = sorted(all_blocks - set(is_blocks))

        is_log = block_log[:, list(is_blocks)].sum(axis=1)
        oos_log = block_log[:, oos_blocks].sum(axis=1)

        best_idx = int(np.argmax(is_log))
        oos_rank_relative = np.sum(oos_log < oos_log[best_idx]) / n_strategies

        is_oos_records.append({
            "best_idx": best_idx,
            "oos_rank_relative": oos_rank_relative,
            "is_median": float(np.median(is_log)),
        })

    pbo = float(np.mean([r["oos_rank_relative"] < 0.5 for r in is_oos_records]))

    return {
        "pbo": pbo,
        "n_strategies": n_strategies,
        "n_splits": n_splits,
        "n_combinations": len(is_oos_records),
        "interpretation": "低过拟合风险" if pbo < 0.25 else ("中等风险" if pbo < 0.5 else "高过拟合风险"),
    }


def calc_dsr(returns: np.ndarray, n_trials: int) -> dict:
    """
    计算 DSR（Deflated Sharpe Ratio），逐笔口径（不年化）。
    returns: 策略的逐笔收益率序列。
    n_trials: 候选策略总数——本项目用"目前正式验证过的假设/变体总数"，不是1。

    与指南5.3.2代码的差异：指南用"年化Sharpe + 基准 sqrt(log N)"，这里统一用逐笔Sharpe，
    基准相应换成 sqrt(2·ln N)/sqrt(T)（Bailey & López de Prado 原文的"N次随机试验的
    期望最大Sharpe"在逐期单位下的近似）。如果照搬指南的 sqrt(log N) 配逐笔Sharpe，
    基准会高到任何策略都不可能通过，单位不一致。
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    T = len(returns)
    if T < 2 or returns.std() == 0:
        return {"dsr_prob": float("nan"), "note": "样本太少或方差为0，无法计算DSR"}

    sr_estimated = returns.mean() / returns.std()
    sr_benchmark = np.sqrt(2 * np.log(n_trials)) / np.sqrt(T) if n_trials > 1 else 0.0

    skewness = skew(returns)
    kurtosis_ = kurt(returns, fisher=True)

    se_sr = np.sqrt(
        max(1e-12, 1 + skewness * sr_estimated - (kurtosis_ - 1) / 4 * sr_estimated ** 2)
        / (T - 1)
    )

    dsr_z = (sr_estimated - sr_benchmark) / se_sr if se_sr > 0 else 0.0
    dsr_prob = float(norm.cdf(dsr_z))

    return {
        "sr_estimated": float(sr_estimated),
        "sr_benchmark": float(sr_benchmark),
        "dsr_prob": dsr_prob,
        "dsr_z": float(dsr_z),
        "n_trials": n_trials,
        "n_obs": T,
        "interpretation": "PASS" if dsr_prob > 0.95 else "FAIL",
    }

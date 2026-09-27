# -*- coding: utf-8 -*-
"""
信号簇代表之间的相关性与增量信息（不看盈亏，只看信号结构）。

对每个频率：
  1) 代表因子：research.icir_clusters 聚类后每簇 |t| 最大者，方向统一为 ICIR>0
  2) 因子值 Spearman 相关（样本内 / 样本外）
  3) 月度 Rank IC 序列的相关：两个信号是否在同一时期一起有效/失效（分散化的关键）
  4) 增量信息：每个代表对其余代表做线性回归（系数只用样本内估计，样本外直接套用），
     残差的月度 ICIR（样本内 / 样本外）= 扣除其他信号后剩下的独立信息
  5) 组合：等权、样本内 ICIR 加权 的 ICIR，对比最好的单因子
用法：python -m research.factor_correlation [5MIN 15MIN ...]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from factors.core import rolling_mad_zscore
from factors.evaluate import HORIZONS, forward_return, ic_stats, rank_ic
from factors.registry import get_factors
from research.icir_clusters import candidates

SPLIT = pd.Timestamp("2020-01-01")
SHORT = {"TrendPullbackLowVolume": "Pullback", "VWAPDeviation": "VWAPDev", "RangeVWAPReversion": "RangeVWAP",
         "VolWeightedCloseThrust": "VWCT", "SqueezeReleaseMomentum": "Squeeze", "VolumeClimaxReversal": "Climax",
         "VolConfirmedBreakout": "VCB", "TSMomentumVolScaled": "TSMom", "TrendEfficiencyVolume": "TrendEff",
         "MTFTrendResonance": "MTFTrend", "MTFPullbackResonance": "MTFPullback"}


def factor_values(freq: str, names: list[str]) -> pd.DataFrame:
    """计算并缓存因子值（未取方向）。"""
    path = f"data/cache/factors_{freq}.pkl"
    cache = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame()
    miss = [n for n in names if n not in cache.columns]
    if miss:
        df = pd.read_pickle(f"data/cache/{freq}.pkl")
        new = pd.DataFrame({n: get_factors([n])[0](df, freq) for n in miss})
        cache = pd.concat([cache, new], axis=1) if len(cache) else new
        cache.to_pickle(path)
    return cache[names]


def representatives(freq: str) -> dict[str, float]:
    """每簇取 |t| 最大者；返回 {因子名: 方向(+1/-1)}。与 icir_clusters 相同的聚类口径。"""
    cand = candidates(freq)
    F = factor_values(freq, list(cand))
    F = F.loc["2015":"2019"].mul(pd.Series({n: np.sign(m["icir"]["icir"]) for n, m in cand.items()}))
    C = F.dropna().rank().corr()
    lab = fcluster(linkage(squareform(1 - C.abs().values, checks=False), "average"), t=0.4, criterion="distance")
    reps = {}
    for c in set(lab):
        names = [n for n, l in zip(C.index, lab) if l == c]
        best = max(names, key=lambda n: abs(cand[n]["icir"]["ic_t"]))
        reps[best] = float(np.sign(cand[best]["icir"]["icir"]))
    return reps


def monthly_ic(f: pd.Series, y: pd.Series) -> pd.Series:
    key = f.index.to_period("M")
    return pd.Series({k: rank_ic(f[key == k], y[key == k]) for k in key.unique()})


def label(n: str, s: float) -> str:
    return ("+" if s > 0 else "−") + SHORT.get(n, n)


def main(freq: str) -> None:
    df = pd.read_pickle(f"data/cache/{freq}.pkl")
    reps = representatives(freq)
    F = factor_values(freq, list(reps)).mul(pd.Series(reps))
    F.columns = [label(n, reps[n]) for n in reps]
    y = forward_return(df, HORIZONS[freq][1])
    ins, oos = F.index < SPLIT, F.index >= SPLIT

    print(f"\n{'#' * 100}\n{freq}（主持有期 h={HORIZONS[freq][1]}）  代表因子：{', '.join(F.columns)}")
    print("\n[1] 因子值 Spearman 相关   左下：样本内 2009-2019 | 右上：样本外 2020-")
    ci, co = F[ins].dropna().rank().corr(), F[oos].dropna().rank().corr()
    M = ci.copy()
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            M.iloc[i, j] = co.iloc[i, j]
    print(M.round(2).to_string())

    ics = pd.DataFrame({c: monthly_ic(F[c], y) for c in F})
    print("\n[2] 月度 IC 序列相关（全样本）：是否同时有效/同时失效")
    print(ics.corr().round(2).to_string())

    print("\n[3] 单因子 vs 残差（扣除其他代表后）的月度 ICIR   回归系数只用样本内估计")
    rows = []
    Z = F.fillna(0.0)
    for c in F:
        others = [o for o in F if o != c]
        X, t = Z.loc[ins, others].to_numpy(), Z.loc[ins, c].to_numpy()
        beta = np.linalg.lstsq(np.c_[np.ones(len(X)), X], t, rcond=None)[0]
        resid = Z[c] - (beta[0] + Z[others].to_numpy() @ beta[1:])
        r2 = 1 - np.var(t - np.c_[np.ones(len(X)), X] @ beta) / np.var(t)
        s_in, s_out = ic_stats(F[c], y, freq, ins), ic_stats(F[c], y, freq, oos)
        r_in, r_out = ic_stats(resid, y, freq, ins), ic_stats(resid, y, freq, oos)
        rows.append({"因子": c, "被其他因子解释R²": round(r2, 2),
                     "ICIR内": s_in["icir"], "ICIR外": s_out["icir"],
                     "残差ICIR内": r_in["icir"], "残差ICIR外": r_out["icir"], "残差t外": r_out["ic_t"]})
    print(pd.DataFrame(rows).set_index("因子").to_string())

    print("\n[4] 组合因子的月度 ICIR（各成分先按样本内 ICIR 统一方向；组合后再做滚动 MAD+Z）")
    w_icir = pd.Series({c: ic_stats(F[c], y, freq, ins)["icir"] for c in F})
    combos = {"等权": F.mean(axis=1), "样本内ICIR加权": (F * w_icir).sum(axis=1) / w_icir.abs().sum()}
    out = []
    for c in F:
        out.append({"信号": c, "ICIR内": ic_stats(F[c], y, freq, ins)["icir"], "ICIR外": ic_stats(F[c], y, freq, oos)["icir"],
                    "t外": ic_stats(F[c], y, freq, oos)["ic_t"], "IC均值外": ic_stats(F[c], y, freq, oos)["ic_mean"]})
    from factors.core import params
    for k, v in combos.items():
        z = rolling_mad_zscore(v, params(freq)["norm"])
        a, b = ic_stats(z, y, freq, ins), ic_stats(z, y, freq, oos)
        out.append({"信号": f"组合·{k}", "ICIR内": a["icir"], "ICIR外": b["icir"], "t外": b["ic_t"], "IC均值外": b["ic_mean"]})
    print(pd.DataFrame(out).set_index("信号").to_string())
    print("  样本内 ICIR 权重:", w_icir.round(2).to_dict())


if __name__ == "__main__":
    for fq in sys.argv[1:] or ["5MIN", "15MIN"]:
        main(fq)

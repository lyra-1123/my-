# -*- coding: utf-8 -*-
"""
过拟合检验：CSCV / PBO（Bailey, Borwein, López de Prado, Zhu 2014）与 DSR（Bailey & López de Prado 2014）。

两个层面：
  A 参数层：30MIN 趋势尾部 3 对象 × chan{16,24,32,48,64} × entry{1,1.25,1.5,1.75,2} × exit{0,0.3,0.6} = 225 个配置
  B 研究层：因子库全部"因子 × 频率"在统一规则下的策略（研究过程实际比较过的集合）
收益：日度，"ATR·今"口径（收益/上一根 ATR、逐年去漂移、成本按最近 1 年 ATR 折算），2009-2026 全样本。
CSCV：S=16 块，C(16,8)=12870 种切分；每种切分下取样本内夏普最高者，看它在样本外的相对排名 w，logit λ=ln(w/(1-w))；
      PBO = P(λ ≤ 0)。另报告：样本内最优在样本外的夏普分布、样本外亏损概率。
DSR：对所选策略（默认参数等权组合），在不同试验次数 N 下计算；N_eff 用收益相关矩阵特征值的参与比估计。
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from factors.core import atr, params
from factors.evaluate import SPREAD, SWAP, execution_signal, positions, swap_units
from factors.registry import get_factors

OUT = "data/cache/daily_returns_{}.pkl"


# ---------------------------------------------------------------------------
# 日度收益矩阵
# ---------------------------------------------------------------------------
def daily_atr_returns(df: pd.DataFrame, z: pd.Series, freq: str, entry=1.5, exit_=0.3) -> pd.Series:
    a = atr(df, params(freq)["atr"])
    a_prev = a.shift(1)
    atr_now = float(a[df.index >= df.index[-1] - pd.Timedelta(days=365)].median())
    move = df["open"].shift(-1) - df["open"]
    m = move / a_prev
    m = m - m.groupby(df.index.year).transform("mean")
    pos = positions(execution_signal(z, freq), entry, exit_)
    dpos = pos.diff().abs().fillna(pos.abs())
    r = (pos * m - dpos * SPREAD / 2 / atr_now - pos.abs() * swap_units(df.index) * SWAP / atr_now).fillna(0)
    d = r.groupby(r.index.normalize()).sum()
    return d[d.index.dayofweek < 5]


def matrix_A() -> pd.DataFrame:
    path = OUT.format("A")
    if os.path.exists(path):
        return pd.read_pickle(path)
    from research.trend_tail_validation import CHANS, ENTRIES, EXITS, FREQ, df, factor_z
    cols = {}
    for n in ("TrendEff", "VWAPDev", "等权"):
        for c in CHANS:
            z = factor_z(n, c)
            for e, x in itertools.product(ENTRIES, EXITS):
                cols[f"{n}|{c}|{e}|{x}"] = daily_atr_returns(df, z, FREQ, e, x)
    M = pd.DataFrame(cols).fillna(0.0)
    M.to_pickle(path)
    return M


def matrix_B() -> pd.DataFrame:
    path = OUT.format("B")
    if os.path.exists(path):
        return pd.read_pickle(path)
    cols = {}
    for freq in ("5MIN", "15MIN", "30MIN", "1H", "4H", "1D"):
        df = pd.read_pickle(f"data/cache/{freq}.pkl")
        for spec in get_factors():
            if freq in spec.freqs:
                cols[f"{spec.name}|{freq}"] = daily_atr_returns(df, spec(df, freq), freq)
    M = pd.DataFrame(cols).fillna(0.0)
    M = M[M.index.dayofweek < 5]
    M.to_pickle(path)
    return M


# ---------------------------------------------------------------------------
# CSCV / PBO
# ---------------------------------------------------------------------------
def cscv(M: pd.DataFrame, S: int = 16) -> dict:
    X = M.to_numpy()
    T = (len(X) // S) * S
    X = X[-T:]
    blocks = np.array_split(np.arange(T), S)
    n = np.array([len(b) for b in blocks], float)
    s1 = np.array([X[b].sum(0) for b in blocks])            # S × N
    s2 = np.array([(X[b] ** 2).sum(0) for b in blocks])
    lam, oos_sr_of_best, oos_loss = [], [], []
    N = X.shape[1]
    for comb in itertools.combinations(range(S), S // 2):
        ins = np.zeros(S, bool); ins[list(comb)] = True
        def sr(mask):
            cnt = n[mask].sum(); mu = s1[mask].sum(0) / cnt
            var = s2[mask].sum(0) / cnt - mu ** 2
            return mu / np.sqrt(np.maximum(var, 1e-18))
        sr_in, sr_out = sr(ins), sr(~ins)
        best = int(np.argmax(sr_in))
        rank = (sr_out < sr_out[best]).sum() + 0.5 * ((sr_out == sr_out[best]).sum() - 1)   # 0..N-1
        w = (rank + 1) / (N + 1)
        lam.append(math.log(w / (1 - w)))
        oos_sr_of_best.append(sr_out[best] * math.sqrt(252))
        oos_loss.append(sr_out[best] < 0)
    lam = np.array(lam)
    return {"N": N, "splits": len(lam), "PBO": float((lam <= 0).mean()), "lambda_median": float(np.median(lam)),
            "oos_sharpe_of_is_best_median": float(np.median(oos_sr_of_best)),
            "oos_sharpe_of_is_best_p5_p95": [float(np.percentile(oos_sr_of_best, 5)), float(np.percentile(oos_sr_of_best, 95))],
            "prob_oos_loss": float(np.mean(oos_loss))}


# ---------------------------------------------------------------------------
# DSR
# ---------------------------------------------------------------------------
def n_eff(M: pd.DataFrame) -> float:
    ev = np.linalg.eigvalsh(np.corrcoef(M.to_numpy().T))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def dsr(r: pd.Series, sr_trials: np.ndarray, N: float) -> dict:
    r = r.to_numpy(); T = len(r)
    sr = r.mean() / r.std()
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    V = np.var(sr_trials)
    em = 0.5772156649
    sr0 = math.sqrt(V) * ((1 - em) * norm.ppf(1 - 1 / N) + em * norm.ppf(1 - 1 / (N * math.e))) if N > 1 else 0.0
    denom = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"N": round(N, 1), "SR_annual": round(sr * math.sqrt(252), 3), "SR0_annual": round(sr0 * math.sqrt(252), 3),
            "PSR(SR0=0)": round(float(norm.cdf(sr * math.sqrt(T - 1) / denom)), 4),
            "DSR": round(float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom)), 4),
            "skew": round(float(g3), 2), "kurt": round(float(g4), 1), "T_days": T}


def main() -> None:
    A, B = matrix_A(), matrix_B()
    out = {}
    for tag, M in (("A 参数层（225 个参数配置）", A), ("B 研究层（因子库全部 因子×频率）", B)):
        c = cscv(M)
        out[tag] = c
        print(f"\n{'=' * 90}\n{tag}：N={c['N']}，CSCV 切分 {c['splits']}")
        print(f"  PBO = {c['PBO']:.3f}   λ 中位数 = {c['lambda_median']:+.2f}")
        print(f"  样本内最优在样本外的年化夏普：中位 {c['oos_sharpe_of_is_best_median']:+.2f}  "
              f"[5%~95%: {c['oos_sharpe_of_is_best_p5_p95'][0]:+.2f} ~ {c['oos_sharpe_of_is_best_p5_p95'][1]:+.2f}]   样本外亏损概率 {c['prob_oos_loss']:.3f}")
        print(f"  有效独立试验数 N_eff ≈ {n_eff(M):.1f}")

    sel = A["等权|32|1.5|0.3"]
    n_trials_logged = sum(1 for _ in open("reports/rule_trials.csv", encoding="utf-8")) - 1
    sr_A = (A.mean() / A.std()).to_numpy()
    sr_B = (B.mean() / B.std()).to_numpy()
    all_sr = np.r_[sr_A, sr_B]
    NA, NB = n_eff(A), n_eff(B)
    print(f"\n{'=' * 90}\nDSR：所选策略 = 30MIN 趋势尾部等权组合（chan=32 / 1.5 / 0.3），日度 ATR·今 收益，2009-2026")
    rows = [("只算参数层，N=N_eff(A)", dsr(sel, sr_A, NA)),
            ("只算参数层，N=225（全部当独立）", dsr(sel, sr_A, 225)),
            ("研究层，N=N_eff(B)", dsr(sel, all_sr, NB)),
            (f"全部研究，N=N_eff(A)+N_eff(B)+登记试验{n_trials_logged}", dsr(sel, all_sr, NA + NB + n_trials_logged)),
            ("极端保守，N=225+库内全部+登记试验", dsr(sel, all_sr, 225 + B.shape[1] + n_trials_logged))]
    t = pd.DataFrame([{"情形": k, **v} for k, v in rows]).set_index("情形")
    print(t.to_string())
    out["DSR"] = {k: v for k, v in rows}
    json.dump(out, open("reports/overfitting_tests.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()

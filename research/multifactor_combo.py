# -*- coding: utf-8 -*-
"""
多因子组合（预登记；成分与组合方式只用样本内 2009-2019 信息选择）。

成分池（每个频率）：因子库中支持该频率、且样本内满足
  ① 去漂移 ATR 每笔边际 per_trip_atr_in > 0；② 2009-2019 年逐年 ATR 边际为正的年份 ≥ 60%。
组合方式：
  A 全部成分等权（z 平均后再 MAD_Z）
  B 簇代表等权（样本内因子值 |ρ|≥0.6 聚类，每簇取样本内 per_trip_atr_in 最高者）
  C 策略层等权（各成分独立交易，日度 ATR·今 收益平均；参照，不合成信号）
  D 样本内最好的单因子（对照）
统一执行：迟滞 1.5/0.3 + 换日前平仓。选择：样本内 ATR·今 夏普最高；样本外只跑一次；CSCV + DSR。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from factors.core import atr, params, rolling_mad_zscore, trading_day
from factors.evaluate import SPREAD, SWAP, execution_signal, positions, swap_units
from factors.registry import REGISTRY, get_factors
from research.factor_correlation import factor_values
from research.overfitting_tests import cscv, dsr, matrix_B, n_eff

SPLIT = pd.Timestamp("2020-01-01")


def pool(freq: str) -> dict:
    get_factors()
    out = {}
    for name, spec in REGISTRY.items():
        path = f"reports/factors/{name}.json"
        if freq not in spec.freqs or not os.path.exists(path):
            continue
        m = json.load(open(path, encoding="utf-8"))["results"].get(freq)
        if not m or "atr_edge" not in m:
            continue
        e = m["atr_edge"]
        yrs = {int(k): v for k, v in e["per_trip_atr_year"].items() if int(k) < 2020}
        ratio = np.mean([v > 0 for v in yrs.values()]) if yrs else 0
        if e["per_trip_atr_in"] > 0 and ratio >= 0.6:
            out[name] = {"edge_in": e["per_trip_atr_in"], "year_ratio_in": round(float(ratio), 2)}
    return out


class Book:
    def __init__(self, freq):
        self.freq = freq
        self.df = pd.read_pickle(f"data/cache/{freq}.pkl")
        d = self.df
        self.move = d["open"].shift(-1) - d["open"]
        a = atr(d, params(freq)["atr"])
        self.a_prev = a.shift(1)
        m = self.move / self.a_prev
        self.m_atr = m - m.groupby(d.index.year).transform("mean")
        self.atr_now = float(a[d.index >= d.index[-1] - pd.Timedelta(days=365)].median())
        self.swu = swap_units(d.index)

    def daily(self, z: pd.Series):
        pos = positions(execution_signal(z, self.freq), 1.5, 0.3)
        dpos = pos.diff().abs().fillna(pos.abs())
        sw = pos.abs() * self.swu
        pa = (pos * self.m_atr - dpos * SPREAD / 2 / self.atr_now - sw * SWAP / self.atr_now).fillna(0)
        pu = (pos * self.move - dpos * SPREAD / 2 - sw * SWAP).fillna(0)
        return pa.groupby(trading_day(pa.index)).sum(), pu, pos


def sh(d):
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")


def stats(tag, d, pu, pos):
    ins_d, oos_d = d.index < SPLIT, d.index >= SPLIT
    ins = pu.index < SPLIT
    L, S = (pos * pu.index.to_series().map(lambda _: 1)).where(pos > 0, 0), None
    return {"组合": tag, "夏普内": round(sh(d[ins_d]), 2), "夏普外": round(sh(d[oos_d]), 2), "夏普全期": round(sh(d), 2),
            "美元内": round(pu[ins].sum()), "美元外": round(pu[~ins].sum()),
            "多头外": round(float(pu[~ins][pos[~ins] > 0].sum())), "空头外": round(float(pu[~ins][pos[~ins] < 0].sum())),
            "年正(全期)": round(float((d.groupby(d.index.year).sum() > 0).mean()), 2)}


def main(freq: str):
    P = pool(freq)
    names = list(P)
    B = Book(freq)
    F = factor_values(freq, names)
    # 簇（样本内因子值相关）
    C = F.loc[:"2019"].dropna().rank().corr().fillna(0)
    lab = fcluster(linkage(squareform(1 - C.abs().values, checks=False), "average"), t=0.4, criterion="distance") if len(names) > 1 else [1]
    reps = [max([n for n, l in zip(C.index, lab) if l == c], key=lambda n: P[n]["edge_in"]) for c in sorted(set(lab))]
    print(f"\n{'#' * 100}\n{freq}：成分池 {len(names)} 个 → {len(reps)} 个簇")
    for c in sorted(set(lab)):
        mem = [n for n, l in zip(C.index, lab) if l == c]
        print(f"  簇{c}: " + ", ".join(f"{n}{'★' if n in reps else ''}({P[n]['edge_in']:+.3f})" for n in mem))

    norm = params(freq)["norm"]
    rows, daily = [], {}
    zA = rolling_mad_zscore(F.fillna(0).mean(axis=1), norm)
    zB = rolling_mad_zscore(F[reps].fillna(0).mean(axis=1), norm)
    for tag, z in (("A 全部等权", zA), ("B 簇代表等权", zB)):
        d, pu, pos = B.daily(z); daily[tag] = d; rows.append(stats(tag, d, pu, pos))
    singles = {}
    for n in names:
        d, pu, pos = B.daily(F[n]); singles[n] = (d, pu, pos)
    dC = pd.concat([singles[n][0] for n in names], axis=1).fillna(0).mean(axis=1)
    puC = pd.concat([singles[n][1] for n in names], axis=1).fillna(0).mean(axis=1)
    posC = pd.concat([singles[n][2] for n in names], axis=1).fillna(0).mean(axis=1)
    daily["C 策略层等权"] = dC; rows.append(stats("C 策略层等权", dC, puC, posC))
    best = max(names, key=lambda n: sh(singles[n][0][singles[n][0].index < SPLIT]))
    daily[f"D 单因子 {best}"] = singles[best][0]; rows.append(stats(f"D 单因子 {best}", *singles[best]))
    t = pd.DataFrame(rows).set_index("组合")
    t["选中"] = ["★" if v == t["夏普内"].max() else "" for v in t["夏普内"]]
    print(t.to_string())
    for n in names:
        daily[f"单:{n}"] = singles[n][0]
    M = pd.DataFrame(daily).fillna(0)
    c = cscv(M)
    print(f"  CSCV（4 个组合方式 + {len(names)} 个单因子，共 {M.shape[1]}）：PBO={c['PBO']:.3f}，样本内最优在样本外夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f}")
    return t, daily


if __name__ == "__main__":
    out = {}
    for fq in sys.argv[1:] or ["30MIN", "1H"]:
        out[fq] = main(fq)
    Bm = matrix_B()
    NB = n_eff(Bm)
    n_logged = sum(1 for _ in open("reports/rule_trials.csv", encoding="utf-8")) - 1
    sr_B = (Bm.mean() / Bm.std()).to_numpy()
    print(f"\nDSR（研究层 N_eff≈{NB:.1f}，另报告加上登记试验 {n_logged}）")
    for fq, (t, daily) in out.items():
        sel = t["夏普内"].idxmax()
        d = daily[sel].reindex(Bm.index).fillna(0)
        r1, r2 = dsr(d, sr_B, NB), dsr(d, sr_B, NB + n_logged)
        print(f"  {fq} 样本内选中 {sel}：全期夏普 {r1['SR_annual']:.2f}  DSR@N_eff {r1['DSR']:.3f}  DSR@N_eff+登记 {r2['DSR']:.3f}")

# -*- coding: utf-8 -*-
"""
候选策略稳健性 + 过拟合检验（预登记，检验而非选参），通用版本（任意频率）。

第八批候选（1H，统一执行：迟滞 + 换日前平仓）：
  C1 HighAnchorMomentum 单因子
  C2 HighAnchorMomentum + ContinuousInfoMomentum + QuietTrend 等权（再做 MAD+Z，norm=1000）
默认参数：anchor_days=250，entry=1.5，exit=0.3（先验值）。
网格：anchor_days∈{120,180,250,375,500} × entry∈{1.0,1.25,1.5,1.75,2.0} × exit∈{0,0.3,0.6}（每个候选 75 个）。
输出：参数平原（样本内/外夏普）、walk-forward（扩展窗口选 anchor×entry）、成本敏感性、逐年、多空、CSCV/PBO、DSR。
收益口径："ATR·今"（收益/上一根 ATR，逐年去漂移，成本按最近 1 年 ATR 折算），按交易日（纽约 17:00）汇总。
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from factors.core import atr, params, rolling_mad_zscore, trading_day
from factors.evaluate import SPREAD, SWAP, execution_signal, positions, swap_units
from factors.registry import get_factors
from research.overfitting_tests import cscv, dsr, n_eff

FREQ = "1H"
SPLIT = pd.Timestamp("2020-01-01")
ANCHORS, ENTRIES, EXITS = (120, 180, 250, 375, 500), (1.0, 1.25, 1.5, 1.75, 2.0), (0.0, 0.3, 0.6)
DEFAULT = (250, 1.5, 0.3)

df = pd.read_pickle(f"data/cache/{FREQ}.pkl")
move = df["open"].shift(-1) - df["open"]
a_prev = atr(df, params(FREQ)["atr"]).shift(1)
m_atr = move / a_prev
m_atr = m_atr - m_atr.groupby(df.index.year).transform("mean")
atr_now = float(atr(df, params(FREQ)["atr"])[df.index >= df.index[-1] - pd.Timedelta(days=365)].median())
swu = swap_units(df.index)
TD = trading_day(df.index)
INS_D = None

HA = get_factors(["HighAnchorMomentum"])[0].func
CI = get_factors(["ContinuousInfoMomentum"])[0].func
QT = get_factors(["QuietTrend"])[0].func
_ci, _qt = CI(df, FREQ).fillna(0), QT(df, FREQ).fillna(0)


def signal(cand: str, anchor: int) -> pd.Series:
    ha = HA(df, FREQ, anchor_days=anchor)
    if cand == "C1":
        return ha
    return rolling_mad_zscore((ha.fillna(0) + _ci + _qt) / 3, params(FREQ)["norm"])


def pnl(pos, sx=1.0, wx=1.0):
    dpos = pos.diff().abs().fillna(pos.abs())
    sw = pos.abs() * swu
    pa = (pos * m_atr - dpos * SPREAD * sx / 2 / atr_now - sw * SWAP * wx / atr_now).fillna(0)
    pu = (pos * move - dpos * SPREAD * sx / 2 - sw * SWAP * wx).fillna(0)
    return pa, pu


def daily(x):
    return x.groupby(trading_day(x.index)).sum()


def sh(x):
    d = daily(x)
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")


def main() -> None:
    INS = df.index < SPLIT
    all_daily = {}
    for cand in ("C1", "C2"):
        res = {}
        for an in ANCHORS:
            zx = execution_signal(signal(cand, an), FREQ)
            for e, x in itertools.product(ENTRIES, EXITS):
                res[(an, e, x)] = pnl(positions(zx, e, x))
                all_daily[f"{cand}|{an}|{e}|{x}"] = daily(res[(an, e, x)][0])
        print(f"\n{'#' * 100}\n{cand}")
        for seg, m in (("样本内", INS), ("样本外", ~INS)):
            t = pd.DataFrame({e: {an: sh(res[(an, e, 0.3)][0][m]) for an in ANCHORS} for e in ENTRIES}); t.index.name = "anchor＼entry"
            print(f"\n[1a] 夏普（ATR·今）{seg}，exit=0.3\n" + t.round(2).to_string())
        t = pd.DataFrame({x: {e: f"{sh(res[(250, e, x)][0][INS]):.2f} / {sh(res[(250, e, x)][0][~INS]):.2f}" for e in ENTRIES} for x in EXITS}); t.index.name = "entry＼exit"
        print("\n[1b] 夏普 内/外，anchor=250\n" + t.to_string())
        years = sorted(set(df.index.year)); wa, wu, picks = [], [], []
        for y in years:
            tr = df.index.year < y
            if len(set(df.index.year[tr])) < 3:
                continue
            best = max(itertools.product(ANCHORS, ENTRIES), key=lambda ae: sh(res[(ae[0], ae[1], 0.3)][0][tr]))
            te = df.index.year == y
            wa.append(res[(best[0], best[1], 0.3)][0][te]); wu.append(res[(best[0], best[1], 0.3)][1][te]); picks.append((y, best))
        wa, wu = pd.concat(wa), pd.concat(wu)
        da, du = res[DEFAULT]
        print("\n[2] Walk-forward 每年所选:", " ".join(f"{y}:{a}/{e}" for y, (a, e) in picks))
        for tag, a_, u_ in (("WF", wa, wu), ("固定默认", da.loc[wa.index], du.loc[wa.index])):
            o = a_.index >= SPLIT
            print(f"   {tag:<6} 夏普 全期 {sh(a_):+.2f} | 2020后 {sh(a_[o]):+.2f}   美元 全期 {u_.sum():+.0f} | 2020后 {u_[o].sum():+.0f}")
        pos = positions(execution_signal(signal(cand, 250), FREQ), 1.5, 0.3)
        print("\n[3] 成本敏感性 夏普 内/外 | 美元 内/外")
        for sx, wx in ((1, 1), (2, 1), (3, 1), (2, 2)):
            pa, pu = pnl(pos, sx, wx)
            print(f"   点差×{sx} 过夜×{wx}: {sh(pa[INS]):+.2f} / {sh(pa[~INS]):+.2f} | {pu[INS].sum():+.0f} / {pu[~INS].sum():+.0f}")
        pa, pu = pnl(pos)
        yr = pd.DataFrame({"美元": pu.groupby(df.index.year).sum().round(0), "ATR·今": pa.groupby(df.index.year).sum().round(1)}).T
        print("\n[4] 逐年\n" + yr.to_string())
        L, S = (pos * move)[pos > 0], (pos * move)[pos < 0]
        print(f"   多头毛利 内/外 {L[INS[pos > 0]].sum():+.0f} / {L[~INS[pos > 0]].sum():+.0f}；空头毛利 内/外 {S[INS[pos < 0]].sum():+.0f} / {S[~INS[pos < 0]].sum():+.0f}")
        eq = pu.cumsum(); print(f"   全期夏普 {sh(pa):+.2f}；全期美元 {pu.sum():+.0f}；最大回撤 {(eq - eq.cummax()).min():+.0f}$")

    M = pd.DataFrame(all_daily).fillna(0.0)
    print(f"\n{'#' * 100}\n过拟合检验")
    for cand in ("C1", "C2"):
        sub = M[[c for c in M if c.startswith(cand)]]
        c = cscv(sub)
        print(f"  {cand} 参数层：N={c['N']} N_eff≈{n_eff(sub):.1f} PBO={c['PBO']:.3f} 样本内最优在样本外夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f} "
              f"[5%~95%: {c['oos_sharpe_of_is_best_p5_p95'][0]:+.2f}~{c['oos_sharpe_of_is_best_p5_p95'][1]:+.2f}] 亏损概率 {c['prob_oos_loss']:.3f}")
    return M


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
日内反转诊断（只用样本内 2009-2019 做条件选择，样本外只用于验证）。

核心指标：按信号方向交易时，每笔的平均毛利（美元/盎司）= sign(z) * (open_{t+1+h} - open_{t+1})。
与点差 0.2 对比：只有 均值 >> 0.2 的条件才可能扣费后盈利。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from factors.core import atr, params, relative_volume
from factors.registry import get_factors

SPLIT = pd.Timestamp("2020-01-01")
SIGNALS = ["VolWeightedCloseThrust", "VolumeClimaxReversal", "TrendPullbackLowVolume"]


def fwd_usd(df, h):
    return df["open"].shift(-(1 + h)) - df["open"].shift(-1)


def table(sig, fwd, mask, groups, name):
    """按分组统计：样本数、每笔平均毛利（美元）、胜率。"""
    rows = []
    for g, m in groups.items():
        sel = mask & m & (sig.abs() > 1.5)
        pnl = (np.sign(sig[sel]) * fwd[sel]).dropna()
        if len(pnl) < 200:
            continue
        rows.append({name: g, "n": len(pnl), "毛利$": round(pnl.mean(), 3),
                     "t值": round(pnl.mean() / (pnl.std() / np.sqrt(len(pnl))), 1),
                     "胜率": round((pnl > 0).mean(), 3)})
    return pd.DataFrame(rows).set_index(name)


def main(freq: str) -> None:
    df = pd.read_pickle(f"data/cache/{freq}.pkl")
    p = params(freq)
    ins, oos = df.index < SPLIT, df.index >= SPLIT
    a = atr(df, p["atr"])
    rv = relative_volume(df, p["vol_base"], p["intraday"])
    hour = pd.Series(df.index.hour, index=df.index)
    specs = {s.name: s for s in get_factors(SIGNALS)}
    fac = {n: specs[n](df, freq) * (-1 if n == "VolWeightedCloseThrust" else 1) for n in SIGNALS}

    print(f"\n{'#' * 80}\n{freq}   中位 ATR ${a[ins].median():.2f}（样本内）/ ${a[oos].median():.2f}（样本外）")
    for n, z in fac.items():
        print(f"\n=== {n}{'（取反）' if n == 'VolWeightedCloseThrust' else ''} ===")
        # 1) 持有期 × 信号强度
        rows = []
        for h in (1, 3, 6, 12, 24):
            f = fwd_usd(df, h)
            for lo, hi in ((1.5, 2), (2, 3), (3, 99)):
                sel = ins & (z.abs() > lo) & (z.abs() <= hi)
                pnl = (np.sign(z[sel]) * f[sel]).dropna()
                rows.append({"h": h, "|z|": f"{lo}-{hi}", "n": len(pnl), "毛利$": round(pnl.mean(), 3)})
        print("样本内：持有期 × 信号强度（每笔平均毛利，美元/盎司）")
        print(pd.DataFrame(rows).pivot(index="|z|", columns="h", values="毛利$").to_string())

        f6 = fwd_usd(df, 6)
        # 2) 交易时段（UTC）
        sess = {"亚盘 00-07": (hour < 7), "伦敦 07-12": (hour >= 7) & (hour < 12),
                "伦纽重叠 12-16": (hour >= 12) & (hour < 16), "纽约尾 16-21": (hour >= 16) & (hour < 21),
                "收盘/换日 21-24": (hour >= 21)}
        # 3) 波动率状态：ATR 相对其 1000 根滚动中位数（无未来函数）
        atr_rel = a / a.rolling(1000, min_periods=200).median().shift(1)
        vol_state = {"低波动 <0.8": atr_rel < 0.8, "常态 0.8-1.25": (atr_rel >= 0.8) & (atr_rel < 1.25),
                     "高波动 >=1.25": atr_rel >= 1.25}
        # 4) 量能状态
        vs = {"缩量 <0.8": rv < 0.8, "常态 0.8-2": (rv >= 0.8) & (rv < 2), "放量 >=2": rv >= 2}
        for title, groups, nm in (("时段", sess, "时段"), ("波动率状态", vol_state, "波动"), ("当根量能", vs, "量能")):
            ti = table(z, f6, ins, groups, nm)
            to = table(z, f6, oos, groups, nm)
            print(f"\n按{title}（h=6，|z|>1.5）  样本内 | 样本外")
            print(ti.join(to, lsuffix="_内", rsuffix="_外").to_string())


if __name__ == "__main__":
    for fq in (sys.argv[1:] or ["5MIN", "15MIN", "30MIN"]):
        main(fq)

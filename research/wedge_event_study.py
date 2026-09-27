# -*- coding: utf-8 -*-
"""
三推楔形事件研究：确认后 t+1 开盘入场、持有 h 根，按反转方向计收益（ATR 单位，逐年去漂移）。
与"三推但无动能递减"对照组、与点差成本（0.2$/ATR_t）比较；按强度三分位拆分；样本内/样本外分开。
"""
import numpy as np
import pandas as pd

from factors.core import atr, params
from factors.library.price_action import wedge_events

SPLIT = pd.Timestamp("2020-01-01")
rows = []
for fq in ("5MIN", "15MIN", "30MIN", "1H", "4H", "1D"):
    d = pd.read_pickle(f"data/cache/{fq}.pkl")
    a = atr(d, params(fq)["atr"])
    an = a.to_numpy()
    for h in (3, 6, 12, 24):
        fwd = (d["open"].shift(-(1 + h)) - d["open"].shift(-1)) / a
        ex = (fwd - fwd.groupby(d.index.year).transform("mean")).to_numpy()
        for tag, dim in (("楔形（动能递减）", True), ("对照（无递减）", False)):
            ev = wedge_events(d, an, diminishing=dim)
            t = np.array([e[0] for e in ev]); side = np.array([e[1] for e in ev]); st = np.array([e[2] for e in ev])
            ok = t < len(d) - h - 2
            t, side, st = t[ok], side[ok], st[ok]
            r = side * ex[t]
            cost = 0.2 / an[t]
            ins = d.index[t] < SPLIT
            for seg, m in (("内", ins), ("外", ~ins)):
                rr = r[m]
                if len(rr) < 20:
                    continue
                rows.append({"频率": fq, "h": h, "组": tag, "段": seg, "n": len(rr),
                             "反转收益ATR": round(float(np.nanmean(rr)), 3),
                             "t": round(float(np.nanmean(rr) / (np.nanstd(rr) / np.sqrt(len(rr)))), 2),
                             "胜率": round(float(np.nanmean(rr > 0)), 3),
                             "成本ATR": round(float(np.nanmean(cost[m])), 3),
                             "近3年成本ATR": round(float(np.nanmean(cost[m][d.index[t][m] >= d.index[-1] - pd.Timedelta(days=1095)])) if (d.index[t][m] >= d.index[-1] - pd.Timedelta(days=1095)).any() else float("nan"), 3)})
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for fq in T["频率"].unique():
    x = T[T["频率"] == fq].pivot_table(index=["组", "h"], columns="段", values=["n", "反转收益ATR", "t", "胜率"], sort=False)
    print(f"\n==== {fq}（成本约 {T[T['频率'] == fq]['近3年成本ATR'].iloc[0]} ATR，近 3 年）\n" + x.to_string())
T.to_csv("reports/wedge_event_study.csv", index=False)

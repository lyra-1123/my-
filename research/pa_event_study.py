# -*- coding: utf-8 -*-
"""
价格行为顺势形态事件研究：形态 K 线收盘后，下一根开盘按信号方向入场，持有 h 根；ATR 单位、逐年去漂移。
H2 vs 对照 H1；旗形 vs 对照 无旗杆区间突破。成本 = 最近 1 年 0.2$/ATR 中位数。
"""
import numpy as np
import pandas as pd

from factors.core import atr, params
from factors.library.price_action_cont import flag_events, second_entry_events

SPLIT = pd.Timestamp("2020-01-01")
KINDS = {"H2 第二次入场": lambda d, a: second_entry_events(d, a, 2), "H1 第一次入场（对照）": lambda d, a: second_entry_events(d, a, 1),
         "旗形突破": lambda d, a: flag_events(d, a, True), "无旗杆区间突破（对照）": lambda d, a: flag_events(d, a, False)}
rows = []
for fq in ("5MIN", "15MIN", "30MIN", "1H", "4H"):
    d = pd.read_pickle(f"data/cache/{fq}.pkl")
    a = atr(d, params(fq)["atr"]); an = a.to_numpy()
    cost_now = float((0.2 / a[d.index >= d.index[-1] - pd.Timedelta(days=365)]).median())
    evs = {k: f(d, an) for k, f in KINDS.items()}
    for h in (3, 6, 12, 24):
        fwd = (d["open"].shift(-(1 + h)) - d["open"].shift(-1)) / a
        ex = (fwd - fwd.groupby(d.index.year).transform("mean")).to_numpy()
        for k, ev in evs.items():
            t = np.array([e[0] for e in ev]); side = np.array([e[1] for e in ev])
            ok = t < len(d) - h - 2; t, side = t[ok], side[ok]
            r = side * ex[t]; ins = d.index[t] < SPLIT
            rec3 = d.index[t] >= d.index[-1] - pd.Timedelta(days=1095)
            for seg, m in (("内", ins), ("外", ~ins), ("近3年", rec3)):
                rr = r[m]
                if len(rr) < 15:
                    continue
                rows.append({"频率": fq, "成本": round(cost_now, 3), "形态": k, "h": h, "段": seg, "n": len(rr),
                             "收益ATR": round(float(np.nanmean(rr)), 3), "t": round(float(np.nanmean(rr) / (np.nanstd(rr) / np.sqrt(len(rr)))), 2)})
T = pd.DataFrame(rows)
for fq in T["频率"].unique():
    x = T[T["频率"] == fq].pivot_table(index=["形态", "h"], columns="段", values=["n", "收益ATR", "t"], sort=False)
    x = x.reindex(columns=["内", "外", "近3年"], level=1)
    print(f"\n==== {fq}（当前成本 {T[T['频率'] == fq]['成本'].iloc[0]} ATR）\n" + x.to_string())
T.to_csv("reports/pa_cont_event_study.csv", index=False)

# -*- coding: utf-8 -*-
"""
事件研究：长期上涨趋势中"逢跌买入"是否有超越牛市 beta 的边际。
事件：已收盘日线趋势 T > 0.5（60/120 日波动率缩放动量），且 4H 收盘价距 3 日最高价的回撤 ≥ k×ATR(4H)。
结果：事件后 h 天（4H 频率，每天 6 根）的前瞻收益（t+1 开盘入场），以 ATR 计，逐年减去"同一年所有 K 线的平均前瞻收益"（去漂移）。
对照：上涨趋势中的全部 K 线（不管是否回调）。
"""
import numpy as np
import pandas as pd

from factors.core import atr
from factors.library.trend_reversion import long_trend

df = pd.read_pickle("data/cache/4H.pkl")
a = atr(df, 14)
T = long_trend(df, "4H")
dd = (df["high"].rolling(18, min_periods=18).max() - df["close"]) / a
rows = []
for h_days in (1, 3, 5):
    h = 6 * h_days
    fwd = (df["open"].shift(-(1 + h)) - df["open"].shift(-1)) / a
    ex = fwd - fwd.groupby(df.index.year).transform("mean")                 # 去漂移
    for seg, m in (("2009-2019", df.index.year < 2020), ("2020-", df.index.year >= 2020)):
        up = m & (T > 0.5)
        base = ex[up].dropna()
        rows.append({"持有": f"{h_days}天", "区间": seg, "事件": "上涨趋势·全部K线", "n": len(base),
                     "去漂移收益(ATR)": round(base.mean(), 3), "t": round(base.mean() / base.std() * np.sqrt(len(base) / h), 2)})
        for k in (2, 3, 4):
            ev = ex[up & (dd >= k)].dropna()
            rows.append({"持有": f"{h_days}天", "区间": seg, "事件": f"上涨趋势·回撤≥{k}ATR", "n": len(ev),
                         "去漂移收益(ATR)": round(ev.mean(), 3), "t": round(ev.mean() / ev.std() * np.sqrt(len(ev) / h), 2)})
t = pd.DataFrame(rows)
print(t.pivot_table(index=["事件"], columns=["持有", "区间"], values=["去漂移收益(ATR)"], sort=False).round(3).to_string())
print("\n（t 值已按重叠持有期近似折算：√(n/h)）")
print(t.pivot_table(index=["事件"], columns=["持有", "区间"], values=["t"], sort=False).to_string())

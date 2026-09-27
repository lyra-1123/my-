# -*- coding: utf-8 -*-
"""
家族：趋势中的回调（第三批，2026-09-27）。
由前两批的实证规律推导：
  (1) ≤30MIN 上所有动量/趋势因子在 1~6 根的持有期上 IC 为负（短期反转，年度一致性 ~100%）；
  (2) 同样的趋势因子在 12 根以上持有期、迟滞持仓回测中多空两侧均盈利（中期动量）。
→ 顺中期趋势、逆短期波动入场：在上升趋势中买"缩量回调"，下降趋势中卖"缩量反弹"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, relative_volume, rolling_mad_zscore
from ..registry import register


@register(
    name="TrendPullbackLowVolume",
    cn_name="趋势中缩量回调",
    family="pullback",
    hypothesis="中期趋势由信息缓慢扩散驱动而延续，短期逆势波动多为流动性冲击/获利了结而回吐；"
               "回调时缩量说明没有新的反向信息（只是趋势资金暂歇），顺势入场可同时吃到反转与动量",
    formula="T = MEAN_L∈trend[1:](logret_L/(σ√L))；PB = max(-sign(T)*(C-C.shift(k))/ATR, 0)；"
            "W = 1+max(-TS_MEAN(log RelVol,k),0)；MAD_Z( sign(T)*min(|T|,3)*PB*W )，k=thrust",
    risks=["趋势拐点处'回调'实为反转，逆势加仓扩大亏损", "依赖两套窗口（趋势/回调），参数空间更大",
           "牛市样本中做多回调天然占优，须看空头侧表现"],
    added="2026-09-27",
)
def factor_trend_pullback_low_volume(df: pd.DataFrame, freq: str = "30MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    lc = np.log(d["close"])
    Ls = p["trend"][1:]
    sigma = lc.diff().rolling(max(Ls), min_periods=max(Ls) // 2).std()
    trend = pd.concat([(lc - lc.shift(L)) / (sigma * np.sqrt(L) + EPS) for L in Ls], axis=1).mean(axis=1)
    s = np.sign(trend)

    k = p["thrust"]
    move = (d["close"] - d["close"].shift(k)) / (atr(d, p["atr"]) + EPS)
    pullback = (-s * move).clip(lower=0)                                   # 只取逆趋势方向的位移
    rv = relative_volume(d, p["vol_base"], p["intraday"])
    dry = (-np.log(rv.clip(lower=EPS)).rolling(k, min_periods=k).mean()).clip(lower=0)
    raw = s * trend.abs().clip(upper=3) * pullback * (1.0 + dry)
    return rolling_mad_zscore(raw, p["norm"])

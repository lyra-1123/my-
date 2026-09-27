# -*- coding: utf-8 -*-
"""
家族：日内反转的可交易化（第五批，2026-09-27）。

诊断（research/reversal_diagnostics.py，条件只在样本内 2009-2019 上观察）：
  - 纯反转信号（VWCT 取反、VolumeClimaxReversal）在 |z|>1.5 时每笔毛利只有 0.01~0.05 美元（5MIN），
    远低于 0.2 点差：反转集中在温和区间，而温和区间的价格位移本来就小。
  - 趋势中回调（TrendPullbackLowVolume）的每笔毛利随持有期单调上升：
    15MIN 从 h=6 的约 0.14 美元升到 h=24 的约 0.25 美元，30MIN h=24 约 0.37~0.42 美元。
  → 要覆盖成本：(1) 拉长持仓；(2) 让一次回归的幅度更大（回归到"价值中枢"而不是回吐一根 K 线）；
    (3) 避开换日时段（21-23 UTC 的真实点差远大于 0.2，回测的成本假设在该时段失效）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import (EPS, HTF_MAP, atr, check_input, htf_feature, params, relative_volume,
                    rolling_mad_zscore, vol_scaled_momentum)
from ..registry import register

ROLLOVER_UTC = (21, 23)   # [21:00, 23:00) UTC：美市收盘到换日，流动性最差、点差最宽


def _rollover_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.hour >= ROLLOVER_UTC[0]) & (idx.hour < ROLLOVER_UTC[1])


@register(
    name="PullbackSwing",
    cn_name="趋势回调波段版",
    family="reversal",
    hypothesis="趋势中的缩量回调同时吃到短期反转与中期动量，且毛利随持有期上升（样本内诊断）；"
               "用信号持续性把持仓拉长到数小时，使单笔期望超过点差；换日时段不开新仓",
    formula="E = TrendPullbackLowVolume 原始信号（换日时段置 0）；MAD_Z( EWM(E, halflife=2*thrust) )",
    risks=["持仓更长 → 单笔回撤更大，趋势拐点时亏损放大", "EWM 半衰期是先验设定，需做参数平原检验",
           "与 TrendPullbackLowVolume 高度相关，只能二选一"],
    freqs=("5MIN", "15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_pullback_swing(df: pd.DataFrame, freq: str = "15MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    lc = np.log(d["close"])
    Ls = p["trend"][1:]
    sigma = lc.diff().rolling(max(Ls), min_periods=max(Ls) // 2).std()
    trend = pd.concat([(lc - lc.shift(L)) / (sigma * np.sqrt(L) + EPS) for L in Ls], axis=1).mean(axis=1)
    s = np.sign(trend)
    k = p["thrust"]
    move = (d["close"] - d["close"].shift(k)) / (atr(d, p["atr"]) + EPS)
    pullback = (-s * move).clip(lower=0)
    rv = relative_volume(d, p["vol_base"], p["intraday"])
    dry = (-np.log(rv.clip(lower=EPS)).rolling(k, min_periods=k).mean()).clip(lower=0)
    entry = s * trend.abs().clip(upper=3) * pullback * (1.0 + dry)
    entry[_rollover_mask(d.index)] = 0.0                            # 换日时段不产生新信号
    raw = entry.ewm(halflife=2 * k, adjust=False).mean()            # 信号持续 → 持仓拉长
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="RangeVWAPReversion",
    cn_name="震荡市 VWAP 均值回归",
    family="reversal",
    hypothesis="无趋势（两个已收盘大周期动量都弱）时，做市与区间交易者主导，价格围绕近期成交成本（VWAP）回归；"
               "偏离 VWAP 越远，回归幅度越大，比'回吐一根 K 线'的单笔期望更高；有趋势时回归会被趋势压制，因此不交易",
    formula="dev=(C-VWAP_n)/ATR，n=2*chan；range=clip(1-MAX(|z_h1|,|z_h2|)/1.5,0,1)；"
            "MAD_Z( -dev * range )，换日时段置 0",
    risks=["震荡转趋势的突破初期逆势亏损（没有止损）", "range 门槛 1.5 是先验设定",
           "VWAP 使用 tick volume 加权，数据源敏感"],
    freqs=("5MIN", "15MIN", "30MIN", "1H"),
    added="2026-09-27",
)
def factor_range_vwap_reversion(df: pd.DataFrame, freq: str = "15MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    n = 2 * p["chan"]
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    vwap = (tp * d["volume"]).rolling(n, min_periods=n).sum() / (d["volume"].rolling(n, min_periods=n).sum() + EPS)
    dev = (d["close"] - vwap) / (atr(d, p["atr"]) + EPS)
    h1, h2 = HTF_MAP[freq]
    z1 = htf_feature(d, freq, h1, vol_scaled_momentum).abs()
    z2 = htf_feature(d, freq, h2, vol_scaled_momentum).abs()
    rng = (1.0 - pd.concat([z1, z2], axis=1).max(axis=1) / 1.5).clip(0, 1)   # 1 = 完全无趋势
    raw = -dev * rng
    raw[_rollover_mask(d.index)] = 0.0
    return rolling_mad_zscore(raw, p["norm"])

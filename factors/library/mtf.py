# -*- coding: utf-8 -*-
"""
家族：多周期共振动量（第四批，2026-09-27）。

设计依据（lessons.md）：
  L1 日内单周期动量在 ≤15MIN 呈稳定反转；L3 短期反转与中期动量并存。
  → 单一周期的"动量"混杂了不同持有期参与者的相反行为。异质市场假说（Müller et al. 1997）认为
    不同周期的参与者（日内交易者 / 波段 / CTA）对同一价格路径的反应不同；
    只有当多个周期的趋势方向一致时，各类参与者才同向交易，缺少对手盘，动量才更容易延续。

大周期特征全部通过 core.htf_feature 计算，只使用已收盘的大周期 K 线。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import (EPS, HTF_MAP, atr, check_input, htf_feature, params, relative_volume,
                    rolling_mad_zscore, vol_scaled_momentum)
from ..registry import register


def _htf_trends(d: pd.DataFrame, freq: str) -> tuple[pd.Series, pd.Series]:
    """两个更高周期上的波动率缩放动量（已对齐到基准频率，截断到 ±3）。"""
    h1, h2 = HTF_MAP[freq]
    z1 = htf_feature(d, freq, h1, vol_scaled_momentum).clip(-3, 3)
    z2 = htf_feature(d, freq, h2, vol_scaled_momentum).clip(-3, 3)
    return z1, z2


@register(
    name="MTFTrendResonance",
    cn_name="多周期趋势共振",
    family="mtf",
    hypothesis="基准周期与两个更高周期的波动率缩放动量同向时，日内/波段/趋势资金同向交易、缺少对手盘，"
               "动量延续概率上升；方向分歧时信号按一致度打折（异质市场假说）",
    formula="z_b=VSM(base,trend[0])，z_h1/z_h2=VSM(已收盘HTF,(5,20))；"
            "agree=|MEAN(sign z)|；MAD_Z( MEAN(z_b,z_h1,z_h2) * agree )",
    risks=["三个周期同向时往往已处于趋势后段，追高风险", "HTF 选择（HTF_MAP）是先验设定，存在隐性参数",
           "牛市样本中同向做多占优，需看空头侧"],
    added="2026-09-27",
)
def factor_mtf_trend_resonance(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    L = p["trend"][0]
    lc = np.log(d["close"])
    sigma = lc.diff().rolling(L, min_periods=L // 2).std()
    zb = ((lc - lc.shift(L)) / (sigma * np.sqrt(L) + EPS)).clip(-3, 3)
    z1, z2 = _htf_trends(d, freq)
    comps = pd.concat([zb, z1, z2], axis=1)
    agree = np.sign(comps).mean(axis=1).abs()          # 全同向 = 1；2:1 = 1/3
    raw = comps.mean(axis=1) * agree
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="HTFTrendLTFBreakout",
    cn_name="大周期定向·小周期放量突破",
    family="mtf",
    hypothesis="小周期放量突破单独看是反转（L1），但当两个大周期趋势同向时，突破代表趋势资金在回调结束后重新入场，"
               "而非噪声扫单；只保留与大周期同向的突破",
    formula="gate = sign(z_h1)==sign(z_h2)==sign(brk)；"
            "MAD_Z( EWM( brk_ATR*max(log RelVol,0) * min(|MEAN(z_h1,z_h2)|,3) * gate ) )",
    risks=["信号稀疏，样本量小", "与 VolConfirmedBreakout 同源，改善可能只是过滤掉样本",
           "大周期拐点时的第一次突破会被过滤，错过反转初期"],
    added="2026-09-27",
)
def factor_htf_trend_ltf_breakout(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    a = atr(d, p["atr"])
    hh = d["high"].shift(1).rolling(p["chan"], min_periods=p["chan"]).max()
    ll = d["low"].shift(1).rolling(p["chan"], min_periods=p["chan"]).min()
    brk = ((d["close"] - hh).clip(lower=0) + (d["close"] - ll).clip(upper=0)) / (a + EPS)
    vol_conf = np.log(relative_volume(d, p["vol_base"], p["intraday"]).clip(lower=EPS)).clip(lower=0)

    z1, z2 = _htf_trends(d, freq)
    gate = (np.sign(z1) == np.sign(z2)) & (np.sign(brk) == np.sign(z1))
    strength = ((z1 + z2) / 2).abs().clip(upper=3)
    x = (brk * vol_conf * strength).where(gate, 0.0)
    raw = x.ewm(span=max(2, p["thrust"] // 2), adjust=False).mean()
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="MTFPullbackResonance",
    cn_name="多周期共振下的缩量回调",
    family="mtf",
    hypothesis="TrendPullbackLowVolume 的多周期版：趋势方向改由两个已收盘大周期共同确认，"
               "只在大周期共振时买入小周期缩量回调（卖出缩量反弹），过滤单一周期趋势的误判",
    formula="T=MEAN(z_h1,z_h2)，gate=sign(z_h1)==sign(z_h2)；PB=max(-sign(T)*(C-C.shift(k))/ATR,0)；"
            "W=1+max(-TS_MEAN(log RelVol,k),0)；MAD_Z( sign(T)*min(|T|,3)*PB*W*gate )",
    risks=["大周期拐点时'回调'实为反转", "与 TrendPullbackLowVolume 高度相关，只能择一使用",
           "牛市样本中做多回调天然占优"],
    added="2026-09-27",
)
def factor_mtf_pullback_resonance(df: pd.DataFrame, freq: str = "15MIN", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    z1, z2 = _htf_trends(d, freq)
    T = (z1 + z2) / 2
    gate = np.sign(z1) == np.sign(z2)
    s = np.sign(T)
    k = p["thrust"]
    move = (d["close"] - d["close"].shift(k)) / (atr(d, p["atr"]) + EPS)
    pullback = (-s * move).clip(lower=0)
    rv = relative_volume(d, p["vol_base"], p["intraday"])
    dry = (-np.log(rv.clip(lower=EPS)).rolling(k, min_periods=k).mean()).clip(lower=0)
    raw = (s * T.abs().clip(upper=3) * pullback * (1.0 + dry)).where(gate, 0.0)
    return rolling_mad_zscore(raw, p["norm"])

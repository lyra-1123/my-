# -*- coding: utf-8 -*-
"""
家族：趋势 / 趋势质量（第二批，2026-09-27）。
第一批评估发现日内"放量突破"呈反转、只有 1H 以上有弱动量，
因此这一批从更长的窗口和"趋势质量"角度检验动量是否存在。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, relative_volume, rolling_mad_zscore
from ..registry import register


@register(
    name="TSMomentumVolScaled",
    cn_name="波动率缩放时序动量",
    family="trend",
    hypothesis="时间序列动量（Moskowitz-Ooi-Pedersen 2012）：投资者对信息反应不足 + 趋势资金正反馈，"
               "过去收益按波动率标准化后对未来同向收益有预测力；作为动量类的基准因子",
    formula="MAD_Z(MEAN_L[ log(C/C.shift(L)) / (STD(logret, Lmax)*sqrt(L)) ])，L ∈ 频率预设 trend 三档",
    risks=["趋势反转拐点大幅回撤", "长窗口导致信号迟钝", "与黄金长期上涨漂移高度相关（beta 暴露）"],
    added="2026-09-27",
)
def factor_ts_momentum(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    """
    【算子】对三档回看窗口 L 分别计算"收益 / (σ·√L)"（即近似 t 统计量），取平均。
    【风险】牛市样本内做多占优，必须看多空拆分；多窗口平均是为了降低单一参数的过拟合。
    """
    p = params(freq, **kw)
    d = check_input(df)
    lc = np.log(d["close"])
    sigma = lc.diff().rolling(max(p["trend"]), min_periods=max(p["trend"]) // 2).std()
    parts = [(lc - lc.shift(L)) / (sigma * np.sqrt(L) + EPS) for L in p["trend"]]
    raw = pd.concat(parts, axis=1).mean(axis=1)
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="TrendEfficiencyVolume",
    cn_name="放量趋势效率",
    family="trend",
    hypothesis="Kaufman 效率比（净位移/路径长度）衡量趋势'干净程度'：来回拉锯少说明单边力量持续占优；"
               "叠加季节调整后的放量，区分有资金推动的有效趋势与缩量漂移",
    formula="MAD_Z( (C-C.shift(n))/TS_SUM(|ΔC|,n) * (1 + max(TS_MEAN(log RelVol,n),0)) )",
    risks=["震荡市 ER 趋近 0，信号稀少", "与 TSMomentum 相关", "放量加成在事件日被放大"],
    added="2026-09-27",
)
def factor_trend_efficiency_volume(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    n = p["chan"]
    net = d["close"] - d["close"].shift(n)
    path = d["close"].diff().abs().rolling(n, min_periods=n).sum()
    er = net / (path + EPS)                                            # 有方向的效率比 ∈ [-1, 1]
    rv = relative_volume(d, p["vol_base"], p["intraday"])
    vol_trend = np.log(rv.clip(lower=EPS)).rolling(n, min_periods=n).mean().clip(lower=0)
    raw = er * (1.0 + vol_trend)
    return rolling_mad_zscore(raw, p["norm"])


@register(
    name="VWAPDeviation",
    cn_name="滚动 VWAP 偏离",
    family="trend",
    hypothesis="滚动 VWAP 近似近期参与者的平均持仓成本：价格站上 VWAP 说明近期买方整体浮盈、"
               "卖压（解套盘）轻，处置效应下更容易延续；跌破则反之",
    formula="MAD_Z( (C - TS_SUM(TP*V,n)/TS_SUM(V,n)) / ATR )，TP=(H+L+C)/3，n=2*chan",
    risks=["与均线偏离高度同质", "tick volume 权重失真", "趋势末端偏离最大时反而易反转"],
    added="2026-09-27",
)
def factor_vwap_deviation(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    p = params(freq, **kw)
    d = check_input(df)
    n = 2 * p["chan"]
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    vwap = (tp * d["volume"]).rolling(n, min_periods=n).sum() / (d["volume"].rolling(n, min_periods=n).sum() + EPS)
    raw = (d["close"] - vwap) / (atr(d, p["atr"]) + EPS)
    return rolling_mad_zscore(raw, p["norm"])

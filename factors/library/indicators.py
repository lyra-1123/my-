# -*- coding: utf-8 -*-
"""
家族：经典技术指标筛选（第十批，2026-09-27）。

原则（控制多重检验）：
  - 全部使用教科书默认参数，不做任何调参；方向按教科书解读（超买超卖类 = 反转，趋势类/资金流类 = 动量）；
  - 统一评估后，只有通过 ICIR、去重、与在跑策略相关性、稳健性与过拟合检验的才可能成为候选；
  - 所有指标均只用 <= t 的数据（滚动 / EWM / 递推），最终做 MAD_Z。
频率：15MIN / 30MIN / 1H / 4H。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, rolling_mad_zscore
from ..registry import register

FREQS = ("15MIN", "30MIN", "1H", "4H")


def _ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rsi(c, n=14):
    ch = c.diff()
    return 100 - 100 / (1 + _wilder(ch.clip(lower=0), n) / (_wilder(-ch.clip(upper=0), n) + EPS))


def _psar(h, l, step=0.02, mx=0.2):
    """抛物线 SAR（递推，只用当前及之前的数据）。"""
    n = len(h); sar = np.full(n, np.nan)
    if n < 3:
        return pd.Series(sar, index=h.index)
    H, L = h.to_numpy(), l.to_numpy()
    up, af, ep, s = True, step, H[0], L[0]
    for t in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, L[t - 1], L[t - 2] if t >= 2 else L[t - 1])
            if L[t] < s:
                up, s, ep, af = False, ep, L[t], step
            elif H[t] > ep:
                ep, af = H[t], min(af + step, mx)
        else:
            s = max(s, H[t - 1], H[t - 2] if t >= 2 else H[t - 1])
            if H[t] > s:
                up, s, ep, af = True, ep, H[t], step
            elif L[t] < ep:
                ep, af = L[t], min(af + step, mx)
        sar[t] = s
    return pd.Series(sar, index=h.index)


# 每项：(name, cn_name, 类型, 教科书逻辑, 公式, raw 函数(d, a) -> Series)
SPECS = [
    ("RSIReversion", "RSI 超买超卖", "reversion", "RSI 高于 70 超买、低于 30 超卖，价格回归",
     "−(RSI14−50)/50", lambda d, a: -(_rsi(d["close"]) - 50) / 50),
    ("StochReversion", "随机指标 KD 超买超卖", "reversion", "%D 接近 100 超买、接近 0 超卖",
     "%K=(C−L14)/(H14−L14)·100，%D=SMA3(SMA3(%K))；−(%D−50)/50",
     lambda d, a: -(((d["close"] - d["low"].rolling(14).min()) / (d["high"].rolling(14).max() - d["low"].rolling(14).min() + EPS) * 100)
                    .rolling(3).mean().rolling(3).mean() - 50) / 50),
    ("CCIReversion", "CCI 超买超卖", "reversion", "CCI > +100 超买、< −100 超卖",
     "TP=(H+L+C)/3；CCI=(TP−SMA20)/(0.015·平均绝对偏差20)；−CCI/100",
     lambda d, a: -((lambda tp: (tp - tp.rolling(20).mean()) / (0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean() + EPS))
                    ((d["high"] + d["low"] + d["close"]) / 3)) / 100),
    ("BollingerReversion", "布林带 %B 回归", "reversion", "触及上轨超买、触及下轨超卖，回归中轨",
     "%B=(C−(MA20−2σ))/(4σ)；−(2·%B−1)",
     lambda d, a: -((d["close"] - d["close"].rolling(20).mean()) / (2 * d["close"].rolling(20).std() + EPS))),
    ("MFIReversion", "MFI 资金流超买超卖", "reversion", "成交量加权的 RSI：MFI > 80 超买、< 20 超卖",
     "MF=TP·V；MFI14=100−100/(1+正向MF14/负向MF14)；−(MFI−50)/50",
     lambda d, a: (lambda tp: -((100 - 100 / (1 + (tp * d["volume"]).where(tp.diff() > 0, 0).rolling(14).sum()
                                           / ((tp * d["volume"]).where(tp.diff() < 0, 0).rolling(14).sum() + EPS))) - 50) / 50)
     ((d["high"] + d["low"] + d["close"]) / 3)),
    ("UltimateOscReversion", "终极振荡器超买超卖", "reversion", "综合 7/14/28 周期的买压，> 70 超买、< 30 超卖",
     "BP=C−min(L,C₋₁)，TR=max(H,C₋₁)−min(L,C₋₁)；UO=100·(4A7+2A14+A28)/7；−(UO−50)/50",
     lambda d, a: (lambda bp, tr: -((100 * (4 * bp.rolling(7).sum() / (tr.rolling(7).sum() + EPS) + 2 * bp.rolling(14).sum() / (tr.rolling(14).sum() + EPS)
                                            + bp.rolling(28).sum() / (tr.rolling(28).sum() + EPS)) / 7) - 50) / 50)
     (d["close"] - np.minimum(d["low"], d["close"].shift(1)), np.maximum(d["high"], d["close"].shift(1)) - np.minimum(d["low"], d["close"].shift(1)))),
    ("DPOReversion", "去趋势价格振荡回归", "reversion", "价格相对 n/2+1 根之前的 20 期均线偏离过大后回归",
     "DPO=C−SMA20.shift(11)；−DPO/ATR", lambda d, a: -(d["close"] - d["close"].rolling(20).mean().shift(11)) / (a + EPS)),
    ("MACDMomentum", "MACD 柱动量", "momentum", "MACD 柱为正且扩大代表上涨动能增强",
     "MACD=EMA12−EMA26，信号=EMA9(MACD)；(MACD−信号)/ATR",
     lambda d, a: (lambda m: (m - _ema(m, 9)) / (a + EPS))(_ema(d["close"], 12) - _ema(d["close"], 26))),
    ("TRIXMomentum", "TRIX 动量", "momentum", "三重平滑 EMA 的变化率，过滤噪声后的趋势",
     "TRIX=100·ΔEMA15³/EMA15³；TRIX−EMA9(TRIX)",
     lambda d, a: (lambda e: (lambda tr: tr - _ema(tr, 9))(100 * e.pct_change()))(_ema(_ema(_ema(d["close"], 15), 15), 15))),
    ("ADXTrend", "ADX 趋势强度", "momentum", "ADX 高说明趋势强，方向由 DI+ 与 DI− 决定",
     "DI±=Wilder14(±DM)/ATR14；ADX=Wilder14(|DI+−DI−|/(DI++DI−))；(DI+−DI−)/(DI++DI−)·ADX/25",
     lambda d, a: (lambda up, dn, tr: (lambda dip, dim: (dip - dim) / (dip + dim + EPS) * _wilder((dip - dim).abs() / (dip + dim + EPS), 14) * 100 / 25)
                   (_wilder(up, 14) / (tr + EPS), _wilder(dn, 14) / (tr + EPS)))
     ((d["high"].diff()).where((d["high"].diff() > -d["low"].diff()) & (d["high"].diff() > 0), 0.0),
      (-d["low"].diff()).where((-d["low"].diff() > d["high"].diff()) & (-d["low"].diff() > 0), 0.0), atr(d, 14))),
    ("AroonTrend", "Aroon 趋势", "momentum", "最近 25 期内新高越近越强、新低越近越弱",
     "AroonUp=100·(25−距最高点期数)/25，Down 同理；(Up−Down)/100",
     lambda d, a: ((d["high"].rolling(26).apply(np.argmax, raw=True) - d["low"].rolling(26).apply(np.argmin, raw=True)) / 25)),
    ("IchimokuTrend", "一目均衡表趋势", "momentum", "转换线在基准线之上且价格在云上方为多头",
     "转换=(H9+L9)/2，基准=(H26+L26)/2，云=((转换+基准)/2 与 (H52+L52)/2).shift(26)；(转换−基准)/ATR + sign(C−云中值)",
     lambda d, a: (lambda tk, kj, sb: (tk - kj) / (a + EPS) + np.sign(d["close"] - (((tk + kj) / 2).shift(26) + sb.shift(26)) / 2))
     ((d["high"].rolling(9).max() + d["low"].rolling(9).min()) / 2, (d["high"].rolling(26).max() + d["low"].rolling(26).min()) / 2,
      (d["high"].rolling(52).max() + d["low"].rolling(52).min()) / 2)),
    ("PSARTrend", "抛物线 SAR 趋势", "momentum", "价格在 SAR 之上为多头，距离越远趋势越强",
     "SAR(0.02, 0.2)；(C−SAR)/ATR", lambda d, a: (d["close"] - _psar(d["high"], d["low"])) / (a + EPS)),
    ("KeltnerBreakout", "Keltner 通道突破", "momentum", "价格突破 EMA20 ± 2ATR 通道代表趋势启动",
     "(C−EMA20)/(2·ATR)", lambda d, a: (d["close"] - _ema(d["close"], 20)) / (2 * a + EPS)),
    ("DonchianTrend", "Donchian 通道位置", "momentum", "海龟交易：价格处在 20 期通道上沿为多头",
     "2·(C−L20)/(H20−L20)−1",
     lambda d, a: 2 * (d["close"] - d["low"].rolling(20).min()) / (d["high"].rolling(20).max() - d["low"].rolling(20).min() + EPS) - 1),
    ("VortexTrend", "Vortex 趋势", "momentum", "正向涡流大于负向涡流为多头",
     "VM±=|H−L₋₁|、|L−H₋₁|；VI±=Σ14 VM±/Σ14 TR；VI+−VI−",
     lambda d, a: (lambda tr: ((d["high"] - d["low"].shift(1)).abs().rolling(14).sum() - (d["low"] - d["high"].shift(1)).abs().rolling(14).sum()) / (tr.rolling(14).sum() + EPS))
     (np.maximum(d["high"], d["close"].shift(1)) - np.minimum(d["low"], d["close"].shift(1)))),
    ("OBVMomentum", "OBV 能量潮动量", "momentum", "成交量领先价格：OBV 上升说明资金流入",
     "OBV=Σsign(ΔC)·V；(OBV−OBV₋₂₀)/Σ20 V",
     lambda d, a: (lambda obv: (obv - obv.shift(20)) / (d["volume"].rolling(20).sum() + EPS))((np.sign(d["close"].diff()).fillna(0) * d["volume"]).cumsum())),
    ("CMFFlow", "Chaikin 资金流", "momentum", "收盘靠近高点且放量代表资金累积",
     "MFM=((C−L)−(H−C))/(H−L)；CMF=Σ20(MFM·V)/Σ20 V",
     lambda d, a: (((d["close"] - d["low"]) - (d["high"] - d["close"])) / (d["high"] - d["low"] + EPS) * d["volume"]).rolling(20).sum()
     / (d["volume"].rolling(20).sum() + EPS)),
    ("ForceIndex", "Elder 强力指数", "momentum", "价格变动 × 成交量衡量多空力量",
     "FI=EMA13(ΔC·V)；FI/(ATR·EMA13(V))", lambda d, a: _ema(d["close"].diff() * d["volume"], 13) / (a * _ema(d["volume"], 13) + EPS)),
]


def _make(name, cn, kind, hypo, formula, fn):
    @register(name=name, cn_name=cn, family=f"indicator_{kind}",
              hypothesis=f"教科书解读：{hypo}（默认参数，不调参）", formula=f"MAD_Z( {formula} )",
              risks=["经典指标多为动量/反转的不同写法，易与已有因子高度相关", "默认参数针对日线设计，用于日内频率未必合适",
                     "20 个指标 × 4 个频率同时筛选，多重检验风险高"],
              freqs=FREQS, added="2026-09-27")
    def f(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
        p = params(freq, **kw)
        d = check_input(df)
        raw = fn(d, atr(d, 14)).replace([np.inf, -np.inf], np.nan)
        return rolling_mad_zscore(raw, p["norm"])
    f.__name__ = f"factor_{name}"
    return f


for _s in SPECS:
    globals()[f"factor_{_s[0]}"] = _make(*_s)

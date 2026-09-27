# -*- coding: utf-8 -*-
"""
家族：价格行为学（第十一批，2026-09-27）—— 三推楔形反转（Al Brooks "Three Pushes / Wedge"）。

摆动点必须事后确认：high[i] 为 [i−k, i+k] 内最高 → 摆动高点，在 i+k 根收盘时才可用（代码中统一 shift(k)）。
所有频率使用同一套先验参数：k=3，形态跨度 ≤60 根，规模 ≥2 ATR，信号有效期 12 根。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, rolling_mad_zscore
from ..registry import register

K, SPAN, MIN_SIZE_ATR, WINDOW = 3, 60, 2.0, 12


def confirmed_pivots(d: pd.DataFrame, k: int = K):
    """返回 (高点确认列表, 低点确认列表)：[(确认位置 t, 摆动点位置 i, 价格)]，t = i + k。"""
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    hmax = d["high"].rolling(2 * k + 1, center=True).max().to_numpy()
    lmin = d["low"].rolling(2 * k + 1, center=True).min().to_numpy()
    his = [(i + k, i, h[i]) for i in np.flatnonzero(h == hmax) if i + k < len(h)]
    los = [(i + k, i, l[i]) for i in np.flatnonzero(l == lmin) if i + k < len(l)]
    return his, los


def wedge_events(d: pd.DataFrame, a: np.ndarray, diminishing: bool = True, k: int = K):
    """
    三推事件：[(确认位置 t, 方向 −1/+1, 强度, 失效价)]。
    顶部（方向 −1）：H1<H2<H3、L1<L2（两推之间的最低点）、动能递减 (H3−H2)<(H2−H1)（diminishing=False 时要求 ≥，作对照）、
    (H3−L1)≥2ATR、跨度 ≤ SPAN。底部对称。只用确认时刻 t 之前（含）的数据。
    """
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    his, los = confirmed_pivots(d, k)
    ev = []
    for pts, side in ((his, -1), (los, +1)):
        for j in range(2, len(pts)):
            (t, i3, p3), (_, i2, p2), (_, i1, p1) = pts[j], pts[j - 1], pts[j - 2]
            if i3 - i1 > SPAN or i2 - i1 < 2 or i3 - i2 < 2:
                continue
            at = a[t]
            if not np.isfinite(at) or at <= 0:
                continue
            if side == -1:
                if not (p1 < p2 < p3):
                    continue
                q1, q2 = l[i1:i2].min(), l[i2:i3].min()          # 两推之间的回调低点
                if not q1 < q2:
                    continue
                push2, push3, size = p2 - p1, p3 - p2, p3 - q1
            else:
                if not (p1 > p2 > p3):
                    continue
                q1, q2 = h[i1:i2].max(), h[i2:i3].max()
                if not q1 > q2:
                    continue
                push2, push3, size = p1 - p2, p2 - p3, q1 - p3
            if size < MIN_SIZE_ATR * at:
                continue
            decay = 1 - push3 / (push2 + EPS)
            if diminishing and decay <= 0:
                continue
            if not diminishing and decay > 0:
                continue
            strength = (abs(decay) if diminishing else 1.0) * min(size / at / 2, 3)
            ev.append((t, side, strength, p3))
    return sorted(ev)


def events_to_signal(d: pd.DataFrame, ev, window: int = WINDOW) -> pd.Series:
    """事件在确认后 window 根内有效；期间若价格重新突破第三推极值（形态失败）则作废。"""
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    out = np.zeros(len(d))
    for t, side, s, p3 in ev:
        for u in range(t, min(t + window, len(d))):
            if (side == -1 and h[u] > p3) or (side == 1 and l[u] < p3):
                break
            out[u] = side * s
    return pd.Series(out, index=d.index)


def sparse_mad_z(x: pd.Series, n_obs: int = 1000) -> pd.Series:
    nz = x[x != 0]
    z = pd.Series(0.0, index=x.index)
    if len(nz):
        z[nz.index] = rolling_mad_zscore(nz, n_obs).to_numpy()
    return z.fillna(0.0)


@register(
    name="ThreePushWedgeReversal",
    cn_name="三推楔形反转",
    family="price_action",
    hypothesis="三次推动创新高但力度递减（动能背离）：追价资金耗尽、早期多头了结、逆势者在前高附近布空，上方失去新增买盘 → 反转；底部对称",
    formula="摆动点：前后各 3 根极值（3 根后确认）；顶部：H1<H2<H3、回调低点 L1<L2、(H3−H2)<(H2−H1)、(H3−L1)≥2ATR、跨度≤60 根；"
            "强度=(1−推3/推2)·min(规模/2ATR,3)；确认后 12 根内有效，突破 H3 作废；只在非零观测上 MAD_Z",
    risks=["形态主观性：摆动点参数（k=3）决定识别结果", "强趋势中'三推'之后往往继续推（与 L11 尾部延续相冲突）",
           "事件稀少，高周期（4H/1D）样本量不足", "确认滞后 3 根，入场时已错过部分反转"],
    freqs=("5MIN", "15MIN", "30MIN", "1H", "4H", "1D"),
    added="2026-09-27",
)
def factor_three_push_wedge(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    d = check_input(df)
    a = atr(d, params(freq, **kw)["atr"]).to_numpy()
    return sparse_mad_z(events_to_signal(d, wedge_events(d, a, diminishing=True)))


@register(
    name="ThreePushNoDecay",
    cn_name="三推（无动能递减，对照）",
    family="price_action",
    hypothesis="对照组：三推创新高但第三推不弱于第二推（没有动能背离）；若与 ThreePushWedgeReversal 表现相同，说明'动能递减'条件没有信息量",
    formula="同三推楔形，但要求 (H3−H2) ≥ (H2−H1)；方向同样按反转处理；强度=min(规模/2ATR,3)",
    risks=["仅作对照，不作为候选"],
    freqs=("5MIN", "15MIN", "30MIN", "1H", "4H", "1D"),
    added="2026-09-27",
)
def factor_three_push_no_decay(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    d = check_input(df)
    a = atr(d, params(freq, **kw)["atr"]).to_numpy()
    return sparse_mad_z(events_to_signal(d, wedge_events(d, a, diminishing=False)))

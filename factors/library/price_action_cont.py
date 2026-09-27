# -*- coding: utf-8 -*-
"""
家族：价格行为学 · 顺势延续形态（第十二批，2026-09-27）。
  H2/L2 第二次入场（Al Brooks）及对照 H1/L1；旗形突破及对照"无旗杆的区间突破"。
所有频率使用同一套先验参数；信号在形态 K 线收盘时产生（只用 <= t 的数据），有效期 6 根，失败价被触及即作废。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params
from ..registry import register
from .price_action import events_to_signal, sparse_mad_z

FREQS = ("5MIN", "15MIN", "30MIN", "1H", "4H")
WIN = 6
POLE, FLAG, POLE_ATR, POLE_ER, NOPOLE_ATR = 8, 8, 3.0, 0.6, 1.0


def second_entry_events(d: pd.DataFrame, a: np.ndarray, which: int = 2):
    """H{which}/L{which} 事件：[(t, 方向, 强度, 失败价)]。"""
    h, l, c = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    ema = d["close"].ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    n = len(h); ev = []
    for side in (1, -1):
        ext, cnt, since, pb_ext = np.nan, 0, 0, np.nan
        for t in range(25, n):
            at = a[t]
            if not np.isfinite(at) or at <= 0 or not np.isfinite(ema[t - 5]):
                continue
            trending = side * (ema[t] - ema[t - 5]) > 0.2 * at
            if not trending:
                ext, cnt, since, pb_ext = np.nan, 0, 0, np.nan
                continue
            hi, lo = (h[t], l[t]) if side == 1 else (-l[t], -h[t])
            prev_hi = h[t - 1] if side == 1 else -l[t - 1]
            if np.isnan(ext) or hi > ext:                       # 创新高（下跌趋势为创新低）→ 重置
                ext, cnt, since, pb_ext = hi, 0, 0, lo
                continue
            since += 1
            pb_ext = min(pb_ext, lo)
            depth = ext - pb_ext
            if since >= 3 and depth >= 0.5 * at and hi > prev_hi:
                cnt += 1
                if cnt == which:
                    strength = min(side * (ema[t] - ema[t - 5]) / at, 3)
                    fail = pb_ext if side == 1 else -pb_ext
                    ev.append((t, side, strength, fail))
    return sorted(ev)


def flag_events(d: pd.DataFrame, a: np.ndarray, with_pole: bool = True):
    """旗形突破（with_pole=True）或无旗杆的区间突破（对照）。"""
    h, l, c = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    n = len(c); ev = []
    for t in range(POLE + FLAG + 2, n):
        at = a[t]
        if not np.isfinite(at) or at <= 0:
            continue
        f0, f1 = t - FLAG, t                                  # 旗面 [t−8, t−1]
        p0, p1 = f0 - POLE, f0                                # 旗杆 [t−16, t−9]
        pole = c[p1 - 1] - c[p0 - 1]
        fh, fl = h[f0:f1].max(), l[f0:f1].min()
        for side in (1, -1):
            mv = side * pole
            if with_pole:
                path = np.abs(np.diff(c[p0 - 1:p1])).sum()
                if mv < POLE_ATR * at or mv / (path + EPS) < POLE_ER:
                    continue
                ref = abs(pole)
            else:
                if abs(pole) >= NOPOLE_ATR * at:
                    continue
                ref = POLE_ATR * at                           # 用同样的"旗面 ≤ 1.5 ATR"标准
            if fh - fl > 0.5 * ref:
                continue
            if side == 1 and (fl < c[p1 - 1] - 0.5 * ref or c[t] <= fh):
                continue
            if side == -1 and (fh > c[p1 - 1] + 0.5 * ref or c[t] >= fl):
                continue
            strength = min(ref / at / 3, 3) if with_pole else 1.0
            ev.append((t, side, strength, fl if side == 1 else fh))
    return ev


def _factor(events_fn):
    def f(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
        d = check_input(df)
        a = atr(d, params(freq, **kw)["atr"]).to_numpy()
        return sparse_mad_z(events_to_signal(d, events_fn(d, a), WIN))
    return f


_COMMON_RISK = ["形态识别依赖先验参数", "信号在形态 K 线收盘后于下一根开盘成交，实际（突破挂单）入场价更好/更差都有可能",
                "延续形态与 TT30/HA1H 等趋势策略可能相关"]

register(name="SecondEntryH2", cn_name="第二次入场 H2/L2", family="price_action_cont",
         hypothesis="上升趋势中回调的第一次恢复（H1）常失败，洗出过早入场者；第二次恢复（H2）时回调卖盘耗尽、趋势资金重新入场，成功率更高（Brooks）；下跌趋势 L2 对称",
         formula="EMA20 5 根内上升 >0.2ATR；回调：未创新高、回撤 ≥0.5ATR、≥3 根；回调中'最高价>上一根最高价'第 2 次出现 → H2；6 根内有效，跌破回调低点作废",
         risks=_COMMON_RISK, freqs=FREQS, added="2026-09-27")(_factor(lambda d, a: second_entry_events(d, a, 2)))
register(name="FirstEntryH1", cn_name="第一次入场 H1/L1（对照）", family="price_action_cont",
         hypothesis="对照组：回调中第 1 次恢复尝试即入场；用于检验'等待第二次'是否有信息量",
         formula="同 H2，但计数到第 1 次即发出信号", risks=["仅作对照"], freqs=FREQS, added="2026-09-27")(
    _factor(lambda d, a: second_entry_events(d, a, 1)))
register(name="FlagBreakout", cn_name="旗形突破", family="price_action_cont",
         hypothesis="急速单边走势（旗杆）代表新信息/大资金进场，随后的窄幅整理（旗面）是换手，突破旗面意味着换手完成、趋势延续",
         formula="旗杆：前 16~9 根净涨 ≥3ATR 且效率比 ≥0.6；旗面：最近 8 根振幅 ≤ 旗杆 50%、回撤 ≤50%；收盘突破旗面高点 → 做多；6 根内有效，跌破旗面低点作废；熊旗对称",
         risks=_COMMON_RISK, freqs=FREQS, added="2026-09-27")(_factor(lambda d, a: flag_events(d, a, True)))
register(name="RangeBreakoutNoPole", cn_name="无旗杆的区间突破（对照）", family="price_action_cont",
         hypothesis="对照组：整理与突破条件相同，但前面没有急速走势（旗杆净变动 <1ATR）；用于检验'旗杆'是否有信息量",
         formula="旗面振幅 ≤1.5ATR、回撤约束同旗形；前 8 根净变动 <1ATR；收盘突破整理区间",
         risks=["仅作对照"], freqs=FREQS, added="2026-09-27")(_factor(lambda d, a: flag_events(d, a, False)))

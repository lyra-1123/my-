# -*- coding: utf-8 -*-
"""
非统一的执行规则（供模拟盘引擎使用）。统一规则（迟滞 + 换日前平仓）仍在 factors.evaluate 中实现。

state_trail：每日照样在执行窗口内平仓；窗口外只要迟滞状态仍在（未出现 |z|<exit 或反向）就按原方向持有/回补；
             叠加移动止损：从本段持仓（连续迟滞状态）的最有利价格回撤 trail×日线 ATR(14)（已收盘日线）即出场，
             盘中按最高/最低价触发，跳空按开盘价成交；止损后同方向须等迟滞状态复位才能再入场。
与 research/ha1h_exit_rules.py 的 simulate(mode="state", exit_th, trail) 逐根一致（见 paper/selftest.py）。
"""
from __future__ import annotations

import numpy as np


def state_trail(z, o, h, l, flat, atrd, entry=1.5, exit_th=0.3, trail=2.0):
    """
    返回：pos_open（第 t 根开盘时持仓）、pnl（该根价格差，1 盎司）、dpos（第 t 根开盘时的仓位变动量，计点差）、
          held_end（该根收盘后仍持仓，计过夜费）、stop_fill（该根盘中止损成交价，否则 NaN）、
          next_stop（第 t 根收盘后为下一根设置的止损线，否则 NaN）、final_pos（最后一根收盘后的目标仓位）。
    """
    n = len(z)
    pos_open = np.zeros(n); pnl = np.zeros(n); dpos = np.zeros(n); held_end = np.zeros(n)
    stop_fill = np.full(n, np.nan); next_stop = np.full(n, np.nan)
    pos, hs, block, best, lvl = 0.0, 0.0, 0.0, 0.0, np.nan
    for t in range(n):
        pos_open[t] = pos
        nxt = o[t + 1] if t + 1 < n else np.nan
        stopped = False
        if pos != 0 and not np.isnan(lvl):
            if pos > 0 and l[t] <= lvl:
                px = min(o[t], lvl); stopped = True
            elif pos < 0 and h[t] >= lvl:
                px = max(o[t], lvl); stopped = True
        if stopped:
            pnl[t] = pos * (px - o[t]); stop_fill[t] = px
            if t + 1 < n:
                dpos[t + 1] += abs(pos)
            block, pos = pos, 0.0
        else:
            pnl[t] = pos * (nxt - o[t]) if t + 1 < n else 0.0
            held_end[t] = abs(pos)
            if pos > 0:
                best = max(best, h[t])
            elif pos < 0:
                best = min(best, l[t])
        zt = z[t]
        if zt > entry:
            hs_new = 1.0
        elif zt < -entry:
            hs_new = -1.0
        elif (exit_th == 0 and hs != 0 and zt * hs <= 0) or (exit_th > 0 and abs(zt) < exit_th):
            hs_new = 0.0
        else:
            hs_new = hs
        if hs_new != hs:
            block = 0.0 if hs_new != block else block
            if hs_new == 0:
                block = 0.0
        episode_start = hs_new != 0 and hs_new != hs
        hs = hs_new
        tgt = 0.0 if flat[t] else hs
        if tgt != 0 and tgt == block:
            tgt = 0.0
        if tgt != pos:
            if t + 1 < n:
                dpos[t + 1] += abs(tgt - pos)
            if tgt != 0 and (pos == 0 or episode_start or np.sign(tgt) != np.sign(pos)):
                if episode_start or best == 0 or np.sign(tgt) != np.sign(pos):
                    best = nxt
            pos = tgt
        if episode_start and pos != 0:
            best = nxt
        lvl = (best - pos * trail * atrd[t]) if pos != 0 else np.nan
        next_stop[t] = lvl
    return pos_open, pnl, dpos, held_end, stop_fill, next_stop, pos

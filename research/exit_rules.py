# -*- coding: utf-8 -*-
"""
出场规则检验（预登记）：TT30-EW-v1 的入场不变，只改出场。结果作为 v2 候选，不改动在跑的 v1。

候选（运行前写定）：
  E0 现行：信号出场（|z|<0.3）
  E1 信号出场 + ATR 移动止损：止损线 = 持仓期间最有利价格 ∓ k×ATR，k∈{2,3,4}
  E2 信号出场 + 固定止损：入场价 ∓ k×ATR（入场时的 ATR），k∈{1.5,2.5}
  E3 只用移动止损出场（不看信号消退），k∈{3,4}；反向极端信号仍然反手
  E4 信号出场 + 固定止盈：入场价 ± k×ATR，k∈{3,5}
共同规则：
  - 换日前平仓不变；止损/止盈后同方向需等 |z|<0.3 后再次突破 1.5 才能重新入场
  - 盘中触发：止损线/止盈线只用上一根收盘时的信息计算；本根最低（最高）价触及即按该价成交，
    跳空越过则按开盘价成交；同一根内止损与止盈都触及时按止损处理（保守）
  - 最有利价格在每根 K 线收盘后更新（含入场那根）；ATR = ATR(32)，用上一根收盘时的值
选择：样本内（2009-2019）ATR·今 口径夏普最高，且必须优于 E0；样本外只跑一次；10 个规则做 CSCV（PBO）。
"""
from __future__ import annotations

import csv
from datetime import date

import numpy as np
import pandas as pd

from factors.core import atr, trading_day
from factors.evaluate import ENTRY, EXIT, SPREAD, SWAP, execution_signal, swap_units
from research.overfitting_tests import cscv
from research.rule_adaptation import TRIALS_CSV
from research.trend_tail_validation import FREQ, df, factor_z

SPLIT = pd.Timestamp("2020-01-01")
RULES = [("E0 信号出场", dict()),
         *[(f"E1 信号+移动止损 k={k}", dict(trail=k)) for k in (2, 3, 4)],
         *[(f"E2 信号+固定止损 k={k}", dict(stop=k)) for k in (1.5, 2.5)],
         *[(f"E3 只用移动止损 k={k}", dict(trail=k, no_signal_exit=True)) for k in (3, 4)],
         *[(f"E4 信号+固定止盈 k={k}", dict(tp=k)) for k in (3, 5)]]


def simulate(z, o, h, l, a, trail=None, stop=None, tp=None, no_signal_exit=False):
    """
    逐根模拟。返回每根 K 线：持仓（开盘时）、毛利（价格差，1 盎司）、仓位变动量（算点差）、是否持仓过换日。
    约定：pos[t] 为第 t 根开盘时的持仓；target 在第 t 根收盘时决定，t+1 开盘执行。
    """
    n = len(z)
    pos_open = np.zeros(n); pnl = np.zeros(n); dpos = np.zeros(n); held_end = np.zeros(n)
    pos, entry_px, entry_atr, best = 0.0, 0.0, 0.0, 0.0
    level_stop = level_tp = np.nan
    block = 0.0                                          # 被止损/止盈的方向，等信号复位才解除
    for t in range(n):
        # ---- 1) 开盘：执行上一根收盘时的目标（在循环末尾已写入 pos 与 dpos[t]）
        pos_open[t] = pos
        # ---- 2) 盘中：检查止损/止盈（只对开盘就持有的仓位）
        exit_px = None
        if pos != 0:
            if pos > 0:
                if not np.isnan(level_stop) and l[t] <= level_stop:
                    exit_px = min(o[t], level_stop)
                elif not np.isnan(level_tp) and h[t] >= level_tp:
                    exit_px = max(o[t], level_tp)
            else:
                if not np.isnan(level_stop) and h[t] >= level_stop:
                    exit_px = max(o[t], level_stop)
                elif not np.isnan(level_tp) and l[t] <= level_tp:
                    exit_px = min(o[t], level_tp)
        nxt_open = o[t + 1] if t + 1 < n else np.nan
        if exit_px is not None:
            pnl[t] = pos * (exit_px - o[t])
            if t + 1 < n:
                dpos[t + 1] += abs(pos)                  # 平仓点差记在下一根（与开盘成交同一口径）
            block, pos = pos, 0.0
            held_end[t] = 0.0
        else:
            pnl[t] = pos * (nxt_open - o[t]) if t + 1 < n else 0.0
            held_end[t] = abs(pos)
            if pos > 0:
                best = max(best, h[t])
            elif pos < 0:
                best = min(best, l[t])
        # ---- 3) 收盘：决定下一根的目标仓位
        zt = z[t]
        s = 1.0 if zt > ENTRY else (-1.0 if zt < -ENTRY else 0.0)
        if abs(zt) < EXIT:
            block = 0.0
        tgt = pos
        if s != 0 and s != pos and s != block:
            tgt = s
        elif pos != 0 and abs(zt) < EXIT and not no_signal_exit:
            tgt = 0.0
        elif pos != 0 and no_signal_exit and zt == 0.0:          # 换日窗口（z 被置 0）仍然平仓
            tgt = 0.0
        if tgt != pos:
            if t + 1 < n:
                dpos[t + 1] += abs(tgt - pos)
            if tgt != 0:
                entry_px, entry_atr = nxt_open, a[t]
                best = nxt_open
            pos = tgt
        # ---- 4) 为下一根计算止损/止盈线（只用到本根收盘为止的信息）
        level_stop = level_tp = np.nan
        if pos != 0:
            if trail is not None:
                level_stop = best - pos * trail * a[t]
            if stop is not None:
                s_lvl = entry_px - pos * stop * entry_atr
                level_stop = s_lvl if np.isnan(level_stop) else (max(level_stop, s_lvl) if pos > 0 else min(level_stop, s_lvl))
            if tp is not None:
                level_tp = entry_px + pos * tp * entry_atr
    return pos_open, pnl, dpos, held_end


def main() -> None:
    z = np.nan_to_num(execution_signal(factor_z("等权", 32), FREQ).to_numpy())
    o, h, l = (df[c].to_numpy() for c in ("open", "high", "low"))
    a_s = atr(df, 32)
    a = a_s.to_numpy()
    a_prev = a_s.shift(1).to_numpy()
    yr = df.index.year
    atr_now = float(a_s[df.index >= df.index[-1] - pd.Timedelta(days=365)].median())
    drift = pd.Series((df["open"].shift(-1) - df["open"]).to_numpy() / a_prev, index=df.index).groupby(yr).transform("mean").to_numpy()
    swu = swap_units(df.index).to_numpy()
    td = trading_day(df.index)
    ins = df.index < SPLIT

    rows, daily = [], {}
    for name, kw in RULES:
        pos_open, pnl, dpos, held_end = simulate(z, o, h, l, np.nan_to_num(a), **kw)
        swap_cost = held_end * swu * SWAP
        usd = pnl - dpos * SPREAD / 2 - swap_cost
        r = pnl / a_prev - pos_open * drift - dpos * SPREAD / 2 / atr_now - swap_cost / atr_now
        r = pd.Series(np.nan_to_num(r), index=df.index)
        d = r.groupby(td).sum()
        daily[name] = d
        u = pd.Series(usd, index=df.index)
        # 逐笔：以持仓段切分
        p = pd.Series(pos_open, index=df.index)
        seg = ((p != p.shift(1)) | (dpos > 0)).cumsum()
        tr = u[p != 0].groupby(seg[p != 0]).sum()
        def sh(x): return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")
        tri, tro = tr[tr.index.isin(seg[ins & (p != 0)])], tr[tr.index.isin(seg[~ins & (p != 0)])]
        eq = u[~ins].cumsum()
        rows.append({"规则": name, "夏普内": round(sh(d[d.index < SPLIT]), 2), "夏普外": round(sh(d[d.index >= SPLIT]), 2),
                     "美元内": round(u[ins].sum()), "美元外": round(u[~ins].sum()),
                     "笔数外": len(tro), "胜率外": round(float((tro > 0).mean()), 2),
                     "均盈/均亏外": f"{tro[tro > 0].mean():+.1f}/{tro[tro <= 0].mean():+.1f}",
                     "最大单笔亏外": round(float(tro.min()), 1), "回撤外$": round(float((eq - eq.cummax()).min())),
                     "偏度外": round(float(d[d.index >= SPLIT].skew()), 2)})
    t = pd.DataFrame(rows).set_index("规则")
    base_in = t.loc["E0 信号出场", "夏普内"]
    cand = t[(t.index != "E0 信号出场") & (t["夏普内"] > base_in)]
    sel = cand["夏普内"].idxmax() if len(cand) else "E0 信号出场"
    t["选中"] = ["★" if i == sel else "" for i in t.index]
    print(t.to_string())
    print(f"\n样本内选中：{sel}（E0 样本内夏普 {base_in}）")
    M = pd.DataFrame(daily).fillna(0.0)
    c = cscv(M)
    print(f"10 个出场规则的 CSCV：PBO = {c['PBO']:.3f}；样本内最优在样本外的夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f}")
    t.to_csv("reports/exit_rules.csv")
    with open(TRIALS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for name, r in t.iterrows():
            w.writerow([date.today(), "TT30-EW 出场规则", FREQ, "trend-tail", "exit", name, "IS sharpe_atr_now (must beat E0)",
                        r["夏普内"], r["美元内"], r["夏普外"], r["美元外"], "", len(t), ("SELECTED " if name == sel else "") + name])


if __name__ == "__main__":
    main()

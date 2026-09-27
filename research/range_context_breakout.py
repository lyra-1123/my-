# -*- coding: utf-8 -*-
"""
区间背景突破（RangeContextBreakout）——预登记检验。线索来自 reports/context_pattern_study.md（同一份数据，
因此本脚本只能检验稳健性与一致性，独立验证只能靠前向测试）。

默认定义：背景 = 前 60 根 ER≤0.15 且 20 根 K 线重叠度≥0.5；突破 = 最近 FLAG=8 根振幅≤1.5ATR、此前 8 根净变动<1ATR、
本根收盘突破 8 根高/低点；方向 = 突破方向。
事件驱动回测：下一根开盘入场、持有 H=24 根（≤1H 遇换日前平仓窗口提前在窗口首根开盘平仓），单一仓位不叠加；
成本：点差 0.2$ + 过夜费 0.47$（周三 ×3）。ATR·今 口径：收益/入场时 ATR、逐年去漂移、成本/最近 1 年 ATR。
网格：ER{0.10,0.15,0.20} × 重叠{0.4,0.5,0.6} × FLAG{6,8,12} × H{12,24,36}（81 个）→ 平原 + CSCV。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from factors.core import atr, params, trading_day
from factors.evaluate import BAR, SPREAD, SWAP, swap_units
from research.overfitting_tests import cscv, n_eff

SPLIT = pd.Timestamp("2020-01-01")
DEFAULT = dict(er=0.15, ov=0.5, flag=8, h=24)


class Data:
    def __init__(self, freq):
        self.freq = freq
        d = pd.read_pickle(f"data/cache/{freq}.pkl")
        self.d = d
        self.o, self.h, self.l, self.c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        a = atr(d, params(freq)["atr"]); self.a = a.to_numpy()
        self.atr_now = float(a[d.index >= d.index[-1] - pd.Timedelta(days=365)].median())
        c = d["close"]
        self.er = ((c - c.shift(60)).abs() / (c.diff().abs().rolling(60, min_periods=60).sum() + 1e-12)).to_numpy()
        inter = (np.minimum(d["high"], d["high"].shift(1)) - np.maximum(d["low"], d["low"].shift(1))).clip(lower=0)
        union = np.maximum(d["high"], d["high"].shift(1)) - np.minimum(d["low"], d["low"].shift(1))
        self.ov = (inter / (union + 1e-12)).rolling(20, min_periods=20).mean().to_numpy()
        ny = d.index.tz_localize("UTC").tz_convert("America/New_York")
        mins = ny.hour * 60 + ny.minute
        start = 17 * 60 - 2 * pd.Timedelta(BAR[freq]).seconds // 60
        self.flat = np.asarray((mins >= start) & (mins < 18 * 60)) if freq in ("5MIN", "15MIN", "30MIN", "1H") else np.zeros(len(d), bool)
        self.swu = swap_units(d.index).to_numpy()
        m = (d["open"].shift(-1) - d["open"]) / a.shift(1)
        self.drift_bar = m.groupby(d.index.year).transform("mean").to_numpy()
        self.td = trading_day(d.index)

    def events(self, er, ov, flag):
        c, h, l, a = self.c, self.h, self.l, self.a
        ev = []
        for t in range(2 * flag + 61, len(c)):
            at = a[t]
            if not np.isfinite(at) or at <= 0 or not (self.er[t] <= er and self.ov[t] >= ov):
                continue
            f0 = t - flag
            fh, fl = h[f0:t].max(), l[f0:t].min()
            if fh - fl > 1.5 * at or abs(c[f0 - 1] - c[f0 - 1 - flag]) >= 1.0 * at:
                continue
            if c[t] > fh:
                ev.append((t, 1))
            elif c[t] < fl:
                ev.append((t, -1))
        return ev

    def backtest(self, ev, hold):
        n = len(self.o); busy = -1; trades = []
        for t, side in ev:
            e = t + 1
            if e <= busy or e >= n - 1 or self.flat[e]:
                continue
            x = min(e + hold, n - 1)
            fl = np.flatnonzero(self.flat[e + 1:x + 1])
            if len(fl):
                x = e + 1 + fl[0]
            bars = x - e
            move = side * (self.o[x] - self.o[e])
            swaps = self.swu[e:x].sum() * SWAP
            usd = move - SPREAD - swaps
            r = move / self.a[e - 1] - side * np.nansum(self.drift_bar[e:x]) - (SPREAD + swaps) / self.atr_now
            trades.append((self.d.index[e], self.td[x], side, bars, usd, r))
            busy = x
        return pd.DataFrame(trades, columns=["entry", "exit_day", "side", "bars", "usd", "r"])


def daily_r(tr, td_index):
    return tr.groupby("exit_day")["r"].sum().reindex(td_index, fill_value=0.0)


def sh(x):
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")


def summarize(D, tr, tag):
    days = pd.Index(sorted(set(D.td)))
    dr = daily_r(tr, days)
    ins_t, ins_d = tr["entry"] < SPLIT, days < SPLIT
    out = {"配置": tag, "笔数内": int(ins_t.sum()), "笔数外": int((~ins_t).sum())}
    for seg, mt, md in (("内", ins_t, ins_d), ("外", ~ins_t, ~ins_d)):
        t = tr[mt]
        out[f"夏普{seg}"] = round(sh(dr[md]), 2)
        out[f"美元{seg}"] = round(float(t["usd"].sum()))
        out[f"R每笔{seg}"] = round(float(t["r"].mean()), 3) if len(t) else np.nan
        out[f"胜率{seg}"] = round(float((t["usd"] > 0).mean()), 2) if len(t) else np.nan
        out[f"多/空{seg}"] = f"{t['usd'][t.side > 0].sum():+.0f}/{t['usd'][t.side < 0].sum():+.0f}"
    return out, dr


def main(freqs):
    rows = []
    for fq in freqs:
        D = Data(fq)
        tr = D.backtest(D.events(DEFAULT["er"], DEFAULT["ov"], DEFAULT["flag"]), DEFAULT["h"])
        s, _ = summarize(D, tr, f"{fq} 默认")
        rows.append(s)
    print("[1] 默认定义，各频率（事件驱动，持有 24 根）\n" + pd.DataFrame(rows).set_index("配置").to_string())

    D = Data("30MIN")
    grid, rows, allg = {}, [], {}
    for er, ov, flag, hold in itertools.product((0.10, 0.15, 0.20), (0.4, 0.5, 0.6), (6, 8, 12), (12, 24, 36)):
        tr = D.backtest(D.events(er, ov, flag), hold)
        s, dr = summarize(D, tr, f"ER{er}/OV{ov}/F{flag}/H{hold}")
        rows.append(s); allg[s["配置"]] = dr
    G = pd.DataFrame(rows).set_index("配置")
    print(f"\n[2] 30MIN 网格（81 个）：样本内夏普>0 的比例 {(G['夏普内'] > 0).mean():.2f}，样本外>0 {(G['夏普外'] > 0).mean():.2f}，"
          f"两段都>0 {((G['夏普内'] > 0) & (G['夏普外'] > 0)).mean():.2f}")
    print("    样本内夏普分位 [10%,50%,90%]:", np.round(G["夏普内"].quantile([.1, .5, .9]).to_numpy(), 2),
          " 样本外:", np.round(G["夏普外"].quantile([.1, .5, .9]).to_numpy(), 2))
    for dim, key in (("ER 阈值", "ER"), ("重叠阈值", "OV"), ("整理长度", "F"), ("持有期", "H")):
        grp = G.groupby(G.index.map(lambda s: s.split("/")[["ER", "OV", "F", "H"].index(key)]))[["夏普内", "夏普外", "笔数外"]].median()
        print(f"    按{dim}的中位数：\n" + grp.round(2).to_string())
    M = pd.DataFrame(allg).fillna(0.0)
    c = cscv(M)
    print(f"\n[3] 网格 CSCV：PBO={c['PBO']:.3f}，N_eff≈{n_eff(M):.1f}，样本内最优在样本外夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f}，亏损概率 {c['prob_oos_loss']:.3f}")
    tr = D.backtest(D.events(DEFAULT["er"], DEFAULT["ov"], DEFAULT["flag"]), DEFAULT["h"])
    y = tr.groupby(tr["entry"].dt.year).agg(笔数=("usd", "size"), 美元=("usd", "sum"), R=("r", "sum"), 胜率=("usd", lambda s: (s > 0).mean()))
    print("\n[4] 30MIN 默认 逐年\n" + y.round(2).T.to_string())
    print(f"    平均持仓 {tr['bars'].mean() / 2:.1f} 小时；最大单笔 {tr['usd'].max():+.1f}/{tr['usd'].min():+.1f}$")
    tr.to_pickle("data/cache/rcb30_trades.pkl")
    daily_r(tr, pd.Index(sorted(set(D.td)))).to_pickle("data/cache/rcb30_daily.pkl")


if __name__ == "__main__":
    main(sys.argv[1:] or ["15MIN", "30MIN", "1H", "4H"])

# -*- coding: utf-8 -*-
"""
异常大 K 线事件研究（预注册：reports/big_candle_prereg.md）。

事件：30MIN / 1H K 线 实体 |C−O|（定义 B）或 振幅 H−L（定义 R）> k × 日ATR(14)（日线纽约 17:00 日切，只用前一交易日及以前）；
      方向 sign(C−O)；同一频率/定义下前一事件后 8 根内的新事件不计。
交易：下一根开盘逆 K 线方向（fade）入场，持有 h∈{1,2,4,8} 根；跨入换日前平仓窗口则在窗口首根开盘提前出场；入场落在窗口内跳过。
收益：R = 方向×价格变动 / 入场前一根 ATR − 方向×持有期逐年平均漂移；扣成本 0.2$ 按最近 1 年 ATR 中位数（当前成本）或当时 ATR（历史成本）折算。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from factors.core import atr, params, trading_day
from factors.evaluate import BAR, SPREAD
from paper.engine import compute
from paper.specs import get_spec

SPLIT = pd.Timestamp("2020-01-01")
FREQS = ("30MIN", "1H")
DEFS = {"B 实体": "body", "R 振幅": "range"}
KS = (1.0, 0.75)
HS = (1, 2, 4, 8)
DEDUP = 8
LOOKBACK = {"5MIN": 288, "30MIN": 48, "1H": 24}
RUNS = {"default": (("30MIN", "1H"), (1.0, 0.75)), "5min": (("5MIN",), (0.5, 0.35))}   # 预注册：big_candle_prereg.md / big_candle_5min_prereg.md


class Data:
    def __init__(self, freq: str):
        self.freq = freq
        d = pd.read_pickle(f"data/cache/{freq}.pkl")
        self.d = d
        self.o, self.h, self.l, self.c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        a = atr(d, params(freq)["atr"])
        self.a = a.to_numpy()
        self.atr_now = float(a[d.index >= d.index[-1] - pd.Timedelta(days=365)].median())
        d1 = pd.read_pickle("data/cache/1D.pkl")
        atr_d = atr(d1, 14).shift(1)                                     # 交易日 D 用 D−1 及以前的日线
        self.td = trading_day(d.index)
        self.atr_d = pd.Series(self.td).map(atr_d).to_numpy()
        ny = d.index.tz_localize("UTC").tz_convert("America/New_York")
        self.mins = np.asarray(ny.hour * 60 + ny.minute)
        start = 17 * 60 - 2 * pd.Timedelta(BAR[freq]).seconds // 60
        self.flat = (self.mins >= start) & (self.mins < 18 * 60)
        m = (d["open"].shift(-1) - d["open"]) / a.shift(1)
        self.drift = m.groupby(d.index.year).transform("mean").to_numpy()
        n = LOOKBACK[freq]
        c = d["close"]
        self.trend = ((c.shift(1) - c.shift(1 + n)) / self.atr_d).to_numpy()

    def events(self, kind: str, k: float) -> list[int]:
        size = np.abs(self.c - self.o) if kind == "body" else (self.h - self.l)
        cand = np.flatnonzero((size > k * self.atr_d) & np.isfinite(self.atr_d) & (self.c != self.o))
        out, last = [], -10 ** 9
        for t in cand:
            if t - last > DEDUP:
                out.append(t)
                last = t
        return out

    def session(self, t: int) -> str:
        m, bar = self.mins[t], pd.Timedelta(BAR[self.freq]).seconds // 60
        if m <= 8 * 60 + 30 < m + bar:
            return "美国数据(08:30NY)"
        if 8 * 60 <= m < 17 * 60:
            return "美国其他"
        if 3 * 60 <= m < 8 * 60:
            return "欧洲"
        return "亚洲"

    def context(self, t: int, side_candle: int) -> str:
        T = self.trend[t]
        if not np.isfinite(T) or abs(T) < 0.5:
            return "无趋势"
        return "顺势" if np.sign(T) == side_candle else "逆势"

    def trades(self, ev: list[int], h: int) -> pd.DataFrame:
        n, rows = len(self.o), []
        for t in ev:
            e = t + 1
            if e + 1 >= n or self.flat[e] or not np.isfinite(self.a[t]):
                continue
            x = min(e + h, n - 1)
            fl = np.flatnonzero(self.flat[e + 1:x + 1])
            if len(fl):
                x = e + 1 + fl[0]
            cs = int(np.sign(self.c[t] - self.o[t]))
            side = -cs
            move = side * (self.o[x] - self.o[e])
            r = move / self.a[t] - side * np.nansum(self.drift[e:x])
            rows.append({"t": self.d.index[t], "entry": self.d.index[e], "exit": self.d.index[x], "candle": cs, "side": side,
                         "bars": x - e, "usd": move - SPREAD, "r": r, "r_now": r - SPREAD / self.atr_now,
                         "r_hist": r - SPREAD / self.a[t], "session": self.session(t), "ctx": self.context(t, cs),
                         "size_atr_d": float(max(abs(self.c[t] - self.o[t]), 0) / self.atr_d[t])})
        return pd.DataFrame(rows)


def tstat(x: pd.Series) -> float:
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float("nan")


def cell_stats(T: pd.DataFrame, cost_now: float, last: pd.Timestamp) -> dict:
    ins, rec = T.t < SPLIT, T.t >= last - pd.Timedelta(days=1095)
    yr = T.groupby(T.t.dt.year).r_now.mean()
    q = T.r_now.quantile([0.05, 0.95])
    trim = T.r_now[(T.r_now >= q.iloc[0]) & (T.r_now <= q.iloc[1])]
    top = T.r.sort_values(ascending=False)
    k10 = max(1, int(len(T) * 0.1))
    return {"n内": int(ins.sum()), "n外": int((~ins).sum()),
            "R内": round(T.r_now[ins].mean(), 3), "R外": round(T.r_now[~ins].mean(), 3), "R近3年": round(T.r[rec].mean(), 3),
            "当前成本": round(cost_now, 3), "t": round(tstat(T.r_now), 2), "逐年>0": round(float((yr > 0).mean()), 2),
            "中位": round(T.r_now.median(), 3), "胜率": round(float((T.r_now > 0).mean()), 2),
            "截尾内": round(trim[ins.reindex(trim.index)].mean(), 3), "截尾外": round(trim[~ins.reindex(trim.index)].mean(), 3),
            "前10%占总": round(float(top.iloc[:k10].sum() / T.r.sum()), 2) if T.r.sum() != 0 else np.nan,
            "阳/阴(fade)": f"{T.r_now[T.candle > 0].mean():+.3f}/{T.r_now[T.candle < 0].mean():+.3f}",
            "美元内/外": f"{T.usd[ins].sum():+.0f}/{T.usd[~ins].sum():+.0f}", "历史成本R": round(T.r_hist.mean(), 3)}


def verdict(s: dict) -> str:
    y, x = s["阳/阴(fade)"].split("/")
    ok = (s["R内"] > 0 and s["R外"] > 0 and s["t"] >= 3 and s["逐年>0"] >= 0.6 and s["R近3年"] > 1.5 * s["当前成本"]
          and s["截尾内"] > 0 and s["截尾外"] > 0 and float(y) > 0 and float(x) > 0)
    if ok:
        return "✅ 可交易回吐"
    return "↗ 延续（fade 显著为负）" if s["t"] <= -3 else "❌"


def live_positions() -> dict:
    """在跑策略每根 K 线的持仓与净 R（模拟盘引擎口径）。"""
    out = {}
    for sid in ("TT30-EW-v1", "HA1H-v1"):
        spec = get_spec(sid)
        bars = pd.read_pickle(f"data/cache/{spec.freq}.pkl")
        c = compute(spec, bars)
        out[sid] = c[["pos", "net_R"]]
    return out


def attach_live(T: pd.DataFrame, live: dict) -> pd.DataFrame:
    T = T.copy()
    close_t = T["entry"]                                               # 事件 K 线收盘 = 入场 K 线开盘时刻
    for sid, c in live.items():
        pos = pd.merge_asof(pd.DataFrame({"k": close_t.to_numpy()}), c["pos"].rename("p").rename_axis("k").reset_index(),
                            on="k", direction="backward")["p"].to_numpy()
        rel = np.where(pos == 0, "空仓", np.where(np.sign(pos) == T["candle"].to_numpy(), "与K线同向", "与K线反向"))
        T[f"{sid}_rel"] = rel
        cum = c["net_R"].fillna(0).cumsum()
        a = cum.reindex(T["entry"], method="ffill").to_numpy()
        b = cum.reindex(T["exit"], method="ffill").to_numpy()
        T[f"{sid}_R"] = b - a                                          # 在跑策略在 fade 持有窗口内的净 R（含其自身持仓）
    return T


def main(run: str = "default") -> None:
    FREQS, KS = RUNS[run]
    k0 = KS[0]                                                          # 主检验阈值
    sfx = "" if run == "default" else f"_{run}"
    pd.set_option("display.width", 280)
    live = live_positions()
    rows, keep = [], {}
    for fq in FREQS:
        D = Data(fq)
        for (dname, kind), k in itertools.product(DEFS.items(), KS):
            ev = D.events(kind, k)
            for h in HS:
                T = D.trades(ev, h)
                s = cell_stats(T, SPREAD / D.atr_now, D.d.index[-1])
                rows.append({"频率": fq, "定义": dname, "k": k, "h": h, **s, "判定": verdict(s)})
                keep[(fq, dname, k, h)] = T
    R = pd.DataFrame(rows)
    print(f"[1] 全部 {len(FREQS) * len(DEFS) * len(KS) * len(HS)} 个预注册单元（fade 方向；R 为每笔 ATR，已去漂移、扣当前成本；R近3年 未扣成本，与 1.5×当前成本比较）")
    print(R.to_string(index=False))
    R.to_csv(f"reports/big_candle_cells{sfx}.csv", index=False)

    print(f"\n[2] 主检验（k={k0}）分组：每笔 R（扣当前成本）与笔数，样本内 | 样本外")
    for fq, dname in itertools.product(FREQS, DEFS):
        print(f"\n--- {fq} {dname} k={k0} ---")
        for grp in ("candle", "session", "ctx"):
            tab = {}
            for h in HS:
                T = keep[(fq, dname, k0, h)]
                T = T.assign(seg=np.where(T.t < SPLIT, "内", "外"), candle=T.candle.map({1: "大阳(做空)", -1: "大阴(做多)"}))
                g = T.groupby([grp, "seg"]).r_now.agg(["mean", "size"])
                tab[f"h={h}"] = g.apply(lambda r: f"{r['mean']:+.3f}({int(r['size'])})", axis=1)
            print(pd.DataFrame(tab).to_string())

    print(f"\n[3] 极端事件检查（k={k0}，h=4）：逐年 fade R")
    for fq, dname in itertools.product(FREQS, DEFS):
        T = keep[(fq, dname, k0, 4)]
        print(f"   {fq} {dname}：", T.groupby(T.t.dt.year).r_now.mean().round(2).to_dict())
        top = T.reindex(T.r.abs().sort_values(ascending=False).index).head(5)
        print("      |R| 最大 5 个：", [(str(x.t)[:16], x.session, round(x.r, 2)) for x in top.itertuples()])

    print(f"\n[4] 与在跑策略的关系（k={k0}，h=4，全样本）")
    for fq, dname in itertools.product(FREQS, DEFS):
        T = attach_live(keep[(fq, dname, k0, 4)], live)
        print(f"\n--- {fq} {dname} k={k0} h=4：{len(T)} 个事件 ---")
        for sid in live:
            g = T.groupby(f"{sid}_rel").agg(事件数=("r_now", "size"), fade_R=("r_now", "mean"), 在跑策略窗口R=(f"{sid}_R", "mean"))
            g["占比"] = (g["事件数"] / len(T)).round(2)
            print(f"  {sid}（'与K线同向' = 在跑持仓与大K线同向，fade 与之冲突）\n" + g.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "default")

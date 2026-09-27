# -*- coding: utf-8 -*-
"""
锁仓（双开多空）策略检验（预登记：reports/hedge_lock_prereg.md）。

每个工作日伦敦 08:00、纽约 08:30（当地时间）开锁：多、空各 1 单位，开仓价 P = 该分钟 M1 开盘价；A = 最后一根已收盘 15MIN 的 ATR(32)。
  L 锁仓后砍亏损腿：价格首次触及 P±kA → 平亏损腿（亏 kA），盈利腿持有到 4 小时后开盘价出场。
  T 两头吃波动：多单止盈 P+kA、空单止盈 P−kA；一腿止盈后另一腿继续等自己的止盈，否则 4 小时后出场。
同一根 M1 同时触及两侧（或同一根里第二腿也止盈）时，取两种顺序中较差的结果；第二腿从下一根 M1 起算。
等价单边版：只在触发后按剩余腿方向开 1 单位，盈亏与锁仓版相同，少付一次点差（未触发的轮次不交易）。
成本：每腿开平 0.2$；不跨纽约 17:00，无过夜费。ATR 口径：美元 / A，扣除按净持仓方向 × 持有分钟数 × 当年每分钟平均漂移。
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from factors.core import atr

SPLIT = pd.Timestamp("2020-01-01")
KS = (1.0, 2.0, 3.0)
HOLD = 240
SPREAD = 0.2
SESSIONS = (("伦敦 08:00", "Europe/London", 8 * 60), ("纽约 08:30", "America/New_York", 8 * 60 + 30))


def load():
    m1 = pd.read_pickle(sorted(glob.glob("data/cache/m1_*.pkl"))[-1])
    b15 = pd.read_pickle("data/cache/15MIN.pkl")
    a15 = atr(b15, 32)
    a_avail = pd.Series(a15.to_numpy(), index=b15.index + pd.Timedelta("15min"))     # 收盘后才可用
    idx = m1.index
    events = []
    for name, tz, mins in SESSIONS:
        loc = idx.tz_localize("UTC").tz_convert(tz)
        hit = np.flatnonzero((loc.hour * 60 + loc.minute == mins) & (loc.dayofweek < 5))
        events += [(i, name) for i in hit]
    events.sort()
    A = pd.merge_asof(pd.DataFrame({"t": idx[[e[0] for e in events]]}), a_avail.rename("a").rename_axis("t").reset_index(),
                      on="t", direction="backward")["a"].to_numpy()
    yr = idx.year
    drift = pd.Series(np.r_[0.0, np.diff(m1["close"].to_numpy())], index=idx).groupby(yr).mean()   # 每分钟平均漂移（$）
    a_now = float(a15[b15.index >= b15.index[-1] - pd.Timedelta(days=365)].median())
    return m1, events, A, drift, a_now


def first_touch(h, l, up, dn, start):
    """从 start 起首次触及 up（+1）或 dn（−1）的位置；同一根同时触及返回 0。"""
    iu = np.flatnonzero(h[start:] >= up); idn = np.flatnonzero(l[start:] <= dn)
    ju = iu[0] + start if len(iu) else None; jd = idn[0] + start if len(idn) else None
    if ju is None and jd is None:
        return None, None
    if jd is None or (ju is not None and ju < jd):
        return ju, 1
    if ju is None or jd < ju:
        return jd, -1
    return ju, 0


def simulate(m1, events, A, drift, k, kind):
    o, h, l = (m1[c].to_numpy() for c in ("open", "high", "low"))
    idx = m1.index
    rows = []
    for (i, sess), a in zip(events, A):
        x = i + HOLD
        if not np.isfinite(a) or a <= 0 or x >= len(o) or (idx[x] - idx[i]) > pd.Timedelta(minutes=HOLD + 30):
            continue                                                       # 跳过缺数据 / 跨休市的轮次
        P, K = o[i], k * a
        H, L = h[i:x], l[i:x]
        j, d = first_touch(H, L, P + K, P - K, 0)
        dpm = float(drift.get(idx[i].year, 0.0))
        outcomes = []
        for dd in ([d] if d in (1, -1) else ([1, -1] if d == 0 else [None])):
            if dd is None:                                                 # 未触发：锁仓版两腿原价平仓
                outcomes.append((0.0, 0, 0, 0))
                continue
            trig = P + dd * K
            if kind == "L":
                side = dd                                                  # 剩余腿 = 顺势腿
                pnl = side * (o[x] - trig)
                held = HOLD - j
            else:
                side = -dd                                                 # 剩余腿 = 逆势腿，止盈在另一侧 P−dd·K
                tp = P - dd * K
                rest = np.flatnonzero((L[j + 1:] <= tp) if side < 0 else (H[j + 1:] >= tp))
                if len(rest):
                    pnl = side * (tp - trig); held = rest[0] + 1
                else:
                    pnl = side * (o[x] - trig); held = HOLD - j
            outcomes.append((pnl, side, held, 1))
        pnl, side, held, trig_ = min(outcomes, key=lambda z: z[0])       # 不利方向
        de = side * dpm * held
        rows.append({"t": idx[i], "sess": sess, "a": a, "trig": trig_, "side": side, "pnl": pnl,
                     "usd_single": pnl - SPREAD * trig_, "usd_lock": pnl - 2 * SPREAD,
                     "r_single": (pnl - de - SPREAD * trig_) / a, "r_lock": (pnl - de - 2 * SPREAD) / a})
    return pd.DataFrame(rows)


def summarize(E, a_now, tag):
    out = []
    rec = E.t >= E.t.max() - pd.Timedelta(days=1095)
    for ver, col, cost in (("单边等价", "single", SPREAD), ("锁仓", "lock", 2 * SPREAD)):
        r = E[f"r_{col}"]
        row = {"结构": tag, "版本": ver, "轮次": len(E), "触发率": round(float(E.trig.mean()), 2)}
        for seg, m in (("内", E.t < SPLIT), ("外", E.t >= SPLIT), ("近3年", rec)):
            row[f"R/轮{seg}"] = round(float(r[m].mean()), 4)
        row["t(全)"] = round(float(r.mean() / (r.std() / np.sqrt(len(r)))), 2)
        yr = r.groupby(E.t.dt.year).mean()
        row["逐年>0"] = round(float((yr > 0).mean()), 2)
        row["当前成本ATR"] = round(cost / a_now, 3)
        row["美元内/外"] = f"{E[f'usd_{col}'][E.t < SPLIT].sum():+.0f}/{E[f'usd_{col}'][E.t >= SPLIT].sum():+.0f}"
        tr = E[E.trig == 1]
        row["多/空美元(外)"] = f"{tr.usd_single[(tr.side > 0) & (tr.t >= SPLIT)].sum():+.0f}/{tr.usd_single[(tr.side < 0) & (tr.t >= SPLIT)].sum():+.0f}"
        out.append(row)
    return out


def main() -> None:
    m1, events, A, drift, a_now = load()
    rows, allE = [], {}
    for kind, name in (("L", "L 锁仓后砍亏损腿"), ("T", "T 两头吃波动")):
        for k in KS:
            E = simulate(m1, events, A, drift, k, kind)
            allE[(kind, k)] = E
            rows += summarize(E, a_now, f"{name} k={k:g}")
    pd.set_option("display.width", 260)
    T = pd.DataFrame(rows)
    print(f"开锁轮次：伦敦 08:00 + 纽约 08:30，每个工作日 2 次；最长持有 {HOLD} 分钟；当前 15MIN ATR 中位数 {a_now:.2f}$\n")
    print(T.to_string(index=False))
    for key in (("L", 1.0), ("T", 1.0)):
        E = allE[key]
        print(f"\n{key} 分时段（单边等价 R/轮 内/外）：",
              {s: (round(float(g.r_single[g.t < SPLIT].mean()), 3), round(float(g.r_single[g.t >= SPLIT].mean()), 3)) for s, g in E.groupby("sess")})
        print(f"{key} 逐年单边等价 R/轮：", E.groupby(E.t.dt.year).r_single.mean().round(3).to_dict())
    T.to_csv("reports/hedge_lock.csv", index=False)


if __name__ == "__main__":
    main()

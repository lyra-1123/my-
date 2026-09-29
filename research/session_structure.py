# -*- coding: utf-8 -*-
"""
第十四批 B：日内时段结构（15MIN，市价单，点差 0.3$），预登记：reports/batch14_prereg.md。

B1 IntradayMomentum   纽约 08:30~09:30 收益方向 → 14:00 开盘入场、16:00 开盘平仓；安慰剂用 11:00~12:00 方向
B2 SessionSeasonality 每个时段（亚盘 18-02、伦敦 02-08、纽约 08-16，纽约时间）过去 60 个交易日该时段收益的 t 值，
                      |t|≥1.5 时按符号持有该时段；安慰剂用过去 60 日全天收益的 t 值
用法：python -m research.session_structure [--oos]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from factors.core import atr, trading_day
from research.multifactor_combo import sh

SPLIT = pd.Timestamp("2020-01-01")
COST = 0.3


def opens():
    d = pd.read_pickle("data/cache/15MIN.pkl")
    a = atr(d, 32).shift(1)
    ny = d.index.tz_localize("UTC").tz_convert("America/New_York")
    key = pd.DataFrame({"td": trading_day(d.index), "m": ny.hour * 60 + ny.minute, "o": d["open"].to_numpy(), "a": a.to_numpy()})
    O = key.pivot_table(index="td", columns="m", values="o", aggfunc="first")
    A = key.pivot_table(index="td", columns="m", values="a", aggfunc="first")
    cost_now = float((COST / atr(d, 32)[d.index >= d.index[-1] - pd.Timedelta(days=365)]).median())
    return O, A, cost_now


def seg_ret(O, A, m0, m1):
    """(open@m1 − open@m0) / ATR@m0，按交易日。"""
    return (O[m1] - O[m0]) / A[m0]


def dedrift(r: pd.Series) -> pd.Series:
    return r - r.groupby(r.index.year).transform("mean")


def trades_b1(O, A):
    x = seg_ret(O, A, 14 * 60, 16 * 60)
    ex = dedrift(x.dropna())
    s_main = np.sign(O[9 * 60 + 30] - O[8 * 60 + 30])
    s_plac = np.sign(O[12 * 60] - O[11 * 60])
    out = {}
    for g, s in (("main", s_main), ("placebo", s_plac)):
        df = pd.DataFrame({"side": s, "ex": ex}).dropna()
        df = df[df.side != 0]
        out[g] = pd.DataFrame({"r": df.side * df.ex, "side": df.side, "sess": "NY尾盘"})
    return out


SESS = {"亚盘": (18 * 60, 2 * 60), "伦敦": (2 * 60, 8 * 60), "纽约": (8 * 60, 16 * 60)}


def trades_b2(O, A):
    R = pd.DataFrame({k: seg_ret(O, A, *v) for k, v in SESS.items()})
    full = R.sum(axis=1, min_count=3)
    def tstat(x):
        m = x.rolling(60, min_periods=40).mean().shift(1)
        s = x.rolling(60, min_periods=40).std().shift(1)
        return m / (s / np.sqrt(x.rolling(60, min_periods=40).count().shift(1)))
    tf = tstat(full)
    out = {"main": [], "placebo": []}
    for k in SESS:
        ex = dedrift(R[k].dropna())
        for g, t in (("main", tstat(R[k])), ("placebo", tf)):
            side = np.sign(t).where(t.abs() >= 1.5)
            df = pd.DataFrame({"side": side, "ex": ex}).dropna()
            out[g].append(pd.DataFrame({"r": df.side * df.ex, "side": df.side, "sess": k}))
    return {g: pd.concat(v).sort_index() for g, v in out.items()}


def stats(x):
    r = x["r"]
    n = len(r)
    return {"n": n, "ATR": r.mean(), "t": r.mean() / (r.std() / np.sqrt(n)) if n > 2 else np.nan,
            "多": r[x.side > 0].mean(), "空": r[x.side < 0].mean()}


def diff_t(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true")
    args = ap.parse_args()
    O, A, cn = opens()
    print(f"当前成本 0.3$ = {cn:.4f} ATR(15MIN)；入选需每笔 ≥ {2 * cn:.4f}")
    EV = {"B1 IntradayMomentum": trades_b1(O, A), "B2 SessionSeasonality": trades_b2(O, A)}
    seg = lambda x, oos: x[x.index >= SPLIT] if oos else x[x.index < SPLIT]
    passed = []
    for name, ev in EV.items():
        m, p = stats(seg(ev["main"], False)), stats(seg(ev["placebo"], False))
        dt = diff_t(seg(ev["main"], False)["r"], seg(ev["placebo"], False)["r"])
        crit = {"S1 ≥2×成本": m["ATR"] >= 2 * cn, "S2 t≥2.5": m["t"] >= 2.5, "S3 优于安慰剂 t≥2": dt >= 2, "S4 n≥200": m["n"] >= 200}
        ok = all(crit.values())
        print(f"\n[样本内] {name}：n={m['n']} 每笔 {m['ATR']:+.4f} ATR t={m['t']:+.2f} 多 {m['多']:+.4f} 空 {m['空']:+.4f}；"
              f"安慰剂 n={p['n']} {p['ATR']:+.4f} (t={p['t']:+.2f})；差 t={dt:+.2f}")
        if name.startswith("B2"):
            for k in SESS:
                x = seg(ev["main"], False)
                x = x[x.sess == k]
                print(f"    {k}: n={len(x)} 每笔 {x.r.mean():+.4f} 多 {x.r[x.side > 0].mean():+.4f}({(x.side > 0).sum()}) 空 {x.r[x.side < 0].mean():+.4f}({(x.side < 0).sum()})")
        print("  " + "  ".join(f"{'✅' if v else '❌'}{k}" for k, v in crit.items()) + f" → {'入选' if ok else '不入选'}")
        if ok:
            passed.append(name)
    print(f"\n样本内入选：{passed or '无'}")
    if not args.oos:
        return
    from research.failed_factor_combo import frozen_daily
    ref = frozen_daily()
    log = []
    for name, ev in EV.items():
        mi = stats(seg(ev["main"], False))
        if name not in passed:
            log.append([name, mi, None, "样本内未入选，未跑样本外"])
            continue
        mo = stats(seg(ev["main"], True))
        dt = diff_t(seg(ev["main"], True)["r"], seg(ev["placebo"], True)["r"])
        daily = (ev["main"]["r"] - cn).groupby(level=0).sum()
        J = ref.join(daily.rename("new"), how="outer").fillna(0.0)
        info, o4, o5 = [], True, True
        for oos in (False, True):
            j = seg(J, oos)
            cc = j.corr()["new"]
            b0, b1 = sh(j["TT30-EW-v1"] + j["HA1H-v1"]), sh(j.sum(axis=1))
            info.append(f"{'外' if oos else '内'} 相关 {cc['TT30-EW-v1']:+.2f}/{cc['HA1H-v1']:+.2f} 单独 {sh(j['new']):.2f} 组合 {b0:.2f}→{b1:.2f}")
            o4 &= bool((cc[["TT30-EW-v1", "HA1H-v1"]].abs() < 0.3).all())
            o5 &= b1 >= b0
        res = {"O1 >成本且 t≥2": mo["ATR"] > cn and mo["t"] >= 2, "O2 优于安慰剂": dt > 0, "O3 多空都为正": mo["多"] > 0 and mo["空"] > 0,
               "O4 相关 < 0.3": o4, "O5 组合夏普不降": o5}
        print(f"\n[样本外] {name}：n={mo['n']} 每笔 {mo['ATR']:+.4f} t={mo['t']:+.2f} 多 {mo['多']:+.4f} 空 {mo['空']:+.4f}；差 t={dt:+.2f}\n  " + "；".join(info))
        print("  " + "  ".join(f"{'✅' if v else '❌'}{k}" for k, v in res.items()))
        log.append([name, mi, mo, "通过" if all(res.values()) else "未通过：" + ",".join(k for k, v in res.items() if not v)])
        if all(res.values()):
            daily.to_pickle(f"data/cache/{name.split()[0]}_daily.pkl")
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for name, mi, mo, note in log:
            w.writerow(["2026-09-29", name, "15MIN", "batch14 session structure", "event", "cost=0.3$", "IS event-study S1-S4",
                        round(mi["t"], 2), round(mi["ATR"], 4), "" if mo is None else round(mo["t"], 2),
                        "" if mo is None else round(mo["ATR"], 4), "" if mo is None else round(mo["ATR"] / cn, 2), 2,
                        f"t列为事件t值、净利列为每笔ATR；{note}"])
    print(f"已登记 {len(log)} 条")


if __name__ == "__main__":
    main()

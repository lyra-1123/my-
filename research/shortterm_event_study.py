# -*- coding: utf-8 -*-
"""
第十三批：短线（15MIN，持有 1 小时）新信息来源事件研究（预登记：reports/shortterm_batch13_prereg.md）。

E1 FixReversal        伦敦金定盘（10:30 / 15:00 伦敦）前 60 分钟位移 ≥1 ATR → 反向
E2 MacroImpulse       纽约 08:30 / 10:00 开始的 15MIN K 线 |收益| ≥1.5 ATR → 顺向
E3 PrevDayFalseBreak  当日首次刺破前一交易日高/低点但收盘收回 → 反向
E4 RoundNumberReject  盘中穿过 50$ 整数关口但收盘收回 → 反向
每个事件带安慰剂对照（去掉关键条件）。信号 t 收盘，t+1 开盘入场，持有 h 根；ATR 单位、逐年去漂移；成本按 0.3$。

用法：python -m research.shortterm_event_study          # 样本内判定
      python -m research.shortterm_event_study --oos    # 入选事件的样本外（只跑一次）+ 组合检验 + 登记
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from factors.core import atr, trading_day
from research.multifactor_combo import sh

SPLIT = pd.Timestamp("2020-01-01")
FREQ, H, H2 = "15MIN", 4, 8
COST, COST_OLD = 0.3, 0.2
BAR = pd.Timedelta("15min")


def load():
    d = pd.read_pickle(f"data/cache/{FREQ}.pkl")
    a = atr(d, 32)
    lon = d.index.tz_localize("UTC").tz_convert("Europe/London")
    ny = d.index.tz_localize("UTC").tz_convert("America/New_York")
    return d, a, (lon.hour * 60 + lon.minute).to_numpy(), (ny.hour * 60 + ny.minute).to_numpy()


def first_per_day(t: np.ndarray, td: np.ndarray) -> np.ndarray:
    """同一交易日只保留第一次（t 已升序）。"""
    if len(t) == 0:
        return t
    _, i = np.unique(td[t], return_index=True)
    return t[np.sort(i)]


def events(d, a, lonm, nym):
    """{事件名: {"main": (t, side), "placebo": (t, side)}}"""
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    an = a.to_numpy()
    td = trading_day(d.index).to_numpy()
    idx = d.index
    N = len(d)
    ok_time = ~((nym >= 16 * 60) & (nym < 18 * 60))            # 换日前后不入场
    out = {}

    # E1：t 为结束于定盘时刻的 K 线（开始于 10:15 / 14:45 伦敦）
    contig = np.r_[[False] * 3, (idx[3:] - idx[:-3]) == 3 * BAR]
    pre = np.full(N, np.nan)
    pre[3:] = (c[3:] - o[:-3]) / an[3:]
    end = lonm + 15
    fix = np.isin(end, (10 * 60 + 30, 15 * 60))
    plac = np.isin(end, [m for m in range(8 * 60 + 30, 16 * 60 + 1, 30) if m not in (10 * 60 + 30, 15 * 60)])
    base = contig & (np.abs(pre) >= 1.0) & ok_time
    out["E1 FixReversal"] = {k: (lambda t: (t, -np.sign(pre[t])))(first_per_day(np.where(base & m)[0], td))
                             for k, m in (("main", fix), ("placebo", plac))}

    # E2：纽约 08:30 / 10:00 开始的 K 线
    r = (c - o) / an
    mac = np.isin(nym, (8 * 60 + 30, 10 * 60))
    plac = (nym >= 7 * 60) & (nym < 12 * 60) & ~mac
    base = (np.abs(r) >= 1.5) & ok_time
    out["E2 MacroImpulse"] = {k: (lambda t: (t, np.sign(r[t])))(first_per_day(np.where(base & m)[0], td))
                              for k, m in (("main", mac), ("placebo", plac))}

    # E3：前一交易日高低点
    s = pd.DataFrame({"h": h, "l": l, "td": td})
    dh, dl = s.groupby("td")["h"].max(), s.groupby("td")["l"].min()
    ph = s["td"].map(dh.shift(1)).to_numpy()
    pl = s["td"].map(dl.shift(1)).to_numpy()
    cum_h = s.groupby("td")["h"].cummax().shift(1).where(s["td"].eq(s["td"].shift(1))).to_numpy()
    cum_l = s.groupby("td")["l"].cummin().shift(1).where(s["td"].eq(s["td"].shift(1))).to_numpy()
    first_up = (h > ph) & ~(cum_h > ph)                          # 当日第一次刺破前日高点
    first_dn = (l < pl) & ~(cum_l < pl)
    fb_up, tb_up = first_up & (c < ph), first_up & (c >= ph)
    fb_dn, tb_dn = first_dn & (c > pl), first_dn & (c <= pl)
    def two_sided(up, dn):
        t = np.sort(np.r_[np.where(up & ok_time)[0], np.where(dn & ok_time)[0]])
        side = np.where(up[t], -1.0, 1.0)
        return t, side
    out["E3 PrevDayFalseBreak"] = {"main": two_sided(fb_up, fb_dn), "placebo": two_sided(tb_up, tb_dn)}

    # E4：50$ 整数关口 vs 偏移 25$ 的安慰剂水平
    def reject(off):
        lu = np.ceil((o - off) / 50) * 50 + off                   # 开盘价上方最近的水平
        ld = np.floor((o - off) / 50) * 50 + off
        up = (lu > o) & (h >= lu) & (c < lu) & ok_time
        dn = (ld < o) & (l <= ld) & (c > ld) & ok_time
        t = np.sort(np.r_[first_per_day(np.where(up)[0], td), first_per_day(np.where(dn)[0], td)])
        return t, np.where(up[t], -1.0, 1.0)
    out["E4 RoundNumberReject"] = {"main": reject(0.0), "placebo": reject(25.0)}
    return out


class Fwd:
    def __init__(self, d, a):
        self.d, self.a = d, a
        o = d["open"]
        td = trading_day(d.index)
        self.cost_now = {c: float((c / a[d.index >= d.index[-1] - pd.Timedelta(days=365)]).median()) for c in (COST, COST_OLD)}
        self.ex, self.usd, self.valid = {}, {}, {}
        for hh in (H, H2):
            move = o.shift(-(1 + hh)) - o.shift(-1)
            f = move / a
            self.ex[hh] = (f - f.groupby(d.index.year).transform("mean")).to_numpy()
            self.usd[hh] = move.to_numpy()
            ent, ext = pd.Series(d.index).shift(-1), pd.Series(d.index).shift(-(1 + hh))
            same_day = pd.Series(td).shift(-1).eq(pd.Series(td).shift(-(1 + hh)))
            self.valid[hh] = (same_day & ((ext - ent) <= hh * BAR + pd.Timedelta("1h"))).to_numpy()

    def rets(self, t, side, hh=H):
        m = self.valid[hh][t] & np.isfinite(self.ex[hh][t])      # 去掉 ATR 尚未形成的开头几根
        t, side = t[m], side[m]
        return t, side * self.ex[hh][t], side * self.usd[hh][t]


def tstat(x):
    return float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def summarize(name, ev, F, seg_mask_fn):
    rows = {}
    for g in ("main", "placebo"):
        t, side = ev[g]
        t, r, u = F.rets(t, side)
        m = seg_mask_fn(t)
        keep = F.valid[H][ev[g][0]] & np.isfinite(F.ex[H][ev[g][0]])
        rr, uu, ss = r[m], u[m], side[keep][m]
        rows[g] = {"n": len(rr), "ATR": np.mean(rr) if len(rr) else np.nan, "t": tstat(rr),
                   "$/笔毛": np.mean(uu) if len(uu) else np.nan,
                   "多ATR": np.mean(rr[ss > 0]) if (ss > 0).any() else np.nan,
                   "空ATR": np.mean(rr[ss < 0]) if (ss < 0).any() else np.nan, "_r": rr}
        t8, r8, _ = F.rets(ev[g][0], ev[g][1], H2)
        rows[g]["h8 ATR"] = np.mean(r8[seg_mask_fn(t8)]) if len(r8) else np.nan
    a, b = rows["main"]["_r"], rows["placebo"]["_r"]
    diff_t = (np.mean(a) - np.mean(b)) / np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) if len(a) > 2 and len(b) > 2 else np.nan
    return rows, float(diff_t)


def daily_strategy(F, t, side):
    """事件策略：每个事件 1 单位、持有 4 根；ATR·今 口径（去漂移收益 − 0.3$ 按当前 ATR 折算），按入场交易日汇总。"""
    t, r, _ = F.rets(t, side)
    net = r - F.cost_now[COST]
    return pd.Series(net, index=trading_day(F.d.index[t + 1])).groupby(level=0).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true")
    args = ap.parse_args()
    d, a, lonm, nym = load()
    F = Fwd(d, a)
    EV = events(d, a, lonm, nym)
    cn = F.cost_now[COST]
    ins = lambda t: d.index[t] < SPLIT
    print(f"当前成本（最近 1 年 15MIN ATR 中位数折算）：0.3$ = {cn:.4f} ATR，0.2$ = {F.cost_now[COST_OLD]:.4f} ATR；主持有期 h={H} 根")
    passed = []
    for name, ev in EV.items():
        rows, dt = summarize(name, ev, F, ins)
        m = rows["main"]
        crit = {"S1 ≥2×成本": m["ATR"] >= 2 * cn, "S2 t≥2.5": m["t"] >= 2.5, "S3 优于安慰剂 t≥2": dt >= 2, "S4 n≥200": m["n"] >= 200}
        ok = all(crit.values())
        print(f"\n[样本内 2009-2019] {name}")
        for g in ("main", "placebo"):
            x = rows[g]
            print(f"  {'事件  ' if g == 'main' else '安慰剂'} n={x['n']:5d}  每笔 {x['ATR']:+.4f} ATR (t={x['t']:+.2f})  毛利 {x['$/笔毛']:+.3f}$  "
                  f"多 {x['多ATR']:+.4f} 空 {x['空ATR']:+.4f}  h8 {x['h8 ATR']:+.4f}")
        print(f"  事件−安慰剂 t={dt:+.2f}  " + "  ".join(f"{'✅' if v else '❌'}{k}" for k, v in crit.items()) + f"  → {'入选' if ok else '不入选'}")
        if ok:
            passed.append(name)
    print(f"\n样本内入选：{passed or '无'}")
    if not args.oos:
        print("（样本外只在 --oos 时运行一次）")
        return

    from research.failed_factor_combo import frozen_daily
    ref = frozen_daily()
    oos = lambda t: d.index[t] >= SPLIT
    log = []
    for name, ev in EV.items():
        rows_in, dt_in = summarize(name, ev, F, ins)
        if name not in passed:
            log.append([name, rows_in["main"], None, "样本内未入选，未跑样本外"])
            continue
        rows, dt = summarize(name, ev, F, oos)
        m = rows["main"]
        s = daily_strategy(F, *ev["main"])
        J = ref.join(s.rename("new"), how="outer").fillna(0.0)
        res = {"O1 >成本且 t≥2": m["ATR"] > cn and m["t"] >= 2, "O2 优于安慰剂": dt > 0,
               "O3 多空都为正": m["多ATR"] > 0 and m["空ATR"] > 0}
        info = []
        o4, o5 = True, True
        for seg, sel in (("内", J.index < SPLIT), ("外", J.index >= SPLIT)):
            j = J[sel]
            cc = j.corr()["new"]
            b0, b1 = sh(j["TT30-EW-v1"] + j["HA1H-v1"]), sh(j.sum(axis=1))
            info.append(f"{seg}：相关 TT30 {cc['TT30-EW-v1']:+.2f} HA1H {cc['HA1H-v1']:+.2f}，单独夏普 {sh(j['new']):.2f}，组合 {b0:.2f}→{b1:.2f}")
            o4 &= bool((cc[["TT30-EW-v1", "HA1H-v1"]].abs() < 0.3).all())
            o5 &= b1 >= b0
        res["O4 相关 < 0.3"], res["O5 组合夏普不降"] = o4, o5
        print(f"\n[样本外 2020-] {name}：n={m['n']} 每笔 {m['ATR']:+.4f} ATR (t={m['t']:+.2f}) 毛利 {m['$/笔毛']:+.3f}$ "
              f"多 {m['多ATR']:+.4f} 空 {m['空ATR']:+.4f}；安慰剂 {rows['placebo']['ATR']:+.4f}（差 t={dt:+.2f}）")
        print("  " + "；".join(info))
        print("  " + "  ".join(f"{'✅' if v else '❌'}{k}" for k, v in res.items()) + f"  → {'通过' if all(res.values()) else '未通过'}")
        log.append([name, rows_in["main"], m, "通过" if all(res.values()) else "未通过：" + ",".join(k for k, v in res.items() if not v)])
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for name, mi, mo, note in log:
            w.writerow(["2026-09-29", name, FREQ, "batch13 short-term events", "event", f"h={H};cost=0.3$",
                        "IS event-study S1-S4", round(mi["t"], 2), round(mi["ATR"], 4),
                        "" if mo is None else round(mo["t"], 2), "" if mo is None else round(mo["ATR"], 4),
                        "" if mo is None else round(mo["ATR"] / cn, 2), 4, f"t列为事件t值、净利列为每笔ATR；{note}"])
    print(f"\n已登记 {len(log)} 条试验到 reports/rule_trials.csv")


if __name__ == "__main__":
    main()

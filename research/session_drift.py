# -*- coding: utf-8 -*-
"""
第十四批（补）：亚盘夜间做多 / 伦敦早盘做空（预登记：reports/batch14b_session_drift_prereg.md）。
1H K 线开盘价成交（纽约时间），点差 0.3$；每笔收益 / 入场前 1H ATR，减去当年平均每小时收益 × 持有小时数。
用法：python -m research.session_drift
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from factors.core import atr, trading_day
from research.multifactor_combo import sh
from research.overfitting_tests import dsr, matrix_B, n_eff

SPLIT = pd.Timestamp("2020-01-01")
COST = 0.3


def build():
    d = pd.read_pickle("data/cache/1H.pkl")
    a = atr(d, 24)
    ap = a.shift(1)
    mu = ((d["open"].shift(-1) - d["open"]) / ap).groupby(d.index.year).mean()      # 当年平均每小时收益（ATR）
    ny = d.index.tz_localize("UTC").tz_convert("America/New_York")
    K = pd.DataFrame({"td": trading_day(d.index), "h": ny.hour, "o": d["open"].to_numpy(), "a": ap.to_numpy()})
    O = K.pivot_table(index="td", columns="h", values="o", aggfunc="first")
    A = K.pivot_table(index="td", columns="h", values="a", aggfunc="first")
    cost_now = float((COST / a[d.index >= d.index[-1] - pd.Timedelta(days=365)]).median())
    return O, A, mu, cost_now


def leg(O, A, mu, h0, h1, hours, side):
    r = side * ((O[h1] - O[h0]) / A[h0] - hours * O.index.year.map(mu).to_numpy())
    return r.dropna()


def main():
    O, A, mu, cn = build()
    print(f"当前成本 0.3$ = {cn:.4f} ATR(1H)")
    L = {"B3 AsiaNightLong": {"19": leg(O, A, mu, 19, 2, 7, 1), "20": leg(O, A, mu, 20, 2, 6, 1)},
         "B4 LondonMorningShort": {"19": leg(O, A, mu, 2, 6, 4, -1)}}
    L["B5 Pair"] = {k: pd.concat([L["B3 AsiaNightLong"][k], L["B4 LondonMorningShort"]["19"]], axis=1).dropna().sum(axis=1)
                    for k in ("19", "20")}
    ntr = {"B3 AsiaNightLong": 1, "B4 LondonMorningShort": 1, "B5 Pair": 2}
    from research.failed_factor_combo import frozen_daily
    ref = frozen_daily()
    seg = lambda x, oos: x[x.index >= SPLIT] if oos else x[x.index < SPLIT]

    def st(r, n):
        net = r - n * cn
        return {"n": len(r), "每笔": r.mean() / n, "t": r.mean() / (r.std() / np.sqrt(len(r))), "夏普": sh(net),
                "年正": (net.groupby(net.index.year).sum() > 0).mean(), "年正数": int((net.groupby(net.index.year).sum() > 0).sum())}
    rows = {k: {"IS": st(seg(v["19"], False), ntr[k])} for k, v in L.items()}
    print("\n样本内 2009-2019：")
    for k, v in rows.items():
        x = v["IS"]
        print(f"  {k:22} n={x['n']} 每笔 {x['每笔']:+.4f} ATR (t={x['t']:+.2f}) 净夏普 {x['夏普']:.2f} 年正 {x['年正']:.2f}")
    sel = max(rows, key=lambda k: rows[k]["IS"]["夏普"])
    print(f"选中（样本内净夏普最高）：{sel}")

    print(f"\n{'#' * 80}\n样本外 2020-01 ~ 2026-09（只跑一次）：")
    for k, v in L.items():
        x = st(seg(v["19"], True), ntr[k])
        rows[k]["OOS"] = x
        print(f"  {k:22} n={x['n']} 每笔 {x['每笔']:+.4f} ATR (t={x['t']:+.2f}) 净夏普 {x['夏普']:.2f} 年正 {x['年正数']}/7")
    r = L[sel]["19"]
    o = rows[sel]["OOS"]
    net = r - ntr[sel] * cn
    res = {f"P1 每笔 {o['每笔']:+.4f} ≥ 成本 {cn:.4f} 且 t={o['t']:.2f} ≥ 2": o["每笔"] >= cn and o["t"] >= 2,
           f"P2 样本外年份为正 {o['年正数']}/7 ≥ 5": o["年正数"] >= 5}
    if "20" in L[sel]:
        x20 = st(seg(L[sel]["20"], True), ntr[sel])
        res[f"P3 20:00 入场 每笔 {x20['每笔']:+.4f} > 成本"] = x20["每笔"] > cn
    else:
        res["P3 （B4 无 20:00 变体，按通过计）"] = True
    J = ref.join(net.rename("new"), how="outer").fillna(0.0)
    info, p4, p5 = [], True, True
    for oos in (False, True):
        j = seg(J, oos)
        cc = j.corr()["new"]
        b0, b1 = sh(j["TT30-EW-v1"] + j["HA1H-v1"]), sh(j.sum(axis=1))
        info.append(f"{'外' if oos else '内'} 相关 {cc['TT30-EW-v1']:+.2f}/{cc['HA1H-v1']:+.2f} 组合 {b0:.2f}→{b1:.2f}")
        p4 &= bool((cc[["TT30-EW-v1", "HA1H-v1"]].abs() < 0.3).all())
        p5 &= b1 >= b0
    res["P4 相关 < 0.3：" + "；".join(info)] = p4
    res["P5 组合夏普不降"] = p5
    Bm = matrix_B()
    Bm = Bm.loc[:, Bm.std() > 0]
    nb = n_eff(Bm)
    x = net.reindex(Bm.index.union(net.index)).fillna(0.0)
    dd = dsr(x, (Bm.mean() / Bm.std()).to_numpy(), nb + 27)
    res[f"P6 DSR {dd['DSR']:.3f} ≥ 0.5（N={nb:.1f}+27，全样本夏普 {dd['SR_annual']:.2f}）"] = dd["DSR"] >= 0.5
    print(f"\n判定 {sel}：")
    for k, v in res.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(res.values())
    print("  结论：" + ("全部通过 → 作为候选交给用户" if ok else "未全部通过"))
    yr = net.groupby(net.index.year).sum()
    print("  逐年净收益（ATR·今）：" + " ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))
    net.to_pickle("data/cache/session_drift_selected.pkl")
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for k in L:
            i, oo = rows[k]["IS"], rows[k]["OOS"]
            w.writerow(["2026-09-29", k, "1H", "batch14b session drift", "fixed-time", "entry19;cost=0.3$", "IS net sharpe",
                        round(i["夏普"], 2), round(i["每笔"], 4), round(oo["夏普"], 2), round(oo["每笔"], 4), round(oo["每笔"] / cn, 2),
                        27, ("SELECTED " if k == sel else "") + "净利列为每笔去漂移ATR；逐小时扫描 24 次计入 N"])
    print("已登记 3 条")


if __name__ == "__main__":
    main()

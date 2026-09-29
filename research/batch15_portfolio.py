# -*- coding: utf-8 -*-
"""
第十五批：日线级状态因子作为组合第三成员的检验（预登记：reports/batch15_prereg.md）。
用法：python -m research.batch15_portfolio
"""
from __future__ import annotations

import csv
import itertools

import numpy as np
import pandas as pd

from factors.core import params, rolling_mad_zscore
from factors.registry import get_factors
from paper.specs import get_spec
from research.failed_factor_combo import daily_r, pos_s0
from research.multifactor_combo import Book, sh
from research.overfitting_tests import dsr, matrix_B, n_eff

SPLIT = pd.Timestamp("2020-01-01")
SM = 1.5
NAMES = ["DailyRangePos5", "DailyRangePos21", "DailyMomentum5", "MA200Distance"]
seg = lambda x, oos: x[x.index >= SPLIT] if oos else x[x.index < SPLIT]


def ref_daily():
    out = {}
    for sid in ("TT30-EW-v1", "HA1H-v1"):
        s = get_spec(sid)
        B = Book(s.freq)
        out[sid] = daily_r(B, pos_s0(s.signal(B.df), s.freq, s.entry, s.exit), SM)["r"]
    return pd.DataFrame(out).fillna(0.0)


def check(name, z, B, ref, n_trials):
    d = daily_r(B, pos_s0(z, "1H"), SM)
    J = ref.join(d["r"].rename("new"), how="outer").fillna(0.0)
    res, info = {}, {}
    for oos in (False, True):
        j = seg(J, oos)
        cc = j.corr()["new"]
        info["外" if oos else "内"] = {"夏普": sh(j["new"]), "相关TT30": cc["TT30-EW-v1"], "相关HA1H": cc["HA1H-v1"],
                                      "组合前": sh(j["TT30-EW-v1"] + j["HA1H-v1"]), "组合后": sh(j.sum(axis=1))}
    i, o = info["内"], info["外"]
    od = seg(d, True)
    grid = []
    for e, x in itertools.product((1.0, 1.5, 2.0), (0.0, 0.3, 0.6)):
        g = daily_r(B, pos_s0(z, "1H", e, x), SM)
        grid.append((sh(seg(g, False)["r"]), sh(seg(g, True)["r"])))
    gi, go = sum(a > 0 for a, _ in grid), sum(b > 0 for _, b in grid)
    ins_ok = i["夏普"] >= 0.3 and abs(i["相关TT30"]) < 0.5 and abs(i["相关HA1H"]) < 0.5
    res["IS入选：夏普≥0.3 且相关<0.5"] = ins_ok
    res["1 OOS夏普≥0.3 且多空为正"] = o["夏普"] >= 0.3 and od["long"].sum() > 0 and od["short"].sum() > 0
    res["2 OOS相关<0.5"] = abs(o["相关TT30"]) < 0.5 and abs(o["相关HA1H"]) < 0.5
    res["3 组合夏普内外都上升"] = i["组合后"] > i["组合前"] and o["组合后"] > o["组合前"]
    res[f"4 平原 内{gi}/9 外{go}/9"] = gi >= 7 and go >= 7
    print(f"\n== {name}")
    for k, v in info.items():
        print(f"  样本{k}：单独夏普 {v['夏普']:.2f}，相关 TT30 {v['相关TT30']:+.2f} / HA1H {v['相关HA1H']:+.2f}，组合 {v['组合前']:.2f}→{v['组合后']:.2f}")
    print(f"  样本外多/空（ATR·今）：{od['long'].sum():+.1f} / {od['short'].sum():+.1f}")
    print("  " + "  ".join(f"{'✅' if v else '❌'}{k}" for k, v in res.items()))
    return d, info, res


def main():
    ref = ref_daily()
    B = Book("1H")
    print(f"现状 TT30+HA1H（点差 0.3$）组合夏普：内 {sh(seg(ref.sum(axis=1), False)):.2f} / 外 {sh(seg(ref.sum(axis=1), True)):.2f}")
    Z = {n: get_factors([n])[0](B.df, "1H") for n in NAMES}
    out, passed = {}, []
    for n in NAMES:
        out[n] = check(n, Z[n], B, ref, len(NAMES))
        if all(out[n][2].values()):
            passed.append(n)
    Bm = matrix_B()
    Bm = Bm.loc[:, Bm.std() > 0]
    nb = n_eff(Bm)
    for n in NAMES:
        x = out[n][0]["r"].reindex(Bm.index.union(out[n][0].index)).fillna(0.0)
        r = dsr(x, (Bm.mean() / Bm.std()).to_numpy(), nb + len(NAMES))
        print(f"  DSR {n}：{r['DSR']:.3f}（全样本夏普 {r['SR_annual']:.2f}，N={nb:.1f}+{len(NAMES)}）")
    print(f"\n通过：{passed or '无'}")
    rows = [(n, out[n]) for n in NAMES]
    if len(passed) >= 2:
        zc = rolling_mad_zscore(pd.concat([Z[n] for n in passed], axis=1).fillna(0).mean(axis=1), params("1H")["norm"])
        out["等权"] = check("通过者等权 " + "+".join(passed), zc, B, ref, 1)
        rows.append(("通过者等权", out["等权"]))
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for n, (d, info, res) in rows:
            w.writerow(["2026-09-29", n, "1H", "batch15 daily-state", "hyst", "entry=1.5;exit=0.3;spread=0.3", "IS sharpe≥0.3 & corr<0.5",
                        round(info["内"]["夏普"], 2), "", round(info["外"]["夏普"], 2), "", "", len(rows),
                        ("通过" if all(res.values()) else "未通过") + f"；组合 内 {info['内']['组合前']:.2f}→{info['内']['组合后']:.2f} 外 {info['外']['组合前']:.2f}→{info['外']['组合后']:.2f}"])
    print(f"已登记 {len(rows)} 条")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
TT30 与 HA1H 方向冲突时的组合层规则（预登记：reports/conflict_rules_prereg.md）。
C0 独立 / C1 冲突时 HA1H 平仓 / C2 HA1H 跟随 TT30 / C3 冲突时 TT30 平仓 / C4 都平仓。
用法：python -m research.conflict_rules
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from factors.evaluate import execution_signal, targets
from paper.specs import get_spec
from research.failed_factor_combo import daily_r
from research.multifactor_combo import Book, sh

SPLIT = pd.Timestamp("2020-01-01")


def main():
    tt, ha = get_spec("TT30-EW-v1"), get_spec("HA1H-v1")
    B30, B1 = Book("30MIN"), Book("1H")
    t30 = targets(execution_signal(tt.signal(B30.df), "30MIN"), tt.entry, tt.exit)      # 30MIN 收盘后的目标
    t1 = targets(execution_signal(ha.signal(B1.df), "1H"), ha.entry, ha.exit)           # 1H 收盘后的目标
    # 1H 收盘时可见的 TT30 目标：收盘于 t+1h 的 30MIN K 线（开始于 t+30min）；缺失时用最近已收盘的
    tt_on_1h = t30.reindex(t30.index.union(t1.index + pd.Timedelta("30min"))).ffill().reindex(t1.index + pd.Timedelta("30min"))
    tt_on_1h.index = t1.index
    # 30MIN 收盘时可见的 HA1H 目标：开始于 floor(s+30min, 1h) − 1h 的 1H K 线
    key = (t30.index + pd.Timedelta("30min")).floor("h") - pd.Timedelta("1h")
    ha_on_30 = pd.Series(t1.reindex(t1.index.union(key.unique())).ffill().reindex(key).to_numpy(), index=t30.index).fillna(0)

    conf1 = (np.sign(tt_on_1h) * np.sign(t1) < 0)
    conf30 = (np.sign(ha_on_30) * np.sign(t30) < 0)
    ha_rules = {"C0": t1, "C1": t1.where(~conf1, 0.0), "C2": t1.where(~conf1, tt_on_1h), "C3": t1, "C4": t1.where(~conf1, 0.0)}
    tt_rules = {"C0": t30, "C1": t30, "C2": t30, "C3": t30.where(~conf30, 0.0), "C4": t30.where(~conf30, 0.0)}
    desc = {"C0": "独立（现状）", "C1": "冲突时HA1H平仓", "C2": "HA1H跟随TT30", "C3": "冲突时TT30平仓", "C4": "冲突时都平仓"}

    # 诊断：冲突频率与冲突期间 HA1H 的收益（C0 下）
    p1 = t1.shift(1).fillna(0)
    c_pos = conf1.shift(1).fillna(False).astype(bool)
    r_bar = (p1 * B1.m_atr).fillna(0)
    held = p1 != 0
    print(f"HA1H 持仓时间中与 TT30 方向冲突的比例：样本内 {c_pos[held & (p1.index < SPLIT)].mean():.1%}，样本外 {c_pos[held & (p1.index >= SPLIT)].mean():.1%}")
    for seg, m in (("内", p1.index < SPLIT), ("外", p1.index >= SPLIT)):
        a, b = r_bar[held & c_pos & m], r_bar[held & ~c_pos & m]
        print(f"  样本{seg}：HA1H 每根 K 线收益（ATR，去漂移毛）冲突时 {a.mean():+.4f}（{len(a)} 根） vs 不冲突 {b.mean():+.4f}（{len(b)} 根）")

    rows, D = [], {}
    for c in ha_rules:
        for mult, tag in ((1.0, "0.2"), (1.5, "0.3")):
            d = (daily_r(B1, ha_rules[c].shift(1).fillna(0), mult)["r"].rename("HA")
                 .to_frame().join(daily_r(B30, tt_rules[c].shift(1).fillna(0), mult)["r"].rename("TT"), how="outer").fillna(0))
            s = d.sum(axis=1)
            D[(c, tag)] = s
            rows.append({"规则": c, "说明": desc[c], "点差": tag, "IS": sh(s[s.index < SPLIT]), "OOS": sh(s[s.index >= SPLIT]),
                         "IS_HA": sh(d.HA[d.index < SPLIT]), "OOS_HA": sh(d.HA[d.index >= SPLIT])})
    T = pd.DataFrame(rows)
    t2 = T[T["点差"] == "0.2"].set_index("规则")
    sel = t2["IS"].idxmax()
    print("\n组合（TT30+HA1H）日度 ATR·今 夏普：")
    print(T.round(2).to_string(index=False))
    t3 = T[T["点差"] == "0.3"].set_index("规则")
    print(f"\n样本内选中：{sel}（{desc[sel]}）IS {t2.loc[sel, 'IS']:.2f} vs C0 {t2.loc['C0', 'IS']:.2f}")
    ok = sel != "C0" and t2.loc[sel, "OOS"] >= t2.loc["C0", "OOS"] and t3.loc[sel, "OOS"] >= t3.loc["C0", "OOS"]
    print("判定：" + ("通过" if ok else ("样本内没有规则优于现状" if sel == "C0" else "样本外不优于现状 → 未通过")))
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for c in ha_rules:
            w.writerow(["2026-09-29", "TT30/HA1H 冲突规则", "30MIN+1H", "portfolio conflict", c, desc[c], "IS portfolio sharpe",
                        round(t2.loc[c, "IS"], 2), "", round(t2.loc[c, "OOS"], 2), "", "", 5,
                        ("SELECTED " if c == sel else "") + f"点差0.3: IS {t3.loc[c, 'IS']:.2f} OOS {t3.loc[c, 'OOS']:.2f}"])
    print("已登记 5 条")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
TT30 + HA1H 的仓位管理规则枚举（预登记：reports/position_sizing_prereg.md）。

逐笔交易来自冻结规格的信号 + 统一执行规则；两个策略共用一个账户，入场时按已实现净值计算盎司数（1 盎司 = 0.01 手，整数，至少 1）。
成本：点差 0.3$/开平（实测口径）+ 过夜费。回撤按日度净值（两个策略每天换日前平仓，日终净值 = 市值）。
用法：python -m research.position_sizing          # 样本内选择
      python -m research.position_sizing --oos    # 样本外只跑一次 + 登记
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np
import pandas as pd

from factors.core import atr, params, trading_day
from factors.evaluate import SWAP, swap_units
from research.failed_factor_combo import pos_s0

SPLIT = pd.Timestamp("2020-01-01")
SPREAD_REAL = 0.3
E0 = 10_000.0
DD_SELECT = 0.20 / 1.5
GRIDS = {"R1 固定手数": [1, 2, 3, 5, 8, 10, 15, 20],
         "R2 按净值比例": [1, 2, 3, 5, 8, 10, 15, 20],
         "R3 ATR定风险": [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02],
         "R4 ATR定风险+回撤降仓": [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02],
         "R5 ATR定风险+单日止损": [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]}


def trades() -> pd.DataFrame:
    """逐笔交易：entry/exit 时间、策略、方向、每盎司净盈亏、入场 ATR（美元）。"""
    from paper.specs import get_spec
    out = []
    for sid in ("TT30-EW-v1", "HA1H-v1"):
        s = get_spec(sid)
        d = pd.read_pickle(f"data/cache/{s.freq}.pkl")
        pos = pos_s0(s.signal(d), s.freq, s.entry, s.exit).to_numpy()
        move = (d["open"].shift(-1) - d["open"]).fillna(0).to_numpy()
        swu = swap_units(d.index).to_numpy()
        a = atr(d, params(s.freq)["atr"]).shift(1).to_numpy()
        idx = d.index
        chg = np.r_[True, pos[1:] != pos[:-1]]
        seg = np.cumsum(chg)
        for g in np.unique(seg[pos != 0]):
            b = np.where(seg == g)[0]
            i0, i1 = b[0], b[-1]
            side = pos[i0]
            pnl = side * move[b].sum() - SPREAD_REAL - swu[b].sum() * SWAP
            out.append((idx[i0], idx[min(i1 + 1, len(idx) - 1)], sid, side, pnl, a[i0]))
    T = pd.DataFrame(out, columns=["entry", "exit", "sid", "side", "pnl_oz", "atr"]).dropna()
    T["exit_day"] = trading_day(pd.DatetimeIndex(T["exit"]))
    T["entry_day"] = trading_day(pd.DatetimeIndex(T["entry"]))
    return T.sort_values("entry").reset_index(drop=True)


def simulate(T: pd.DataFrame, rule: str, x: float, e0: float = E0) -> pd.Series:
    """事件驱动：按时间处理入场/出场（同一时刻先出场后入场）。返回日度净值。"""
    ev = [(r.exit, 0, i) for i, r in T.iterrows()] + [(r.entry, 1, i) for i, r in T.iterrows()]
    ev.sort()
    eq, peak = e0, e0
    size = {}
    day_pnl: dict = {}
    daily: dict = {}
    for t, kind, i in ev:
        r = T.loc[i]
        if kind == 0:
            if i not in size:
                continue
            p = size.pop(i) * r.pnl_oz
            eq += p
            peak = max(peak, eq)
            day_pnl[r.exit_day] = day_pnl.get(r.exit_day, 0.0) + p
            daily[r.exit_day] = eq
            if eq <= 0:
                break
            continue
        if rule.startswith("R5") and day_pnl.get(r.entry_day, 0.0) < -0.02 * eq:
            continue
        if rule.startswith("R1"):
            n = x
        elif rule.startswith("R2"):
            n = math.floor(eq / 10_000 * x)
        else:
            n = math.floor(eq * x / r.atr)
            if rule.startswith("R4") and eq < 0.9 * peak:
                n = math.floor(n / 2)
        size[i] = max(int(n), 1)
    s = pd.Series(daily).sort_index()
    return s


def metrics(eq: pd.Series, e0: float = E0) -> dict:
    if len(eq) == 0:
        return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    full = pd.concat([pd.Series([e0], index=[eq.index[0] - pd.Timedelta("1D")]), eq])
    dd = (full / full.cummax() - 1).min()
    dp = full.diff().dropna()
    dp = dp[dp != 0]
    neg = (dp < 0).astype(int)
    streak = int(neg.groupby((neg != neg.shift()).cumsum()).sum().max()) if len(neg) else 0
    ret = full.pct_change().dropna()
    return {"期末净值": round(float(eq.iloc[-1])), "年化": (eq.iloc[-1] / e0) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0,
            "最大回撤": float(dd), "盈利天数占比": float((dp > 0).mean()), "最长连亏天数": streak,
            "单日最大亏损": float(ret.min()), "日度夏普": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else np.nan}


def run(T, seg, e0=E0):
    TT = T[T.entry < SPLIT] if seg == "IS" else T[T.entry >= SPLIT]
    rows = []
    for rule, grid in GRIDS.items():
        for x in grid:
            m = metrics(simulate(TT.reset_index(drop=True), rule, x, e0), e0)
            rows.append({"规则": rule, "档": x, **m})
    return pd.DataFrame(rows)


def fmt(D):
    D = D.copy()
    for c in ("年化", "最大回撤", "盈利天数占比", "单日最大亏损"):
        D[c] = (D[c] * 100).round(1).astype(str) + "%"
    D["日度夏普"] = D["日度夏普"].round(2)
    return D.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true")
    args = ap.parse_args()
    T = trades()
    print(f"逐笔交易：{T.groupby('sid').size().to_dict()}；每盎司净盈亏合计 样本内 {T[T.entry < SPLIT].pnl_oz.sum():+.0f}$ / 样本外 {T[T.entry >= SPLIT].pnl_oz.sum():+.0f}$（点差 0.3$）")
    IS = run(T, "IS")
    print(f"\n样本内 2009-2019（起始 {E0:,.0f}$）：\n" + fmt(IS))
    ok = IS[IS["最大回撤"] >= -DD_SELECT]
    sel = ok["年化"].idxmax()
    print(f"\n选中（样本内回撤 ≤ {DD_SELECT:.1%} 中年化最高）：{IS.loc[sel, '规则']} 档 {IS.loc[sel, '档']}  "
          f"年化 {IS.loc[sel, '年化']:.1%} 回撤 {IS.loc[sel, '最大回撤']:.1%}")
    if not args.oos:
        print("（样本外只在 --oos 时运行一次）")
        return
    OOS = run(T, "OOS")
    print(f"\n样本外 2020-01 ~ 2026-09（重新从 {E0:,.0f}$ 起）——只跑一次：\n" + fmt(OOS))
    rule, x = IS.loc[sel, "规则"], IS.loc[sel, "档"]
    o = OOS.loc[sel]
    g = OOS[OOS["规则"] == rule].reset_index(drop=True)
    k = list(GRIDS[rule]).index(x)
    nb = g.iloc[max(k - 1, 0):k + 2]
    res = {"V1 样本外回撤 ≤ 20%": o["最大回撤"] >= -0.20, "V2 样本外年化 > 0": o["年化"] > 0,
           "V3 相邻档样本外回撤 ≤ 30%": bool((nb["最大回撤"] >= -0.30).all())}
    print(f"\n判定 {rule} 档 {x}：样本外 年化 {o['年化']:.1%} 回撤 {o['最大回撤']:.1%} 盈利天数 {o['盈利天数占比']:.0%} 最长连亏 {o['最长连亏天数']} 天")
    for kk, v in res.items():
        print(f"  {'✅' if v else '❌'} {kk}")
    base = OOS[(OOS["规则"] == "R1 固定手数") & (OOS["档"] == 1)].iloc[0]
    print(f"  对照 固定 1 盎司（现状）：样本外年化 {base['年化']:.1%} 回撤 {base['最大回撤']:.1%}")
    TO = T[T.entry >= SPLIT].reset_index(drop=True)
    for e0 in (5_000.0,):
        m = metrics(simulate(TO, rule, x, e0), e0)
        print(f"  敏感性：起始 {e0:,.0f}$ → 年化 {m['年化']:.1%} 回撤 {m['最大回撤']:.1%}")
    eq = simulate(TO, rule, x)
    yr = eq.groupby(eq.index.year).last()
    prev = pd.concat([pd.Series([E0]), yr.iloc[:-1]]).to_numpy()
    print("  逐年收益：" + "  ".join(f"{y}:{v / p - 1:+.0%}" for y, v, p in zip(yr.index, yr.to_numpy(), prev)))
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for i in IS.index:
            w.writerow(["2026-09-29", "TT30+HA1H 仓位管理", "trade", "portfolio sizing", IS.loc[i, "规则"], f"x={IS.loc[i, '档']};E0=10000;spread=0.3",
                        "IS CAGR s.t. IS maxDD<=13.3%", round(IS.loc[i, "日度夏普"], 2), round(IS.loc[i, "年化"], 4),
                        round(OOS.loc[i, "日度夏普"], 2), round(OOS.loc[i, "年化"], 4), "", len(IS),
                        ("SELECTED " if i == sel else "") + f"净利列为年化；IS回撤 {IS.loc[i, '最大回撤']:.1%} OOS回撤 {OOS.loc[i, '最大回撤']:.1%}"])
    print(f"\n已登记 {len(IS)} 条试验到 reports/rule_trials.csv")


if __name__ == "__main__":
    main()

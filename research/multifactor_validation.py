# -*- coding: utf-8 -*-
"""
30MIN 多因子"簇代表等权"组合的验证（预登记）。
  1) 滚动选择（walk-forward）：每年 Y 只用 Y 之前的信息重做"成分池 → 聚类 → 簇代表"，交易 Y 年；对比固定成分（2009-2019 选定）
     成分池：该年之前逐年 ATR 边际（evaluate 的 per_trip_atr_year，逐年独立计算）均值 > 0 且为正年份 ≥ 60%，至少 3 年
     聚类：Y 之前 5 年的因子值 Spearman 相关，|ρ|≥0.6；代表：之前各年边际均值最高者
  2) 参数平原：entry×exit；聚类门槛 |ρ|≥{0.5,0.6,0.7}；成分池年份门槛 {50%,60%,70%}
  3) 成本：点差 ×2 / ×3
  4) 逐年、多空；参数层 CSCV
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from factors.core import params, rolling_mad_zscore, trading_day
from factors.evaluate import SPREAD, SWAP, execution_signal, positions
from factors.registry import REGISTRY, get_factors
from research.factor_correlation import factor_values
from research.multifactor_combo import Book, sh
from research.overfitting_tests import cscv, n_eff

FREQ, SPLIT = "30MIN", pd.Timestamp("2020-01-01")
get_factors()
NAMES = [n for n, s in REGISTRY.items() if FREQ in s.freqs]
EDGES = {}
for n in NAMES:
    try:
        m = json.load(open(f"reports/factors/{n}.json", encoding="utf-8"))["results"][FREQ]
        EDGES[n] = {int(k): v for k, v in m["atr_edge"]["per_trip_atr_year"].items()}
    except Exception:
        pass
NAMES = [n for n in NAMES if n in EDGES]
F = factor_values(FREQ, NAMES).fillna(0.0)
B = Book(FREQ)
NORM = params(FREQ)["norm"]


def select(before_year: int, ratio_min=0.6, rho=0.6, corr_years=None):
    pool = {}
    for n in NAMES:
        v = [x for y, x in EDGES[n].items() if y < before_year]
        if len(v) >= 3 and np.mean(v) > 0 and np.mean([x > 0 for x in v]) >= ratio_min:
            pool[n] = np.mean(v)
    if not pool:
        return []
    names = list(pool)
    if len(names) == 1:
        return names
    lo = str(before_year - (corr_years or 99)) if corr_years else None
    sub = F.loc[lo:str(before_year - 1), names]
    C = sub.iloc[::4].rank().corr().fillna(0)
    lab = fcluster(linkage(squareform(1 - C.abs().values, checks=False), "average"), t=1 - rho, criterion="distance")
    return [max([n for n, l in zip(names, lab) if l == c], key=lambda n: pool[n]) for c in sorted(set(lab))]


_zcache = {}


def composite(reps):
    key = tuple(sorted(reps))
    if key not in _zcache:
        _zcache[key] = rolling_mad_zscore(F[list(key)].mean(axis=1), NORM)
    return _zcache[key]


def run(z, entry=1.5, exit_=0.3, sx=1.0):
    pos = positions(execution_signal(z, FREQ), entry, exit_)
    dpos = pos.diff().abs().fillna(pos.abs())
    sw = pos.abs() * B.swu
    pa = (pos * B.m_atr - dpos * SPREAD * sx / 2 / B.atr_now - sw * SWAP / B.atr_now).fillna(0)
    pu = (pos * B.move - dpos * SPREAD * sx / 2 - sw * SWAP).fillna(0)
    return pa, pu, pos


def daily(x):
    return x.groupby(trading_day(x.index)).sum()


def main():
    ins = B.df.index < SPLIT
    fixed = select(2020)
    print(f"固定成分（2009-2019 选定）：{fixed}")
    zf = composite(fixed)

    # 1) 滚动选择
    wa, wu, picks = [], [], []
    for y in range(2012, 2027):
        reps = select(y, corr_years=5)
        if not reps:
            continue
        pa, pu, _ = run(composite(reps))
        m = B.df.index.year == y
        wa.append(pa[m]); wu.append(pu[m]); picks.append((y, len(reps)))
    wa, wu = pd.concat(wa), pd.concat(wu)
    fa, fu, _ = run(zf)
    print("\n[1] 滚动选择 每年成分数:", " ".join(f"{y}:{k}" for y, k in picks))
    for tag, a, u in (("滚动选择", wa, wu), ("固定成分", fa.loc[wa.index], fu.loc[wa.index])):
        d = daily(a); o = d.index >= SPLIT
        print(f"   {tag}: 夏普 2012-2019 {sh(d[~o]):+.2f} | 2020后 {sh(d[o]):+.2f} | 全期 {sh(d):+.2f}   美元 2020后 {u[u.index >= SPLIT].sum():+.0f}")

    # 2) 参数平原
    grid = {}
    t = {}
    for e, x in itertools.product((1.0, 1.25, 1.5, 1.75, 2.0), (0.0, 0.3, 0.6)):
        pa, pu, _ = run(zf, e, x); d = daily(pa); grid[f"thr|{e}|{x}"] = d
        t.setdefault(x, {})[e] = f"{sh(d[d.index < SPLIT]):.2f} / {sh(d[d.index >= SPLIT]):.2f}"
    print("\n[2a] 开平仓阈值（夏普 内/外）\n" + pd.DataFrame(t).rename_axis("entry＼exit").to_string())
    rows = []
    for rho in (0.5, 0.6, 0.7):
        for rm in (0.5, 0.6, 0.7):
            reps = select(2020, ratio_min=rm, rho=rho)
            pa, pu, _ = run(composite(reps)); d = daily(pa); grid[f"sel|{rho}|{rm}"] = d
            rows.append({"聚类|ρ|≥": rho, "年份门槛": rm, "成分数": len(reps), "夏普内": round(sh(d[d.index < SPLIT]), 2), "夏普外": round(sh(d[d.index >= SPLIT]), 2)})
    print("\n[2b] 选择门槛\n" + pd.DataFrame(rows).to_string(index=False))

    # 3) 成本
    print("\n[3] 成本敏感性（夏普 内/外 | 美元 内/外）")
    for sx in (1, 2, 3):
        pa, pu, _ = run(zf, sx=sx); d = daily(pa)
        print(f"   点差×{sx}: {sh(d[d.index < SPLIT]):+.2f} / {sh(d[d.index >= SPLIT]):+.2f} | {pu[ins].sum():+.0f} / {pu[~ins].sum():+.0f}")

    # 4) 逐年、多空、逐笔
    pa, pu, pos = run(zf)
    yr = pd.DataFrame({"美元": pu.groupby(pu.index.year).sum().round(0), "ATR·今": pa.groupby(pa.index.year).sum().round(1)}).T
    print("\n[4] 逐年\n" + yr.to_string())
    seg = (pos != pos.shift(1)).cumsum()
    tr = pu[pos != 0].groupby(seg[pos != 0]).agg(["sum", "size"])
    st = pos[pos != 0].groupby(seg[pos != 0]).first()
    t0 = pd.Series(pos[pos != 0].index, index=pos[pos != 0].index).groupby(seg[pos != 0]).first()
    for tag, m in (("样本内", t0 < SPLIT), ("样本外", t0 >= SPLIT)):
        s = tr["sum"][m]; side = st[m]
        print(f"   {tag}: {len(s)} 笔，胜率 {(s > 0).mean():.2f}，平均盈 {s[s > 0].mean():+.2f}$ / 亏 {s[s <= 0].mean():+.2f}$，"
              f"多头 {s[side > 0].sum():+.0f}$ / 空头 {s[side < 0].sum():+.0f}$，平均持仓 {tr['size'][m].mean() / 2:.1f} 小时")
    eq = pu.cumsum(); print(f"   最大回撤 {(eq - eq.cummax()).min():+.0f}$")
    M = pd.DataFrame(grid).fillna(0)
    c = cscv(M)
    print(f"\n[5] 参数层 CSCV（{M.shape[1]} 个配置，N_eff≈{n_eff(M):.1f}）：PBO={c['PBO']:.3f}，样本内最优在样本外夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f}")
    daily(fa).to_pickle("data/cache/mf30_fixed_daily.pkl")


if __name__ == "__main__":
    main()

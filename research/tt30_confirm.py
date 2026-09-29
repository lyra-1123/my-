# -*- coding: utf-8 -*-
"""
TT30 信号强弱与确认条件（预登记：reports/tt30_confirm_prereg.md）。
交易级操作：过滤 = 不满足条件的交易整笔不做；加权 = 满足条件的交易 2 盎司。随机对照 200 次。
用法：python -m research.tt30_confirm
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from factors.core import htf_feature, relative_volume
from factors.evaluate import execution_signal, targets
from paper.specs import get_spec
from research.failed_factor_combo import daily_r
from research.multifactor_combo import Book, sh

SPLIT = pd.Timestamp("2020-01-01")
SM = 1.5                 # 点差 0.3$ = 0.2$ × 1.5
NRAND = 200


def setup():
    tt, ha = get_spec("TT30-EW-v1"), get_spec("HA1H-v1")
    B = Book("30MIN")
    d = B.df
    z = tt.signal(d)
    pos = targets(execution_signal(z, "30MIN"), tt.entry, tt.exit).shift(1).fillna(0.0)
    p = pos.to_numpy()
    new = (p != 0) & (np.r_[0.0, p[:-1]] != p)
    tid = np.cumsum(new) * (p != 0)                       # 每根 K 线所属交易编号（0 = 空仓）
    e = np.where(new)[0]
    s = e - 1                                             # 信号 K 线
    side = p[e]
    # 条件
    zs = np.abs(z.to_numpy()[s])
    d1 = htf_feature(d, "30MIN", "1D", lambda h: np.sign(h["close"] - h["close"].rolling(50, min_periods=50).mean())).to_numpy()[s]
    rv = relative_volume(d, 20, True).to_numpy()[s]
    ny = d.index[e].tz_localize("UTC").tz_convert("America/New_York").hour
    B1 = Book("1H")
    t1 = targets(execution_signal(ha.signal(B1.df), "1H"), ha.entry, ha.exit)
    key = (d.index[s] + pd.Timedelta("30min")).floor("h") - pd.Timedelta("1h")
    ha_at = t1.reindex(t1.index.union(key.unique())).ffill().reindex(key).fillna(0).to_numpy()
    K = {"K1 强信号|z|≥2": zs >= 2.0, "K2 日线趋势同向": d1 == side, "K3 放量": rv >= 1.0,
         "K4 非亚盘": ~((ny >= 18) | (ny < 2)), "K5 HA1H不反向": ha_at * side >= 0}
    tr_time = d.index[e]
    return B, pos, tid, K, zs, side, tr_time


def with_weights(B, pos, tid, w):
    """w：每笔交易的权重（0 / 1 / 2），按交易编号映射到每根 K 线。"""
    ww = np.r_[0.0, w][tid]
    return daily_r(B, pos * ww, SM)


def main():
    B, pos, tid, K, zs, side, tt = setup()
    ntr = len(zs)
    ins_tr = tt < SPLIT
    base = daily_r(B, pos, SM)
    seg = lambda x, oos: x[x.index >= SPLIT] if oos else x[x.index < SPLIT]
    b_is, b_oos = sh(seg(base, False)["r"]), sh(seg(base, True)["r"])
    print(f"TT30 共 {ntr} 笔（样本内 {ins_tr.sum()}）；现状夏普（ATR·今，点差 0.3$）内 {b_is:.2f} / 外 {b_oos:.2f}")

    # 诊断：|z| 分档的每笔去漂移收益
    g = (pos * B.m_atr).groupby(tid).sum().drop(0, errors="ignore").to_numpy()
    q = pd.qcut(zs[ins_tr], 5, duplicates="drop")
    print("\n诊断（样本内）：入场信号 |z| 分档 → 每笔去漂移毛收益（ATR）")
    print(pd.Series(g[ins_tr]).groupby(q, observed=True).agg(["count", "mean"]).round(3).to_string())
    for k, c in K.items():
        print(f"  {k}：满足 {c[ins_tr].mean():.0%}；满足/不满足 每笔 {g[ins_tr & c].mean():+.3f} / {g[ins_tr & ~c].mean():+.3f}")

    rng = np.random.default_rng(7)
    rows, D = [], {}
    for k, c in K.items():
        pr = c[ins_tr].mean()
        for mode in ("过滤", "加权"):
            w = np.where(c, 1.0, 0.0) if mode == "过滤" else np.where(c, 2.0, 1.0)
            d = with_weights(B, pos, tid, w)
            D[(k, mode)] = d
            rnd_is, rnd_oos = [], []
            for _ in range(NRAND):
                m = rng.random(ntr) < pr
                wr = np.where(m, 1.0, 0.0) if mode == "过滤" else np.where(m, 2.0, 1.0)
                dr = with_weights(B, pos, tid, wr)
                rnd_is.append(sh(seg(dr, False)["r"]))
                rnd_oos.append(sh(seg(dr, True)["r"]))
            yb = base["r"].groupby(base.index.year).sum()
            yv = d["r"].groupby(d.index.year).sum()
            rows.append({"条件": k, "方式": mode, "IS": sh(seg(d, False)["r"]), "随机IS90%": np.quantile(rnd_is, .9),
                         "OOS": sh(seg(d, True)["r"]), "随机OOS中位": np.median(rnd_oos),
                         "年份更好": float((yv.reindex(yb.index).fillna(0) > yb).mean()),
                         "OOS多": seg(d, True)["long"].sum(), "OOS空": seg(d, True)["short"].sum()})
    T = pd.DataFrame(rows)
    sel = T["IS"].idxmax()
    pd.set_option("display.width", 250)
    print(f"\n{'#' * 90}\n全部 10 个（现状 内 {b_is:.2f} / 外 {b_oos:.2f}）：")
    print(T.round(2).to_string(index=False))
    x = T.loc[sel]
    c = {"IS 高于现状与随机90%": x["IS"] > b_is and x["IS"] > x["随机IS90%"],
         "OOS 不低于现状且高于随机中位": x["OOS"] >= b_oos and x["OOS"] > x["随机OOS中位"],
         "≥60% 年份更好": x["年份更好"] >= 0.6, "OOS 多空都为正": x["OOS多"] > 0 and x["OOS空"] > 0}
    print(f"\n样本内选中：{x['条件']} · {x['方式']}")
    print("判定：" + "  ".join(f"{'✅' if v else '❌'}{k}" for k, v in c.items()) + f" → {'通过' if all(c.values()) else '未通过'}")
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        wtr = csv.writer(fh)
        for i, r in T.iterrows():
            wtr.writerow(["2026-09-29", "TT30 确认条件", "30MIN", "trend-tail", r["方式"], r["条件"], "IS sharpe_atr_now @0.3 vs base & random",
                          round(r["IS"], 2), "", round(r["OOS"], 2), "", "", 10,
                          ("SELECTED " if i == sel else "") + f"随机IS90% {r['随机IS90%']:.2f} 随机OOS中位 {r['随机OOS中位']:.2f} 年份更好 {r['年份更好']:.2f}"])


if __name__ == "__main__":
    main()

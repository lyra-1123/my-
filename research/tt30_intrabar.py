# -*- coding: utf-8 -*-
"""
TT30 用盘中临时 z 提前入场/出场（预登记：reports/tt30_intrabar_prereg.md）。

临时 z：每根 5MIN 收盘时，当前 30MIN K 线用已走部分（累计高/低/收/量），之前的 30MIN K 线用完整数据。
  - 两个因子的滚动和（路径长度、ln 相对量、TP×V、V）= 前 n−1 根完整值之和 + 当前临时值；ATR 用 Wilder 递推一步；
  - MAD_Z：均值、标准差精确包含当前临时值；滚动中位数与 MAD 用上一根的值（近似，窗口 1000）；
  - 30MIN 收盘那一刻用正式 z（与 TT30 完全一致）。
用法：python -m research.tt30_intrabar
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from numba import njit

from factors.core import EPS, atr, rolling_mad_zscore, trading_day
from paper.specs import get_spec
from research.multifactor_combo import sh

SPLIT = pd.Timestamp("2020-01-01")
N, W, VB = 32, 1000, 20
N2 = 2 * N


def mad_parts(x: pd.Series, window: int = W):
    """与 core.rolling_mad_zscore 相同的中间量：med、scale、截断后序列。"""
    minp = max(30, window // 4)
    med = x.rolling(window, min_periods=minp).median()
    mad = (x - med).abs().rolling(window, min_periods=minp).median()
    scale = (1.4826 * mad).where(mad > EPS)
    clipped = x.clip(lower=med - 3 * scale, upper=med + 3 * scale)
    return med, scale, clipped


def partial_z(xp, i, x_full: pd.Series):
    """当前 K 线取临时值 xp 时的 MAD_Z（i 为该 30MIN K 线的位置）。"""
    med, scale, cl = mad_parts(x_full)
    s1 = cl.rolling(W - 1, min_periods=W - 1).sum().shift(1).to_numpy()[i]
    s2 = (cl ** 2).rolling(W - 1, min_periods=W - 1).sum().shift(1).to_numpy()[i]
    m, sc = med.shift(1).to_numpy()[i], scale.shift(1).to_numpy()[i]
    c = np.where(np.isfinite(sc), np.clip(xp, m - 3 * sc, m + 3 * sc), xp)
    mu = (s1 + c) / W
    var = (s2 + c ** 2 - W * mu ** 2) / (W - 1)
    sd = np.sqrt(np.clip(var, 0, None))
    return np.where(sd > EPS, (c - mu) / (sd + EPS), 0.0)


def provisional():
    d30 = pd.read_pickle("data/cache/30MIN.pkl")
    d5 = pd.read_pickle("data/cache/5MIN.pkl")
    C, H, L, V = (d30[k] for k in ("close", "high", "low", "volume"))
    # 30MIN 完整值
    absd = C.diff().abs()
    tod = d30.index.hour * 60 + d30.index.minute
    base = V.groupby(tod).transform(lambda s: s.shift(1).rolling(VB, min_periods=max(5, VB // 4)).median())
    lnrv = np.log((V / (base + EPS)).clip(lower=EPS))
    er = (C - C.shift(N)) / (absd.rolling(N, min_periods=N).sum() + EPS)
    rawA = er * (1 + lnrv.rolling(N, min_periods=N).mean().clip(lower=0))
    tp = (H + L + C) / 3
    a30 = atr(d30, N)
    rawB = (C - (tp * V).rolling(N2, min_periods=N2).sum() / (V.rolling(N2, min_periods=N2).sum() + EPS)) / (a30 + EPS)
    zA, zB = rolling_mad_zscore(rawA, W), rolling_mad_zscore(rawB, W)
    rawZ = 0.5 * zA.fillna(0) + 0.5 * zB.fillna(0)
    z_full = rolling_mad_zscore(rawZ, W)
    # 与冻结规格对账
    tt = get_spec("TT30-EW-v1")
    z_spec = tt.signal(d30)
    print(f"复现 TT30 z 与规格的最大差异：{np.nanmax(np.abs(z_full - z_spec)):.2e}")

    # 5MIN 行 → 所属 30MIN K 线与临时 OHLCV
    b = d5.index.floor("30min")
    i = d30.index.get_indexer(b)
    assert (i >= 0).all()
    g = pd.Series(i, index=d5.index)
    Hp = d5["high"].groupby(g.values).cummax().to_numpy()
    Lp = d5["low"].groupby(g.values).cummin().to_numpy()
    Vp = d5["volume"].groupby(g.values).cumsum().to_numpy()
    Cp = d5["close"].to_numpy()
    last = np.r_[i[1:] != i[:-1], True]

    sh1 = lambda s, k: s.rolling(k, min_periods=k).sum().shift(1).to_numpy()[i]
    Cprev = C.shift(1).to_numpy()[i]
    net = Cp - C.shift(N).to_numpy()[i]
    path = sh1(absd, N - 1) + np.abs(Cp - Cprev)
    rv_p = np.log(np.clip(Vp / (base.to_numpy()[i] + EPS), EPS, None))
    vt = np.clip((sh1(lnrv, N - 1) + rv_p) / N, 0, None)
    A_p = net / (path + EPS) * (1 + vt)
    tpp = (Hp + Lp + Cp) / 3
    vwap = (sh1(tp * V, N2 - 1) + tpp * Vp) / (sh1(V, N2 - 1) + Vp + EPS)
    ap = a30.shift(1).to_numpy()[i]
    tr = np.maximum.reduce([Hp - Lp, np.abs(Hp - Cprev), np.abs(Lp - Cprev)])
    atr_p = ap + (tr - ap) / N
    B_p = (Cp - vwap) / (atr_p + EPS)
    zA_p = np.nan_to_num(partial_z(A_p, i, rawA))
    zB_p = np.nan_to_num(partial_z(B_p, i, rawB))
    z_p = partial_z(0.5 * zA_p + 0.5 * zB_p, i, rawZ)
    zf = z_full.to_numpy()[i]
    ok = last & np.isfinite(zf) & np.isfinite(z_p)
    print(f"收盘时刻 临时 z 与正式 z 的差异（近似误差）：中位 {np.median(np.abs(z_p[ok] - zf[ok])):.4f}，99% 分位 {np.quantile(np.abs(z_p[ok] - zf[ok]), .99):.4f}")
    z = np.where(last, zf, z_p)
    # 执行层：所属 30MIN K 线开始于纽约 [16:00, 18:00) 时置 0（与 execution_signal 相同）
    ny = b.tz_localize("UTC").tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    z = np.where((mins >= 16 * 60) & (mins < 18 * 60), 0.0, z)
    minute_in_bin = ((d5.index - b).total_seconds() // 60).astype(int)
    half = np.asarray(minute_in_bin == 10)          # 5MIN 标签 :10/:40 → 收盘于 K 线走完一半
    return d5, z, last, half


@njit(cache=True)
def _targets(z, upd_entry, upd_exit, entry, exit_):
    n = len(z)
    out = np.zeros(n)
    cur = 0.0
    for k in range(n):
        x = z[k]
        if np.isfinite(x):
            if upd_entry[k] and x > entry:
                cur = 1.0
            elif upd_entry[k] and x < -entry:
                cur = -1.0
            elif upd_exit[k] and abs(x) < exit_:
                cur = 0.0
        out[k] = cur
    return out


def main():
    d5, z, last, half = provisional()
    allr = np.ones(len(z), bool)
    V = {"V0 仅30分钟收盘（现状）": (last, last), "V1 每5分钟开平": (allr, allr), "V2 每5分钟开仓/收盘平仓": (allr, last),
         "V3 收盘开仓/每5分钟平仓": (last, allr), "V4 每15分钟开平": (last | half, last | half)}
    move = (d5["open"].shift(-1) - d5["open"]).fillna(0).to_numpy()
    td = trading_day(d5.index)
    rows = []
    for k, (ue, ux) in V.items():
        pos = np.r_[0.0, _targets(z, ue, ux, 1.5, 0.3)[:-1]]
        dp = np.abs(np.diff(np.r_[0.0, pos]))
        r = {"变体": k}
        for seg, m in (("内", d5.index < SPLIT), ("外", d5.index >= SPLIT)):
            yrs = 11 if seg == "内" else 6.75
            trips = dp[m].sum() / 2
            r[f"笔/年{seg}"] = round(trips / yrs)
            r[f"持仓小时{seg}"] = round((pos[m] != 0).sum() * 5 / 60 / max(trips, 1), 1)
            for s in (0.3, 0.4):
                net = pos[m] * move[m] - dp[m] * s / 2
                r[f"夏普{seg}@{s}"] = round(sh(pd.Series(net).groupby(td[m]).sum()), 2)
                r[f"美元{seg}@{s}"] = round(float(net.sum()))
            net = pos[m] * move[m] - dp[m] * 0.15
            r[f"多{seg}$"] = round(float(net[pos[m] > 0].sum()))
            r[f"空{seg}$"] = round(float(net[pos[m] < 0].sum()))
        rows.append(r)
    T = pd.DataFrame(rows).set_index("变体")
    pd.set_option("display.width", 250)
    for seg in ("内", "外"):
        print(f"\n样本{'内 2009-2019' if seg == '内' else '外 2020-2026.09'}：\n" + T[[c for c in T.columns if seg in c]].to_string())
    base = T.iloc[0]
    cand = T.iloc[1:]
    sel = cand["夏普内@0.3"].idxmax()
    x = T.loc[sel]
    c = {"样本内夏普高于 V0": x["夏普内@0.3"] > base["夏普内@0.3"], "样本外夏普不低于 V0": x["夏普外@0.3"] >= base["夏普外@0.3"],
         "样本外多空都为正": x["多外$"] > 0 and x["空外$"] > 0, "点差0.4 样本外不低于 V0": x["夏普外@0.4"] >= base["夏普外@0.4"]}
    print(f"\n样本内选中：{sel}")
    print("判定：" + "  ".join(f"{'✅' if v else '❌'}{kk}" for kk, v in c.items()) + f" → {'通过' if all(c.values()) else '未通过'}")
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for k, r in T.iterrows():
            w.writerow(["2026-09-29", "TT30 盘中临时z执行", "30MIN→5MIN", "trend-tail", "intrabar", k, "IS usd sharpe @0.3 vs V0",
                        r["夏普内@0.3"], r["美元内@0.3"], r["夏普外@0.3"], r["美元外@0.3"], "", 5, ("SELECTED " if k == sel else "")])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
第十四批 A：限价单剥头皮（5MIN 信号 + M1 保守撮合），预登记：reports/batch14_prereg.md。

撮合（只有 BID 的 M1）：买入限价 L 需 bid_low + s ≤ L − b（ask 穿价），卖出限价 L 需 bid_high ≥ L + b；成交价 = L。
止损/到时按市价（多卖 bid、空买 bid+s，跳空按开盘）；同根先算止损；成交那根只检查止损。
ATR·今 口径：s、b 按 (当时 ATR5 / 当前 ATR5) 折算，收益 = 价格盈亏 / 当时 ATR5。
用法：python -m research.limit_scalp          # 样本内选择
      python -m research.limit_scalp --oos    # 样本外只跑一次 + 检验 + 登记
"""
from __future__ import annotations

import argparse
import csv
import itertools

import numpy as np
import pandas as pd
from numba import njit

from factors.core import atr, params, rolling_mad_zscore, trading_day
from research.factor_correlation import factor_values
from research.multifactor_combo import sh
from research.overfitting_tests import cscv, dsr, matrix_B, n_eff

SPLIT = pd.Timestamp("2020-01-01")
S_REAL, BUF = 0.3, 0.05
STOP, VALID_MIN, HOLD_MIN = 2.0, 15, 30
STRONG = ["IntradayReversalCore", "IntradayRegimeScalp", "AnchoredVWAPBandReversion",
          "RangeVWAPReversion", "VolumeClimaxReversal", "TrendPullbackLowVolume"]
WEAK = ["ThreePushWedgeReversal", "PullbackSwing", "MTFPullbackResonance"]
SIGNALS = {"S1 IRC": ["IntradayReversalCore"], "S2 IRS": ["IntradayRegimeScalp"], "S3 AVB": ["AnchoredVWAPBandReversion"],
           "S4 强反转6等权": STRONG, "S5 强弱9等权": STRONG + WEAK}
GRID = list(itertools.product((1.5, 2.0), (0.0, 0.25, 0.5), (0.5, 1.0)))


@njit(cache=True)
def _sim(tm, o, h, l, c, flat, j0s, sides, closes, atrs, ss, bs, k, m, stop, valid_ns, hold_ns):
    n = len(tm)
    E = len(j0s)
    out_j = np.full(E, -1, np.int64)
    out_pnl = np.zeros(E)
    busy = -1
    for e in range(E):
        j0 = j0s[e]
        if j0 <= busy or j0 >= n or flat[j0]:
            continue
        side, a, s, b = sides[e], atrs[e], ss[e], bs[e]
        L = closes[e] - side * k * a
        jf = -1
        j = j0
        while j < n and tm[j] < tm[j0] + valid_ns:
            if flat[j]:
                break
            if side > 0 and l[j] + s <= L - b:
                jf = j
                break
            if side < 0 and h[j] >= L + b:
                jf = j
                break
            j += 1
        if jf < 0:
            busy = j
            continue
        tp = L + side * m * a
        sl = L - side * stop * a
        j = jf
        px = np.nan
        while True:
            if j >= n:
                j = n - 1
                px = c[j] if side > 0 else c[j] + s
                break
            if j > jf and (tm[j] >= tm[jf] + hold_ns or flat[j]):
                px = o[j] if side > 0 else o[j] + s
                break
            if side > 0:
                if l[j] <= sl:
                    px = min(o[j], sl) if j > jf else sl
                    break
                if j > jf and h[j] >= tp + b:
                    px = tp
                    break
            else:
                if h[j] + s >= sl:
                    px = max(o[j] + s, sl) if j > jf else sl
                    break
                if j > jf and l[j] + s <= tp - b:
                    px = tp
                    break
            j += 1
        out_j[e] = jf
        out_pnl[e] = side * (px - L)
        busy = j
    return out_j, out_pnl


class Market:
    def __init__(self):
        m1 = pd.read_pickle("data/cache/1MIN_raw.pkl")
        self.idx = m1.index
        self.tm = m1.index.asi8
        self.o, self.h, self.l, self.c = (m1[k].to_numpy() for k in ("open", "high", "low", "close"))
        ny = m1.index.tz_localize("UTC").tz_convert("America/New_York")
        mins = ny.hour * 60 + ny.minute
        self.flat = np.asarray((mins >= 16 * 60) & (mins < 18 * 60))
        d5 = pd.read_pickle("data/cache/5MIN.pkl")
        self.d5 = d5
        a = atr(d5, params("5MIN")["atr"])
        self.a5 = a.to_numpy()
        self.atr_now = float(a[d5.index >= d5.index[-1] - pd.Timedelta(days=365)].median())
        self.j0 = np.searchsorted(self.tm, (d5.index + pd.Timedelta("5min")).asi8)
        F = factor_values("5MIN", STRONG + WEAK).fillna(0.0)
        norm = params("5MIN")["norm"]
        self.z = {k: (F[v[0]] if len(v) == 1 else rolling_mad_zscore(F[v].mean(axis=1), norm)).to_numpy()
                  for k, v in SIGNALS.items()}

    def run(self, sig, zin, k, m, s=S_REAL, b=BUF, usd=False):
        z = self.z[sig]
        ev = np.where((np.abs(z) > zin) & np.isfinite(self.a5))[0]
        ev = ev[self.j0[ev] < len(self.tm)]
        a = self.a5[ev]
        scale = np.ones(len(ev)) if usd else a / self.atr_now
        jf, pnl = _sim(self.tm, self.o, self.h, self.l, self.c, self.flat, self.j0[ev].astype(np.int64),
                       np.sign(z[ev]), self.d5["close"].to_numpy()[ev], a, s * scale, b * scale, k, m, STOP,
                       VALID_MIN * 60 * 10**9, HOLD_MIN * 60 * 10**9)
        ok = jf >= 0
        t = pd.DatetimeIndex(self.idx[jf[ok]])
        r = pnl[ok] if usd else pnl[ok] / a[ok]
        side = np.sign(z[ev])[ok]
        return pd.DataFrame({"r": r, "side": side, "td": trading_day(t)}, index=t)


def daily(tr: pd.DataFrame) -> pd.DataFrame:
    g = tr.groupby("td")
    return pd.DataFrame({"r": g["r"].sum(), "long": tr[tr.side > 0].groupby("td")["r"].sum(),
                         "short": tr[tr.side < 0].groupby("td")["r"].sum()}).fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true")
    args = ap.parse_args()
    from research.failed_factor_combo import frozen_daily
    ref = frozen_daily()
    M = Market()
    print(f"当前 5MIN ATR 中位数 {M.atr_now:.2f}$：点差 0.3$ = {S_REAL / M.atr_now:.3f} ATR")
    rows, D, TR = [], {}, {}
    for sig in SIGNALS:
        for zin, k, m in GRID:
            key = f"{sig}|z{zin}|k{k}|m{m}"
            tr = M.run(sig, zin, k, m)
            d = daily(tr)
            D[key], TR[key] = d, tr
            din = d[d.index < SPLIT]
            tin = tr[tr.index < SPLIT]
            c = ref[ref.index < SPLIT].join(din["r"].rename("x"), how="outer").fillna(0).corr()["x"]
            rows.append({"key": key, "signal": sig, "IS夏普": sh(din["r"]), "IS笔/年": len(tin) / 11,
                         "IS每笔ATR": tin["r"].mean(), "IS胜率": (tin["r"] > 0).mean(),
                         "IS多": din["long"].sum(), "IS空": din["short"].sum(),
                         "相关TT30": c["TT30-EW-v1"], "相关HA1H": c["HA1H-v1"]})
    T = pd.DataFrame(rows).set_index("key")
    T["门槛"] = (T[["相关TT30", "相关HA1H"]].abs() < 0.3).all(axis=1)
    pd.set_option("display.width", 250)
    print("\n样本内 2009-2019（ATR·今 口径），全部 60 个配置：")
    print(T.drop(columns="signal").round(3).to_string())
    sel = T[T["门槛"]]["IS夏普"].idxmax()
    print(f"\n样本内夏普 > 0：{(T['IS夏普'] > 0).sum()}/60；选中 {sel}（IS 夏普 {T.loc[sel, 'IS夏普']:.2f}）")
    if not args.oos:
        print("（样本外只在 --oos 时运行一次）")
        return

    seg = lambda x, oos: x[x.index >= SPLIT] if oos else x[x.index < SPLIT]
    for k, d in D.items():
        T.loc[k, "OOS夏普"] = sh(seg(d, True)["r"])
    print(f"\n{'#' * 90}\n样本外 2020-01 ~ 2026-09（只跑一次）")
    print(T[["IS夏普", "OOS夏普"]].groupby(T["signal"]).median().round(2).to_string())
    print(f"全部 60 个：样本外夏普 > 0 的 {(T['OOS夏普'] > 0).sum()}，IS/OOS 秩相关 {T['IS夏普'].rank().corr(T['OOS夏普'].rank()):+.2f}")
    d, tr = D[sel], TR[sel]
    o, to = seg(d, True), seg(tr, True)
    print(f"选中 {sel}：样本外 夏普 {sh(o['r']):.2f}，{len(to) / 6.75:.0f} 笔/年，每笔 {to['r'].mean():+.4f} ATR，胜率 {(to['r'] > 0).mean():.0%}，"
          f"多 {o['long'].sum():+.1f} 空 {o['short'].sum():+.1f}")
    res = {"A1 样本外夏普 ≥ 0.5": sh(o["r"]) >= 0.5, "A2 多空都 > 0": o["long"].sum() > 0 and o["short"].sum() > 0}
    g = T[T["signal"] == T.loc[sel, "signal"]]
    res[f"A3 平原 内 {(g['IS夏普'] > 0).sum()}/12 外 {(g['OOS夏普'] > 0).sum()}/12"] = (g["IS夏普"] > 0).sum() >= 9 and (g["OOS夏普"] > 0).sum() >= 9
    MM = pd.DataFrame({k: v["r"] for k, v in D.items()}).fillna(0.0)
    cs = cscv(MM)
    res[f"A4 PBO {cs['PBO']:.3f} < 0.3（IS 最优 OOS 中位 {cs['oos_sharpe_of_is_best_median']:+.2f}）"] = cs["PBO"] < 0.3
    Bm = matrix_B()
    Bm = Bm.loc[:, Bm.std() > 0]
    MM2 = MM.loc[:, MM.std() > 0]
    ns, nb = n_eff(MM2), n_eff(Bm)
    x = d["r"].reindex(Bm.index.union(d.index)).fillna(0.0)
    dd = dsr(x, np.r_[(MM2.mean() / MM2.std()).to_numpy(), (Bm.mean() / Bm.std()).to_numpy()], ns + nb)
    res[f"A5 DSR {dd['DSR']:.3f} ≥ 0.5（N={ns:.1f}+{nb:.1f}，全样本夏普 {dd['SR_annual']:.2f}）"] = dd["DSR"] >= 0.5
    J = ref.join(d["r"].rename("new"), how="outer").fillna(0.0)
    info, ok6 = [], True
    for oos in (False, True):
        j = seg(J, oos)
        cc = j.corr()["new"]
        b0, b1 = sh(j["TT30-EW-v1"] + j["HA1H-v1"]), sh(j.sum(axis=1))
        info.append(f"{'外' if oos else '内'} 相关 {cc['TT30-EW-v1']:+.2f}/{cc['HA1H-v1']:+.2f} 组合 {b0:.2f}→{b1:.2f}")
        ok6 &= bool((cc[["TT30-EW-v1", "HA1H-v1"]].abs() < 0.3).all() and b1 >= b0)
    res["A6 " + "；".join(info)] = ok6
    zin, k, m = (float(x[1:]) for x in sel.split("|")[1:])
    rob = daily(M.run(T.loc[sel, "signal"], zin, k, m, s=0.4, b=0.1))
    res[f"A7 点差0.4/缓冲0.1 样本外夏普 {sh(seg(rob, True)['r']):.2f} > 0"] = sh(seg(rob, True)["r"]) > 0
    usd = M.run(T.loc[sel, "signal"], zin, k, m, usd=True)
    print(f"美元口径（固定 0.3$、1 盎司）：样本内 {seg(usd, False)['r'].sum():+.0f}$，样本外 {seg(usd, True)['r'].sum():+.0f}$；"
          f"逐年 " + " ".join(f"{y}:{v:+.0f}" for y, v in usd["r"].groupby(usd.index.year).sum().items()))
    print("\n判定：")
    for kk, v in res.items():
        print(f"  {'✅' if v else '❌'} {kk}")
    print("  结论：" + ("全部通过" if all(res.values()) else "未全部通过"))
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for k2, r in T.iterrows():
            w.writerow(["2026-09-29", f"限价单剥头皮:{r['signal']}", "5MIN+M1", "batch14 limit scalp", "limit",
                        k2.split("|", 1)[1] + ";stop=2ATR;valid15m;hold30m;s=0.3;b=0.05", "IS sharpe_atr_now (corr<0.3 gate)",
                        round(r["IS夏普"], 2), round(seg(D[k2], False)["r"].sum(), 1), round(r["OOS夏普"], 2),
                        round(seg(D[k2], True)["r"].sum(), 1), "", 60, ("SELECTED " if k2 == sel else "") + f"每笔ATR内 {r['IS每笔ATR']:+.4f}"])
    MM.to_pickle("data/cache/limit_scalp_daily.pkl")
    D[sel]["r"].to_pickle("data/cache/limit_scalp_selected.pkl")
    print("\n已登记 60 条试验")


if __name__ == "__main__":
    main()

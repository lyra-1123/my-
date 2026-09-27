# -*- coding: utf-8 -*-
"""
第十三批（预登记：reports/intraday_batch13_prereg.md）的补充检验。

[1] SettlementMomentum 事件研究（主检验）：每个工作日按 sign(R) 在窗口起点开盘入场、窗口终点开盘出场，
    R = 当日（纽约 18:00 起）到窗口起点的累计对数收益。窗口：结算 12:30~13:30（主），对照 10:30~11:30、14:30~15:30。
    收益 = side × (O_出 − O_入)/ATR_入场前一根 − side × 持有期内逐年平均漂移；成本 = 最近 1 年 0.2$/ATR 中位数。
    另报美元（1 盎司，扣 0.2$ 点差）、多空拆分与逐年为正比例。
[2] RealizedSkewReversal 残差 ICIR：扣除 IntradayReversalCore 后的独立信息（回归系数只用样本内）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.core import atr, params, trading_day
from factors.evaluate import HORIZONS, forward_return, ic_stats
from factors.library.intraday_info import BAR_MIN, ny_clock
from factors.registry import get_factors

SPLIT = pd.Timestamp("2020-01-01")
WINDOWS = {"结算 12:30~13:30（主）": (750, 810), "对照 10:30~11:30": (630, 690), "对照 14:30~15:30": (870, 930)}


def event_study(freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_pickle(f"data/cache/{freq}.pkl")
    bar = BAR_MIN[freq]
    a = atr(d, params(freq)["atr"])
    atr_now = float(a[d.index >= d.index[-1] - pd.Timedelta(days=365)].median())
    clock = ny_clock(d.index)
    td = pd.Series(trading_day(d.index), index=d.index)
    wk = np.asarray(d.index.tz_localize("UTC").tz_convert("America/New_York").dayofweek < 5)
    first_open = d["open"].groupby(td).transform("first")
    m = (d["open"].shift(-1) - d["open"]) / a.shift(1)
    drift = m.groupby(d.index.year).transform("mean")
    o = d["open"]
    rows, trades = [], []
    for name, (s, e) in WINDOWS.items():
        ent = pd.Series(np.flatnonzero((clock == s) & wk), dtype=int)                 # 窗口起点那根 K 线（开盘入场）
        ext_idx = pd.Series(np.flatnonzero((clock == e) & wk), index=td.iloc[np.flatnonzero((clock == e) & wk)].to_numpy())
        ext_idx = ext_idx[~ext_idx.index.duplicated()]
        T = []
        for i in ent:
            day = td.iloc[i]
            if day not in ext_idx.index or i < 1:
                continue
            x = int(ext_idx[day])
            if x <= i or x - i != (e - s) // bar:
                continue
            R = np.log(d["close"].iloc[i - 1] / first_open.iloc[i - 1])                # 截至窗口起点（上一根收盘）
            if R == 0 or not np.isfinite(a.iloc[i - 1]):
                continue
            side = float(np.sign(R))
            move = side * (o.iloc[x] - o.iloc[i])
            r_atr = move / a.iloc[i - 1] - side * drift.iloc[i:x].sum()
            T.append((d.index[i], side, move - 0.2, r_atr, r_atr - 0.2 / atr_now))
        T = pd.DataFrame(T, columns=["t", "side", "usd", "r", "r_net"])
        T["win"] = name
        trades.append(T)
        for seg, msk in (("内", T.t < SPLIT), ("外", T.t >= SPLIT), ("近3年", T.t >= d.index[-1] - pd.Timedelta(days=1095))):
            x = T[msk]
            yr = x.groupby(x.t.dt.year)["r"].mean()
            rows.append({"频率": freq, "窗口": name, "段": seg, "n": len(x), "每笔ATR": round(x.r.mean(), 4),
                         "t": round(x.r.mean() / (x.r.std() / np.sqrt(len(x))), 2), "当前成本ATR": round(0.2 / atr_now, 4),
                         "扣成本每笔ATR": round(x.r_net.mean(), 4), "美元": round(x.usd.sum()),
                         "多/空美元": f"{x.usd[x.side > 0].sum():+.0f}/{x.usd[x.side < 0].sum():+.0f}",
                         "多/空ATR": f"{x.r[x.side > 0].mean():+.3f}/{x.r[x.side < 0].mean():+.3f}",
                         "逐年>0": round(float((yr > 0).mean()), 2)})
    return pd.DataFrame(rows), pd.concat(trades)


def residual_icir(freq: str) -> dict:
    d = pd.read_pickle(f"data/cache/{freq}.pkl")
    f = get_factors(["RealizedSkewReversal"])[0](d, freq)
    g = get_factors(["IntradayReversalCore"])[0](d, freq)
    y = forward_return(d, HORIZONS[freq][1])
    Z = pd.concat([f, g], axis=1, keys=["c", "irc"]).dropna()
    ins = Z.index < SPLIT
    beta = np.polyfit(Z["irc"][ins], Z["c"][ins], 1)
    resid = Z["c"] - np.polyval(beta, Z["irc"])
    yy = y.reindex(Z.index)
    return {"频率": freq, "与IRC相关(内)": round(float(Z[ins].corr().iloc[0, 1]), 3),
            "原始ICIR内/外": (ic_stats(Z["c"], yy, freq, ins)["icir"], ic_stats(Z["c"], yy, freq, ~ins)["icir"]),
            "残差ICIR内/外": (ic_stats(resid, yy, freq, ins)["icir"], ic_stats(resid, yy, freq, ~ins)["icir"]),
            "残差t内/外": (ic_stats(resid, yy, freq, ins)["ic_t"], ic_stats(resid, yy, freq, ~ins)["ic_t"])}


def main() -> None:
    pd.set_option("display.width", 250)
    allrows, alltr = [], []
    for fq in ("15MIN", "30MIN"):
        r, t = event_study(fq)
        allrows.append(r); t["freq"] = fq; alltr.append(t)
    R = pd.concat(allrows)
    print("[1] SettlementMomentum 事件研究（每工作日 1 笔，sign(当日到窗口起点收益) 方向，持有 1 小时）")
    print(R.to_string(index=False))
    T = pd.concat(alltr)
    for fq in ("15MIN", "30MIN"):
        x = T[(T.freq == fq) & (T.win.str.startswith("结算"))]
        print(f"\n    {fq} 结算窗口逐年每笔 ATR：", x.groupby(x.t.dt.year)["r"].mean().round(3).to_dict())
    R.to_csv("reports/intraday_batch13_event_study.csv", index=False)
    print("\n[2] RealizedSkewReversal 扣除 IntradayReversalCore 后的残差 ICIR（系数只用样本内估计）")
    print(pd.DataFrame([residual_icir(fq) for fq in ("15MIN", "30MIN", "1H")]).to_string(index=False))


if __name__ == "__main__":
    main()


def dsr_impact() -> None:
    """[3] L27：本批 8 个"因子×频率"加入研究层后，在跑策略（TT30-EW-v1、HA1H-v1）的 DSR 变化。"""
    from paper.specs import SPECS
    from research.ha1h_exit_rules import evaluate as ha_eval, simulate as ha_sim
    from research.overfitting_tests import daily_atr_returns, dsr, matrix_B, n_eff
    B = matrix_B()
    const = [c for c in B.columns if B[c].std() == 0]           # 全期不开仓的"因子×频率"：相关矩阵无定义，剔除
    B = B.drop(columns=const)
    print(f"剔除全期无交易的列 {len(const)} 个：{const}")
    new = [c for c in B.columns if c.split("|")[0] in ("IntradaySeasonality", "SettlementMomentum", "RealizedSkewReversal")]
    old = B.drop(columns=new)
    tt = next(s for s in SPECS if s.id == "TT30-EW-v1")
    d30 = pd.read_pickle("data/cache/30MIN.pkl")
    sel = {"TT30-EW-v1": daily_atr_returns(d30, tt.signal(d30), "30MIN"), "HA1H-v1": ha_eval(ha_sim("strict", 0.3))[0]}
    n_logged = sum(1 for _ in open("reports/rule_trials.csv", encoding="utf-8")) - 1
    print(f"\n[3] 研究层规模对 DSR 的影响（日度 ATR·今，全样本；登记试验 {n_logged} 次，含本批）")
    for tag, M in (("本批之前", old), ("加入本批", B)):
        sr = (M.mean() / M.std()).to_numpy(); ne = n_eff(M)
        for k, r in sel.items():
            r = r.reindex(M.index).fillna(0.0)
            a, b = dsr(r, sr, ne), dsr(r, sr, ne + n_logged)
            print(f"   {tag}：N={M.shape[1]}，N_eff={ne:.1f}  {k:<11} SR={a['SR_annual']:.2f}  SR0={a['SR0_annual']:.2f}  "
                  f"DSR@N_eff={a['DSR']:.3f}  DSR@N_eff+登记={b['DSR']:.3f}")

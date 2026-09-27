# -*- coding: utf-8 -*-
"""
30MIN 趋势尾部策略稳健性验证（预登记，检验而非选参）。

对象：TrendEfficiencyVolume、VWAPDeviation（原方向=动量）及二者等权组合；30MIN；统一执行（迟滞 + 换日前平仓）。
默认参数：chan=32，entry=1.5，exit=0.3（均为入库时的先验值）。
检验：
  1) 参数平原：chan∈{16,24,32,48,64} × entry∈{1.0,1.25,1.5,1.75,2.0} × exit∈{0,0.3,0.6}
  2) Walk-forward：每年用此前全部年份（至少 3 年）选 (chan, entry)（exit=0.3），交易下一年；对比固定默认参数
  3) 成本敏感性：点差 ×{1,1.5,2,3}，过夜费 ×{1,2}
  4) 逐年 / 逐笔：净利、笔数、胜率、持仓时长、最大回撤、多空拆分
收益口径：
  - "ATR·今"：每根收益 / 上一根 ATR（固定 ATR 窗口 32，逐年去漂移），成本按最近 1 年 ATR 折算 → 与价格水平无关，反映今天的成本环境
  - "美元"：真实历史美元，0.01 手，点差 0.2 + 过夜费 0.47（周三 ×3）
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from factors.core import atr, rolling_mad_zscore, params, trading_day
from factors.evaluate import SPREAD, SWAP, execution_signal, positions, swap_units
from factors.registry import get_factors

FREQ = "30MIN"
SPLIT = pd.Timestamp("2020-01-01")
CHANS, ENTRIES, EXITS = (16, 24, 32, 48, 64), (1.0, 1.25, 1.5, 1.75, 2.0), (0.0, 0.3, 0.6)
DEFAULT = (32, 1.5, 0.3)

df = pd.read_pickle(f"data/cache/{FREQ}.pkl")
move = df["open"].shift(-1) - df["open"]
a_prev = atr(df, 32).shift(1)
m_atr = move / a_prev
m_atr = m_atr - m_atr.groupby(df.index.year).transform("mean")
atr_now = float(atr(df, 32)[df.index >= df.index[-1] - pd.Timedelta(days=365)].median())
swu = swap_units(df.index)
INS = df.index < SPLIT


def factor_z(name: str, chan: int) -> pd.Series:
    te = get_factors(["TrendEfficiencyVolume"])[0].func
    vw = get_factors(["VWAPDeviation"])[0].func
    if name == "TrendEff":
        return te(df, FREQ, chan=chan)
    if name == "VWAPDev":
        return vw(df, FREQ, chan=chan)
    raw = 0.5 * te(df, FREQ, chan=chan).fillna(0) + 0.5 * vw(df, FREQ, chan=chan).fillna(0)
    return rolling_mad_zscore(raw, params(FREQ)["norm"])


def pnl(pos: pd.Series, spread_x=1.0, swap_x=1.0) -> tuple[pd.Series, pd.Series]:
    dpos = pos.diff().abs().fillna(pos.abs())
    sw = pos.abs() * swu
    p_atr = (pos * m_atr - dpos * SPREAD * spread_x / 2 / atr_now - sw * SWAP * swap_x / atr_now).fillna(0)
    p_usd = (pos * move - dpos * SPREAD * spread_x / 2 - sw * SWAP * swap_x).fillna(0)
    return p_atr, p_usd


def daily(x: pd.Series) -> pd.Series:
    return x.groupby(trading_day(x.index)).sum()


def sharpe(x: pd.Series) -> float:
    d = daily(x)
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else float("nan")


def trades(pos: pd.Series, p_usd: pd.Series) -> pd.DataFrame:
    seg = (pos != pos.shift(1)).cumsum()
    g = pd.DataFrame({"pos": pos, "pnl": p_usd, "seg": seg})[pos != 0]
    t = g.groupby("seg").agg(side=("pos", "first"), bars=("pos", "size"), pnl=("pnl", "sum"))
    t["start"] = g.groupby("seg").apply(lambda x: x.index[0])
    return t


def main() -> None:
    Z = {(n, c): execution_signal(factor_z(n, c), FREQ) for n in ("TrendEff", "VWAPDev", "等权") for c in CHANS}
    res = {}
    for (n, c), z in Z.items():
        for e, x in itertools.product(ENTRIES, EXITS):
            pa, pu = pnl(positions(z, e, x))
            res[(n, c, e, x)] = (pa, pu)

    for n in ("TrendEff", "VWAPDev", "等权"):
        print(f"\n{'#' * 100}\n{n}")
        # 1) 参数平原
        for seg, m in (("样本内", INS), ("样本外", ~INS)):
            t = pd.DataFrame({e: {c: sharpe(res[(n, c, e, 0.3)][0][m]) for c in CHANS} for e in ENTRIES})
            t.index.name = "chan＼entry"
            print(f"\n[1a] 夏普（ATR·今）{seg}，exit=0.3")
            print(t.round(2).to_string())
        t = pd.DataFrame({x: {e: f"{sharpe(res[(n, 32, e, x)][0][INS]):.2f} / {sharpe(res[(n, 32, e, x)][0][~INS]):.2f}" for e in ENTRIES} for x in EXITS})
        t.index.name = "entry＼exit"
        print("\n[1b] 夏普 内/外，chan=32")
        print(t.to_string())

        # 2) Walk-forward（扩展窗口，≥3 年）
        years = sorted(set(df.index.year))
        wf_atr, wf_usd, picks = [], [], []
        for y in years:
            train = (df.index.year < y)
            if len(set(df.index.year[train])) < 3:
                continue
            best = max(itertools.product(CHANS, ENTRIES), key=lambda ce: sharpe(res[(n, ce[0], ce[1], 0.3)][0][train]))
            test = df.index.year == y
            pa, pu = res[(n, best[0], best[1], 0.3)]
            wf_atr.append(pa[test]); wf_usd.append(pu[test]); picks.append((y, best))
        wa, wu = pd.concat(wf_atr), pd.concat(wf_usd)
        da, du = res[(n, *DEFAULT)]
        span = wa.index
        print("\n[2] Walk-forward（每年用此前所有年份选 chan×entry，exit=0.3）")
        print("   每年所选参数:", " ".join(f"{y}:{c}/{e}" for y, (c, e) in picks))
        for tag, a, u in (("WF", wa, wu), ("固定默认", da.loc[span], du.loc[span])):
            oos = a.index >= SPLIT
            print(f"   {tag:<6} 夏普ATR·今 全期 {sharpe(a):+.2f} | 2020后 {sharpe(a[oos]):+.2f}   美元净利 全期 {u.sum():+.0f} | 2020后 {u[oos].sum():+.0f}")

        # 3) 成本敏感性（默认参数）
        pos = positions(Z[(n, 32)], 1.5, 0.3)
        print("\n[3] 成本敏感性（默认参数）  夏普ATR·今 内/外 | 美元净利 内/外")
        for sx, wx in ((1, 1), (1.5, 1), (2, 1), (3, 1), (1, 2), (2, 2)):
            pa, pu = pnl(pos, sx, wx)
            print(f"   点差×{sx:<3} 过夜×{wx}:  {sharpe(pa[INS]):+.2f} / {sharpe(pa[~INS]):+.2f}  |  {pu[INS].sum():+7.0f} / {pu[~INS].sum():+7.0f}")

        # 4) 逐年与逐笔（默认参数）
        pa, pu = pnl(pos)
        tr = trades(pos, pu)
        tr["year"] = pd.DatetimeIndex(tr["start"]).year
        eq = pu.cumsum()
        yr = pd.DataFrame({
            "美元净利": pu.groupby(df.index.year).sum().round(0),
            "ATR·今收益": pa.groupby(df.index.year).sum().round(1),
            "笔数": tr.groupby("year").size(),
            "胜率": tr.groupby("year")["pnl"].apply(lambda s: (s > 0).mean()).round(2),
            "平均持仓(小时)": (tr.groupby("year")["bars"].mean() / 2).round(1),
            "年内最大回撤$": pu.groupby(df.index.year).apply(lambda s: (s.cumsum() - s.cumsum().cummax()).min()).round(0),
        })
        print("\n[4] 逐年（默认参数）")
        print(yr.T.to_string())
        for seg, m in (("样本内", tr["start"] < SPLIT), ("样本外", tr["start"] >= SPLIT)):
            t = tr[m]
            print(f"   {seg}: {len(t)} 笔，胜率 {(t.pnl > 0).mean():.2f}，平均盈利 {t.pnl[t.pnl > 0].mean():+.2f}$ / 平均亏损 {t.pnl[t.pnl <= 0].mean():+.2f}$，"
                  f"多头 {t.pnl[t.side > 0].sum():+.0f}$ / 空头 {t.pnl[t.side < 0].sum():+.0f}$，平均持仓 {t.bars.mean() / 2:.1f} 小时")
        print(f"   全期最大回撤 {(eq - eq.cummax()).min():+.0f}$；年化夏普（ATR·今）全期 {sharpe(pa):+.2f}")


if __name__ == "__main__":
    main()

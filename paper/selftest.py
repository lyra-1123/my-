# -*- coding: utf-8 -*-
"""
模拟盘自检：python -m paper.selftest
改动 factors/ 或 paper/ 代码后、以及每次加入新策略前都要跑。
  1) 对账：引擎的仓位与美元净利 == 研究评估器（factors.evaluate）的结果
  2) 指纹：参数的微小变化必须改变行为指纹
  3) 未走完的 K 线：M1 在半根 K 线处截断时，该根必须被丢弃
  4) 前向流程：以过去某日为 forward_start 模拟一段前向，账本/日度/状态判断可正常生成
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from factors.data_loader import resample_ohlcv
from factors.evaluate import backtest, execution_signal
from paper.engine import BAR, compute, daily_from, fingerprint, load_bars, trades_from
from paper.onboard import bands
from paper.report import status
from paper.specs import SPECS


def main() -> None:
    ok = True
    for spec in [s for s in SPECS if s.status in ("active", "candidate", "shadow")]:
        bars, last_m1 = load_bars("data", spec.freq)
        c = compute(spec, bars)
        oos = bars.index >= pd.Timestamp("2020-01-01")
        if spec.rule == "state_trail":
            from research.ha1h_exit_rules import SPREAD as _SP, SWAP as _SW, SWU, df as rdf, simulate
            po, pn, dp, he = simulate("state", spec.exit, trail=dict(spec.rule_params)["trail"])
            m = rdf.index >= pd.Timestamp("2020-01-01")
            ref = {"net": round(float((pn - dp * _SP / 2 - he * SWU * _SW)[m].sum()), 1),
                   "trips": int(round(float(dp[m].sum() / 2)))}
            c_trips = int(round(float(c.loc[oos, "spread"].sum() / (_SP / 2 * spec.lots * 100) / 2)))
        else:
            ref = backtest(bars[oos], execution_signal(spec.signal(bars), spec.freq)[oos])
            c_trips = None
        eng = round(float(c.loc[oos, "net"].sum()), 1)
        eng_trips = c_trips if c_trips is not None else int(round(float(c.loc[oos, "pos"].diff().abs().fillna(c.loc[oos, "pos"].abs()).sum() / 2)))
        r1 = abs(eng - ref["net"]) < 0.2 and eng_trips == ref["trips"]
        print(f"[1] {spec.id} 对账：引擎 {eng:+.1f}$/{eng_trips} 笔 vs 评估器 {ref['net']:+.1f}$/{ref['trips']} 笔 → {'OK' if r1 else 'FAIL'}")

        comp = list(spec.components)
        name, w, kw = comp[0]
        if kw:   # 扰动第一个成分的第一个参数（+1）
            kw2 = ((kw[0][0], kw[0][1] + 1),) + tuple(kw[1:])
            alt = dataclasses.replace(spec, components=tuple([(name, w, kw2)] + comp[1:]))
        else:    # 成分没有显式参数时，扰动开仓阈值（对任何策略都会改变目标仓位）
            alt = dataclasses.replace(spec, entry=spec.entry + 0.05)
        r2 = fingerprint(spec, bars) != fingerprint(alt, bars)
        print(f"[2] {spec.id} 指纹对参数变化敏感 → {'OK' if r2 else 'FAIL'}")

        m1 = pd.read_pickle(sorted(__import__('glob').glob('data/cache/m1_*.pkl'))[0]).loc["2026-09-01":"2026-09-10 14:17"]
        rb = resample_ohlcv(m1, spec.freq)
        dur = pd.Timedelta(BAR[spec.freq])
        kept = rb[rb.index + dur <= m1.index[-1] + pd.Timedelta("1min")]
        r3 = kept.index[-1] + dur <= m1.index[-1] + pd.Timedelta("1min") and len(rb) == len(kept) + 1
        print(f"[3] 截断于 {m1.index[-1]}：最后保留 {kept.index[-1]}，丢弃 {rb.index[-1]} → {'OK' if r3 else 'FAIL'}")

        fwd = c.loc["2026-06-01":]
        d, tr = daily_from(fwd), trades_from(fwd, spec.lots)
        b = bands(daily_from(c.loc["2020-01-01":"2026-05-31"]))
        st = status(float(d["net_R"].sum()), len(d), b)
        r4 = len(d) > 50 and len(tr) > 5 and abs(tr["net"].sum() - d["net_usd"].sum()) < 0.25 * 2 + 1e-6
        print(f"[4] 模拟前向 2026-06-01~：{len(d)} 个交易日，{len(tr)} 笔，逐笔合计 {tr['net'].sum():+.1f}$ vs 日度合计 {d['net_usd'].sum():+.1f}$；状态：{st} → {'OK' if r4 else 'FAIL'}")
        ok &= r1 and r2 and r3 and r4
    print("\n自检", "全部通过" if ok else "有失败项")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

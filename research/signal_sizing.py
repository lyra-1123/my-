# -*- coding: utf-8 -*-
"""
信号强弱决定仓位（预登记：reports/sizing_hedge_prereg.md 第一部分）。

对在跑策略 TT30-EW-v1、HA1H-v1（只读规格，不修改）：
  检验 1：入场强度 s=|z|（开仓信号那根 K 线）分三档 [1.5,2)、[2,2.5)、≥2.5，看每笔 R 是否随档位上升；
  检验 2：S0 固定 1 单位 / S1 入场定档 1-2-3 单位 / S2 持仓期间按当前 |z| 档位动态加减仓；
          S1 另做随机对照（手数在交易间打乱 1000 次）。
口径与模拟盘引擎一致：R = (仓位×价格变动 − 点差 − 过夜费) / 上一根 ATR，按交易日汇总；成本为历史真实 0.2$ + 过夜费。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.core import trading_day
from factors.evaluate import SPREAD, SWAP, swap_units
from paper.engine import compute, load_bars
from paper.specs import get_spec

SPLIT = pd.Timestamp("2020-01-01")
BUCKETS = (2.0, 2.5)                      # [1.5,2) → 1，[2,2.5) → 2，≥2.5 → 3
N_PERM = 1000
rng = np.random.default_rng(20260927)


def bucket(s):
    return 1 + (np.asarray(s) >= BUCKETS[0]).astype(int) + (np.asarray(s) >= BUCKETS[1]).astype(int)


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")


def prepare(sid: str):
    spec = get_spec(sid)
    bars = load_bars("data", spec.freq)[0]
    c = compute(spec, bars)
    pos = c["pos"].to_numpy()
    prev = np.r_[0.0, pos[:-1]]
    start = (pos != 0) & (pos != prev)
    tid = np.cumsum(start) - 1
    tid = np.where(pos != 0, tid, -1)
    s_entry = np.abs(c["z_exec"].shift(1).to_numpy())[start]              # 开仓信号那根 K 线的 |z|
    z_prev = np.abs(c["z_exec"].shift(1).fillna(0).to_numpy())
    move = (c["open"].shift(-1) - c["open"]).to_numpy()
    a = c["atr_prev"].to_numpy()
    swu = swap_units(c.index).to_numpy()
    td = trading_day(c.index)
    return spec, c, pos, tid, start, s_entry, z_prev, move, a, swu, td


def bar_R(size, move, a, swu):
    dsize = np.abs(np.diff(np.r_[0.0, size]))
    return (size * move - dsize * SPREAD / 2 - np.abs(size) * swu * SWAP) / a


def trade_matrix(pos, tid, move, a, swu, td):
    """每笔交易（1 单位）对各交易日的 R 贡献：毛利 + 过夜费 + 入场/出场各半个点差。S1 的日度 R = 手数向量 @ 矩阵。"""
    n_tr = tid.max() + 1
    days = pd.Index(sorted(set(td)))
    di = days.get_indexer(td)
    M = np.zeros((n_tr, len(days)))
    ok = (tid >= 0) & np.isfinite(a) & np.isfinite(move)
    np.add.at(M, (tid[ok], di[ok]), (pos[ok] * move[ok] - np.abs(pos[ok]) * swu[ok] * SWAP) / a[ok])
    first = np.flatnonzero((tid >= 0) & (np.r_[-1, tid[:-1]] != tid))
    last = np.flatnonzero((tid >= 0) & (np.r_[tid[1:], -1] != tid))
    for idx in (first, np.minimum(last + 1, len(tid) - 1)):              # 入场在首根开盘、出场在末根之后那根开盘
        good = np.isfinite(a[idx])
        np.add.at(M, (tid[first][good] if idx is first else tid[last][good], di[idx][good]), -SPREAD / 2 / a[idx][good])
    return M, days


def main() -> None:
    info_rows, rule_rows = [], []
    for sid in ("TT30-EW-v1", "HA1H-v1"):
        spec, c, pos, tid, start, s, z_prev, move, a, swu, td = prepare(sid)
        # ---- 检验 1：每笔 R 与入场强度 ----
        R1 = pd.Series(bar_R(pos, move, a, swu), index=c.index)
        chk = float(np.nansum(R1.to_numpy()) - np.nansum(c["net_R"].to_numpy()))
        tr = pd.DataFrame({"tid": tid, "R": R1.to_numpy()})[tid >= 0].groupby("tid")["R"].sum()
        T = pd.DataFrame({"s": s, "R": tr.to_numpy(), "t": c.index[start]})
        for seg, m in (("内", T.t < SPLIT), ("外", T.t >= SPLIT)):
            x = T[m]
            row = {"策略": sid, "段": seg, "笔数": len(x), "Spearman(s,R)": round(float(x.s.corr(x.R, method="spearman")), 3)}
            for lo, hi, lab in ((1.5, 2.0, "[1.5,2)"), (2.0, 2.5, "[2,2.5)"), (2.5, 99, "≥2.5")):
                g = x[(x.s >= lo) & (x.s < hi)].R
                row[f"{lab} n"] = len(g)
                row[f"{lab} R/笔"] = round(float(g.mean()), 3)
                row[f"{lab} t"] = round(float(g.mean() / (g.std() / np.sqrt(len(g)))), 2) if len(g) > 2 else np.nan
            info_rows.append(row)
        # ---- 检验 2：仓位规则 ----
        m1 = bucket(s)
        size = {"S0 固定": pos.astype(float),
                "S1 入场定档": pos * np.where(tid >= 0, m1[np.maximum(tid, 0)], 0),
                "S2 动态跟随": pos * np.where(pos != 0, bucket(z_prev), 0)}
        daily = {}
        for k, sz in size.items():
            r = pd.Series(np.nan_to_num(bar_R(sz.astype(float), move, a, swu)), index=c.index)
            daily[k] = r.groupby(td).sum()
            eq = daily[k].cumsum()
            avg = float(np.abs(sz[sz != 0]).mean())
            rule_rows.append({"策略": sid, "规则": k, "夏普内": round(sharpe(daily[k][daily[k].index < SPLIT]), 3),
                              "夏普外": round(sharpe(daily[k][daily[k].index >= SPLIT]), 3), "全样本": round(sharpe(daily[k]), 3),
                              "平均手数": round(avg, 2), "最大回撤R/平均手数": round(float((eq - eq.cummax()).min()) / avg, 1),
                              "加减仓次数": int((np.abs(np.diff(np.r_[0.0, sz])) > 0).sum())})
        # ---- 随机对照：S1 手数在交易间打乱 ----
        M, days = trade_matrix(pos, tid, move, a, swu, td)
        ins = days < SPLIT
        base = m1 @ M
        err = float(np.abs(base - daily["S1 入场定档"].reindex(days).fillna(0).to_numpy()).max())
        perm = np.array([rng.permutation(m1) @ M for _ in range(N_PERM)])
        def sh(X, msk):
            X = X[..., msk]
            return X.mean(-1) / X.std(-1) * np.sqrt(252)
        p_in, p_out = sh(perm, ins), sh(perm, ~ins)
        rule_rows.append({"策略": sid, "规则": "S1 随机对照 95% 分位", "夏普内": round(float(np.percentile(p_in, 95)), 3),
                          "夏普外": round(float(np.percentile(p_out, 95)), 3), "全样本": np.nan, "平均手数": round(float(m1.mean()), 2),
                          "最大回撤R/平均手数": np.nan, "加减仓次数": np.nan})
        rule_rows.append({"策略": sid, "规则": "S1 在随机对照中的分位", "夏普内": round(float((p_in < sh(base, ins)).mean()), 3),
                          "夏普外": round(float((p_out < sh(base, ~ins)).mean()), 3), "全样本": np.nan, "平均手数": np.nan,
                          "最大回撤R/平均手数": np.nan, "加减仓次数": np.nan})
        print(f"{sid}：与引擎 net_R 对账差 {chk:.2e}；交易矩阵复现 S1 最大误差 {err:.2e}；交易 {len(T)} 笔")
        dist = pd.Series(m1).value_counts().sort_index()
        print(f"   入场档位分布（1/2/3 单位）：{dist.to_dict()}")
    pd.set_option("display.width", 250)
    print("\n[1] 入场强度 s 与每笔 R（引擎口径，含真实历史成本）\n" + pd.DataFrame(info_rows).to_string(index=False))
    print("\n[2] 仓位规则（日度夏普；随机对照 = S1 手数在交易间打乱 1000 次）\n" + pd.DataFrame(rule_rows).to_string(index=False))


if __name__ == "__main__":
    main()

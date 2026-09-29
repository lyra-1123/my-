# -*- coding: utf-8 -*-
"""
未通过 ICIR 筛选的因子：按逻辑桶组合 + 仓位规则（预登记：reports/failed_factor_combo_prereg.md）。

逻辑桶：MR 均值回归 / MB 动量突破 / RR 区间反转 / PB 趋势回调（按因子假设事先划分，方向不翻转）。
信号配置：每频率 15 个桶子集 + 2 个状态切换（G1 教科书 / G2 反向）；仓位规则 S0 固定 / S1 强度分层 / S2 桶独立净额。
口径：日度 ATR·今 收益（同 research/multifactor_combo.py）。选择只用样本内 2009-2019；样本外只跑一次。

用法：
  python -m research.failed_factor_combo            # 只输出样本内（选择阶段），不打印任何样本外数字
  python -m research.failed_factor_combo --oos      # 选择固定后，样本外只跑一次 + 全部检验，并登记 rule_trials.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os

import numpy as np
import pandas as pd

from factors.core import params, rolling_mad_zscore, trading_day
from factors.evaluate import HORIZONS, SPREAD, SWAP, execution_signal, forward_return, ic_stats, targets
from research.factor_correlation import factor_values
from research.multifactor_combo import Book, sh
from research.overfitting_tests import cscv, dsr, matrix_B, n_eff

SPLIT = pd.Timestamp("2020-01-01")
FREQS = ("30MIN", "1H")
TIER = 2.5                                    # S1：持仓期间 |z| ≥ 2.5 时 2 单位

# 事先划分的逻辑桶（只列名字；某频率下是否入池由 ICIR 筛选结果决定）
BUCKETS = {
    "MR": ("AsiaSessionReversion", "CCIReversion", "MFIReversion", "StochReversion", "RSI2PullbackInTrend",
           "UptrendDipReversion", "VolumeClimaxReversal", "WeekendGapReversion"),
    "MB": ("AroonTrend", "CMFFlow", "ForceIndex", "HTFTrendLTFBreakout", "IchimokuTrend", "LondonNYSessionMomentum",
           "MACDMomentum", "OBVMomentum", "PSARTrend", "SqueezeReleaseMomentum", "TRIXMomentum",
           "VolConfirmedBreakout", "VortexTrend"),
    "RR": ("RangeVWAPReversion", "ThreePushWedgeReversal"),
    "PB": ("MTFPullbackResonance", "PullbackSwing", "SecondEntryH2", "TrendPullbackLowVolume"),
}
EXCLUDE = {"ThreePushNoDecay", "FirstEntryH1", "RangeBreakoutNoPole", "FlagBreakout", "IntradayReversalCore"}
SUBSETS = [c for k in range(1, 5) for c in itertools.combinations(("MR", "MB", "RR", "PB"), k)]


def failed(freq: str) -> dict:
    """该频率下未通过 ICIR 筛选（|t|<5 或样本内外反号）且 t 可计算的因子 → {名字: 桶}。"""
    out = {}
    for b, names in BUCKETS.items():
        for n in names:
            p = f"reports/factors/{n}.json"
            if n in EXCLUDE or not os.path.exists(p):
                continue
            m = json.load(open(p, encoding="utf-8"))["results"].get(freq)
            if not m:
                continue
            t, i, o = m["icir"]["ic_t"], m["icir_in"]["icir"], m["icir_oos"]["icir"]
            if t != t:
                continue
            if abs(t) < 5 or np.sign(i) != np.sign(o):
                out[n] = b
    return out


# ---------------------------------------------------------------------------
# 仓位与日度收益
# ---------------------------------------------------------------------------
def pos_s0(z, freq, entry=1.5, exit_=0.3):
    return targets(execution_signal(z, freq), entry, exit_).shift(1).fillna(0.0)


def pos_s1(z, freq, entry=1.5, exit_=0.3):
    ze = execution_signal(z, freq)
    st = targets(ze, entry, exit_)
    strong = (ze * st >= TIER)                 # 与持仓同向且 |z| ≥ 2.5
    return (st * (1 + strong.astype(float))).shift(1).fillna(0.0)


def daily_r(B: Book, pos: pd.Series, spread_mult: float = 1.0) -> pd.DataFrame:
    """日度 ATR·今 收益，另给出多头/空头部分（成本按该 K 线仓位方向归属，平仓的成本归属原方向）。"""
    dpos = pos.diff().abs().fillna(pos.abs())
    cost = dpos * SPREAD * spread_mult / 2 / B.atr_now + pos.abs() * B.swu * SWAP / B.atr_now
    r = (pos * B.m_atr - cost).fillna(0.0)
    side = np.sign(pos).replace(0, np.nan).fillna(np.sign(pos.shift(1))).fillna(0)
    td = trading_day(r.index)
    return pd.DataFrame({"r": r.groupby(td).sum(), "long": r.where(side > 0, 0).groupby(td).sum(),
                         "short": r.where(side < 0, 0).groupby(td).sum()})


# ---------------------------------------------------------------------------
# 信号
# ---------------------------------------------------------------------------
class Signals:
    def __init__(self, freq: str):
        self.freq = freq
        self.pool = failed(freq)
        self.norm = params(freq)["norm"]
        self.F = factor_values(freq, list(self.pool)).fillna(0.0)
        self.bz = {b: rolling_mad_zscore(self.F[[n for n, bb in self.pool.items() if bb == b]].mean(axis=1), self.norm)
                   for b in BUCKETS if any(bb == b for bb in self.pool.values())}
        df = pd.read_pickle(f"data/cache/{freq}.pkl")
        c = df["close"]
        er = (c - c.shift(60)).abs() / c.diff().abs().rolling(60).sum()
        self.trend = (er > er.rolling(1000, min_periods=250).median().shift(1)).reindex(self.F.index).fillna(False)

    def combo(self, subset) -> pd.Series:
        return rolling_mad_zscore(pd.concat([self.bz[b] for b in subset], axis=1).fillna(0).mean(axis=1), self.norm)

    def regime(self, kind: str) -> pd.Series:
        mom = pd.concat([self.bz["MB"], self.bz["PB"]], axis=1).fillna(0).mean(axis=1)
        rev = pd.concat([self.bz["MR"], self.bz["RR"]], axis=1).fillna(0).mean(axis=1)
        a, b = (mom, rev) if kind == "G1" else (rev, mom)
        return rolling_mad_zscore(a.where(self.trend, b), self.norm)

    def configs(self):
        """[(名称, 规则, 仓位函数(entry, exit))]，共 17×2 + 11 个。"""
        out = []
        sig = {"+".join(s): self.combo(s) for s in SUBSETS}
        sig["G1 趋势→MB+PB/区间→MR+RR"] = self.regime("G1")
        sig["G2 趋势→MR+RR/区间→MB+PB"] = self.regime("G2")
        fq = self.freq
        for k, z in sig.items():
            out.append((k, "S0", lambda e, x, z=z: pos_s0(z, fq, e, x)))
            out.append((k, "S1", lambda e, x, z=z: pos_s1(z, fq, e, x)))
        for s in SUBSETS:
            if len(s) > 1:
                out.append(("+".join(s), "S2", lambda e, x, s=s: pd.concat([pos_s0(self.bz[b], fq, e, x) for b in s], axis=1).mean(axis=1)))
        return out


def frozen_daily() -> pd.DataFrame:
    """已登记策略（只读调用其规格的信号，不改任何东西）的日度 ATR·今 收益，用于相关性与组合检验。"""
    from paper.specs import get_spec
    out = {}
    for sid in ("TT30-EW-v1", "HA1H-v1"):
        s = get_spec(sid)
        B = Book(s.freq)
        z = s.signal(B.df)
        out[sid] = daily_r(B, pos_s0(z, s.freq, s.entry, s.exit))["r"]
    return pd.DataFrame(out).fillna(0.0)


def icir_diag(S: Signals) -> list[str]:
    """H1 诊断：桶信号的样本内月度 ICIR vs 桶内成分 ICIR 中位数。"""
    df = pd.read_pickle(f"data/cache/{S.freq}.pkl")
    y = forward_return(df, HORIZONS[S.freq][1])
    ins = df.index < SPLIT
    lines = []
    for b, z in S.bz.items():
        mem = [n for n, bb in S.pool.items() if bb == b]
        mi = [ic_stats(S.F[n], y, S.freq, ins)["icir"] for n in mem]
        bi = ic_stats(z, y, S.freq, ins)["icir"]
        lines.append(f"  {S.freq} {b}（{len(mem)} 个）：桶 ICIR 内 {bi:+.2f}  成分中位 {np.nanmedian(mi):+.2f}  "
                     f"成分 |ICIR| 中位 {np.nanmedian(np.abs(mi)):.2f}  → {'支持' if bi > np.nanmedian(mi) and bi > 0 else '不支持'} H1")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true")
    args = ap.parse_args()
    ref = frozen_daily()
    ref_in = ref[ref.index < SPLIT]

    rows, daily, fns = [], {}, {}
    diag = []
    for fq in FREQS:
        S = Signals(fq)
        B = Book(fq)
        print(f"\n{fq} 成分池 {len(S.pool)}：" + "；".join(f"{b}={[n for n, bb in S.pool.items() if bb == b]}" for b in S.bz))
        diag += icir_diag(S)
        for name, rule, fn in S.configs():
            key = f"{fq}|{name}|{rule}"
            d = daily_r(B, fn(1.5, 0.3))
            daily[key], fns[key] = d, (B, fn)
            din = d[d.index < SPLIT]
            c = ref_in.join(din["r"].rename("x"), how="outer").fillna(0).corr()["x"]
            rows.append({"key": key, "freq": fq, "signal": name, "rule": rule, "IS夏普": sh(din["r"]),
                         "IS多头": din["long"].sum(), "IS空头": din["short"].sum(),
                         "IS相关TT30": c["TT30-EW-v1"], "IS相关HA1H": c["HA1H-v1"]})
    T = pd.DataFrame(rows).set_index("key")
    T["相关门槛"] = (T[["IS相关TT30", "IS相关HA1H"]].abs() < 0.5).all(axis=1)
    sel = T[T["相关门槛"]]["IS夏普"].idxmax()
    top = T["IS夏普"].idxmax()
    pd.set_option("display.width", 250)
    print("\nH1 诊断（样本内月度 ICIR）：\n" + "\n".join(diag))
    print(f"\n样本内（2009-2019）全部 {len(T)} 个配置，按 IS 夏普排序（前 25）：")
    print(T.sort_values("IS夏普", ascending=False).head(25).drop(columns=["freq", "signal", "rule"]).round(2).to_string())
    print(f"\n样本内夏普 > 0 的配置：{(T['IS夏普'] > 0).sum()}/{len(T)}；中位 {T['IS夏普'].median():.2f}")
    print(f"选中（满足相关门槛、IS 夏普最高）：{sel}  IS 夏普 {T.loc[sel, 'IS夏普']:.2f}")
    print(f"不加门槛的第一名：{top}  IS 夏普 {T.loc[top, 'IS夏普']:.2f}")
    if not args.oos:
        print("\n（选择阶段结束。样本外只在 --oos 时运行一次。）")
        return
    oos_phase(T, sel, top, daily, fns, ref)


def oos_phase(T, sel, top, daily, fns, ref) -> None:
    def seg(d, oos):
        return d[d.index >= SPLIT] if oos else d[d.index < SPLIT]

    # 1) 全部配置的样本外（只为 CSCV/登记，选择已固定）
    for k, d in daily.items():
        o = seg(d, True)
        T.loc[k, "OOS夏普"] = sh(o["r"])
        T.loc[k, "OOS多头"], T.loc[k, "OOS空头"] = o["long"].sum(), o["short"].sum()
    print(f"\n{'#' * 100}\n样本外（2020-01 ~ 2026-09）——只跑一次")
    print(T.loc[[sel, top] if sel != top else [sel]].round(2).to_string())
    print(f"全部 {len(T)} 个配置：样本外夏普 > 0 的 {(T['OOS夏普'] > 0).sum()}，中位 {T['OOS夏普'].median():.2f}；"
          f"IS/OOS 夏普秩相关 {T['IS夏普'].rank().corr(T['OOS夏普'].rank()):+.2f}")
    print("\n按信号类型汇总（样本内 / 样本外夏普中位数）：")
    print(T.groupby(["freq", "rule"])[["IS夏普", "OOS夏普"]].median().round(2).to_string())

    res = {}
    r_sel = daily[sel]
    o = seg(r_sel, True)
    res["A1 样本外夏普 ≥ 0.3"] = (round(sh(o["r"]), 2), sh(o["r"]) >= 0.3)
    res["A2 样本外多空都 > 0"] = ((round(o["long"].sum(), 1), round(o["short"].sum(), 1)), o["long"].sum() > 0 and o["short"].sum() > 0)

    # 3) 参数平原
    B, fn = fns[sel]
    grid = []
    for e, x in itertools.product((1.0, 1.5, 2.0), (0.0, 0.3, 0.6)):
        d = daily_r(B, fn(e, x))
        grid.append({"entry": e, "exit": x, "IS": sh(seg(d, False)["r"]), "OOS": sh(seg(d, True)["r"])})
    G = pd.DataFrame(grid)
    print("\n参数平原（选中配置）：\n" + G.round(2).to_string(index=False))
    res["A3 平原 ≥7/9 内外为正"] = (f"内 {(G.IS > 0).sum()}/9 外 {(G.OOS > 0).sum()}/9", (G.IS > 0).sum() >= 7 and (G.OOS > 0).sum() >= 7)

    # 4) CSCV
    M = pd.DataFrame({k: d["r"] for k, d in daily.items()}).fillna(0.0)
    c = cscv(M)
    res["A4 PBO < 0.3"] = (f"PBO {c['PBO']:.3f}；IS 最优在 OOS 夏普中位 {c['oos_sharpe_of_is_best_median']:+.2f}", c["PBO"] < 0.3)

    # 5) DSR
    Bm = matrix_B()
    Bm = Bm.loc[:, Bm.std() > 0]                 # 去掉零方差列（从不开仓的稀疏因子），否则相关矩阵含 NaN
    M = M.loc[:, M.std() > 0]
    ns, nb = n_eff(M), n_eff(Bm)
    n_logged = sum(1 for _ in open("reports/rule_trials.csv", encoding="utf-8")) - 1
    sr_all = np.r_[(M.mean() / M.std()).to_numpy(), (Bm.mean() / Bm.std()).to_numpy()]
    x = r_sel["r"].reindex(Bm.index.union(r_sel.index)).fillna(0.0)
    d1 = dsr(x, sr_all, ns + nb)
    d2 = dsr(x, sr_all, ns + nb + n_logged + len(M.columns))
    print(f"\nDSR：本研究 N_eff {ns:.1f} + 研究层 N_eff {nb:.1f} → {d1}\n保守（+ 已登记 {n_logged} + 本次 {M.shape[1]}）→ {d2}")
    res["A5 DSR ≥ 0.5"] = (d1["DSR"], d1["DSR"] >= 0.5)

    # 6) 组合
    J = ref.join(r_sel["r"].rename("new"), how="outer").fillna(0.0)
    ok6, info = True, []
    for oos in (False, True):
        j = seg(J, oos)
        cc = j.corr()["new"]
        b0, b1 = sh(j["TT30-EW-v1"] + j["HA1H-v1"]), sh(j.sum(axis=1))
        info.append(f"{'外' if oos else '内'}：相关 TT30 {cc['TT30-EW-v1']:+.2f} HA1H {cc['HA1H-v1']:+.2f}，组合夏普 {b0:.2f}→{b1:.2f}")
        ok6 &= bool(abs(cc["TT30-EW-v1"]) < 0.5 and abs(cc["HA1H-v1"]) < 0.5 and b1 >= b0)
    res["A6 相关 < 0.5 且组合夏普不降"] = ("；".join(info), ok6)

    # 组合层仓位分配（P0/P1/P2，样本内选）
    jin = seg(J, False)
    w1 = 1 / jin.std()
    P = {"P0 等权": pd.Series(1.0, index=J.columns), "P1 逆波动(样本内)": w1 / w1.mean(),
         "P2 新策略半权": pd.Series({"TT30-EW-v1": 1.0, "HA1H-v1": 1.0, "new": 0.5})}
    prow = []
    for k, w in P.items():
        s = (J * w).sum(axis=1)
        prow.append({"方案": k, "IS": sh(seg(s, False)), "OOS": sh(seg(s, True)), "权重": w.round(2).to_dict()})
    PT = pd.DataFrame(prow).set_index("方案")
    psel = PT["IS"].idxmax()
    base = ref.sum(axis=1)
    print(f"\n组合层仓位分配（对照：TT30+HA1H 等权 IS {sh(seg(base, False)):.2f} / OOS {sh(seg(base, True)):.2f}）：")
    print(PT.round(2).to_string())
    print(f"  样本内选中：{psel}")

    # 7) 点差 ×2
    d2x = daily_r(B, fn(1.5, 0.3), spread_mult=2.0)
    res["A7 点差×2 样本外夏普 > 0"] = (round(sh(seg(d2x, True)["r"]), 2), sh(seg(d2x, True)["r"]) > 0)

    print(f"\n{'=' * 100}\n判定：{sel}")
    for k, (v, ok) in res.items():
        print(f"  {'✅' if ok else '❌'} {k}：{v}")
    print("  结论：" + ("全部通过 → 可作为影子/候选版本提交用户决定" if all(ok for _, ok in res.values()) else "未全部通过 → 不建议上模拟盘"))

    # 登记全部试验
    with open("reports/rule_trials.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for k, r in T.iterrows():
            din, dout = seg(daily[k], False), seg(daily[k], True)
            note = ("SELECTED " if k == sel else "") + f"桶组合 {r['signal']} {r['rule']}；IS相关 TT30 {r['IS相关TT30']:+.2f} HA1H {r['IS相关HA1H']:+.2f}"
            w.writerow(["2026-09-29", f"未达标因子桶组合:{r['signal']}", r["freq"], "failed-ICIR buckets", r["rule"],
                        "entry=1.5;exit=0.3" + (f";tier={TIER}" if r["rule"] == "S1" else ""),
                        "IS sharpe_atr_now (corr<0.5 gate)", round(r["IS夏普"], 2), round(din["r"].sum(), 1),
                        round(r["OOS夏普"], 2), round(dout["r"].sum(), 1), "", len(T), note])
        for k, r in PT.iterrows():
            w.writerow(["2026-09-29", f"组合层仓位分配(TT30+HA1H+{sel})", "daily", "portfolio", "alloc", k, "IS portfolio sharpe",
                        round(r["IS"], 2), "", round(r["OOS"], 2), "", "", len(PT), ("SELECTED " if k == psel else "") + str(r["权重"])])
    M.to_pickle("data/cache/failed_combo_daily.pkl")
    print(f"\n已登记 {len(T) + len(PT)} 条试验到 reports/rule_trials.csv")


if __name__ == "__main__":
    main()

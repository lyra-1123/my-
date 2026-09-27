# -*- coding: utf-8 -*-
"""
因子评估器 + 因子库文档生成。

用法（仓库根目录）：
    python -m factors.evaluate                         # 评估全部已注册因子
    python -m factors.evaluate --factors A B           # 只评估指定因子
    python -m factors.evaluate --library-only          # 只根据已有 JSON 重建 FACTOR_LIBRARY.md

输出：
    reports/factors/<因子名>.json    每个频率的完整指标
    FACTOR_LIBRARY.md                因子库总表 + 每个因子的卡片（自动生成，勿手改）

评估口径（所有因子统一，禁止针对单个因子改口径）：
    - 样本内 < split <= 样本外；参数不在样本内调优（用 core.FREQ_PRESETS 先验参数）
    - 信号 t 收盘产生，t+1 开盘成交；前瞻收益用开盘价
    - 成本：0.01 手（1 盎司）点差 0.2 美元，每单位仓位变化收半个点差；
      过夜费 0.47 美元/0.01 手/天，纽约 17:00 换日时持仓即收（多空都收），周三 3 倍
    - 执行：日内频率（≤1H）换日前平仓、换日后 1 小时不开仓（execution_signal）；4H/1D 持仓过夜
    - 迟滞开平仓：|z|>1.5 开仓，|z|<0.3 平仓
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from .core import ALL_FREQS, atr, check_input, params, trading_day
from .data_loader import load_m1, resample_ohlcv
from .registry import REGISTRY, FactorSpec, get_factors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports", "factors")
LIBRARY_MD = os.path.join(ROOT, "FACTOR_LIBRARY.md")

HORIZONS = {"5MIN": (1, 6, 24), "15MIN": (1, 4, 16), "30MIN": (1, 4, 12),
            "1H": (1, 4, 12), "4H": (1, 3, 6), "1D": (1, 3, 5)}
SPREAD = 0.2           # 美元 / 0.01 手 / 开平一次
SWAP = 0.47            # 美元 / 0.01 手 / 每次换日（多空都收），周三 3 倍
ENTRY, EXIT = 1.5, 0.3


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def forward_return(df: pd.DataFrame, h: int) -> pd.Series:
    """【仅评估用标签】t+1 开盘入场、t+1+h 开盘出场的对数收益。"""
    o = df["open"]
    return np.log(o.shift(-(1 + h)) / o.shift(-1))


def rank_ic(f: pd.Series, y: pd.Series) -> float:
    x = pd.concat([f, y], axis=1).dropna()
    x = x[x.iloc[:, 0] != 0]          # 稀疏因子只在有信号的 K 线上计算 IC
    if len(x) < 30:
        return float("nan")
    return float(x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank()))


def ic_stats(f: pd.Series, y: pd.Series, freq: str, mask=None) -> dict:
    """
    ICIR：按期计算 Rank IC，ICIR = mean(IC)/std(IC)，t = ICIR*sqrt(期数)。
    衡量信号稳定性：|ICIR| 越大，IC 的方向和大小越稳定。
    计算周期必须远长于因子的持续时间：≤1H 用月度（约 213 期）；4H/1D 的趋势类因子回看达数周，
    月内几乎不变，月度 IC 只反映"月内偏离"而漏掉跨月信息，因此用年度（18 期，统计功效较低）。
    """
    if mask is not None:
        f, y = f[mask], y[mask]
    key = f.index.to_period("Y" if freq in ("4H", "1D") else "M")
    ics = pd.Series({k: rank_ic(f[key == k], y[key == k]) for k in key.unique()}).dropna()
    if len(ics) < 6 or ics.std() == 0:
        return {"ic_mean": float("nan"), "icir": float("nan"), "ic_t": float("nan"), "n_periods": int(len(ics))}
    icir = ics.mean() / ics.std()
    return {"ic_mean": round(float(ics.mean()), 4), "icir": round(float(icir), 3),
            "ic_t": round(float(icir * np.sqrt(len(ics))), 2), "n_periods": int(len(ics))}


def targets(z: pd.Series, entry: float = ENTRY, exit_: float = EXIT) -> pd.Series:
    """迟滞规则：第 t 根收盘后的目标仓位（尚未执行）。"""
    raw = pd.Series(np.nan, index=z.index)
    raw[z > entry] = 1.0
    raw[z < -entry] = -1.0
    raw[z.abs() < exit_] = 0.0
    return raw.ffill().fillna(0.0)


def positions(z: pd.Series, entry: float = ENTRY, exit_: float = EXIT) -> pd.Series:
    """迟滞信号 -> 实际持仓（已 shift 到 t+1 开盘执行）。"""
    return targets(z, entry, exit_).shift(1).fillna(0.0)


def swap_units(index: pd.DatetimeIndex) -> pd.Series:
    """
    每根 K 线持仓需要支付的过夜费单位数。
    换日时点 = 纽约 17:00（与日线切分一致）。持仓 pos_t 覆盖 [open_t, open_t+1)，
    若 t 与 t+1 属于不同交易日，则跨过一次换日：收 1 份；该交易日为周三时收 3 份（覆盖周末），周五不另收。
    """
    ny = index.tz_localize("UTC").tz_convert("America/New_York") + pd.Timedelta(hours=7)
    day = pd.Series(ny.normalize().tz_localize(None), index=index)
    cross = day.shift(-1).ne(day) & day.shift(-1).notna()
    mult = np.where(day.dt.dayofweek == 2, 3.0, 1.0)
    return (cross.astype(float) * mult).rename("swap_units")


INTRADAY_FLAT = ("5MIN", "15MIN", "30MIN", "1H")   # 这些频率执行"换日前平仓"
BAR = {"5MIN": "5min", "15MIN": "15min", "30MIN": "30min", "1H": "1h", "4H": "4h", "1D": "1D"}


def execution_signal(z: pd.Series, freq: str) -> pd.Series:
    """
    执行层规则（不改变因子本身，IC 仍用原始因子计算）：
    日内频率在 [纽约17:00 - 2根K线, 纽约18:00) 内把信号置 0：
      - 迟滞规则在换日前平仓（提前两根：t 收盘出信号、t+1 开盘成交），不支付过夜费；
      - 换日后 1 小时（点差最宽的时段）不开新仓。
    4H/1D 本身就是多日持仓，不做处理，照常支付过夜费。
    已知残留：美国假日提前收盘（纽约 14:00 左右）的日子，平仓窗口内没有 K 线，仓位会被带过换日（每年约 10 次）。
    """
    if freq not in INTRADAY_FLAT:
        return z
    ny = z.index.tz_localize("UTC").tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    start = 17 * 60 - 2 * pd.Timedelta(BAR[freq]).seconds // 60
    flat = (mins >= start) & (mins < 18 * 60)
    return z.where(~flat, 0.0)


def backtest(df: pd.DataFrame, z: pd.Series, spread: float = SPREAD, oz: float = 1.0) -> dict:
    pos = positions(z)
    move = df["open"].shift(-1) - df["open"]
    bar = pos * move * oz
    bar_ex = pos * (move - move.mean()) * oz          # 去掉样本期平均漂移（牛市 beta）后的毛利
    spread_cost = pos.diff().abs().fillna(pos.abs()) * spread / 2.0 * oz
    swap_cost = pos.abs() * swap_units(df.index) * SWAP * oz
    cost = spread_cost + swap_cost
    net = (bar - cost).fillna(0.0)
    trips = float(pos.diff().abs().sum() / 2.0)
    eq = net.cumsum()
    daily = net.groupby(trading_day(net.index)).sum()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else float("nan")
    return {
        "gross": round(float(bar.sum()), 1),
        "cost": round(float(cost.sum()), 1),
        "spread_cost": round(float(spread_cost.sum()), 1),
        "swap_cost": round(float(swap_cost.sum()), 1),
        "net": round(float(net.sum()), 1),
        "net_ex_drift": round(float((bar_ex - cost).sum()), 1),
        "long_gross": round(float(bar[pos > 0].sum()), 1),
        "short_gross": round(float(bar[pos < 0].sum()), 1),
        "trips": int(round(trips)),
        "gross_per_trip": round(float(bar.sum() / max(trips, 1)), 3),
        "time_in_mkt": round(float((pos != 0).mean()), 3),
        "mdd": round(float((eq - eq.cummax()).min()), 1),
        "sharpe": round(sharpe, 2),
    }


def atr_edge(df: pd.DataFrame, z: pd.Series, freq: str) -> dict:
    """
    以 ATR 为单位、逐年去漂移后的每笔毛利（与价格水平/波动率无关），逐年统计。
    固定 0.2 美元点差在不同波动时期对应的 ATR 成本差异巨大（2015 年 15MIN≈0.15 ATR，2026 年≈0.02 ATR），
    因此还要与"当前"成本比较：(点差 + 过夜费 × 平均每笔跨越的换日单位数) / 最近 1 年 ATR 中位数。
    """
    a = atr(df, params(freq)["atr"]).shift(1)
    pos = positions(z)
    yr = df.index.year
    m_atr = (df["open"].shift(-1) - df["open"]) / a
    m_atr = m_atr - m_atr.groupby(yr).transform("mean")      # 逐年去漂移（剔除当年金价单边行情的 beta）
    g = pos * m_atr
    trips = pos.diff().abs() / 2
    per_year = (g.groupby(yr).sum() / trips.groupby(yr).sum().replace(0, np.nan)).dropna()
    last = df.index >= df.index[-1] - pd.Timedelta(days=365)
    recent = df.index >= df.index[-1] - pd.Timedelta(days=3 * 365)
    split = pd.Timestamp(SPLIT_DEFAULT)
    ins, oos = df.index < split, df.index >= split
    # 该策略平均每笔开平要跨越的过夜费单位数（持仓习惯的属性，用全样本估计）
    swap_per_trip = float((pos.abs() * swap_units(df.index)).sum() / max(trips.sum(), 1))

    def pt(m):
        n = float(trips[m].sum())
        return round(float(g[m].sum() / n), 4) if n > 0 else float("nan")
    return {
        "per_trip_atr_in": pt(ins), "per_trip_atr_oos": pt(oos), "per_trip_atr_recent3y": pt(recent),
        "per_trip_atr_year": {int(k): round(float(v), 4) for k, v in per_year.items()},
        "per_trip_atr_year_pos_ratio": round(float((per_year > 0).mean()), 2) if len(per_year) else float("nan"),
        "swap_units_per_trip": round(swap_per_trip, 3),
        "cost_now_atr": round(float(((SPREAD + SWAP * swap_per_trip) / a[last]).median()), 4),
    }


def lookahead_ok(spec: FactorSpec, df: pd.DataFrame, freq: str, n: int = 20000) -> bool:
    """截断数据后历史因子值必须完全不变。"""
    tail = df.tail(n)
    cut = int(len(tail) * 0.7)
    full = spec(tail, freq).iloc[:cut]
    part = spec(tail.iloc[:cut], freq)
    return bool(np.nan_to_num((full - part).abs().max()) < 1e-9)


def verdict(m: dict) -> str:
    """统一判定规则（详见 .claude/skills/xauusd-factor-mining/references/evaluation.md）。"""
    ii, io, cons, bt, rv = m["ic_in"], m["ic_oos"], m["ic_year_pos_ratio"], m["bt_oos"], m["bt_oos_reversed"]
    if any(pd.isna(v) for v in (ii, io, cons)):
        return "⚪ 数据不足"
    ic_ok = ii > 0.01 and io > 0.01 and cons >= 0.7
    if (ic_ok and bt["net"] > 0 and bt.get("net_ex_drift", 0) > 0 and bt["trips"] >= 100
            and bt["long_gross"] > 0 and bt["short_gross"] > 0):
        return "✅ 候选"
    if ic_ok and bt["net"] <= 0:
        return "💸 有效但成本不可行"
    e = m.get("atr_edge")
    if (e and e["per_trip_atr_in"] > 0 and e["per_trip_atr_oos"] > 0 and e["per_trip_atr_year_pos_ratio"] >= 0.75
            and e["per_trip_atr_recent3y"] > 1.5 * e["cost_now_atr"] and bt["trips"] >= 100):
        return "⏳ 当前波动下可行"
    if ii > 0 and io > 0 and bt["net"] > 0 and bt.get("net_ex_drift", 0) > 0 and bt["trips"] >= 30:
        return "🟡 观察"
    if ii < -0.01 and io < -0.01 and cons <= 0.3:
        return "🔄 方向相反" + ("（反向可盈利）" if rv["net"] > 0 else "")
    return "❌ 拒绝"


def evaluate_freq(spec: FactorSpec, df: pd.DataFrame, freq: str, split: pd.Timestamp) -> dict:
    fac = spec(df, freq)
    zx = execution_signal(fac, freq)          # 回测/ATR 边际使用执行层信号
    ins, oos = fac.index < split, fac.index >= split
    hs = HORIZONS[freq]
    main_h = hs[1]
    ic = {}
    for h in hs:
        y = forward_return(df, h)
        ic[h] = (rank_ic(fac[ins], y[ins]), rank_ic(fac[oos], y[oos]))
    y = forward_return(df, main_h)
    icir_all, icir_in, icir_oos = ic_stats(fac, y, freq), ic_stats(fac, y, freq, ins), ic_stats(fac, y, freq, oos)
    years = sorted(set(df.index.year))
    ic_year = {int(yr): rank_ic(fac[df.index.year == yr], y[df.index.year == yr]) for yr in years}
    valid = [v for v in ic_year.values() if not np.isnan(v)]
    m = {
        "freq": freq, "main_h": main_h,
        "lookahead_ok": lookahead_ok(spec, df, freq),
        "cost_hurdle": round(float(SPREAD / atr(check_input(df), params(freq)["atr"]).median()), 3),
        "ic": {str(h): [round(a, 4), round(b, 4)] for h, (a, b) in ic.items()},
        "ic_in": round(ic[main_h][0], 4), "ic_oos": round(ic[main_h][1], 4),
        "icir": icir_all, "icir_in": icir_in, "icir_oos": icir_oos,
        "ic_year": {k: round(v, 4) for k, v in ic_year.items()},
        "ic_year_pos_ratio": round(float(np.mean([v > 0 for v in valid])), 2) if valid else float("nan"),
        "bt_oos": backtest(df[oos], zx[oos]),
        "bt_oos_reversed": backtest(df[oos], -zx[oos]),
        "signal_coverage": round(float((fac.abs() > ENTRY).mean()), 4),
        "atr_edge": atr_edge(df, zx, freq),
    }
    m["verdict"] = verdict(m)
    return m


# ---------------------------------------------------------------------------
# 因子库文档
# ---------------------------------------------------------------------------
def _fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def build_library_md() -> str:
    get_factors()
    rows, cards = [], []
    for name, spec in REGISTRY.items():
        path = os.path.join(REPORT_DIR, f"{name}.json")
        rep = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
        if rep is None:
            rows.append(f"| {name} | {spec.cn_name} | {spec.family} | 未评估 | | |")
            continue
        res = rep["results"]
        best = max(res.values(), key=lambda m: m["bt_oos"]["net"])
        verdicts = " ".join(f"{f}:{res[f]['verdict'].split()[0]}" for f in res)
        rows.append(f"| [{name}](#{name.lower()}) | {spec.cn_name} | {spec.family} | {verdicts} | "
                    f"{best['freq']} ({best['bt_oos']['net']:+.0f}$) | {rep['evaluated']} |")
        t = ["| 频率 | 判定 | IC内 | IC外 | ICIR内/外(t) | 年度IC>0占比 | 样本外净利$ | 去漂移净利$ | 多头毛利 | 空头毛利 | 开平次数 | 单笔毛利 | 反向净利 | 每笔毛利ATR 内/外/近3年 | 年度ATR毛利>0 | 当前成本ATR(点差+过夜) | 夏普外 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for f, m in res.items():
            b, r = m["bt_oos"], m["bt_oos_reversed"]
            t.append(f"| {f} (h={m['main_h']}) | {m['verdict']} | {_fmt(m['ic_in'],4)} | {_fmt(m['ic_oos'],4)} | "
                     f"{_fmt(m['icir_in']['icir'],2)} / {_fmt(m['icir_oos']['icir'],2)} ({_fmt(m['icir']['ic_t'],1)}) | "
                     f"{_fmt(m['ic_year_pos_ratio'],2)} | {b['net']:+.0f} | {b.get('net_ex_drift', float('nan')):+.0f} | {b['long_gross']:+.0f} | {b['short_gross']:+.0f} | "
                     f"{b['trips']} | {b['gross_per_trip']:+.3f} | {r['net']:+.0f} | "
                     + (f"{e['per_trip_atr_in']:+.3f} / {e['per_trip_atr_oos']:+.3f} / {e['per_trip_atr_recent3y']:+.3f} | "
                        f"{_fmt(e['per_trip_atr_year_pos_ratio'],2)} | {e['cost_now_atr']:.3f} |"
                        if (e := m.get("atr_edge")) else "— | — | — |")
                     + f" {_fmt(b.get('sharpe'), 2)} |")
        la = all(m["lookahead_ok"] for m in res.values())
        cards.append("\n".join([
            f"### {name}", "",
            f"**{spec.cn_name}** · 家族 `{spec.family}` · 入库 {spec.added} · 评估 {rep['evaluated']} · "
            f"未来函数自检 {'✅' if la else '❌'}", "",
            f"- **逻辑**：{spec.hypothesis}",
            f"- **算子**：`{spec.formula}`",
            f"- **风险**：{'；'.join(spec.risks)}",
            f"- **代码**：`factors/library/{spec.func.__module__.split('.')[-1]}.py::{spec.func.__name__}`", "",
            *t, ""]))
    head = [
        "# XAUUSD 因子库", "",
        "> 本文件由 `python -m factors.evaluate` 自动生成，请勿手改。",
        f"> 数据：Dukascopy XAUUSD M1（BID，UTC），样本内 < {SPLIT_DEFAULT} ≤ 样本外。"
        "成本：0.01 手点差 0.2 美元 + 过夜费 0.47 美元/天（纽约 17:00 换日，多空都收，周三 3 倍）；开仓 |z|>1.5、平仓 |z|<0.3；信号 t 收盘、t+1 开盘成交；≤1H 换日前平仓。", "",
        "判定：✅ 候选 · ⏳ 当前波动下可行 · 🟡 观察 · 💸 有效但成本不可行 · 🔄 方向相反 · ❌ 拒绝 · ⚪ 数据不足"
        "（规则见 `.claude/skills/xauusd-factor-mining/references/evaluation.md`）", "",
        "## 总表", "",
        "| 因子 | 中文名 | 家族 | 各频率判定 | 样本外最佳（净利/0.01手） | 评估日期 |",
        "|---|---|---|---|---|---|",
        *rows, "", "## 因子卡片", "", *cards,
    ]
    return "\n".join(head)


SPLIT_DEFAULT = "2020-01-01"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--factors", nargs="*")
    ap.add_argument("--freqs", nargs="*", default=ALL_FREQS)
    ap.add_argument("--split", default=SPLIT_DEFAULT)
    ap.add_argument("--library-only", action="store_true")
    args = ap.parse_args()

    if not args.library_only:
        specs = get_factors(args.factors)
        split = pd.Timestamp(args.split)
        cache = os.path.join(args.data_dir, "cache")
        os.makedirs(cache, exist_ok=True)
        os.makedirs(REPORT_DIR, exist_ok=True)
        m1 = None
        bars = {}
        for freq in args.freqs:
            pk = os.path.join(cache, f"{freq}.pkl")
            if os.path.exists(pk):
                bars[freq] = pd.read_pickle(pk)
            else:
                m1 = load_m1(args.data_dir) if m1 is None else m1
                bars[freq] = resample_ohlcv(m1, freq)
                bars[freq].to_pickle(pk)
        for spec in specs:
            res = {}
            for freq in args.freqs:
                if freq not in spec.freqs:
                    continue
                m = evaluate_freq(spec, bars[freq], freq, split)
                res[freq] = m
                b = m["bt_oos"]
                print(f"{spec.name:<26}{freq:>6}  IC内 {m['ic_in']:+.4f}  IC外 {m['ic_oos']:+.4f}  "
                      f"ICIR {m['icir_in']['icir']:+.2f}/{m['icir_oos']['icir']:+.2f} t={m['icir']['ic_t']:+.1f}  "
                      f"年度>0 {m['ic_year_pos_ratio']:.2f}  净利 {b['net']:+8.0f}  去漂移 {b['net_ex_drift']:+8.0f}  次数 {b['trips']:>5}  "
                      f"自检 {'OK' if m['lookahead_ok'] else 'FAIL'}  {m['verdict']}")
            out = os.path.join(REPORT_DIR, f"{spec.name}.json")
            if os.path.exists(out):                       # 只重跑部分频率时，保留其他频率的已有结果
                prev = json.load(open(out, encoding="utf-8"))["results"]
                res = {f: res.get(f, prev.get(f)) for f in ALL_FREQS if f in res or f in prev}
            rep = {"name": spec.name, "evaluated": str(date.today()), "split": args.split, "results": res}
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, ensure_ascii=False, indent=1)

    with open(LIBRARY_MD, "w", encoding="utf-8") as fh:
        fh.write(build_library_md())
    print(f"\n因子库已更新: {LIBRARY_MD}")


if __name__ == "__main__":
    main()

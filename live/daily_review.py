# -*- coding: utf-8 -*-
"""
MT5 模拟盘每日复盘（数据层）：python -m live.daily_review [--day 2026-09-29] [--all-missing]

读取 MT5 的真实成交记录（history_deals）+ 程序日志（decisions.csv / orders.csv），与"同一份 MT5 K 线上、按 0.2 点差计算的模型账本"逐项对照，
输出 live/reports/review_<交易日>.md 和 live/reports/daily_summary.csv（每策略每日一行，供累计统计）。

交易日按纽约 17:00 划分（与过夜费、模型账本一致）：交易日 D = [D 前一日 17:00 NY, D 当日 17:00 NY)。
这是"数据层"：只做客观计算和基于事先写死阈值的检查。深度复盘（教练评语）按
.claude/skills/mt5-daily-review/SKILL.md 在此报告基础上撰写。

怀疑原则（写死在代码里）：
  - 评价的是"是否按规则执行"和"执行成本"，不是当日盈亏；当日盈亏只换算成模型日波动的倍数（σ），|z| < 2 视为噪声。
  - 实际与模型的差额拆成：执行成本差（滑点 + 点差超出 0.2 的部分）与其他（信号/时点/止损不一致），后者必须逐笔解释。
  - 模拟盘点差大于实盘：另给"按 0.2 点差折算"的实际盈亏，但不据此美化结果。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from factors.core import trading_day
from live import config as C
from live.mt5_api import mt5
from live.mt5_data import DUR, bars_range, connect, deals_between, fetch_bars, utc_now_from_server
from paper.engine import compute, daily_from, trades_from
from paper.specs import get_spec

REPORT_DIR = os.path.join(os.path.dirname(C.LOG_DIR), "reports")
REAL_SPREAD = 0.20            # 实盘点差（模型成本假设）
REASON = {0: "手动(客户端)", 1: "手动(手机)", 2: "手动(网页)", 3: "程序", 4: "止损", 5: "止盈", 6: "强平", 7: "展期", 8: "保证金调整"}
MANUAL_REASONS = {0, 1, 2}

# 事先写死的检查阈值
TH = {"align_green": 0.95, "align_red": 0.90,          # 逐根仓位一致率
      "slip_yellow": 0.10, "slip_red": 0.30,           # 单边滑点（$/盎司，已扣除点差）
      "latency_yellow": 60, "latency_red": 300,        # 成交距 K 线开盘的秒数
      "z_noise": 2.0, "z_investigate": 3.0}            # 当日盈亏 / 模型日波动


def day_bounds(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """交易日标签 → [开始, 结束) 的 UTC 时间。"""
    end_ny = (pd.Timestamp(day) + pd.Timedelta(hours=17)).tz_localize("America/New_York")
    end = end_ny.tz_convert("UTC").tz_localize(None)
    start = ((pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=17)).tz_localize("America/New_York")
             .tz_convert("UTC").tz_localize(None))
    return start, end


def last_closed_day(now_utc: pd.Timestamp) -> pd.Timestamp:
    """最近一个已经收盘的交易日。"""
    return trading_day(pd.DatetimeIndex([now_utc]))[0] - pd.offsets.BDay(1)


def _read_log(name: str) -> pd.DataFrame:
    p = os.path.join(C.LOG_DIR, name)
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    return df


# ---------------------------------------------------------------------------
# 成交 → 逐笔交易
# ---------------------------------------------------------------------------
def round_trips(deals: pd.DataFrame, magic: int) -> pd.DataFrame:
    d = deals[deals["magic"] == magic]
    rows = []
    for pid, g in d.groupby("position_id"):
        ins, outs = g[g["entry"] == 0], g[g["entry"].isin([1, 3])]
        if ins.empty:
            continue
        i = ins.iloc[0]
        side = 1 if i["type"] == 0 else -1
        r = {"position": pid, "side": side, "lots": i["volume"], "entry_utc": i["utc"], "entry_px": i["price"],
             "entry_reason": int(i["reason"]), "exit_utc": pd.NaT, "exit_px": np.nan, "exit_reason": None,
             "profit": g["profit"].sum(), "swap": g["swap"].sum(), "commission": g["commission"].sum() + g["fee"].sum(),
             "is_open": outs.empty}
        if not outs.empty:
            o = outs.iloc[-1]
            r.update(exit_utc=o["utc"], exit_px=float((outs["price"] * outs["volume"]).sum() / outs["volume"].sum()),
                     exit_reason=int(o["reason"]))
        rows.append(r)
    tr = pd.DataFrame(rows)
    if not tr.empty:
        tr["net"] = tr["profit"] + tr["swap"] + tr["commission"]
    return tr


def spread_at(dec: pd.DataFrame, sid: str, t: pd.Timestamp) -> float:
    """成交前最近一次决策记录的点差（程序在下单前读取）；没有记录时返回 NaN。"""
    if dec.empty or pd.isna(t):
        return np.nan
    x = dec[(dec["strategy"] == sid) & (dec["utc_time"] <= t + pd.Timedelta(seconds=5))]
    x = x[pd.to_numeric(x["spread"], errors="coerce").notna()]
    return float(x["spread"].iloc[-1]) if len(x) else np.nan


def enrich(tr: pd.DataFrame, sid: str, bars: pd.DataFrame, c: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    """每笔：参考价（成交所在 K 线开盘价，BID）、执行成本、延迟、最大浮盈/浮亏、ATR 单位。"""
    if tr.empty:
        return tr
    spec = get_spec(sid)
    dur = DUR[spec.freq]
    out = []
    for _, r in tr.iterrows():
        oz = r["lots"] * 100
        eb = r["entry_utc"].floor(dur)
        ref_in = float(bars["open"].get(eb, np.nan))
        s_in = spread_at(dec, sid, r["entry_utc"])
        # 执行成本（$/盎司，正 = 对自己不利），以 BID 开盘价为基准：买入含整个点差，卖出不含
        cost_in = r["side"] * (r["entry_px"] - ref_in)
        cost_out, s_out, ref_out, lat_out = np.nan, np.nan, np.nan, np.nan
        if not r["is_open"]:
            xb = r["exit_utc"].floor(dur)
            s_out = spread_at(dec, sid, r["exit_utc"])
            if r["exit_reason"] == 4:           # 止损：参考价为模型的止损成交价（跳空按开盘价）
                sf = c["stop_fill"].get(xb, np.nan) if "stop_fill" in c else np.nan
                if np.isnan(sf):
                    sf = c["stop_fill"].get(r["exit_utc"].floor("1h"), np.nan) if "stop_fill" in c else np.nan
                ref_out = sf
            else:
                ref_out = float(bars["open"].get(xb, np.nan))
                lat_out = (r["exit_utc"] - xb).total_seconds()
            cost_out = -r["side"] * (r["exit_px"] - ref_out)
        # 点差部分：买入那一边付出整个点差（开多 = 开仓时，开空 = 平仓时）
        spread_paid = s_in if r["side"] > 0 else (s_out if not r["is_open"] else 0.0)
        total_cost = cost_in + (cost_out if not r["is_open"] else 0.0)
        slip = total_cost - (spread_paid if np.isfinite(spread_paid) else 0.0)
        # 最大浮盈 / 浮亏（M1 K 线，BID）
        end = r["exit_utc"] if not r["is_open"] else pd.Timestamp.now("UTC").tz_localize(None)
        m1 = bars_range("1MIN", r["entry_utc"].floor("1min"), end)
        if len(m1):
            hi, lo = m1["high"].max(), m1["low"].min()
            mfe = (hi - r["entry_px"]) if r["side"] > 0 else (r["entry_px"] - lo)
            mae = (r["entry_px"] - lo) if r["side"] > 0 else (hi - r["entry_px"])
        else:
            mfe = mae = np.nan
        atr_ = float(c["atr_prev"].get(eb, np.nan))
        move = r["side"] * (r["exit_px"] - r["entry_px"]) if not r["is_open"] else np.nan
        out.append({**r.to_dict(), "sid": sid, "ref_in": ref_in, "latency_in_s": (r["entry_utc"] - eb).total_seconds(),
                    "latency_out_s": lat_out, "ref_out": ref_out, "cost_in": cost_in, "cost_out": cost_out,
                    "spread_in": s_in, "spread_out": s_out, "spread_paid": spread_paid, "slip_ex_spread": slip,
                    "net_at_real_spread": (r["net"] + ((spread_paid if np.isfinite(spread_paid) else REAL_SPREAD) - REAL_SPREAD) * oz)
                    if not r["is_open"] else np.nan,
                    "mfe": mfe, "mae": mae, "capture": (move / mfe) if (mfe and mfe > 0 and np.isfinite(move)) else np.nan,
                    "atr": atr_, "R": (r["net"] / oz / atr_) if atr_ and not r["is_open"] else np.nan,
                    "hold_min": ((r["exit_utc"] if not r["is_open"] else end) - r["entry_utc"]).total_seconds() / 60})
    return pd.DataFrame(out)


def live_position_path(deals: pd.DataFrame, magic: int, times: pd.DatetimeIndex, lag_s=120) -> pd.Series:
    """每根 K 线开盘后 lag_s 秒时的实际净手数（由成交记录重建）。"""
    d = deals[deals["magic"] == magic].copy()
    if d.empty:
        return pd.Series(0.0, index=times)
    sgn = np.where(d["type"] == 0, 1.0, -1.0)
    d["delta"] = sgn * d["volume"]
    cum = d.set_index("utc")["delta"].cumsum()
    idx = cum.index.searchsorted(times + pd.Timedelta(seconds=lag_s), side="right") - 1
    return pd.Series(np.where(idx >= 0, cum.to_numpy()[np.clip(idx, 0, None)], 0.0), index=times).round(2)


# ---------------------------------------------------------------------------
# 单策略复盘
# ---------------------------------------------------------------------------
def review_strategy(sid, cfg, day, start, end, deals_hist, dec, orders):
    spec = get_spec(sid)
    lots = cfg["lots"]
    bars = fetch_bars(spec.freq, C.HISTORY_BARS[spec.freq])
    c = compute(spec, bars)
    in_day = (c.index >= start) & (c.index < end)
    cd = c[in_day]
    # 模型（同一份 MT5 K 线，0.2 点差）
    daily = daily_from(c)
    model_day = float(daily["net_usd"].get(day, 0.0))
    hist = daily["net_usd"].iloc[-261:-1]
    sigma = float(hist[hist != 0].std()) if (hist != 0).sum() > 20 else np.nan
    mtr = trades_from(c, lots)
    mtr_day = mtr[(mtr["entry_time"] >= start) & (mtr["entry_time"] < end)]
    # 实际
    # 上线时点：该策略第一条决策记录（没有日志时退回第一笔成交）
    started = deals_hist[deals_hist["magic"] == cfg["magic"]]
    first_dec = dec.loc[dec["strategy"] == sid, "utc_time"].min() if not dec.empty else pd.NaT
    live_start = first_dec if pd.notna(first_dec) else (started["utc"].min() if len(started) else pd.NaT)
    tr_all = round_trips(deals_hist, cfg["magic"])
    tr = tr_all[(tr_all["entry_utc"] < end) & ((tr_all["exit_utc"] >= start) | tr_all["is_open"])] if len(tr_all) else tr_all
    tr = enrich(tr, sid, bars, c, dec)
    closed_today = tr[(~tr["is_open"]) & (tr["exit_utc"] >= start) & (tr["exit_utc"] < end)] if len(tr) else tr
    live_day = float(closed_today["net"].sum()) if len(closed_today) else 0.0
    # 逐根仓位一致（只看程序已在运行的 K 线）
    t_ok = cd.index[(cd.index >= live_start)] if pd.notna(live_start) else cd.index[:0]
    model_pos = (cd.loc[t_ok, "pos"] * lots).round(2)
    live_pos = live_position_path(deals_hist, cfg["magic"], t_ok)
    mism = model_pos[(model_pos - live_pos).abs() > 1e-9]
    align = 1 - len(mism) / len(t_ok) if len(t_ok) else np.nan
    # 日志审计
    dd = dec[(dec["strategy"] == sid) & (dec["utc_time"] >= start) & (dec["utc_time"] < end)] if not dec.empty else dec
    od = orders[(orders["strategy"] == sid) & (orders["utc_time"] >= start) & (orders["utc_time"] < end)] if not orders.empty else orders
    n_err = int((dd["action"] == "ERROR").sum()) if len(dd) else 0
    notes = dd["note"].dropna().astype(str) if len(dd) else pd.Series(dtype=str)
    n_spread_block = int(notes.str.contains("点差").sum())
    n_stale = int(notes.str.contains("过期").sum())
    n_order_fail = int((pd.to_numeric(od["retcode"], errors="coerce") != 10009).sum()) if len(od) else 0
    manual = int(deals_hist[(deals_hist["magic"] == cfg["magic"]) & (deals_hist["utc"] >= start) & (deals_hist["utc"] < end)
                            & deals_hist["reason"].isin(MANUAL_REASONS)].shape[0])
    swap_trips = int((closed_today["swap"].abs() > 1e-9).sum()) if len(closed_today) else 0
    z = live_day / sigma if sigma and np.isfinite(sigma) and sigma > 0 else np.nan
    return {"sid": sid, "name": spec.name, "freq": spec.freq, "status": spec.status, "lots": lots, "bars": cd, "trips": tr,
            "closed": closed_today, "model_trades": mtr_day, "model_day": model_day, "live_day": live_day, "sigma": sigma, "z": z,
            "z_model": model_day / sigma if sigma else np.nan, "align": align, "n_bars_live": len(t_ok), "mismatch": mism,
            "live_pos_at_mismatch": live_pos.reindex(mism.index), "n_err": n_err, "n_spread_block": n_spread_block,
            "n_stale": n_stale, "n_order_fail": n_order_fail, "manual": manual, "swap_trips": swap_trips,
            "live_start": live_start, "decisions": dd, "daily_model": daily, "tr_all": tr_all, "c": c}


def flags_for(r) -> list[tuple[str, str]]:
    f = []
    if r["manual"]:
        f.append(("🔴", f"{r['manual']} 笔手动成交（魔术号属于本策略但来源是客户端/手机/网页）：破坏了规则执行，当日数据不能用于评价策略"))
    if r["n_err"]:
        f.append(("🔴", f"程序报错 {r['n_err']} 次（decisions.csv 中 action=ERROR）"))
    if r["n_order_fail"]:
        f.append(("🔴", f"下单失败 {r['n_order_fail']} 次（orders.csv 回执不是 10009）"))
    if r["swap_trips"]:
        f.append(("🔴", f"{r['swap_trips']} 笔交易被收取过夜费：按规则应在纽约 17:00 前平仓，需查明原因"))
    if np.isfinite(r["align"]):
        lvl = "🟢" if r["align"] >= TH["align_green"] else ("🟡" if r["align"] >= TH["align_red"] else "🔴")
        if lvl != "🟢":
            f.append((lvl, f"逐根仓位一致率 {r['align']:.1%}（{len(r['mismatch'])}/{r['n_bars_live']} 根不一致）"))
    if r["n_spread_block"]:
        f.append(("🟡", f"{r['n_spread_block']} 次因点差过宽未开仓（模拟盘点差问题，实盘不会发生，但造成与模型的偏离）"))
    if r["n_stale"]:
        f.append(("🟡", f"{r['n_stale']} 次数据过期未操作（断线/终端卡顿？）"))
    cl = r["closed"]
    if len(cl):
        slip = cl["slip_ex_spread"].dropna()
        if len(slip) and slip.mean() > TH["slip_yellow"] * 2:
            f.append(("🔴" if slip.mean() > TH["slip_red"] * 2 else "🟡", f"扣除点差后的往返滑点均值 {slip.mean():+.2f} $/盎司"))
        lat = pd.concat([cl["latency_in_s"], cl["latency_out_s"]]).dropna()
        if len(lat) and lat.max() > TH["latency_yellow"]:
            f.append(("🔴" if lat.max() > TH["latency_red"] else "🟡", f"最慢成交距 K 线开盘 {lat.max():.0f} 秒"))
    if np.isfinite(r["z"]) and abs(r["z"]) >= TH["z_investigate"]:
        f.append(("🟡", f"当日盈亏 {r['z']:+.1f}σ，超出正常波动，需确认是行情极端还是执行问题"))
    return f


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _f(x, fmt):
    try:
        return "" if x is None or not np.isfinite(float(x)) else format(float(x), fmt)
    except (TypeError, ValueError):
        return ""


def _fmt_t(t):
    return "" if pd.isna(t) else pd.Timestamp(t).tz_localize("UTC").tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M")


def cumulative(r, prereg, end):
    """程序上线以来到本交易日结束的累计：实际 vs 模型（同一时段、同一份 MT5 K 线）及登记区间。"""
    tr, c = r["tr_all"], r["c"]
    if pd.isna(r["live_start"]):
        return None
    cl = tr[(~tr["is_open"]) & (tr["exit_utc"] < end)] if len(tr) else pd.DataFrame(columns=["net"])
    seg = c[(c.index >= r["live_start"].floor(DUR[r["freq"]])) & (c.index < end)]
    n_days = int(trading_day(seg.index).nunique()) if len(seg) else 0
    band = None
    if prereg:
        keys = sorted(int(k) for k in prereg.get("bands_cum_R", {}))
        k = max([x for x in keys if x <= n_days], default=None)
        if k:
            band = (k, prereg["bands_cum_R"][str(k)])
    return {"days": n_days, "trips": len(cl), "live_cum": float(cl["net"].sum()), "model_cum": float(seg["net"].sum()),
            "win": float((cl["net"] > 0).mean()) if len(cl) else np.nan, "band": band}


def write_report(day, results, account, now_utc, foreign=None, ctx=None) -> str:
    L = []
    d = pd.Timestamp(day).strftime("%Y-%m-%d")
    s, e = day_bounds(day)
    L.append(f"# MT5 模拟盘复盘 · 交易日 {d}（数据层）\n")
    L.append(f"区间：{_fmt_t(s)} → {_fmt_t(e)}（北京时间；纽约 17:00 为界） · 生成于 {_fmt_t(now_utc)} · "
             f"账户 {account.login}@{account.server} · 余额 {account.balance:.2f} · 净值 {account.equity:.2f} {account.currency}\n")
    L.append("> 本页是自动计算的客观数据与规则检查；教练评语按 `.claude/skills/mt5-daily-review/SKILL.md` 在此基础上撰写。"
             "模型 = 同一份 MT5 K 线、按规则成交、0.2 点差；σ = 模型近一年非零交易日的日盈亏标准差。\n")
    if ctx:
        L.append("## 0. 当日行情（MT5 30MIN，BID）\n")
        L.append(f"- 开 {ctx['open']:.2f} / 高 {ctx['high']:.2f} / 低 {ctx['low']:.2f} / 收 {ctx['close']:.2f}，涨跌 {ctx['chg']:+.2f}（{ctx['chg_atr']:+.2f} 倍日 ATR）")
        L.append(f"- 振幅 {ctx['range']:.2f} = {ctx['range_atr']:.2f} 倍日 ATR(14)（{ctx['atr']:.2f}），在近 250 日中位于 {ctx['range_pct']:.0%} 分位；"
                 f"趋势效率 {ctx['eff']:.2f}（|收−开| / 30 分钟路径长度，越接近 1 越单边）")
        L.append(f"- 最大单根 30MIN 波动 {ctx['max_bar']:.2f}（{_fmt_t(ctx['max_bar_t'])}）\n")
    # 总览
    L.append("## 1. 当日总览\n")
    L.append("| 策略 | 状态 | 实际净盈亏$ | 按0.2点差折算$ | 模型$ | 差额$ | 当日 z | 平仓笔数(实际/模型) | 仓位一致率 | 检查 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    tot = {"live": 0.0, "adj": 0.0, "model": 0.0}
    for r in results:
        cl = r["closed"]
        adj = float(cl["net_at_real_spread"].sum()) if len(cl) else 0.0
        fl = flags_for(r)
        worst = "🔴" if any(x[0] == "🔴" for x in fl) else ("🟡" if fl else "🟢")
        mt = r["model_trades"]
        L.append(f"| {r['name']}（{r['sid']}） | {r['status']} | {r['live_day']:+.2f} | {adj:+.2f} | {r['model_day']:+.2f} | "
                 f"{r['live_day'] - r['model_day']:+.2f} | {_f(r['z'], '+.2f')} | {len(cl)}/{int((~mt['is_open']).sum()) if len(mt) else 0} | "
                 f"{_f(r['align'], '.0%')} | {worst} |")
        if r["status"] == "active":
            tot["live"] += r["live_day"]; tot["adj"] += adj; tot["model"] += r["model_day"]
    L.append(f"| **组合（仅 active）** | | **{tot['live']:+.2f}** | **{tot['adj']:+.2f}** | **{tot['model']:+.2f}** | "
             f"**{tot['live'] - tot['model']:+.2f}** | | | | |\n")
    # 规则检查
    L.append("## 2. 规则与执行检查（阈值事先写死）\n")
    anyf = False
    if foreign is not None and len(foreign):
        L.append(f"- 🔴 **账户**：当日有 {len(foreign)} 笔不属于本程序魔术号的 {C.SYMBOL} 成交（手动单或其他 EA），会占用保证金、干扰熔断统计")
        anyf = True
    for r in results:
        for lvl, msg in flags_for(r):
            L.append(f"- {lvl} **{r['name']}**：{msg}"); anyf = True
    if not anyf:
        L.append("- 🟢 无违例：无手动干预、无报错、无下单失败、无过夜、仓位与模型一致、滑点与延迟在阈值内")
    L.append("")
    # 逐策略
    L.append("## 3. 逐策略明细\n")
    for r in results:
        L.append(f"### {r['name']}（{r['sid']}，{r['freq']}，{r['status']}）\n")
        cd = r["bars"]
        if len(cd):
            zx = cd["z_exec"]
            L.append(f"- 信号：当日 {len(cd)} 根 K 线，|z|>入场阈值 {int((zx.abs() > get_spec(r['sid']).entry).sum())} 根，"
                     f"z 范围 {zx.min():+.2f} ~ {zx.max():+.2f}；模型当日 {len(r['model_trades'])} 笔开仓，持仓时间占比 {(cd['pos'] != 0).mean():.0%}")
        if np.isfinite(r["sigma"]):
            L.append(f"- 结果的统计意义：实际 {r['live_day']:+.2f}$ = {r['z']:+.2f}σ，模型 {r['model_day']:+.2f}$ = {r['z_model']:+.2f}σ"
                     f"（σ = {r['sigma']:.1f}$；|z| < {TH['z_noise']:.0f} 属于日常噪声）")
        tr = r["trips"]
        if len(tr):
            L.append("\n| 开仓(北京) | 方向 | 手数 | 开仓价 | 参考价 | 延迟s | 平仓(北京) | 平仓价 | 平仓原因 | 净$ | 按0.2折算$ | R | 滑点(扣点差)$/oz | MFE | MAE | 兑现率 | 持仓分钟 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for _, t in tr.iterrows():
                L.append(f"| {_fmt_t(t['entry_utc'])} | {'多' if t['side'] > 0 else '空'} | {t['lots']:.2f} | {t['entry_px']:.2f} | {t['ref_in']:.2f} | "
                         f"{t['latency_in_s']:.0f} | {'持仓中' if t['is_open'] else _fmt_t(t['exit_utc'])} | "
                         f"{'' if t['is_open'] else _f(t['exit_px'], '.2f')} | {'' if t['is_open'] else REASON.get(t['exit_reason'], t['exit_reason'])} | "
                         f"{t['net']:+.2f} | {'' if t['is_open'] else _f(t['net_at_real_spread'], '+.2f')} | "
                         f"{_f(t['R'], '+.2f')} | "
                         f"{_f(t['slip_ex_spread'], '+.2f')} | "
                         f"{_f(t['mfe'], '.2f')} | {_f(t['mae'], '.2f')} | {_f(t['capture'], '.0%')} | {t['hold_min']:.0f} |")
            L.append("")
        else:
            L.append("- 当日无实际交易")
        mt = r["model_trades"]
        if len(mt):
            L.append("模型当日开仓：" + "；".join(f"{_fmt_t(x.entry_time)} {'多' if x.side == 'LONG' else '空'} → "
                                              f"{'持仓中' if x.is_open else _fmt_t(x.exit_time)} {x.net:+.2f}$" for x in mt.itertuples()))
        if len(r["mismatch"]):
            L.append("\n仓位不一致的 K 线（模型 vs 实际手数）：" + "；".join(
                f"{_fmt_t(t)} {m:+.2f}/{r['live_pos_at_mismatch'].get(t, np.nan):+.2f}" for t, m in r["mismatch"].head(12).items()))
        cum = r.get("cum")
        if cum:
            b = cum["band"]
            btxt = f"；登记区间（{b[0]} 日累计 R）5%/50%/95% = {b[1]['p5']:+.1f}/{b[1]['p50']:+.1f}/{b[1].get('p95', float('nan')):+.1f}" if b else ""
            L.append(f"\n- 上线以来：{cum['days']} 个交易日，平仓 {cum['trips']} 笔，胜率 {_f(cum['win'], '.0%') or '—'}，"
                     f"实际累计 {cum['live_cum']:+.2f}$ vs 模型同期 {cum['model_cum']:+.2f}${btxt}")
        L.append("")
    # 待解释事项
    L.append("## 4. 必须逐条解释的偏离（教练复盘从这里开始）\n")
    k = 0
    for r in results:
        diff = r["live_day"] - r["model_day"]
        cl = r["closed"]
        exec_part = -float(((cl["cost_in"] + cl["cost_out"]) - REAL_SPREAD).mul(cl["lots"] * 100).sum()) if len(cl) else 0.0
        other = diff - exec_part
        if abs(diff) > 0.5 or len(r["mismatch"]):
            k += 1
            L.append(f"{k}. **{r['name']}** 实际 − 模型 = {diff:+.2f}$，其中执行成本差 ≈ {exec_part:+.2f}$（滑点 + 点差超出 0.2 的部分），"
                     f"其他 ≈ {other:+.2f}$（信号/成交时点/止损/跨日归属不一致；不一致 K 线 {len(r['mismatch'])} 根）")
    if k == 0:
        L.append("- 无：实际与模型的差额均小于 0.5$，且逐根仓位一致")
    L.append("")
    return "\n".join(L)


def market_context(start, end, day):
    try:
        b = fetch_bars("30MIN", C.HISTORY_BARS["30MIN"])
        seg = b[(b.index >= start) & (b.index < end)]
        if seg.empty:
            return None
        td = trading_day(b.index)
        d = b.groupby(td).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
        atr_ = tr.rolling(14).mean().shift(1)
        rng = d["high"] - d["low"]
        day = pd.Timestamp(day)
        a = float(atr_.get(day, np.nan))
        hist = rng.loc[:day].iloc[-251:-1]
        path = seg["close"].diff().abs().sum()
        o, c_ = float(seg["open"].iloc[0]), float(seg["close"].iloc[-1])
        bar_rng = seg["high"] - seg["low"]
        return {"open": o, "high": float(seg["high"].max()), "low": float(seg["low"].min()), "close": c_, "chg": c_ - o,
                "chg_atr": (c_ - o) / a, "range": float(seg["high"].max() - seg["low"].min()),
                "range_atr": float(seg["high"].max() - seg["low"].min()) / a, "atr": a,
                "range_pct": float((hist < float(seg["high"].max() - seg["low"].min())).mean()),
                "eff": abs(c_ - o) / path if path else np.nan, "max_bar": float(bar_rng.max()), "max_bar_t": bar_rng.idxmax()}
    except Exception:
        return None


def update_summary(day, results):
    path = os.path.join(REPORT_DIR, "daily_summary.csv")
    rows = []
    for r in results:
        cl = r["closed"]
        rows.append({"day": pd.Timestamp(day).strftime("%Y-%m-%d"), "strategy": r["sid"], "status": r["status"],
                     "live_usd": round(r["live_day"], 2),
                     "live_usd_at_0.2": round(float(cl["net_at_real_spread"].sum()) if len(cl) else 0.0, 2),
                     "model_usd": round(r["model_day"], 2), "trips": len(cl),
                     "live_R": round(float(cl["R"].sum()) if len(cl) else 0.0, 3), "z": round(r["z"], 2) if np.isfinite(r["z"]) else "",
                     "align": round(r["align"], 3) if np.isfinite(r["align"]) else "",
                     "slip_ex_spread_mean": round(float(cl["slip_ex_spread"].mean()), 3) if len(cl) else "",
                     "flags": " | ".join(f"{a}{b}" for a, b in flags_for(r))})
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["day"] != new["day"].iloc[0]]
        new = pd.concat([old, new], ignore_index=True).sort_values(["day", "strategy"])
    new.to_csv(path, index=False, encoding="utf-8-sig")


def run_review(day: pd.Timestamp | None = None, now_utc: pd.Timestamp | None = None) -> str:
    now_utc = now_utc or utc_now_from_server()
    day = pd.Timestamp(day) if day is not None else last_closed_day(now_utc)
    start, end = day_bounds(day)
    deals_hist = deals_between(end - pd.Timedelta(days=120), end)
    dec, orders = _read_log("decisions.csv"), _read_log("orders.csv")
    results = []
    for sid, cfg in C.STRATEGIES.items():
        if not cfg["enabled"]:
            continue
        r = review_strategy(sid, cfg, day, start, end, deals_hist, dec, orders)
        pj = os.path.join("paper", "state", sid, "prereg.json")
        r["cum"] = cumulative(r, json.load(open(pj, encoding="utf-8")) if os.path.exists(pj) else None, end)
        results.append(r)
    os.makedirs(REPORT_DIR, exist_ok=True)
    ours = {v["magic"] for v in C.STRATEGIES.values()}
    foreign = deals_hist[(deals_hist["utc"] >= start) & (deals_hist["utc"] < end) & ~deals_hist["magic"].isin(ours)
                         & deals_hist["entry"].isin([0, 1, 2, 3])]
    md = write_report(day, results, mt5.account_info(), now_utc, foreign, market_context(start, end, day))
    path = os.path.join(REPORT_DIR, f"review_{day:%Y-%m-%d}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    update_summary(day, results)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None, help="交易日（纽约 17:00 为界），默认最近一个已收盘的交易日")
    args = ap.parse_args()
    connect()
    path = run_review(args.day)
    print(open(path, encoding="utf-8").read())
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()

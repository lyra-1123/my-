# -*- coding: utf-8 -*-
"""
模拟盘引擎：确定性重放。

每次运行都从全部历史数据重新计算信号与仓位（因子只用 <= t 的数据，结果可复现），
再从 forward_start 起生成前向账本。成交模型与研究评估完全一致：
  - 第 t 根收盘出信号，第 t+1 根开盘成交；
  - 点差 0.2$/0.01 手/开平一次（每单位仓位变化收半个点差）；过夜费 0.47$/0.01 手/次，纽约 17:00 换日，周三 ×3；
  - ≤1H 频率换日前平仓、换日后 1 小时不开仓。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from factors.core import atr, params, trading_day
from factors.data_loader import load_m1, resample_ohlcv
from factors.evaluate import BAR, SPREAD, SWAP, execution_signal, swap_units, targets
from paper.specs import FINGERPRINT_WINDOW, StrategySpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "paper", "state")


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_bars(data_dir: str, freq: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """读取 M1（按文件名/大小/修改时间缓存），重采样，并丢弃最后一根未走完的 K 线。"""
    files = sorted(glob.glob(os.path.join(data_dir, "DAT_ASCII_XAUUSD_M1_*.csv*")))
    key = hashlib.md5("|".join(f"{f}:{os.path.getsize(f)}:{os.path.getmtime(f)}" for f in files).encode()).hexdigest()[:12]
    cache = os.path.join(data_dir, "cache", f"m1_{key}.pkl")
    if os.path.exists(cache):
        m1 = pd.read_pickle(cache)
    else:
        m1 = load_m1(data_dir)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        for old in glob.glob(os.path.join(data_dir, "cache", "m1_*.pkl")):
            os.remove(old)
        m1.to_pickle(cache)
    last_m1 = m1.index[-1]
    bars = resample_ohlcv(m1, freq)
    dur = pd.Timedelta(BAR[freq])
    complete = bars.index + dur <= last_m1 + pd.Timedelta("1min")
    return bars[complete], last_m1


# ---------------------------------------------------------------------------
# 信号与仓位
# ---------------------------------------------------------------------------
def compute(spec: StrategySpec, bars: pd.DataFrame) -> pd.DataFrame:
    if spec.rule != "standard":
        return compute_rule(spec, bars)
    z = spec.signal(bars)
    zx = execution_signal(z, spec.freq)
    tgt = targets(zx, spec.entry, spec.exit)
    out = pd.DataFrame({"open": bars["open"], "close": bars["close"], "z": z, "z_exec": zx,
                        "target": tgt, "pos": tgt.shift(1).fillna(0.0)})
    a = atr(bars, params(spec.freq)["atr"])
    out["atr_prev"] = a.shift(1)
    oz = spec.lots * 100
    move = bars["open"].shift(-1) - bars["open"]
    dpos = out["pos"].diff().abs().fillna(out["pos"].abs())
    out["gross"] = out["pos"] * move * oz
    out["spread"] = dpos * SPREAD / 2 * oz
    out["swap"] = out["pos"].abs() * swap_units(bars.index) * SWAP * oz
    out["net"] = out["gross"] - out["spread"] - out["swap"]
    # R 单位：收益与成本都除以当时（上一根）的 ATR，便于跨价格水平比较
    out["net_R"] = (out["pos"] * move - dpos * SPREAD / 2 - out["pos"].abs() * swap_units(bars.index) * SWAP) / out["atr_prev"]
    return out


def compute_rule(spec: StrategySpec, bars: pd.DataFrame) -> pd.DataFrame:
    """非统一执行规则（paper/rules.py）。输出列与 compute 相同，另有 stop_fill / next_stop。"""
    from factors.core import htf_feature
    from paper.rules import state_trail
    if spec.rule != "state_trail":
        raise ValueError(f"未知执行规则 {spec.rule}")
    z = spec.signal(bars)
    ny = bars.index.tz_localize("UTC").tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    start = 17 * 60 - 2 * pd.Timedelta(BAR[spec.freq]).seconds // 60
    flat = np.asarray((mins >= start) & (mins < 18 * 60))
    atrd = np.nan_to_num(htf_feature(bars, spec.freq, "1D", lambda b: atr(b, 14)).to_numpy())
    kw = dict(spec.rule_params)
    pos_open, pnl, dpos, held_end, stop_fill, next_stop, final = state_trail(
        np.nan_to_num(z.to_numpy()), bars["open"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy(),
        flat, atrd, spec.entry, spec.exit, kw.get("trail", 2.0))
    oz = spec.lots * 100
    out = pd.DataFrame({"open": bars["open"], "close": bars["close"], "z": z,
                        "z_exec": z.where(~flat, 0.0), "pos": pos_open}, index=bars.index)
    out["target"] = out["pos"].shift(-1).fillna(final)
    a = atr(bars, params(spec.freq)["atr"])
    out["atr_prev"] = a.shift(1)
    sw = held_end * swap_units(bars.index).to_numpy() * SWAP
    out["gross"] = pnl * oz
    out["spread"] = dpos * SPREAD / 2 * oz
    out["swap"] = sw * oz
    out["net"] = out["gross"] - out["spread"] - out["swap"]
    out["net_R"] = (pnl - dpos * SPREAD / 2 - sw) / out["atr_prev"]
    out["stop_fill"] = stop_fill
    out["next_stop"] = next_stop
    return out


def fingerprint(spec: StrategySpec, bars: pd.DataFrame) -> str:
    """行为指纹：固定历史区间上的信号（6 位小数）与目标仓位；非统一规则另加实际持仓。"""
    c = compute(spec, bars).loc[FINGERPRINT_WINDOW[0]:FINGERPRINT_WINDOW[1]]
    payload = np.round(c["z"].fillna(0).to_numpy(), 6).tobytes() + c["target"].to_numpy().tobytes()
    if spec.rule != "standard":
        payload += spec.rule.encode() + repr(spec.rule_params).encode() + c["pos"].to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------
def trades_from(c: pd.DataFrame, lots: float) -> pd.DataFrame:
    """把持仓序列切成逐笔交易。每笔点差：已平仓 = 开 + 平 = 0.2$×盎司；未平仓 = 只计开仓的一半。"""
    oz = lots * 100
    p = c["pos"]
    seg = (p != p.shift(1)).cumsum()
    cols = ["entry_time", "side", "entry_open", "exit_time", "exit_open", "bars", "gross", "spread", "swap", "net", "is_open"]
    g = c[p != 0].assign(seg=seg[p != 0])
    rows = []
    for _, t in g.groupby("seg"):
        i_last = c.index.get_loc(t.index[-1])
        closed = i_last + 1 < len(c)
        stop_px = t["stop_fill"].iloc[-1] if "stop_fill" in t else np.nan
        stopped = not np.isnan(stop_px)
        rows.append({"entry_time": t.index[0], "side": "LONG" if t["pos"].iloc[0] > 0 else "SHORT",
                     "entry_open": t["open"].iloc[0],
                     "exit_time": (t.index[-1] if stopped else c.index[i_last + 1]) if closed else pd.NaT,
                     "exit_open": (stop_px if stopped else c["open"].iloc[i_last + 1]) if closed else np.nan,
                     "bars": len(t), "gross": t["gross"].sum(),
                     "spread": SPREAD * oz * (1.0 if closed else 0.5), "swap": t["swap"].sum(), "is_open": not closed})
    tr = pd.DataFrame(rows, columns=cols)
    tr["net"] = tr["gross"] - tr["spread"] - tr["swap"]
    return tr


def daily_from(c: pd.DataFrame) -> pd.DataFrame:
    d = c.groupby(trading_day(c.index)).agg(net_usd=("net", "sum"), net_R=("net_R", "sum"), pos_end=("pos", "last"))
    d["cum_usd"] = d["net_usd"].cumsum()
    d["cum_R"] = d["net_R"].cumsum()
    return d


def next_action(c: pd.DataFrame, spec: StrategySpec, last_m1: pd.Timestamp) -> dict:
    last = c.iloc[-1]
    cur, tgt = float(last["pos"]), float(last["target"])
    lots = spec.lots
    if tgt == cur:
        act = "HOLD" if cur != 0 else "FLAT（无操作）"
    elif tgt == 0:
        act = f"CLOSE {'LONG' if cur > 0 else 'SHORT'} {lots} 手"
    elif cur == 0:
        act = f"{'BUY' if tgt > 0 else 'SELL'} {lots} 手"
    else:
        act = f"REVERSE → {'BUY' if tgt > 0 else 'SELL'} {lots} 手（先平后开）"
    bar_end = c.index[-1] + pd.Timedelta(BAR[spec.freq])
    return {"strategy": spec.id, "as_of_bar": str(c.index[-1]), "bar_close_utc": str(bar_end), "last_m1_utc": str(last_m1),
            "z": round(float(last["z"]), 3), "z_exec": round(float(last["z_exec"]), 3),
            "current_position": cur, "target_position": tgt,
            "action_at_next_open": act, "entry": spec.entry, "exit": spec.exit, "rule": spec.rule,
            "stop_for_next_bar": (round(float(last["next_stop"]), 3) if "next_stop" in c and not np.isnan(last["next_stop"]) else None),
            "in_flat_window": bool(last["z_exec"] == 0 and last["z"] != 0)}

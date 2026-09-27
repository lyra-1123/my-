"""
signal_support_resistance.py
===============================
假设7：支撑/阻力（00_方案/hypothesis_support_resistance.md）。价位识别见 lib/goldq/levels.py。

两种用法：
  filter  只保留"较低底落在事先已存在的支撑/阻力位附近（<= NEAR_ATR×ATR）"的双底，双顶对称；
          价位必须在第一个底/顶出现之前就已知
  bounce  最低价触及支撑（<= 价位 + TOUCH_ATR×ATR）且收盘回到价位上方、前一根收盘也在上方 → 做多；
          阻力对称做空。止损 = min(当根最低, 价位) - STOP_BUFFER_ATR×ATR；同方向 COOLDOWN_BARS 根内不重复
出场统一：给定止损 + 2R 止盈，不限持仓（遇 >3 小时停盘平仓）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from goldq.exits import ExitRule  # noqa: E402
from goldq.levels import (htf_swing_levels, near_level, prev_day_week_levels,  # noqa: E402
                          round_number_levels, swing_cluster_levels)
from signal_double_top_bottom import TIMEFRAMES, build_features, load_m1, load_tf  # noqa: E402,F401

METHODS = {"cluster": "多次触及", "htf": "大周期高低点", "prev_dw": "前日/前周高低点", "round": "整数关口$50"}
USES = {"filter": "过滤双顶双底", "bounce": "触及反弹"}
HTF_FOR = {"M5": "4h", "M15": "4h", "H1": "D"}
NEAR_ATR = 0.5
TOUCH_ATR = 0.1
STOP_BUFFER_ATR = 0.2
COOLDOWN_BARS = 6
ROUND_STEP = 50.0
EXIT_RULE = ExitRule("given_2R", "given", "fixed_r", max_bars=None, r_multiple=2.0)


def level_sets(feat: pd.DataFrame, tf: str) -> dict[str, np.ndarray]:
    return {
        "cluster": swing_cluster_levels(feat),
        "htf": htf_swing_levels(feat, HTF_FOR[tf]),
        "prev_dw": prev_day_week_levels(feat),
        "round": round_number_levels(feat, ROUND_STEP),
    }


def filter_signal(feat: pd.DataFrame, patterns: pd.DataFrame, levels: np.ndarray) -> tuple[pd.Series, pd.Series]:
    atr, base_sig = feat["atr"].to_numpy(), feat["signal"].to_numpy()
    sig = np.zeros(len(feat), dtype=int)
    stop = np.full(len(feat), np.nan)
    for p in patterns.itertuples():
        if base_sig[p.t] != p.direction:
            continue
        price = p.bottom * p.direction  # 双顶存的是取负后的价格
        if near_level(price, levels[p.i1], NEAR_ATR * atr[p.i2]):
            sig[p.t], stop[p.t] = p.direction, p.stop_dist
    return pd.Series(sig, index=feat.index), pd.Series(stop, index=feat.index)


def bounce_signal(feat: pd.DataFrame, levels: np.ndarray) -> tuple[pd.Series, pd.Series]:
    low, high, close, atr = (feat[c].to_numpy(dtype=float) for c in ("low", "high", "close", "atr"))
    prev_close = np.r_[np.nan, close[:-1]]
    L = levels
    tol = (TOUCH_ATR * atr)[:, None]
    with np.errstate(invalid="ignore"):
        long_ok = (low[:, None] <= L + tol) & (close[:, None] > L) & (prev_close[:, None] > L)
        short_ok = (high[:, None] >= L - tol) & (close[:, None] < L) & (prev_close[:, None] < L)
    support = np.where(long_ok, L, -np.inf).max(axis=1)
    resistance = np.where(short_ok, L, np.inf).min(axis=1)
    long_any, short_any = long_ok.any(axis=1), short_ok.any(axis=1)

    sig = np.zeros(len(feat), dtype=int)
    stop = np.full(len(feat), np.nan)
    last = {1: -10**9, -1: -10**9}
    for t in np.flatnonzero((long_any ^ short_any) & np.isfinite(atr)):
        d = 1 if long_any[t] else -1
        if t - last[d] <= COOLDOWN_BARS:
            continue
        if d == 1:
            dist = close[t] - (min(low[t], support[t]) - STOP_BUFFER_ATR * atr[t])
        else:
            dist = (max(high[t], resistance[t]) + STOP_BUFFER_ATR * atr[t]) - close[t]
        sig[t], stop[t], last[d] = d, dist, t
    return pd.Series(sig, index=feat.index), pd.Series(stop, index=feat.index)


def all_signals(feat: pd.DataFrame, patterns: pd.DataFrame, tf: str) -> dict[tuple[str, str], tuple[pd.Series, pd.Series]]:
    out = {}
    for method, lv in level_sets(feat, tf).items():
        out[("filter", method)] = filter_signal(feat, patterns, lv)
        out[("bounce", method)] = bounce_signal(feat, lv)
    return out


def describe_levels(feat: pd.DataFrame, tf: str) -> pd.DataFrame:
    """最后一根bar当时已知的各类价位，供在 MT5 图上核对（UTC，bid 价）。"""
    rows = []
    for method, lv in level_sets(feat, tf).items():
        vals = np.sort(lv[-1][np.isfinite(lv[-1])])
        rows.append({"方法": METHODS[method], "价位": ", ".join(f"{v:.2f}" for v in np.unique(vals.round(2)))})
    return pd.DataFrame(rows)

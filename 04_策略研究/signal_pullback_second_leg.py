"""
signal_pullback_second_leg.py
================================
假设9：趋势回调 50% 入场，等第二段（ABC 等幅）。00_方案/hypothesis_pullback_second_leg.md。

做多（做空把价格取负后复用同一套逻辑）：
  A   已确认的摆动低点（左右各 SWING_K 根），第 a+SWING_K 根收盘确认
  B   A 之后到上一根为止的最高价（新高出现时 B 上移）
  挂单 A 确认后，满足以下条件时在 L = (A+B)/2 挂买入限价：
        B - A >= LEG_ATR × ATR（上一根的 ATR）
        大一级周期上一根已收盘的 K 线收盘价在 EMA50 上方（M5 看 M15，M15 看 H1）
        B 出现之后价格还没碰过 L（否则这次 50% 机会已经错过，等下一个新高）
  成交 某根最低价 <= L：按 min(L, 开盘价) 成交；开盘价已低于 A 的不成交
  撤单 跌破 A，或 A 之后 MAX_WAIT 根仍未成交
  止损 A（起涨点，按用户描述不加缓冲）
  止盈 成交价 + (B - A)：第二段高度 = 第一段高度；恰好在 50% 成交时盈亏比 1:2
  每个 A 最多一笔；同一根 K 线上多个 A 同时成交只取第一个
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.exits import ExitRule  # noqa: E402
from goldq.levels import SWING_K, swing_lows  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from compute_indicators import compute_atr, compute_ema  # noqa: E402

TIMEFRAMES = {"M5": "5min", "M15": "15min"}
HTF_FOR = {"M5": "15min", "M15": "1h"}
LEG_ATR_GRID = [2.0, 3.0, 5.0]
MAX_WAIT = 100
HTF_EMA = 50
EXIT_RULE = ExitRule("stopA_target_leg", "given", "given", max_bars=None)


def htf_trend(ltf: pd.DataFrame, m1: pd.DataFrame, ltf_rule: str, htf_rule: str) -> pd.Series:
    """每根小周期K线收盘时，最近一根已收盘的大周期K线：收盘>EMA50 为 +1，< 为 -1。"""
    htf = resample_ohlcv(m1, rule=htf_rule)
    htf["trend"] = np.sign(htf["close"] - compute_ema(htf["close"], HTF_EMA))
    htf["known_at"] = htf["time_utc"] + pd.Timedelta(htf_rule)
    left = pd.DataFrame({"known_at": ltf["time_utc"] + pd.Timedelta(ltf_rule), "row": ltf.index})
    merged = pd.merge_asof(left.sort_values("known_at"), htf[["known_at", "trend"]].sort_values("known_at"),
                           on="known_at", direction="backward")
    return merged.set_index("row")["trend"].reindex(ltf.index).fillna(0)


def detect(low: np.ndarray, high: np.ndarray, open_: np.ndarray, close: np.ndarray,
           atr: np.ndarray, trend_ok: np.ndarray, leg_atr: float) -> list[dict]:
    """做多方向的识别；trend_ok[j] 表示第 j 根收盘时大周期为多头。"""
    n = len(close)
    out = []
    for a in swing_lows(low, SWING_K):
        a_low = low[a]
        b, min_since_b = -np.inf, np.inf
        for j in range(a + 1, min(a + MAX_WAIT, n - 1) + 1):
            if j > a + SWING_K and np.isfinite(b):
                level = (a_low + b) / 2
                leg = b - a_low
                if (leg >= leg_atr * atr[j - 1] and trend_ok[j - 1] and min_since_b > level
                        and open_[j] > a_low and low[j] <= level):
                    fill = min(level, open_[j])
                    out.append({"t": j, "a": a, "a_price": a_low, "b_price": b, "fill": fill,
                                "stop_dist": fill - a_low, "target_dist": leg,
                                "post_hi": max(fill, close[j]), "post_lo": low[j]})
                    break
            if low[j] < a_low:
                break
            if high[j] > b:
                b, min_since_b = high[j], np.inf
            else:
                min_since_b = min(min_since_b, low[j])
    return out


def build_features(ltf: pd.DataFrame, trend: pd.Series) -> pd.DataFrame:
    df = ltf.copy()
    df["atr"] = compute_atr(df, 14, max_gap_minutes=180)
    df["htf_trend"] = trend.to_numpy()
    return df


def signals_for(feat: pd.DataFrame, leg_atr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (逐bar信号列, 形态明细)。逐bar列：signal / entry_price / entry_hi / entry_lo / stop_dist / target_dist。"""
    o, h, l, c, atr = (feat[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "atr"))
    tr = feat["htf_trend"].to_numpy()
    n = len(feat)
    cols = {k: np.full(n, np.nan) for k in ("entry_price", "entry_hi", "entry_lo", "stop_dist", "target_dist")}
    signal = np.zeros(n, dtype=int)
    pats = []
    for d, pl in ((1, detect(l, h, o, c, atr, tr > 0, leg_atr)),
                  (-1, detect(-h, -l, -o, -c, atr, tr < 0, leg_atr))):
        for p in pl:
            t = p["t"]
            if signal[t] != 0:
                continue
            signal[t] = d
            cols["entry_price"][t] = p["fill"] * d
            cols["stop_dist"][t] = p["stop_dist"]
            cols["target_dist"][t] = p["target_dist"]
            # 取负空间里的 [post_lo, post_hi] 换回真实价格
            cols["entry_hi"][t] = p["post_hi"] if d == 1 else -p["post_lo"]
            cols["entry_lo"][t] = p["post_lo"] if d == 1 else -p["post_hi"]
            pats.append({"direction": d, "t": t, "a": p["a"], "A": p["a_price"] * d,
                         "B": p["b_price"] * d, "fill": p["fill"] * d})
    out = pd.DataFrame(cols, index=feat.index)
    out["signal"] = signal
    return out, pd.DataFrame(pats)


def load_m1(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    return fetch_bars(symbol=symbol, start=start, end=end)


def load_features(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    ltf = resample_ohlcv(m1, rule=TIMEFRAMES[tf])
    return build_features(ltf, htf_trend(ltf, m1, TIMEFRAMES[tf], HTF_FOR[tf]))


def describe_patterns(feat: pd.DataFrame, pats: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
    if pats.empty:
        return pats
    ex = pats.sort_values("t").tail(n_examples)
    times = feat["time_utc"].to_numpy()
    return pd.DataFrame({
        "方向": np.where(ex["direction"] > 0, "做多", "做空"),
        "起涨点A时间": times[ex["a"].to_numpy()], "A": ex["A"].round(2).to_numpy(),
        "B": ex["B"].round(2).to_numpy(),
        "50%成交时间": times[ex["t"].to_numpy()], "成交价": ex["fill"].round(2).to_numpy(),
        "止盈": (ex["fill"] + ex["direction"] * (ex["B"] - ex["A"]).abs()).round(2).to_numpy(),
    })

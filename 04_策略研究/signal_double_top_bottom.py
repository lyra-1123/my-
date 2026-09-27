"""
signal_double_top_bottom.py
==============================
假设6：双顶 / 双底（00_方案/hypothesis_double_top_bottom.md）。

双底（双顶对称，做法是把价格取负后复用同一套识别）：
  1. 摆动低点：low[i] 严格低于左边 SWING_K 根、且不高于右边 SWING_K 根的最低价；
     右边 SWING_K 根走完（i+SWING_K 收盘）才算确认
  2. 两个摆动低点 i1 < i2：间隔 MIN_SEP~MAX_SEP 根；|low[i1]-low[i2]| <= TOL_ATR×ATR(i2)；
     两底之间没有更低的价格；颈线 = 两底之间最高价，颈线 - 较低底 >= MIN_DEPTH_ATR×ATR(i2)
     （对每个新确认的 i2，取满足条件的最近一个 i1）
  3. i2 确认后到 i2+BREAKOUT_WINDOW 根内，收盘价从颈线下方（含）上穿到颈线上方的那根入场做多；
     期间最低价跌破较低底则形态作废
  4. 止损距离 = 入场价 - (较低底 - STOP_BUFFER_ATR×ATR(入场))；
     形态目标距离 = 颈线 + 形态高度(颈线-较低底) - 入场价（<=0 则该笔不适用形态目标出场）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.exits import ExitRule  # noqa: E402
from goldq.levels import swing_lows  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from compute_indicators import compute_atr  # noqa: E402

SWING_K = 3
MIN_SEP, MAX_SEP = 5, 50
TOL_ATR = 0.5
MIN_DEPTH_ATR = 1.0
BREAKOUT_WINDOW = 30
STOP_BUFFER_ATR = 0.2
ATR_PERIOD = 14
ATR_GAP_MINUTES = 180
TIMEFRAMES = {"M5": "5min", "M15": "15min", "H1": "1h"}

EXIT_RULES = [
    ExitRule("pattern_target", "given", "given", max_bars=None),
    ExitRule("pattern_trail", "given_trail", "none", max_bars=None, atr_stop_mult=1.5),
    ExitRule("pattern_2R", "given", "fixed_r", max_bars=None, r_multiple=2.0),
]


def detect_double_bottoms(low: np.ndarray, high: np.ndarray, close: np.ndarray,
                          atr: np.ndarray) -> list[dict]:
    """返回每个入场信号：{t, i1, i2, bottom, neck, stop_dist, target_dist}。做双顶时传入取负的价格。"""
    n = len(close)
    swings = swing_lows(low, SWING_K)
    out = []
    for p2, i2 in enumerate(swings):
        conf = i2 + SWING_K
        if conf >= n:
            break
        a = atr[i2]
        if not np.isfinite(a) or a <= 0:
            continue

        pattern = None
        p1 = p2 - 1
        while p1 >= 0 and i2 - swings[p1] <= MAX_SEP:
            i1 = swings[p1]
            p1 -= 1
            if i2 - i1 < MIN_SEP or abs(low[i1] - low[i2]) > TOL_ATR * a:
                continue
            bottom = min(low[i1], low[i2])
            if low[i1:i2 + 1].min() < bottom:
                continue
            neck = high[i1:i2 + 1].max()
            if neck - bottom < MIN_DEPTH_ATR * a:
                continue
            pattern = (i1, bottom, neck)
            break
        if pattern is None:
            continue

        i1, bottom, neck = pattern
        for t in range(conf, min(i2 + BREAKOUT_WINDOW, n - 1) + 1):
            if low[t] < bottom:
                break
            if close[t] > neck and close[t - 1] <= neck:
                at = atr[t]
                if not np.isfinite(at) or at <= 0:
                    break
                stop_dist = close[t] - (bottom - STOP_BUFFER_ATR * at)
                target_dist = neck + (neck - bottom) - close[t]
                out.append({"t": t, "i1": i1, "i2": i2, "bottom": bottom, "neck": neck,
                            "stop_dist": stop_dist,
                            "target_dist": target_dist if target_dist > 0 else np.nan})
                break
    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (feat, patterns)。patterns 不放进 df.attrs：pandas 合并数据时会比较 attrs，放 DataFrame 会报错。"""
    df = df.copy()
    df["atr"] = compute_atr(df, ATR_PERIOD, max_gap_minutes=ATR_GAP_MINUTES)
    low, high, close, atr = (df[c].to_numpy(dtype=float) for c in ("low", "high", "close", "atr"))

    n = len(df)
    signal = np.zeros(n, dtype=int)
    stop_dist = np.full(n, np.nan)
    target_dist = np.full(n, np.nan)
    patterns = []
    for d, pats in ((1, detect_double_bottoms(low, high, close, atr)),
                    (-1, detect_double_bottoms(-high, -low, -close, atr))):
        for p in pats:
            t = p["t"]
            if signal[t] == -d:  # 同一根bar既突破双底颈线又跌破双顶颈线：不交易
                signal[t], stop_dist[t], target_dist[t] = 0, np.nan, np.nan
                continue
            if signal[t] == d:
                continue
            signal[t], stop_dist[t], target_dist[t] = d, p["stop_dist"], p["target_dist"]
            patterns.append({"direction": d, **p})

    df["signal"] = signal
    df["stop_dist"] = stop_dist
    df["target_dist"] = target_dist
    return df, pd.DataFrame(patterns)


def load_m1(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    return fetch_bars(symbol=symbol, start=start, end=end)


def load_tf(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    return resample_ohlcv(m1, rule=TIMEFRAMES[tf])


def describe_patterns(feat: pd.DataFrame, pats: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
    """把识别出的形态换成时间和价位，方便在 MT5 图上人工核对。价位是 bid 价（Dukascopy）。"""
    if pats.empty:
        return pats
    times = feat["time_utc"].to_numpy()
    ex = pats.sort_values("t").tail(n_examples).copy()
    sign = ex["direction"]
    return pd.DataFrame({
        "类型": np.where(sign > 0, "双底", "双顶"),
        "第一个底/顶": times[ex["i1"].to_numpy()],
        "第二个底/顶": times[ex["i2"].to_numpy()],
        "底/顶价位": (ex["bottom"] * sign).round(2).to_numpy(),
        "颈线": (ex["neck"] * sign).round(2).to_numpy(),
        "突破入场": times[ex["t"].to_numpy()],
        "入场价": feat["close"].to_numpy()[ex["t"].to_numpy()],
    })


if __name__ == "__main__":
    m1 = load_m1()
    for tf in TIMEFRAMES:
        feat, _ = build_features(load_tf(m1, tf))
        s = feat["signal"]
        print(f"{tf}: 双底 {int((s == 1).sum())} 个，双顶 {int((s == -1).sum())} 个")

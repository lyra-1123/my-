"""
levels.py
==========
支撑/阻力位的客观识别。每种方法返回形如 (n_bars, K) 的数组：第 t 行是"第 t 根bar开盘前已经知道的"
价位（不足 K 个用 NaN 补齐），严格因果。价位不分支撑/阻力：在价格下方就是支撑，上方就是阻力
（角色互换）。

  swing_cluster   最近 lookback 根内，>= min_touches 个已确认摆动高/低点聚在 tol_atr×ATR 以内
                  → 取均值为一个价位，存活 lookback 根；保留最近 K 个
  htf_swings      大周期（H4 / 交易日）最近 n_each 个已确认摆动高点 + n_each 个摆动低点
  prev_day_week   前一交易日 高/低/收 + 前一交易周 高/低（交易日以纽约 17:00 为界，
                  与多数 MT5 券商日线一致）
  round_numbers   前一根收盘价附近的 step 美元整数关口（上下各两个）
"""

from bisect import bisect_right

import numpy as np
import pandas as pd

SWING_K = 3


def swing_lows(low: np.ndarray, k: int = SWING_K) -> np.ndarray:
    """low[i] 严格低于左边 k 根、不高于右边 k 根的索引；在 i+k 收盘时才确认。"""
    s = pd.Series(low)
    left_min = s.shift(1).rolling(k).min()
    right_min = s.shift(-1).rolling(k).min().shift(-(k - 1))
    return np.flatnonzero(((s < left_min) & (s <= right_min)).to_numpy())


def swing_highs(high: np.ndarray, k: int = SWING_K) -> np.ndarray:
    return swing_lows(-np.asarray(high), k)


def trading_date(time_utc: pd.Series) -> pd.Series:
    """纽约 17:00 为日界的交易日（周日晚开盘归入周一）。"""
    t = pd.to_datetime(time_utc)
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    ny = t.dt.tz_convert("America/New_York")
    return (ny + pd.Timedelta(hours=7)).dt.tz_localize(None).dt.normalize()


def _pad(rows: list[list[float]], k: int) -> np.ndarray:
    out = np.full((len(rows), k), np.nan)
    for i, r in enumerate(rows):
        r = r[-k:]
        out[i, :len(r)] = r
    return out


def swing_cluster_levels(df: pd.DataFrame, lookback: int = 200, min_touches: int = 3,
                         tol_atr: float = 0.5, max_levels: int = 10) -> np.ndarray:
    high, low, atr = (df[c].to_numpy(dtype=float) for c in ("high", "low", "atr"))
    n = len(df)
    pts = [(i + SWING_K, high[i], i) for i in swing_highs(high)] + \
          [(i + SWING_K, low[i], i) for i in swing_lows(low)]
    pts = sorted(p for p in pts if p[0] < n)

    created = []  # (可用起始行, 失效行, 价位)
    recent = []   # (确认行, 价格)
    for conf, price, i in pts:
        recent = [(c, p) for c, p in recent if conf - c <= lookback]
        tol = tol_atr * atr[i] if np.isfinite(atr[i]) else np.nan
        if np.isfinite(tol):
            near = [p for _, p in recent if abs(p - price) <= tol]
            if len(near) + 1 >= min_touches:
                created.append((conf + 1, conf + 1 + lookback, float(np.mean(near + [price]))))
        recent.append((conf, price))

    out = np.full((n, max_levels), np.nan)
    if not created:
        return out
    # 逐段填充：在任意两个事件（创建/失效）之间，活跃价位集合不变
    # created 按起始行排序，存活期相同所以也按失效行排序：[lo, hi) 就是在 a 行活跃的价位
    starts = [s for s, _, _ in created]
    bounds = sorted({0, n} | {s for s in starts if s < n} | {e for _, e, _ in created if e < n})
    lo = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        while lo < len(created) and created[lo][1] <= a:
            lo += 1
        hi = bisect_right(starts, a)
        act = [lv for _, _, lv in created[max(lo, hi - max_levels):hi]]
        if act:
            out[a:b, :len(act)] = act
    return out


def htf_swing_levels(df: pd.DataFrame, htf: str, n_each: int = 5) -> np.ndarray:
    """htf: "4h" 或 "D"（交易日）。大周期摆动点在其右侧第 SWING_K 根大周期bar结束后才可用。"""
    if htf == "D":
        key = trading_date(df["time_utc"])
    else:
        t = pd.to_datetime(df["time_utc"])
        key = t.dt.floor(htf)
    codes, uniques = pd.factorize(key, sort=True)
    g = pd.DataFrame({"g": codes, "high": df["high"].to_numpy(), "low": df["low"].to_numpy()}).groupby("g")
    hh, ll = g["high"].max().to_numpy(), g["low"].min().to_numpy()

    events = sorted([(i + SWING_K, "H", hh[i]) for i in swing_highs(hh)] +
                    [(i + SWING_K, "L", ll[i]) for i in swing_lows(ll)])
    per_htf = np.full((len(uniques), 2 * n_each), np.nan)
    highs, lows = [], []
    ev = 0
    for j in range(len(uniques)):
        while ev < len(events) and events[ev][0] < j:  # 在第 j 根大周期bar开始前已确认
            _, kind, price = events[ev]
            (highs if kind == "H" else lows).append(price)
            ev += 1
        row = highs[-n_each:] + lows[-n_each:]
        per_htf[j, :len(row)] = row
    return per_htf[codes]


def prev_day_week_levels(df: pd.DataFrame) -> np.ndarray:
    td = trading_date(df["time_utc"])
    frame = pd.DataFrame({"td": td.to_numpy(), "high": df["high"].to_numpy(),
                          "low": df["low"].to_numpy(), "close": df["close"].to_numpy()})
    daily = frame.groupby("td").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev_daily = daily.shift(1)
    iso = daily.index.isocalendar()
    week_key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    weekly = daily.groupby(week_key.to_numpy()).agg(high=("high", "max"), low=("low", "min"))
    prev_weekly = weekly.shift(1)

    d = prev_daily.reindex(td.to_numpy())
    w = prev_weekly.reindex(week_key.reindex(td.to_numpy()).to_numpy())
    return np.column_stack([d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(),
                            w["high"].to_numpy(), w["low"].to_numpy()])


def round_number_levels(df: pd.DataFrame, step: float = 50.0) -> np.ndarray:
    prev_close = df["close"].shift(1).to_numpy(dtype=float)
    base = np.floor(prev_close / step) * step
    return np.column_stack([base - step, base, base + step, base + 2 * step])


def near_level(price: float, levels_row: np.ndarray, tol: float) -> bool:
    return bool(np.any(np.abs(levels_row - price) <= tol))

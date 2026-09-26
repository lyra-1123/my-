"""
signal_mtf_golden_cross_cascade.py
====================================
假设见 00_方案/hypothesis_mtf_golden_cross_cascade.md。

信号：1H 水下金叉（+走窄）之后，30min/15min 也处于金叉后有效期内，
在 5min 金叉confirm 那一刻入场。

严格因果：每个5min时刻只使用"已经完全走完"的高周期K线（confirm_time = bar起始 + 周期长度），
不使用尚未收盘的高周期bar。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from compute_macd import compute_macd  # noqa: E402

NARROWING_LOOKBACK_1H = 5
VALID_BARS_1H = 3    # 1H 金叉后有效期：3 根（3 小时）
VALID_BARS_30M = 6   # 30min 金叉后有效期：6 根（3 小时）
VALID_BARS_15M = 12  # 15min 金叉后有效期：12 根（3 小时）

TIMEFRAME_RULES = {"1h": "1h", "30m": "30min", "15m": "15min", "5m": "5min"}


def _bullish_state_with_bars_since(hist: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    返回 (is_bullish, bars_in_state)：
    - is_bullish[i] = HIST[i] >= 0（当前处于金叉后，尚未死叉）
    - bars_in_state[i] = 这个状态已经持续了几根（0 = 状态刚开始，即金叉/死叉confirm的那一根）
    """
    is_bullish = hist >= 0
    state_change = is_bullish != is_bullish.shift(1).fillna(is_bullish.iloc[0])
    group_id = state_change.cumsum()
    bars_in_state = group_id.groupby(group_id).cumcount()
    return is_bullish, bars_in_state


def compute_1h_valid_state(df_1h: pd.DataFrame) -> pd.Series:
    """1H：只有"水下金叉 + 走窄"触发的多头状态才算数，且必须在 VALID_BARS_1H 根以内。"""
    macd = compute_macd(df_1h["close"])
    dif, dea, hist = macd["dif"], macd["dea"], macd["hist"]

    pre_hist = hist.shift(1)
    pre_dif = dif.shift(1)
    pre_dea = dea.shift(1)
    hist_n_bars_before_cross = hist.shift(1 + NARROWING_LOOKBACK_1H)

    is_underwater = (pre_dif < 0) & (pre_dea < 0)
    is_narrowing = pre_hist > hist_n_bars_before_cross

    is_bullish, bars_in_state = _bullish_state_with_bars_since(hist)

    # 该 bullish 状态开始那一根（bars_in_state==0）是否满足水下+走窄，broadcast到整个状态区间
    state_change = is_bullish != is_bullish.shift(1).fillna(is_bullish.iloc[0])
    group_id = state_change.cumsum()
    qualified_start = (is_underwater & is_narrowing).groupby(group_id).transform("first")

    valid = is_bullish & qualified_start & (bars_in_state <= VALID_BARS_1H)
    valid.name = "valid_1h"
    return valid


def compute_lower_tf_valid_state(df: pd.DataFrame, valid_bars: int) -> pd.Series:
    """30min/15min：任意金叉都算数，只要求当前仍在有效期内、未死叉。"""
    macd = compute_macd(df["close"])
    is_bullish, bars_in_state = _bullish_state_with_bars_since(macd["hist"])
    valid = is_bullish & (bars_in_state <= valid_bars)
    return valid


def align_to_5m(df_5m: pd.DataFrame, higher_tf_df: pd.DataFrame, valid_series: pd.Series,
                 period: pd.Timedelta, col_name: str) -> pd.DataFrame:
    """
    把高周期的 valid 状态严格因果地映射到 5min 时间线上：
    只用"已经完全走完"的高周期K线（confirm_time = bar起始时间 + 周期长度）。
    """
    higher = pd.DataFrame({
        "confirm_time": higher_tf_df["time_utc"] + period,
        col_name: valid_series.values,
    }).sort_values("confirm_time")

    merged = pd.merge_asof(
        df_5m.sort_values("time_utc"),
        higher,
        left_on="time_utc",
        right_on="confirm_time",
        direction="backward",
    )
    return merged


def generate_signal(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    df_m1 = fetch_bars(symbol=symbol, start=start, end=end)

    df_1h = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["1h"])
    df_30m = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["30m"])
    df_15m = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["15m"])
    df_5m = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["5m"])

    valid_1h = compute_1h_valid_state(df_1h)
    valid_30m = compute_lower_tf_valid_state(df_30m, VALID_BARS_30M)
    valid_15m = compute_lower_tf_valid_state(df_15m, VALID_BARS_15M)

    merged = align_to_5m(df_5m, df_1h, valid_1h, pd.Timedelta(hours=1), "valid_1h")
    merged = align_to_5m(merged, df_30m, valid_30m, pd.Timedelta(minutes=30), "valid_30m")
    merged = align_to_5m(merged, df_15m, valid_15m, pd.Timedelta(minutes=15), "valid_15m")

    macd_5m = compute_macd(merged["close"])
    pre_hist_5m = macd_5m["hist"].shift(1)
    golden_cross_5m = (pre_hist_5m < 0) & (macd_5m["hist"] >= 0)

    higher_tf_ok = merged["valid_1h"].fillna(False) & merged["valid_30m"].fillna(False) \
        & merged["valid_15m"].fillna(False)

    merged["signal"] = (golden_cross_5m & higher_tf_ok).astype(int)
    return merged


if __name__ == "__main__":
    df = generate_signal()
    n_signals = int(df["signal"].sum())
    print(f"[信号] M5 共 {len(df)} 根，触发 {n_signals} 次 "
          f"({n_signals / len(df) * 100:.4f}%)")
    print(df[df["signal"] == 1][["time_utc", "close"]].head(10).to_string(index=False))

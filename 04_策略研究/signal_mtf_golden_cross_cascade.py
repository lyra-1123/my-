"""
signal_mtf_golden_cross_cascade.py
====================================
假设见 00_方案/hypothesis_mtf_golden_cross_cascade.md。

信号：1H 水下金叉（+走窄）之后，30min/15min 也处于金叉后有效期内，
在 5min 金叉confirm 那一刻入场。

严格因果：每个5min时刻只使用"已经完全走完"的高周期K线（confirm_time = bar起始 + 周期长度），
不使用尚未收盘的高周期bar。共享逻辑见 lib/goldq/macd_cross.py。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from goldq.macd_cross import qualified_valid_state, simple_valid_state, golden_cross_now, align_to_lower_tf  # noqa: E402

NARROWING_LOOKBACK_1H = 5
VALID_BARS_1H = 3    # 1H 金叉后有效期：3 根（3 小时）
VALID_BARS_30M = 6   # 30min 金叉后有效期：6 根（3 小时）
VALID_BARS_15M = 12  # 15min 金叉后有效期：12 根（3 小时）

TIMEFRAME_RULES = {"1h": "1h", "30m": "30min", "15m": "15min", "5m": "5min"}


def generate_signal(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    df_m1 = fetch_bars(symbol=symbol, start=start, end=end)

    df_1h = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["1h"])
    df_30m = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["30m"])
    df_15m = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["15m"])
    df_5m = resample_ohlcv(df_m1, rule=TIMEFRAME_RULES["5m"])

    valid_1h = qualified_valid_state(df_1h, VALID_BARS_1H, NARROWING_LOOKBACK_1H)
    valid_30m = simple_valid_state(df_30m, VALID_BARS_30M)
    valid_15m = simple_valid_state(df_15m, VALID_BARS_15M)

    merged = align_to_lower_tf(df_5m, df_1h, valid_1h, pd.Timedelta(hours=1), "valid_1h")
    merged = align_to_lower_tf(merged, df_30m, valid_30m, pd.Timedelta(minutes=30), "valid_30m")
    merged = align_to_lower_tf(merged, df_15m, valid_15m, pd.Timedelta(minutes=15), "valid_15m")

    golden_cross_5m = golden_cross_now(merged)

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

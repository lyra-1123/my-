"""
signal_single_htf_golden_cross.py
====================================
假设见 00_方案/hypothesis_single_htf_golden_cross.md。

信号：单一大周期（1H / 30min / 15min，三选一）水下金叉+走窄，
在 5min 金叉confirm 那一刻入场。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from goldq.macd_cross import qualified_valid_state, golden_cross_now, align_to_lower_tf  # noqa: E402

NARROWING_LOOKBACK = 5

# 有效期窗口沿用"3小时等效根数"的约定
HTF_CONFIG = {
    "1h": {"rule": "1h", "valid_bars": 3, "period": pd.Timedelta(hours=1)},
    "30m": {"rule": "30min", "valid_bars": 6, "period": pd.Timedelta(minutes=30)},
    "15m": {"rule": "15min", "valid_bars": 12, "period": pd.Timedelta(minutes=15)},
}


def generate_signal(higher_tf: str, symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    if higher_tf not in HTF_CONFIG:
        raise ValueError(f"higher_tf 必须是 {list(HTF_CONFIG)} 之一，收到: {higher_tf}")
    cfg = HTF_CONFIG[higher_tf]

    df_m1 = fetch_bars(symbol=symbol, start=start, end=end)
    df_htf = resample_ohlcv(df_m1, rule=cfg["rule"])
    df_5m = resample_ohlcv(df_m1, rule="5min")

    valid_htf = qualified_valid_state(df_htf, cfg["valid_bars"], NARROWING_LOOKBACK)
    merged = align_to_lower_tf(df_5m, df_htf, valid_htf, cfg["period"], "valid_htf")

    golden_cross_5m = golden_cross_now(merged)
    merged["signal"] = (golden_cross_5m & merged["valid_htf"].fillna(False)).astype(int)
    return merged


if __name__ == "__main__":
    for htf in HTF_CONFIG:
        df = generate_signal(htf)
        n_signals = int(df["signal"].sum())
        print(f"[{htf}] M5 共 {len(df)} 根，触发 {n_signals} 次 ({n_signals / len(df) * 100:.4f}%)")

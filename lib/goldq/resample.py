"""
把 fetch_bars() 返回的 M1 数据重采样成任意周期（M5/M15/H1...）。

⚠️ 时刻铁律：resample(label="left", closed="left") 之后，每根 bar 的 time_utc 是该
bar 的**开始**时刻，但这根 bar 只有等到 time_utc + 周期长度 才算走完/可信。
比如 16:00 这根 M5 bar，实盘中要到 16:05 才能确认它的 OHLC——用这份数据做信号回测时，
"在 bar i 触发信号"实际执行时刻是 bar i 的 time_utc + 周期，不是 time_utc 本身。
"""

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    d = df.set_index("time_utc")
    out = d.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    out = out.dropna(subset=["open"])
    return out.reset_index()

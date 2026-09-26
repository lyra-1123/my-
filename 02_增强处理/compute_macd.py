"""
compute_macd.py
================
标准 MACD(12,26,9)。全部用 pandas ewm(adjust=False)（递归 EMA，和交易软件一致），
只用当前和过去的 close，不看未来——任何调用方都不需要再做额外的因果性检查。
"""

import pandas as pd

FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9


def compute_macd(
    close: pd.Series,
    fast: int = FAST_PERIOD,
    slow: int = SLOW_PERIOD,
    signal: int = SIGNAL_PERIOD,
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist}, index=close.index)

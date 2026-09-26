"""
compute_indicators.py
=======================
EMA / ATR / RSI，全部只用当前和过去的数据（因果安全），ATR/RSI 用 Wilder 平滑
（ewm(alpha=1/period, adjust=False)），跟交易软件默认一致。
"""

import pandas as pd


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_mfi(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Money Flow Index：成交量加权版的RSI。用简单滚动求和（不是Wilder平滑，是MFI的标准算法）。"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical_price * df["volume"]

    price_up = typical_price > typical_price.shift(1)
    positive_flow = raw_money_flow.where(price_up, 0.0)
    negative_flow = raw_money_flow.where(~price_up, 0.0)

    positive_sum = positive_flow.rolling(period).sum()
    negative_sum = negative_flow.rolling(period).sum()
    money_flow_ratio = positive_sum / negative_sum
    return 100 - (100 / (1 + money_flow_ratio))

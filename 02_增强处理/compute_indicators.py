"""
compute_indicators.py
=======================
EMA / ATR / RSI / ADX / MFI，全部只用当前和过去的数据（因果安全）。
ATR/RSI/ADX 用 Wilder 平滑（ewm(alpha=1/period, adjust=False)），与 TradingView 一致；
注意 MT5 自带 ATR 是 TR 的简单移动平均，数值会有出入。
"""

import pandas as pd


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14, max_gap_minutes: int | None = None) -> pd.Series:
    """
    max_gap_minutes: 给定时，与上一根bar间隔超过该分钟数（周末/假期/每日停盘）的bar，
    TR 只取当根 high-low，不把跳空算进去——否则停盘后第一根的跳空会让 ATR 虚高很久。
    None = 标准算法（包含跳空）。
    """
    prev_close = df["close"].shift(1)
    if max_gap_minutes is not None:
        gap = df["time_utc"].diff() > pd.Timedelta(minutes=max_gap_minutes)
        prev_close = prev_close.mask(gap)
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


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX（趋势强度），Wilder平滑，标准算法。用于第5章多Regime切分（趋势 vs 震荡）。"""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = compute_atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def compute_mfi(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Money Flow Index：成交量加权版的RSI。用简单滚动求和（不是Wilder平滑，是MFI的标准算法）。"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical_price * df["volume"]

    prev_tp = typical_price.shift(1)
    positive_flow = raw_money_flow.where(typical_price > prev_tp, 0.0)
    negative_flow = raw_money_flow.where(typical_price < prev_tp, 0.0)  # 持平两边都不计（标准MFI）

    positive_sum = positive_flow.rolling(period).sum()
    negative_sum = negative_flow.rolling(period).sum()
    money_flow_ratio = positive_sum / negative_sum
    return 100 - (100 / (1 + money_flow_ratio))

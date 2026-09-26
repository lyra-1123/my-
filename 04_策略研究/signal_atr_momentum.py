"""
signal_atr_momentum.py
========================
假设见 00_方案/hypothesis_atr_momentum.md。

信号（双向）：
  多头regime（close > EMA20）+ ATR扩张(>=1.5倍50根均值) + (MACD金叉 或 RSI12跌破30) -> signal=+1
  空头regime（close < EMA20）+ ATR扩张(>=1.5倍50根均值) + (MACD死叉 或 RSI12涨破70) -> signal=-1

全部只用当前及过去的bar，严格因果。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from compute_macd import compute_macd  # noqa: E402
from compute_indicators import compute_ema, compute_atr, compute_rsi  # noqa: E402

EMA_PERIOD = 20
ATR_PERIOD = 14
ATR_BASELINE_LOOKBACK = 50
ATR_EXPANSION_MULTIPLIER = 1.5
RSI_PERIOD = 12
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


def generate_signal(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    df_m1 = fetch_bars(symbol=symbol, start=start, end=end)
    df = resample_ohlcv(df_m1, rule="5min")

    ema20 = compute_ema(df["close"], EMA_PERIOD)
    atr = compute_atr(df, ATR_PERIOD)
    atr_baseline = atr.shift(1).rolling(ATR_BASELINE_LOOKBACK).mean()  # 不含当前bar
    atr_expanded = atr >= ATR_EXPANSION_MULTIPLIER * atr_baseline

    macd = compute_macd(df["close"])
    hist = macd["hist"]
    pre_hist = hist.shift(1)
    golden_cross = (pre_hist < 0) & (hist >= 0)
    dead_cross = (pre_hist >= 0) & (hist < 0)

    rsi = compute_rsi(df["close"], RSI_PERIOD)
    pre_rsi = rsi.shift(1)
    rsi_cross_below_30 = (pre_rsi >= RSI_OVERSOLD) & (rsi < RSI_OVERSOLD)
    rsi_cross_above_70 = (pre_rsi <= RSI_OVERBOUGHT) & (rsi > RSI_OVERBOUGHT)

    is_bull_regime = df["close"] > ema20
    is_bear_regime = df["close"] < ema20

    long_trigger = golden_cross | rsi_cross_below_30
    short_trigger = dead_cross | rsi_cross_above_70

    long_signal = is_bull_regime & atr_expanded & long_trigger
    short_signal = is_bear_regime & atr_expanded & short_trigger

    df["signal"] = 0
    df.loc[long_signal, "signal"] = 1
    df.loc[short_signal, "signal"] = -1

    return df


if __name__ == "__main__":
    df = generate_signal()
    n_long = int((df["signal"] == 1).sum())
    n_short = int((df["signal"] == -1).sum())
    print(f"[信号] M5 共 {len(df)} 根，做多信号 {n_long} 次，做空信号 {n_short} 次 "
          f"(合计 {(n_long+n_short)/len(df)*100:.4f}%)")
    print(df[df["signal"] != 0][["time_utc", "close", "signal"]].head(10).to_string(index=False))

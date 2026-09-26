"""
signal_atr_momentum_v2.py
============================
假设见 00_方案/hypothesis_atr_momentum_v2.md。相对假设4 v1 (signal_atr_momentum.py) 的改动：
RSI -> MFI，目标金额固定$3 -> 1倍当前ATR（target_series()）。

信号（双向）：
  多头regime（close > EMA20）+ ATR扩张(>=1.5倍50根均值) + (MACD金叉 或 MFI跌破20) -> signal=+1
  空头regime（close < EMA20）+ ATR扩张(>=1.5倍50根均值) + (MACD死叉 或 MFI涨破80) -> signal=-1

build_features() 只算一次指标，signal_from_features() 按参数出信号——第5章做参数邻域
（CSCV/PBO需要的"变体家族"）时不用重复加载和重采样600多万行M1数据。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from compute_macd import compute_macd  # noqa: E402
from compute_indicators import compute_ema, compute_atr, compute_mfi  # noqa: E402

EMA_PERIOD = 20
ATR_PERIOD = 14
ATR_BASELINE_LOOKBACK = 50
ATR_EXPANSION_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 1.0  # 反弹/下跌目标 = 这个倍数 * 信号bar当时的ATR
MFI_PERIOD = 12
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: M5 OHLCV。返回在 df 上追加了信号所需全部中间特征的副本。"""
    df = df.copy()
    df["ema20"] = compute_ema(df["close"], EMA_PERIOD)
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["atr_baseline"] = df["atr"].shift(1).rolling(ATR_BASELINE_LOOKBACK).mean()  # 不含当前bar

    hist = compute_macd(df["close"])["hist"]
    pre_hist = hist.shift(1)
    df["golden_cross"] = (pre_hist < 0) & (hist >= 0)
    df["dead_cross"] = (pre_hist >= 0) & (hist < 0)

    mfi = compute_mfi(df, MFI_PERIOD)
    pre_mfi = mfi.shift(1)
    df["mfi_cross_below"] = (pre_mfi >= MFI_OVERSOLD) & (mfi < MFI_OVERSOLD)
    df["mfi_cross_above"] = (pre_mfi <= MFI_OVERBOUGHT) & (mfi > MFI_OVERBOUGHT)
    return df


def signal_from_features(feat: pd.DataFrame,
                         atr_expansion_multiplier: float = ATR_EXPANSION_MULTIPLIER) -> pd.Series:
    atr_expanded = feat["atr"] >= atr_expansion_multiplier * feat["atr_baseline"]
    long_signal = (feat["close"] > feat["ema20"]) & atr_expanded & (feat["golden_cross"] | feat["mfi_cross_below"])
    short_signal = (feat["close"] < feat["ema20"]) & atr_expanded & (feat["dead_cross"] | feat["mfi_cross_above"])

    signal = pd.Series(0, index=feat.index, name="signal")
    signal[long_signal] = 1
    signal[short_signal] = -1
    return signal


def load_m5(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    return resample_ohlcv(fetch_bars(symbol=symbol, start=start, end=end), rule="5min")


def generate_signal(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    df = build_features(load_m5(symbol, start, end))
    df["signal"] = signal_from_features(df)
    return df


def target_series(df: pd.DataFrame, atr_target_multiplier: float = ATR_TARGET_MULTIPLIER) -> pd.Series:
    """反弹/下跌目标 = atr_target_multiplier 倍信号bar当时的ATR。"""
    return atr_target_multiplier * df["atr"]


if __name__ == "__main__":
    df = generate_signal()
    n_long = int((df["signal"] == 1).sum())
    n_short = int((df["signal"] == -1).sum())
    print(f"[信号] M5 共 {len(df)} 根，做多信号 {n_long} 次，做空信号 {n_short} 次 "
          f"(合计 {(n_long+n_short)/len(df)*100:.4f}%)")
    print(df[df["signal"] != 0][["time_utc", "close", "atr", "signal"]].head(10).to_string(index=False))

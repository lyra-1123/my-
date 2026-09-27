"""
signal_atr_mfi_reversal.py
=============================
假设见 00_方案/hypothesis_atr_mfi_reversal.md。

build_features() 算一次指标和"本次极值区期间ATR放大倍数的最大值"，
signal_from_features(feat, x) 按放大倍数门槛 x 出信号，第4/5章测多个 x 不用重算。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.exits import ExitRule  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from compute_indicators import compute_atr, compute_mfi  # noqa: E402

ATR_PERIOD = 14
ATR_GAP_MINUTES = 180
ATR_BASELINE_LOOKBACK = 50
MFI_PERIOD = 12
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
EXPANSION_GRID = [1.5, 2.0, 2.5, 3.0]
EXIT_RULE = ExitRule("trail1.5ATR_nolimit", stop="atr_trail", take_profit="none",
                     max_bars=None, atr_stop_mult=1.5)


def _episode_peak(in_zone: pd.Series, ratio: pd.Series) -> pd.Series:
    """每个连续 in_zone 区段内，截至当前bar的放大倍数最大值；区段外为 NaN。"""
    episode = (in_zone != in_zone.shift()).cumsum()
    return ratio.where(in_zone).groupby(episode).cummax().where(in_zone)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = compute_atr(df, ATR_PERIOD, max_gap_minutes=ATR_GAP_MINUTES)
    df["atr_ratio"] = df["atr"] / df["atr"].shift(1).rolling(ATR_BASELINE_LOOKBACK).mean()
    mfi = compute_mfi(df, MFI_PERIOD)
    df["mfi"] = mfi
    pre = mfi.shift(1)

    oversold, overbought = mfi < MFI_OVERSOLD, mfi > MFI_OVERBOUGHT
    df["long_trigger"] = (pre < MFI_OVERSOLD) & (mfi >= MFI_OVERSOLD)
    df["short_trigger"] = (pre > MFI_OVERBOUGHT) & (mfi <= MFI_OVERBOUGHT)
    # 离开那一根本身也算进"期间"：取 上一根为止的区段最大值 与 当根放大倍数 的较大者
    df["long_peak_ratio"] = pd.concat([_episode_peak(oversold, df["atr_ratio"]).shift(1),
                                       df["atr_ratio"]], axis=1).max(axis=1)
    df["short_peak_ratio"] = pd.concat([_episode_peak(overbought, df["atr_ratio"]).shift(1),
                                        df["atr_ratio"]], axis=1).max(axis=1)
    return df


def signal_from_features(feat: pd.DataFrame, expansion: float) -> pd.Series:
    signal = pd.Series(0, index=feat.index, name="signal")
    signal[feat["long_trigger"] & (feat["long_peak_ratio"] >= expansion)] = 1
    signal[feat["short_trigger"] & (feat["short_peak_ratio"] >= expansion)] = -1
    return signal


def load_m5(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    return resample_ohlcv(fetch_bars(symbol=symbol, start=start, end=end), rule="5min")


if __name__ == "__main__":
    feat = build_features(load_m5())
    for x in EXPANSION_GRID:
        s = signal_from_features(feat, x)
        print(f"x={x}: 做多 {int((s == 1).sum())} 次，做空 {int((s == -1).sum())} 次")

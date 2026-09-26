"""
signal_macd_underwater_cross.py
=================================
假设见 00_方案/hypothesis_macd_underwater_cross.md。

信号：M5 K线上，MACD(12,26,9) 水下金叉 + 金叉前绿柱已走窄。

严格因果：判断第 i 根 bar 是否触发信号，只使用第 i 根及更早的 HIST/DIF/DEA 值，
不使用任何未来数据。信号在第 i 根 bar 收盘后才能确认——对应实盘执行时刻是
i 的 time_utc + 5 分钟（见 lib/goldq/resample.py 顶部说明）。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_增强处理"))
from goldq.datastore import fetch_bars  # noqa: E402
from goldq.resample import resample_ohlcv  # noqa: E402
from compute_macd import compute_macd  # noqa: E402

NARROWING_LOOKBACK = 5  # v1 默认参数，走窄比较的回溯根数，后续可网格搜索


def generate_signal(df_m5: pd.DataFrame, narrowing_lookback: int = NARROWING_LOOKBACK) -> pd.Series:
    """
    df_m5: 必须有 close 列，index 按时间正序排列（fetch_bars/resample_ohlcv 的输出即可直接传入）。
    返回：与 df_m5 等长的 0/1 Series，1 = 信号在这根 bar 触发（bar 收盘时确认）。
    """
    macd = compute_macd(df_m5["close"])
    dif, dea, hist = macd["dif"], macd["dea"], macd["hist"]

    pre_hist = hist.shift(1)
    pre_dif = dif.shift(1)
    pre_dea = dea.shift(1)
    hist_n_bars_before_cross = hist.shift(1 + narrowing_lookback)

    is_golden_cross = (pre_hist < 0) & (hist >= 0)
    is_underwater = (pre_dif < 0) & (pre_dea < 0)
    is_narrowing = pre_hist > hist_n_bars_before_cross  # 交叉前一根比 N 根前更接近 0

    signal = (is_golden_cross & is_underwater & is_narrowing).astype(int)
    signal.name = "signal"
    return signal


def build_m5_with_signal(symbol: str = "XAUUSD", start=None, end=None) -> pd.DataFrame:
    df_m1 = fetch_bars(symbol=symbol, start=start, end=end)
    df_m5 = resample_ohlcv(df_m1, rule="5min")
    df_m5["signal"] = generate_signal(df_m5)
    return df_m5


if __name__ == "__main__":
    df = build_m5_with_signal()
    n_signals = int(df["signal"].sum())
    print(f"[信号] M5 共 {len(df)} 根，触发 {n_signals} 次 "
          f"({n_signals / len(df) * 100:.3f}%)")
    print(df[df["signal"] == 1][["time_utc", "close"]].head(10).to_string(index=False))

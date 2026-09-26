"""
MACD 金叉状态判定的共享逻辑，被 04_策略研究/signal_*.py 里所有基于
"某周期金叉后到现在还有效"这个概念的假设复用（假设2、假设3都用得到）。
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "02_增强处理"))
from compute_macd import compute_macd  # noqa: E402


def bullish_state_with_bars_since(hist: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    返回 (is_bullish, bars_in_state)：
    - is_bullish[i] = HIST[i] >= 0（当前处于金叉后，尚未死叉）
    - bars_in_state[i] = 这个状态已经持续了几根（0 = 状态刚开始，即金叉/死叉confirm的那一根）
    """
    is_bullish = hist >= 0
    state_change = is_bullish != is_bullish.shift(1).fillna(is_bullish.iloc[0])
    group_id = state_change.cumsum()
    bars_in_state = group_id.groupby(group_id).cumcount()
    return is_bullish, bars_in_state


def qualified_valid_state(df: pd.DataFrame, valid_bars: int, narrowing_lookback: int) -> pd.Series:
    """
    只有"水下金叉 + 走窄"触发的多头状态才算数，且必须在 valid_bars 根以内。
    （原假设1/假设2里给1H用的定义，这里泛化成可用于任意周期。）
    """
    macd = compute_macd(df["close"])
    dif, dea, hist = macd["dif"], macd["dea"], macd["hist"]

    pre_hist = hist.shift(1)
    pre_dif = dif.shift(1)
    pre_dea = dea.shift(1)
    hist_n_bars_before_cross = hist.shift(1 + narrowing_lookback)

    is_underwater = (pre_dif < 0) & (pre_dea < 0)
    is_narrowing = pre_hist > hist_n_bars_before_cross

    is_bullish, bars_in_state = bullish_state_with_bars_since(hist)

    state_change = is_bullish != is_bullish.shift(1).fillna(is_bullish.iloc[0])
    group_id = state_change.cumsum()
    qualified_start = (is_underwater & is_narrowing).groupby(group_id).transform("first")

    return is_bullish & qualified_start & (bars_in_state <= valid_bars)


def simple_valid_state(df: pd.DataFrame, valid_bars: int) -> pd.Series:
    """任意金叉都算数，只要求当前仍在有效期内、未死叉（假设2里给30min/15min用的定义）。"""
    macd = compute_macd(df["close"])
    is_bullish, bars_in_state = bullish_state_with_bars_since(macd["hist"])
    return is_bullish & (bars_in_state <= valid_bars)


def golden_cross_now(df: pd.DataFrame) -> pd.Series:
    """本周期这根bar本身刚发生金叉confirm（不要求水下/走窄），用作入场触发条件。"""
    macd = compute_macd(df["close"])
    pre_hist = macd["hist"].shift(1)
    return (pre_hist < 0) & (macd["hist"] >= 0)


def align_to_lower_tf(df_lower: pd.DataFrame, higher_tf_df: pd.DataFrame, valid_series: pd.Series,
                       period: pd.Timedelta, col_name: str) -> pd.DataFrame:
    """
    把高周期的 valid 状态严格因果地映射到低周期时间线上：
    只用"已经完全走完"的高周期K线（confirm_time = bar起始时间 + 周期长度）。
    """
    higher = pd.DataFrame({
        "confirm_time": higher_tf_df["time_utc"] + period,
        col_name: valid_series.values,
    }).sort_values("confirm_time")

    return pd.merge_asof(
        df_lower.sort_values("time_utc"),
        higher,
        left_on="time_utc",
        right_on="confirm_time",
        direction="backward",
    )

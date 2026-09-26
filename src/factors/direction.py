"""Directional (long/short) signals — distinct from library.py's regime-
classification factors (which only ever answer "is now safe to average
down", never "which way is price going"). Everything here is backward-
looking only (see library.py's look-ahead audit note in
reports/00_progress.md; the same rule applies here: rolling/shift(positive)
only, no future data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma_direction(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """+1 = fast MA above slow MA (long bias), -1 = below (short bias),
    0 = exactly equal (rare, treated as "no direction" downstream)."""
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    return np.sign(fast_ma - slow_ma)


def align_htf_direction(ltf_df: pd.DataFrame, htf_df: pd.DataFrame, htf_direction: pd.Series,
                         htf_bar_seconds: int) -> pd.Series:
    """Map a higher-timeframe direction series onto the lower-timeframe's
    bar index, WITHOUT look-ahead: an HTF bar starting at `time` only
    becomes known once it closes (time + htf_bar_seconds), so it's only
    valid for LTF bars at or after that close time — same convention as
    the H1->H4 alignment in Phase 2's build_htf_context."""
    ctx = pd.DataFrame({
        "valid_from": htf_df["time"] + pd.Timedelta(seconds=htf_bar_seconds),
        "htf_direction": htf_direction,
    }).dropna().sort_values("valid_from")
    merged = pd.merge_asof(
        ltf_df[["time"]].sort_values("time"), ctx,
        left_on="time", right_on="valid_from", direction="backward",
    )
    return merged["htf_direction"]


def cross_timeframe_signal(ltf_direction: pd.Series, htf_direction_aligned: pd.Series,
                            wait_bars: int = 4, reduced_size: float = 0.5) -> pd.DataFrame:
    """"Look at the big timeframe, trade the small one": full-size entry in
    the shared direction when LTF and HTF agree; if they disagree, stand
    down for `wait_bars` LTF bars, then defer to the HTF direction at
    `reduced_size` (rather than waiting indefinitely for the LTF to catch
    up — the HTF view is treated as authoritative once the grace period
    passes).
    """
    aligned = (ltf_direction == htf_direction_aligned) & (ltf_direction != 0)
    # length of the current run of consecutive misaligned bars, ending at
    # each row (resets to 0 the instant `aligned` is True)
    run_id = (aligned != aligned.shift()).cumsum()
    misalign_streak = (~aligned).groupby(run_id).cumcount() + 1
    misalign_streak = misalign_streak.where(~aligned, 0)

    deferred = (~aligned) & (misalign_streak >= wait_bars)

    direction = pd.Series(0.0, index=ltf_direction.index)
    size = pd.Series(0.0, index=ltf_direction.index)
    direction = direction.mask(aligned, ltf_direction)
    size = size.mask(aligned, 1.0)
    direction = direction.mask(deferred, htf_direction_aligned)
    size = size.mask(deferred, reduced_size)

    return pd.DataFrame({
        "aligned": aligned, "misalign_streak": misalign_streak, "deferred": deferred,
        "direction": direction, "size": size,
    })

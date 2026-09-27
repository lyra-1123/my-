"""
sessions.py
============
交易时段划分，全部按当地时间（随夏令时切换），与真实的流动性切换一致。

  亚洲盘        交易日开始（纽约 17:00）→ 伦敦 08:00
  伦敦上午      伦敦 08:00 → 纽约 08:00（COMEX 开盘 08:20、美国数据 08:30 之前）
  伦敦纽约重叠  纽约 08:00 → 伦敦 16:00
  纽约下午      伦敦 16:00 → 交易日结束（纽约 17:00）
"""

import numpy as np
import pandas as pd

SESSIONS = ["亚洲盘", "伦敦上午", "伦敦纽约重叠", "纽约下午"]


def _utc(time_utc: pd.Series) -> pd.Series:
    t = pd.to_datetime(time_utc)
    return t.dt.tz_localize("UTC") if t.dt.tz is None else t


def local_time(time_utc: pd.Series, tz: str) -> pd.Series:
    return _utc(time_utc).dt.tz_convert(tz)


def london_time(time_utc: pd.Series) -> pd.Series:
    return local_time(time_utc, "Europe/London")


def ny_time(time_utc: pd.Series) -> pd.Series:
    return local_time(time_utc, "America/New_York")


def _minutes(t: pd.Series) -> np.ndarray:
    return (t.dt.hour * 60 + t.dt.minute).to_numpy()


def session_label(time_utc: pd.Series) -> pd.Series:
    """按 bar 开始时间归入四个时段。"""
    ldn, ny = _minutes(london_time(time_utc)), _minutes(ny_time(time_utc))
    after_ny_close = ny >= 17 * 60
    label = np.select(
        [after_ny_close | (ldn < 8 * 60),
         ny < 8 * 60,
         ldn < 16 * 60],
        [SESSIONS[0], SESSIONS[1], SESSIONS[2]],
        default=SESSIONS[3],
    )
    return pd.Series(label, index=time_utc.index)

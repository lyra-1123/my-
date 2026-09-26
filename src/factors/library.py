"""Candidate factors for XAUUSD regime classification.

For a martingale strategy the relevant question is not "which direction will
price go" but "is the market currently choppy/mean-reverting (safe to average
down into) or trending (a martingale ladder can run away and blow up)". Every
factor here is computed from information available at time t only (no
look-ahead); labels in `labels.py` look forward and are for evaluation only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (10, 20, 50, 100)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _log_return(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"]).diff()


# ---------- volatility ----------

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return _true_range(df).rolling(n).mean()


def realized_vol(df: pd.DataFrame, n: int) -> pd.Series:
    return _log_return(df).rolling(n).std()


def bollinger_width(df: pd.DataFrame, n: int, k: float = 2.0) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    return (2 * k * sd) / ma


def keltner_width(df: pd.DataFrame, n: int, mult: float = 2.0) -> pd.Series:
    ema = df["close"].ewm(span=n, adjust=False).mean()
    return (2 * mult * atr(df, n)) / ema


def vol_of_vol(df: pd.DataFrame, n: int) -> pd.Series:
    return realized_vol(df, n).rolling(n).std()


# ---------- trend strength / persistence ----------

def adx(df: pd.DataFrame, n: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(df)
    atr_n = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(n).mean()


def adx_slope(df: pd.DataFrame, n: int) -> pd.Series:
    a = adx(df, n)
    return a - a.shift(max(n // 4, 1))


def ma_slope(df: pd.DataFrame, n: int) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    return (ma - ma.shift(n)) / (n * df["close"])


def efficiency_ratio(df: pd.DataFrame, n: int) -> pd.Series:
    """Kaufman's Efficiency Ratio over the trailing n bars: net move / path
    length. Near 1 = trending, near 0 = choppy. Usable both as a backward
    factor and (shifted forward) as the regime label — see labels.py."""
    net = (df["close"] - df["close"].shift(n)).abs()
    path = df["close"].diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


def variance_ratio_2(df: pd.DataFrame, n: int) -> pd.Series:
    """Lo-MacKinlay 2-period variance ratio, estimated over trailing n bars.
    >1 = trending/positively autocorrelated, <1 = mean-reverting."""
    r1 = _log_return(df)
    r2 = np.log(df["close"]).diff(2)
    var_r1 = r1.rolling(n).var()
    var_r2 = r2.rolling(n).var()
    return var_r2 / (2 * var_r1.replace(0, np.nan))


def autocorr_returns(df: pd.DataFrame, n: int) -> pd.Series:
    """Rolling lag-1 autocorrelation of returns over trailing n bars.
    Positive = momentum/trending, negative = mean-reverting."""
    r = _log_return(df)
    return r.rolling(n).corr(r.shift(1))


def streak_length(df: pd.DataFrame) -> pd.Series:
    """Number of consecutive bars (including current) moving in the same
    direction as the current bar."""
    sign = np.sign(df["close"].diff())
    streak_id = (sign != sign.shift()).cumsum()
    return sign.groupby(streak_id).cumcount() + 1


# ---------- oscillators / mean-reversion ----------

def rsi(df: pd.DataFrame, n: int) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def zscore_vs_ma(df: pd.DataFrame, n: int) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    return (df["close"] - ma) / sd.replace(0, np.nan)


def stochastic_k(df: pd.DataFrame, n: int) -> pd.Series:
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)


def stochastic_d(df: pd.DataFrame, n: int) -> pd.Series:
    return stochastic_k(df, n).rolling(3).mean()


def williams_r(df: pd.DataFrame, n: int) -> pd.Series:
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def cci(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    mad = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def roc(df: pd.DataFrame, n: int) -> pd.Series:
    return df["close"].pct_change(n)


def dist_from_high(df: pd.DataFrame, n: int) -> pd.Series:
    hh = df["high"].rolling(n).max()
    return (df["close"] - hh) / hh


def dist_from_low(df: pd.DataFrame, n: int) -> pd.Series:
    ll = df["low"].rolling(n).min()
    return (df["close"] - ll) / ll


def donchian_position(df: pd.DataFrame, n: int) -> pd.Series:
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return (df["close"] - ll) / (hh - ll).replace(0, np.nan)


def skew_returns(df: pd.DataFrame, n: int) -> pd.Series:
    return _log_return(df).rolling(n).skew()


def kurt_returns(df: pd.DataFrame, n: int) -> pd.Series:
    return _log_return(df).rolling(n).kurt()


def mfi(df: pd.DataFrame, n: int) -> pd.Series:
    """Money Flow Index: volume-weighted RSI. Our `volume` column is a
    tick-count proxy, not real traded volume (see reports/00_progress.md),
    so treat this as a weaker signal than a true-volume MFI would be."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_flow = tp * df["volume"]
    direction = np.sign(tp.diff())
    pos_flow = raw_flow.where(direction > 0, 0.0).rolling(n).sum()
    neg_flow = raw_flow.where(direction < 0, 0.0).rolling(n).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


def choppiness_index(df: pd.DataFrame, n: int) -> pd.Series:
    """Purpose-built for exactly our question: 100 near choppy/range-bound,
    0 near a strong sustained trend. tr_sum uses SUM of true range (not mean
    like ATR) over the window, per the standard CHOP formula."""
    tr_sum = _true_range(df).rolling(n).sum()
    hi_lo_range = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    return 100 * np.log10(tr_sum / hi_lo_range.replace(0, np.nan)) / np.log10(n)


def aroon_up(df: pd.DataFrame, n: int) -> pd.Series:
    return df["high"].rolling(n + 1).apply(lambda x: 100 * x.argmax() / n, raw=True)


def aroon_down(df: pd.DataFrame, n: int) -> pd.Series:
    return df["low"].rolling(n + 1).apply(lambda x: 100 * x.argmin() / n, raw=True)


def parkinson_vol(df: pd.DataFrame, n: int) -> pd.Series:
    """Parkinson (1980) high-low range volatility estimator: more efficient
    than close-to-close realized_vol since it uses the whole bar's range."""
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)))


def garman_klass_vol(df: pd.DataFrame, n: int) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility estimator: uses open/close as
    well as the range, captures overnight-gap variance too."""
    hl = 0.5 * np.log(df["high"] / df["low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
    return np.sqrt((hl - co).rolling(n).mean())


def linreg_r2(df: pd.DataFrame, n: int) -> pd.Series:
    """R^2 of a linear (time-trend) fit to close over the trailing n bars:
    how well a straight line explains the path, direction-agnostic — high
    = persistent trend of either sign, low = choppy/directionless."""
    idx = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    return df["close"].rolling(n).corr(idx) ** 2


def avg_gap(df: pd.DataFrame, n: int) -> pd.Series:
    """Average bar-to-bar opening gap (|open_t - close_{t-1}|) over the
    trailing n bars — a microstructure/discontinuity measure distinct from
    ATR (which is dominated by intrabar range, not inter-bar jumps)."""
    gap = (df["open"] - df["close"].shift(1)).abs() / df["close"].shift(1)
    return gap.rolling(n).mean()


def macd_histogram(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / df["close"]


# ---------- time ----------

def session_bucket(df: pd.DataFrame) -> pd.Series:
    """Coarse UTC trading-session label. NOT independently verified against a
    live UTC feed (see reports/00_progress.md) — treat session-based findings
    as provisional until that's checked."""
    hour = df["time"].dt.hour
    bins = [-1, 7, 13, 16, 21, 24]
    labels = ["asia", "london_open", "ny_overlap", "ny_only", "late"]
    return pd.cut(hour, bins=bins, labels=labels).astype(str)


def day_of_week(df: pd.DataFrame) -> pd.Series:
    return df["time"].dt.dayofweek


# windowed indicator -> function, applied for every n in WINDOWS
WINDOWED_FACTORS = {
    "atr": atr,
    "realized_vol": realized_vol,
    "bb_width": bollinger_width,
    "keltner_width": keltner_width,
    "vol_of_vol": vol_of_vol,
    "adx": adx,
    "adx_slope": adx_slope,
    "ma_slope": ma_slope,
    "efficiency_ratio": efficiency_ratio,
    "variance_ratio_2": variance_ratio_2,
    "autocorr_returns": autocorr_returns,
    "rsi": rsi,
    "zscore_vs_ma": zscore_vs_ma,
    "stochastic_k": stochastic_k,
    "stochastic_d": stochastic_d,
    "williams_r": williams_r,
    "cci": cci,
    "roc": roc,
    "dist_from_high": dist_from_high,
    "dist_from_low": dist_from_low,
    "donchian_position": donchian_position,
    "skew_returns": skew_returns,
    "kurt_returns": kurt_returns,
    "mfi": mfi,
    "choppiness_index": choppiness_index,
    "aroon_up": aroon_up,
    "aroon_down": aroon_down,
    "parkinson_vol": parkinson_vol,
    "garman_klass_vol": garman_klass_vol,
    "linreg_r2": linreg_r2,
    "avg_gap": avg_gap,
}

# Gold's price went from ~$860 to ~$5300+ over this sample, so any raw
# dollar-denominated (or otherwise price-scale-dependent) factor level means
# something different in 2009 vs 2026. We percentile-rank EVERY windowed
# factor against its own trailing PCTRANK_WINDOWS-bar history, regardless of
# whether it looks already-bounded (RSI, ADX etc. are conventionally 0-100,
# but their *typical* range still drifts across volatility regimes) — v3
# only did this for a hand-picked subset; v4 checks all of them rather than
# assuming which ones need it.
PCTRANK_WINDOWS = (500, 2000)  # ~3 weeks and ~12 weeks of H1 bars


def build_factor_table(df: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """df: OHLCV with a `time` column (e.g. H1 bars). Returns a DataFrame of
    factor columns aligned to df's index, all backward-looking only."""
    cols = {}
    for name, fn in WINDOWED_FACTORS.items():
        for n in windows:
            cols[f"{name}_{n}"] = fn(df, n)

    for name in WINDOWED_FACTORS:
        for n in windows:
            base = cols[f"{name}_{n}"]
            for pw in PCTRANK_WINDOWS:
                cols[f"{name}_{n}_pctrank{pw}"] = base.rolling(pw).rank(pct=True)

    cols["macd_hist"] = macd_histogram(df)
    cols["streak_length"] = streak_length(df)
    cols["hour"] = df["time"].dt.hour
    cols["session"] = session_bucket(df)
    cols["day_of_week"] = day_of_week(df)
    cols["time"] = df["time"]
    return pd.DataFrame(cols, index=df.index)

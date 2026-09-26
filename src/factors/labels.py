"""Forward-looking regime labels, used ONLY for evaluating factors — never
as model inputs (they use future data by construction)."""
from __future__ import annotations

from .library import efficiency_ratio


def forward_efficiency_ratio(df, horizon: int):
    """ER computed over the window [t, t+horizon]; near 1 = strong trend
    that persisted (dangerous for a martingale ladder), near 0 = choppy/
    mean-reverting (the regime a martingale grid is designed for)."""
    return efficiency_ratio(df, horizon).shift(-horizon)

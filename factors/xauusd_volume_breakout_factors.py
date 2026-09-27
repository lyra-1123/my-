# -*- coding: utf-8 -*-
"""
XAUUSD 放量突破 · 短期动量延续因子库
=====================================

输入数据：仅使用 open / high / low / close / volume 五个字段（DatetimeIndex，升序）。
所有因子在 t 根 K 线收盘时刻可计算，只用到 <= t 的信息；交易在 t+1 根 K 线开盘执行。

包含 3 个不同角度的因子：
    1. VolConfirmedBreakout   量能确认的通道突破因子   —— "价格位置"角度（突破了什么）
    2. VolWeightedCloseThrust 放量收盘强度推力因子     —— "K 线内部微观结构"角度（谁在主导）
    3. SqueezeReleaseMomentum 波动压缩后放量释放因子   —— "波动率状态切换"角度（何时突破最可信）

统一后处理：滚动 MAD 去极值 + 滚动 Z-Score 标准化（单品种时间序列，只能用滚动窗口，
不能用全样本统计量，否则就是未来函数）。

交易成本：0.01 手（=1 盎司）点差 0.2 美元。按"每单位仓位变化支付半个点差"计，
即一次完整开平仓（round trip）= 0.2 美元 / 0.01 手。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 频率参数预设（窗口以"根 K 线"为单位）
#    说明：不同频率的噪声水平、成交量季节性差异很大，统一一套参数会严重失真。
#    这些是"合理起点"而非优化结果，实盘前请做样本外 / walk-forward 验证。
# ---------------------------------------------------------------------------
FREQ_PRESETS = {
    #         通道   ATR  量能基准  推力窗口  压缩短/长  标准化窗口  是否日内
    "1D":    dict(chan=20, atr=14, vol_base=20, thrust=5,  sq_short=5,  sq_long=60,  norm=250,  intraday=False),
    "4H":    dict(chan=30, atr=14, vol_base=20, thrust=6,  sq_short=6,  sq_long=90,  norm=500,  intraday=True),
    "1H":    dict(chan=24, atr=24, vol_base=20, thrust=6,  sq_short=8,  sq_long=120, norm=750,  intraday=True),
    "30MIN": dict(chan=32, atr=32, vol_base=20, thrust=8,  sq_short=8,  sq_long=160, norm=1000, intraday=True),
    "15MIN": dict(chan=32, atr=32, vol_base=20, thrust=8,  sq_short=12, sq_long=192, norm=1500, intraday=True),
    "5MIN":  dict(chan=36, atr=48, vol_base=20, thrust=12, sq_short=12, sq_long=288, norm=2000, intraday=True),
}

EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. 通用工具函数
# ---------------------------------------------------------------------------
def _check_input(df: pd.DataFrame) -> pd.DataFrame:
    """校验字段：只允许 OHLCV，杜绝使用数据字典之外的字段。"""
    need = ["open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"缺少字段: {miss}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index 必须是 DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index 必须按时间升序")
    return df[need].astype(float)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder ATR。TR 用到前一根收盘价（t-1），t 时刻已知，无未来函数。"""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def relative_volume(df: pd.DataFrame, n_days: int, intraday: bool) -> pd.Series:
    """
    相对成交量 = 当前成交量 / 历史"同类"K 线成交量的中位数。

    - 日内频率：黄金成交量有极强的日内季节性（亚盘清淡、伦敦开盘/美盘开盘/数据公布放量）。
      若直接和前 N 根 K 线比，伦敦开盘那根 K 线永远"放量"，因子会退化成时段哑变量。
      因此按"同一时刻(time-of-day)"分组，用该时刻过去 n_days 天的成交量中位数作为基准。
    - 日频：直接用前 n_days 根日线的中位数。
    - 基准一律 shift(1)，不包含当前 K 线本身。
    """
    vol = df["volume"]
    if intraday:
        tod = df.index.hour * 60 + df.index.minute
        base = vol.groupby(tod).transform(
            lambda s: s.shift(1).rolling(n_days, min_periods=max(5, n_days // 4)).median()
        )
    else:
        base = vol.shift(1).rolling(n_days, min_periods=max(5, n_days // 4)).median()
    return vol / (base + EPS)


def rolling_mad_zscore(x: pd.Series, window: int, n_mad: float = 3.0) -> pd.Series:
    """
    滚动 MAD 去极值 + 滚动 Z-Score 标准化（纯时间序列版本，无未来函数）。

    步骤：
      1) med_t = 过去 window 根（含当前）的中位数
      2) MAD_t = 过去 window 根 |x - med| 的中位数（近似：|x_i - med_i| 使用各自时点的中位数，
         避免 O(N*W) 内存；对于平稳的因子序列与精确 MAD 差异极小）
      3) 截断到 [med - n_mad*1.4826*MAD, med + n_mad*1.4826*MAD]
      4) 对截断后的序列做滚动均值/标准差 Z-Score
    注意：横截面 MAD 在单品种上不适用；全样本 MAD 会引入未来信息。
    """
    minp = max(30, window // 4)
    med = x.rolling(window, min_periods=minp).median()
    mad = (x - med).abs().rolling(window, min_periods=minp).median()
    scale = 1.4826 * mad
    clipped = x.clip(lower=med - n_mad * scale, upper=med + n_mad * scale)
    mu = clipped.rolling(window, min_periods=minp).mean()
    sd = clipped.rolling(window, min_periods=minp).std()
    z = (clipped - mu) / (sd + EPS)
    # 标准差为 0（因子长期为 0 的稀疏段）时置 0，避免除零放大
    return z.where(sd > EPS, 0.0)


# ---------------------------------------------------------------------------
# 2. 因子一：量能确认的通道突破因子  VolConfirmedBreakout
# ---------------------------------------------------------------------------
def factor_vol_confirmed_breakout(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    """
    【逻辑】
      收盘价突破过去 N 根 K 线的最高价/最低价，意味着市场接受了一个新的价格区间。
      但"无量突破"往往是流动性稀薄时段的扫止损行为（stop hunt），随后回落；
      "放量突破"说明有真实的新增资金/信息驱动（知情交易者入场），
      加上突破后追涨的趋势跟踪资金、被套方止损盘的连锁反应 → 短期动量延续。

    【算子】
      hh = TS_MAX(high, N).shift(1)        # 前 N 根最高价（不含当前）
      ll = TS_MIN(low,  N).shift(1)
      brk = (close - hh)/ATR  if close > hh
            (close - ll)/ATR  if close < ll
            0                 otherwise     # 以 ATR 衡量的突破幅度（有方向）
      vol_conf = max(log(RelVol), 0)         # 只有高于常态的成交量才给予确认权重
      raw = EWM(brk * vol_conf, span=k)      # 突破信号在随后几根 K 线内衰减保留
      factor = MAD_ZSCORE(raw)

    【风险点】
      - XAUUSD 为场外市场，volume 通常是经纪商的 tick volume，而非真实成交量；
        不同经纪商数据差异大，换数据源须重新验证。
      - 非农/CPI/FOMC 等数据公布时"放量突破"极多且常见来回扫（whipsaw），
        因子在事件时点可能出现大幅反向亏损 —— 建议实盘叠加事件时间过滤。
      - 信号稀疏：多数时间为 0，Z-Score 后分布呈尖峰厚尾，阈值不要设太低。
      - 风格暴露：本质是趋势跟踪（CTA 动量）暴露，在震荡市会持续小额回撤。
    """
    p = {**FREQ_PRESETS[freq], **kw}
    d = _check_input(df)
    a = atr(d, p["atr"])
    hh = d["high"].shift(1).rolling(p["chan"], min_periods=p["chan"]).max()
    ll = d["low"].shift(1).rolling(p["chan"], min_periods=p["chan"]).min()

    up = (d["close"] - hh).clip(lower=0)
    dn = (d["close"] - ll).clip(upper=0)
    brk = (up + dn) / (a + EPS)             # 突破幅度，单位：ATR

    rv = relative_volume(d, p["vol_base"], p["intraday"])
    vol_conf = np.log(rv.clip(lower=EPS)).clip(lower=0)   # 缩量突破权重为 0

    raw = (brk * vol_conf).ewm(span=max(2, p["thrust"] // 2), adjust=False).mean()
    return rolling_mad_zscore(raw, p["norm"]).rename("VolConfirmedBreakout")


# ---------------------------------------------------------------------------
# 3. 因子二：放量收盘强度推力因子  VolWeightedCloseThrust
# ---------------------------------------------------------------------------
def factor_vol_weighted_close_thrust(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    """
    【逻辑】
      K 线收盘价在全天振幅中的位置（CLV）反映收盘时刻多空谁占上风：
      收在最高附近 = 买方一直主导到收盘，卖方无力反扑。
      若这种"强收盘"同时伴随放量和振幅扩张，说明是主动性的大单推动而非噪声，
      此类订单流具有持续性（机构拆单执行、信息逐步扩散 —— Kyle 模型中的知情交易者
      会把订单分散在多个时段），因此近 M 根 K 线的"量加权收盘强度"对下一段收益有预测力。

    【算子】
      CLV   = ((C-L) - (H-C)) / (H-L)            ∈[-1,1] 收盘位置
      BODY  = (C-O) / (H-L)                      ∈[-1,1] 有方向的实体占比（排除长影线的假强势）
      s     = 0.5*(CLV + BODY)                   单根 K 线的方向强度
      RX    = (H-L) / ATR.shift(1)               振幅扩张倍数（与前一根的 ATR 比，避免自包含）
      w     = RelVol                             量能权重
      raw   = TS_SUM(s*RX*w, M) / TS_SUM(w, M)   量加权平均推力
      factor = MAD_ZSCORE(raw)

    【风险点】
      - 高频（5MIN）下单根 K 线的 CLV 噪声大，点差 0.2 相对 5 分钟振幅不可忽略，
        该因子在 5MIN 上换手高、净收益最容易被成本吞噬。
      - 日线的"最高/最低/收盘"依赖经纪商的日切时间（纽约 17:00 或 GMT 0:00），
        不同日切得到的 CLV 不同 —— 存在数据定义敏感性。
      - 与因子一有一定相关性（强势突破 K 线通常也是强收盘），组合时注意去相关/正交化。
      - 过拟合风险：CLV 与 BODY 的等权组合是先验设定，不要在样本内网格搜索权重。
    """
    p = {**FREQ_PRESETS[freq], **kw}
    d = _check_input(df)
    rng = (d["high"] - d["low"])
    rng_safe = rng.where(rng > EPS)                        # 一字线（H==L）视为无信息

    clv = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / rng_safe
    body = (d["close"] - d["open"]) / rng_safe
    s = (0.5 * (clv + body)).fillna(0.0)

    rx = (rng / (atr(d, p["atr"]).shift(1) + EPS)).clip(upper=5.0)  # 限制极端振幅的放大作用
    w = relative_volume(d, p["vol_base"], p["intraday"]).clip(upper=10.0)

    m = p["thrust"]
    num = (s * rx * w).rolling(m, min_periods=m).sum()
    den = w.rolling(m, min_periods=m).sum()
    raw = num / (den + EPS)
    return rolling_mad_zscore(raw, p["norm"]).rename("VolWeightedCloseThrust")


# ---------------------------------------------------------------------------
# 4. 因子三：波动压缩后放量释放因子  SqueezeReleaseMomentum
# ---------------------------------------------------------------------------
def factor_squeeze_release_momentum(df: pd.DataFrame, freq: str = "1H", **kw) -> pd.Series:
    """
    【逻辑】
      波动率聚集（volatility clustering）：长时间低波动盘整期间，市场参与者的止损单、
      突破挂单在区间两端不断堆积，同时新信息被"压抑"未定价。
      一旦出现放量方向性移动，堆积的挂单被连锁触发，外加注意力驱动的跟风交易
      （行为金融中的注意力效应 / 羊群效应），导致低波动→高波动的状态切换并伴随趋势延续。
      相比因子一，本因子强调"突破之前的环境"：同样的放量突破，发生在压缩之后可信度更高。

    【算子】
      ratio     = ATR_short / ATR_long                     短期波动相对长期波动
      squeeze   = max(-log(TS_MIN(ratio, K).shift(1)), 0)  过去 K 根内出现过的最强压缩程度（不含当前）
      disp      = (C - C.shift(K)) / (ATR_long.shift(1)*sqrt(K))   近 K 根的标准化位移（方向）
      surge     = TS_MEAN(max(log(RelVol),0), 3)           最近 3 根的放量程度
      raw       = squeeze * disp * surge
      factor    = MAD_ZSCORE(raw)

    【风险点】
      - 三项相乘使因子高度非线性且稀疏，对各窗口参数敏感 —— 过拟合风险最高的一个，
        必须做参数平原（parameter plateau）检验：邻近参数的表现应平滑，不应呈孤峰。
      - 假突破（failed breakout）在压缩后同样常见；本因子不包含止损逻辑，实盘需配合
        基于 ATR 的硬止损。
      - 节假日/亚盘低流动性时段天然"低波动"，会被误判为压缩；日内频率下 relative_volume
        已做时段校正，但 ATR 比值没有，建议在 5MIN/15MIN 上额外排除亚盘信号。
      - 风格暴露：波动率突破（long gamma 性质），在均值回归主导的行情中持续失效。
    """
    p = {**FREQ_PRESETS[freq], **kw}
    d = _check_input(df)
    k = p["sq_short"]
    atr_s = atr(d, k)
    atr_l = atr(d, p["sq_long"])

    ratio = atr_s / (atr_l + EPS)
    squeeze = (-np.log(ratio.clip(lower=EPS))).rolling(k, min_periods=k).max().shift(1).clip(lower=0)

    disp = (d["close"] - d["close"].shift(k)) / (atr_l.shift(1) * np.sqrt(k) + EPS)

    rv = relative_volume(d, p["vol_base"], p["intraday"])
    surge = np.log(rv.clip(lower=EPS)).clip(lower=0).rolling(3, min_periods=3).mean()

    raw = squeeze * disp * surge
    return rolling_mad_zscore(raw, p["norm"]).rename("SqueezeReleaseMomentum")


def compute_all_factors(df: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
    """一次性计算 3 个因子。"""
    return pd.concat(
        [factor_vol_confirmed_breakout(df, freq),
         factor_vol_weighted_close_thrust(df, freq),
         factor_squeeze_release_momentum(df, freq)],
        axis=1,
    )


# ---------------------------------------------------------------------------
# 5. 评估：IC + 含成本的简易回测
# ---------------------------------------------------------------------------
def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    【仅用于评估的标签，禁止作为因子输入】
    t 时刻收盘出信号 → t+1 开盘入场 → t+1+horizon 开盘出场 的对数收益。
    用开盘价而不是收盘价作为成交价，避免"用收盘价出信号又按收盘价成交"的隐性未来函数。
    """
    o = df["open"]
    return np.log(o.shift(-(1 + horizon)) / o.shift(-1))


def rank_ic(factor: pd.Series, fwd: pd.Series) -> float:
    """时间序列 Spearman Rank IC。"""
    x = pd.concat([factor, fwd], axis=1).dropna()
    if len(x) < 30:
        return np.nan
    return float(x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank()))


def backtest_with_cost(
    df: pd.DataFrame,
    factor: pd.Series,
    entry: float = 1.5,
    exit_: float = 0.3,
    spread: float = 0.2,
    lots: float = 0.01,
) -> dict:
    """
    含点差成本的迟滞（hysteresis）信号回测，结果以美元计。

    - |z| > entry 开仓（方向 = sign(z)）；|z| < exit_ 平仓；介于两者之间维持原仓位。
      迟滞区间是控制换手、对抗 0.2 点差最有效的手段。
    - 信号在 t 收盘产生，t+1 开盘成交：pos.shift(1)。
    - 每根 K 线盈亏 = pos * (open_{t+1} - open_t) * 盎司数，0.01 手 = 1 盎司。
    - 成本 = |Δpos| * spread/2 * 盎司数（一次完整开平 = 一个点差 0.2 美元/0.01 手）。
    """
    oz = lots * 100.0
    z = factor.reindex(df.index)
    raw_pos = pd.Series(np.nan, index=df.index)
    raw_pos[z > entry] = 1.0
    raw_pos[z < -entry] = -1.0
    raw_pos[z.abs() < exit_] = 0.0
    raw_pos = raw_pos.ffill().fillna(0.0)

    pos = raw_pos.shift(1).fillna(0.0)                     # t+1 开盘执行
    bar_pnl = pos * (df["open"].shift(-1) - df["open"]) * oz
    cost = pos.diff().abs().fillna(pos.abs()) * (spread / 2.0) * oz
    net = (bar_pnl - cost).fillna(0.0)

    n_round_trips = pos.diff().abs().sum() / 2.0
    gross = bar_pnl.sum()
    total_cost = cost.sum()
    return {
        "gross_usd": round(float(gross), 2),
        "cost_usd": round(float(total_cost), 2),
        "net_usd": round(float(net.sum()), 2),
        "round_trips": int(round(n_round_trips)),
        "gross_per_trip_usd": round(float(gross / max(n_round_trips, 1)), 3),
        "time_in_market": round(float((pos != 0).mean()), 3),
        "max_drawdown_usd": round(float((net.cumsum() - net.cumsum().cummax()).min()), 2),
    }


def cost_hurdle_report(df: pd.DataFrame, freq: str, spread: float = 0.2) -> float:
    """
    成本门槛：点差 / 中位 ATR。该值越大，说明该频率越难覆盖成本。
    经验上 > 0.1 时，需要较高的单笔期望（大 entry 阈值、长持有）才能覆盖成本。
    """
    a = atr(_check_input(df), FREQ_PRESETS[freq]["atr"])
    return float(spread / a.median())


# ---------------------------------------------------------------------------
# 6. 未来函数自检：截断数据后，历史因子值必须完全不变
# ---------------------------------------------------------------------------
def check_no_lookahead(df: pd.DataFrame, freq: str, cut: int | None = None) -> bool:
    cut = cut or int(len(df) * 0.7)
    full = compute_all_factors(df, freq)
    part = compute_all_factors(df.iloc[:cut], freq)
    diff = (full.iloc[:cut] - part).abs().max().max()
    return bool(np.nan_to_num(diff) < 1e-9)


# ---------------------------------------------------------------------------
# 7. Demo：用合成数据跑通流程（真实研究请替换为你的 XAUUSD OHLCV 数据）
# ---------------------------------------------------------------------------
def _make_synthetic(n: int = 20000, freq: str = "15min", seed: int = 7) -> pd.DataFrame:
    """合成带日内成交量季节性的 15 分钟 K 线，仅用于验证代码可运行，不代表真实表现。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    hour = idx.hour.values
    season = np.where((hour >= 7) & (hour < 11), 2.0, 1.0) * np.where((hour >= 13) & (hour < 17), 2.5, 1.0)
    vol_shock = rng.lognormal(0, 0.5, n)
    sigma = 0.0008 * np.sqrt(season) * np.sqrt(vol_shock)
    ret = rng.standard_normal(n) * sigma
    close = 2000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.standard_normal((2, n))) * sigma * close * 0.5
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = (1000 * season * vol_shock).round()
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


if __name__ == "__main__":
    FREQ = "15MIN"
    data = _make_synthetic()

    print("未来函数自检通过:", check_no_lookahead(data, FREQ))
    print(f"成本门槛 spread/ATR = {cost_hurdle_report(data, FREQ):.3f}")

    fac = compute_all_factors(data, FREQ)
    print("\n因子描述统计:\n", fac.describe().T.round(3))
    print("\n因子相关性:\n", fac.corr().round(3))

    for h in (1, 4, 16):
        fwd = forward_return(data, h)
        ics = {c: round(rank_ic(fac[c], fwd), 4) for c in fac}
        print(f"\nRankIC (h={h}):", ics)

    print("\n含成本回测（0.01 手，点差 0.2）:")
    for c in fac:
        print(c, backtest_with_cost(data, fac[c]))

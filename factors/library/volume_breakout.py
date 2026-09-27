# -*- coding: utf-8 -*-
"""
家族：放量 + 突破 → 短期动量延续（第一批入库因子，2026-09-27）。
真实数据评估结论见 FACTOR_LIBRARY.md。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import EPS, atr, check_input, params, relative_volume, rolling_mad_zscore
from ..registry import register


# ---------------------------------------------------------------------------
# 因子一：量能确认的通道突破因子  VolConfirmedBreakout
# ---------------------------------------------------------------------------
@register(
    name='VolConfirmedBreakout',
    cn_name='量能确认的通道突破',
    family='breakout',
    hypothesis='放量突破 N 根高/低点代表新信息与新资金入场，止损盘与趋势跟随资金推动动量延续；缩量突破视为扫止损',
    formula='MAD_Z(EWM(brk_ATR * max(log RelVol,0)))，brk_ATR=(C-TS_MAX(H,N).shift1)/ATR 或 (C-TS_MIN(L,N).shift1)/ATR',
    risks=['tick volume 数据源差异', '数据公布时点 whipsaw', '信号稀疏厚尾', '趋势跟踪风格暴露'],
    added="2026-09-27",
)
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
    p = params(freq, **kw)
    d = check_input(df)
    a = atr(d, p["atr"])
    hh = d["high"].shift(1).rolling(p["chan"], min_periods=p["chan"]).max()
    ll = d["low"].shift(1).rolling(p["chan"], min_periods=p["chan"]).min()

    up = (d["close"] - hh).clip(lower=0)
    dn = (d["close"] - ll).clip(upper=0)
    brk = (up + dn) / (a + EPS)             # 突破幅度，单位：ATR

    rv = relative_volume(d, p["vol_base"], p["intraday"])
    vol_conf = np.log(rv.clip(lower=EPS)).clip(lower=0)   # 缩量突破权重为 0

    raw = (brk * vol_conf).ewm(span=max(2, p["thrust"] // 2), adjust=False).mean()
    return rolling_mad_zscore(raw, p["norm"])


# ---------------------------------------------------------------------------
# 因子二：放量收盘强度推力因子  VolWeightedCloseThrust
# ---------------------------------------------------------------------------
@register(
    name='VolWeightedCloseThrust',
    cn_name='放量收盘强度推力',
    family='microstructure',
    hypothesis='强收盘(CLV)+实体占比+振幅扩张且放量，代表主动大单拆单执行（Kyle），订单流有持续性',
    formula='MAD_Z(TS_SUM(0.5*(CLV+BODY)*RX*RelVol,M)/TS_SUM(RelVol,M))',
    risks=['高频成本吞噬', '日切定义敏感', '与突破因子相关', 'CLV/BODY 权重先验'],
    added="2026-09-27",
)
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
    p = params(freq, **kw)
    d = check_input(df)
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
    return rolling_mad_zscore(raw, p["norm"])


# ---------------------------------------------------------------------------
# 因子三：波动压缩后放量释放因子  SqueezeReleaseMomentum
# ---------------------------------------------------------------------------
@register(
    name='SqueezeReleaseMomentum',
    cn_name='波动压缩后放量释放',
    family='volatility',
    hypothesis='低波动盘整堆积止损/突破挂单，放量方向性移动触发连锁反应与注意力跟风，波动率状态切换伴随趋势',
    formula='MAD_Z(max(-log TS_MIN(ATR_s/ATR_l,K).shift1,0) * ΔC_K/(ATR_l*sqrt K) * TS_MEAN(max(log RelVol,0),3))',
    risks=['三项乘积高度非线性、参数敏感', '假突破无止损', '亚盘天然低波动被误判为压缩', 'long-gamma 风格暴露'],
    added="2026-09-27",
)
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
    p = params(freq, **kw)
    d = check_input(df)
    k = p["sq_short"]
    atr_s = atr(d, k)
    atr_l = atr(d, p["sq_long"])

    ratio = atr_s / (atr_l + EPS)
    squeeze = (-np.log(ratio.clip(lower=EPS))).rolling(k, min_periods=k).max().shift(1).clip(lower=0)

    disp = (d["close"] - d["close"].shift(k)) / (atr_l.shift(1) * np.sqrt(k) + EPS)

    rv = relative_volume(d, p["vol_base"], p["intraday"])
    surge = np.log(rv.clip(lower=EPS)).clip(lower=0).rolling(3, min_periods=3).mean()

    raw = squeeze * disp * surge
    return rolling_mad_zscore(raw, p["norm"])

# -*- coding: utf-8 -*-
"""
模拟盘策略规格（冻结）。

每个策略一份 StrategySpec。上线后规格与行为都被冻结：
  - 行为指纹：在固定历史区间（FINGERPRINT_WINDOW）上的信号值与目标仓位的哈希，登记时写入 state/<id>/prereg.json；
    之后每次运行都会重算并比较。挖掘新因子时若改动了公共代码并影响到在跑策略，运行会报错。
  - 需要修改在跑策略时：不要改原规格，新增一个版本（id 后缀 -v2），旧版本 status 改为 "retired"。
新增策略：在 SPECS 里追加一项（status="candidate"），运行 `python -m paper.onboard <id>` 做准入检查并登记。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from factors.core import rolling_mad_zscore
from factors.registry import get_factors

FINGERPRINT_WINDOW = ("2024-01-01", "2024-12-31")


@dataclass(frozen=True)
class StrategySpec:
    id: str                                   # 唯一标识，含版本号，如 "TT30-EW-v1"
    description: str
    freq: str                                 # 5MIN / 15MIN / 30MIN / 1H / 4H / 1D
    components: tuple                         # ((因子名, 权重, {参数覆盖}), ...)，权重符号即方向
    norm: int                                 # 组合后滚动 MAD+Z 的窗口（单成分且权重=1 时不再标准化）
    entry: float = 1.5
    exit: float = 0.3
    lots: float = 0.01                        # 0.01 手 = 1 盎司
    forward_start: str = ""                   # 前向测试起始日（UTC 日期），此前为回测
    status: str = "candidate"                 # candidate / active / retired
    notes: str = ""
    evidence: dict = field(default_factory=dict, hash=False, compare=False)

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """组合信号（>0 看多），只用 <= t 的数据。"""
        parts = []
        for name, w, kw in self.components:
            f = get_factors([name])[0].func(df, self.freq, **dict(kw))
            parts.append(w * f.fillna(0.0))
        raw = sum(parts)
        if len(self.components) == 1 and abs(self.components[0][1]) == 1:
            return raw
        return rolling_mad_zscore(raw, self.norm)


SPECS: list[StrategySpec] = [
    StrategySpec(
        id="TT30-EW-v1",
        description="30MIN 趋势尾部：放量趋势效率 + VWAP 偏离 等权组合，迟滞开平仓，换日前平仓",
        freq="30MIN",
        components=(("TrendEfficiencyVolume", 0.5, (("chan", 32),)),
                    ("VWAPDeviation", 0.5, (("chan", 32),))),
        norm=1000, entry=1.5, exit=0.3, lots=0.01,
        forward_start="2026-09-28",
        status="active",
        notes="参数为入库先验值，未调参；见 reports/trend_tail_validation.md、reports/overfitting_tests.md",
        evidence={"oos_sharpe_atr_2020_": 1.04, "cscv_oos_sharpe_median": 0.41, "research_PBO": 0.034,
                  "DSR_Neff16": 0.69, "param_PBO": 0.61},
    ),
]


SPECS.append(
    StrategySpec(
        id="HA1H-v1",
        description="1H 52 周高点锚定动量：价格在已收盘日线 250 日高低区间中的位置，迟滞开平仓，换日前平仓",
        freq="1H",
        components=(("HighAnchorMomentum", 1.0, (("anchor_days", 250),)),),
        norm=1000, entry=1.5, exit=0.3, lots=0.01,
        forward_start="2026-09-28",
        status="candidate",
        notes="第八批；参数为先验值；见 reports/batch8_candidate_validation.txt、reports/trend_continuation_batch8.md",
        evidence={"oos_sharpe_atr_2020_": 0.59, "is_sharpe_atr": 0.62, "cscv_oos_sharpe_median": 0.61,
                  "param_PBO": 0.111, "research_PBO": 0.027, "DSR_Neff20": 0.279},
    )
)


def get_spec(sid: str) -> StrategySpec:
    for s in SPECS:
        if s.id == sid:
            return s
    raise KeyError(f"未知策略 {sid}；已有：{[s.id for s in SPECS]}")

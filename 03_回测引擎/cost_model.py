"""
cost_model.py
==============
交易成本模型（指南 3.5 节 P0-8，本项目尚未做真实成本标定，这里是 v1 占位实现）。

⚠️ spread_usd / slippage_usd 是通用placeholder，不是任何具体broker的真实点差——
上实盘前必须用第7.2节"Broker选择5维评估"里选定的broker真实点差数据替换这两个默认值，
否则回测成本会失真。
"""

from dataclasses import dataclass


@dataclass
class SimpleCostModel:
    spread_usd: float = 0.30       # 点差（一次往返只算一次，买卖价差已经隐含在成交价里）
    slippage_usd: float = 0.05     # 单边滑点，开仓平仓各算一次

    def round_trip_cost(self) -> float:
        return self.spread_usd + 2 * self.slippage_usd

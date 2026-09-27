"""
cost_model.py
==============
交易成本模型（指南 3.5 节 P0-8，本项目尚未做真实成本标定，这里是 v1 占位实现）。

spread_usd = 0.20：2026-09-27 用户给出的实际点差。
⚠️ slippage_usd 仍是占位值，上实盘前用第7.2节选定broker的实测滑点替换。
"""

from dataclasses import dataclass


@dataclass
class SimpleCostModel:
    spread_usd: float = 0.20       # 点差（一次往返只算一次，买卖价差已经隐含在成交价里）
    slippage_usd: float = 0.05     # 单边滑点，开仓平仓各算一次

    def round_trip_cost(self) -> float:
        return self.spread_usd + 2 * self.slippage_usd

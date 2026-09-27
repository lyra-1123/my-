# -*- coding: utf-8 -*-
"""
因子注册表。每个因子用 @register(...) 登记元数据，评估器和因子库文档都从这里读取。

新增因子只需：在 factors/library/ 下写函数 + 加装饰器 + 在 library/__init__.py 导入模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class FactorSpec:
    name: str                    # 因子名（英文驼峰，直观）
    cn_name: str                 # 中文名
    family: str                  # 角度/家族：breakout / microstructure / volatility / trend / reversal / session
    hypothesis: str              # 一句话逻辑（经济学/行为金融依据）
    formula: str                 # 算子表达式
    risks: list[str]             # 风险点
    func: Callable[..., pd.Series]
    freqs: tuple[str, ...] = ("5MIN", "15MIN", "30MIN", "1H", "4H", "1D")  # 适用频率
    added: str = ""              # 入库日期
    tags: list[str] = field(default_factory=list)

    def __call__(self, df: pd.DataFrame, freq: str, **kw) -> pd.Series:
        return self.func(df, freq, **kw).rename(self.name)


REGISTRY: dict[str, FactorSpec] = {}


def register(**meta):
    """装饰器：登记因子。函数签名必须为 f(df, freq, **kw) -> pd.Series（已做 MAD+Z 标准化，>0 看多）。"""
    def deco(func):
        spec = FactorSpec(func=func, **meta)
        if spec.name in REGISTRY:
            raise ValueError(f"因子重名: {spec.name}")
        REGISTRY[spec.name] = spec
        return func
    return deco


def get_factors(names: list[str] | None = None) -> list[FactorSpec]:
    from . import library  # noqa: F401  触发注册
    if not names:
        return list(REGISTRY.values())
    miss = [n for n in names if n not in REGISTRY]
    if miss:
        raise KeyError(f"未注册的因子: {miss}；已注册: {list(REGISTRY)}")
    return [REGISTRY[n] for n in names]

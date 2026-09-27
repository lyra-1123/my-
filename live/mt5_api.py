# -*- coding: utf-8 -*-
"""MT5 接口：优先使用官方 MetaTrader5 包；设置环境变量 XAU_MT5_MOCK=1 时使用 live/mock_mt5.py（仅用于测试）。"""
import os

if os.environ.get("XAU_MT5_MOCK") == "1":
    from live import mock_mt5 as mt5  # noqa: F401
else:
    import MetaTrader5 as mt5  # noqa: F401

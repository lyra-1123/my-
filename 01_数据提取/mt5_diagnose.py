#!/usr/bin/env python3
"""
mt5_diagnose.py
================
诊断脚本：MT5的copy_rates_range持续报Invalid params时，
先用不依赖日期参数的copy_rates_from_pos测API本身是否正常，缩小问题范围。

运行环境：本地机器，需登录好MT5终端，pip install MetaTrader5 pandas
"""

import datetime

import MetaTrader5 as mt5

print(f"[信息] MetaTrader5包版本: {mt5.__version__}")

if not mt5.initialize():
    print(f"[错误] initialize失败: {mt5.last_error()}")
    raise SystemExit

print(f"[信息] terminal_info: {mt5.terminal_info()}")
print(f"[信息] account_info: {mt5.account_info()}")

SYMBOL = "XAUUSD"

selected = mt5.symbol_select(SYMBOL, True)
print(f"\n[信息] symbol_select结果: {selected}")

print(f"\n[测试1] copy_rates_from_pos(最新100根M1)")
rates1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 100)
if rates1 is None:
    print(f"  失败: {mt5.last_error()}")
else:
    print(f"  成功，拿到 {len(rates1)} 条，最后一条: {rates1[-1]}")

print(f"\n[测试2] copy_rates_from(当前时间往前100根)")
rates2 = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_M1, datetime.datetime.now(), 100)
if rates2 is None:
    print(f"  失败: {mt5.last_error()}")
else:
    print(f"  成功，拿到 {len(rates2)} 条")

print(f"\n[测试3] copy_rates_range(最近1天, naive datetime)")
end = datetime.datetime.now()
start = end - datetime.timedelta(days=1)
rates3 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start, end)
if rates3 is None:
    print(f"  失败: {mt5.last_error()}")
else:
    print(f"  成功，拿到 {len(rates3)} 条")

mt5.shutdown()

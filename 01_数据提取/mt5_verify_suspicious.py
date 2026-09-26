#!/usr/bin/env python3
"""
mt5_verify_suspicious.py
==========================
验证之前探测到的"1条"数据到底是不是真实K线（而不是broker占位记录）。
同时逐日探测某个窗口，确认真实数据开始的确切日期。

运行环境：本地机器，需登录好MT5终端，pip install MetaTrader5 pandas
"""

import calendar
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSD"

if not mt5.initialize():
    raise RuntimeError(f"MT5初始化失败: {mt5.last_error()}")

# 抽查几个可疑月份，看看返回的到底是什么时间、什么价格
test_months = [
    (2025, 12), (2020, 1), (2016, 1), (2009, 1), (1996, 10),
]

for year, month in test_months:
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1)
    end = datetime(year, month, last_day, 23, 59, 59)

    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start, end)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        print(f"[请求 {year}-{month:02d}] 实际返回的那些数据:")
        print(df.to_string())
        print()

print("\n" + "=" * 50)
print("逐日探测某个窗口，确认真实数据开始的确切日期:")
d = datetime(2026, 5, 20)
while d < datetime(2026, 6, 10):
    start = d
    end = d + timedelta(days=1)
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start, end)
    n = len(rates) if rates is not None else 0
    print(f"  {d.strftime('%Y-%m-%d')}: {n} 条")
    d += timedelta(days=1)

mt5.shutdown()

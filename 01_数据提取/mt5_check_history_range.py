#!/usr/bin/env python3
"""
mt5_check_history_range.py
============================
探测MT5服务器上某品种M1数据实际保存的历史深度。

按月倒推探测，用 terminal_info().maxbars 而不是单次 copy_rates_range 能拉的最大条数
（早期版本单次拉全量2000-2026年会超限报错 (-2, 'Invalid params')）。
连续3个月无数据就停止往前探测，判定为已到服务器实际保存历史的起点。

运行环境：本地机器，需登录好MT5终端，pip install MetaTrader5 pandas
"""

import calendar
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSD"
MISS_STREAK_LIMIT = 3
MAX_MONTHS_TO_PROBE = 360  # 最多往前探测30年


def main() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5初始化失败: {mt5.last_error()}")

    print(f"[信息] 品种: {SYMBOL}")
    print(f"[信息] terminal maxbars: {mt5.terminal_info().maxbars}\n")

    now = datetime.now()
    year, month = now.year, now.month

    earliest_found = None
    miss_streak = 0
    monthly_counts = []

    for _ in range(MAX_MONTHS_TO_PROBE):
        last_day = calendar.monthrange(year, month)[1]
        start = datetime(year, month, 1)
        end = datetime(year, month, last_day, 23, 59, 59)

        rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start, end)

        if rates is not None and len(rates) > 0:
            n = len(rates)
            print(f"[{year}-{month:02d}] 有数据: {n} 条")
            monthly_counts.append((year, month, n))
            earliest_found = (year, month)
            miss_streak = 0
        else:
            err = mt5.last_error()
            print(f"[{year}-{month:02d}] 无数据 (err={err})")
            miss_streak += 1
            if miss_streak >= MISS_STREAK_LIMIT:
                print(f"\n连续{miss_streak}个月无数据，停止往前探测。")
                break

        month -= 1
        if month == 0:
            month = 12
            year -= 1

    print(f"\n{'=' * 50}")
    if earliest_found:
        ey, em = earliest_found
        print(f"[结论] 服务器实际保存的M1历史最早到: {ey}年{em}月")
        print(f"共探测到 {len(monthly_counts)} 个有数据的月份。")
        print("窗口之外的历史深度需要用 dukascopy_export_historical.py 补充。")
    else:
        print("[结论] 没有探测到任何数据，请检查SYMBOL名称或MT5连接状态。")

    mt5.shutdown()


if __name__ == "__main__":
    main()

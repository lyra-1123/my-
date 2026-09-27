# -*- coding: utf-8 -*-
"""
MT5 模拟盘实时程序配置（在你的 Windows 电脑上按实际情况修改）。
"""

# ---------------- 连接 ----------------
SYMBOL = "XAUUSD"               # MT5 里的品种名；有些经纪商带后缀（如 "XAUUSD." / "XAUUSDm"），以"市场报价"窗口为准
MT5_PATH = None                 # 需要指定终端路径时填，例如 r"C:\Program Files\MetaTrader 5\terminal64.exe"

# 服务器时间 → UTC 的换算方式（先运行 python -m live.check_mt5 核对）：
#   "ny+7"      服务器时间 = 纽约时间 + 7 小时（冬令时 GMT+2、夏令时 GMT+3，多数经纪商如此）
#   "fixed:+2"  固定偏移（小时），例如服务器常年 GMT+2
SERVER_TZ = "ny+7"

# ---------------- 策略 ----------------
# id 必须与 paper/specs.py 一致；lots 为每个策略的手数（0.01 手 = 1 盎司，先用 check_mt5 核对合约大小）
STRATEGIES = {
    "TT30-EW-v1":      {"enabled": True, "magic": 3001, "lots": 0.01},   # 日内30min强势跟随
    "HA1H-v1":         {"enabled": True, "magic": 3002, "lots": 0.01},   # 年内高低位顺势
    "HA1H-TS2-shadow": {"enabled": True, "magic": 3003, "lots": 0.01},   # 影子：移动止损版
    "MF30-EW-shadow":  {"enabled": True, "magic": 3004, "lots": 0.01},   # 影子：30min多逻辑组合
}

# 每个频率从 MT5 取多少根 K 线做计算（需要在 MT5"工具→选项→图表→图表最大K线数"里设为足够大，例如"无限"）
HISTORY_BARS = {"30MIN": 30000, "1H": 20000}

# ---------------- 执行与风控 ----------------
BAR_CLOSE_DELAY_SEC = 8         # K 线收盘后等待几秒再取数据（确保最后一根已完整）
DEVIATION_POINTS = 30           # 市价单允许的最大滑点（点）
MAX_SPREAD_USD = 0.60           # 点差超过该值时不开新仓（平仓不受限）
MAX_DATA_AGE_MIN = 35           # 最新完整 K 线距现在超过该分钟数时不交易（数据中断保护）
DEMO_ONLY = True                # 只允许在模拟账户上运行
KILL_FILE = "live/STOP"         # 存在此文件时：平掉本程序的全部仓位并暂停
DAILY_LOSS_LIMIT_USD = None     # 组合熔断：当日（纽约 17:00 起）本程序全部策略的已实现 + 浮动亏损超过该值时平仓并暂停到下一交易日；None 为关闭
ALIGN_ON_START = False          # 启动时是否立即按目标仓位对齐（False：等到下一根 K 线收盘再操作，与模型成交时点一致）
LOG_DIR = "live/logs"

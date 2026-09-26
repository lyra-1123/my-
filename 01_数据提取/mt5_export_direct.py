#!/usr/bin/env python3
"""
mt5_export_direct.py
=====================
最终兜底方案：MT5 服务器上 XAUUSD 的 M1 历史是"滚动窗口"（实测约100天），
这段窗口内的数据可以直接从 MT5 全量导出（OHLC + tick_volume 都是真实的），
不需要再和 Dukascopy 合并。

⚠️ 时区铁律：MT5 返回的 time 是 broker 服务器时间，不是 UTC。
本脚本用 07_配置参数/data_paths.yaml 里的 mt5_utc_offset_hours 做校准——
上线前必须先用 mt5_fill_volume.py 的 verify_overlap() 或 mt5_verify_suspicious.py
核对真实偏移量，禁止假设默认值 0 是对的（详见 99_结果/traps_log.md）。

运行环境：本地 Windows 机器，需要装好 MT5 客户端并登录，pip install MetaTrader5 pandas pyyaml
"""

import calendar
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from goldq.config import load_data_paths  # noqa: E402

SYMBOL = "XAUUSD"


def init_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5初始化失败: {mt5.last_error()}")


def detect_earliest_available(symbol: str) -> tuple[datetime, datetime]:
    probe_start = datetime(2000, 1, 1)
    probe_end = datetime.now()
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, probe_start, probe_end)
    if rates is None or len(rates) == 0:
        raise RuntimeError("探测失败，没有拉到任何数据，请检查MT5连接和maxbars设置")
    df = pd.DataFrame(rates)
    earliest = pd.to_datetime(df["time"].min(), unit="s")
    latest = pd.to_datetime(df["time"].max(), unit="s")
    print(f"[探测] 当前服务器真实历史窗口: {earliest} ~ {latest} (共{len(df)}条)")
    return earliest.to_pydatetime(), latest.to_pydatetime()


def export_month(symbol: str, year: int, month: int, utc_offset_hours: int) -> pd.DataFrame:
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1)
    end = datetime(year, month, last_day, 23, 59, 59)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s") - timedelta(hours=utc_offset_hours)
    return df


def save_as_dukascopy_format(df: pd.DataFrame, out_path: Path) -> None:
    out = pd.DataFrame()
    out["datetime_str"] = df["time"].dt.strftime("%Y%m%d %H%M%S")
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    out["volume"] = df["tick_volume"]  # real_volume 外汇/贵金属常年为0，用tick_volume
    out.to_csv(out_path, sep=";", header=False, index=False)
    print(f"[输出] {out_path} ({len(out)}行)")


def main() -> None:
    cfg = load_data_paths()
    output_dir = Path(cfg["raw_dir"])
    utc_offset_hours = cfg.get("mt5_utc_offset_hours", 0)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_mt5()

    earliest, latest = detect_earliest_available(SYMBOL)
    print(f"\n即将导出真实数据窗口: {earliest.date()} ~ {latest.date()}")
    print(f"（窗口之外的月份不会导出，请继续使用Dukascopy文件）")
    print(f"（UTC 校准偏移 = {utc_offset_hours} 小时，来自 07_配置参数/data_paths.yaml）\n")

    year, month = earliest.year, earliest.month
    end_year, end_month = latest.year, latest.month

    while (year, month) <= (end_year, end_month):
        df = export_month(SYMBOL, year, month, utc_offset_hours)
        if not df.empty:
            fname = f"DAT_ASCII_XAUUSD_M1_{year}{month:02d}_MT5REAL.csv"
            out_path = output_dir / fname
            save_as_dukascopy_format(df, out_path)
        else:
            print(f"[跳过] {year}-{month:02d} 无数据")

        month += 1
        if month == 13:
            month = 1
            year += 1

    print(f"\n完成。合并导出到: {output_dir}")
    print("文件名带 _MT5REAL 后缀，volume列是真实tick_volume。")
    print("可以直接替换掉对应月份原有的Dukascopy文件（哪些月份重叠自行判断）。")

    mt5.shutdown()


if __name__ == "__main__":
    main()

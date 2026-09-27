# -*- coding: utf-8 -*-
"""
从本机 MT5 终端导出券商自己的报价数据（研究用；独立脚本，不引用、不修改 live/ 的任何代码）。

导出两类数据：
  1) M1 K 线 + 每分钟点差（copy_rates_range 的 spread 字段，单位"点"，按 symbol 的 point 换算为美元）
       → <out>/BROKER_<品种>_M1_<年份>.csv   列：time_utc;open;high;low;close;tick_volume;spread_usd
     券商保留的 M1 历史通常较长（数年）。价格为 BID。
  2) 逐笔 tick（copy_ticks_range，带 bid/ask）
       → <out>/BROKER_<品种>_TICKS_<年月>.csv 列：time_utc_ms;bid;ask
     券商服务器保留的 tick 历史有限（常见几个月），脚本从 --end 往回逐日取，连续 10 个工作日取不到就停止。

前提：MT5 终端已登录（与实盘程序同一个终端即可，运行期间不影响实盘程序），pip install MetaTrader5 pandas。
时间：MT5 返回"服务器时间"。多数黄金经纪商的服务器时间 = 纽约时间 + 7 小时（与 live/config.py 的 SERVER_TZ="ny+7" 相同）；
      若你的券商不同，用 --server-tz fixed:2 这类写法（服务器时间 = UTC + 2 小时，不随夏令时变化）。

用法（Windows 命令行，在脚本所在目录）：
  python export_broker_data.py                                  # 默认：XAUUSD，M1 从 2009 年起能取多少取多少，tick 取最近 2 年内能取到的
  python export_broker_data.py --symbol XAUUSD --what m1        # 只导出 M1 + 点差
  python export_broker_data.py --what ticks --tick-days 30      # 只导出最近 30 天 tick（先试小范围）
  python export_broker_data.py --path "C:\\Program Files\\MetaTrader 5\\terminal64.exe"   # 需要指定终端时
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("缺少 MetaTrader5 包：pip install MetaTrader5")


def server_to_utc(idx: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    if mode == "ny+7":
        ny = (idx - pd.Timedelta(hours=7)).tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        return ny.tz_convert("UTC").tz_localize(None)
    if mode.startswith("fixed:"):
        return idx - pd.Timedelta(hours=float(mode.split(":")[1]))
    raise ValueError(f"未知 --server-tz：{mode}")


def srv_arg(ts: pd.Timestamp):
    # MetaTrader5 包把 datetime 当作"秒数"使用；传带 UTC 时区的对象，避免被本机时区再换算一次（与 live/mt5_data.py 相同做法）
    return pd.Timestamp(ts).tz_localize("UTC").to_pydatetime()


def export_m1(sym: str, start: pd.Timestamp, end: pd.Timestamp, out: str, tz: str, point: float) -> None:
    print(f"[M1] {sym} {start.date()} ~ {end.date()}（服务器时间），按月请求……")
    frames = []
    a = start
    while a < end:
        b = min(a + pd.DateOffset(months=1), end)
        r = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, srv_arg(a), srv_arg(b))
        if r is not None and len(r):
            frames.append(pd.DataFrame(r))
        a = b
    if not frames:
        print(f"  没有取到 M1 数据：{mt5.last_error()}（可在 MT5 选项→图表 把'图表中最大K线数'调到'无限'后重试）")
        return
    df = pd.concat(frames).drop_duplicates("time").sort_values("time")
    idx = server_to_utc(pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s")), tz)
    df = df.assign(t=idx).dropna(subset=["t"])
    df["spread_usd"] = df["spread"] * point
    for y, g in df.groupby(df["t"].dt.year):
        path = os.path.join(out, f"BROKER_{sym}_M1_{y}.csv")
        with open(path, "w", encoding="ascii") as fh:
            fh.write("time_utc;open;high;low;close;tick_volume;spread_usd\n")
            for row in g.itertuples(index=False):
                fh.write(f"{row.t:%Y%m%d %H%M%S};{row.open:.3f};{row.high:.3f};{row.low:.3f};{row.close:.3f};"
                         f"{int(row.tick_volume)};{row.spread_usd:.3f}\n")
        print(f"  写出 {path}：{len(g)} 行，点差中位数 {g['spread_usd'].median():.3f}$")
    print(f"  M1 覆盖 {df['t'].min()} ~ {df['t'].max()}（UTC）")


def export_ticks(sym: str, end: pd.Timestamp, days: int, out: str, tz: str) -> None:
    print(f"[TICK] {sym} 从 {end.date()} 往回最多 {days} 天，逐日请求……")
    by_month: dict[str, list] = {}
    misses, got = 0, 0
    day = end.normalize()
    for _ in range(days):
        day -= pd.Timedelta(days=1)
        if day.weekday() == 5:
            continue
        t = mt5.copy_ticks_range(sym, srv_arg(day), srv_arg(day + pd.Timedelta(days=1)), mt5.COPY_TICKS_ALL)
        if t is None or len(t) == 0:
            if day.weekday() < 5:
                misses += 1
                if misses >= 10:
                    print(f"  连续 10 个工作日取不到 tick，停止（券商 tick 历史大概到 {(day + pd.Timedelta(days=14)).date()} 为止）")
                    break
            continue
        misses = 0
        d = pd.DataFrame(t)[["time_msc", "bid", "ask"]]
        d = d[(d["bid"] > 0) & (d["ask"] > 0)]
        utc = server_to_utc(pd.DatetimeIndex(pd.to_datetime(d["time_msc"], unit="ms")), tz)
        d = d.assign(ms=utc.astype("datetime64[ms]").asi8).dropna()          # 与 pandas 的内部精度无关
        by_month.setdefault(f"{day:%Y%m}", []).append(d)
        got += 1
        if got % 20 == 0:
            print(f"  已取 {got} 天（{day.date()}），当日 {len(d)} 笔", flush=True)
    for ym, parts in sorted(by_month.items()):
        d = pd.concat(parts).sort_values("ms")
        path = os.path.join(out, f"BROKER_{sym}_TICKS_{ym}.csv")
        with open(path, "w", encoding="ascii") as fh:
            fh.write("time_utc_ms;bid;ask\n")
            for ms, b, a in zip(d["ms"].to_numpy(), d["bid"].to_numpy(), d["ask"].to_numpy()):
                fh.write(f"{int(ms)};{b:.3f};{a:.3f}\n")
        sp = (d["ask"] - d["bid"])
        print(f"  写出 {path}：{len(d)} 笔，点差中位数 {sp.median():.3f}$（95% 分位 {sp.quantile(0.95):.3f}$）")
    if not by_month:
        print(f"  没有取到任何 tick：{mt5.last_error()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="导出券商 M1（含点差）与 tick（bid/ask）")
    ap.add_argument("--symbol", default="XAUUSD", help="MT5 里的品种名（有后缀的券商如 XAUUSD. / XAUUSDm）")
    ap.add_argument("--what", default="both", choices=["both", "m1", "ticks"])
    ap.add_argument("--m1-start", default="2009-01-01")
    ap.add_argument("--tick-days", type=int, default=730, help="tick 往回最多取多少天")
    ap.add_argument("--out", default="broker_data")
    ap.add_argument("--server-tz", default="ny+7")
    ap.add_argument("--path", default=None, help="MT5 终端 terminal64.exe 路径（可选）")
    a = ap.parse_args()
    if not (mt5.initialize(path=a.path) if a.path else mt5.initialize()):
        sys.exit(f"MT5 初始化失败：{mt5.last_error()}（确认终端已打开并登录）")
    try:
        if not mt5.symbol_select(a.symbol, True):
            sys.exit(f"无法选中品种 {a.symbol}：{mt5.last_error()}（检查'市场报价'里的品种名）")
        info = mt5.symbol_info(a.symbol)
        os.makedirs(a.out, exist_ok=True)
        now_srv = pd.Timestamp(mt5.symbol_info_tick(a.symbol).time, unit="s") + pd.Timedelta(days=1)
        print(f"品种 {a.symbol}：point={info.point}，当前点差 {info.spread * info.point:.3f}$，服务器时间约 {now_srv - pd.Timedelta(days=1)}")
        if a.what in ("both", "m1"):
            export_m1(a.symbol, pd.Timestamp(a.m1_start), now_srv, a.out, a.server_tz, info.point)
        if a.what in ("both", "ticks"):
            export_ticks(a.symbol, now_srv, a.tick_days, a.out, a.server_tz)
    finally:
        mt5.shutdown()
    print(f"\n完成。把 {os.path.abspath(a.out)} 里的 CSV 传到 Drive 的「dukascopy导出」文件夹。")


if __name__ == "__main__":
    main()

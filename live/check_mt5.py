# -*- coding: utf-8 -*-
"""
上线前第 1 步：python -m live.check_mt5 [--dukascopy-dir 路径]
检查：连接与账户（模拟？对冲？）、品种合约（0.01 手是否 = 1 盎司）、服务器时区偏移、历史深度；
可选：与 Dukascopy 的 30MIN 收益做滞后相关，确认时区对齐（最佳滞后应为 0）。
"""
import argparse
import time

import numpy as np
import pandas as pd

from live import config as C
from live.mt5_api import mt5
from live.mt5_data import connect, fetch_bars, server_to_utc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dukascopy-dir", default=None)
    args = ap.parse_args()
    connect()
    acc, info, tick = mt5.account_info(), mt5.symbol_info(C.SYMBOL), mt5.symbol_info_tick(C.SYMBOL)
    print("== 账户")
    print(f"  登录 {acc.login}，服务器 {acc.server}，模拟账户：{acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO}，"
          f"对冲模式：{acc.margin_mode == mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING}，余额 {acc.balance} {acc.currency}")
    print("== 品种", C.SYMBOL)
    print(f"  合约大小 {info.trade_contract_size}（0.01 手 = {info.trade_contract_size * 0.01:g} 盎司），小数位 {info.digits}，"
          f"最小手数 {info.volume_min}，步长 {info.volume_step}，止损最小距离 {info.trade_stops_level} 点，成交方式 {info.filling_mode}")
    print(f"  当前点差 {tick.ask - tick.bid:.3f}（配置的上限 {C.MAX_SPREAD_USD}）")
    srv = pd.Timestamp(tick.time, unit="s"); utc = pd.Timestamp(time.time(), unit="s")
    off = (srv - utc).total_seconds() / 3600
    pred = (srv - server_to_utc(pd.DatetimeIndex([srv]))[0]).total_seconds() / 3600
    print("== 时区")
    fresh = abs(off - round(off)) < 0.1 and abs(off) < 14   # 报价在几分钟内 → 偏移接近整数小时
    verdict = ("一致" if abs(round(off) - pred) < 0.5 else "不一致，请修改 SERVER_TZ") if fresh else \
        "最新报价不是实时的（休市/断线），无法据此判断，以下面的滞后相关表为准或开盘后再跑"
    print(f"  最新报价服务器时间 {srv}，本机 UTC {utc}，实测偏移约 {off:+.2f} 小时；"
          f"按 SERVER_TZ='{C.SERVER_TZ}' 推算的偏移 {pred:+.0f} 小时 → {verdict}")
    print("== 历史深度（需要：30MIN ≥ 6000 根，1H ≥ 7000 根）")
    for fq in ("30MIN", "1H"):
        print(f"  正在读取 {fq}（第一次可能要从服务器下载历史，需等待）…", flush=True)
        b = fetch_bars(fq, C.HISTORY_BARS[fq])
        print(f"  {fq}: {len(b)} 根，{b.index[0]} ~ {b.index[-1]} UTC → {'足够' if len(b) >= {'30MIN': 6000, '1H': 7000}[fq] else '不足，请调大图表最大 K 线数'}")
    if args.dukascopy_dir:
        from factors.data_loader import load_m1, resample_ohlcv
        print("  正在读取 Dukascopy M1 全部历史（需要几分钟）…", flush=True)
        d = resample_ohlcv(load_m1(args.dukascopy_dir), "30MIN")
        m = fetch_bars("30MIN", C.HISTORY_BARS["30MIN"])
        step = pd.Timedelta("30min")
        # 只用"上一根正好是 30 分钟前"的收益，避免跨缺口的收益污染相关系数
        rd = np.log(d["close"]).diff().where(d.index.to_series().diff() == step)
        rm = np.log(m["close"]).diff().where(m.index.to_series().diff() == step)
        print("== MT5 30MIN 历史中超过 4 天的缺口（周末约 2 天，属正常）")
        gaps = m.index.to_series().diff()
        big = gaps[gaps > pd.Timedelta(days=4)]
        for t, g in big.tail(15).items():
            print(f"  {t - g} → {t}（{g.days} 天）")
        if len(big) > 15:
            print(f"  …共 {len(big)} 个")
        if len(big) == 0:
            print("  无")
        print("== 与 Dukascopy 的对齐（30MIN 收益在不同时间平移下的相关，最佳应为 0 小时）")
        for lag in range(-6, 7):
            x = pd.concat([rd, rm.shift(lag, freq=step)], axis=1, sort=True).dropna()
            print(f"  平移 {lag * 0.5:+.1f} 小时：相关 {x.corr().iloc[0, 1]:.3f}（{len(x)} 根重叠）")
        both = pd.concat([d[["close", "volume"]], m[["close", "volume"]], rd, rm], axis=1, sort=True,
                         keys=["duka", "mt5", "rd", "rm"]).dropna()
        both.columns = ["dc", "dv", "mc", "mv", "rd", "rm"]
        print("== 按年份：收益相关 / 收盘价差 / 成交量相关（近几年最重要，实时计算只用最近约 1~2 年）")
        for y, g in both.groupby(both.index.year):
            diff = g.mc - g.dc
            print(f"  {y}：{len(g):6d} 根，收益相关 {g.rd.corr(g.rm):.3f}，价差中位 {diff.median():+.2f} / 绝对值95%分位 {diff.abs().quantile(.95):.2f}，"
                  f"成交量相关 {np.corrcoef(np.log1p(g.dv), np.log1p(g.mv))[0, 1]:.3f}")
    mt5.shutdown()


if __name__ == "__main__":
    main()

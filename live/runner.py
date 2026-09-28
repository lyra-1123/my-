# -*- coding: utf-8 -*-
"""
MT5 模拟盘实时程序：python -m live.runner [--once]

每根 30 分钟 K 线收盘后（UTC :00 / :30 + 延迟），对每个启用的策略：
  1) 从 MT5 取 K 线并换算为 UTC，丢弃未走完的最后一根；
  2) 用与模拟盘完全相同的代码（paper.engine.compute）计算目标仓位；
  3) 按魔术号把实际持仓对齐到目标（先平后开）；影子版本 HA1H-TS2 设置移动止损；
  4) 风控：STOP 文件 → 平仓暂停；单日亏损熔断；点差过宽不开新仓；数据过期不交易；只允许模拟账户。
所有决策写入 live/logs/decisions.csv，所有订单写入 live/logs/orders.csv。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd

from factors.core import trading_day
from live import broker
from live import config as C
from live.mt5_api import mt5
from live.mt5_data import DUR, connect, fetch_bars, utc_now_from_server
from paper.engine import compute
from paper.specs import get_spec

STATE = os.path.join(C.LOG_DIR, "state.json")


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {"last_bar": {}, "paused_until": None}


def save_state(s):
    os.makedirs(C.LOG_DIR, exist_ok=True)
    json.dump(s, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


def log_decision(row):
    path = os.path.join(C.LOG_DIR, "decisions.csv")
    new = not os.path.exists(path)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def preflight():
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError(f"读取账户失败：{mt5.last_error()}")
    if C.DEMO_ONLY and acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        raise RuntimeError("当前不是模拟账户，DEMO_ONLY=True，拒绝运行")
    enabled = [s for s, v in C.STRATEGIES.items() if v["enabled"]]
    if len(enabled) > 1 and acc.margin_mode != mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
        raise RuntimeError("账户是净额（netting）模式：多个策略会合并成一个仓位，无法按魔术号区分。请使用对冲（hedging）模拟账户")
    magics = [v["magic"] for v in C.STRATEGIES.values()]
    if len(set(magics)) != len(magics):
        raise RuntimeError("魔术号重复")


def daily_pnl(now_utc) -> float:
    """当日（纽约 17:00 起）本程序所有魔术号的已实现 + 浮动盈亏（美元）。"""
    magics = {v["magic"] for v in C.STRATEGIES.values()}
    day_start_ny = (pd.Timestamp(now_utc).tz_localize("UTC").tz_convert("America/New_York") + pd.Timedelta(hours=7)).normalize() - pd.Timedelta(hours=7)
    start_utc = day_start_ny.tz_convert("UTC").tz_localize(None)
    realized = 0.0
    deals = mt5.history_deals_get(start_utc.to_pydatetime(), (now_utc + pd.Timedelta(days=1)).to_pydatetime()) or ()
    for d in deals:
        if d.magic in magics and d.symbol == C.SYMBOL:
            realized += d.profit + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0)
    floating = sum(p.profit + getattr(p, "swap", 0.0) for p in (mt5.positions_get(symbol=C.SYMBOL) or ()) if p.magic in magics)
    return realized + floating


def process(sid: str, cfg: dict, now_utc: pd.Timestamp, state: dict, halt: str | None, force=False, halt_orders: str = ""):
    spec = get_spec(sid)
    bars = fetch_bars(spec.freq, C.HISTORY_BARS[spec.freq], now_utc)
    last_bar = bars.index[-1]
    if not force and state["last_bar"].get(sid) == str(last_bar):
        return
    age_min = (now_utc - (last_bar + DUR[spec.freq])).total_seconds() / 60
    c = compute(spec, bars)
    row = c.iloc[-1]
    target = float(row["target"])
    stop = float(row["next_stop"]) if "next_stop" in c and np.isfinite(row["next_stop"]) else None
    desired = round(target * cfg["lots"], 2)
    current = broker.net_lots(cfg["magic"])
    spread = broker.spread_usd()
    note, action = "", "HOLD"
    if halt:
        desired, note = 0.0, halt
    elif len(bars) < C.MIN_BARS[spec.freq]:
        note, desired = f"连续历史只有 {len(bars)} 根（需要 {C.MIN_BARS[spec.freq]}），不交易", 0.0
    elif age_min > C.MAX_DATA_AGE_MIN:
        note, desired = f"数据过期 {age_min:.0f} 分钟，不交易", current
    failed = []
    if abs(desired - current) > 1e-9 and not halt_orders:
        if current != 0 and (desired == 0 or np.sign(desired) != np.sign(current) or abs(desired) != abs(current)):
            action = "CLOSE"
            if not broker.close_all(sid, cfg["magic"], now_utc):
                failed.append(f"平仓失败：{broker.LAST_ERROR[0]}")
        if desired != 0 and not failed:
            if spread > C.MAX_SPREAD_USD:
                note += f" 点差 {spread:.2f} 过宽，不开仓"
            elif state.get("suspect", {}).get(sid):
                note += " 此前成交后持仓与目标不符，已停止为该策略开新仓，请人工核对后删除 state.json 中的 suspect 项"
            else:
                if stop is None and "trail_dist" in c:
                    # 模型的初始止损 = 下一根开盘价 ∓ trail×日线ATR；实盘此时下一根开盘价就是现价
                    px = broker.entry_price(int(np.sign(desired)))
                    stop = px - np.sign(desired) * float(row["trail_dist"])
                action = "OPEN_LONG" if desired > 0 else "OPEN_SHORT"
                if current != 0:
                    action = "REVERSE_" + action.split("_")[1]
                if not broker.open_position(sid, cfg["magic"], abs(desired), int(np.sign(desired)), now_utc, sl=stop):
                    failed.append(f"开仓失败：{broker.LAST_ERROR[0]}")
        after = broker.net_lots(cfg["magic"])
        if not failed and abs(after - desired) > 1e-9:
            # 回执成功但按魔术号查不到应有的持仓：停止继续开仓，防止重复下单
            state.setdefault("suspect", {})[sid] = True
            failed.append(f"回执成功但持仓为 {after}（目标 {desired}），已停止为该策略开新仓")
    elif abs(desired - current) > 1e-9 and halt_orders:
        note += f" 未下单：{halt_orders}"
        action = "BLOCKED"
    if failed:
        action = "ORDER_FAILED"
        note += " " + "；".join(failed)
        print(f"!!! {sid} 下单失败：{'；'.join(failed)}")
    state["last_bar"][sid] = str(last_bar)
    if action != "HOLD" or note.strip():
        print(f"  {sid}（{spec.name}）：{action}，目标 {desired:+.2f} 手，原持仓 {current:+.2f} 手"
              + (f"，止损 {stop:.2f}" if stop is not None else "") + (f"，备注：{note.strip()}" if note.strip() else ""))
    log_decision({"utc_time": str(now_utc), "strategy": sid, "name": spec.name, "bar": str(last_bar), "z": round(float(row["z"]), 4),
                  "z_exec": round(float(row["z_exec"]), 4), "target": target, "desired_lots": desired, "current_lots": current,
                  "action": action, "stop": stop, "spread": round(spread, 3), "data_age_min": round(age_min, 1), "note": note.strip()})


def cycle(state: dict, force=False):
    now_utc = utc_now_from_server()
    halt = None
    if os.path.exists(C.KILL_FILE):
        halt = "STOP 文件存在：平仓并暂停"
    elif state.get("paused_until") and now_utc < pd.Timestamp(state["paused_until"]):
        halt = f"熔断暂停至 {state['paused_until']}"
    elif C.DAILY_LOSS_LIMIT_USD is not None:
        pnl = daily_pnl(now_utc)
        if pnl < -abs(C.DAILY_LOSS_LIMIT_USD):
            nxt = trading_day(pd.DatetimeIndex([now_utc]))[0] + pd.Timedelta(days=1)
            state["paused_until"] = str((nxt.tz_localize("America/New_York") - pd.Timedelta(hours=7)).tz_convert("UTC").tz_localize(None))
            halt = f"当日亏损 {pnl:.2f}$ 超过熔断线，暂停至 {state['paused_until']}"
    ok, why = broker.trading_allowed()
    if not ok:
        print(f"!!! 无法下单：{why}")
    for sid, cfg in C.STRATEGIES.items():
        if cfg["enabled"]:
            try:
                process(sid, cfg, now_utc, state, halt, force, "" if ok else why)
            except Exception as e:  # 单个策略出错不影响其他策略
                log_decision({"utc_time": str(now_utc), "strategy": sid, "name": "", "bar": "", "z": "", "z_exec": "", "target": "",
                              "desired_lots": "", "current_lots": "", "action": "ERROR", "stop": "", "spread": "", "data_age_min": "",
                              "note": repr(e)})
    # 每个交易日收盘（纽约 17:00）后的第一次运行：自动生成前一交易日的复盘数据报告
    if C.AUTO_DAILY_REVIEW:
        try:
            from live.daily_review import last_closed_day, run_review
            day = last_closed_day(now_utc)
            if state.get("last_review") != str(day.date()):
                path = run_review(day, now_utc)
                state["last_review"] = str(day.date())
                print(f"已生成复盘报告：{path}")
        except Exception as e:
            log_decision({"utc_time": str(now_utc), "strategy": "DAILY_REVIEW", "name": "", "bar": "", "z": "", "z_exec": "", "target": "",
                          "desired_lots": "", "current_lots": "", "action": "ERROR", "stop": "", "spread": "", "data_age_min": "",
                          "note": repr(e)})
    save_state(state)


def next_wakeup(now: pd.Timestamp) -> pd.Timestamp:
    return now.floor("30min") + pd.Timedelta("30min") + pd.Timedelta(seconds=C.BAR_CLOSE_DELAY_SEC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只运行一次（用于测试或由计划任务调用）")
    args = ap.parse_args()
    connect(); preflight()
    ok, why = broker.trading_allowed()
    if not ok:
        raise SystemExit(f"无法下单：{why}。开启后再启动。")
    state = load_state()
    if args.once or C.ALIGN_ON_START:
        cycle(state, force=C.ALIGN_ON_START)
        if args.once:
            return
    print("已启动。每根 30 分钟 K 线收盘后运行；Ctrl+C 退出；创建 live/STOP 文件可平仓暂停。")
    while True:
        wake = next_wakeup(pd.Timestamp(time.time(), unit="s"))
        time.sleep(max(1.0, (wake - pd.Timestamp(time.time(), unit="s")).total_seconds()))
        cycle(state)
        print(f"{pd.Timestamp(time.time(), unit='s'):%Y-%m-%d %H:%M:%S} UTC 已处理，详见 {C.LOG_DIR}/decisions.csv")


if __name__ == "__main__":
    main()

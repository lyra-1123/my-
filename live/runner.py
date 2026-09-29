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
from live.logutil import append_row, flush_pending
from live.mt5_api import mt5
from live.mt5_data import DUR, connect, fetch_bars, server_to_utc, utc_now_from_server
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
    try:
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=1, default=str)
    except OSError as e:
        print(f"!!! 状态文件写入失败（不影响本次交易）：{e!r}")


DECISION_FIELDS = ["utc_time", "beijing_time", "strategy", "name", "bar", "z", "z_exec", "target", "desired_lots", "current_lots",
                   "action", "signal_price", "bid", "ask", "close_fill", "open_fill", "slip_close_vs_signal", "slip_open_vs_signal",
                   "slip_vs_quote", "stop", "spread", "data_age_min", "note"]


def log_decision(row):
    """写决策日志；文件被占用时暂存，绝不让程序崩溃（见 live/logutil.py）。"""
    try:
        row["utc_time"] = pd.Timestamp(row["utc_time"]).strftime("%Y-%m-%d %H:%M:%S")   # 统一到秒，避免格式混杂
        row.setdefault("beijing_time", pd.Timestamp(row["utc_time"]).tz_localize("UTC").tz_convert("Asia/Shanghai")
                       .strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass
    append_row(os.path.join(C.LOG_DIR, "decisions.csv"), row, DECISION_FIELDS)


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


def expected_last_bar(freq: str, now_utc: pd.Timestamp) -> pd.Timestamp:
    """此刻应当已经走完的最后一根 K 线的开盘时间。"""
    d = DUR[freq]
    return pd.Timestamp(now_utc).floor(d) - d


def fetch_bars_retry(freq: str, now_utc: pd.Timestamp) -> pd.DataFrame:
    """
    取 K 线；取数失败，或刚收盘的那根还没到（终端短暂断线/同步慢），等 RETRY_WAIT_SEC 秒重试，最多 RETRY_TIMES 次。
    休市（周末、换日休市的一小时）时最后一根本来就不会更新，不重试。
    """
    exp = expected_last_bar(freq, now_utc)
    ny = pd.Timestamp(exp).tz_localize("UTC").tz_convert("America/New_York")
    in_break = ny.hour == 17 or ny.weekday() == 5 or (ny.weekday() == 4 and ny.hour >= 17) or (ny.weekday() == 6 and ny.hour < 18)
    last_err = None
    for k in range(C.RETRY_TIMES + 1):
        try:
            bars = fetch_bars(freq, C.HISTORY_BARS[freq], now_utc)
            if bars.index[-1] >= exp or in_break:   # 每日换日休市 / 周末：本来就没有这根 K 线
                return bars
            tick = mt5.symbol_info_tick(C.SYMBOL)
            if tick is None or (now_utc - server_to_utc(pd.DatetimeIndex([pd.Timestamp(tick.time, unit="s")]))[0]) > pd.Timedelta(minutes=10):
                return bars   # 报价也没在动：休市或长时间断线，按原逻辑处理（数据过期保护）
            last_err = f"最新 K 线 {bars.index[-1]} 尚未更新到 {exp}"
        except Exception as e:
            bars, last_err = None, repr(e)
            try:   # 与终端的连接断了（例如终端重启）：重新连接
                connect()
            except Exception:
                pass
        if k < C.RETRY_TIMES:
            print(f"  {freq} 数据未就绪（{last_err}），{C.RETRY_WAIT_SEC} 秒后重试（{k + 1}/{C.RETRY_TIMES}）")
            time.sleep(C.RETRY_WAIT_SEC)
            now_utc = now_utc + pd.Timedelta(seconds=C.RETRY_WAIT_SEC)
    if bars is None:
        raise RuntimeError(f"重试 {C.RETRY_TIMES} 次后仍取不到 {freq} K 线：{last_err}")
    return bars


def process(sid: str, cfg: dict, now_utc: pd.Timestamp, state: dict, halt: str | None, force=False, halt_orders: str = ""):
    spec = get_spec(sid)
    bars = fetch_bars_retry(spec.freq, now_utc)
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
    broker.FILLS.clear()
    tick0 = mt5.symbol_info_tick(C.SYMBOL)   # 下单前的报价
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
    elif stop is not None and desired != 0 and not halt_orders and broker.positions(cfg["magic"]):
        # 持仓与目标一致（影子 HA1H-TS2）：把服务器止损同步到模型的移动止损
        tick = mt5.symbol_info_tick(C.SYMBOL)
        if tick is not None and ((desired > 0 and stop >= tick.bid) or (desired < 0 and stop <= tick.ask)):
            note += f" 模型止损 {stop:.2f} 已越过现价，未更新服务器止损"
        elif not broker.set_stop(sid, cfg["magic"], stop, now_utc):
            failed.append(f"更新止损失败：{broker.LAST_ERROR[0]}")
    if failed:
        action = "ORDER_FAILED"
        note += " " + "；".join(failed)
        print(f"!!! {sid} 下单失败：{'；'.join(failed)}")
    state["last_bar"][sid] = str(last_bar)
    if action != "HOLD" or note.strip():
        print(f"  {sid}（{spec.name}）：{action}，目标 {desired:+.2f} 手，原持仓 {current:+.2f} 手"
              + (f"，止损 {stop:.2f}" if stop is not None else "") + (f"，备注：{note.strip()}" if note.strip() else ""))
    # 价格与滑点（$/盎司，正 = 对自己不利）：
    #   信号价 = 信号 K 线收盘价（BID；模型按下一根开盘价成交，与之几乎相同）
    #   相对信号价的滑点 = 含点差的全部执行成本；相对报价的滑点 = 下单报价到成交价的纯滑点
    sig_px = round(float(row["close"]), 2)
    closes = [f for f in broker.FILLS if f["closing"] and f["fill_price"]]
    opens = [f for f in broker.FILLS if not f["closing"] and f["fill_price"]]
    close_px = float(np.mean([f["fill_price"] for f in closes])) if closes else None
    open_px = float(opens[-1]["fill_price"]) if opens else None
    side_new, side_old = np.sign(desired), np.sign(current)
    slip_open = round(side_new * (open_px - sig_px), 3) if open_px else None
    slip_close = round(-side_old * (close_px - sig_px), 3) if close_px else None
    slip_q = [(1 if not f["closing"] else -1) * (side_new if not f["closing"] else side_old) * (f["fill_price"] - f["req_price"])
              for f in broker.FILLS if f["fill_price"] and f["req_price"]]
    if broker.FILLS:
        print(f"    信号价 {sig_px:.2f}" + (f"，平仓成交 {close_px:.2f}（滑点 {slip_close:+.2f}）" if close_px else "")
              + (f"，开仓成交 {open_px:.2f}（滑点 {slip_open:+.2f}）" if open_px else ""))
    log_decision({"utc_time": str(now_utc), "strategy": sid, "name": spec.name, "bar": str(last_bar),
                  "signal_price": sig_px, "bid": getattr(tick0, "bid", ""), "ask": getattr(tick0, "ask", ""),
                  "close_fill": round(close_px, 2) if close_px else "", "open_fill": round(open_px, 2) if open_px else "",
                  "slip_close_vs_signal": "" if slip_close is None else slip_close,
                  "slip_open_vs_signal": "" if slip_open is None else slip_open,
                  "slip_vs_quote": round(float(np.mean(slip_q)), 3) if slip_q else "",
                  "z": round(float(row["z"]), 4),
                  "z_exec": round(float(row["z_exec"]), 4), "target": target, "desired_lots": desired, "current_lots": current,
                  "action": action, "stop": stop, "spread": round(spread, 3), "data_age_min": round(age_min, 1), "note": note.strip()})
    return action, note.strip()


def alert_file() -> str:
    return os.path.join(C.LOG_DIR, "ALERT.txt")


def raise_alert(now_utc, problems: list[str], state: dict):
    """下单失败 / 不允许下单 / 报错：醒目打印、响铃，并写 live/logs/ALERT.txt（live.status 会显示）。"""
    state.setdefault("alert_since", str(now_utc))
    state["alert_cycles"] = state.get("alert_cycles", 0) + 1
    bj = pd.Timestamp(now_utc).tz_localize("UTC").tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M")
    lines = [f"[北京 {bj}] {p}" for p in problems]
    print("\n" + "!" * 70)
    print(f"!!! 告警：程序无法按目标下单（自 {state['alert_since']} UTC 起连续 {state['alert_cycles']} 轮）")
    for x in lines:
        print("!!!   " + x)
    print("!!! 仓位正在与模型脱节。处理好后程序会在下一根 K 线自动补齐，告警自动解除。")
    print("!" * 70 + "\n")
    try:
        os.makedirs(C.LOG_DIR, exist_ok=True)
        new = not os.path.exists(alert_file())
        with open(alert_file(), "a", encoding="utf-8") as fh:
            if new:
                fh.write(f"告警开始（UTC {state['alert_since']}）：程序无法按目标下单，仓位与模型脱节。问题解决后自动删除本文件。\n")
            fh.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"!!! 告警文件写入失败：{e!r}")
    if os.name == "nt":
        try:
            import winsound
            for _ in range(3):
                winsound.MessageBeep(winsound.MB_ICONHAND)
                time.sleep(0.4)
        except Exception:
            pass


def clear_alert(state: dict):
    if state.pop("alert_since", None) is None and not os.path.exists(alert_file()):
        return
    n = state.pop("alert_cycles", 0)
    try:
        if os.path.exists(alert_file()):
            os.remove(alert_file())
    except OSError:
        pass
    print(f"告警解除：本轮下单正常（此前连续 {n} 轮异常）")


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
    problems = [] if ok else [f"无法下单：{why}"]
    if not ok:
        print(f"!!! 无法下单：{why}")
    for sid, cfg in C.STRATEGIES.items():
        if cfg["enabled"]:
            try:
                res = process(sid, cfg, now_utc, state, halt, force, "" if ok else why)
                if res and res[0] == "ORDER_FAILED":
                    problems.append(f"{sid}：{res[1]}")
            except Exception as e:  # 单个策略出错不影响其他策略
                problems.append(f"{sid}：程序报错 {e!r}")
                log_decision({"utc_time": str(now_utc), "strategy": sid, "name": "", "bar": "", "z": "", "z_exec": "", "target": "",
                              "desired_lots": "", "current_lots": "", "action": "ERROR", "stop": "", "spread": "", "data_age_min": "",
                              "note": repr(e)})
    # 已停用的策略不再被管理：如果它的魔术号下还有持仓，这笔单没人平，必须提醒
    for sid, cfg in C.STRATEGIES.items():
        if not cfg["enabled"] and broker.positions(cfg["magic"]):
            problems.append(f"{sid} 已停用，但魔术号 {cfg['magic']} 仍有持仓 {broker.net_lots(cfg['magic']):+.2f} 手，"
                            "程序不会再管理它：请在 MT5 里手动平仓")
    if problems:
        raise_alert(now_utc, problems, state)
    else:
        clear_alert(state)
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
            print(f"!!! 复盘报告生成失败：{e!r}（交易不受影响；可手动运行 python -m live.daily_review 查看详细错误）")
            log_decision({"utc_time": str(now_utc), "strategy": "DAILY_REVIEW", "name": "", "bar": "", "z": "", "z_exec": "", "target": "",
                          "desired_lots": "", "current_lots": "", "action": "ERROR", "stop": "", "spread": "", "data_age_min": "",
                          "note": repr(e)})
    for name in ("decisions.csv", "orders.csv"):
        flush_pending(os.path.join(C.LOG_DIR, name))
    save_state(state)


def next_wakeup(now: pd.Timestamp) -> pd.Timestamp:
    return now.floor("30min") + pd.Timedelta("30min") + pd.Timedelta(seconds=C.BAR_CLOSE_DELAY_SEC)


def disable_quick_edit():
    """
    Windows 命令行的"快速编辑模式"：在窗口里点一下鼠标就会进入选择状态，程序下次打印时整个进程被挂起，
    直到按 Esc/回车。对无人值守的交易程序很危险，这里在启动时关掉它（只影响本窗口）。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)                  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            ENABLE_QUICK_EDIT, ENABLE_EXTENDED_FLAGS = 0x0040, 0x0080
            k32.SetConsoleMode(h, (mode.value & ~ENABLE_QUICK_EDIT) | ENABLE_EXTENDED_FLAGS)
    except Exception:
        pass


def main():
    disable_quick_edit()
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
        try:
            cycle(state)
            print(f"{pd.Timestamp(time.time(), unit='s'):%Y-%m-%d %H:%M:%S} UTC 已处理，详见 {C.LOG_DIR}/decisions.csv")
        except Exception as e:   # 任何意外都不能让程序退出：打印后等下一根 K 线再试
            import traceback
            traceback.print_exc()
            print(f"!!! {pd.Timestamp(time.time(), unit='s'):%Y-%m-%d %H:%M:%S} UTC 本次运行出错：{e!r}；程序继续运行，下一根 K 线重试")


if __name__ == "__main__":
    main()

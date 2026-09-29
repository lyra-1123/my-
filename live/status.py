# -*- coding: utf-8 -*-
"""
实时状态：python -m live.status [--watch 60]

对每个启用的策略显示：
  - 实际持仓（MT5，按魔术号）与模型目标是否一致
  - 信号 z（最近几根已收盘 K 线）与进出场规则，离开仓/平仓/反手还差多少
  - 盘中预估：把尚未走完的 K 线按现价当作收盘，z 大约是多少、会触发什么（仅供参考，程序只在 K 线收盘时决策）
  - 换日前强制平仓的时间；移动止损版的止损位与距离
  - 程序是否在线（最近一次决策距今多久）
只读：不下单、不改任何文件，可以和 runner 同时运行。
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from live import broker
from live import config as C
from live.mt5_api import mt5
from live.mt5_data import DUR, connect, fetch_bars, utc_now_from_server
from paper.engine import compute
from paper.specs import get_spec

NEAR = 0.30   # 离阈值不到这么多个 z 单位时标注"接近"
SIDE = {1: "多", -1: "空", 0: "空仓"}


def _bj(t) -> str:
    return pd.Timestamp(t).tz_localize("UTC").tz_convert("Asia/Shanghai").strftime("%m-%d %H:%M")


def _dur(td: pd.Timedelta) -> str:
    m = int(td.total_seconds() // 60)
    return f"{m // 60}小时{m % 60:02d}分" if m >= 60 else f"{m}分"


def next_decision(freq: str, now: pd.Timestamp) -> pd.Timestamp:
    d = DUR[freq]
    return now.floor(d) + d + pd.Timedelta(seconds=C.BAR_CLOSE_DELAY_SEC)


def force_flat_time(freq: str, now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """下一次换日前强制平仓的成交时刻（纽约 17:00 − 1 根 K 线）与恢复开仓的最早决策时刻（纽约 18:00 收盘的 K 线）。"""
    ny = pd.Timestamp(now).tz_localize("UTC").tz_convert("America/New_York")
    d = DUR[freq]
    flat_ny = ny.normalize() + pd.Timedelta(hours=17) - d
    if ny >= flat_ny:
        flat_ny += pd.Timedelta(days=1)
    reopen_ny = flat_ny + d + pd.Timedelta(hours=1) + d
    to_utc = lambda t: t.tz_convert("UTC").tz_localize(None)  # noqa: E731
    return to_utc(flat_ny), to_utc(reopen_ny)


def rule_text(pos: int, z: float, entry: float, ex: float) -> tuple[str, list[str]]:
    """按迟滞规则描述下一步会发生什么，以及离各个阈值的距离。"""
    notes = []
    if pos == 0:
        up, dn = entry - z, z + entry
        what = f"空仓：z > +{entry:.2f} 开多，z < −{entry:.2f} 开空"
        notes.append(f"距开多还差 {max(up, 0):.2f}" + ("（接近）" if 0 < up <= NEAR else ""))
        notes.append(f"距开空还差 {max(dn, 0):.2f}" + ("（接近）" if 0 < dn <= NEAR else ""))
    else:
        s = "多" if pos > 0 else "空"
        # 持仓时：|z| < 出场线 → 平仓；反向超过入场线 → 反手；其余（包括反向但未过入场线）继续持有
        to_exit = (z - ex) if pos > 0 else (-ex - z)
        to_rev = (z + entry) if pos > 0 else (entry - z)
        what = f"持{s}：|z| < {ex:.2f} 平仓，z {'<' if pos > 0 else '>'} {'−' if pos > 0 else '+'}{entry:.2f} 反手，其余继续持有"
        if abs(z) < ex:
            notes.append("已进入平仓区")
        elif (pos > 0 and z > 0) or (pos < 0 and z < 0):
            notes.append(f"距平仓还差 {to_exit:.2f}" + ("（接近）" if to_exit <= NEAR else ""))
        else:
            notes.append("z 已反向但未过入场线：按规则继续持有（在 ±出场线内才平仓）")
        notes.append(f"距反手还差 {max(to_rev, 0):.2f}")
    return what, notes


def strategy_status(sid: str, cfg: dict, now: pd.Timestamp, dec: pd.DataFrame) -> list[str]:
    spec = get_spec(sid)
    L = [f"■ {spec.name}（{sid}，{spec.freq}，{spec.status}）"]
    bars = fetch_bars(spec.freq, C.HISTORY_BARS[spec.freq], now)
    c = compute(spec, bars)
    last = c.index[-1]
    row = c.iloc[-1]
    tgt = int(row["target"])
    z = float(row["z"])
    zs = " ".join(f"{v:+.2f}" for v in c["z"].iloc[-5:])
    # 实际持仓
    ps = broker.positions(cfg["magic"])
    lots = broker.net_lots(cfg["magic"])
    live_side = int(np.sign(lots))
    if ps:
        pr = sum(p.profit for p in ps)
        px = np.average([p.price_open for p in ps], weights=[p.volume for p in ps])
        sl = [p.sl for p in ps if p.sl]
        L.append(f"  实际持仓：{SIDE[live_side]} {abs(lots):.2f} 手 @ {px:.2f}，浮动盈亏 {pr:+.2f}$" + (f"，止损 {sl[0]:.2f}" if sl else ""))
    else:
        L.append("  实际持仓：空仓")
    want = round(tgt * cfg["lots"], 2)
    ok = abs(want - lots) < 1e-9
    L.append(f"  模型目标（{_bj(last + DUR[spec.freq])} 收盘的 K 线）：{SIDE[tgt]}" + ("  ✓ 与实际一致" if ok else
             f"  ✗ 与实际不一致（实际 {lots:+.2f}，应为 {want:+.2f}；下次决策 {_bj(next_decision(spec.freq, now))} 会对齐，若持续不一致请检查日志）"))
    L.append(f"  信号 z = {z:+.2f}（最近 5 根：{zs}）")
    what, notes = rule_text(tgt, z, spec.entry, spec.exit)
    L.append(f"  规则：{what}")
    L.append(f"  距离：{'；'.join(notes)}")
    # 盘中预估
    try:
        bp = fetch_bars(spec.freq, C.HISTORY_BARS[spec.freq], now, include_partial=True)
        if len(bp) and bp.index[-1] > last:
            cp = compute(spec, bp)
            zp, tp = float(cp["z"].iloc[-1]), int(cp["target"].iloc[-1])
            chg = "维持" if tp == tgt else f"将变为 {SIDE[tp]}"
            L.append(f"  盘中预估（当前未走完的 K 线按现价收盘）：z ≈ {zp:+.2f} → {chg}；"
                     f"{_bj(bp.index[-1] + DUR[spec.freq])} 收盘时正式判断（含成交量的指标在 K 线走完前偏低，仅供参考）")
    except Exception as e:  # 预估失败不影响其他信息
        L.append(f"  盘中预估不可用：{e!r}")
    # 换日前平仓
    flat_t, reopen_t = force_flat_time(spec.freq, now)
    if tgt != 0 or lots != 0:
        L.append(f"  换日前强制平仓：北京 {_bj(flat_t)} 左右（还有 {_dur(flat_t - now)}）；之后到 {_bj(reopen_t)} 前不开新仓")
    else:
        L.append(f"  今日最后可开仓的决策：北京 {_bj(flat_t - DUR[spec.freq])} 之前（之后到 {_bj(reopen_t)} 暂停开仓）")
    # 移动止损版
    if spec.rule == "state_trail" and ps:
        sl = [p.sl for p in ps if p.sl]
        tick = mt5.symbol_info_tick(C.SYMBOL)
        if sl and tick:
            ref = tick.bid if live_side > 0 else tick.ask
            dist = (ref - sl[0]) if live_side > 0 else (sl[0] - ref)
            L.append(f"  移动止损：{sl[0]:.2f}，距现价 {dist:.2f}$" + ("（接近）" if dist < 10 else "") +
                     "；止损后同方向需信号先复位（|z| < 出场线）才能再入场")
    # 最近一次决策
    if not dec.empty:
        d = dec[dec["strategy"] == sid]
        if len(d):
            r = d.iloc[-1]
            note = r.get("note")
            L.append(f"  最近决策：北京 {_bj(r['utc_time'])} {r['action']}" + (f"（{note}）" if isinstance(note, str) and note else ""))
    return L


def read_decisions() -> pd.DataFrame:
    p = os.path.join(C.LOG_DIR, "decisions.csv")
    try:
        d = pd.read_csv(p)
        pend = p[:-4] + ".pending.csv"
        if os.path.exists(pend):
            d = pd.concat([d, pd.read_csv(pend)], ignore_index=True)
        d["utc_time"] = pd.to_datetime(d["utc_time"], format="mixed")
        return d.sort_values("utc_time", kind="stable")
    except Exception:
        return pd.DataFrame()


def render() -> str:
    now = utc_now_from_server()
    dec = read_decisions()
    tick = mt5.symbol_info_tick(C.SYMBOL)
    acc = mt5.account_info()
    L = [f"XAUUSD 策略实时状态 · 北京 {_bj(now)}:{pd.Timestamp(now).second:02d} · BID {tick.bid:.2f} / ASK {tick.ask:.2f}（点差 {tick.ask - tick.bid:.2f}）"
         f" · 净值 {acc.equity:.2f} {acc.currency}"]
    # runner 每次运行结束都会写 state.json（休市时也写），用它的修改时间判断程序是否在线
    st = os.path.join(C.LOG_DIR, "state.json")
    if not os.path.exists(st):
        L.append("程序状态：没有找到 state.json（程序还没运行过？）")
    else:
        age = pd.Timedelta(seconds=time.time() - os.path.getmtime(st))
        alive = age <= pd.Timedelta(minutes=35)
        L.append(f"程序状态：最近一次运行在 {_dur(age)}前 → " + ("在线" if alive else "!!! 超过 35 分钟没有运行，请检查 runner 窗口（是否卡住/已退出）"))
    ok, why = broker.trading_allowed()
    if not ok:
        L.append(f"!!! 无法下单：{why}")
    if os.path.exists(C.KILL_FILE):
        L.append("!!! STOP 文件存在：程序会平仓并暂停")
    alert = os.path.join(C.LOG_DIR, "ALERT.txt")
    if os.path.exists(alert):
        try:
            lines = open(alert, encoding="utf-8").read().strip().splitlines()
        except OSError:
            lines = ["（ALERT.txt 读取失败）"]
        L.append("!!! 告警（live/logs/ALERT.txt）：")
        L += ["!!!   " + x for x in lines[:1] + lines[1:][-6:]]
    L.append("")
    for sid, cfg in C.STRATEGIES.items():
        if cfg["enabled"]:
            try:
                L += strategy_status(sid, cfg, now, dec)
            except Exception as e:
                L.append(f"■ {sid}：读取失败 {e!r}")
            L.append("")
    L.append(f"z 的含义：|z| 越大信号越强。开仓线 ±入场阈值，平仓线 ±出场阈值；\"接近\" = 离阈值不到 {NEAR}。程序只在每根 K 线收盘后 8 秒决策。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="每隔多少秒刷新一次（0 = 只显示一次）")
    args = ap.parse_args()
    connect()
    while True:
        text = render()
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(text)
        if not args.watch:
            break
        time.sleep(max(5, args.watch))


if __name__ == "__main__":
    main()

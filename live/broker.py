# -*- coding: utf-8 -*-
"""持仓查询与下单（按魔术号区分策略）。"""
from __future__ import annotations

import csv
import os

import pandas as pd

from live import config as C
from live.mt5_api import mt5


def fill_mode():
    info = mt5.symbol_info(C.SYMBOL)
    fm = getattr(info, "filling_mode", 0)
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def positions(magic: int):
    ps = mt5.positions_get(symbol=C.SYMBOL) or ()
    return [p for p in ps if p.magic == magic]


def net_lots(magic: int) -> float:
    return round(sum(p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume for p in positions(magic)), 2)


def _log(row: dict):
    path = os.path.join(C.LOG_DIR, "orders.csv")
    new = not os.path.exists(path)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


LAST_ERROR = [""]   # 最近一次失败订单的回执说明（runner 写进决策日志）


def trading_allowed() -> tuple[bool, str]:
    """终端和账户是否允许程序下单（"算法交易"按钮、账户权限）。"""
    ti, acc = mt5.terminal_info(), mt5.account_info()
    if ti is not None and not getattr(ti, "trade_allowed", True):
        return False, "MT5 终端未开启'算法交易'（工具栏按钮需为绿色；工具→选项→智能交易→允许算法交易）"
    if acc is not None and not getattr(acc, "trade_expert", True):
        return False, "该账户不允许程序（EA）交易"
    if acc is not None and not getattr(acc, "trade_allowed", True):
        return False, "该账户当前不允许交易"
    return True, ""


def _send(req: dict, sid: str, utc_now) -> bool:
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    if not ok:
        LAST_ERROR[0] = f"{getattr(res, 'retcode', None)} {getattr(res, 'comment', mt5.last_error())}"
    _log({"utc_time": str(utc_now), "strategy": sid, "action": req.get("action"), "type": req.get("type"),
          "volume": req.get("volume"), "req_price": req.get("price"), "sl": req.get("sl"), "position": req.get("position"),
          "retcode": getattr(res, "retcode", None), "fill_price": getattr(res, "price", None),
          "comment": getattr(res, "comment", str(mt5.last_error()))})
    return ok


def close_all(sid: str, magic: int, utc_now) -> bool:
    ok = True
    for p in positions(magic):
        tick = mt5.symbol_info_tick(C.SYMBOL)
        buy_back = p.type == mt5.POSITION_TYPE_SELL
        ok &= _send({"action": mt5.TRADE_ACTION_DEAL, "symbol": C.SYMBOL, "volume": p.volume, "position": p.ticket,
                     "type": mt5.ORDER_TYPE_BUY if buy_back else mt5.ORDER_TYPE_SELL,
                     "price": tick.ask if buy_back else tick.bid, "deviation": C.DEVIATION_POINTS, "magic": magic,
                     "comment": sid[:31], "type_time": mt5.ORDER_TIME_GTC, "type_filling": fill_mode()}, sid, utc_now)
    return ok


def open_position(sid: str, magic: int, lots: float, side: int, utc_now, sl: float | None = None) -> bool:
    tick = mt5.symbol_info_tick(C.SYMBOL)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": C.SYMBOL, "volume": lots,
           "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
           "price": tick.ask if side > 0 else tick.bid, "deviation": C.DEVIATION_POINTS, "magic": magic,
           "comment": sid[:31], "type_time": mt5.ORDER_TIME_GTC, "type_filling": fill_mode()}
    if sl is not None:
        req["sl"] = round(sl, mt5.symbol_info(C.SYMBOL).digits)
    return _send(req, sid, utc_now)


def set_stop(sid: str, magic: int, sl: float, utc_now) -> bool:
    ok = True
    digits = mt5.symbol_info(C.SYMBOL).digits
    for p in positions(magic):
        if abs((p.sl or 0) - sl) < 10 ** -digits:
            continue
        ok &= _send({"action": mt5.TRADE_ACTION_SLTP, "symbol": C.SYMBOL, "position": p.ticket, "sl": round(sl, digits),
                     "tp": p.tp or 0.0, "magic": magic}, sid, utc_now)
    return ok


def entry_price(side: int) -> float:
    """止损参考价：与模型一致用 BID（Dukascopy 数据为 BID）。"""
    return float(mt5.symbol_info_tick(C.SYMBOL).bid)


def spread_usd() -> float:
    t = mt5.symbol_info_tick(C.SYMBOL)
    return float(t.ask - t.bid) if t else float("inf")

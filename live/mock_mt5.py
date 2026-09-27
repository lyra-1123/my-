# -*- coding: utf-8 -*-
"""
测试用的 MT5 模拟接口（只实现本程序用到的部分）。行情来自 data/cache 的 Dukascopy K 线，按"纽约时间 + 7 小时"转成服务器时间；
点差固定 0.2；持仓按魔术号记录；止损在每次推进时间时按 K 线最高/最低价触发。只用于 live/selftest.py。
"""
from collections import namedtuple

import numpy as np
import pandas as pd

TIMEFRAME_M30, TIMEFRAME_H1, TIMEFRAME_D1, TIMEFRAME_M1 = 30, 16385, 16408, 1
TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
ORDER_TIME_GTC = 0
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
TRADE_RETCODE_DONE = 10009
ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 0, 2

_BARS = {}
NOW_UTC = None
_pos, _deals, _ticket = [], [], [1000]
_last_stop_check = None
Pos = namedtuple("Pos", "ticket symbol magic type volume price_open sl tp profit swap")
Res = namedtuple("Res", "retcode price comment")
Tick = namedtuple("Tick", "time bid ask")
Info = namedtuple("Info", "trade_contract_size digits volume_min volume_step trade_stops_level filling_mode")
Acc = namedtuple("Acc", "login server trade_mode margin_mode balance currency")
Deal = namedtuple("Deal", "magic symbol profit commission swap")


def _bars(tf):
    if tf not in _BARS:
        fq = {TIMEFRAME_M30: "30MIN", TIMEFRAME_H1: "1H"}[tf]
        b = pd.read_pickle(f"data/cache/{fq}.pkl")
        srv = b.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
        _BARS[tf] = (b, srv)
    return _BARS[tf]


def _srv(ts_utc):
    return ts_utc.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)


def initialize(*a, **k): return True
def shutdown(): return True
def last_error(): return (0, "ok")
def symbol_select(*a): return True
def account_info(): return Acc(1, "mock", ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_MARGIN_MODE_RETAIL_HEDGING, 10000.0, "USD")
def symbol_info(sym): return Info(100.0, 2, 0.01, 0.01, 0, 1)


def _last_price():
    b, _ = _bars(TIMEFRAME_M30)
    i = b.index.searchsorted(NOW_UTC, side="right") - 1
    return float(b["open"].iloc[i]) if b.index[i] + pd.Timedelta("30min") > NOW_UTC else float(b["close"].iloc[i])


def symbol_info_tick(sym):
    bid = _last_price()
    return Tick(int(_srv(NOW_UTC).timestamp()), bid, bid + 0.2)


def copy_rates_from_pos(sym, tf, start, count):
    b, srv = _bars(tf)
    m = b.index <= NOW_UTC
    bb, ss = b[m].iloc[-count:], srv[m][-count:]
    arr = np.zeros(len(bb), dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")])
    arr["time"] = ((ss - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy(); arr["open"] = bb["open"]; arr["high"] = bb["high"]
    arr["low"] = bb["low"]; arr["close"] = bb["close"]; arr["tick_volume"] = np.maximum(bb["volume"].to_numpy() * 1000, 1).astype("i8")
    return arr


def positions_get(symbol=None): return tuple(_pos)
def history_deals_get(a, b): return tuple(_deals)


def order_send(req):
    if req["action"] == TRADE_ACTION_SLTP:
        for i, p in enumerate(_pos):
            if p.ticket == req["position"]:
                _pos[i] = p._replace(sl=req["sl"])
        return Res(TRADE_RETCODE_DONE, 0.0, "sltp")
    px = req["price"]
    if "position" in req:
        p = next(p for p in _pos if p.ticket == req["position"])
        _pos.remove(p)
        _deals.append(Deal(p.magic, p.symbol, (px - p.price_open) * (1 if p.type == POSITION_TYPE_BUY else -1) * p.volume * 100, 0.0, 0.0))
        return Res(TRADE_RETCODE_DONE, px, "close")
    _ticket[0] += 1
    _pos.append(Pos(_ticket[0], req["symbol"], req["magic"], POSITION_TYPE_BUY if req["type"] == ORDER_TYPE_BUY else POSITION_TYPE_SELL,
                    req["volume"], px, req.get("sl", 0.0) or 0.0, 0.0, 0.0, 0.0))
    return Res(TRADE_RETCODE_DONE, px, "open")


def advance_to(t_utc):
    """推进时间，并按 1H K 线最高/最低价触发止损（与模拟盘影子规则的 1H 粒度一致）。"""
    global NOW_UTC, _last_stop_check
    b, _ = _bars(TIMEFRAME_H1)
    if _last_stop_check is not None:
        seg = b[(b.index >= _last_stop_check) & (b.index + pd.Timedelta("1h") <= t_utc)]
        for ts, bar in seg.iterrows():
            for p in list(_pos):
                if p.sl and ((p.type == POSITION_TYPE_BUY and bar.low <= p.sl) or (p.type == POSITION_TYPE_SELL and bar.high >= p.sl)):
                    fill = min(bar.open, p.sl) if p.type == POSITION_TYPE_BUY else max(bar.open, p.sl)
                    _pos.remove(p)
                    _deals.append(Deal(p.magic, p.symbol, (fill - p.price_open) * (1 if p.type == POSITION_TYPE_BUY else -1) * p.volume * 100, 0.0, 0.0))
            _last_stop_check = ts + pd.Timedelta("1h")
    else:
        _last_stop_check = t_utc.floor("1h")
    NOW_UTC = t_utc

# -*- coding: utf-8 -*-
"""
15min策略评估公共模块：数据读取、信号回放、ETF成交模拟、绩效统计。

信号引擎直接复用盯盘脚本 monitor_15min_signal.py 里的 prepare_indicators /
replay_and_check_latest，保证评估用的判断逻辑跟实盘推送完全一致。已验证：把软止损2
改回AND后，用该引擎回放2014-10~2026-09全历史，跟回测交易明细1434笔逐笔一致(终值84.92x)。
"""

import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENGINE = os.path.join(HERE, 'monitor_15min_signal.py')
ETF_TICK = 0.001


# ========================= 数据读取 =========================

def _read_ifind_table(path, names):
    """读iFinD导出的表格：前4行是 代码名称/中文列名/英文列名/同花顺iFinD，之后是数据。"""
    if path.lower().endswith(('.xlsx', '.xls')):
        raw = pd.read_excel(path, header=None, skiprows=4)
    else:
        raw = pd.read_csv(path, header=None, skiprows=4, encoding='gbk', dtype=str)
    raw = raw.iloc[:, :len(names)]
    raw.columns = names
    return raw


def _to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')


def load_index_15min(path):
    """中证1000指数15min K线。支持两种格式：
    1. iFinD导出的 000852-15min.xlsx (日期/时间/开高低收/成交量/成交额)
    2. 盯盘脚本的缓存 ifind_cache/000852_SH_15min_cache.csv (date,open,high,low,close)
    返回列: date, open, high, low, close"""
    if path.lower().endswith('.csv'):
        head = pd.read_csv(path, nrows=1)
        if 'date' in head.columns:
            df = pd.read_csv(path, parse_dates=['date'])
            return df[['date', 'open', 'high', 'low', 'close']].sort_values('date').reset_index(drop=True)
    raw = _read_ifind_table(path, ['d', 't', 'open', 'high', 'low', 'close', 'vol', 'amt'])
    raw = raw.dropna(subset=['close'])
    df = pd.DataFrame({
        'date': pd.to_datetime(raw['d'].astype(str).str[:10] + ' ' + raw['t'].astype(str)),
        'open': _to_num(raw['open']), 'high': _to_num(raw['high']),
        'low': _to_num(raw['low']), 'close': _to_num(raw['close']),
    })
    return df.sort_values('date').reset_index(drop=True)


def load_etf_1min(path, index_df=None):
    """512100 1分钟K线(iFinD导出，CSV为GBK编码)。返回以时间为索引的DataFrame，
    列: open, high, low, close, vol(复权后价格), raw_open, raw_close(原始价格，用于算tick成本)。
    传入index_df时自动识别份额折算(ETF/指数比值单日跳变>20%)，把折算前的价格乘上折算系数。"""
    raw = _read_ifind_table(path, ['d', 't', 'open', 'high', 'low', 'close', 'vol'])
    raw = raw.dropna(subset=['close'])
    e = pd.DataFrame({
        'open': _to_num(raw['open']).values, 'high': _to_num(raw['high']).values,
        'low': _to_num(raw['low']).values, 'close': _to_num(raw['close']).values,
        'vol': _to_num(raw['vol']).values,
    }, index=pd.to_datetime(raw['d'].astype(str) + ' ' + raw['t'].astype(str)))
    e = e.dropna(subset=['close']).sort_index()
    e['raw_open'] = e['open']
    e['raw_close'] = e['close']
    if index_df is not None:
        idx_daily = index_df.set_index('date')['close'].groupby(lambda x: x.date()).last()
        etf_daily = e['close'].groupby(e.index.date).last()
        ratio = (idx_daily / etf_daily).dropna()
        jump = ratio / ratio.shift(1)
        for day, j in jump[(jump - 1).abs() > 0.2].items():
            factor = 1 / j  # 折算前价格需要乘的系数
            before = e.index < pd.Timestamp(day)
            for c in ['open', 'high', 'low', 'close']:
                e.loc[before, c] = e.loc[before, c] * factor
            print(f'[ETF] 识别到份额折算: {day}，折算前价格已乘以 {factor:.4f}')
    return e


# ========================= 信号回放 =========================

def load_engine(path=DEFAULT_ENGINE):
    """加载盯盘脚本作为信号引擎。可以传入实盘电脑上正在跑的那份脚本路径，
    这样评估用的逻辑就跟实盘100%一致。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到信号引擎脚本 {path}。请把实盘用的 monitor_15min_signal.py 复制到 strategy_eval/ 下，'
            f'或用 --engine 指定它的路径(该文件不入库，仓库是公开的)。')
    spec = importlib.util.spec_from_file_location('signal_engine', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_signals(engine, index_df, key='IM', params_override=None):
    """在指数15min数据上回放状态机，返回逐笔交易：
    en/ex(信号bar收盘时刻), entry_type, reason, p0/p1(指数价格), ret_index"""
    params = dict(engine.INDEX_CONFIGS[key]['params'])
    if params_override:
        params.update(params_override)
    df = engine.prepare_indicators(index_df[['date', 'open', 'high', 'low', 'close']])
    events = engine.replay_and_check_latest(df, params)['all_events']
    rows, cur = [], None
    for ev in events:
        if ev['kind'] == 'ENTRY':
            cur = ev
        elif cur is not None:
            rows.append(dict(en=df.loc[cur['i'], 'date'], ex=df.loc[ev['i'], 'date'],
                             entry_type=cur['reason'], reason=ev['reason'],
                             p0=cur['price'], p1=ev['price']))
            cur = None
    tr = pd.DataFrame(rows)
    if len(tr):
        tr['ret_index'] = tr['p1'] / tr['p0'] - 1
    open_pos = None
    if cur is not None:
        open_pos = dict(en=df.loc[cur['i'], 'date'], entry_type=cur['reason'], p0=cur['price'])
    return tr, open_pos


# ========================= ETF成交模拟 =========================

class EtfBook:
    """按信号时刻在ETF 1分钟数据上找成交价。1分钟bar的时间标签是该分钟结束时刻
    (09:31 = 09:30~09:31)，09:30是开盘集合竞价。"""

    def __init__(self, etf):
        self.e = etf
        self.idx = etf.index
        self.days = np.array(sorted(set(self.idx.date)))

    def price_after(self, ts, delay_min=0):
        """信号在ts时刻确认后的成交价。delay_min=0: 下一分钟bar的开盘价(信号一出立刻下单)；
        delay_min=d: 第d分钟bar的收盘价(模拟人工/程序延迟)。跨午休/隔夜自动顺延到下一个可交易分钟。
        返回 (复权价, 原始价, 成交时刻)"""
        k = self.idx.searchsorted(ts, side='right')
        col, raw_col = ('open', 'raw_open') if delay_min == 0 else ('close', 'raw_close')
        k = k + max(delay_min - 1, 0)
        if k < len(self.idx) and self.idx[k].strftime('%H:%M') == '09:30':
            k += 1  # 不参与开盘集合竞价，用09:31
            col, raw_col = 'open', 'raw_open'
        if k >= len(self.idx):
            return None
        return self.e[col].iloc[k], self.e[raw_col].iloc[k], self.idx[k]

    def next_day_open(self, ts):
        i = self.days.searchsorted(ts.date(), side='right')
        if i >= len(self.days):
            return None
        k = self.idx.searchsorted(pd.Timestamp(self.days[i]) + pd.Timedelta('9h31m'))
        return self.e['open'].iloc[k], self.e['raw_open'].iloc[k], self.idx[k]


def simulate_etf(trades, book, t0=True, comm=0.0001, tick_mult=0.5, delay_min=0):
    """把指数信号搬到ETF上成交，返回逐笔收益DataFrame。
    t0=True: 有底仓，当天买的可以当天卖(实际卖的是底仓)；
    t0=False: 纯T+1，当天买入当天出场信号推迟到次日09:31开盘卖，期间若再次出现入场信号则继续持有。
    comm: 单边佣金率(万一免五=0.0001)；tick_mult: 单边滑点，以tick(0.001元)为单位。"""
    tr = trades[(trades['en'] >= book.idx[0]) & (trades['ex'] <= book.idx[-1])]

    def cost(raw_px):
        return comm + tick_mult * ETF_TICK / raw_px

    out = []
    if t0:
        for _, x in tr.iterrows():
            a, b = book.price_after(x['en'], delay_min), book.price_after(x['ex'], delay_min)
            if a is None or b is None:
                continue
            out.append(dict(en=x['en'], ex=x['ex'], fill_en=a[2], fill_ex=b[2], px_en=a[0], px_ex=b[0],
                            ret_index=x['ret_index'],
                            ret=b[0] / a[0] - 1 - cost(a[1]) - cost(b[1])))
        return pd.DataFrame(out)

    pos, pend, buy = False, None, None
    for _, x in tr.iterrows():
        if pend is not None and pend[2] <= x['en']:
            out.append(dict(en=buy[3], ex=pend[3], fill_en=buy[2], fill_ex=pend[2], px_en=buy[0],
                            px_ex=pend[0], ret=pend[0] / buy[0] - 1 - cost(buy[1]) - cost(pend[1])))
            pos, pend = False, None
        if pos and pend is not None:
            pend = None  # 推迟卖出期间又出现入场信号：继续持有
        elif not pos:
            a = book.price_after(x['en'], delay_min)
            if a is None:
                continue
            buy, pos = (a[0], a[1], a[2], x['en']), True
        b = book.price_after(x['ex'], delay_min)
        if b is None:
            continue
        if b[2].date() <= buy[2].date():
            nd = book.next_day_open(x['ex'])
            if nd is not None:
                pend = (nd[0], nd[1], nd[2], x['ex'])
        else:
            out.append(dict(en=buy[3], ex=x['ex'], fill_en=buy[2], fill_ex=b[2], px_en=buy[0],
                            px_ex=b[0], ret=b[0] / buy[0] - 1 - cost(buy[1]) - cost(b[1])))
            pos = False
    if pend is not None:
        out.append(dict(en=buy[3], ex=pend[3], fill_en=buy[2], fill_ex=pend[2], px_en=buy[0],
                        px_ex=pend[0], ret=pend[0] / buy[0] - 1 - cost(buy[1]) - cost(pend[1])))
    return pd.DataFrame(out)


# ========================= 统计 =========================

def summarize(ret, start, end):
    """逐笔收益序列的汇总：笔数/单笔平均/胜率/复利终值/年化/逐笔口径最大回撤"""
    ret = pd.Series(ret).reset_index(drop=True)
    if ret.empty:
        return dict(n=0)
    eq = (1 + ret).cumprod()
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-9)
    return dict(n=len(ret), avg_pct=round(ret.mean() * 100, 3), win=round((ret > 0).mean(), 3),
                final=round(eq.iloc[-1], 2), cagr=round(eq.iloc[-1] ** (1 / years) - 1, 3),
                mdd=round((eq / eq.cummax() - 1).min(), 3))

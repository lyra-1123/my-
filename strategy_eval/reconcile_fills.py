# -*- coding: utf-8 -*-
"""
实盘成交 vs 回测信号 对账工具。

回答三个问题：
  1. 实盘有没有漏单/多单(每个信号都执行了吗？有没有信号之外的成交？)
  2. 执行成本多大(从信号确认到成交的延迟、相对信号时刻ETF价格的滑点，单位tick和%)
  3. 实盘单笔收益跟回测差多少，差距来自哪里

用法:
  python reconcile_fills.py --fills 我的成交.csv --index ifind_cache/000852_SH_15min_cache.csv
  python reconcile_fills.py --fills 我的成交.csv --index data/000852-15min.xlsx --etf data/512100-1min.CSV

成交记录CSV格式见 fills_template.csv，必填列: 成交时间, 方向(买/卖), 成交价；
可选列: 数量, 手续费, 信号时ETF价(没有ETF分钟数据时用它当滑点基准)。
只记录策略仓位的买卖，不要记底仓的调整。
"""

import argparse

import numpy as np
import pandas as pd

from common import DEFAULT_ENGINE, ETF_TICK, EtfBook, load_engine, load_etf_1min, load_index_15min, \
    run_signals

GROUP_GAP = pd.Timedelta(minutes=10)  # 同方向、间隔10分钟内的多笔成交视为同一张单拆开成交
SESSIONS = [('09:30', '11:30'), ('13:00', '15:00')]


def trading_minutes(t0, t1):
    """t0到t1之间经过的交易时段分钟数(跳过午休、隔夜和周末)。15:00的信号次日09:35成交算5分钟。"""
    if pd.isna(t1) or t1 <= t0:
        return 0.0
    total = 0.0
    for day in pd.bdate_range(t0.normalize(), t1.normalize()):
        for a, b in SESSIONS:
            lo = max(t0, day + pd.Timedelta(a + ':00'))
            hi = min(t1, day + pd.Timedelta(b + ':00'))
            if hi > lo:
                total += (hi - lo).total_seconds() / 60
    return total


def load_fills(path):
    for enc in ('utf-8-sig', 'gbk'):
        try:
            f = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    f.columns = [c.strip() for c in f.columns]
    f['成交时间'] = pd.to_datetime(f['成交时间'])
    f['方向'] = f['方向'].astype(str).str.strip().map({'买': 'BUY', '买入': 'BUY', 'B': 'BUY', 'BUY': 'BUY',
                                                   '卖': 'SELL', '卖出': 'SELL', 'S': 'SELL', 'SELL': 'SELL'})
    if f['方向'].isna().any():
        raise ValueError('方向列只能填 买/卖')
    for c in ['数量', '手续费', '信号时ETF价']:
        if c not in f.columns:
            f[c] = np.nan
    f = f.sort_values('成交时间').reset_index(drop=True)

    # 合并拆单：按数量加权均价
    orders, cur = [], None
    for _, r in f.iterrows():
        q = r['数量'] if pd.notna(r['数量']) else 1.0
        if cur and cur['side'] == r['方向'] and r['成交时间'] - cur['last'] <= GROUP_GAP:
            cur['amt'] += r['成交价'] * q
            cur['qty'] += q
            cur['fee'] += r['手续费'] if pd.notna(r['手续费']) else 0
            cur['last'] = r['成交时间']
            continue
        cur = dict(side=r['方向'], first=r['成交时间'], last=r['成交时间'], amt=r['成交价'] * q, qty=q,
                   fee=r['手续费'] if pd.notna(r['手续费']) else 0, ref_in_file=r['信号时ETF价'])
        orders.append(cur)
    out = pd.DataFrame(orders)
    out['px'] = out['amt'] / out['qty']
    return out


def signal_events(trades, open_pos):
    ev = [dict(t=x['en'], side='BUY', idx_px=x['p0'], why=x['entry_type']) for _, x in trades.iterrows()]
    ev += [dict(t=x['ex'], side='SELL', idx_px=x['p1'], why=x['reason']) for _, x in trades.iterrows()]
    if open_pos:
        ev.append(dict(t=open_pos['en'], side='BUY', idx_px=open_pos['p0'], why=open_pos['entry_type']))
    return pd.DataFrame(ev).sort_values('t').reset_index(drop=True)


def match(signals, orders, max_delay):
    """每个信号匹配它之后max_delay分钟内第一张同方向、尚未匹配的订单。"""
    used = set()
    rows = []
    for _, s in signals.iterrows():
        cand = orders[(orders['side'] == s['side']) & (orders['first'] >= s['t'])
                      & (orders['first'] <= s['t'] + pd.Timedelta(days=5))
                      & (~orders.index.isin(used))]
        cand = cand[[trading_minutes(s['t'], t) <= max_delay for t in cand['first']]]
        if len(cand):
            o = cand.iloc[0]
            used.add(cand.index[0])
            rows.append(dict(signal_t=s['t'], side=s['side'], why=s['why'], idx_px=s['idx_px'],
                             fill_t=o['first'], fill_px=o['px'], qty=o['qty'], fee=o['fee'],
                             ref_in_file=o['ref_in_file'], order_id=cand.index[0]))
        else:
            rows.append(dict(signal_t=s['t'], side=s['side'], why=s['why'], idx_px=s['idx_px'],
                             fill_t=pd.NaT, fill_px=np.nan, qty=np.nan, fee=np.nan,
                             ref_in_file=np.nan, order_id=np.nan))
    m = pd.DataFrame(rows)
    extra = orders[~orders.index.isin(used)]
    return m, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fills', required=True, help='实盘成交记录CSV')
    ap.add_argument('--index', required=True, help='中证1000 15min数据，需覆盖首笔成交前至少60个交易日')
    ap.add_argument('--etf', help='512100 1分钟数据(可选，用来取信号时刻ETF价格做滑点基准)')
    ap.add_argument('--engine', default=DEFAULT_ENGINE, help='信号引擎脚本(建议传实盘那份)')
    ap.add_argument('--max-delay', type=int, default=60, help='信号后多少个交易分钟内的成交算作执行了该信号(不算午休/隔夜)')
    ap.add_argument('--out', default='reconcile_detail.csv')
    args = ap.parse_args()

    engine = load_engine(args.engine)
    index_df = load_index_15min(args.index)
    orders = load_fills(args.fills)
    trades, open_pos = run_signals(engine, index_df)
    signals = signal_events(trades, open_pos)

    start = orders['first'].min().normalize()
    end = min(orders['last'].max(), index_df['date'].iloc[-1])
    signals = signals[(signals['t'] >= start) & (signals['t'] <= end)].reset_index(drop=True)
    if index_df['date'].iloc[0] > start - pd.Timedelta(days=90):
        print('[提醒] 指数数据开始日期离首笔成交不到90天，状态机可能还没进入跟实盘一致的状态，'
              '最开始几个信号的对账结果仅供参考。\n')

    m, extra = match(signals, orders, args.max_delay)

    # 滑点基准：信号确认后下一分钟ETF开盘价(有ETF数据时)，否则用成交记录里的"信号时ETF价"
    book = EtfBook(load_etf_1min(args.etf, index_df)) if args.etf else None
    refs = []
    for _, r in m.iterrows():
        ref = np.nan
        if book is not None:
            p = book.price_after(r['signal_t'])
            if p is not None and p[2] - r['signal_t'] < pd.Timedelta(hours=20):
                ref = p[1]  # 用原始价格，跟实盘成交价同口径
        if np.isnan(ref) and pd.notna(r['ref_in_file']):
            ref = r['ref_in_file']
        refs.append(ref)
    m['ref_px'] = refs
    sign = np.where(m['side'] == 'BUY', 1, -1)
    m['delay_min'] = [trading_minutes(a, b) if pd.notna(b) else np.nan
                      for a, b in zip(m['signal_t'], m['fill_t'])]
    m['slip_tick'] = sign * (m['fill_px'] - m['ref_px']) / ETF_TICK
    m['slip_pct'] = sign * (m['fill_px'] / m['ref_px'] - 1) * 100
    m['fee_rate_pct'] = m['fee'] / (m['fill_px'] * m['qty']) * 100

    # 逐笔：配对 买信号 -> 下一个卖信号
    pairs = []
    buys = m[m['side'] == 'BUY'].reset_index(drop=True)
    sells = m[m['side'] == 'SELL'].reset_index(drop=True)
    for _, b in buys.iterrows():
        s = sells[sells['signal_t'] > b['signal_t']]
        if s.empty:
            continue
        s = s.iloc[0]
        row = dict(entry_signal=b['signal_t'], exit_signal=s['signal_t'], why_exit=s['why'],
                   ret_index=s['idx_px'] / b['idx_px'] - 1,
                   ret_etf_signal=s['ref_px'] / b['ref_px'] - 1 if pd.notna(b['ref_px']) and pd.notna(s['ref_px'])
                   else np.nan,
                   ret_real=s['fill_px'] / b['fill_px'] - 1 if pd.notna(b['fill_px']) and pd.notna(s['fill_px'])
                   else np.nan)
        if pd.notna(row['ret_real']):
            fee = (b['fee_rate_pct'] if pd.notna(b['fee_rate_pct']) else 0) + \
                  (s['fee_rate_pct'] if pd.notna(s['fee_rate_pct']) else 0)
            row['ret_real_net'] = row['ret_real'] - fee / 100
        pairs.append(row)
    pairs = pd.DataFrame(pairs)

    # ---------- 输出 ----------
    done = m['fill_t'].notna()
    print(f'对账区间: {start.date()} ~ {end}')
    print(f'策略信号 {len(m)} 个(买{(m.side == "BUY").sum()}/卖{(m.side == "SELL").sum()})，'
          f'已执行 {done.sum()} 个，漏执行 {(~done).sum()} 个；信号之外的成交 {len(extra)} 张单\n')

    x = m[done]
    print('【执行延迟】信号bar收盘 -> 首笔成交(交易分钟，不含午休/隔夜)')
    print(f'  中位数 {x["delay_min"].median():.1f} 分钟，平均 {x["delay_min"].mean():.1f} 分钟，'
          f'最长 {x["delay_min"].max():.1f} 分钟')
    y = x.dropna(subset=['slip_tick'])
    if len(y):
        print('\n【滑点】相对信号确认后下一分钟ETF开盘价，正数=吃亏')
        print(f'  平均 {y["slip_tick"].mean():+.2f} tick ({y["slip_pct"].mean():+.4f}%)，'
              f'中位数 {y["slip_tick"].median():+.2f} tick，样本 {len(y)} 个')
        print(f'  买入平均 {y[y.side == "BUY"]["slip_tick"].mean():+.2f} tick，'
              f'卖出平均 {y[y.side == "SELL"]["slip_tick"].mean():+.2f} tick')
        avg = y['slip_tick'].mean()
        if avg <= 0.5:
            verdict = '<=0.5tick，跟回测"底仓T+0, 0.5tick"情景一致或更好'
        elif avg <= 1.0:
            verdict = '0.5~1tick，介于回测0.5tick与1tick情景之间，年化预期要往下调'
        else:
            verdict = '>1tick，比回测最悲观情景还差，优先改进下单方式(限价单/挂单)'
        print(f'  判断: {verdict}')
    else:
        print('\n【滑点】没有可用的基准价：请传 --etf 分钟数据，或在成交记录里填"信号时ETF价"')
    if x['fee_rate_pct'].notna().any():
        print(f'\n【手续费】平均单边费率 {x["fee_rate_pct"].mean():.4f}%')

    if len(pairs):
        p = pairs.dropna(subset=['ret_real'])
        print(f'\n【逐笔收益】已完成的往返交易 {len(pairs)} 笔，其中实盘两边都成交 {len(p)} 笔')
        if len(p):
            print(f'  指数口径(回测)     单笔平均 {p["ret_index"].mean() * 100:+.3f}%')
            if p['ret_etf_signal'].notna().any():
                print(f'  ETF信号时刻价格    单笔平均 {p["ret_etf_signal"].mean() * 100:+.3f}%'
                      f'   (与指数的差 = ETF跟踪/折溢价)')
            print(f'  实盘成交价         单笔平均 {p["ret_real"].mean() * 100:+.3f}%'
                  f'   (与上一行的差 = 延迟+滑点)')
            if 'ret_real_net' in p:
                print(f'  实盘扣手续费       单笔平均 {p["ret_real_net"].mean() * 100:+.3f}%')

    missed = m[~done]
    if len(missed):
        print('\n【漏执行的信号】')
        print(missed[['signal_t', 'side', 'why']].to_string(index=False))
    if len(extra):
        print('\n【信号之外的成交】(可能是手动交易、底仓调整，或延迟超过 --max-delay)')
        print(extra[['first', 'side', 'px', 'qty']].to_string(index=False))

    if args.out.endswith('.xlsx'):
        with pd.ExcelWriter(args.out) as w:
            m.to_excel(w, sheet_name='信号对账', index=False)
            pairs.to_excel(w, sheet_name='逐笔收益', index=False)
        print(f'\n明细已写入 {args.out}')
    else:
        m.to_csv(args.out, index=False, encoding='utf-8-sig')
        pairs_path = args.out.replace('.csv', '_pairs.csv')
        pairs.to_csv(pairs_path, index=False, encoding='utf-8-sig')
        print(f'\n明细已写入 {args.out} 和 {pairs_path}')


if __name__ == '__main__':
    main()

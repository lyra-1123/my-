# -*- coding: utf-8 -*-
"""
用中证1000指数15min算信号、用512100 ETF真实1分钟价格成交的回测，对比不同成交假设。

用法:
  python backtest_etf.py --index data/000852-15min.xlsx --etf data/512100-1min.CSV
  python backtest_etf.py ... --engine D:/.../monitor_15min_signal.py   # 用实盘那份脚本做引擎
  python backtest_etf.py ... --starts 2021-01-01 2024-01-01 --out trades.csv
"""

import argparse

import pandas as pd

from common import DEFAULT_ENGINE, EtfBook, load_engine, load_etf_1min, load_index_15min, \
    run_signals, simulate_etf, summarize

SCENARIOS = [
    ('底仓T+0, 万一, 0滑点', dict(t0=True, comm=0.0001, tick_mult=0)),
    ('底仓T+0, 万一, 0.5tick', dict(t0=True, comm=0.0001, tick_mult=0.5)),
    ('底仓T+0, 万一, 1tick', dict(t0=True, comm=0.0001, tick_mult=1.0)),
    ('底仓T+0, 万一, 0.5tick, 延迟5分钟', dict(t0=True, comm=0.0001, tick_mult=0.5, delay_min=5)),
    ('纯T+1, 万一, 0.5tick', dict(t0=False, comm=0.0001, tick_mult=0.5)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True, help='中证1000 15min数据(iFinD导出xlsx或盯盘缓存csv)')
    ap.add_argument('--etf', required=True, help='512100 1分钟数据(iFinD导出CSV)')
    ap.add_argument('--engine', default=DEFAULT_ENGINE, help='信号引擎脚本路径(默认用仓库里的盯盘脚本)')
    ap.add_argument('--starts', nargs='+', default=['2021-01-01', '2022-08-01', '2024-01-01'])
    ap.add_argument('--out', help='把"底仓T+0, 万一, 0.5tick"情景的逐笔明细写到这个CSV')
    args = ap.parse_args()

    engine = load_engine(args.engine)
    index_df = load_index_15min(args.index)
    etf = load_etf_1min(args.etf, index_df)
    book = EtfBook(etf)
    trades, _ = run_signals(engine, index_df)
    end = min(index_df['date'].iloc[-1], etf.index[-1])
    print(f'指数数据 {index_df["date"].iloc[0]} ~ {index_df["date"].iloc[-1]}，'
          f'ETF数据 {etf.index[0]} ~ {etf.index[-1]}，全历史信号 {len(trades)} 笔\n')

    rows = []
    for start in args.starts:
        tr = trades[(trades['en'] >= start) & (trades['ex'] <= end)]
        rows.append(dict(起始=start, 情景='指数价格成交(回测口径)', **summarize(tr['ret_index'], start, end)))
        for name, kw in SCENARIOS:
            res = simulate_etf(tr, book, **kw)
            rows.append(dict(起始=start, 情景=name, **summarize(res['ret'], start, end)))
            if args.out and name == '底仓T+0, 万一, 0.5tick' and start == args.starts[0]:
                res.to_csv(args.out, index=False, encoding='utf-8-sig')
    out = pd.DataFrame(rows).rename(columns=dict(n='笔数', avg_pct='单笔平均%', win='胜率', final='终值',
                                                 cagr='年化', mdd='逐笔最大回撤'))
    print(out.to_string(index=False))
    if args.out:
        print(f'\n逐笔明细已写入 {args.out}')


if __name__ == '__main__':
    main()

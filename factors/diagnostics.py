# -*- coding: utf-8 -*-
"""样本外诊断：多空拆分（排除牛市漂移）、反向信号（反转假设）含成本回测、VWCT 分年度 IC。
用法：python factors/diagnostics.py（需先把 M1 数据放在 data/）"""
import sys; import os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from data_loader import load_m1, resample_ohlcv
from xauusd_volume_breakout_factors import compute_all_factors, forward_return, rank_ic, backtest_with_cost
m1=load_m1("data"); split=pd.Timestamp("2020-01-01")
def ls_split(df,f,entry=1.5,exit_=0.3):
    z=f; p=pd.Series(np.nan,index=df.index); p[z>entry]=1; p[z<-entry]=-1; p[z.abs()<exit_]=0
    p=p.ffill().fillna(0).shift(1).fillna(0); pnl=p*(df.open.shift(-1)-df.open)
    return round(pnl[p>0].sum(),0), round(pnl[p<0].sum(),0)
for freq in ["5MIN","15MIN","30MIN","1H","4H","1D"]:
    df=resample_ohlcv(m1,freq); fac=compute_all_factors(df,freq); oos=df.index>=split
    print(f"\n== {freq}  OOS buy&hold 1oz: {df.open[oos].iloc[-1]-df.open[oos].iloc[0]:.0f} USD")
    for c in fac:
        L,S=ls_split(df[oos],fac[c][oos]); rev=backtest_with_cost(df[oos],-fac[c][oos])
        print(f"  {c:<24} 动量: 多头毛利 {L:>7} 空头毛利 {S:>7} | 反向(反转)净利 {rev['net_usd']:>8} 交易 {rev['round_trips']}")
    if freq in ("5MIN","15MIN","30MIN"):
        fwd=forward_return(df,{"5MIN":6,"15MIN":4,"30MIN":4}[freq]); c="VolWeightedCloseThrust"
        yr={y:round(rank_ic(fac[c][df.index.year==y],fwd[df.index.year==y]),3) for y in range(2009,2027)}
        print("  VWCT 分年度 IC:",yr)

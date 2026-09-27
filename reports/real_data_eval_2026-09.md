# XAUUSD 真实数据评估结果（Dukascopy M1 2009-01 ~ 2026-09，样本外 = 2020-01-01 起）

## run_real_data.py
```
M1: 5,688,363 根, 2008-12-31 16:00:00 ~ 2026-09-25 20:59:00

======================================================================
5MIN: 1,243,696 根
  未来函数自检: True
  成本门槛 spread/ATR: 0.202
                         IC_in                  IC_oos                
h                           1       6       24      1       6       24
factor                                                                
SqueezeReleaseMomentum -0.0102 -0.0125 -0.0095 -0.0097 -0.0159 -0.0113
VolConfirmedBreakout   -0.0186 -0.0307 -0.0301 -0.0098 -0.0157 -0.0147
VolWeightedCloseThrust -0.0361 -0.0401 -0.0300 -0.0237 -0.0248 -0.0178
  样本外含成本回测（0.01 手，点差 0.2 美元）:
    VolConfirmedBreakout     {'gross_usd': 1044.84, 'cost_usd': 924.0, 'net_usd': 120.84, 'round_trips': 4620, 'gross_per_trip_usd': 0.226, 'time_in_market': 0.255, 'max_drawdown_usd': -1467.86}
    VolWeightedCloseThrust   {'gross_usd': 58.07, 'cost_usd': 2324.0, 'net_usd': -2265.93, 'round_trips': 11620, 'gross_per_trip_usd': 0.005, 'time_in_market': 0.285, 'max_drawdown_usd': -2893.07}
    SqueezeReleaseMomentum   {'gross_usd': 50.95, 'cost_usd': 2296.4, 'net_usd': -2245.45, 'round_trips': 11482, 'gross_per_trip_usd': 0.004, 'time_in_market': 0.141, 'max_drawdown_usd': -2528.22}

======================================================================
15MIN: 425,761 根
  未来函数自检: True
  成本门槛 spread/ATR: 0.113
                         IC_in                  IC_oos                
h                           1       4       16      1       4       16
factor                                                                
SqueezeReleaseMomentum -0.0157 -0.0158  0.0012 -0.0098 -0.0118 -0.0074
VolConfirmedBreakout   -0.0157 -0.0209 -0.0177 -0.0057 -0.0109 -0.0095
VolWeightedCloseThrust -0.0395 -0.0406 -0.0209 -0.0260 -0.0226 -0.0107
  样本外含成本回测（0.01 手，点差 0.2 美元）:
    VolConfirmedBreakout     {'gross_usd': -137.93, 'cost_usd': 347.2, 'net_usd': -485.13, 'round_trips': 1736, 'gross_per_trip_usd': -0.079, 'time_in_market': 0.243, 'max_drawdown_usd': -903.44}
    VolWeightedCloseThrust   {'gross_usd': -1445.38, 'cost_usd': 1055.6, 'net_usd': -2500.98, 'round_trips': 5278, 'gross_per_trip_usd': -0.274, 'time_in_market': 0.284, 'max_drawdown_usd': -2707.4}
    SqueezeReleaseMomentum   {'gross_usd': -589.31, 'cost_usd': 972.8, 'net_usd': -1562.11, 'round_trips': 4864, 'gross_per_trip_usd': -0.121, 'time_in_market': 0.172, 'max_drawdown_usd': -1662.52}

======================================================================
30MIN: 214,303 根
  未来函数自检: True
  成本门槛 spread/ATR: 0.078
                         IC_in                  IC_oos                
h                           1       4       12      1       4       12
factor                                                                
SqueezeReleaseMomentum -0.0160 -0.0105  0.0009 -0.0079 -0.0098 -0.0057
VolConfirmedBreakout   -0.0135 -0.0135 -0.0061 -0.0044 -0.0063  0.0027
VolWeightedCloseThrust -0.0368 -0.0261 -0.0183 -0.0196 -0.0157 -0.0084
  样本外含成本回测（0.01 手，点差 0.2 美元）:
    VolConfirmedBreakout     {'gross_usd': 348.24, 'cost_usd': 170.6, 'net_usd': 177.64, 'round_trips': 853, 'gross_per_trip_usd': 0.408, 'time_in_market': 0.239, 'max_drawdown_usd': -400.07}
    VolWeightedCloseThrust   {'gross_usd': -404.61, 'cost_usd': 516.2, 'net_usd': -920.81, 'round_trips': 2581, 'gross_per_trip_usd': -0.157, 'time_in_market': 0.28, 'max_drawdown_usd': -1372.98}
    SqueezeReleaseMomentum   {'gross_usd': 131.36, 'cost_usd': 611.2, 'net_usd': -479.84, 'round_trips': 3056, 'gross_per_trip_usd': 0.043, 'time_in_market': 0.181, 'max_drawdown_usd': -669.84}

======================================================================
1H: 107,572 根
  未来函数自检: True
  成本门槛 spread/ATR: 0.054
                         IC_in                  IC_oos                
h                           1       4       12      1       4       12
factor                                                                
SqueezeReleaseMomentum -0.0184 -0.0136 -0.0069 -0.0093 -0.0036  0.0050
VolConfirmedBreakout   -0.0134 -0.0183 -0.0213  0.0026  0.0104  0.0170
VolWeightedCloseThrust -0.0336 -0.0222 -0.0070 -0.0162 -0.0147  0.0004
  样本外含成本回测（0.01 手，点差 0.2 美元）:
    VolConfirmedBreakout     {'gross_usd': 1029.2, 'cost_usd': 95.3, 'net_usd': 933.9, 'round_trips': 476, 'gross_per_trip_usd': 2.16, 'time_in_market': 0.226, 'max_drawdown_usd': -671.59}
    VolWeightedCloseThrust   {'gross_usd': 124.62, 'cost_usd': 314.8, 'net_usd': -190.18, 'round_trips': 1574, 'gross_per_trip_usd': 0.079, 'time_in_market': 0.279, 'max_drawdown_usd': -867.6}
    SqueezeReleaseMomentum   {'gross_usd': 501.23, 'cost_usd': 306.6, 'net_usd': 194.63, 'round_trips': 1533, 'gross_per_trip_usd': 0.327, 'time_in_market': 0.183, 'max_drawdown_usd': -499.87}

======================================================================
4H: 28,784 根
  未来函数自检: True
  成本门槛 spread/ATR: 0.027
                         IC_in                  IC_oos                
h                            1       3       6       1       3       6
factor                                                                
SqueezeReleaseMomentum  0.0013  0.0094  0.0079 -0.0065 -0.0032  0.0020
VolConfirmedBreakout   -0.0038  0.0007  0.0051 -0.0193 -0.0355 -0.0555
VolWeightedCloseThrust -0.0098  0.0006 -0.0045 -0.0027  0.0085  0.0085
  样本外含成本回测（0.01 手，点差 0.2 美元）:
    VolConfirmedBreakout     {'gross_usd': -1073.56, 'cost_usd': 19.2, 'net_usd': -1092.76, 'round_trips': 96, 'gross_per_trip_usd': -11.183, 'time_in_market': 0.184, 'max_drawdown_usd': -1372.29}
    VolWeightedCloseThrust   {'gross_usd': 886.09, 'cost_usd': 79.0, 'net_usd': 807.09, 'round_trips': 395, 'gross_per_trip_usd': 2.243, 'time_in_market': 0.282, 'max_drawdown_usd': -668.99}
    SqueezeReleaseMomentum   {'gross_usd': 1093.37, 'cost_usd': 94.4, 'net_usd': 998.97, 'round_trips': 472, 'gross_per_trip_usd': 2.316, 'time_in_market': 0.196, 'max_drawdown_usd': -368.46}

======================================================================
1D: 5,002 根
  未来函数自检: True
  成本门槛 spread/ATR: 0.011
                         IC_in                  IC_oos                
h                            1       3       5       1       3       5
factor                                                                
SqueezeReleaseMomentum -0.0097  0.0059  0.0178  0.0015  0.0057  0.0120
VolConfirmedBreakout   -0.0138 -0.0168 -0.0141 -0.0155 -0.0239 -0.0429
VolWeightedCloseThrust -0.0002  0.0258  0.0283 -0.0058 -0.0053 -0.0099
  样本外含成本回测（0.01 手，点差 0.2 美元）:
    VolConfirmedBreakout     {'gross_usd': -222.3, 'cost_usd': 7.6, 'net_usd': -229.9, 'round_trips': 38, 'gross_per_trip_usd': -5.85, 'time_in_market': 0.221, 'max_drawdown_usd': -643.91}
    VolWeightedCloseThrust   {'gross_usd': -330.73, 'cost_usd': 16.6, 'net_usd': -347.33, 'round_trips': 83, 'gross_per_trip_usd': -3.985, 'time_in_market': 0.293, 'max_drawdown_usd': -806.6}
    SqueezeReleaseMomentum   {'gross_usd': 433.7, 'cost_usd': 11.0, 'net_usd': 422.7, 'round_trips': 55, 'gross_per_trip_usd': 7.885, 'time_in_market': 0.131, 'max_drawdown_usd': -265.2}
```

## diagnostics.py
```

== 5MIN  OOS buy&hold 1oz: 2764 USD
  VolConfirmedBreakout     动量: 多头毛利   777.0 空头毛利   268.0 | 反向(反转)净利 -1968.84 交易 4620
  VolWeightedCloseThrust   动量: 多头毛利  1384.0 空头毛利 -1326.0 | 反向(反转)净利 -2382.07 交易 11620
  SqueezeReleaseMomentum   动量: 多头毛利   415.0 空头毛利  -364.0 | 反向(反转)净利 -2347.35 交易 11482
  VWCT 分年度 IC: {2009: -0.032, 2010: -0.044, 2011: -0.037, 2012: -0.052, 2013: -0.045, 2014: -0.054, 2015: -0.031, 2016: -0.036, 2017: -0.057, 2018: -0.027, 2019: -0.025, 2020: -0.048, 2021: -0.039, 2022: -0.024, 2023: -0.012, 2024: -0.013, 2025: -0.022, 2026: -0.016}

== 15MIN  OOS buy&hold 1oz: 2765 USD
  VolConfirmedBreakout     动量: 多头毛利   107.0 空头毛利  -245.0 | 反向(反转)净利  -209.27 交易 1736
  VolWeightedCloseThrust   动量: 多头毛利  -547.0 空头毛利  -898.0 | 反向(反转)净利   389.78 交易 5278
  SqueezeReleaseMomentum   动量: 多头毛利  -143.0 空头毛利  -447.0 | 反向(反转)净利  -383.49 交易 4864
  VWCT 分年度 IC: {2009: -0.033, 2010: -0.06, 2011: -0.055, 2012: -0.046, 2013: -0.04, 2014: -0.046, 2015: -0.039, 2016: -0.04, 2017: -0.033, 2018: -0.016, 2019: -0.026, 2020: -0.032, 2021: -0.033, 2022: -0.023, 2023: -0.016, 2024: -0.025, 2025: -0.014, 2026: -0.02}

== 30MIN  OOS buy&hold 1oz: 2768 USD
  VolConfirmedBreakout     动量: 多头毛利   256.0 空头毛利    92.0 | 反向(反转)净利  -518.84 交易 853
  VolWeightedCloseThrust   动量: 多头毛利   -20.0 空头毛利  -385.0 | 反向(反转)净利  -111.59 交易 2581
  SqueezeReleaseMomentum   动量: 多头毛利   216.0 空头毛利   -85.0 | 反向(反转)净利  -742.56 交易 3056
  VWCT 分年度 IC: {2009: -0.022, 2010: -0.049, 2011: -0.039, 2012: -0.042, 2013: -0.023, 2014: -0.036, 2015: -0.045, 2016: -0.029, 2017: 0.016, 2018: 0.004, 2019: -0.007, 2020: -0.03, 2021: -0.025, 2022: -0.013, 2023: -0.017, 2024: -0.017, 2025: 0.001, 2026: -0.012}

== 1H  OOS buy&hold 1oz: 2772 USD
  VolConfirmedBreakout     动量: 多头毛利   478.0 空头毛利   551.0 | 反向(反转)净利  -1124.5 交易 476
  VolWeightedCloseThrust   动量: 多头毛利   619.0 空头毛利  -494.0 | 反向(反转)净利  -439.42 交易 1574
  SqueezeReleaseMomentum   动量: 多头毛利   356.0 空头毛利   145.0 | 反向(反转)净利  -807.83 交易 1533

== 4H  OOS buy&hold 1oz: 2772 USD
  VolConfirmedBreakout     动量: 多头毛利  -109.0 空头毛利  -965.0 | 反向(反转)净利  1054.36 交易 96
  VolWeightedCloseThrust   动量: 多头毛利   869.0 空头毛利    17.0 | 反向(反转)净利  -965.09 交易 395
  SqueezeReleaseMomentum   动量: 多头毛利   422.0 空头毛利   671.0 | 反向(反转)净利 -1187.77 交易 472

== 1D  OOS buy&hold 1oz: 2747 USD
  VolConfirmedBreakout     动量: 多头毛利    17.0 空头毛利  -240.0 | 反向(反转)净利    214.7 交易 38
  VolWeightedCloseThrust   动量: 多头毛利    83.0 空头毛利  -414.0 | 反向(反转)净利   314.13 交易 83
  SqueezeReleaseMomentum   动量: 多头毛利   194.0 空头毛利   240.0 | 反向(反转)净利   -444.7 交易 55
```

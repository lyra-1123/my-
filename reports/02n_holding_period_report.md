# 持有期N搜索报告(信号设计步骤3-1)

## 方法

第一次用真实的入场出场规则打分(此前所有walk-forward都是用fwd_return在触发的那根bar直接算固定horizon收益，不是真实的交易模拟)：**下一根bar开盘价入场**，**持有到第N根bar收盘价出场**(暂不设止损止盈，止损止盈是下一步02o的工作)。对8个单因子+9对组合各自在N∈(6, 12, 24, 48, 96, 192)(根M5，即30min~16h)网格上搜索，每个候选自己选自己的最优N(不强制共用一个N)。

## 完整扫描结果

| 候选 | N(根/小时) | OOS Sharpe(年化) | 年触发 | 折数为正 |
|---|---|---|---|---|
| vol_of_vol_100 | 6/0.5h | +0.59 | 640 | 5/5 |
| vol_of_vol_100 | 12/1.0h | +0.70 | 640 | 5/5 |
| vol_of_vol_100 | 24/2.0h | +0.75 | 640 | 5/5 |
| vol_of_vol_100 | 48/4.0h | +0.40 | 640 | 3/5 |
| vol_of_vol_100 | 96/8.0h | +0.35 | 640 | 4/5 |
| vol_of_vol_100 | 192/16.0h | +0.33 | 640 | 4/5 |
| adx_50_pctrank24000 | 6/0.5h | +0.25 | 865 | 3/5 |
| adx_50_pctrank24000 | 12/1.0h | +0.27 | 865 | 3/5 |
| adx_50_pctrank24000 | 24/2.0h | +0.10 | 865 | 2/5 |
| adx_50_pctrank24000 | 48/4.0h | +0.62 | 865 | 4/5 |
| adx_50_pctrank24000 | 96/8.0h | +0.54 | 865 | 4/5 |
| adx_50_pctrank24000 | 192/16.0h | +0.33 | 865 | 3/5 |
| bb_width_100 | 6/0.5h | +0.74 | 525 | 5/5 |
| bb_width_100 | 12/1.0h | +0.53 | 525 | 3/5 |
| bb_width_100 | 24/2.0h | +0.07 | 525 | 2/5 |
| bb_width_100 | 48/4.0h | +0.04 | 525 | 2/5 |
| bb_width_100 | 96/8.0h | -0.20 | 525 | 2/5 |
| bb_width_100 | 192/16.0h | -0.30 | 525 | 2/5 |
| bb_width_300_pctrank6000 | 6/0.5h | -0.06 | 231 | 3/5 |
| bb_width_300_pctrank6000 | 12/1.0h | +0.24 | 231 | 4/5 |
| bb_width_300_pctrank6000 | 24/2.0h | +0.38 | 231 | 3/5 |
| bb_width_300_pctrank6000 | 48/4.0h | +0.34 | 231 | 4/5 |
| bb_width_300_pctrank6000 | 96/8.0h | +0.25 | 231 | 3/5 |
| bb_width_300_pctrank6000 | 192/16.0h | +0.09 | 231 | 3/5 |
| parkinson_vol_100_pctrank24000 | 6/0.5h | +0.33 | 460 | 4/5 |
| parkinson_vol_100_pctrank24000 | 12/1.0h | +0.36 | 460 | 4/5 |
| parkinson_vol_100_pctrank24000 | 24/2.0h | +0.03 | 460 | 3/5 |
| parkinson_vol_100_pctrank24000 | 48/4.0h | -0.50 | 460 | 0/5 |
| parkinson_vol_100_pctrank24000 | 96/8.0h | -0.62 | 460 | 0/5 |
| parkinson_vol_100_pctrank24000 | 192/16.0h | -0.41 | 460 | 1/5 |
| garman_klass_vol_20_pctrank6000 | 6/0.5h | +0.42 | 1526 | 3/5 |
| garman_klass_vol_20_pctrank6000 | 12/1.0h | +0.29 | 1526 | 4/5 |
| garman_klass_vol_20_pctrank6000 | 24/2.0h | +0.19 | 1526 | 4/5 |
| garman_klass_vol_20_pctrank6000 | 48/4.0h | -0.07 | 1526 | 4/5 |
| garman_klass_vol_20_pctrank6000 | 96/8.0h | -0.57 | 1526 | 2/5 |
| garman_klass_vol_20_pctrank6000 | 192/16.0h | -0.56 | 1526 | 1/5 |
| parkinson_vol_60_pctrank6000 | 6/0.5h | +0.57 | 745 | 4/5 |
| parkinson_vol_60_pctrank6000 | 12/1.0h | +0.58 | 745 | 4/5 |
| parkinson_vol_60_pctrank6000 | 24/2.0h | -0.11 | 745 | 3/5 |
| parkinson_vol_60_pctrank6000 | 48/4.0h | -0.80 | 745 | 1/5 |
| parkinson_vol_60_pctrank6000 | 96/8.0h | -0.75 | 745 | 1/5 |
| parkinson_vol_60_pctrank6000 | 192/16.0h | -0.24 | 745 | 2/5 |
| garman_klass_vol_240 | 6/0.5h | +0.20 | 189 | 4/5 |
| garman_klass_vol_240 | 12/1.0h | +0.37 | 189 | 4/5 |
| garman_klass_vol_240 | 24/2.0h | +0.29 | 189 | 4/5 |
| garman_klass_vol_240 | 48/4.0h | +0.22 | 189 | 3/5 |
| garman_klass_vol_240 | 96/8.0h | +0.61 | 189 | 4/5 |
| garman_klass_vol_240 | 192/16.0h | +0.03 | 189 | 2/5 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | 6/0.5h | +0.21 | 21 | 3/5 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | 12/1.0h | +0.49 | 21 | 4/5 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | 24/2.0h | +0.61 | 21 | 5/5 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | 48/4.0h | +0.52 | 21 | 5/5 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | 96/8.0h | +0.34 | 21 | 4/5 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | 192/16.0h | +0.38 | 21 | 3/5 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | 6/0.5h | +0.37 | 24 | 4/5 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | 12/1.0h | +0.53 | 24 | 4/5 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | 24/2.0h | +0.65 | 24 | 4/5 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | 48/4.0h | +0.35 | 24 | 3/5 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | 96/8.0h | +0.01 | 24 | 3/5 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | 192/16.0h | -0.01 | 24 | 2/5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | 6/0.5h | +0.50 | 46 | 5/5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | 12/1.0h | +0.53 | 46 | 5/5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | 24/2.0h | +0.32 | 46 | 3/5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | 48/4.0h | +0.15 | 46 | 3/5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | 96/8.0h | +0.23 | 46 | 3/5 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | 192/16.0h | +0.56 | 46 | 3/5 |
| realized_vol_100_pctrank2000+roc_10 | 6/0.5h | +0.05 | 38 | 3/5 |
| realized_vol_100_pctrank2000+roc_10 | 12/1.0h | +0.39 | 38 | 4/5 |
| realized_vol_100_pctrank2000+roc_10 | 24/2.0h | +0.36 | 38 | 4/5 |
| realized_vol_100_pctrank2000+roc_10 | 48/4.0h | -0.14 | 38 | 3/5 |
| realized_vol_100_pctrank2000+roc_10 | 96/8.0h | -0.01 | 38 | 3/5 |
| realized_vol_100_pctrank2000+roc_10 | 192/16.0h | -0.01 | 38 | 3/5 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | 6/0.5h | +0.20 | 21 | 3/5 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | 12/1.0h | +0.26 | 21 | 3/5 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | 24/2.0h | +0.11 | 21 | 3/5 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | 48/4.0h | +0.00 | 21 | 2/5 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | 96/8.0h | +0.36 | 21 | 2/5 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | 192/16.0h | +0.38 | 21 | 4/5 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | 6/0.5h | +0.28 | 52 | 3/5 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | 12/1.0h | +0.35 | 52 | 4/5 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | 24/2.0h | -0.20 | 52 | 1/5 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | 48/4.0h | -0.02 | 52 | 2/5 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | 96/8.0h | +0.02 | 52 | 4/5 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | 192/16.0h | +0.16 | 52 | 4/5 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | 6/0.5h | +0.03 | 35 | 3/5 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | 12/1.0h | +0.27 | 35 | 3/5 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | 24/2.0h | +0.41 | 35 | 3/5 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | 48/4.0h | +0.07 | 35 | 3/5 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | 96/8.0h | -0.35 | 35 | 2/5 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | 192/16.0h | -0.21 | 35 | 2/5 |
| avg_gap_50+roc_10 | 6/0.5h | +0.37 | 57 | 4/5 |
| avg_gap_50+roc_10 | 12/1.0h | +0.24 | 57 | 3/5 |
| avg_gap_50+roc_10 | 24/2.0h | +0.09 | 57 | 3/5 |
| avg_gap_50+roc_10 | 48/4.0h | -0.12 | 57 | 3/5 |
| avg_gap_50+roc_10 | 96/8.0h | +0.00 | 57 | 2/5 |
| avg_gap_50+roc_10 | 192/16.0h | +0.06 | 57 | 4/5 |
| stochastic_d_100_pctrank2000+keltner_width_20 | 6/0.5h | +0.51 | 24 | 5/5 |
| stochastic_d_100_pctrank2000+keltner_width_20 | 12/1.0h | +0.12 | 24 | 3/5 |
| stochastic_d_100_pctrank2000+keltner_width_20 | 24/2.0h | +0.13 | 24 | 2/5 |
| stochastic_d_100_pctrank2000+keltner_width_20 | 48/4.0h | -0.02 | 24 | 2/5 |
| stochastic_d_100_pctrank2000+keltner_width_20 | 96/8.0h | +0.01 | 24 | 3/5 |
| stochastic_d_100_pctrank2000+keltner_width_20 | 192/16.0h | +0.12 | 24 | 3/5 |

## 每个候选选出的最优N

| 候选 | 类型 | 最优N(根/小时) | OOS Sharpe(年化) | 年触发 | 达标 |
|---|---|---|---|---|---|
| vol_of_vol_100 | single | 24/2.0h | +0.75 | 640 | 是 |
| adx_50_pctrank24000 | single | 48/4.0h | +0.62 | 865 | 是 |
| bb_width_100 | single | 6/0.5h | +0.74 | 525 | 是 |
| bb_width_300_pctrank6000 | single | 24/2.0h | +0.38 | 231 | 是 |
| parkinson_vol_100_pctrank24000 | single | 12/1.0h | +0.36 | 460 | 是 |
| garman_klass_vol_20_pctrank6000 | single | 6/0.5h | +0.42 | 1526 | 是 |
| parkinson_vol_60_pctrank6000 | single | 12/1.0h | +0.58 | 745 | 是 |
| garman_klass_vol_240 | single | 96/8.0h | +0.61 | 189 | 是 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | pair | 24/2.0h | +0.61 | 21 | 是 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | pair | 24/2.0h | +0.65 | 24 | 是 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | pair | 192/16.0h | +0.56 | 46 | 是 |
| realized_vol_100_pctrank2000+roc_10 | pair | 12/1.0h | +0.39 | 38 | 是 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | pair | 192/16.0h | +0.38 | 21 | 是 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | pair | 12/1.0h | +0.35 | 52 | 是 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | pair | 24/2.0h | +0.41 | 35 | 是 |
| avg_gap_50+roc_10 | pair | 6/0.5h | +0.37 | 57 | 是 |
| stochastic_d_100_pctrank2000+keltner_width_20 | pair | 6/0.5h | +0.51 | 24 | 是 |

完整数据：reports/02n_holding_period_scan.csv；最优N汇总：reports/02n_holding_period_winners.csv（02o止损止盈搜索会读取这份文件）。


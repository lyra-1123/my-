# 完整流程Walk-Forward验证报告(信号设计步骤6)

## 方法

02n/02o/02p选持有期N和分方向止损止盈时，是看全部5折汇总的OOS表现选一次、然后固定用到所有折——这本身是一种轻度"偷看"（用来评分的OOS折同时也是用来选参数的折）。这一步把N和止损止盈的选择**搬到每一折内部**：每一折只用该折的训练区间自己选N、再选多空各自的止损止盈，choices冻结后只应用到该折的测试区间，5折汇总。窗口(单因子用02k优化窗口/组合用02i原窗口)不在这一步重新开放搜索(视为已经在02m单独做过PBO检验的特征工程决定)。

## 结果：真正嵌套walk-forward vs 之前"看全部折选一次"的对比

| 候选 | 之前(偷看)年化Sharpe | 嵌套walk-forward年化Sharpe | 差距 | 折数为正 | 是否通过筛选 |
|---|---|---|---|---|---|
| avg_gap_50+garman_klass_vol_20_pctrank500 | +0.65 | +0.67 | +0.02 | 2/5 | 否 |
| adx_50_pctrank24000 | +0.62 | +0.60 | -0.02 | 4/5 | 是 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | +0.53 | +0.58 | +0.05 | 3/5 | 是 |
| garman_klass_vol_20_pctrank6000 | +0.58 | +0.57 | -0.01 | 4/5 | 是 |
| vol_of_vol_100 | +0.92 | +0.51 | -0.42 | 5/5 | 是 |
| stochastic_d_100_pctrank2000+keltner_width_20 | +0.75 | +0.46 | -0.29 | 5/5 | 是 |
| bb_width_300_pctrank6000 | +0.46 | +0.44 | -0.02 | 4/5 | 是 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | +0.78 | +0.37 | -0.42 | 3/5 | 是 |
| avg_gap_50+roc_10 | +0.50 | +0.36 | -0.13 | 4/5 | 是 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | +0.76 | +0.31 | -0.45 | 4/5 | 是 |
| garman_klass_vol_240 | +0.61 | +0.17 | -0.44 | 3/5 | 是 |
| bb_width_100 | +0.74 | +0.10 | -0.64 | 3/5 | 是 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | +0.64 | +0.07 | -0.56 | 4/5 | 是 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | +0.63 | -0.25 | -0.88 | 3/5 | 否 |
| realized_vol_100_pctrank2000+roc_10 | +0.39 | -0.26 | -0.66 | 3/5 | 否 |
| parkinson_vol_100_pctrank24000 | +0.48 | -0.38 | -0.86 | 3/5 | 否 |
| parkinson_vol_60_pctrank6000 | +0.65 | -0.41 | -1.06 | 4/5 | 否 |

**12/17个候选在真正嵌套的walk-forward下仍然通过筛选**(OOS Sharpe>0 + >=3/5折为正 + 触发频率达标)。

## 参数稳定性：每折自己选出的N和止损止盈是否一致

见完整数据(reports/02s_fold_choices.csv)——如果同一个候选5折选出的N/止损类型来回跳变，说明之前"看全部折选一次"的参数本身就不稳定，只是偶然在全样本上表现好。

## 结论与下一步

- 完整数据：reports/02s_nested_walk_forward_results.csv（每候选结果）、reports/02s_fold_choices.csv（每折的参数选择）。
- 下一步（步骤7）：从这一步真正OOS验证过的结果，反推出正式的入场出场条件(不再是每折不同的候选参数，而是给出一套用于实盘/阶段3b的最终推荐参数)。


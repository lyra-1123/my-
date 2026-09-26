# 净头寸互锁策略实现与验证报告(信号设计步骤5b)

## 方法

实现用户选定的互锁策略：**净头寸相抵，反向信号先平仓**——同方向的新触发在已有仓位存活期内被忽略(不加仓)；反方向的新触发立即把当前仓位强制平仓(按新触发那根bar的开盘价)，然后开新方向的仓位。单个候选自己的多空重叠、和多个候选一起跑的组合层面冲突，都用同一套`simulate_net_position`逻辑处理(用户要求两者policy一致)。

对比口径："基线"=假装每笔交易都能各自独立跑完全程、互不干扰(02o/02p用的口径)；"净头寸"=按上面的互锁规则实际会发生的结果(有的交易被反向信号提前砍仓)。都用全样本固定阈值(不是walk-forward逐折)，年化用交易笔数/12.2年折算。

## 单个候选：自身反手修正前后对比

| 候选 | 基线年化Sharpe | 净头寸年化Sharpe | 强制平仓笔数/总笔数 |
|---|---|---|---|
| garman_klass_vol_20_pctrank6000 | +1.01 | +1.01 | 0/20244 |
| adx_50_pctrank2000+autocorr_returns_50_pctrank2000 | +0.87 | +0.87 | 0/292 |
| parkinson_vol_100_pctrank24000 | +0.89 | +0.80 | 0/6437 |
| avg_gap_50+roc_10 | +0.77 | +0.76 | 0/759 |
| vol_of_vol_100 | +0.70 | +0.74 | 147/9535 |
| parkinson_vol_100_pctrank2000+aroon_up_10_pctrank500 | +0.70 | +0.73 | 0/776 |
| parkinson_vol_60_pctrank6000 | +0.87 | +0.73 | 1/9819 |
| bb_width_300_pctrank6000 | +0.75 | +0.68 | 0/3305 |
| kurt_returns_100+mean_reversion_speed_50_pctrank500 | +0.65 | +0.62 | 9/602 |
| realized_vol_100_pctrank2000+roc_10 | +0.58 | +0.58 | 0/556 |
| avg_gap_50+zscore_vs_ma_100_pctrank500 | +0.59 | +0.58 | 0/484 |
| garman_klass_vol_240 | +0.43 | +0.51 | 3/2564 |
| adx_50_pctrank24000 | +0.57 | +0.42 | 420/11420 |
| avg_gap_50+garman_klass_vol_20_pctrank500 | +0.40 | +0.37 | 0/263 |
| bb_width_100 | +0.33 | +0.31 | 0/7792 |
| realized_vol_100_pctrank2000+variance_ratio_2_50_pctrank2000 | +0.37 | +0.31 | 3/304 |
| stochastic_d_100_pctrank2000+keltner_width_20 | +0.22 | +0.22 | 0/319 |

## 组合层面：多个候选一起跑，净头寸互锁前后对比

| 组合范围 | 朴素合并年化Sharpe(忽略冲突) | 净头寸年化Sharpe(实际会发生的) | 强制平仓笔数/总笔数 |
|---|---|---|---|
| 8个单因子 | +1.77 | +1.35 | 10597/51194 |
| 9对组合 | +1.37 | +1.25 | 152/3878 |
| 全部17个 | +2.14 | +1.54 | 11481/52830 |

## 结论与下一步

- 完整数据：reports/02r_self_net_position.csv（单候选）、reports/02r_ensemble_net_position.csv（组合层面）。
- 这套互锁逻辑(`src/factors/execution.py::simulate_net_position`)是可复用的基础设施，阶段3b把多个候选接入真实马丁引擎时可以直接复用同一个函数。
- 下一步（步骤6）：Walk-Forward验证——把目前这套(窗口+N+分方向止损止盈+互锁策略)完整流程放进真正的多折walk-forward里再走一遍，而不是像这一步一样用全样本固定阈值。


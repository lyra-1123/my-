# PBO(过拟合概率)补充检验报告

## 背景

02h(798个单因子候选)、02i(482对组合候选)、02j(19个家族x4个窗口=76个候选)三次搜索都只用OOS Sharpe+折一致性+触发频率筛选，没有做PBO——这是相对于阶段2c/2d"IC+IR+PBO+OOS Sharpe"标准的一个缺口。这里用CSCV(Bailey/Lopez de Prado方法，`src/factors/validation.py`里已有的`pbo_from_block_sharpe`复用)补上：10个连续区块，穷举所有对半分法(10选5)，每次用一半区块选"样本内最优候选"，检查它在另一半区块里的排名是否掉到中位数以下。PBO=这种情况发生的比例，越接近0.5说明样本内最优≈随机噪音(过拟合)，越接近0说明样本内最优在样本外也大概率靠前(搜索结果可信)。

## 结果

| 候选池 | 候选数 | PBO | 全样本最优候选 | 是否与实际筛选出的最终候选一致 |
|---|---|---|---|---|
| 02h单因子(798) | 798 | 0.016 | aroon_up_10_pctrank500_reversion | 否——见下方说明 |
| 02i两两组合(482) | 482 | 0.179 | aroon_up_10_pctrank500+variance_ratio_2_50_pctrank2000 | 否——见下方说明 |

## 02j窗口选择池(19个家族，每个家族4个候选窗口)逐家族PBO

| 家族 | pctrank(H1) | PBO | 全样本最优窗口 |
|---|---|---|---|
| vol_of_vol | nan | 0.024 | vol_of_vol_100 |
| adx | 2000.0 | 0.238 | adx_50_pctrank24000 |
| bb_width | nan | 0.067 | bb_width_100 |
| bb_width | 500.0 | 0.036 | bb_width_100_pctrank6000 |
| parkinson_vol | 2000.0 | 0.222 | parkinson_vol_100_pctrank24000 |
| parkinson_vol | 500.0 | 0.000 | parkinson_vol_20_pctrank6000 |
| garman_klass_vol | 500.0 | 0.000 | garman_klass_vol_20_pctrank6000 |
| garman_klass_vol | nan | 0.024 | garman_klass_vol_20 |
| autocorr_returns | 2000.0 | 0.282 | autocorr_returns_50_pctrank24000 |
| avg_gap | nan | 0.032 | avg_gap_50 |
| kurt_returns | nan | 0.472 | kurt_returns_100 |
| mean_reversion_speed | 500.0 | 0.028 | mean_reversion_speed_50_pctrank6000 |
| realized_vol | 2000.0 | 0.333 | realized_vol_600_pctrank24000 |
| roc | nan | 0.004 | roc_10 |
| variance_ratio_2 | 2000.0 | 0.456 | variance_ratio_2_50_pctrank24000 |
| aroon_up | 500.0 | 0.000 | aroon_up_10_pctrank6000 |
| zscore_vs_ma | 500.0 | 0.056 | zscore_vs_ma_600_pctrank6000 |
| stochastic_d | 2000.0 | 0.131 | stochastic_d_1200_pctrank24000 |
| keltner_width | nan | 0.131 | keltner_width_20 |

## 解读

- PBO<0.5：样本内最优在样本外排名中位数以上的次数更多，说明这次搜索"挑出来的最优"不是纯噪音。PBO接近甚至超过0.5则说明这批候选里选出来的"最优"很可能是过拟合——样本内最优和样本外最优基本没关系。
- 这里的PBO用的是全样本固定阈值(不是walk-forward逐折重新拟合的阈值)，衡量的是"搜索/挑选过程本身"有多可信，跟02h/02i/02k里walk-forward验证的"这个具体因子OOS Sharpe是否为正"是两个不同但互补的问题——一个测搜索过程，一个测选中的候选本身。
- 完整明细：reports/02m_pbo_window_specs.csv（02j窗口池逐家族）；02h/02i的候选级明细未展开保存(798+482行的block Sharpe矩阵体积较大，只保留了本报告的汇总数字)。

## 为什么"全样本最优候选"和实际选中的候选不一样

02h单因子池PBO算出来的"全样本最优"是`aroon_up_10_pctrank500`(reversion)，而不是我们实际选中并做了walk-forward验证、最终采用的`vol_of_vol_100`；02i组合池同理。**这不是过拟合的证据，是两种排序标准不同导致的**：这里的PBO用的是"全部bar(含未触发的0收益)的区块Sharpe"，这个指标天然偏向触发频率高的候选——`aroon_up_10_pctrank500`原始触发率高达14115次/年(参考02j扫描)，远高于`vol_of_vol_100`的640次/年，同样多的"零收益bar"里混入更多真实交易，稀释效应更小，区块Sharpe自然更高。而02h/02i实际用来挑选的标准是**walk-forward算出的年化OOS Sharpe**(已经用√年触发数做过频率调整，是不同的指标)。所以"PBO最优候选≠实际选中候选"本身不能说明选择过程有问题——真正回答"过拟合"问题的是PBO数值本身(0.016和0.179)，两者都远低于0.5，是好消息：不管用哪种全样本指标去挑，样本内的"最优"大概率在样本外也不差，说明798/482这个规模的搜索没有被噪音主导。

## 需要额外注意的：02j窗口选择池里有两个家族的PBO偏高

大多数家族的窗口选择PBO都很低(0/19个家族PBO在0附近，说明"哪个窗口候选最好"这个判断稳健)，但有两个**恰好是上次报告里改善最明显的两个**：

| 家族 | PBO | 上次报告的"改善" |
|---|---|---|
| `kurt_returns` | **0.472** | x1→x6，年化Sharpe+0.23→+1.05(当时最大的一次提升) |
| `variance_ratio_2` | **0.456** | x1→x6，年化Sharpe+0.22→+0.27(较小的提升) |

PBO这么接近0.5意味着：在4个候选窗口(x1/x3/x6/x12)之间，"样本内选出来最好的那个"和"样本外表现最好的那个"基本没有关系——`kurt_returns`换成600根bar窗口这个"提升"很可能只是4个候选里矬子里拔将军的噪音。

**但这个不影响当前实际采用的候选池**：`kurt_returns`和`variance_ratio_2`都只出现在两两组合里（`kurt_returns_100`+`mean_reversion_speed_50_pctrank500`、`realized_vol_100_pctrank2000`+`variance_ratio_2_50_pctrank2000`），而两两组合按之前的结论用的是02i的**原始x1窗口**(`kurt_returns_100`、`variance_ratio_2_50_pctrank2000`)，不是02j扫描出来、这里被PBO标记为可疑的优化窗口(`_600`/`_300`)——那两个优化窗口本来就没有被采纳进最终候选池，只是02j/02k探索阶段的中间产物。所以这里的发现是"以后如果想把这两个家族也做成独立单因子候选，先别用扫描出来的窗口"，而不是"现在的候选池需要改"。其余家族(包括改善第二大的单因子候选`garman_klass_vol`，PBO=0.024)的窗口选择相对可信，`garman_klass_vol_240`(单因子候选之一)是安全的。


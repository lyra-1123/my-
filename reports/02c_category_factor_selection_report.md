# 分类别因子最终筛选报告（阶段2c）

## 方法

阶段2b是先在每个指标家族内选“全样本IC最强”的代表变体，再筛这个代表变体——问题是全样本IC最强的变体不一定是样本外Sharpe最好的变体。这一步改成**直接在四大类别（均值回归/动量/波动率/价格行为）下的全部314个变体里逐个测**（每个变体在4小时和8小时上都算IR和5折walk-forward样本外Sharpe），通过标准是**OOS Sharpe在4小时和8小时上同时为正**（硬指标），IR用来在候选里排序（不是硬性门槛，只是次要参考）。

**均值回归类一开始颗粒无收**：原本7个家族(RSI/z-score/stochastic/CCI/Donchian/Williams %R)一共84个变体，没有一个能同时通过——这些指标本质上都是“现在偏离均值多远”，彼此高度相关（比如donchian_position、stochastic_k、williams_r数学上是同一个量的仿射变换），单纯换窗口/百分位排名解决不了“这一类指标本身缺乏regime预测力”的问题。所以新增了两个结构不同的构造：`ma_cross_count`(价格穿越均线的**频率**，不是距离)和`mean_reversion_speed`(滚动AR(1)回归估计偏离均值后被拉回的**速度**，不是当前偏离了多少)——后者奏效了。

## 各类别最终候选

| 分类 | 家族 | 最优变体 | IR(4h) | IR(8h) | OOS Sharpe年化(4h) | OOS Sharpe年化(8h) |
|---|---|---|---|---|---|---|
| mean_reversion | mean_reversion_speed | `mean_reversion_speed_20` | 0.343 | 0.506 | +0.253 | +0.250 |
| momentum | autocorr_returns | `autocorr_returns_100` | 0.382 | 0.431 | +0.489 | +0.075 |
| volatility | vol_of_vol | `vol_of_vol_10_pctrank500` | 0.228 | 0.255 | +0.180 | +0.147 |
| price_action | dist_from_high | `dist_from_high_20_pctrank500` | 0.246 | 0.383 | +0.208 | +0.054 |

## 每个类别通过“OOS Sharpe两个horizon都为正”的候选数量（体现搜索的穷尽程度）

| 分类 | 测试变体数 | 双horizon OOS Sharpe均为正的变体数 |
|---|---|---|
| mean_reversion | 108 | 8 |
| momentum | 61 | 8 |
| volatility | 84 | 1 |
| price_action | 61 | 10 |

## 结论与下一步

- 四大类别现在**全部有验证过的候选**：`mean_reversion_speed_20`（均值回归）、`autocorr_returns_100`（动量）、`vol_of_vol_10_pctrank500`（波动率）、`dist_from_high_20_pctrank500`（价格行为）。
- `mean_reversion_speed_20`是这轮新加的构造，在两个horizon上的IR(0.34/0.51)和OOS Sharpe(+0.25/+0.25)在全项目里都算表现均衡的（4小时和8小时量级接近，不像有些因子在一个horizon上很强、另一个转负）——说明“均值回归速度”这个角度是有效的，“均值回归距离”（RSI/z-score那一类）这个角度对regime分类没用，这是个有意义的区分。
- 这4个候选都只在“因子本身+简化测试床”层面验证过，阶段3要把它们接入真实马丁引擎重新验证；分类别覆盖只是保证“思路多样性”，不代表4个都要用——阶段3可以先用4个都测一遍，再看哪个（或哪几个组合）对真实回测的最大回撤/破产概率改善最大。
- 完整的314变体×2horizon扫描数据在`reports/02c_category_variant_scan.csv`。


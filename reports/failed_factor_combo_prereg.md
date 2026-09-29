# 预登记：未通过 ICIR 筛选的因子，按逻辑分组组合 + 仓位分配（2026-09-29）

> 本文件在跑任何组合回测之前写好并提交，之后不修改。脚本：`research/failed_factor_combo.py`；结果：`reports/failed_factor_combo.md`。
> 在跑策略（TT30-EW-v1、HA1H-v1、HA1H-TS2-shadow、MF30-EW-shadow）的代码、参数、指纹一律不动；本研究只新增研究脚本和报告。

## 0. 动机与先验预期
- 用户要求：之前 ICIR 不达标（月度 |t|<5 或样本内外 ICIR 反号）的因子，按逻辑（均值回归、动量突破、区间反转等）分组、枚举组合，并加上仓位分配与管理，看能否组合出可用的策略。
- 先验：单个因子 ICIR 不显著，说明信息弱或不稳定。如果各成分的噪声互相独立、信息同向，等权组合能提高 ICIR（分散化）；如果成分本身没有信息，组合只会把噪声加在一起。已有证据偏向后者：L14（相关低 ≠ 有独立信息）、L26（换指标写法不带来新信息）、L27（研究规模扩大后 DSR 普遍下降）。**预期大概率不通过**，但按用户要求做一次完整、诚实的检验。

## 1. 假设
- **H1（分散化）**：同一逻辑桶内，未达标因子等权组合后，样本内（2009-2019）月度 ICIR 高于桶内成分 ICIR 的中位数。
- **H2（可交易）**：存在一个"桶子集 × 仓位规则"，在样本内选出后，样本外（2020-2026.09）ATR·今 夏普 ≥ 0.3，且多空两侧都盈利。
- **H3（组合增益）**：选出的策略与 TT30-EW-v1、HA1H-v1 的日度收益相关 < 0.5，加入后组合夏普在样本内和样本外都上升。

## 2. 成分池（固定，不再增删）
- 频率：**30MIN、1H**（≤15MIN 付不起点差，L2/L15/L25；4H/1D 样本少、过夜费高，L5/L12）。
- 入池条件：因子库中该"因子×频率"**未通过** ICIR 筛选（月度 |t|<5，或样本内外 ICIR 反号），且 t 可计算。
- 排除：对照组因子（ThreePushNoDecay、FirstEntryH1、RangeBreakoutNoPole）；t 无法计算的稀疏事件因子（FlagBreakout）。
- 方向：一律使用因子自身假设的方向（代码里 >0 = 看多），**不按样本外结果翻转**。
- 说明：筛选条件本身用到了样本外 ICIR 的符号。入池的是"失败"的因子，这种偏差对样本外表现是不利的（偏保守），不会抬高结果。

### 逻辑桶（按因子假设事先划分）
| 桶 | 逻辑 | 30MIN 成员 | 1H 成员 |
|---|---|---|---|
| MR 均值回归 | 偏离后回归、超跌/超买反转 | AsiaSessionReversion, UptrendDipReversion, VolumeClimaxReversal, WeekendGapReversion | AsiaSessionReversion, CCIReversion, MFIReversion, StochReversion, RSI2PullbackInTrend, UptrendDipReversion, VolumeClimaxReversal, WeekendGapReversion |
| MB 动量突破 | 突破/动量延续 | HTFTrendLTFBreakout, LondonNYSessionMomentum, MACDMomentum, SqueezeReleaseMomentum, TRIXMomentum, VolConfirmedBreakout | AroonTrend, CMFFlow, ForceIndex, HTFTrendLTFBreakout, IchimokuTrend, LondonNYSessionMomentum, MACDMomentum, OBVMomentum, PSARTrend, SqueezeReleaseMomentum, TRIXMomentum, VolConfirmedBreakout, VortexTrend |
| RR 区间反转 | 震荡区间/形态力竭反转 | RangeVWAPReversion, ThreePushWedgeReversal | RangeVWAPReversion, ThreePushWedgeReversal |
| PB 趋势回调 | 顺大逆小 | PullbackSwing, SecondEntryH2 | MTFPullbackResonance, PullbackSwing, SecondEntryH2, TrendPullbackLowVolume |

## 3. 枚举（全部登记进 reports/rule_trials.csv，不论好坏）
信号构造：桶信号 = MAD_Z(桶内成分等权平均, norm)；组合信号 = MAD_Z(所选桶信号等权平均, norm)；norm 用 FREQ_PRESETS 先验值；缺失按 0。

**信号配置（每个频率 17 个，共 34 个）**
- 15 个桶子集：{MR, MB, RR, PB} 的全部非空子集；
- 2 个状态切换：ER60 = |C−C₋₆₀| / Σ₆₀|ΔC|，状态阈值 = ER60 过去 1000 根的滚动中位数（shift 1，无未来函数）：
  - G1 教科书：趋势状态用 mean(MB, PB)，区间状态用 mean(MR, RR)；
  - G2 反向（L30：黄金在区间后突破延续、趋势末段回吐）：趋势状态用 mean(MR, RR)，区间状态用 mean(MB, PB)。

**仓位规则（在信号配置上叠加）**
- S0 固定：迟滞 1.5/0.3，±1 单位（统一执行规则，换日前平仓）；
- S1 强度分层：迟滞状态同 S0；持仓期间 |z| ≥ 2.5（同向）时 2 单位，否则 1 单位；每单位变化收半个点差；
- S2 桶独立交易后净额合并（仅多桶子集和 G1/G2 之外的 11 个子集）：每个桶各自按迟滞 1.5/0.3 交易，仓位 = 各桶仓位的平均（反向相抵），按净仓位变化收点差。

试验数：34 (S0) + 34 (S1) + 22 (S2) = **90**。

**组合层仓位分配（第二阶段，只对选中的新策略，共 3 个）**：TT30 + HA1H + 新策略
- P0 等权（各 1 单位 R）；P1 逆波动（权重 ∝ 1/样本内日度 R 标准差，只用样本内估计）；P2 新策略半权（0.5）。
- 按样本内组合夏普选择，样本外只跑一次。

## 4. 选择规则（只用样本内 2009-2019）
- 口径：日度 ATR·今 收益（收益/上一根 ATR、逐年去漂移、成本按最近 1 年 ATR 折算；与 research/multifactor_combo.py 相同）。
- 在"样本内与 TT30、HA1H 日度相关都 < 0.5"的配置中，选**样本内夏普最高**的一个。同时报告不加相关性门槛时的第一名。
- 选定后样本外只跑一次。之后不再增加桶、成分、状态定义或仓位规则。

## 5. 判定标准（全部满足才建议作为新的影子/候选版本，是否上模拟盘由用户决定）
| 编号 | 标准 |
|---|---|
| A1 | 样本外 ATR·今 夏普 ≥ 0.3 |
| A2 | 样本外多头、空头的 ATR 收益都 > 0（排除牛市 beta） |
| A3 | 参数平原：选中配置在 entry {1.0,1.5,2.0} × exit {0,0.3,0.6} 的 9 个组合中，样本内 ≥7 个为正、样本外 ≥7 个为正 |
| A4 | CSCV（90 个配置，16 块）PBO < 0.3 |
| A5 | DSR ≥ 0.5，N = 本研究 N_eff + 因子库研究层 N_eff（同时报告加上全部登记试验的保守口径） |
| A6 | 样本内、样本外与 TT30、HA1H 的日度相关都 < 0.5，且加入后组合夏普在样本内、样本外都不下降 |
| A7 | 点差 ×2 时样本外夏普仍 > 0 |

H1 只做诊断（月度 ICIR，样本内），不作为交易判定。

## 6. 不做的事
- 不根据样本外结果改桶划分、成分、方向、阈值或仓位规则；
- 不重测已经否定过的方向（止损/止盈 L21、波动率门槛 L18、反转择时 L16）；
- 不修改 live/、live/config.py、paper/specs.py 中的已登记策略，也不修改公共代码（factors/core.py、factors/evaluate.py）；
- 模拟盘前几天的结果不参与任何判断。

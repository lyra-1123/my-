# 项目进度总览

XAUUSD量化马丁格尔策略：数据清洗 → 因子挖掘 → 回测 → 策略成型 → 多agent审核 → 策略运行 → 复盘。

约定：每个阶段对应一个编号脚本（`scripts/NN_xxx.py`）+ 一份编号报告（`reports/NN_xxx_report.md`），
报告记录该阶段的输入、方法、结论和遗留问题，脚本可重复运行复现该阶段的产出。本文件是索引，
每个阶段完成/更新后同步维护。

| 阶段 | 状态 | 脚本 | 报告 | 一句话结论 |
|---|---|---|---|---|
| 1. 数据清洗 | ✅ 完成 | `scripts/01_build_clean_dataset.py` | `reports/01_data_quality_report.md` | 635万条M1数据，0坏点、0未解释缺口，已产出M1~D1多周期parquet |
| 2. 因子挖掘 | ✅ 完成(v4，新因子类型+全面百分位化+系统性组合搜索) | `scripts/02_factor_mining.py` | `reports/02_factor_mining_report.md` + `reports/02_factor_candidate_pool.csv` | 383个因子中286个通过\|IC\|≥0.01；新增Choppiness Index/Aroon/Parkinson-GK波动率/linreg_r²/avg_gap，其中Choppiness Index表现强(\|IC\|~0.08)且符号与bb_width/adx相反，两者互相印证"波动率/趋势会均值回归"；穷举15个代表因子的两两组合，最优对(bb_width_50+efficiency_ratio_20)把未来ER压低17.7%，好于v1~v3手选组合(~15%)；8因子平均合成打分反而不如两两组合(9.9%<17.7%，简单平均稀释信号)，阶段3不建议用平均合成分数；session/星期几乎无区分力 |
| 2b. 因子分类筛选(IC+IR+PBO+OOS Sharpe) | ✅ 完成(v6，被2c取代) | `scripts/02b_factor_screening.py` | `reports/02b_factor_screening_report.md` | 见提交历史：v1→v6从12→2→11→0→0→4个家族级候选(autocorr_returns/choppiness_index等)。核心局限是"先选家族代表变体、再筛"，代表变体不一定是OOS表现最好的——这个局限在2c里解决了 |
| 2c. 分类别因子最终筛选(变体级搜索) | ✅ 完成 | `scripts/02c_category_factor_selection.py` | `reports/02c_category_factor_selection_report.md` + `reports/02c_category_variant_scan.csv` | 不再"先选家族代表"，直接对均值回归/动量/波动率/价格行为4大类下全部314个变体逐个测(4h+8h的IR+5折walk-forward OOS Sharpe)，通过标准=双horizon OOS Sharpe同时为正。**均值回归类原本7个家族84个变体颗粒无收**（RSI/zscore/stochastic/CCI/Donchian/Williams %R本质都是"距离均值多远"，彼此仿射相关，缺乏regime预测力）——新增`mean_reversion_speed`(AR(1)回归估计的均值回归**速度**，不是距离)后奏效。**四大类最终候选**：均值回归=`mean_reversion_speed_20`(IR 0.34/0.51, OOS Sharpe +0.25/+0.25，全项目里最均衡的因子之一)、动量=`autocorr_returns_100`(IR 0.38/0.43, OOS +0.49/+0.08)、波动率=`vol_of_vol_10_pctrank500`(IR 0.23/0.26, OOS +0.18/+0.15，84个波动率变体里唯一双horizon都过的)、价格行为=`dist_from_high_20_pctrank500`(IR 0.25/0.38, OOS +0.21/+0.05)。这4个是阶段3的正式候选池 |
| 2d. 组合信号设计(投票规则) | ✅ 完成 | `scripts/02d_combined_signal_design.py` | `reports/02d_combined_signal_report.md` + `reports/02d_combined_signal_results.csv` | 把阶段2c的4个类别代表因子做成投票信号("至少K/4票安全"允许开新层)，K=1~4在4h和8h上**全部**OOS Sharpe为正，但强度和激活率此消彼长：K=4最强(4h:+0.59/5折全正，8h:+0.23/4折正)但激活率仅0.7~0.8%(几乎不开单)；K=1激活率54%但8h Sharpe很弱(+0.04)；**K=3是较好的折中**(4h:+0.35/8h:+0.33，两个horizon都4/5折为正，激活率~4.7~4.8%)。这只是简化测试床上的稳健性检验，非严格walk-forward，阶段3b要接入真实马丁引擎(`allow_entry`参数已预留)重新验证 |
| 信号设计-步骤1：跨周期一致性 | ✅ 完成(需返工) | `scripts/02e_cross_timeframe_consistency.py` | `reports/02e_cross_timeframe_consistency_report.md` | 大周期H1(MA20/50方向)+小周期M5(同样MA20/50)，一致全仓/不一致等4根后按H1半仓。**结果：跨周期完整逻辑年化Sharpe+0.31，反而不如直接用H1方向全仓(+0.44)**；仅M5方向单独看是负的(-0.14)，说明M5用MA20/50判断方向本身没有信息量，导致"不一致"状态占49%时间、其中45%被砍成半仓，纯粹拖累一个本来有效的H1信号。结论：问题不在"看大做小"思路本身，而在M5方向定义需要换（或换一对周期）。这些Sharpe未计入点差滑点，且用的是占位持有期(12根M5)，最终参数在步骤3定 |
| 信号设计-步骤2：定义signal_long/short | ✅ 完成(v2) | `scripts/02f_signal_definition.py` | `reports/02f_signal_definition_report.md` | v1(H1，3条件AND：mean_reversion_speed+autocorr_returns+MFI)信号过稀(18年仅6/12次触发)；**v2改到M5周期+去掉最弱的条件A，只用B(autocorr_returns_100高位+上根bar方向)+MFI(20超买超卖回归穿越)两条件AND**，触发频率回升到年均~115次(可用量级)。但edge偏弱且多空不对称：signal_long年化Sharpe约+0.10，signal_short仅+0.02(接近0)——对多头方向明显更有效，可能与样本期金价长期偏多头有关。B/MFI的窗口(100/20根)是直接搬用阶段2c在H1上验证过的bar数，**尚未在M5粒度上重新验证**，且用的是占位持有期(1小时)，步骤3定真实参数后需重新验证 |
| 信号设计：M5方向性因子系统性挖掘 | ✅ 完成(单因子无edge) | `scripts/02g_build_m5_factors.py` + `scripts/02h_directional_factor_mining.py` | `reports/02h_directional_factor_mining_report.md` + 2个scan CSV | 对M5上全部399个因子变体×2种方向化模式(动量延续/反转穿越)=798种组合做系统性挖掘(阶段1全样本粗筛，阶段2对top25做5折walk-forward验证)。**诚实结论：没有一个单因子有像样的方向性edge**——最强的garman_klass_vol_20 OOS Sharpe也只有+0.002(未年化，年化约+0.02，统计上和噪音无法区分)。这不代表步骤2的发现是错的——步骤2用的是autocorr_returns+MFI的AND组合(年化Sharpe~0.10)，本次只测单因子，没测组合；说明和阶段2的regime分类任务一样，方向性edge也主要在因子组合层面，不在单因子层面。下一步应做系统性两两组合搜索，而非继续找单一因子 |
| 3. 回测(基线，无过滤器) | ✅ 完成 | `scripts/03_baseline_backtest.py` | `reports/03_baseline_backtest_report.md` | 长仓ATR网格马丁(初始0.01手/2倍加仓/最多8层/1xATR(14)间距和止盈/$10000初始资金/1:200杠杆)：权益从$10000稳定涨到峰值$104,961(2012-10-09)，随后半年内回撤95%到$4,787，最终被2013年4月中旬黄金历史级暴跌一根H1 bar打出-$105,418强平，账户**破产**(2013-04-15)，此后不再交易。验证了马丁格尔的核心风险：稳定盈利可以持续数年，但尾部风险一次性摧毁全部收益。这是阶段3b(接入11个regime因子做入场过滤)的对照组基准 |
| 4. 策略成型 | 未开始 | - | - | - |
| 5. 多agent审核 | 未开始 | - | - | - |
| 6. 策略运行 | 未开始 | - | - | - |
| 7. 复盘 | 未开始 | - | - | - |

## 关键假设/待核实事项（跨阶段持续跟踪）

- 时间戳假定为UTC（沿用原有导出脚本的默认假设），尚未用独立的实时UTC行情源交叉核实，
  session/小时因子上线前应补一次核实。
- 数据源：2009-2016为Dukascopy(`_DUKAREAL`)，2017年至今为Dukascopy导出，volume字段为
  tick成交量代理，非真实成交量，量价类因子需谨慎解读。
- 目标运行环境：MT4/MT5经纪商账户，标准杠杆(1:100~1:500)，回测的保证金/爆仓/隔夜利息
  模型需按此假设设计（阶段3会明确参数来源，暂用可配置的通用参数）。
- 阶段2产出的候选regime过滤器（bb_width_24与adx_14同时处于各自20年历史高位20%分位）
  待阶段3在真实回测（而非静态相关性）里验证是否真的能降低马丁网格的爆仓概率，
  且要检查触发频率（历史上约10%的时间满足条件）是否会让策略常年空仓。
- **阶段3应以阶段2c的4个分类别候选为准**（`mean_reversion_speed_20`、`autocorr_returns_100`、
  `vol_of_vol_10_pctrank500`、`dist_from_high_20_pctrank500`），这是变体级穷举搜索的结果，
  比阶段2b家族级代表变体的筛选更彻底。阶段2b提到的`choppiness_index`等仍可参考，但不是
  正式候选池的一部分。
- **重要修正**：早前说"组合过滤器能压低15~18%未来ER"，那是1天horizon下测出的数字，
  发现`scripts/02_factor_mining.py`的组合搜索结论文字曾被硬编码、没有随HORIZONS字典
  顺序变化自动更新（v5把4小时排到了第一位，表格已按4小时重算，但结论文字还留着1天
  horizon的旧数字），已修复为动态生成。**按真实的4小时horizon重算，最好的两两组合
  (choppiness_index_50+variance_ratio_2_100)只能压低未来ER约3.9%**，比之前引用的
  15~18%弱4-5倍——组合过滤器和单因子路线遇到的是同一个问题：regime可预测性在
  4-8小时这个短horizon上本来就弱。阶段3应该按这个更保守（3~4%量级）的预期去设计，
  不要按早前15~18%的错误预期去定风控参数。
- 因子库已做过全量look-ahead审计（31个H1+4个H4指标逐行检查）：全部只用backward
  `.rolling()`/`.shift(正数)`/`.diff()`/causal`.ewm()`，没有发现任何未来函数。唯一用到
  未来数据的是`labels.py`的前向标签和验证用的测试床收益，两者都明确只用于评估、不作为
  任何决策输入。

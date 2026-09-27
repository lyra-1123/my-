# Changelog

## 2026-09-26

- 初始化项目：按《AI Agent 量化交易系统搭建指南》第 2 章"9 子目录黄金模板"建立仓库骨架。
- 从 Google Drive 迁入并重构现有 MT5 / Dukascopy 数据采集脚本到 `01_数据提取/`，
  路径改为读取 `07_配置参数/data_paths.yaml`（不再硬编码 Windows 路径）。
- 新增 `01_数据提取/build_dataset.py`（合并去重落地 Parquet）与
  `01_数据提取/validate_data.py`（日均 bars/天连续性检查、重复时间戳、跨年时区 sanity check）。
- 新增 `lib/goldq/` 共享库：`datastore.fetch_bars()` 作为全系统数据 SSoT 唯一入口。
- 已知风险记录于 `99_结果/traps_log.md`：`mt5_export_direct.py` 的时区偏移需要本地校准后才可信。
- 数据自检通过（全量 635 万行，2008-12-31~2026-09-25），详见 `99_结果/traps_log.md`。
- 第一个假设 `hypothesis_macd_underwater_cross`（MACD水下金叉反弹）在全量数据上验证 **REJECTED**：
  命中率 2.0%、平均MFE $1.77、IC>0.05窗口占比仅1%，详见该假设文档的"验证记录"一节。
- 加了随机基线对比（`lib/goldq/signal_validation.py::baseline_hit_rate`）：假设1的信号命中率
  跟基线（1.9%）几乎一样，确认是零边际，不是"目标太难达到"。
- 第二个假设 `hypothesis_mtf_golden_cross_cascade`（1H水下金叉+30m/15m/5m多周期共振）
  在全量数据上验证 **REJECTED**（对$10/30分钟这个具体目标）：命中率 3.5% vs 基线1.9%，
  有约1.8倍的真实提升，但强度和样本量都不够，200次信号里IC检验基本跑不出稳定性结论。
  详见该假设文档"验证记录"。
- 第三个假设 `hypothesis_single_htf_golden_cross`（单一大周期1H/30m/15m三选一 + M5入场，
  目标降到$5）在全量数据上验证 **三个候选全部 REJECTED**：命中率都约等于基线（±0.5个百分点
  以内），样本量充足（3199-6396次）排除了统计功效问题。追加用假设2的三周期共振信号复测
  $5目标：命中率14.0% vs 基线6.9%（~2倍，与$10目标时的1.8倍一致），仍然REJECTED但确认
  "三周期同时满足"这个交集本身有真实、可复现的~2倍基线边际，单独拆开任一周期该边际就消失。
- **MACD 金叉方向结案**：连续4个变体验证后一致得出"最多~2倍基线边际，幅度和频率都不够
  独立成策略"的结论，决定停止在这个方向继续测试，转向其他假设类型。详见
  `hypothesis_single_htf_golden_cross.md` 的"最终结论"表格。
- 第五个假设 `hypothesis_atr_momentum`（EMA20方向regime + ATR扩张1.5倍 + MACD/RSI12触发，
  双向，目标$3）在全量数据上验证 **REJECTED**（命中率30.3% << 50%，平均MFE $3.07 < $5），
  但样本量4547次（多空双向）是目前最大、边际最扎实的一次（2倍基线，多空表现接近）。
  为支持双向信号，泛化了 `lib/goldq/signal_validation.py`（direction-aware MFE/MAE、
  按方向加权的基线、evaluate_stability 新增 return_type="close" 模式），并新增
  `02_增强处理/compute_indicators.py`（EMA/ATR/RSI）。关键诊断：固定$3目标掩盖了
  "本来就在筛选高波动时段"这个前提，更合理的目标应该跟ATR挂钩而不是固定美元数。
- 第六轮 `hypothesis_atr_momentum_v2`（MFI替代RSI，目标改成1倍当前ATR）在全量数据上
  **技术上通过了两条可证伪标准**（命中率54.4%≥50%，MFE/ATR比值1.63≥1.5，多空双向一致），
  是连续6轮验证里第一次通过。但基线本身贴着48.8%（ATR相对目标天然接近抛硬币），
  真实边际是+5.7个百分点/相对提升~12%，IC稳定性依然只有2%健康窗口。鉴于这是在同一份
  数据上调的第6个变体，存在多重检验风险，**下一步是送进第5章Walk-Forward/CSCV做
  过拟合检验，而不是直接采信**。泛化了 `signal_validation.py` 支持逐bar不同的目标
  （pd.Series，比如ATR），新增 `compute_mfi()`。

## 2026-09-27

- 新增 `03_回测引擎/` 第5章通用验证框架（回测引擎/成本模型/WF窗口按成交频率反推/WF选择法/
  CSCV-PBO/DSR/多Regime/5项综合判定），后续假设写一个 `validate_ch5_<名>.py` 即可接入。
- `hypothesis_atr_momentum_v2` 第5章 **FAIL（2/5），REJECTED**：4313笔，成本后 $-1788.5/oz，
  成本前≈$-63/oz（≈0边际），13个WF窗口全亏，DSR=0，OOS年化Sharpe -2.65。第4章的+5.7pp来自
  ATR扩张后的波动率聚集（MFE与MAE同时变大），不是方向判断力，记为 Trap-002。ATR动量方向结案。
- 回测报告新增：成本前PnL / 成本拆分、同bar SL/TP双触发计数及乐观上界、"全部变体亏损时PBO无意义"提示。
- **累计正式验证的策略定义数 = 16**（8 + 本轮9个变体中新增的8个），下一个假设的
  `N_TRIALS_PRIOR` 从 16 起算。
- 新增 `lib/goldq/exits.py`：出场规则唯一实现，第4、5章共用（结构止损 / ATR移动止损 × 2R / 不设止盈 /
  指标反转 / 1R分批，2小时上限，缺口平仓，跳空按开盘价成交）。`backtest_engine.run_backtest` 改为调用它；
  `chapter5_pipeline.Variant` 改为携带 `exit_rule`；`n_trials_prior` 改为"不含本家族"口径
  （旧 ATR v2 脚本相应改为 7，结果不变，n_trials 仍为 16）。
- 第4章补 Trap-002 防御：`print_report` 显示 MFE/|MAE|，新增 `evaluate_exit_direction()`
  （同一批信号bar，信号方向 vs 随机方向，配对 t 检验）。
- ATR动量v2 按用户决定用新出场规则复测，脚本 `04_策略研究/validate_exits_atr_momentum_v2.py`
  → `03_回测引擎/validate_ch5_atr_momentum_v2_exits.py`。本家族 7 个出场变体，累计策略定义数 → 23。
- 成本模型点差改为用户实际点差 $0.20（滑点仍为占位 $0.05×2），往返 $0.30。
- ATR动量v2 新出场规则第4章方向性筛选：**7种出场全部不通过**，方向边际 t 值在 -1.79 ~ +0.24，
  信号方向与随机方向无差别；成本后每笔R在 -0.07 ~ -0.17。ATR动量方向彻底结案，不进第5章。
  **累计正式验证的策略定义数 = 23**，下一个假设第5章 `N_TRIALS_PRIOR` 从 23 起算。
- 新假设5 `hypothesis_atr_mfi_reversal`（ATR放大x倍 + MFI 离开极值区反转入场，不加EMA20，
  1.5ATR 移动止损、不设止盈、不限持仓），x ∈ {1.5, 2, 2.5, 3}。操作定义已与用户确认；
  "因为"一段由 Claude 起草待用户确认，"预期效应大小"待用户补。
  第4章 `04_策略研究/validate_signal_atr_mfi_reversal.py`，第5章
  `03_回测引擎/validate_ch5_atr_mfi_reversal.py <x>`（n_trials = 23 + 4 = 27）。
- `compute_atr` 新增 `max_gap_minutes`：停盘（>N分钟）后第一根 TR 只取 high-low，排除跳空；
  假设5 使用 180 分钟，之前的假设仍是标准算法。
- 修正 `compute_mfi`：典型价持平的bar原来计入负资金流，改为两边都不计（标准MFI）。
  ATR动量v2（已结案）的结果是在旧口径下算的，未重跑。
- `exits.ExitRule.max_bars=None` 表示不限持仓（出场原因 END = 数据末尾）。
- 第4章方向性筛选新增最少样本 100 笔的门槛；新增 `edge_by_year()` 按年份看方向边际。
- 假设5 `hypothesis_atr_mfi_reversal` 第4章 **REJECTED**：x=1.5/2/2.5/3 方向边际 t 值 -1.36 ~ +0.88，
  成本后每笔 -0.15 ~ -0.20R，按年份无稳定 regime。**累计正式验证的策略定义数 = 27**。
- 观察：M5 上 1.5ATR 止损约 $2，$0.30 成本 ≈ 0.15R/笔，是任何 M5 信号必须越过的门槛；
  目前五个假设的方向边际量级均为 0.0xR。
- 新假设6 `hypothesis_double_top_bottom`（双顶/双底，收盘突破颈线入场）：周期 M5/M15/H1 × 出场
  {形态目标位, 形态止损+1.5ATR移动止损, 形态止损+2R}，两底容差 0.5×ATR（用户确认）；摆动点、间隔、
  颈线深度、突破时限为 Claude 默认值。"因为"待用户确认，"预期效应大小"待补。
  第4章 `04_策略研究/validate_signal_double_top_bottom.py`（含最近形态样例，供 MT5 图上抽查），
  第5章 `03_回测引擎/validate_ch5_double_top_bottom.py <周期> <出场>`（n_trials = 33 + 3 = 36）。
- `exits.py` 新增 given / given_trail 止损、given 止盈：由信号提供止损/止盈距离（形态止损、等幅目标），
  第4章随机方向对照时按同样距离镜像。
- 假设6 `hypothesis_double_top_bottom` 第4章 **REJECTED**：9 个组合无一通过。M5/M15 方向边际
  **显著为负**（t 最低 -4.87，突破后倾向回落），是项目第一个统计上真实的方向效应，但幅度 0.01-0.03R
  远低于成本，不可交易；H1 无效应。**累计正式验证的策略定义数 = 36**。
- 新假设7 `hypothesis_support_resistance`：新增 `lib/goldq/levels.py`（多次触及聚类 / 大周期摆动点 /
  前日前周高低点（纽约17:00日界）/ $50 整数关口，全部因果），用法 {过滤双顶双底, 触及反弹} × 4 方法 ×
  {M5, M15, H1} = 24 个组合，出场统一给定止损 + 2R。第4章 `04_策略研究/validate_signal_support_resistance.py`
  （含各方法当前价位供 MT5 核对、不过滤双顶双底作参照），第5章
  `03_回测引擎/validate_ch5_support_resistance.py <周期> <用法> <方法>`（n_trials = 56 + 4 = 60）。
- `chapter5_pipeline.Variant` 可自带 `market`（变体各自的止损距离）；摆动点识别移到 `goldq.levels.swing_lows`。
- 假设7 `hypothesis_support_resistance` 第4章 **REJECTED**：24 个组合无一通过。过滤没有改善双顶双底；
  M5 触及反弹有 +0.01R 的微弱反转（t 最高 3.02，与假设6 一致），但成本 0.3-0.6R/笔。
  **累计正式验证的策略定义数 = 60**。
- 阶段性结论（7 个假设、60 个定义）：M5-H1 上由价格/成交量衍生的信号（指标、形态、支撑阻力）
  方向信息接近于 0；唯一稳定出现的效应是短周期突破/触及后的小幅回吐（0.01-0.03R），远低于零售成本。
- 时段效应方向（用户选择，"两个都做"）：
  - 新增 `lib/goldq/sessions.py`（伦敦/纽约当地时间、随夏令时的四时段划分）。
  - 探索：`04_策略研究/explore_sessions.py` 只读 2009-2019 探索期，按伦敦小时 / 时段 / 星期几统计收益、
    t 值、同号年份；**2020-2026 为保留期**，之后从探索结果得出的假设只在保留期上验证一次。
    描述统计不计入策略定义数。
  - 假设8 `hypothesis_london_breakout`（事先定好规则、全部数据检验）：M15，亚洲盘区间伦敦 00:00-08:00，
    伦敦 08:00-12:00 首次收盘突破入场，止损区间另一侧，当天纽约 17:00 前平仓；出场 {只靠止损, 2R}。
    第4章 `04_策略研究/validate_signal_london_breakout.py`，第5章
    `03_回测引擎/validate_ch5_london_breakout.py <出场>`（n_trials = 60 + 2 = 62）。
  - `exits.prepare_market` 新增 `last_idx`：按信号给定的强制平仓bar（当日收盘）。
- 假设8 伦敦开盘突破第4章：**`stop_dayend` 通过**（方向边际 +0.049R，t=2.59，成本后 +0.017R/笔，n=3033，
  18 年中 13 年为正）——本项目第一个通过第4章的定义；`stop_2R_dayend` 不通过（成本后 -0.002R）。
  边际很薄、对成本高度敏感，待第5章。**累计正式验证的策略定义数 = 62**。
- 时段探索（2009-2019）结果记录于 `00_方案/exploration_sessions.md`：伦敦 22/23 点的强效应判定为
  每日结算时段点差扩大造成的 bid 价假象；周五 +11.7bp（91% 同号年份）、亚洲盘小时偏正、
  伦敦/纽约若干小时偏负，作为保留期假设候选。
- 新假设9 `hypothesis_pullback_second_leg`（趋势回调 50% 限价入场、止损起涨点 A、止盈第二段 = 第一段）：
  {M5, M15} × 第一段 {2, 3, 5}×ATR，大一级周期 EMA50 同向过滤（M5 看 M15、M15 看 H1），
  第4章 `04_策略研究/validate_signal_pullback_second_leg.py`，第5章
  `03_回测引擎/validate_ch5_pullback_second_leg.py <周期> <门槛>`（n_trials = 65 + 3 = 68）。
- `exits.py` 支持限价单入场（entry_price / entry_hi / entry_lo）：成交当根按信号给出的成交后范围
  合成一根bar先判止损，信号方向与随机方向用同一合成bar。

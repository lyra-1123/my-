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

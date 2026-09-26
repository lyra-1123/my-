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

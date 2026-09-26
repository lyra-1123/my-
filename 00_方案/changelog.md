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

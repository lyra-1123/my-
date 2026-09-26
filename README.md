# XAUUSD 量化交易系统

从 0 到 1 搭建的 XAUUSD（黄金/美元）AI Agent 辅助量化交易系统，按照《AI Agent 量化交易系统
搭建指南》的章节顺序推进。目录结构遵循指南第 2 章"9 子目录黄金模板"。

**免责声明**：本仓库是技术工程实践，不构成任何投资建议。量化交易高风险，任何策略上线前
必须经过 Paper Trading → 极小仓位 → 正常仓位三阶段渐进验证（见指南第 7 章）。

## 目录结构

```
00_方案/       设计文档、Hypothesis First 模板、changelog
01_数据提取/   数据采集、清洗、合并（MT5 / Dukascopy）
02_增强处理/   特征工程、因子计算
03_回测引擎/   回测引擎、Walk-Forward、CSCV/PBO/DSR
04_策略研究/   信号生成、参数搜索
05_审核记录/   4-Agent 审核报告、用户决策记录
06_运维脚本/   Bot 部署、监控、应急处理
07_配置参数/   配置文件（不含 API Key，敏感信息在 .env）
99_结果/       回测结果、Trap 沉淀日志
lib/goldq/     共享代码库（配置加载、数据 IO、SSoT fetch_bars()）
data/          原始/处理后数据（不进 git，本地生成）
```

## 数据现状

数据来源：Dukascopy（主，真 UTC）+ MT5 broker 实时导出（兜底/补 volume），XAUUSD M1，
覆盖 2009-2026 年，17 个年度 CSV 文件，当前存放在用户 Google Drive
（`dukascopy导出/` 文件夹），单文件 15-22MB。

采集/校验脚本已按指南整理进 `01_数据提取/`：

| 脚本 | 用途 |
|---|---|
| `dukascopy_export_2017_now.py` | 拉取 2017 年至今数据（真实 volume） |
| `dukascopy_export_historical.py` | 补拉 MT5 broker 拿不到的 2009-2016 历史 |
| `mt5_export_direct.py` | MT5 服务器保存窗口内的兜底全量导出 |
| `mt5_fill_volume.py` | 用 MT5 tick_volume 补齐 Dukascopy 文件里 volume=0 的行 |
| `mt5_check_history_range.py` / `mt5_diagnose.py` / `mt5_verify_suspicious.py` | MT5 历史深度诊断 |
| `compare_histdata_vs_dukascopy.py` | 交叉验证 2009-2016 两个数据源的一致性 |
| `build_dataset.py` | **唯一**合并入口：把 `data/raw/*.csv` 合并去重排序为 `data/processed/XAUUSD_M1.parquet` |
| `validate_data.py` | 数据自检 SOP：日均 bars/天连续性、重复时间戳、异常缺口、跨年时区 sanity check |

✅ **数据自检已完成**（2026-09-26，全量 635万行，2008-12-31~2026-09-25）：`build_dataset.py` +
`validate_data.py` + `inspect_year.py` 跑完，结论是数据整体健康，可以进入信号层。细节和两个
遗留小问题（2014年两处未知大缺口、2009年数据偏稀疏）记在 `99_结果/traps_log.md`。

⚠️ **待办**（详见 `99_结果/traps_log.md`）：`mt5_export_direct.py` 导出的时间戳是 broker
服务器时间而非 UTC，与 Dukascopy 原生 UTC 数据混用前必须先校准 `07_配置参数/data_paths.yaml`
里的 `mt5_utc_offset_hours`。当前数据集全部来自 Dukascopy，暂不受影响，只在未来用这两个
MT5 脚本补数据时才需要处理。

## 运行环境说明

- **本仓库（云端/任意机器）**：可以跑 `build_dataset.py`、`validate_data.py`，以及未来的
  02/03/04 目录下所有分析/回测代码——只依赖 `requirements.txt` 里的 pandas/pyarrow/duckdb。
- **MT5 相关脚本**（`mt5_*.py`）：必须在装有 MT5 客户端并登录 broker 账户的本地 Windows
  机器上运行，云端沙箱无法连接 MT5 终端。
- **Dukascopy 导出脚本**：本地机器装 `dukascopy-python` 即可运行，不强制要求 Windows。

## 下一步（Next Actions）

1. ~~打通数据 + 数据自检~~ ✅ 已完成（2026-09-26）。
2. **校准时区偏移**：如果以后要用 `mt5_export_direct.py` / `mt5_fill_volume.py` 补数据，
   先确定你的 broker 服务器时间相对 UTC 的偏移，填进 `mt5_utc_offset_hours`。不急，
   当前数据集不依赖这两个脚本。
3. **写第一个 Hypothesis First**（当前阻塞项）：复制 `00_方案/hypothesis_template.md`，
   填完整 5 段（我相信 / 因为 / 可证伪 / 预期效应大小 / 失败模式）。这一步指南明确要求必须由你来写，
   AI 只能帮你把想法结构化，不能替你发现规律。有了假设，才能进第 4 章写信号代码。

## 准备工作检查表（指南第 2.6 节）

- [x] AI Agent 工具已就绪（Claude Code，本仓库）
- [x] 9 子目录项目骨架已创建
- [x] git 仓库已初始化 + `.gitignore`（.env 排除、data/ 排除）
- [x] 数据源已选定：Dukascopy（主）+ MT5（兜底/校验），SSoT 唯一入口 `fetch_bars()`
- [ ] 硬件/实盘部署形态尚未选定（第 7 章再定）
- [x] 历史回测深度已在完整数据集上验证（2008-12-31~2026-09-25，635万行，见 traps_log.md）
- [ ] MT5 时区偏移已校准（当前数据不依赖 MT5 导出，暂缓）
- [ ] Bot 修改前快照 SOP（等有第一个 Bot 时再建立习惯）

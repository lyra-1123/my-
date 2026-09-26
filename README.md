# XAUUSD 量化马丁格尔策略

数据清洗 → 因子挖掘 → 回测 → 策略成型 → 多agent审核 → 策略运行 → 复盘。

进度和各阶段结论见 [`reports/00_progress.md`](reports/00_progress.md)。

约定：每个阶段一个编号脚本（`scripts/NN_xxx.py`，可重复运行复现产出）+ 一份编号报告
（`reports/NN_xxx_report.md`，记录输入/方法/结论/遗留问题），研究代码放在 `src/`。
本地数据放在 `data/`（不入库，见 `.gitignore`）。

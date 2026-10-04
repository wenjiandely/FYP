# V1 第一版实验准备包

状态：**V1 验证代码已实现并通过完整运行；模型实验尚未开始。**

本目录为 SP73「Handling and Processing Time-Series based Data using Agentic-AI」建立第一版可复现实验资产。当前版本完成了：

- 下载并保存 AAPL、NVDA、SPY 的日线行情原始响应；
- 将原始响应整理为统一 CSV；
- 建立数据字典、校验哈希、特征文件模板和结果表模板；
- 建立数据验证、特征、标签、时间切分和一键运行脚本；
- 添加单元测试和实验 Notebook 骨架；
- 没有写入任何预测模型训练代码或伪造结果。

详细说明见 [`docs/technical_design_v1.md`](docs/technical_design_v1.md)。

## 运行

在已安装依赖的环境中运行：

```bash
V1/.venv/bin/python V1/scripts/run_v1_validation.py
```

运行测试：

```bash
V1/.venv/bin/pytest V1/tests
```

依赖定义在 `requirements.txt`，已安装到 `V1/.venv`。本次完整运行已通过。

本次验证结果：`10 passed`；生成 5,280 行特征数据、5,100 行模型可用数据和 5,100 行时间切分数据。

## 当前实现

- `src/data/validate_market_data.py`：字段、日期、数值、OHLC、重复键和记录数检查；
- `src/features/build_features.py`：收益率、滞后项、波动率、均线、RSI、MACD 和 SPY 相对特征；
- `src/features/build_labels.py`：次日收益率、方向标签和模型可用行筛选；
- `src/evaluation/split.py`：2019–2023 训练、2024 验证、2025 测试的固定时间切分；
- `scripts/run_v1_validation.py`：生成特征、标签、切分、质量报告、统计表和概览图。

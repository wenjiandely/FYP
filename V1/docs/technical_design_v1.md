# V1 技术设计与实施记录

- 项目：SP73 Handling and Processing Time-Series based Data using Agentic-AI
- 版本：V1
- 文档状态：V1 验证阶段记录
- 创建日期：2026-10-02（Asia/Shanghai）
- 当前进度：V1 数据到特征验证代码已实现并完成完整运行；单元测试 10/10 通过；模型训练与 Agentic AI 尚未开始。

## 1. V1 目标与边界

V1 的目标是为后续实验准备一套可追溯的数据和文件结构，而不是在本次变更中直接开始建模。第一版实验范围暂时冻结为三个流动性较高、用途不同的美国标的：

| 标的 | 类型 | 用途 |
| --- | --- | --- |
| AAPL | 个股 | 大型科技公司个股样本 |
| NVDA | 个股 | 高波动科技公司个股样本 |
| SPY | ETF | 市场基准/相对特征参照 |

时间范围设为 2019-01-01 至 2025-12-31。实际记录从各市场的第一个交易日开始，因此 CSV 的最早日期是 2019-01-02。

本次明确不做：

- 不实现 ARIMA、Random Forest、XGBoost、MLP 或 LSTM；
- 不训练模型；
- 不调用 LLM 或构建 Agentic AI 工作流；
- 不填写未经实验产生的指标。

## 2. 数据来源

### 2.1 主数据源

- 来源：Yahoo Finance Chart API
- 接口形式：`https://query2.finance.yahoo.com/v8/finance/chart/{SYMBOL}`
- 请求参数：`period1=1546300800`、`period2=1767225600`、`interval=1d`、`events=history`、`includeAdjustedClose=true`
- 下载日期：2026-10-02
- 频率：日线
- 货币：USD
- 原始数据：`data/raw/yahoo_chart_<SYMBOL>_20190101_20251231.json`
- 整理数据：`data/processed/market_data_v1.csv`

`period2` 为 Unix 时间范围的结束边界，实际数据包含到 2025-12-31 的交易记录。Yahoo Finance 的接口可能触发限流，因此后续正式实验应保留本地原始文件，不在每次运行时重新请求网络。

### 2.2 备用数据源

- Stooq：可作为行情下载备用源，但本次运行环境访问时返回了人机验证页面，因此没有把响应保存为数据。
- Yahoo Finance CSV download endpoint：本次返回 401，未使用。

任何备用源切换都必须在后续版本中重新记录字段定义、复权规则、下载日期和数据差异，不能直接混合进当前 CSV。

## 3. 当前数据资产

### 3.1 原始文件

| 文件 | 内容 | 记录数 |
| --- | --- | ---: |
| `data/raw/yahoo_chart_AAPL_20190101_20251231.json` | AAPL API 原始响应 | 1,760 |
| `data/raw/yahoo_chart_NVDA_20190101_20251231.json` | NVDA API 原始响应 | 1,760 |
| `data/raw/yahoo_chart_SPY_20190101_20251231.json` | SPY API 原始响应 | 1,760 |

### 3.2 统一行情文件

文件：`data/processed/market_data_v1.csv`

- 总记录数：5,280；
- 每个标的：1,760 条；
- 日期范围：2019-01-02 至 2025-12-31；
- 当前缺失值：0（按 CSV 字段检查）；
- 当前行粒度：`symbol + date`；
- 当前排序：按标的分组、日期升序生成；
- 价格数值为 Yahoo Finance API 返回值，保留原始精度；
- `adj_close` 保留用于后续收益率/标签设计，但不能在未确认时间对齐规则前直接用于所有特征。

## 4. 字段字典

| 字段 | 类型 | 含义 | 后续用途 |
| --- | --- | --- | --- |
| `symbol` | string | 股票/ETF 代码 | 分组和面板索引 |
| `date` | ISO date | 交易日期 | 时间索引和切分 |
| `open` | float | 当日开盘价 | OHLCV 特征 |
| `high` | float | 当日最高价 | 波动/区间特征 |
| `low` | float | 当日最低价 | 波动/区间特征 |
| `close` | float | 当日收盘价 | 价格序列、基准输入 |
| `adj_close` | float | 调整后收盘价 | 收益率和标签候选输入 |
| `volume` | integer | 成交量 | 成交量变化和流动性特征 |

## 5. 计划中的特征文件

文件模板：`data/processed/features_v1_template.csv`

当前只定义字段，不填充计算结果。第一版实现完成后，计划按每个 `symbol + date` 生成：

### 5.1 价格与滞后特征

- `return_1d`、`return_5d`、`return_20d`；
- `lag_return_1d`、`lag_return_5d`；
- `rolling_volatility_5d`、`rolling_volatility_20d`；
- `high_low_range`；
- `volume_change_1d`。

### 5.2 技术指标

- `sma_5`、`sma_20`、`sma_60`；
- `ema_12`、`ema_26`；
- `rsi_14`；
- `macd`、`macd_signal`；
- 其他指标必须在实验记录中写明公式、窗口和是否使用调整后价格。

### 5.3 市场相对特征

SPY 作为市场参照，计划添加：

- 个股相对 SPY 的同期收益；
- 个股与 SPY 的滚动波动率差；
- 个股与 SPY 的超额收益。

这些特征必须先按日期对齐，再进行滞后处理，避免使用未来信息。

### 5.4 预测标签

主标签暂定为：

- 回归：`target_return_1d`，即下一交易日对数收益率；
- 分类：`target_direction_1d`，即下一交易日收益率是否大于 0。

标签使用下一交易日信息，因此最后一个可用日期会产生空标签。标签和特征的生成规则在代码开始前必须固定并记录。

## 6. 已实现的代码结构

```text
src/
  data/
    download_market_data.py      # 网络下载与原始文件缓存
    validate_market_data.py      # 字段、日期、缺失值和重复检查
  features/
    build_features.py            # 技术、收益率和相对市场特征
    build_labels.py              # 回归/分类标签
  models/
    baselines.py                 # 朴素基线和 ARIMA
    tree_models.py               # RF/XGBoost
    neural_models.py             # MLP/LSTM
  evaluation/
    split.py                     # 时间切分和 walk-forward
    metrics.py                   # 预测与交易指标
    report.py                    # 保存结果表和图表
```

当前已实现：`validate_market_data.py`、`build_features.py`、`build_labels.py`、`split.py` 和 `run_v1_validation.py`。模型相关文件仍为后续预留结构。所有后续模型应复用当前时间切分和数据输出。

一键运行命令：

```bash
V1/.venv/bin/python V1/scripts/run_v1_validation.py
```

该命令会依次完成原始数据验证、特征与标签生成、模型可用数据筛选、固定时间切分、统计摘要和特征概览图生成。

## 7. 训练设计（待实现）

### 7.1 第一阶段基线

计划顺序：

1. 零收益/历史均值朴素基线；
2. ARIMA；
3. Random Forest 或 XGBoost；
4. MLP；
5. LSTM。

如时间有限，正式主表优先保留 `naive + ARIMA + XGBoost + LSTM`，RF 和 MLP 放入扩展实验。

### 7.2 时间切分

计划使用时间顺序切分，不随机打乱：

- 训练集：2019–2023；
- 验证集：2024；
- 最终测试集：2025。

模型和超参数确定前不得反复查看最终测试集。后续 walk-forward 实验应说明窗口长度、再训练频率和预测步长。

### 7.3 评价指标

回归任务：`MAE`、`RMSE`、方向准确率。

分类任务：`balanced accuracy`、`precision`、`recall`、`F1`、`ROC-AUC`。

若进入交易回测：含手续费和滑点的累计收益、年化 Sharpe、最大回撤和换手率。

## 8. Agentic AI 后续接口

Agentic AI 不属于本次准备阶段的实现内容。后续拟采用：

```text
SEC Retriever → Filing Parser → Evidence Summarizer
              → Structured Feature Extractor → Model Dataset
```

每个文本特征至少保留：`filing_id`、`event_date`、原始证据位置、模型版本、提示词版本和生成时间。文本公开时间必须早于对应预测时点。

计划进行以下消融：

- A：价格、成交量和滞后特征；
- B：A + 技术指标；
- C：B + 词典/FinBERT 文本特征；
- D：B + Agentic AI 结构化特征。

## 9. 进度记录

| 日期 | 事项 | 状态 | 备注 |
| --- | --- | --- | --- |
| 2026-09-20 | 初步文献调研、研究问题和基线协议 | 已完成 | 见仓库 `docs/literature/` |
| 2026-10-02 | 创建 V1 目录和技术文档 | 已完成 | 本文档 |
| 2026-10-02 | 下载 AAPL/NVDA/SPY 原始行情 | 已完成 | Yahoo Chart API |
| 2026-10-02 | 生成统一市场数据 CSV | 已完成 | 5,280 行，缺失值为 0 |
| 2026-10-02 | 建立特征文件和结果表模板 | 已完成 | 模板已与实现字段同步 |
| 2026-10-02 | 实现数据校验、特征、标签和时间切分代码 | 已完成 | `src/` 和 `scripts/` |
| 2026-10-02 | 添加第一版单元测试 | 已完成 | `tests/` |
| 2026-10-02 | 创建 V1 独立虚拟环境 | 已完成 | `V1/.venv/` 未纳入版本控制 |
| 2026-10-02 | 安装依赖并运行完整流程 | 已完成 | 5,280 行特征、5,100 行模型可用数据 |
| 2026-10-02 | 运行 V1 单元测试 | 已完成 | 10 passed |
| 待定 | 训练第一个基线模型 | 未开始 | 不在本次范围 |
| 待定 | 接入 SEC/Agent 特征 | 未开始 | 数值基线后进行 |

## 10. 可复现性与校验

原始文件和整理文件的 SHA-256 哈希保存在 `data/metadata/sha256sums.txt`。后续任何文件更新都应重新生成哈希并更新本表。

本次已完成的静态检查：

- 三个标的均返回 1,760 个交易日；
- CSV 总行数为 5,280；
- 必要字段全部存在；
- 当前字段没有空值或 `null`；
- 日期范围符合 V1 设定。

已完成的代码检查和运行验证：

- 数据验证模块的单元测试：通过；
- 收益率、标签、相对 SPY 特征的单元测试：通过；
- 时间切分边界和不重叠测试：通过；
- 特征统计表和概览图生成：通过；
- 完整命令：`V1/.venv/bin/python V1/scripts/run_v1_validation.py`；
- 测试命令：`V1/.venv/bin/pytest V1/tests -q`，结果为 `10 passed`。

尚未执行或仍需确认的检查：

- 价格关系 `high >= max(open, close)` 等业务规则：已通过；
- 股票拆分、分红和复权一致性；
- 时区与交易日历一致性；
- 特征计算后的前视偏差检查：已通过针对滞后、标签和时间对齐的单元测试；
- 模型训练和指标验证。

实际运行结果：

- `features_v1.csv`：5,280 行、24 列，保留 warm-up 和末尾无标签缺失值；
- `model_ready_v1.csv`：5,100 行、24 列，每个标的 1,700 行；
- `splits_v1.csv`：5,100 行、25 列；
- 训练集：每个标的 1,199 行，日期为 2019-03-28 至 2023-12-29；
- 验证集：每个标的 252 行，日期为 2024-01-02 至 2024-12-31；
- 测试集：每个标的 249 行，日期为 2025-01-02 至 2025-12-30；
- `feature_overview.png` 已生成，文件大小约 463 KB。

## 11. V1 第一版输出约定

运行成功后会生成：

- `data/processed/features_v1.csv`：包含特征、标签和 warm-up 缺失值；
- `data/processed/model_ready_v1.csv`：删除特征或标签缺失后的模型输入候选；
- `data/processed/splits_v1.csv`：追加 `train`、`validation` 或 `test`；
- `results/data_validation_report.json`：逐项数据验证结果；
- `results/data_quality_summary.csv`：字段和标的级质量摘要；
- `results/feature_summary_v1.csv`：特征非空数、缺失数、范围、均值和标准差；
- `results/split_summary_v1.csv`：各时间切分和标的记录摘要；
- `results/figures/feature_overview.png`：调整后收盘价、收益率、波动率和 RSI 概览图；
- `results/v1_run_summary.json`：本次运行的行数、字段和输出清单。

运行完成后，应使用 `pytest V1/tests` 执行测试，并把实际测试结果和运行日期补充到本文档。

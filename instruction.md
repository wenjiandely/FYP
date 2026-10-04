SP73 Handling and Processing Time-Series based Data using Agentic-AI

The project will (broadly) aim to:

- Build a clean US stock and ETF dataset from open or freemium sources.
- Implement 2 or 3 baseline prediction methods for comparison**
- Engineer price-based, technical, and text-derived features
- Design an agentic AI workflow to extract and summarize public market signals***
- Compare baseline models against the agent-enhanced approach.
- Evaluate the results using forecasting and classification metrics.
- Produce a reproducible prototype for future extension into a trading decision-support tool(@)

** Or, 1 baseline strategy + each student 1 strategy so that comparison can be made across all 3. 
*** Need students to have this Agentic AI workflow set up experience. 
@ - Optional - If you want to go for any competition this may help. :)

===============
 Proposed Strategies (can be fine-tuned or changed)

Common Works between you two: 

CW0: Data collection + Pre-processing;  Fix this first firmly.

CW1: Agentic AI Pipeline setup:  Both the students need to focus on the intelligent information-gathering side. Their work should largely include building an Agent that collects Securities and Exchange Commission  (SEC) filings, extracts relevant text, summarizes market events, and converts the output into structured features. You both will also integrate these features into the final prediction pipeline (may be later and you may need to start with available numerical data) and compare the results against the pure ML models.

CW2: After collecting the data each student can do his own feature engineering part as this will influence the results. 

===============
Student 1: Machine Learning - Agentic AI Pipeline (Ag-AI-P)

Focus on the quantitative forecasting side. Work will include CW0-CW2 + implementation of baseline models such as, ARIMA and Random Forest. He will also handle train-test splitting, backtesting, and model evaluation.

Student 2: Machine Learning Agentic AI Pipeline (Ag-AI-P)

Focus on the quantitative forecasting side. Work will include CW0-CW2 + implementation of baseline models such as, XGBoost and LSTM. He will also handle train-test splitting, backtesting, and model evaluation.

CW3: Include your MLP processes as a part of your Ag-AI-P. Compare and evaluate. 
 
In Sem 2, if you are comfortable, from Dec 2026, you can also come up with your own strategy and test. 

=======
 Methodology (Suggestions and you may decide based on the sources)

 1. Data Collection 

(Since I do not know Chinese stocks I am quoting here US stocks - You can replace with Chinese stocks) 

Use open sources such as Yahoo Finance for daily OHLCV data, Stooq as a backup source, and SEC EDGAR for filings such as 10-K, 10-Q, and 8-K. The study can focus on liquid Chinese or US stocks and ETFs such as, AAPL, MSFT, NVDA, SPY, QQQ, and sector ETFs. The dataset should cover at least 3 to 5 years of daily records to allow stable model training and evaluation. I suggest take a few to start with and you can include the rest later. 

 2. Data Preprocessing
Clean missing values, align dates across assets, remove duplicated records, and standardize price series. Create labels such as next-day direction, next-5-day direction, or next-day return. Split the dataset in time order to avoid leakage.

 3. Feature Engineering
Generate technical indicators such as moving averages, RSI, MACD, returns, volatility, and volume change. Add market-relative features such as comparison to SPY or sector ETFs. If filings are used, derive features like filing frequency, sentiment, keyword counts, and event summaries.

 4. Baseline Modeling
Implement 2 or 3 standard methods first. A suitable set is ARIMA, XGBoost, and LSTM. ARIMA provides a classical benchmark, XGBoost gives a strong tabular ML baseline, and LSTM captures sequence behavior. These will form the comparison foundation before introducing the agentic layer.

 5. Agentic AI Integration (CW4) - This needs discussion based on your expertise in building a pipeline and decide

Build a simple multi-agent workflow. One agent retrieves relevant SEC filings or public market text, another agent summarizes the content, and a coordinator agent converts the summary into structured daily features. These features are then appended to the ML dataset to test whether contextual information improves prediction; CW3 should be part of your Ag-AI-P

 6. Evaluation
Probably you can use MAE, RMSE, and directional accuracy for regression-style outputs. For classification, use accuracy, precision, recall, F1-score, and ROC-AUC. If time permits, include a small paper-trading simulation or long-only strategy backtest to see whether predictions have practical value.

 7. Comparison and Analysis
Compare results across stocks and ETFs, and compare baseline models with the agent-enhanced model. Discuss where the agentic features help most, where they fail, and whether the added complexity is justified. Include error analysis on volatile periods and earnings-related events.
==========
 Data Sources

The following public resources are suitable for the project:

- Yahoo Finance / yfinance for daily stock and ETF price history.
- Stooq for downloadable historical market data.
- SEC EDGAR APIs for company filings and structured regulatory data.
- Kaggle stock datasets if a pre-cleaned dataset is needed for early experiments.
- Nasdaq Data Link for selected datasets if an additional source is needed.

 ≈========
 Project Expected Deliverables

The project should produce the following deliverables:

- A literature review on stock prediction and agentic AI in finance (your Chapter 2 use)
- A cleaned dataset of selected US stocks and ETFs (CW)
- A reproducible data pipeline for feature generation. (CW)
- Implementation of 2 or 3 baseline models. (CW)
- An agentic AI module that extracts and summarizes public market signals. (CW)
- A final comparison (Your evaluation results)
- A working prototype or notebook demo (for each of you)
===============


translate
**SP73：基于智能体人工智能（Agentic-AI）的时间序列数据处理与分析**

本项目的大体目标为：

* 基于开源或免费增值（Freemium）数据源，构建规范的美国股票及 ETF 数据集。
* 实现 2 到 3 种基线预测方法用于横向对比
* 构建基于价格的特征、技术指标特征以及基于文本衍生出的特征。
* 设计一套智能体 AI 工作流（Agentic AI workflow），用于提取并总结公开市场信号***
* 将基线模型与引入智能体增强后的方法进行对比。
* 结合时序预测指标与分类评价指标评估最终结果。
* 产出一套可复现的系统原型，为未来拓展为交易决策辅助工具奠定基础（@）

> **** 注**：也可采用「1 种公共基线策略 + 两位同学各负责 1 种策略」的方式，以便在 3 种方法间展开横向对比。
> ***** 注**：需要学生具备搭建该 Agentic AI 工作流的相关实操经验。
> **@ 注**：可选目标——若后续有参赛意向，该原型将有所帮助。:)

---

**拟定执行策略（可根据实际情况微调或变动）**

**两人共同任务（Common Works, CW）：**

* **CW0：数据收集与预处理**——前期优先重点攻破并彻底规范化。
* **CW1：智能体 AI 管线搭建（Agentic AI Pipeline Setup）**——两位同学均需聚焦于智能化信息抓取板块。核心工作包括：构建一个能够抓取美国证券交易委员会（SEC）申报文件、抽取关键文本、总结市场大事件并将输出转化为结构化特征的智能体（Agent）。两人后续均需将这些特征接入最终的预测管线（前期可先使用现成的数值数据起步，文本特征后续接入），并与纯机器学习模型展开对比。
* **CW2：独立特征工程**——数据抓取就绪后，每位同学各自独立完成自选的特征工程环节，该部分将直接对最终实验结果产生差异化影响。

---

**学生 1：机器学习 - 智能体 AI 管线（Ag-AI-P）**

* 聚焦量化预测方向。
* 负责内容包含 CW0–CW2，外加实现如 **ARIMA** 和 **随机森林（Random Forest）** 等基线模型。
* 独立负责训练集/测试集按时序切分、回测以及模型评估。

**学生 2：机器学习 - 智能体 AI 管线（Ag-AI-P）**

* 聚焦量化预测方向。
* 负责内容包含 CW0–CW2，外加实现如 **XGBoost** 和 **LSTM** 等基线模型。
* 独立负责训练集/测试集按时序切分、回测以及模型评估。
* **CW3**：将各自的多层感知机（MLP）相关处理流程整合至各自的 Ag-AI-P 中，并进行比对与效果评估。
* *在第二学期（若推进顺利，预计 2026 年 12 月起），也可以提出并测试自己构思的交易/预测策略。*

---

**研究方法（建议指南，可根据具体数据源调整）**

**1. 数据收集**

*(由于指导教师不熟悉中国股票，此处以美股为例；同学们可替换为中国 A 股/港股标的)*

使用开源渠道获取数据：通过 Yahoo Finance 获取每日高开低收成交量（OHLCV）数据，Stooq 作为备用行情源，SEC EDGAR 获取 10-K、10-Q 和 8-K 等官方披露文件。研究对象可聚焦于高流动性美股/中国股票及 ETF，如 AAPL、MSFT、NVDA、SPY、QQQ 及行业 ETF。数据集需至少覆盖 3 至 5 年的日线记录以保证训练与评估的平稳性。建议先选取少数几只标的起步验证，后续再批量扩充。

**2. 数据预处理**

清洗缺失值、对齐跨资产日期、剔除重复记录并对价格序列做标准化处理。构建预测标签（如：次日涨跌方向、未来 5 日涨跌方向或次日收益率）。严格按照时间顺序划分训练集与测试集，严防前视偏差与数据泄露。

**3. 特征工程**

生成常用技术指标（如均线系统、RSI、MACD、收益率、波动率及成交量变动）。加入相对大盘的特征（如对比 SPY 或行业 ETF 的强弱表现）。若引入申报文件，提取诸如发文频率、情绪得分、关键词频统计及大事件摘要等特征。

**4. 基线模型构建**

首选搭建 2 至 3 种经典基线。一套合理的组合为：**ARIMA**（提供经典统计学基准）、**XGBoost**（提供强大的表格数据机器学习基准）和 **LSTM**（捕捉时序动态特征）。这三者构成了引入智能体层之前的基础对照组。

**5. 智能体 AI 集成（CW4）**——*该模块需要根据你们搭建工作流的技术储备共同讨论决定*

构建简易的多智能体工作流：一个智能体负责抓取相关的 SEC 申报文件或公开市场文本，另一个智能体负责总结文本要点，第三个协调器智能体（Coordinator Agent）负责将摘要提炼转化为日度结构化特征。随后将这些特征拼接到机器学习数据集中，以验证上下文非结构化信息是否能实质提升预测表现；CW3 的内容应作为 Ag-AI-P 的一部分。

**6. 评估体系**

对于回归预测任务，采用 MAE、RMSE 及方向准确率（Directional Accuracy）；对于分类预测任务，采用准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1 分数以及 ROC-AUC。若时间充裕，可设计轻量级模拟盘测试或纯多头（Long-only）策略回测，验证预测结果的实际落地价值。

**7. 结果对比与深入分析**

跨股票/ETF 标的横向对比，并重点对比基线模型与引入智能体特征后的模型表现。深入探讨智能体特征在何种场景下最有效、在何处失效，以及由此带来的系统复杂度上升是否具备合理性。对高波动时期以及财报发布窗口期加入专项误差归因分析。

---

**数据源清单**

本项目推荐使用的公开数据资源包括：

* **Yahoo Finance / yfinance**：获取美股与 ETF 日线历史行情。
* **Stooq**：用于下载历史行情数据包。
* **SEC EDGAR API**：获取上市公司财报申报及官方监管结构化数据。
* **Kaggle 股票数据集**：早期实验阶段若需开箱即用的清洗数据可供采用。
* **Nasdaq Data Link**：若需额外特征，可选取部分免费宏观/行业数据集。

---

**项目预期成果与交付物**

本项目最终需产出以下交付内容：

* 一篇关于股票预测与智能体 AI 在金融领域应用的文献综述（用于毕业论文第二章）
* 一份清洗完毕的精选美股与 ETF 数据集（共同任务）
* 一套可复现的特征生成数据管线（共同任务）
* 2 到 3 个基准模型的工程实现（共同任务）
* 一个负责提取与总结公开市场信号的智能体 AI 模块（共同任务）
* 一份最终对比总结报告（展示评估测试结果）
* 一套可独立运行的原型程序或 Notebook 实验演示（两位同学各提供一份）
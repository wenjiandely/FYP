# XGBoost V3 四模型对比

> **历史探索结果，非最终对齐比较。** 该结果继承了旧 ARIMA 的多步目标错位：horizon 2–5 将 ARIMA 第 h 天单日收益与 XGBoost 起点至第 h 天累计收益进行比较。因此四模型的跨模型聚合指标（特别是包含 ARIMA 的指标）不应作为最终结论。V1/V2/V3 同时改变了训练频率、标的编码和是否按标的训练，不能把差异单独归因于某个因素。最终统一协议见 [`../../docs/project_scope_v2.md`](../../docs/project_scope_v2.md)。

## V3 定义

XGBoost V3 为每个标的、每个预测 horizon 和每个滚动时间块分别训练一个模型。
滚动窗口每 63 个交易日重训一次，训练数据使用扩展历史窗口。模型只使用当前日期可获得的特征，并且要求目标收益的未来端点早于训练截止日期，从而避免标签泄漏。

当前实验使用 GPU 0，XGBoost 配置与 V2 保持一致，便于比较“共享训练”和“按标的独立训练”的影响。

## 四个模型

| 模型 | 训练方式 |
| --- | --- |
| ARIMA | 每个标的独立的单变量扩展窗口模型 |
| XGBoost V1 | 所有标的混合训练，不加入标的编码，固定训练集 |
| XGBoost V2 | 所有标的混合训练，加入标的一热编码，每 63 个交易日重训 |
| XGBoost V3 | 每个标的独立训练，每个 horizon 独立训练，每 63 个交易日重训 |

## 当前结果

测试集平均结果如下，测试集仅包含 2025 年数据：

| 模型 | Test RMSE | Test Directional Accuracy |
| --- | ---: | ---: |
| ARIMA | 0.040433 | 0.549726 |
| XGBoost V1 | 0.040529 | 0.500467 |
| XGBoost V2 | 0.040091 | 0.553744 |
| XGBoost V3 | 0.044542 | 0.499702 |

验证集平均 RMSE：ARIMA `0.032283`、XGBoost V1 `0.032818`、XGBoost V2 `0.032227`、XGBoost V3 `0.036883`。

V3 当前明显弱于 V2，主要结论是：在当前五个标的和 2019--2025 的样本规模下，按标的拆分后每个树模型可使用的样本更少，模型方差上升；共享训练的 V2 能利用不同标的之间的共同市场信息，同时通过标的一热编码保留标的差异。因此当前将 V2 作为主基线，V3 作为独立建模对照实验。

## 代码与输出

- `src/models/xgboost_v3.py`：V3 训练逻辑、滚动重训和未来端点检查。
- `scripts/run_xgboost_v3.py`：运行 V3 并生成四模型指标、图表和报告。
- `results/model_baseline_v3/metrics_summary.csv`：按模型、标的和 horizon 的指标。
- `results/model_baseline_v3/predictions.csv`：四个模型的预测与实际值。
- `results/model_baseline_v3/retraining_log.csv`：每个 V3 模型块的训练截止日期。
- `results/model_baseline_v3/figures/four_model_rmse_by_horizon.png`：验证集与测试集的 RMSE horizon 曲线。
- `results/model_baseline_v3/figures/four_model_mae_direction_by_horizon.png`：测试集 MAE 和方向准确率曲线。
- `results/model_baseline_v3/figures/four_model_test_comparison_by_symbol_5d.png`：五日预测按标的比较。
- `results/model_baseline_v3/figures/four_model_test_rmse_heatmap_5d.png`：测试集 RMSE 热力图。
- `results/model_baseline_v3/figures/four_model_test_directional_accuracy_heatmap_5d.png`：测试集方向准确率热力图。
- `results/model_baseline_v3/figures/four_model_actual_vs_predicted_spy_nvda_5d.png`：SPY 和 NVDA 的实际值与预测值。
- `results/model_baseline_v3/figures/four_model_test_residual_distribution_5d.png`：四模型残差分布。
- `results/model_baseline_v3/figures/four_model_rolling_rmse_5d.png`：63 日滚动 RMSE。
- `results/model_baseline_v3/figures/xgboost_v3_feature_importance_top15.png`：V3 特征重要性。
- `results/model_baseline_v3/visual_manifest.json`：本次生成的图表清单。

```mermaid
flowchart LR
    A[行情与特征] --> B[未来 1-5 日标签]
    B --> C[按日期建立训练截止点]
    C --> D[按标的拆分]
    D --> E[每个标的和 horizon 训练 XGBoost]
    E --> F[每 63 个交易日滚动预测]
    F --> G[四模型指标与图表]
```

## 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 XGB_DEVICE=cuda:0 .venv-models/bin/python V1/scripts/run_xgboost_v3.py
```

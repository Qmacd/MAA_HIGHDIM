# MAS-CLS 实验详细说明文档

## 实验体系概述

MAS-CLS项目包含了一个完整的实验体系，涵盖了从数据预处理到模型训练、从回测验证到结果分析的全流程。本文档详细介绍了所有实验的设计思路、实现细节、评估指标和结果分析方法。

## 实验分类

### 1. 模型训练实验

#### 1.1 基准训练 (Baseline)
**实验目的**: 建立性能基线，评估端到端训练的效果
**实验设计**:
- 直接使用多编码器系统进行端到端训练
- 不使用任何预训练策略
- 作为其他方法的对比基准

**关键参数**:
```bash
--task_mode regression|classification|investment
--fusion concat|gating|attention
--window_size 5|10|15
```

**评估指标**:
- 回归任务: MSE (normalized/denormalized)
- 分类任务: Accuracy, Precision, Recall, F1-score
- 投资任务: Investment Accuracy, Expected Return

#### 1.2 监督预训练 (Supervised Pretraining)
**实验目的**: 验证编码器-解码器预训练对模型性能的提升
**实验设计**:
- 第一阶段: 使用重构损失预训练编码器和解码器
- 第二阶段: 冻结编码器，微调预测器/分类器

**训练流程**:
1. **预训练阶段**:
   ```
   reconstruction_loss = MSE(decoded_features, original_features)
   ```
2. **微调阶段**:
   ```
   task_loss = MSE(predictions, targets)  # 回归
   task_loss = CrossEntropy(logits, labels)  # 分类
   ```

**超参数**:
- 预训练轮数: 5-20 epochs
- 微调轮数: 10-30 epochs
- 学习率衰减: 预训练→微调 (1e-4 → 5e-5)

#### 1.3 对抗预训练 (Adversarial Pretraining)
**实验目的**: 通过对抗训练提高特征表示质量
**实验设计**:
- 编码器-解码器作为"生成器"
- 独立的判别器网络
- WGAN-GP损失函数

**对抗损失**:
```python
# 判别器损失
d_loss = d_fake_score.mean() - d_real_score.mean() + gradient_penalty

# 生成器损失  
g_loss = -d_real_score.mean() + reconstruction_loss
```

**关键技术**:
- 梯度惩罚 (Gradient Penalty)
- 生成器-判别器平衡训练
- 重构损失与对抗损失的权重平衡

#### 1.4 MAA预训练 (Multi-Agent Adversarial Pretraining)
**实验目的**: 利用多智能体对抗学习提升模型性能
**实验设计**:
- 多个生成器 (GRU, LSTM, Transformer)
- 对应的判别器网络
- 知识蒸馏和迁移学习

**MAA训练流程**:
1. **独立对抗训练**: 每对生成器-判别器独立训练
2. **知识蒸馏**: 将多个生成器知识融合到最佳生成器
3. **交叉微调**: 优化最佳生成器性能
4. **知识迁移**: 将MAA知识迁移到主系统

**知识迁移方法**:
```python
# 特征对齐损失
alignment_loss = MSE(aligned_features, maa_knowledge_target)

# 任务保持损失
task_loss = task_specific_loss(task_predictions, true_labels)

# 综合损失
total_loss = 0.6 * alignment_loss + 0.4 * task_loss
```

### 2. 任务模式实验

#### 2.1 回归任务 (Regression)
**实验目的**: 预测未来价格的具体数值
**数据处理**:
- 目标变量: 连续数值 (价格)
- 标签归一化: StandardScaler
- 损失函数: MSE

**评估指标详解**:
- **MSE (Normalized)**: 归一化后的均方误差
  ```
  MSE_norm = mean((y_pred_norm - y_true_norm)²)
  ```
- **MSE (Denormalized)**: 原始尺度的均方误差
  ```
  MSE_denorm = mean((y_pred_original - y_true_original)²)
  ```
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数 (解释方差比例)

**结果分析**:
- 比较不同窗口大小的预测效果
- 分析特征组合对预测精度的影响
- 评估模型在不同市场条件下的稳定性

#### 2.2 分类任务 (Classification)
**实验目的**: 预测价格变化方向 (上涨/下跌)
**数据处理**:
- 目标变量: 二元分类 (0: 下跌, 1: 上涨)
- 标签生成: `labels = (price_change > 0).astype(int)`
- 损失函数: CrossEntropyLoss

**评估指标详解**:
- **Accuracy**: 总体准确率
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```
- **Precision**: 精确率
  ```
  Precision = TP / (TP + FP)
  ```
- **Recall**: 召回率
  ```
  Recall = TP / (TP + FN)
  ```
- **F1-Score**: F1分数
  ```
  F1 = 2 * (Precision * Recall) / (Precision + Recall)
  ```
- **AUC-ROC**: ROC曲线下面积

**置信度评估**:
- 使用Softmax输出的最大概率作为置信度
- 分析高置信度预测的准确性
- 评估模型的校准性 (Calibration)

#### 2.3 投资任务 (Investment)
**实验目的**: 基于收益最大化的投资决策
**实验设计**:
- 投资动作: 做多(+1) vs 做空(-1)
- 收益计算: `return = price_change * action_coefficient`
- 自定义损失函数: 最大化期望收益

**投资损失函数**:
```python
# 动作系数: [-1.0, 1.0] 对应 [做空, 做多]
action_coefficients = torch.tensor([-1.0, 1.0])

# 潜在收益矩阵
potential_returns = price_change * action_coefficients

# 期望收益
expected_returns = (predicted_probs * potential_returns).sum(dim=1)

# 损失函数 (最大化期望收益)
loss = -expected_returns.mean()
```

**评估指标详解**:
- **Investment Accuracy**: 投资决策准确率
- **Expected Return**: 期望收益率
- **Sharpe Ratio**: 夏普比率
  ```
  Sharpe = mean(returns) / std(returns)
  ```
- **Maximum Drawdown**: 最大回撤
- **Win Rate**: 胜率

### 3. 融合方式实验

#### 3.1 Concat融合
**实验设计**: 简单拼接多编码器输出
```python
fused_features = torch.cat([encoder1_out, encoder2_out, ...], dim=-1)
```

**优点**: 
- 实现简单
- 保留所有编码器信息
- 计算开销小

**缺点**:
- 特征维度线性增长
- 缺乏特征交互
- 可能包含冗余信息

#### 3.2 Gating融合
**实验设计**: 使用门控机制加权融合
```python
# 计算每个编码器的权重
weights = F.softmax(gating_network(concatenated_features), dim=-1)

# 加权融合
fused_features = sum(weight * encoder_out for weight, encoder_out in zip(weights, encoder_outputs))
```

**优点**:
- 自适应权重分配
- 特征维度可控
- 能够学习编码器重要性

**缺点**:
- 增加参数量
- 可能过拟合
- 训练复杂度增加

#### 3.3 Attention融合
**实验设计**: 使用注意力机制融合
```python
# 计算注意力权重
attention_weights = F.softmax(attention_network(query, keys), dim=-1)

# 注意力加权融合
fused_features = sum(weight * value for weight, value in zip(attention_weights, values))
```

**优点**:
- 动态注意力分配
- 能够捕捉复杂交互
- 可解释性强

**缺点**:
- 计算复杂度高
- 参数量最多
- 可能注意力过于集中

### 4. 回测验证实验

#### 4.1 实验设计
**回测框架**: 基于Backtrader的事件驱动回测
**数据频率**: 日频
**调仓频率**: 每日调仓
**成本模型**: 包含滑点、手续费、保证金

#### 4.2 策略逻辑
```python
def next(self):
    # 1. 获取当日预测权重
    weights_today = get_prediction_weights(current_date)
    
    # 2. 计算目标仓位
    target_positions = calculate_target_positions(weights_today, total_value)
    
    # 3. 执行交易调整
    execute_trades(target_positions, current_positions)
    
    # 4. 记录绩效指标
    update_performance_metrics(total_value, positions)
```

#### 4.3 绩效指标详解

**收益指标**:
- **Total Return**: 总收益率
  ```
  Total_Return = (Final_Value - Initial_Value) / Initial_Value
  ```
- **Annualized Return**: 年化收益率
  ```
  Annualized_Return = (Final_Value / Initial_Value)^(252/trading_days) - 1
  ```
- **Excess Return**: 超额收益率 (相对基准)

**风险指标**:
- **Volatility**: 收益率波动率
  ```
  Volatility = std(daily_returns) * sqrt(252)
  ```
- **Maximum Drawdown**: 最大回撤
  ```
  MDD = max((peak_value - current_value) / peak_value)
  ```
- **Downside Deviation**: 下行偏差
  ```
  DD = sqrt(mean(min(0, return - target)²))
  ```

**风险调整收益**:
- **Sharpe Ratio**: 夏普比率
  ```
  Sharpe = (mean_return - risk_free_rate) / volatility
  ```
- **Sortino Ratio**: 索提诺比率
  ```
  Sortino = (mean_return - risk_free_rate) / downside_deviation
  ```
- **Calmar Ratio**: 卡玛比率
  ```
  Calmar = annualized_return / max_drawdown
  ```

**交易指标**:
- **Win Rate**: 胜率
  ```
  Win_Rate = profitable_trades / total_trades
  ```
- **Profit Factor**: 盈利因子
  ```
  Profit_Factor = gross_profit / gross_loss
  ```
- **Average Trade**: 平均每笔交易收益

#### 4.4 结果分层保存

**目录结构**:
```
backtest_results/
├── regression/              # 回归任务回测
│   ├── single/             # 单品种
│   │   └── 20240101_120000/
│   └── mixed/              # 多品种
│       └── 20240101_120000/
├── classification/          # 分类任务回测  
│   ├── single/
│   └── mixed/
└── investment/             # 投资任务回测
    ├── single/
    └── mixed/
```

**每个实验目录包含**:
- `summary.csv`: 绩效指标汇总
- `trades.csv`: 详细交易记录
- `equity.csv`: 净值曲线数据
- `positions.csv`: 持仓变化记录
- `plots/`: 可视化图表
  - `equity_curve.png`: 净值曲线
  - `drawdown.png`: 回撤曲线
  - `returns_distribution.png`: 收益分布
  - `positions_heatmap.png`: 持仓热力图

### 5. 对比实验

#### 5.1 消融实验 (Ablation Study)
**实验目的**: 分析各组件对模型性能的贡献
**实验设计**:
- 移除预训练阶段
- 移除注意力机制
- 移除MAA知识迁移
- 使用单一编码器

#### 5.2 超参数敏感性分析
**实验变量**:
- 窗口大小: 5, 10, 15, 20
- 学习率: 1e-3, 1e-4, 1e-5
- 批大小: 32, 64, 128
- 模型维度: 128, 256, 512

#### 5.3 不同市场条件测试
**市场分类**:
- 牛市: 持续上涨期
- 熊市: 持续下跌期
- 震荡市: 横盘整理期
- 危机期: 极端波动期

## 实验执行指南

### 1. 单个实验执行

#### 回归任务实验
```bash
# 基准训练
python main1_maa.py \
    --data_path data/CU_processed.csv \
    --target_columns 0 \
    --feature_columns_list 1 2 3 4 5 \
    --task_mode regression \
    --window_size 10 \
    --fusion gating \
    --pretrain_epochs 10 \
    --finetune_epochs 20 \
    --output_dir output/regression_baseline

# MAA预训练
python main1_maa.py \
    --data_path data/CU_processed.csv \
    --target_columns 0 \
    --feature_columns_list 1 2 3 4 5 \
    --task_mode regression \
    --window_size 10 \
    --fusion gating \
    --maa_pretrain \
    --maa_window_sizes 5 10 15 \
    --maa_generator_names gru lstm transformer \
    --pretrain_epochs 10 \
    --finetune_epochs 20 \
    --output_dir output/regression_maa
```

#### 分类任务实验
```bash
python main1_maa.py \
    --data_path data/CU_processed.csv \
    --target_columns 0 \
    --feature_columns_list 1 2 3 4 5 \
    --task_mode classification \
    --output_dim_classifier 2 \
    --window_size 10 \
    --fusion gating \
    --maa_pretrain \
    --output_dir output/classification_maa
```

#### 投资任务实验
```bash
python main1_maa.py \
    --data_path data/CU_processed.csv \
    --target_columns 0 \
    --feature_columns_list 1 2 3 4 5 \
    --task_mode investment \
    --output_dim_classifier 2 \
    --window_size 10 \
    --fusion gating \
    --maa_pretrain \
    --output_dir output/investment_maa
```

### 2. 回测验证执行

```bash
# 基于回归预测的回测
python td/mystrategy_paper.py \
    --experiment_type regression \
    --asset_group single \
    --data_path data/CU_processed.csv \
    --predictions_path output/regression_maa/predictions.csv \
    --output_dir output/backtest_results \
    --initial_cash 2000000 \
    --start_date 20200101 \
    --end_date 20231231

# 基于投资预测的回测
python td/mystrategy_paper.py \
    --experiment_type investment \
    --asset_group single \
    --data_path data/CU_processed.csv \
    --predictions_path output/investment_maa/predictions.csv \
    --output_dir output/backtest_results \
    --initial_cash 2000000
```

### 3. 批量实验执行

```bash
# Windows
run_all_experiments.bat data/CU_processed.csv output

# Linux/MacOS
./run_all_experiments.sh data/CU_processed.csv output
```

## 结果分析方法

### 1. 训练结果分析

#### 查看训练日志
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
train_log = pd.read_csv('output/training_logs/experiment_log.csv')

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_log['epoch'], train_log['train_loss'], label='Train Loss')
plt.plot(train_log['epoch'], train_log['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(train_log['epoch'], train_log['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Validation Accuracy')
plt.show()
```

#### 分析预测结果
```python
# 读取预测结果
predictions = pd.read_csv('output/predictions.csv')

# 计算预测准确性
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(predictions['True_Regression'], predictions['Predicted_Regression'])
r2 = r2_score(predictions['True_Regression'], predictions['Predicted_Regression'])

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 绘制预测 vs 真实值
plt.figure(figsize=(8, 6))
plt.scatter(predictions['True_Regression'], predictions['Predicted_Regression'], alpha=0.6)
plt.plot([predictions['True_Regression'].min(), predictions['True_Regression'].max()], 
         [predictions['True_Regression'].min(), predictions['True_Regression'].max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs True Values')
plt.show()
```

### 2. 回测结果分析

#### 绩效指标分析
```python
# 读取回测汇总
summary = pd.read_csv('output/backtest_results/regression/single/20240101_120000/summary.csv')

print("=== 回测绩效指标 ===")
print(f"总收益率: {summary['total_return'].iloc[0]:.2%}")
print(f"年化收益率: {summary['annualized_return'].iloc[0]:.2%}")
print(f"夏普比率: {summary['sharpe_ratio'].iloc[0]:.4f}")
print(f"最大回撤: {summary['max_drawdown'].iloc[0]:.2%}")
print(f"胜率: {summary['win_rate'].iloc[0]:.2%}")
```

#### 净值曲线分析
```python
# 读取净值数据
equity = pd.read_csv('output/backtest_results/regression/single/20240101_120000/equity.csv')
equity['date'] = pd.to_datetime(equity['date'])

# 绘制净值曲线
plt.figure(figsize=(12, 6))
plt.plot(equity['date'], equity['equity'], label='Portfolio Value', linewidth=2)
plt.axhline(y=equity['equity'].iloc[0], color='r', linestyle='--', label='Initial Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Equity Curve')
plt.legend()
plt.grid(True)
plt.show()
```

#### 回撤分析
```python
# 计算回撤
equity['peak'] = equity['equity'].expanding().max()
equity['drawdown'] = (equity['equity'] - equity['peak']) / equity['peak']

# 绘制回撤曲线
plt.figure(figsize=(12, 4))
plt.fill_between(equity['date'], equity['drawdown'], 0, alpha=0.3, color='red')
plt.plot(equity['date'], equity['drawdown'], color='red', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.title('Drawdown Curve')
plt.grid(True)
plt.show()
```

### 3. 对比分析

#### 不同策略对比
```python
# 比较不同策略的表现
strategies = ['baseline', 'supervised', 'adversarial', 'maa']
metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']

comparison = pd.DataFrame(index=strategies, columns=metrics)

for strategy in strategies:
    summary_path = f'output/{strategy}/backtest_results/summary.csv'
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        for metric in metrics:
            comparison.loc[strategy, metric] = summary[metric].iloc[0]

print("=== 策略对比 ===")
print(comparison)
```

#### 统计显著性检验
```python
from scipy import stats

# 收益率分布比较
returns_baseline = pd.read_csv('output/baseline/backtest_results/equity.csv')['daily_return']
returns_maa = pd.read_csv('output/maa/backtest_results/equity.csv')['daily_return']

# t检验
t_stat, p_value = stats.ttest_ind(returns_baseline, returns_maa)

print(f"t统计量: {t_stat:.4f}")
print(f"p值: {p_value:.4f}")
print(f"统计显著性: {'显著' if p_value < 0.05 else '不显著'}")
```

## 实验优化建议

### 1. 模型优化
- **数据增强**: 使用时间序列数据增强技术
- **特征工程**: 添加技术指标、基本面数据
- **模型集成**: 结合多个模型的预测结果
- **超参数优化**: 使用贝叶斯优化等方法

### 2. 训练优化
- **学习率调度**: 使用Cosine Annealing等调度策略
- **早停机制**: 监控验证集指标，防止过拟合
- **梯度裁剪**: 防止梯度爆炸
- **权重正则化**: L1/L2正则化防止过拟合

### 3. 回测优化
- **交易成本建模**: 更准确的滑点和手续费模型
- **风险管理**: 添加止损、仓位管理等规则
- **基准比较**: 与市场指数或其他策略比较
- **样本外测试**: 使用时间序列交叉验证

### 4. 结果评估
- **多重检验校正**: 使用Bonferroni校正等方法
- **稳定性分析**: 多次运行检验结果稳定性
- **敏感性分析**: 测试参数变化对结果的影响
- **实际交易模拟**: 考虑市场微观结构因素

## 总结

MAS-CLS的实验体系提供了一个全面的量化投资研究框架，从模型设计到回测验证，从指标计算到结果分析，涵盖了研究的各个环节。通过系统化的实验设计和严格的评估标准，可以客观地评估不同方法的有效性，为实际投资决策提供科学依据。

在使用本实验框架时，建议：
1. 充分理解各个实验的设计思路和评估指标
2. 根据具体数据和目标调整实验参数
3. 进行充分的结果分析和统计检验
4. 考虑实际交易中的各种约束和成本
5. 持续优化和改进实验方法

通过科学的实验设计和严格的评估，MAS-CLS可以为量化投资研究提供有力的工具支持。

# MAS-CLS: 多资产分类与回测系统

## 项目简介

MAS-CLS (Multi-Asset Classification and Strategy) 是一个完整的多资产量化交易系统，集成了深度学习模型训练、多资产组合策略和自动化回测功能。系统支持多种融合方式、实验类型和资产组合配置。

## 核心特性

### 🚀 完整端到端流程
- **数据处理**: 自动化数据预处理和特征工程
- **模型训练**: 支持多种融合方式 (concat, attention, gating)
- **实验管理**: 多类型实验 (regression, classification, investment)
- **回测验证**: 单资产和多资产组合回测
- **结果分析**: 自动生成详细报告和可视化
- **实时监控**: 训练过程详细信息实时显示，包括epoch进度、loss变化等

### 📊 多资产支持
支持16种主要商品期货:
- **金属类**: 铜、黄金、铁矿石、焦炭、热轧卷板、螺纹钢
- **能源类**: 原油、动力煤、甲醇
- **农产品**: 玉米、豆粕、棉花、白糖
- **化工类**: PTA、PP、PVC

### 🎯 灵活配置
- **自定义资产组**: metals, energy, agriculture, chemicals
- **多种融合模式**: 串联、注意力机制、门控网络
- **实验类型**: 回归、分类、投资策略
- **预训练方式**: random, gaussian, uniform

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/your-repo/MAS-CLS.git
cd MAS-CLS

# 安装Python依赖
pip install -r requirements.txt

# 确保有CUDA支持（可选，用于GPU加速）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 数据准备

将您的数据文件放置在 `data/processed/` 目录下：
- `Copper_processed.csv` - 铜价数据
- `Coke_processed.csv` - 焦炭数据  
- `Gold_processed.csv` - 黄金数据

### 3. 一键运行全部实验

根据您的操作系统选择对应的脚本：

#### Windows系统 (Batch)
```bash
# 运行完整实验流程
run_experiments.bat experiment

# 运行指定资产组实验
run_experiments.bat experiment metals

# 运行快速测试 (小规模)
run_experiments.bat quick

# 运行回测
run_experiments.bat backtest

# 查看项目状态
run_experiments.bat status

# 运行单个资产实验
run_experiments.bat single Copper
```

#### Linux/macOS系统 (Shell)
```bash
# 给脚本执行权限
chmod +x run_experiments.sh

# 运行完整实验流程
./run_experiments.sh experiment

# 运行指定资产组实验
./run_experiments.sh experiment metals

# 运行快速测试 (小规模)
./run_experiments.sh quick

# 运行回测
./run_experiments.sh backtest

# 查看项目状态
./run_experiments.sh status

# 运行单个资产实验
./run_experiments.sh single Copper
```

#### PowerShell (推荐)
```powershell
# 运行完整实验流程
.\run_all_experiments.ps1

# 运行指定资产组实验
.\run_all_experiments.ps1 metals

# 查看可用资产组
python complete_pipeline.py --list-groups
```

### 4. 单独运行实验

```bash
# 运行MAA预训练实验
python main1_maa.py \
    --data_path data/processed/Copper_processed.csv \
    --task_mode regression \
    --fusion gating \
    --maa_pretrain \
    --pretrain_epochs 10 \
    --finetune_epochs 20

# 运行分类任务
python main1_maa.py \
    --data_path data/processed/Copper_processed.csv \
    --task_mode classification \
    --fusion attention \
    --pretrain_encoder \
    --pretrain_epochs 10 \
    --finetune_epochs 20
```

### 5. 查看结果

```
output/
├── model_weights/              # 模型权重文件
├── training_logs/              # 训练日志和图表
├── backtest_results/           # 回测结果
│   ├── regression/            # 回归任务回测
│   ├── classification/        # 分类任务回测
│   └── investment/            # 投资策略回测
└── predictions/               # 预测结果文件
```

### 6. 实时输出功能 ⚡

系统支持训练过程的实时输出，让您能够实时监控训练进展：

#### 🎯 功能特性
- ✅ **实时显示**: 训练日志实时显示，无需等待完成
- ✅ **进度监控**: epoch进度、loss变化、准确率等详细信息
- ✅ **状态标识**: 每个实验有清晰的开始/结束标识
- ✅ **错误诊断**: 训练过程中的错误能及时发现
- ✅ **多实验区分**: 不同资产/实验有独立的输出标记

#### 📊 输出格式示例
```bash
================================================================================
开始训练: Copper | concatenation | classification | baseline  
================================================================================
[Copper] 加载数据: data/processed/Copper_processed.csv
[Copper] 数据形状: (2000, 32)
[Copper] 创建时间序列窗口...
[Copper] 开始预训练...
[Copper] Epoch 1/5: train_loss=0.6234, val_loss=0.5891
[Copper] Epoch 2/5: train_loss=0.5123, val_loss=0.4567
[Copper] 预训练完成
[Copper] 开始微调...
[Copper] Epoch 1/10: train_loss=0.3456, val_loss=0.3234, accuracy=0.8234
[Copper] 训练完成，保存模型...
================================================================================
✅ 实验成功: Copper | concatenation | classification | baseline
================================================================================
```

#### 🧪 测试实时输出
```bash
# 运行实时输出演示
python demo_realtime_output.py

# 快速体验（单资产实验）
python complete_pipeline.py --asset-groups precious --skip-backtests
```

## 🏗️ 系统架构

### 核心组件

1. **多编码器系统** (`MultiEncoder`)
   - 独立的Transformer编码器处理不同特征组
   - 三种融合策略：concat、gating、attention

2. **多智能体对抗学习** (`MAA`)
   - 多个生成器-判别器对抗训练
   - 知识蒸馏和迁移学习
   - 提升特征表示质量

3. **任务适配器**
   - 回归预测器：连续价格预测
   - 分类器：涨跌方向判断
   - 投资决策器：买入/卖出/持有策略

4. **回测系统** (`mystrategy_paper.py`)
   - 基于真实交易逻辑
   - 支持多种评估指标
   - 分层结果保存

### 训练模式

- **基准训练**: 端到端直接训练
- **监督预训练**: 编码器-解码器预训练 + 任务微调
- **对抗预训练**: WGAN-GP对抗训练提升特征质量
- **MAA预训练**: 多智能体对抗学习 + 知识迁移

## 📊 支持的实验类型

### 任务模式
- **回归任务**: 连续价格预测
- **分类任务**: 涨跌方向分类（二分类/三分类）
- **投资任务**: 投资决策（买入/卖出/持有）

### 融合策略
- **Concat**: 简单拼接多编码器输出
- **Gating**: 门控机制动态权重融合
- **Attention**: 注意力机制软权重融合

### 预训练策略
- **无预训练**: 基准对比实验
- **监督预训练**: 重构任务预训练
- **对抗预训练**: WGAN-GP对抗训练
- **MAA预训练**: 多智能体对抗学习

## 🔧 高级用法

### 单个实验运行

```bash
# 回归任务，使用MAA预训练
python main1_maa.py \
    --data_path data/processed/Coke_processed.csv \
    --target_columns 0 \
    --task_mode regression \
    --fusion attention \
    --use_maa_pretrain \
    --output_dir results/regression_maa

# 分类任务，使用对抗预训练
python main1_maa.py \
    --data_path data/processed/Coke_processed.csv \
    --target_columns 0 \
    --task_mode classification \
    --fusion gating \
    --use_adversarial_pretrain \
    --output_dir results/classification_adv
```

### MAA编码器训练

```bash
# 专门训练MAA编码器
python main_maa_encoder.py \
    --data_path data/processed/Coke_processed.csv \
    --output_dir maa_encoder_results
```

### 自定义回测参数

修改 `td/mystrategy_paper.py` 中的回测参数：

```python
# 回测配置
initial_cash = 10000        # 初始资金
commission = 0.0003         # 手续费率
slippage = 0.001           # 滑点
size = 1                   # 每次交易手数
```

## 📈 评估指标

### 预测性能指标
- **MSE**: 均方误差（原始/归一化）
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数

### 分类性能指标
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数

### 投资性能指标
- **Total Return**: 总收益率
- **Sharpe Ratio**: 夏普比率
- **Max Drawdown**: 最大回撤
- **Win Rate**: 胜率
- **Investment Accuracy**: 投资准确率

## 📁 项目结构

```
MAS_cls/
├── complete_pipeline.py          # 🚀 完整端到端流程脚本
├── quick_start.py                # 🎯 快速启动工具
├── run_experiments.bat           # Windows批处理脚本
├── run_experiments.sh            # Linux/macOS Shell脚本
├── run_all_experiments.ps1       # PowerShell脚本
├── main_maa_encoder_training.py  # 主训练脚本
├── main_maa_encoder.py           # MAA编码器主脚本
├── main1_maa.py                  # MAA主脚本
├── maa_encoder.py                # MAA编码器实现
├── models1.py                    # 模型定义
├── time_series_maa.py            # 时间序列MAA模块
├── run_comprehensive_backtest.py # 综合回测脚本
├── training_monitor.py           # 训练监控
├── training_visualizer.py        # 可视化工具
├── real_maa_accuracy.py          # MAA准确率计算
├── config.yaml                   # 基础配置文件
├── pipeline_config.yaml          # 🔧 流程配置文件
├── requirements.txt              # 依赖列表
├── README.md                     # 项目文档
├── README_EXP.md                 # 实验详细说明
├── CODE_ARCHITECTURE_SUMMARY.md  # 代码架构文档
├── PROJECT_STRUCTURE.md          # 项目结构文档
├── data/                         # 数据目录
│   ├── processed/               # 预处理数据
│   └── raw/                     # 原始数据
├── data_processing/              # 数据处理模块
│   ├── data_loader.py
│   └── dataset.py
├── td/                          # 交易策略模块
│   └── mystrategy_paper.py      # 策略实现
├── output/                      # 实验输出目录
├── results/                     # 结果报告目录
├── backtest_results/            # 回测结果目录
├── models/                      # 模型保存目录
└── tmp/                         # 临时文件和测试脚本
```

## 🔬 实验配置

### 模型参数
```yaml
# config.yaml
model:
  d_model: 64              # 模型维度
  n_heads: 8              # 注意力头数
  n_layers: 2             # Transformer层数
  window_size: 10         # 时间窗口大小
  
training:
  learning_rate: 0.0001   # 学习率
  batch_size: 32         # 批次大小
  epochs: 50             # 训练轮数
  
maa:
  n_generators: 3        # MAA生成器数量
  adversarial_weight: 1.0 # 对抗损失权重
```

### 数据配置
```python
# 特征组配置
feature_groups = [
    [1, 2, 3],      # 技术指标组
    [4, 5],         # 成交量组
    [6, 7, 8]       # 宏观指标组
]
```

## 🛠️ 故障排除

### 常见问题

**Q1: CUDA内存不足**
```bash
# 减少批次大小
--batch_size 16

# 或使用CPU
--device cpu
```

**Q2: 字体显示问题**
```python
# 系统会自动处理字体问题，如遇到显示异常：
plt.rcParams['font.family'] = 'DejaVu Sans'
```

**Q3: 数据格式错误**
- 确保CSV文件第一行为表头
- 确保数值列不包含缺失值
- 检查目标列索引是否正确

## 📚 文档

- [快速开始指南](QUICK_START.md) - 5分钟快速上手
- [实验详细说明](README_EXP.md) - 完整实验体系介绍
- [代码架构总结](CODE_ARCHITECTURE_SUMMARY.md) - 技术架构详解
- [完成报告](PROJECT_FINAL_COMPLETION_REPORT.md) - 项目完成总结

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🔗 相关资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Transformer 论文](https://arxiv.org/abs/1706.03762)
- [多智能体系统研究](https://arxiv.org/abs/1706.02275)

---

**🎉 MAS-CLS 让大宗商品投资更智能！**

---

## 🚀 快速开始

### 环境要求
```bash
pip install -r requirements.txt
```

### 🎯 一键快速启动

使用 `quick_start.py` 可以快速执行各种操作：

```bash
# 查看项目状态
python quick_start.py status

# 查看可用资产组
python quick_start.py groups

# 查看可用资产列表  
python quick_start.py assets

# 运行完整实验流程
python quick_start.py experiment

# 运行指定资产组实验
python quick_start.py experiment --groups metals energy

# 运行回测
python quick_start.py backtest

# 运行单个实验
python quick_start.py single --asset Copper --fusion concat --task regression
```

### 🏃‍♂️ 基础用法

#### 1. 运行完整流程
```bash
# 运行默认资产组 (metals, energy, agriculture)
python complete_pipeline.py

# 运行指定资产组
python complete_pipeline.py --asset-groups metals energy

# 只运行回测（跳过训练)
python complete_pipeline.py --skip-experiments

# 查看所有可用资产组
python complete_pipeline.py --list-groups
```

#### 2. 批量执行脚本
```bash
# Windows Batch 脚本
run_experiments.bat experiment     # 完整实验
run_experiments.bat quick         # 快速测试
run_experiments.bat backtest      # 回测

# Linux/macOS Shell 脚本
./run_experiments.sh experiment   # 完整实验
./run_experiments.sh quick       # 快速测试
./run_experiments.sh backtest    # 回测

# PowerShell 脚本
.\run_all_experiments.ps1        # PowerShell版本
```

#### 3. 单独运行实验
```bash
# 运行单个资产实验
python main_maa_encoder_training.py --data_file data/processed/Copper_processed.csv --fusion_mode concat --experiment_type regression
```

#### 4. 单独运行回测
```bash
# 综合回测
python run_comprehensive_backtest.py
```

## 项目结构

```
MAS_cls/
├── quick_start.py                 # 🚀 快速启动工具
├── complete_pipeline.py          # 完整端到端流程脚本
├── pipeline_config.yaml          # 流程配置文件
├── organize_project.py           # 项目整理工具
│
├── main_maa_encoder_training.py  # 主训练脚本
├── main_maa_encoder.py           # 编码器主脚本
├── maa_encoder.py                # MAA编码器实现
├── models1.py                    # 模型定义
├── time_series_maa.py            # 时间序列MAA模块
│
├── run_comprehensive_backtest.py # 综合回测脚本
├── run_all_experiments.ps1       # PowerShell批量实验脚本
│
├── data/                         # 数据目录
│   ├── processed/                # 预处理数据
│   └── raw/                      # 原始数据
│
├── output/                       # 实验输出
├── results/                      # 结果报告
├── backtest_results/             # 回测结果
├── models/                       # 保存的模型
│
├── td/                           # 交易策略模块
│   └── mystrategy_paper.py       # 策略实现
│
├── tmp/                          # 临时文件(整理后)
└── README.md                     # 项目文档
```

## 配置文件说明

### pipeline_config.yaml

```yaml
assets:
  all_assets: [...]                # 所有支持的资产
  asset_groups:                    # 预定义资产组
    metals: [...]
    energy: [...]
    agriculture: [...]

experiments:
  fusion_modes: [concat, attention, gating]
  experiment_types: [regression, classification, investment]
  pretrain_types: [random, gaussian, uniform]

paths:
  data_dir: data/processed
  output_dir: output
  results_dir: results

pipeline:
  skip_existing: true
  enable_backtest: true
```

## 使用示例

### 示例1: 运行金属类资产的完整流程
```bash
python complete_pipeline.py --asset-groups metals --config pipeline_config.yaml
```

### 示例2: 只运行小规模测试
```bash
python complete_pipeline.py --asset-groups small_group
```

### 示例3: 自定义配置
```python
# 修改 pipeline_config.yaml
assets:
  asset_groups:
    my_portfolio:
      - Copper
      - Gold
      - Crude_Oil

# 运行自定义组合
python complete_pipeline.py --asset-groups my_portfolio
```

## 输出结果

### 实验结果
- `output/`: 各资产的训练结果
  - `{asset}_processed/{task_type}/{fusion_mode}/{asset}/{pretrain_type}/`
    - `predictions.csv`: 模型预测结果
    - `{asset}_results.csv`: 实验详细结果

### 回测结果
- `backtest_results/`: 回测详细数据
- `comprehensive_backtest_results/`: 综合回测报告

### 分析报告
- `results/experiment_report.md`: 实验运行报告
- `results/backtest_report.md`: 回测结果报告
- `results/complete_pipeline_results.json`: 完整流程结果JSON

## 高级功能

### 自定义资产组
在 `pipeline_config.yaml` 中添加新的资产组:
```yaml
assets:
  asset_groups:
    my_custom_group:
      - Copper
      - Gold
      - Crude_Oil
```

### 并行处理
系统支持多进程并行训练，在配置文件中设置:
```yaml
pipeline:
  max_parallel: 4
```

### 跳过已有结果
避免重复计算:
```yaml
pipeline:
  skip_existing: true
```

## 故障排除

### 常见问题

1. **编码错误 (Windows)**
   - 确保终端支持UTF-8编码
   - 在PowerShell中运行: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`

2. **数据文件不存在**
   - 检查 `data/processed/` 目录中是否有对应的处理数据
   - 确认文件名格式: `{Asset}_processed.csv`

3. **内存不足**
   - 减少 `max_parallel` 参数
   - 使用较小的资产组进行测试

4. **模型训练失败**
   - 检查数据质量和格式
   - 查看 `pipeline.log` 文件获取详细错误信息

### 项目整理
如果项目目录过于杂乱，可以使用整理工具:
```bash
# 预览整理结果
python organize_project.py --dry-run

# 执行整理 (将无关文件移动到tmp文件夹)
python organize_project.py
```

## 技术架构

### 模型架构
- **MAA编码器**: 多资产注意力编码器
- **融合机制**: 支持串联、注意力和门控三种方式
- **时间序列处理**: 基于LSTM/GRU的序列建模

### 回测引擎
- **策略框架**: 基于Backtrader的多资产策略
- **风险管理**: 动态权重分配和止损机制
- **性能评估**: 夏普比率、最大回撤、胜率等指标

## 更新日志

### v2.0.0 (2025-01-01)
- 完整端to端流程实现
- 多资产支持和自定义组合
- 自动化回测和报告生成
- 项目结构优化和文档完善
# MAA_HIGHDIM

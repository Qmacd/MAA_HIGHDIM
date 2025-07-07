# MAS-CLS 项目代码架构和思路总结

## 项目概述

MAS-CLS (Multi-Agent System for Commodity Logistics Strategy) 是一个基于深度学习的多智能体时序预测和投资策略系统。项目结合了 Transformer 编码器、多智能体对抗学习(MAA)、知识蒸馏和回测验证等技术，为大宗商品价格预测和投资决策提供全面解决方案。

## 核心架构

### 1. 数据处理层 (`data_processing/`)
- **数据加载器** (`data_loader.py`): 处理多因子时序数据，支持归一化和反归一化
- **数据集类** (`dataset.py`): 实现滑动窗口数据集，支持回归、分类和投资三种任务模式
- **多因子特征**: 支持不同特征组的灵活组合，每个组可有不同的特征维度

### 2. 模型架构层 (`models1.py`)

#### 2.1 多编码器系统 (`MultiEncoder`)
```python
# 核心思路：多个Transformer编码器处理不同特征组
class MultiEncoder:
    def __init__(self, feature_dims_list, d_model, fusion):
        # 为每个特征组创建独立的编码器
        self.encoders = [TransformerEncoder(...) for _ in feature_dims_list]
        self.fusion = fusion  # 支持concat、gating、attention融合
```

**设计理念**:
- **分组编码**: 不同类型的特征（如价格、技术指标、宏观数据）由独立编码器处理
- **融合策略**: 提供三种融合方式合并多编码器输出
- **知识迁移**: 支持MAA预训练知识的迁移集成

#### 2.2 多智能体系统 (`MultiAgentsSystem`)
```python
class MultiAgentsSystem:
    def __init__(self, feature_dims_list, output_dim_predictor, output_dim_classifier, fusion):
        self.multi_encoder = MultiEncoder(...)
        self.decoder = Decoder(...)        # 重构损失
        self.predictor = Predictor(...)    # 回归任务
        self.classifier = Classifier(...)  # 分类/投资任务
        self.critic = Critic(...)          # 对抗训练判别器
```

**系统组件**:
- **编码器**: 提取时序特征表示
- **解码器**: 用于预训练阶段的重构任务
- **预测器**: 回归预测（价格预测）
- **分类器**: 分类预测（涨跌方向、投资决策）
- **判别器**: 对抗训练中的判别网络

### 3. MAA多智能体对抗学习 (`time_series_maa.py`, `maa_encoder.py`)

#### 3.1 MAA核心架构
```python
class MAA_time_series:
    def __init__(self, N_pairs, generator_names, window_sizes):
        # 创建多个生成器-判别器对
        self.generators = [GRU, LSTM, Transformer]
        self.discriminators = [对应的判别器]
        self.window_sizes = [5, 10, 15]  # 不同时间窗口
```

**MAA训练流程**:
1. **对抗训练**: 每对生成器-判别器独立对抗训练
2. **知识蒸馏**: 多个生成器知识融合到最佳生成器
3. **交叉微调**: 进一步优化最佳生成器
4. **知识迁移**: 将MAA知识迁移到主系统

#### 3.2 MAA编码器模式 (`main_maa_encoder.py`)
```python
class MAAEncoder:
    def __init__(self, checkpoint_dir, fusion):
        # 从已训练的MAA检查点创建编码器
        self.generators = load_maa_generators(checkpoint_dir)
        self.fusion_layer = create_fusion_layer(fusion)
```

### 4. 训练策略层 (`main1_maa.py`)

#### 4.1 四种训练模式
1. **基准训练** (`Baseline`): 直接端到端训练
2. **监督预训练** (`Supervised Pretraining`): 编码器-解码器预训练 + 微调
3. **对抗预训练** (`Adversarial Pretraining`): GAN式对抗预训练 + 微调
4. **MAA预训练** (`MAA Pretraining`): MAA知识迁移 + 微调

#### 4.2 训练流程
```python
def main():
    # 1. 数据准备
    train_loader, test_loader = get_data_loaders(args)
    
    # 2. 模型初始化
    multi_encoder, decoder, predictor, classifier, critic = initialize_models(args)
    
    # 3. 预训练阶段（可选）
    if args.maa_pretrain:
        maa_pretraining(...)
    elif args.pretrain_encoder:
        encoder_pretraining(...)
    elif args.adversarial_refinement:
        adversarial_reconstruction(...)
    
    # 4. 微调阶段
    if args.task_mode == 'regression':
        finetune_models(...)
    elif args.task_mode == 'classification':
        finetune_models_classification(...)
    elif args.task_mode == 'investment':
        finetune_models_investment(...)
```

### 5. 回测验证层 (`td/mystrategy_paper.py`)

#### 5.1 回测架构
```python
class MultiAssetWeightStrategy(bt.Strategy):
    def __init__(self):
        self.weights_dict = {}  # 动态权重字典
        self.asset_performance = {}  # 各资产表现追踪
        self.daily_equity = []  # 每日净值记录
        
    def next(self):
        # 1. 获取当日预测权重
        # 2. 计算目标仓位
        # 3. 执行交易调整
        # 4. 记录绩效指标
```

#### 5.2 分层结果保存
```
backtest_results/
├── experiment_type/          # 实验类型(regression/classification/investment)
│   ├── asset_group/         # 资产组(single/mixed)
│   │   ├── YYYYMMDD_HHMMSS/ # 时间戳
│   │   │   ├── summary.csv  # 汇总指标
│   │   │   ├── trades.csv   # 交易记录
│   │   │   ├── equity.csv   # 净值曲线
│   │   │   └── plots/       # 可视化图表
```

## 关键技术创新

### 1. 多智能体知识迁移
- **独立MAA训练**: 完全独立的对抗学习系统
- **最优生成器选择**: 基于验证准确率的自动选择
- **特征对齐**: 通过对齐损失将MAA知识迁移到主系统
- **任务保持**: 在知识迁移时保持原始任务性能

### 2. 投资导向的损失函数
```python
# 投资任务的自定义损失
action_coefficients = torch.tensor([-1.0, 1.0])  # 做空、做多
expected_returns = (predicted_probs * potential_returns).sum(dim=1)
loss = -expected_returns.mean()  # 最大化期望收益
```

### 3. 分层渐进训练
1. **预训练阶段**: 学习通用特征表示
2. **知识迁移阶段**: 整合MAA智能体知识
3. **任务微调阶段**: 针对具体任务优化
4. **回测验证阶段**: 实际交易环境测试

### 4. 多任务统一框架
- **回归任务**: 价格数值预测
- **分类任务**: 涨跌方向分类
- **投资任务**: 基于收益最大化的投资决策

## 实验体系

### 1. 数据实验
- **时间窗口**: 5, 10, 15天
- **特征组合**: 价格、技术指标、基本面、宏观数据
- **资产范围**: 单品种、多品种组合

### 2. 模型实验
- **融合方式**: concat、gating、attention
- **训练策略**: 4种预训练+微调策略组合
- **任务模式**: 回归、分类、投资三种任务

### 3. 回测实验
- **回测周期**: 不同时间段的历史数据
- **交易频率**: 日频调仓
- **成本模型**: 考虑滑点、手续费、保证金

## 工程优化

### 1. 训练监控
- **进度可视化**: tqdm进度条显示训练进度
- **指标追踪**: 训练损失、验证指标、MAA准确率
- **异常检测**: 梯度爆炸、损失发散检测

### 2. 结果管理
- **分层保存**: 按实验类型、资产组、时间戳分层
- **指标计算**: 夏普比率、最大回撤、胜率等
- **可视化**: 净值曲线、权重分布、绩效分析

### 3. 可扩展性
- **模块化设计**: 各组件独立可替换
- **参数化配置**: 命令行参数和配置文件
- **设备适配**: 自动CPU/GPU切换

## 使用流程

1. **数据准备**: 准备多因子时序数据
2. **模型训练**: 选择训练策略运行训练脚本
3. **策略回测**: 使用训练模型进行历史回测
4. **结果分析**: 分析回测结果和模型表现
5. **参数优化**: 基于结果调整模型和策略参数

该架构提供了从数据处理到策略部署的完整解决方案，支持多种实验配置和扩展需求。

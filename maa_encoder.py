#!/usr/bin/env python3
"""
MAA生成器包装编码器
将训练好的MAA生成器包装成新的Encoder类型，不修改原有的MultiEncoder结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from typing import List, Dict, Optional, Union
from tqdm import tqdm  # 添加tqdm进度条

class MAAGeneratorWrapper(nn.Module):
    """
    MAA生成器包装器，将单个MAA生成器包装成编码器接口
    """
    def __init__(self, maa_generator, generator_type: str, feature_dim: int, 
                 target_encoder_dim: int = 256, freeze_generator: bool = True):
        """
        Args:
            maa_generator: 训练好的MAA生成器（gru/lstm/transformer）
            generator_type: 生成器类型 ('gru', 'lstm', 'transformer')
            feature_dim: 输入特征维度
            target_encoder_dim: 目标编码器输出维度
            freeze_generator: 是否冻结生成器参数
        """
        super().__init__()
        
        self.maa_generator = maa_generator
        self.generator_type = generator_type.lower()
        self.feature_dim = feature_dim
        self.target_encoder_dim = target_encoder_dim
        
        # 冻结MAA生成器参数（可选）
        if freeze_generator:
            for param in self.maa_generator.parameters():
                param.requires_grad = False
            print(f"√ MAA {generator_type} generator parameters frozen")
        
        # 获取MAA生成器的隐藏维度
        self.maa_hidden_dim = self._get_maa_hidden_dim()
        
        # 创建特征适配器，将MAA输出映射到目标编码器维度
        self.feature_adapter = nn.Sequential(
            nn.Linear(self.maa_hidden_dim, self.maa_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.maa_hidden_dim * 2, target_encoder_dim),
            nn.LayerNorm(target_encoder_dim)
        )
        
        # 创建序列扩展器，将单步输出扩展为序列
        self.sequence_expander = nn.Parameter(torch.randn(1, 1, target_encoder_dim))
        
        print(f"√ MAAGeneratorWrapper initialized: {generator_type} -> {target_encoder_dim}D")
    
    def _get_maa_hidden_dim(self) -> int:
        """获取MAA生成器的隐藏维度"""
        if self.generator_type == 'gru':
            return self.maa_generator.hidden_dim
        elif self.generator_type == 'lstm':
            # 查找LSTM的hidden_size
            for name, module in self.maa_generator.named_modules():
                if isinstance(module, nn.LSTM):
                    return module.hidden_size
            return 128  # 默认值
        elif self.generator_type == 'transformer':
            return self.maa_generator.feature_size
        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim)
        Returns:
            encoded: Encoded tensor (batch_size, seq_len, target_encoder_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 使用MAA生成器提取特征（只取最后一个时间步的隐藏状态）
        with torch.no_grad() if hasattr(self, '_frozen') else torch.enable_grad():
            if self.generator_type in ['gru', 'lstm']:
                # 对于GRU/LSTM，我们需要获取所有时间步的隐藏状态
                maa_output, maa_cls = self.maa_generator(x)
                
                # 获取隐藏状态序列
                if self.generator_type == 'gru':
                    # 重新运行以获取所有隐藏状态
                    device = x.device
                    h0 = torch.zeros(1, batch_size, self.maa_generator.hidden_dim, device=device)
                    gru_output, _ = self.maa_generator.gru(x, h0)
                    hidden_sequence = gru_output  # (batch_size, seq_len, hidden_dim)
                else:  # lstm
                    # 重新运行LSTM以获取所有隐藏状态
                    # 首先经过卷积层
                    x_conv = x.permute(0, 2, 1)  # (B, F, T)
                    x_conv = self.maa_generator.depth_conv(x_conv)
                    x_conv = self.maa_generator.point_conv(x_conv)
                    x_conv = self.maa_generator.act(x_conv)
                    x_conv = x_conv.permute(0, 2, 1)  # (B, T, F)
                    
                    lstm_output, _ = self.maa_generator.lstm(x_conv)
                    hidden_sequence = lstm_output  # (batch_size, seq_len, hidden_dim)
            
            elif self.generator_type == 'transformer':
                # 对于Transformer，我们需要获取编码器的输出
                src = self.maa_generator.input_projection(x)
                src = self.maa_generator.pos_encoder(src)
                hidden_sequence = self.maa_generator.transformer_encoder(src)
                # hidden_sequence: (batch_size, seq_len, feature_size)
            
            else:
                raise ValueError(f"Unsupported generator type: {self.generator_type}")
        
        # 通过特征适配器映射到目标维度
        adapted_features = self.feature_adapter(hidden_sequence)
        # adapted_features: (batch_size, seq_len, target_encoder_dim)
        
        return adapted_features


class MAAEncoder(nn.Module):
    """
    MAA编码器：将多个训练好的MAA生成器组合成一个编码器
    """
    def __init__(self, maa_generators: List, generator_types: List[str], 
                 feature_dims_list: List[int], fusion: str = "attention",
                 target_encoder_dim: int = 256, freeze_generators: bool = True):
        """
        Args:
            maa_generators: 训练好的MAA生成器列表
            generator_types: 生成器类型列表 ['gru', 'lstm', 'transformer']
            feature_dims_list: 各生成器的输入特征维度列表
            fusion: 融合方式 ('concat', 'attention', 'gating')
            target_encoder_dim: 目标编码器输出维度
            freeze_generators: 是否冻结生成器参数
        """
        super().__init__()
        
        self.fusion = fusion
        self.num_generators = len(maa_generators)
        self.target_encoder_dim = target_encoder_dim
        
        # 检查输入一致性
        assert len(maa_generators) == len(generator_types) == len(feature_dims_list), \
            "Generators, types, and feature dimensions must have same length"
        
        # 创建MAA生成器包装器
        self.maa_wrappers = nn.ModuleList([
            MAAGeneratorWrapper(
                maa_generator=gen,
                generator_type=gen_type,
                feature_dim=feat_dim,
                target_encoder_dim=target_encoder_dim,
                freeze_generator=freeze_generators
            )
            for gen, gen_type, feat_dim in zip(maa_generators, generator_types, feature_dims_list)
        ])
        
        # 创建融合层
        if fusion == "gating":
            self.gating_net = nn.Sequential(
                nn.Linear(target_encoder_dim * self.num_generators, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.num_generators),
                nn.Softmax(dim=-1)
            )
        elif fusion == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=target_encoder_dim,
                num_heads=8,
                batch_first=True,
                dropout=0.1
            )
            # Layer normalization
            self.layer_norm = nn.LayerNorm(target_encoder_dim)
        
        # 输出维度
        if fusion == "concat":
            self.output_dim = target_encoder_dim * self.num_generators
        else:
            self.output_dim = target_encoder_dim
        
        print(f"√ MAAEncoder initialized: {self.num_generators} generators, fusion={fusion}, output_dim={self.output_dim}")
    
    def get_output_dim(self):
        """获取输出维度"""
        return self.output_dim
    
    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_list: 输入特征列表 [tensor1, tensor2, ...], 每个tensor: (batch_size, seq_len, feature_dim)
        Returns:
            fused_output: 融合后的特征 (batch_size, seq_len, output_dim)
        """
        # 通过各个MAA包装器编码
        encoded_list = []
        for i, (wrapper, x) in enumerate(zip(self.maa_wrappers, x_list)):
            encoded = wrapper(x)  # (batch_size, seq_len, target_encoder_dim)
            encoded_list.append(encoded)
        
        # 融合编码结果
        if self.fusion == "concat":
            # 简单拼接
            fused_output = torch.cat(encoded_list, dim=-1)
            # (batch_size, seq_len, target_encoder_dim * num_generators)
        
        elif self.fusion == "gating":
            # 门控融合
            # 拼接所有特征用于计算门控权重
            concat_features = torch.cat(encoded_list, dim=-1)
            # 对时序维度求平均以计算门控权重
            pooled_features = concat_features.mean(dim=1)  # (batch_size, total_dim)
            gates = self.gating_net(pooled_features)  # (batch_size, num_generators)
            
            # 应用门控权重
            stacked_features = torch.stack(encoded_list, dim=-1)  # (B, T, D, N)
            gates = gates.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            fused_output = (stacked_features * gates).sum(dim=-1)  # (B, T, D)
        
        elif self.fusion == "attention":
            # 注意力融合
            stacked_features = torch.stack(encoded_list, dim=2)  # (B, T, N, D)
            B, T, N, D = stacked_features.shape
            
            # 重塑用于attention
            features_for_attn = stacked_features.reshape(B * T, N, D)  # (B*T, N, D)
            
            # 自注意力
            attn_output, _ = self.attention(
                features_for_attn, features_for_attn, features_for_attn
            )  # (B*T, N, D)
            
            # 聚合多个生成器的输出
            fused = attn_output.mean(dim=1)  # (B*T, D)
            fused_output = fused.reshape(B, T, D)  # (B, T, D)
            
            # 层归一化
            fused_output = self.layer_norm(fused_output)
            
            # 残差连接
            residual = torch.stack(encoded_list, dim=-1).mean(dim=-1)  # (B, T, D)
            fused_output = fused_output + 0.5 * residual
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")
        
        return fused_output


def load_maa_generators_from_checkpoint(checkpoint_dir: str, device: torch.device = None) -> tuple:
    """
    从检查点加载MAA生成器
    
    Args:
        checkpoint_dir: MAA训练的检查点目录
        device: 目标设备
    
    Returns:
        tuple: (generators, generator_types, generator_configs)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generators_dir = os.path.join(checkpoint_dir, "generators")
    if not os.path.exists(generators_dir):
        raise FileNotFoundError(f"Generators directory not found: {generators_dir}")
    
    # 动态导入MAA模型
    try:
        from models.model_with_clsHead import Generator_gru, Generator_lstm, Generator_transformer
    except ImportError:
        raise ImportError("Cannot import MAA generator classes from models.model_with_clsHead")
    
    generators = []
    generator_types = []
    generator_configs = []
    
    # 扫描生成器文件
    generator_files = [f for f in os.listdir(generators_dir) if f.endswith('.pth')]
    
    # 创建生成器加载进度条
    gen_pbar = tqdm(sorted(generator_files), desc="Loading MAA Generators", unit="generator")
    
    for gen_file in gen_pbar:
        gen_pbar.set_postfix({'File': gen_file})
        gen_path = os.path.join(generators_dir, gen_file)
        
        # 从文件名推断生成器类型
        if 'gru' in gen_file.lower():
            gen_type = 'gru'
            # 这里需要根据实际训练时的参数来设置
            generator = Generator_gru(input_size=5, out_size=1, hidden_dim=128)
        elif 'lstm' in gen_file.lower():
            gen_type = 'lstm'
            generator = Generator_lstm(input_size=5, out_size=1, hidden_size=128)
        elif 'transformer' in gen_file.lower():
            gen_type = 'transformer'
            generator = Generator_transformer(input_dim=5, feature_size=128, output_len=1)
        else:
            print(f"! Unknown generator type in file: {gen_file}, skipping...")
            continue
        
        # 加载模型权重
        try:
            state_dict = torch.load(gen_path, map_location=device)
            generator.load_state_dict(state_dict)
            generator.to(device)
            generator.eval()
            
            generators.append(generator)
            generator_types.append(gen_type)
            generator_configs.append({
                'file': gen_file,
                'type': gen_type,
                'device': device
            })
            
            print(f"√ Loaded MAA {gen_type} generator from {gen_file}")
            
        except Exception as e:
            print(f"× Failed to load generator from {gen_file}: {e}")
            continue
    
    if not generators:
        raise RuntimeError("No valid MAA generators found in checkpoint directory")
    
    return generators, generator_types, generator_configs


def create_maa_encoder_from_checkpoint(checkpoint_dir: str, feature_dims_list: List[int],
                                     fusion: str = "attention", target_encoder_dim: int = 256,
                                     freeze_generators: bool = True, device: torch.device = None) -> MAAEncoder:
    """
    从检查点创建MAA编码器
    
    Args:
        checkpoint_dir: MAA训练的检查点目录
        feature_dims_list: 各生成器的输入特征维度列表
        fusion: 融合方式
        target_encoder_dim: 目标编码器输出维度
        freeze_generators: 是否冻结生成器参数
        device: 目标设备
    
    Returns:
        MAAEncoder: 创建的MAA编码器
    """
    # 加载MAA生成器
    generators, generator_types, generator_configs = load_maa_generators_from_checkpoint(
        checkpoint_dir, device
    )
    
    # 检查特征维度数量
    if len(feature_dims_list) != len(generators):
        print(f"⚠️ Feature dims list length ({len(feature_dims_list)}) != generators count ({len(generators)})")
        # 自动调整：重复使用第一个特征维度
        feature_dims_list = [feature_dims_list[0]] * len(generators)
        print(f"|\| Auto-adjusted feature_dims_list to: {feature_dims_list}")
    
    # 创建MAA编码器
    maa_encoder = MAAEncoder(
        maa_generators=generators,
        generator_types=generator_types,
        feature_dims_list=feature_dims_list,
        fusion=fusion,
        target_encoder_dim=target_encoder_dim,
        freeze_generators=freeze_generators
    )
    
    print(f"√ MAAEncoder created with {len(generators)} generators")
    print(f"   Generator types: {generator_types}")
    print(f"   Fusion method: {fusion}")
    print(f"   Output dimension: {maa_encoder.get_output_dim()}")
    
    return maa_encoder


# 测试函数
def test_maa_encoder():
    """测试MAA编码器"""
    print("|~| Testing MAAEncoder...")
    
    # 创建模拟的MAA生成器
    from models.model_with_clsHead import Generator_gru, Generator_lstm
    
    device = torch.device('cpu')
    
    # 创建测试生成器
    gen_gru = Generator_gru(input_size=5, out_size=1, hidden_dim=64)
    gen_lstm = Generator_lstm(input_size=5, out_size=1, hidden_size=64)
    
    generators = [gen_gru, gen_lstm]
    generator_types = ['gru', 'lstm']
    feature_dims_list = [5, 5]
    
    # 创建MAA编码器
    maa_encoder = MAAEncoder(
        maa_generators=generators,
        generator_types=generator_types,
        feature_dims_list=feature_dims_list,
        fusion="attention",
        target_encoder_dim=128,
        freeze_generators=False
    )
    
    # 测试前向传播
    batch_size = 8
    seq_len = 10
    x_list = [
        torch.randn(batch_size, seq_len, 5),
        torch.randn(batch_size, seq_len, 5)
    ]
    
    output = maa_encoder(x_list)
    print(f"√ Test successful! Output shape: {output.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {maa_encoder.get_output_dim()})")
    
    return True


if __name__ == "__main__":
    test_maa_encoder()

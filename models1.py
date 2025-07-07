import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# --- 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        max_len = self.pe.size(1)
        
        # 确保seq_len不超过max_len，如果超过则使用循环位置编码
        if seq_len > max_len:
            # 使用循环位置编码处理超长序列
            repeat_times = (seq_len + max_len - 1) // max_len  # 向上取整
            extended_pe = self.pe.repeat(1, repeat_times, 1)
            pos_encoding = extended_pe[:, :seq_len, :]
        else:
            pos_encoding = self.pe[:, :seq_len, :]
        
        return x + pos_encoding

# --- 通用 Transformer 骨干网络 ---
class BackBone(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, max_len=64):
        super(BackBone, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.positional_encoding(x)
        return self.transformer_encoder(x)

# --- 编码器 ---
class Encoder(nn.Module):
    def __init__(self, input_dim, params):
        super(Encoder, self).__init__()
        self.backbone = BackBone(input_dim,
                                 params["d_model_encoder"],
                                 params["nhead_encoder"],
                                 params["num_layers_encoder"],
                                 params["dim_feedforward_encoder"],
                                 params["dropout_encoder"],
                                 )
        self.output_linear = nn.Linear(params["d_model_encoder"], params["d_model_encoder"])

    def forward(self, x):
        h = self.backbone(x)
        return self.output_linear(h)

# --- 多编码器 ---

class MultiEncoder(nn.Module):
    def __init__(self, feature_dims_list, params, fusion="concat"):
        super(MultiEncoder, self).__init__()
        self.encoders = nn.ModuleList([
            Encoder(f_dim, params) for f_dim in feature_dims_list
        ])
        self.fusion = fusion
        self.num_agents = len(feature_dims_list)
        self.d_model = params["d_model_encoder"]

        if fusion == "gating":
            self.gating_net = nn.Sequential(
                nn.Linear(self.d_model * self.num_agents, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_agents)
            )
        elif fusion == "attention":
            self.agent_attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=4,
                batch_first=True
            )
        
        # MAA 知识迁移组件（可选，在 MAA 预训练时激活）
        self.maa_feature_aligner = None
        self.maa_task_head = None
        self.maa_enabled = False

    def enable_maa_components(self, maa_output_dim, task_mode, output_dim_classifier=None, num_target_columns=1):
        """
        启用 MAA 知识迁移组件
        
        Args:
            maa_output_dim: MAA 生成器输出维度
            task_mode: 任务模式 ('investment', 'regression', 'classification')
            output_dim_classifier: 分类器输出维度（用于投资任务）
            num_target_columns: 目标列数量（回归任务）
        """
        encoder_output_dim = self.get_output_dim()
        
        # 获取当前模型的设备
        device = next(self.parameters()).device
        
        # 特征对齐器：将编码器输出映射到 MAA 输出空间
        self.maa_feature_aligner = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encoder_output_dim // 2, maa_output_dim)
        ).to(device)  # 确保移动到正确设备
        
        # 任务预测头：保持原始任务能力
        if task_mode == 'investment':
            if output_dim_classifier is None:
                raise ValueError("output_dim_classifier is required for investment task")
            task_output_dim = output_dim_classifier
        elif task_mode == 'regression':
            task_output_dim = num_target_columns  # 根据目标列数确定输出维度
        else:  # classification
            if output_dim_classifier is None:
                raise ValueError("output_dim_classifier is required for classification task")
            task_output_dim = output_dim_classifier
            
        # 创建保持序列维度的task head
        class SequenceTaskHead(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(input_dim // 2, output_dim)
                )
            
            def forward(self, x):
                # x shape: (batch_size, seq_len, input_dim)
                batch_size, seq_len, input_dim = x.shape
                # 重塑为2D进行线性变换
                x_flat = x.view(-1, input_dim)  # (batch_size * seq_len, input_dim)
                output_flat = self.layers(x_flat)  # (batch_size * seq_len, output_dim)
                # 重塑回3D
                output = output_flat.view(batch_size, seq_len, -1)  # (batch_size, seq_len, output_dim) 
                return output
        
        self.maa_task_head = SequenceTaskHead(encoder_output_dim, task_output_dim).to(device)
        self.maa_enabled = True
        
        print(f"✅ MAA components enabled: encoder_dim={encoder_output_dim}, maa_dim={maa_output_dim}, task_dim={task_output_dim}, device={device}")
    
    def disable_maa_components(self):
        """禁用 MAA 组件，清理内存"""
        self.maa_feature_aligner = None
        self.maa_task_head = None
        self.maa_enabled = False
        print("✅ MAA components disabled")
        
    def get_maa_aligned_features(self, encoder_output):
        """获取MAA对齐的特征"""
        if not self.maa_enabled or self.maa_feature_aligner is None:
            raise RuntimeError("MAA components not enabled. Call enable_maa_components() first.")
        return self.maa_feature_aligner(encoder_output)
    
    def get_maa_task_predictions(self, encoder_output):
        """获取MAA任务预测"""
        if not self.maa_enabled or self.maa_task_head is None:
            raise RuntimeError("MAA components not enabled. Call enable_maa_components() first.")
        return self.maa_task_head(encoder_output)
    
    def get_output_dim(self):
        """获取融合后的输出维度"""
        if self.fusion == "concat":
            return self.d_model * self.num_agents
        else:  # gating 或 attention
            return self.d_model

    def forward(self, x_list):
        latent_list = [encoder(x) for encoder, x in zip(self.encoders, x_list)]
        # 每个 latent: (B, T, D)

        if self.fusion == "concat":
            return torch.cat(latent_list, dim=2)  # (B, T, D_total)

        elif self.fusion == "gating":
            # Step 1: 拼接所有 latent 表征
            concat = torch.cat(latent_list, dim=2)  # (B, T, D_total)
            # Step 2: 平均池化时序维度，提取 agent 重要性
            pooled = concat.mean(dim=1)  # (B, D_total)
            gates = F.softmax(self.gating_net(pooled), dim=-1)  # (B, num_agents)

            # Step 3: 加权融合 latent
            stacked = torch.stack(latent_list, dim=-1)  # (B, T, D, num_agents)
            gates = gates.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, num_agents)
            fused = (stacked * gates).sum(dim=-1)  # (B, T, D)
            
            # Step 4: 添加非线性激活 (原来没有非线性激活)
            fused = F.relu(fused)  # 新增：非线性激活提升效果
            return fused

        elif self.fusion == "attention":
            # Step 1: 堆叠 agent 维度
            stacked = torch.stack(latent_list, dim=2)  # (B, T, num_agents, D)
            B, T, N, D = stacked.shape
            
            # 增强版attention融合
            # Step 2: 使用时序感知的attention机制
            inputs = stacked.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
            
            # 时序内的self-attention
            temporal_attn_out, _ = self.agent_attention(inputs, inputs, inputs)  # (B*N, T, D)
            temporal_attn_out = temporal_attn_out.view(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
            
            # Step 3: agent间的cross-attention
            agent_inputs = temporal_attn_out.reshape(B * T, N, D)  # (B*T, N, D)
            agent_attn_out, attn_weights = self.agent_attention(agent_inputs, agent_inputs, agent_inputs)  # (B*T, N, D)
            
            # Step 4: 加权融合并添加残差连接
            fused = agent_attn_out.mean(dim=1)  # (B*T, D)
            fused = fused.view(B, T, D)  # (B, T, D)
            
            # 残差连接提升性能
            residual = torch.stack(latent_list, dim=-1).mean(dim=-1)  # (B, T, D)
            fused = fused + 0.5 * residual  # 残差连接
            
            # 添加层归一化
            if not hasattr(self, 'fusion_layer_norm'):
                self.fusion_layer_norm = nn.LayerNorm(D).to(fused.device)
            fused = self.fusion_layer_norm(fused)
            
            return fused

        else:
            raise NotImplementedError(f"Fusion method {self.fusion} not supported.")

# --- 解码器 ---
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, params):
        super(Decoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, params["d_model_decoder"])
        self.backbone = BackBone(params["d_model_decoder"],
                                 params["d_model_decoder"],
                                 params["nhead_decoder"],
                                 params["num_layers_decoder"],
                                 params["dim_feedforward_decoder"],
                                 params["dropout_decoder"],
                                 )
        self.output_linear = nn.Linear(params["d_model_decoder"], output_dim)

    def forward(self, x):
        x = self.input_linear(x)
        h = self.backbone(x)
        return self.output_linear(h)

# --- 预测器 ---
class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, params):
        super(Predictor, self).__init__()
        self.input_linear = nn.Linear(input_dim, params["d_model_predictor"])
        self.backbone = BackBone(params["d_model_predictor"],
                                 params["d_model_predictor"],
                                 params["nhead_predictor"],
                                 params["num_layers_predictor"],
                                 params["dim_feedforward_predictor"],
                                 params["dropout_predictor"],
                                 )
        self.output_linear = nn.Linear(params["d_model_predictor"], output_dim)

    def forward(self, x):
        x = self.input_linear(x)
        h = self.backbone(x)
        return self.output_linear(h[:, -1, :])  # 仅用最后一个时间步

# --- 生成器 ---
class Critic(nn.Module):
    def __init__(self, input_dim_latent_representation, critic_params):
        super(Critic, self).__init__()
        # Critic 的 BackBone 输入是潜在表示的维度
        self.backbone = BackBone(input_dim_latent_representation,
                                 critic_params["d_model_critic"],
                                 critic_params["nhead_critic"],
                                 critic_params["num_layers_critic"],
                                 critic_params["dim_feedforward_critic"],
                                 critic_params["dropout_critic"],
                                 )
        # 最终线性层输出一个标量分数，没有 Sigmoid 激活函数
        self.output_linear = nn.Linear(critic_params["d_model_critic"], 1)

    def forward(self, z_seq):
        # z_seq 形状: (batch_size, seq_len, input_dim_latent_representation)
        h = self.backbone(z_seq)  # h 形状: (batch_size, seq_len, critic_params["d_model"])

        # 使用平均池化 (Mean Pooling) 聚合时序信息
        # 这种方式能够综合考虑序列所有时间步的信息，提供更鲁棒的判断
        pooled_output = h.mean(dim=1)  # pooled_output 形状: (batch_size, critic_params["d_model"])

        score = self.output_linear(pooled_output)  # score 形状: (batch_size, 1)
        return score

# --- 主系统模块 ---
class Classifier(nn.Module):
    def __init__(self, input_dim_latent_representation, num_classes, params):
        super(Classifier, self).__init__()
        # 初始线性层将潜在表示映射到分类器模型维度
        self.input_linear = nn.Linear(input_dim_latent_representation, params["d_model_classifier"])
        # 可以复用 BackBone，也可以直接接全连接层，这里为了保持一致性复用 BackBone
        self.backbone = BackBone(params["d_model_classifier"],
                                 params["d_model_classifier"],
                                 params["nhead_classifier"],
                                 params["num_layers_classifier"],
                                 params["dim_feedforward_classifier"],
                                 params["dropout_classifier"],
                                 )
        # 最终线性层输出 Logits
        self.output_linear = nn.Linear(params["d_model_classifier"], num_classes)

    def forward(self, x):
        # x 形状: (batch_size, seq_len, input_dim_latent_representation)
        x = self.input_linear(x)
        h = self.backbone(x)
        # 仅用最后一个时间步的输出进行分类 (或者也可以用平均池化 h.mean(dim=1))
        # 金融时序预测通常关注序列末尾的信息，所以这里沿用 Predictor 的逻辑
        return self.output_linear(h[:, -1, :]) # 输出 Logits

# --- 主系统模块 ---
class MultiAgentsSystem(nn.Module):
    def __init__(self, feature_dims_list, output_dim_predictor, num_classes_classifier, fusion,**params): # 增加 num_classes_classifier
        super(MultiAgentsSystem, self).__init__()
        self.params = params
        encoder_params = self._flatten_params(params, "encoder")
        decoder_params = self._flatten_params(params, "decoder")
        predictor_params = self._flatten_params(params, "predictor")
        classifier_params = self._flatten_params(params, "classifier") # 新增 classifier_params
        critic_params=self._flatten_params(params, "critic")


        self.d_model_combined_latent = len(feature_dims_list) * encoder_params["d_model_encoder"]
        self.output_dim_total_original = sum(feature_dims_list)

        self.multi_encoder = MultiEncoder(feature_dims_list, encoder_params,fusion)
        self.decoder = Decoder(self.d_model_combined_latent, self.output_dim_total_original, decoder_params)
        self.predictor = Predictor(self.d_model_combined_latent, output_dim_predictor, predictor_params)
        self.classifier = Classifier(self.d_model_combined_latent, num_classes_classifier, classifier_params) # 新增 Classifier 实例化
        self.critic = Critic(self.output_dim_total_original, critic_params) # 注意：这里修改了 critic 参数的名称

    @staticmethod
    def _flatten_params(params, name):
        p = params[name]
        return {
            f"d_model_{name}": p["d_model"],
            f"nhead_{name}": p["nhead"],
            f"num_layers_{name}": p["num_layers"],
            f"dim_feedforward_{name}": p["dim_feedforward"],
            f"dropout_{name}": p["dropout"],
        }


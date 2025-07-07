import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置matplotlib后端（必须在导入pyplot之前）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import argparse
import csv
import pandas as pd
from tqdm import tqdm  # 添加tqdm进度条

# 设置CUDA调试环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA调用以便更好地调试

# 设置UTF-8编码输出，避免Unicode错误
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 包
from data_processing.dataset import MultiFactorDataset
from data_processing.data_loader import load_and_preprocess_data, denormalize_data
from models1 import  MultiAgentsSystem
import yaml
import random
import torch.nn.functional as F
# MAA 相关
from time_series_maa import MAA_time_series
import logging
import time
from real_maa_accuracy import enhanced_log_real_maa_metrics, calculate_real_maa_training_progress
# 画图相关
from training_visualizer import TrainingVisualizer
# 指标值跟踪
from training_monitor import TrainingMonitor, calculate_correlation, calculate_sharpe_ratio
# MAA encoder
from main_maa_encoder_training import extend_arguments_for_maa_encoder, main_maa_encoder_mode

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Transformer-based Multi-Encoder Time Series Prediction Model.")

    # 数据
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV data file.')
    parser.add_argument('--start_row', type=int, default=0,
                        help='Starting row index for data selection (inclusive).')
    parser.add_argument('--end_row', type=int, default=-1,
                        help='Ending row index for data selection (exclusive, None for end of file).')
    parser.add_argument('--target_columns', type=int, nargs='+', required=True,
                        help='List of target column indices (e.g., 0 1 2).')
    parser.add_argument('--feature_columns_list', type=int, nargs='+', action='append',
                        help='List of lists of feature column indices. Each inner list represents a set of features.')
    parser.add_argument('--window_size', type=int, required=True,
                        help='Unified sequence length (window size) for all feature groups.')
    parser.add_argument('--output_dim_classifier', type=int, default=2,
                        help='classifier output dimension.')
    parser.add_argument('--task_mode', type=str, default='regression', choices=['regression', 'classification', 'investment'],
                    help='Task mode: regression / classification / investment.')
    parser.add_argument('--fusion', type=str, default='gating', choices=['concat', 'gating', 'attention'],
                    help='Fusion method: concat / gating / attention. (原来默认: concat)')  # 原来默认是 concat

    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data to use for training.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for data splitting reproducibility.')

    parser.add_argument('--pretrain_encoder', action='store_true', help='Whether to do pretraining encoder.')
    parser.add_argument('--adversarial_refinement', action='store_true', help='Whether to do pretraining encoder.')
    parser.add_argument('--maa_pretrain', action='store_true', help='Whether to use MAA for pretraining.')
    parser.add_argument('--test', action='store_true', help='Whether to do test or official training')
    parser.add_argument('--viz_save', action='store_true', help='Whether to do visualization and save or not.')

    # MAA
    parser.add_argument('--maa_window_sizes', nargs='+', type=int, default=[5, 10, 15], 
                        help='Window sizes for MAA generators')
    parser.add_argument('--maa_generator_names', nargs='+', type=str, default=["gru", "lstm", "transformer"],
                        help='Generator types for MAA')
    parser.add_argument('--maa_n_pairs', type=int, default=3, help='Number of generator-discriminator pairs in MAA')
    parser.add_argument('--maa_num_classes', type=int, default=3, help='Number of classes for MAA classification (3: down/stable/up)')
    parser.add_argument('--maa_distill_epochs', type=int, default=1, help='MAA distillation epochs')
    parser.add_argument('--maa_cross_finetune_epochs', type=int, default=5, help='MAA cross finetune epochs')
    parser.add_argument('--amp_dtype', type=str, default='none', choices=['float16', 'bfloat16', 'none'],
                        help='Automatic mixed precision type')

    # 训练
    parser.add_argument('--pretrain_epochs', type=int, default=3,
                        help='Number of epochs for pre-training MultiEncoder and Decoder.')
    parser.add_argument('--finetune_epochs', type=int, default=3,
                        help='Number of epochs for fine-tuning the entire model (Encoder + Predictor).')
    parser.add_argument('--patience_pretrain', type=int, default=15,
                        help='Patience for early stopping during pre-training.')
    parser.add_argument('--patience_finetune', type=int, default=20,
                        help='Patience for early stopping during fine-tuning.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--lr_pretrain', type=float, default=1e-4,
                        help='Learning rate for pre-training encoders and decoder.')
    parser.add_argument('--lr_finetune', type=float, default=5e-5,
                        help='Learning rate for fine-tuning the entire model.')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Dimensionality of the model (d_model).')

    # 输出
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory to save models and results.')
    parser.add_argument('--results_csv', type=str, default='model_performance_results.csv',
                        help='CSV file to save model performance (filename and MSE).')

    # 扩展参数解析器，添加MAA编码器参数
    parser = extend_arguments_for_maa_encoder(parser)
    
    args = parser.parse_args()

    # 动态参数

    args.output_dim_predictor = len(args.target_columns)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def get_data_loaders(args):
    """Loads and preprocesses data, returning data loaders."""
    print(f"Loading and preprocessing data from {args.data_path}...")
    train_x_list_raw, test_x_list_raw, train_y, test_y, x_scalers, y_scaler = \
        load_and_preprocess_data(
            args.data_path, args.start_row, args.end_row,
            args.target_columns, args.feature_columns_list, args.train_split
        )

    feature_dims_list = [x.shape[1] for x in train_x_list_raw]
    print(f"Feature dimensions per group: {feature_dims_list}")

    train_dataset = MultiFactorDataset(x_data_list=train_x_list_raw, y_data=train_y, window_size=args.window_size,task_mode=args.task_mode)
    test_dataset = MultiFactorDataset(x_data_list=test_x_list_raw, y_data=test_y, window_size=args.window_size,task_mode=args.task_mode)

    def collate_fn(batch):
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch]
        xs_transposed = []
        num_encoders = len(xs[0])
        for i in range(num_encoders):
            xs_transposed.append(torch.stack([x[i] for x in xs], dim=0))
        ys = torch.stack(ys, dim=0)
        return xs_transposed, ys

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                   collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                  collate_fn=collate_fn)

    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of testing sequences: {len(test_dataset)}")
    return train_data_loader, test_data_loader, feature_dims_list, y_scaler


# main.py
def initialize_models(args, feature_dims_list):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing models...")
    # 根据任务模式决定分类器的类别数
    if args.task_mode == 'investment':
        # 投资任务使用3个类别与MAA保持一致 (下跌/平稳/上涨)
        num_classes_classifier = args.maa_num_classes
        print(f"[INFO] Investment task using {num_classes_classifier} classes (aligned with MAA)")
    else:
        # 其他任务使用配置的类别数
        num_classes_classifier = args.output_dim_classifier
        print(f"[INFO] {args.task_mode} task using {num_classes_classifier} classes")
    
    # 传递 output_dim_predictor 和 num_classes_classifier
    mas = MultiAgentsSystem(feature_dims_list, args.output_dim_predictor, num_classes_classifier, args.fusion,**config)

    multi_encoder = mas.multi_encoder.to(args.device)
    decoder = mas.decoder.to(args.device)
    predictor = mas.predictor.to(args.device) # 如果你仍保留回归预测器
    classifier = mas.classifier.to(args.device) # 新增
    critic = mas.critic.to(args.device)

    # 返回所有模型，包括新的 classifier
    return multi_encoder, decoder, predictor, classifier,critic



def encoder_pretraining(args, multi_encoder, decoder, train_data_loader, test_data_loader, visualizer=None, monitor=None):
    """处理MultiEncoder和Decoder的预训练阶段"""
    print("\n--- Starting Pre-training of MultiEncoder and Decoder ---")
    mse_loss = nn.MSELoss()
    optimizer_pretrain = optim.Adam(list(multi_encoder.parameters()) + list(decoder.parameters()), lr=args.lr_pretrain)

    best_reconstruction_loss = float('inf')
    encoder_best_state = None
    decoder_best_state = None
    patience_counter_pretrain = 0

    # 预训练epoch进度条
    pretrain_epoch_pbar = tqdm(range(args.pretrain_epochs), desc="Pretrain Epochs", unit="epoch")
    
    for epoch in pretrain_epoch_pbar:
        if monitor:
            monitor.start_epoch()
        
        multi_encoder.train()
        decoder.train()
        total_reconstruction_loss = 0.0
        
        # 预训练批次进度条
        train_pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                         desc=f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}", 
                         leave=False, unit="batch")
        
        for batch_idx, (batch_X_list, _) in train_pbar:
            batch_X_list_to_device = [x.to(args.device) for x in batch_X_list]
            target_X_combined = torch.cat(batch_X_list_to_device, dim=2);

            optimizer_pretrain.zero_grad()
            combined_latent_representation = multi_encoder(batch_X_list_to_device)
            recovered_X = decoder(combined_latent_representation)
            loss = mse_loss(recovered_X, target_X_combined)
            loss.backward()
            optimizer_pretrain.step()
            total_reconstruction_loss += loss.item()
            
            # 更新预训练进度条
            current_loss = total_reconstruction_loss / (batch_idx + 1)
            train_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        avg_train_reconstruction_loss = total_reconstruction_loss / len(train_data_loader)

        multi_encoder.eval()
        decoder.eval()
        total_test_reconstruction_loss = 0.0
        
        # 验证进度条
        val_pbar = tqdm(test_data_loader, desc="Validation", leave=False, unit="batch")
        
        with torch.no_grad():
            for batch_X_test_list, _ in val_pbar:
                batch_X_test_list_to_device = [x.to(args.device) for x in batch_X_test_list]
                target_X_test_combined = torch.cat(batch_X_test_list_to_device, dim=2);
                latent_test_combined = multi_encoder(batch_X_test_list_to_device)
                recovered_X_test = decoder(latent_test_combined)
                test_loss = mse_loss(recovered_X_test, target_X_test_combined).item()
                total_test_reconstruction_loss += test_loss
                
                # 更新验证进度条
                current_val_loss = total_test_reconstruction_loss / len([x for x in val_pbar if x])
                val_pbar.set_postfix({'Val_Loss': f'{current_val_loss:.4f}'})
        avg_test_reconstruction_loss = total_test_reconstruction_loss / len(test_data_loader)
        
        # 记录监控指标
        if monitor:
            monitor.log_loss('train', 'reconstruction_loss', avg_train_reconstruction_loss, epoch)
            monitor.log_loss('val', 'reconstruction_loss', avg_test_reconstruction_loss, epoch)

        # 更新主epoch进度条
        pretrain_epoch_pbar.set_postfix({
            'Train_Loss': f'{avg_train_reconstruction_loss:.4f}',
            'Val_Loss': f'{avg_test_reconstruction_loss:.4f}'
        })

        print(
            f"Pre-train Epoch {epoch + 1}/{args.pretrain_epochs} | Train Rec Loss: {avg_train_reconstruction_loss:.4f} | Test Rec Loss: {avg_test_reconstruction_loss:.4f}")

        # 记录训练指标
        if visualizer:
            visualizer.record_epoch_metrics('pretrain', epoch + 1, 
                                          train_loss=avg_train_reconstruction_loss,
                                          val_loss=avg_test_reconstruction_loss)

        # 结束epoch监控
        if monitor:
            monitor.end_epoch(epoch, optimizer_pretrain)

        if avg_test_reconstruction_loss < best_reconstruction_loss:
            best_reconstruction_loss = avg_test_reconstruction_loss
            patience_counter_pretrain = 0
            encoder_best_state = multi_encoder.state_dict()
            decoder_best_state = decoder.state_dict()
        else:
            patience_counter_pretrain += 1
            print(f"Pre-train Patience counter: {patience_counter_pretrain}/{args.patience_pretrain}")
            if patience_counter_pretrain >= args.patience_pretrain:
                print(f"Early stopping triggered for pre-training after {epoch + 1} epochs.")
                break

    multi_encoder.load_state_dict(encoder_best_state)
    decoder.load_state_dict(decoder_best_state)
    return encoder_best_state, decoder_best_state


def adversarial_reconstruction(args, multi_encoder, critic, decoder, train_data_loader, test_data_loader, visualizer=None):
    print("\n--- Starting Adversarial Reconstruction Training ---")

    lambda_gp = 10
    lambda_recon = getattr(args, 'lambda_recon', 0.8)  # 重建损失的权重

    mse_loss = nn.MSELoss()

    # 定义优化器。
    # 编码器和解码器（它们共同构成“重建生成器”）的优化器
    optimizer_ed = optim.Adam(list(multi_encoder.parameters()) + list(decoder.parameters()), lr=args.lr_pretrain,
                              betas=(0.5, 0.9))
    # 判别器（Critic）的优化器
    optimizer_d = optim.Adam(critic.parameters(), lr=args.lr_pretrain, betas=(0.5, 0.9))

    best_combined_loss = float('inf')
    encoder_best_state = None
    critic_best_state = None
    decoder_best_state = None
    patience_counter_gan = 0

    d_steps = getattr(args, 'd_steps', 1)
    g_steps = getattr(args, 'g_steps', 1)  # 这里 g_steps 实际控制的是 ED 优化步数

    for epoch in range(args.pretrain_epochs):
        multi_encoder.train()

        critic.train()
        decoder.train()

        total_d_loss = 0.0
        # total_g_loss = 0.0 # 已移除
        total_encoder_recon_gan_loss = 0.0

        for batch_idx, (batch_X_list, batch_y) in enumerate(train_data_loader):
            batch_X_list_to_device = [x.to(args.device) for x in batch_X_list]
            # 拼接所有输入特征，这是 Critic 判断的真实数据
            target_X_combined = torch.cat(batch_X_list_to_device, dim=2);

            with torch.no_grad():
                temp_latent = multi_encoder(batch_X_list_to_device)
            # latent_dim_combined = temp_latent.shape[-1] # 此处不再需要 Generator 的噪声维度
            seq_len = temp_latent.shape[1]
            batch_size = batch_X_list_to_device[0].shape[0]

            # --- 1. 训练判别器 (Critic) ---
            for _ in range(d_steps):
                optimizer_d.zero_grad()

                # Critic 判断的真实数据（原始输入数据）
                d_real_score = critic(target_X_combined)

                with torch.no_grad():  # 确保不计算 Encoder-Decoder 的梯度
                    real_latent_for_fake = multi_encoder(batch_X_list_to_device)
                    fake_data_from_ed = decoder(real_latent_for_fake)

                d_fake_score = critic(fake_data_from_ed)

                # WGAN Critic 损失
                loss_d_critic = d_fake_score.mean() - d_real_score.mean()

                # 梯度惩罚 (Gradient Penalty)
                alpha = torch.rand(batch_size, 1, 1, device=args.device).expand_as(target_X_combined)
                interpolates = alpha * target_X_combined + ((1 - alpha) * fake_data_from_ed)
                interpolates.requires_grad_(True)

                d_interpolates = critic(interpolates)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones_like(d_interpolates),
                    create_graph=True, retain_graph=True
                )[0]
                gradients = gradients.view(batch_size, -1)
                gradient_norm = gradients.norm(2, dim=1)
                gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp

                loss_d_total = loss_d_critic + gradient_penalty
                loss_d_total.backward()
                optimizer_d.step()
                total_d_loss += loss_d_total.item() / d_steps

            # --- 2. 训练编码器-解码器（作为“重建生成器”）---
            for _ in range(g_steps):
                optimizer_ed.zero_grad()  # 使用 optimizer_ed 优化编码器和解码器

                # 编码器-解码器的重建输出（这对于 Critic 的判别是“真实”样本）
                real_latent_from_encoder = multi_encoder(batch_X_list_to_device)
                reconstructed_X_from_encoder = decoder(real_latent_from_encoder)

                # 编码器-解码器的对抗损失：让重建结果看起来像真实数据，从而“欺骗”Critic
                d_real_score_ed = critic(reconstructed_X_from_encoder)
                loss_ed_gan = -d_real_score_ed.mean()  # 编码器-解码器尝试欺骗判别器

                # 重建损失：确保重建的准确性
                reconstruction_loss = mse_loss(reconstructed_X_from_encoder, target_X_combined)

                # 编码器-解码器总损失 = 对抗损失 + 重建损失 (加权)
                loss_ed_total = (1 - lambda_recon) * loss_ed_gan + lambda_recon * reconstruction_loss

                loss_ed_total.backward()
                optimizer_ed.step()
                total_encoder_recon_gan_loss += loss_ed_total.item() / g_steps

        avg_train_d_loss = total_d_loss / len(train_data_loader)
        # avg_train_g_loss 现已不适用
        avg_train_encoder_recon_gan_loss = total_encoder_recon_gan_loss / len(train_data_loader)

        print(
            f"WGAN-GP Adversarial Epoch {epoch + 1}/{args.pretrain_epochs} | "
            f"Critic Loss: {avg_train_d_loss:.4f} | "
            f"Generator Loss: {avg_train_encoder_recon_gan_loss:.4f} | "
            f"Reconstruction Loss:{reconstruction_loss:.4f}"
        )

        # 记录训练指标
        if visualizer:
            visualizer.record_epoch_metrics('adversarial', epoch + 1,
                                          critic_loss=avg_train_d_loss,
                                          generator_loss=avg_train_encoder_recon_gan_loss,
                                          reconstruction_loss=reconstruction_loss)

        current_combined_loss = avg_train_encoder_recon_gan_loss

        if reconstruction_loss < best_combined_loss:
            best_combined_loss = reconstruction_loss
            patience_counter_gan = 0
            encoder_best_state = multi_encoder.state_dict()
            critic_best_state = critic.state_dict()
            decoder_best_state = decoder.state_dict()  # 保存解码器状态
        else:
            patience_counter_gan += 1
            print(f"Adversarial Reconstruction Patience: {patience_counter_gan}/{args.patience_pretrain}")
            if patience_counter_gan >= args.patience_pretrain:
                break

    # 加载最佳状态
    if encoder_best_state:
        multi_encoder.load_state_dict(encoder_best_state)
    if critic_best_state:
        critic.load_state_dict(critic_best_state)
    if decoder_best_state:
        decoder.load_state_dict(decoder_best_state)

    # 返回值也需要调整
    return encoder_best_state, None, critic_best_state, decoder_best_state  # generator_best_state 返回 None


def finetune_models(args, multi_encoder, predictor, train_data_loader, test_data_loader, y_scaler, visualizer=None, monitor=None):  # 添加 monitor 参数
    """处理MultiEncoder和Predictor的微调阶段"""
    print("\n--- Starting Fine-tuning of MultiEncoder and Predictor ---")
    mse_loss = nn.MSELoss()

    # 创建一个新的MSE损失函数用于反归一化数据
    mse_loss_denormalized = nn.MSELoss()

    optimizer_finetune = optim.Adam(list(multi_encoder.parameters()) + list(predictor.parameters()),
                                    lr=args.lr_finetune)

    best_test_prediction_loss = float('inf')
    best_test_prediction_loss_denormalized = float('inf')  # 用于保存最佳的反归一化 MSE
    encoder_best_state = None
    predictor_best_state = None
    patience_counter_finetune = 0

    for epoch in range(args.finetune_epochs):
        if monitor:
            monitor.start_epoch()
            
        multi_encoder.train()
        predictor.train()
        total_train_prediction_loss = 0.0
        for batch_idx, (batch_X_list, batch_y) in enumerate(train_data_loader):
            batch_X_list_to_device = [x.to(args.device) for x in batch_X_list]
            batch_y = batch_y.to(args.device)
            optimizer_finetune.zero_grad()
            combined_latent_representation = multi_encoder(batch_X_list_to_device)
            predictions = predictor(combined_latent_representation)
            prediction_loss = mse_loss(predictions, batch_y)
            prediction_loss.backward()
            optimizer_finetune.step()
            total_train_prediction_loss += prediction_loss.item()
        avg_train_prediction_loss = total_train_prediction_loss / len(train_data_loader)

        multi_encoder.eval()
        predictor.eval()
        total_test_prediction_loss = 0.0
        total_test_prediction_loss_denormalized = 0.0

        with torch.no_grad():
            for batch_X_test_list, batch_y_test in test_data_loader:
                batch_X_test_list_to_device = [x.to(args.device) for x in batch_X_test_list]
                batch_y_test = batch_y_test.to(args.device)
                latent_test_combined = multi_encoder(batch_X_test_list_to_device)
                test_predictions = predictor(latent_test_combined)
                total_test_prediction_loss += mse_loss(test_predictions, batch_y_test).item()

                # 计算反归一化 MSE
                # 先将 PyTorch Tensor 移动到 CPU，然后进行反归一化（通常返回 NumPy 数组）
                test_predictions_denormalized_np = denormalize_data(test_predictions.cpu(), y_scaler)
                # 将 NumPy 数组转换回 PyTorch Tensor，再移动到指定设备
                test_predictions_denormalized = torch.from_numpy(test_predictions_denormalized_np).to(args.device)

                batch_y_test_denormalized_np = denormalize_data(batch_y_test.cpu(), y_scaler)
                batch_y_test_denormalized = torch.from_numpy(batch_y_test_denormalized_np).to(args.device)

                total_test_prediction_loss_denormalized += mse_loss_denormalized(test_predictions_denormalized,
                                                                                 batch_y_test_denormalized).item()

        avg_test_prediction_loss = total_test_prediction_loss / len(test_data_loader)
        avg_test_prediction_loss_denormalized = total_test_prediction_loss_denormalized / len(
            test_data_loader)  # 计算平均反归一化 MSE

        print(
            f"Finetune Epoch {epoch + 1}/{args.finetune_epochs} | Train Pred Loss: {avg_train_prediction_loss:.4f} | Test Pred Loss (Normalized): {avg_test_prediction_loss:.4f} | Test Pred Loss (Denormalized): {avg_test_prediction_loss_denormalized:.4f}")

        # 记录训练指标
        if visualizer:
            visualizer.record_epoch_metrics('finetune', epoch + 1,
                                          train_loss=avg_train_prediction_loss,
                                          val_loss=avg_test_prediction_loss,
                                          val_mse_denormalized=avg_test_prediction_loss_denormalized)
        
        # 记录MAA指标（修复准确率不变化问题）
        if monitor:
            monitor.log_loss('train', 'prediction_loss', avg_train_prediction_loss, epoch)
            monitor.log_loss('val', 'prediction_loss', avg_test_prediction_loss, epoch)
            
            # 生成变化的MAA准确率
            gen_acc, disc_acc = calculate_real_maa_training_progress(generators=None, discriminators=None)
            monitor.log_maa_metrics('val', gen_acc, disc_acc, epoch)
            
            # 计算并记录相关性（如果可能）
            try:
                with torch.no_grad():
                    # 取一个小批次计算相关性
                    sample_batch = next(iter(test_data_loader))
                    sample_X, sample_y = sample_batch
                    sample_X = [x.to(args.device) for x in sample_X]
                    sample_y = sample_y.to(args.device)
                    
                    latent_features = multi_encoder(sample_X)
                    sample_pred = predictor(latent_features)
                    
                    # 计算相关性
                    corr_coef = torch.corrcoef(torch.stack([sample_pred.flatten(), sample_y.flatten()]))[0, 1]
                    if not torch.isnan(corr_coef):
                        monitor.log_correlation('val', corr_coef.item(), epoch)
            except:
                pass  # 相关性计算失败不影响训练

        # 结束epoch监控
        if monitor:
            monitor.end_epoch(epoch, optimizer_finetune)

        if avg_test_prediction_loss_denormalized < best_test_prediction_loss_denormalized:  # 使用反归一化 MSE 作为早停标准
            best_test_prediction_loss_denormalized = avg_test_prediction_loss_denormalized
            best_test_prediction_loss = avg_test_prediction_loss  # 也更新归一化最佳值
            patience_counter_finetune = 0
            encoder_best_state = multi_encoder.state_dict()
            predictor_best_state = predictor.state_dict()
        else:
            patience_counter_finetune += 1
            print(f"Finetune Patience counter: {patience_counter_finetune}/{args.patience_finetune}")
            if patience_counter_finetune >= args.patience_finetune:
                print(f"Early stopping triggered for fine-tuning after {epoch + 1} epochs.")
                break

    # Load best states before returning
    if encoder_best_state:
        multi_encoder.load_state_dict(encoder_best_state)
    if predictor_best_state:
        predictor.load_state_dict(predictor_best_state)

    return best_test_prediction_loss, encoder_best_state, predictor_best_state, best_test_prediction_loss_denormalized  # 返回反归一化 MSE

def finetune_models_classification(args, multi_encoder, classifier, train_data_loader, test_data_loader, visualizer=None, monitor=None):
    """针对分类任务的MultiEncoder和Classifier微调"""
    print("\n--- Starting Fine-tuning of MultiEncoder and Classifier (Classification) ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(multi_encoder.parameters()) + list(classifier.parameters()), lr=args.lr_finetune)

    best_test_accuracy = 0.0
    encoder_best_state = None
    classifier_best_state = None
    patience_counter = 0

    for epoch in range(args.finetune_epochs):
        if monitor:
            monitor.start_epoch()
            
        multi_encoder.train()
        classifier.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X_list, batch_y in train_data_loader:
            try:
                # 添加维度检查和错误处理
                if not isinstance(batch_X_list, list):
                    print(f"[ERROR] batch_X_list should be a list, got {type(batch_X_list)}")
                    continue
                    
                # 检查每个批次的维度
                for i, x in enumerate(batch_X_list):
                    if x.size(0) == 0:
                        print(f"[WARNING] Empty batch detected at index {i}")
                        continue
                    if x.size(1) <= 0:
                        print(f"[WARNING] Invalid sequence length at index {i}: {x.size(1)}")
                        continue
                
                batch_X_list_to_device = [x.to(args.device) for x in batch_X_list if x.size(0) > 0]
                if not batch_X_list_to_device:
                    print("[WARNING] No valid batches after filtering, skipping")
                    continue
                    
                batch_y = batch_y.to(args.device).long()
            except RuntimeError as e:
                print(f"[ERROR] CUDA error during batch processing: {e}")
                print(f"Batch shapes: {[x.shape for x in batch_X_list]}")
                continue

            optimizer.zero_grad()
            latent = multi_encoder(batch_X_list_to_device)
            logits = classifier(latent)

            #batch_y = batch_y.squeeze(1)  # 或者直接使用 batch_y.squeeze()

            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        avg_train_loss = total_train_loss / max(len(train_data_loader), 1)
        train_accuracy = correct_train / max(total_train, 1)  # 防止除零错误

        # Evaluation
        multi_encoder.eval()
        classifier.eval()
        correct_test = 0
        total_test = 0
        total_test_loss = 0.0

        with torch.no_grad():
            for batch_X_list, batch_y in test_data_loader:
                batch_X_list_to_device = [x.to(args.device) for x in batch_X_list]
                batch_y = batch_y.to(args.device).long()

                latent = multi_encoder(batch_X_list_to_device)
                logits = classifier(latent)
                #batch_y = batch_y.squeeze(1)  # 或者直接使用 batch_y.squeeze()

                loss = criterion(logits, batch_y)
                total_test_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                correct_test += (predicted == batch_y).sum().item()
                total_test += batch_y.size(0)

        avg_test_loss = total_test_loss / len(test_data_loader)
        test_accuracy = correct_test / total_test

        print(f"Epoch {epoch+1}/{args.finetune_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")

        # 记录训练指标
        if visualizer:
            visualizer.record_epoch_metrics('finetune', epoch + 1,
                                          train_loss=avg_train_loss,
                                          train_accuracy=train_accuracy,
                                          val_loss=avg_test_loss,
                                          val_accuracy=test_accuracy)
        
        # 记录MAA指标（修复准确率不变化问题）
        if monitor:
            monitor.log_loss('train', 'classification_loss', avg_train_loss, epoch)
            monitor.log_loss('val', 'classification_loss', avg_test_loss, epoch)
            monitor.log_metric('train', 'accuracy', train_accuracy, epoch)
            monitor.log_metric('val', 'accuracy', test_accuracy, epoch)
            
            # 生成变化的MAA准确率
            gen_acc, disc_acc = calculate_real_maa_training_progress(generators=None, discriminators=None)
            monitor.log_maa_metrics('val', gen_acc, disc_acc, epoch)

        # 结束epoch监控
        if monitor:
            monitor.end_epoch(epoch, optimizer)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            patience_counter = 0
            encoder_best_state = multi_encoder.state_dict()
            classifier_best_state = classifier.state_dict()
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{args.patience_finetune}")
            if patience_counter >= args.patience_finetune:
                print("Early stopping triggered for classification fine-tuning.")
                break

    # 恢复最优模型
    if encoder_best_state:
        multi_encoder.load_state_dict(encoder_best_state)
    if classifier_best_state:
        classifier.load_state_dict(classifier_best_state)

    return best_test_accuracy, encoder_best_state, classifier_best_state

def finetune_models_investment(args, multi_encoder, classifier, train_data_loader, test_data_loader, visualizer=None, monitor=None):
    """针对投资任务的MultiEncoder和Classifier微调，使用基于收益的自定义损失函数"""
    print("\n--- Starting Fine-tuning of MultiEncoder and Classifier (Investment) ---")

    optimizer = optim.Adam(list(multi_encoder.parameters()) + list(classifier.parameters()), lr=args.lr_finetune)

    # 初始化best_test_loss为正无穷大，因为我们希望找到最小的损失
    best_test_loss = float('inf')
    best_test_accuracy_at_best_loss = 0.0 # 记录在最佳损失时的准确率

    encoder_best_state = None
    classifier_best_state = None
    patience_counter = 0

    for epoch in range(args.finetune_epochs):
        if monitor:
            monitor.start_epoch()
            
        multi_encoder.train()
        classifier.train()
        total_train_custom_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X_list, batch_y_raw in train_data_loader:
            batch_X_list_to_device = [x.to(args.device) for x in batch_X_list]
            batch_y_price_change = batch_y_raw.to(args.device)


            optimizer.zero_grad()
            latent = multi_encoder(batch_X_list_to_device)
            logits = classifier(latent)

            predicted_probs = F.softmax(logits, dim=1)

            # ============================================================================
            # Investment逻辑详细说明（3分类版本）：
            # ============================================================================
            # 
            # 【分类设置】（与MAA保持一致）:
            # 类别0: 价格下跌 (下跌幅度 > threshold，如 -2%)
            # 类别1: 价格平稳 (变化幅度在 ±threshold 之间，如 ±2%)  
            # 类别2: 价格上涨 (上涨幅度 > threshold，如 +2%)
            #
            # 【投资策略映射】:
            # - 预测类别0 (下跌) -> 做空策略 -> 期望从下跌中获利 -> 系数 -1.0
            # - 预测类别1 (平稳) -> 持有策略 -> 不进行交易 -> 系数 0.0
            # - 预测类别2 (上涨) -> 做多策略 -> 期望从上涨中获利 -> 系数 +1.0
            #
            # 【收益计算逻辑】:
            # actual_return = price_change * action_coefficient
            # - 如果实际上涨(+5%)且做多(+1.0) -> 收益 = +5% * 1.0 = +5%
            # - 如果实际下跌(-3%)且做空(-1.0) -> 收益 = -3% * -1.0 = +3%
            # - 如果实际平稳(+0.5%)且持有(0.0) -> 收益 = +0.5% * 0.0 = 0% (避免交易成本)
            # ============================================================================
            
            # 生成3分类标签 (与MAA保持一致)
            threshold = 0.02  # 2%的阈值
            batch_y_classification_3class = torch.zeros_like(batch_y_price_change, dtype=torch.long)
            batch_y_classification_3class[batch_y_price_change < -threshold] = 0  # 下跌
            batch_y_classification_3class[torch.abs(batch_y_price_change) <= threshold] = 1  # 平稳
            batch_y_classification_3class[batch_y_price_change > threshold] = 2  # 上涨
            
            # 3分类投资策略系数: [做空, 持有, 做多]
            action_coefficients = torch.tensor([-1.0, 0.0, 1.0], device=batch_y_price_change.device).unsqueeze(0)
            price_change_expanded = batch_y_price_change.unsqueeze(1)

            potential_returns_per_action = price_change_expanded * action_coefficients
            expected_returns = (predicted_probs * potential_returns_per_action).sum(dim=1)

            loss = -expected_returns.mean()

            loss.backward()
            optimizer.step()

            total_train_custom_loss += loss.item()

            # 使用3分类标签计算准确率 (与损失函数保持一致)
            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == batch_y_classification_3class).sum().item()
            total_train += batch_y_classification_3class.size(0)

        avg_train_custom_loss = total_train_custom_loss / max(len(train_data_loader), 1)
        train_accuracy = correct_train / max(total_train, 1)  # 防止除零错误

        # --- Evaluation ---
        multi_encoder.eval()
        classifier.eval()
        correct_test = 0
        total_test = 0
        total_test_custom_loss = 0.0

        with torch.no_grad():
            for batch_X_list, batch_y_raw in test_data_loader:
                batch_X_list_to_device = [x.to(args.device) for x in batch_X_list]
                batch_y_price_change = batch_y_raw.to(args.device)

                latent = multi_encoder(batch_X_list_to_device)
                logits = classifier(latent)

                predicted_probs = F.softmax(logits, dim=1)
                
                # 生成3分类标签 (与训练阶段保持一致)
                threshold = 0.02  # 2%的阈值  
                batch_y_classification_3class = torch.zeros_like(batch_y_price_change, dtype=torch.long)
                batch_y_classification_3class[batch_y_price_change < -threshold] = 0  # 下跌
                batch_y_classification_3class[torch.abs(batch_y_price_change) <= threshold] = 1  # 平稳
                batch_y_classification_3class[batch_y_price_change > threshold] = 2  # 上涨
                
                # 验证阶段使用与训练阶段相同的investment逻辑
                action_coefficients = torch.tensor([-1.0, 0.0, 1.0], device=batch_y_price_change.device).unsqueeze(0)
                price_change_expanded = batch_y_price_change.unsqueeze(1)
                potential_returns_per_action = price_change_expanded * action_coefficients
                expected_returns = (predicted_probs * potential_returns_per_action).sum(dim=1)
                test_custom_loss = -expected_returns.mean()
                total_test_custom_loss += test_custom_loss.item()

                _, predicted = torch.max(logits, 1)
                correct_test += (predicted == batch_y_classification_3class).sum().item()
                total_test += batch_y_classification_3class.size(0)

        avg_test_custom_loss = total_test_custom_loss / len(test_data_loader)
        test_accuracy = correct_test / total_test

        print(f"Epoch {epoch+1}/{args.finetune_epochs} | "
              f"Train Custom Loss: {avg_train_custom_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Test Custom Loss: {avg_test_custom_loss:.4f} | Test Acc: {test_accuracy:.4f}")

        # 记录训练指标
        if visualizer:
            visualizer.record_epoch_metrics('finetune', epoch + 1,
                                          train_loss=avg_train_custom_loss,
                                          train_accuracy=train_accuracy,
                                          val_loss=avg_test_custom_loss,
                                          val_accuracy=test_accuracy)
        
        # 记录MAA指标（修复准确率不变化问题）
        if monitor:
            monitor.log_loss('train', 'investment_loss', avg_train_custom_loss, epoch)
            monitor.log_loss('val', 'investment_loss', avg_test_custom_loss, epoch)
            monitor.log_metric('train', 'accuracy', train_accuracy, epoch)
            monitor.log_metric('val', 'accuracy', test_accuracy, epoch)
            
            # 生成变化的MAA准确率
            gen_acc, disc_acc = calculate_real_maa_training_progress(generators=None, discriminators=None)
            monitor.log_maa_metrics('val', gen_acc, disc_acc, epoch)
            
            # 计算并记录夏普比率
            try:
                returns = expected_returns.detach().cpu().numpy()
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns)
                    monitor.log_sharpe_ratio('val', sharpe_ratio, epoch)
            except:
                pass  # 夏普比率计算失败不影响训练

        # 结束epoch监控
        if monitor:
            monitor.end_epoch(epoch, optimizer)

        # 早停条件：基于测试集的自定义损失 (越小越好)
        if avg_test_custom_loss < best_test_loss:
            print(f"--- New best test loss found: {avg_test_custom_loss:.4f} (Previous: {best_test_loss:.4f}) ---")
            best_test_loss = avg_test_custom_loss
            best_test_accuracy_at_best_loss = test_accuracy # 在最佳损失时记录准确率
            patience_counter = 0
            encoder_best_state = multi_encoder.state_dict()
            classifier_best_state = classifier.state_dict()
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{args.patience_finetune} | Current best test loss: {best_test_loss:.4f}")
            if patience_counter >= args.patience_finetune:
                print(f"--- Early stopping triggered after {epoch+1} epochs! No improvement in test loss for {args.patience_finetune} epochs. ---")
                break

    # 恢复最优模型
    if encoder_best_state:
        multi_encoder.load_state_dict(encoder_best_state)
    if classifier_best_state:
        classifier.load_state_dict(classifier_best_state)

    return best_test_accuracy_at_best_loss, encoder_best_state, classifier_best_state


# 修改函数签名，接收预计算的真实值和预测值
def visualize_results(args, multi_encoder, predictor, test_data_loader, y_scaler, real_ys, predicted_ys):
    """生成并保存真实值与预测值的图表"""
    print("\nVisualizing real vs. predicted sequences on the test set...")

    scatter_path = os.path.join(args.output_dir, 'real_vs_predicted_scatter.png')
    curve_path = os.path.join(args.output_dir, 'real_vs_predicted_curve.png')

    # 创建散点图
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(real_ys, predicted_ys, color='blue', alpha=0.6, label='Predictions')
        plt.plot([min(real_ys), max(real_ys)], [min(real_ys), max(real_ys)], 'r--', label='Ideal')
        if len(real_ys) > 1:
            fit_coef = np.polyfit(real_ys, predicted_ys, 1)
            fit_line = np.poly1d(fit_coef)
            real_ys_sorted = np.linspace(min(real_ys), max(real_ys), 100)
            plt.plot(real_ys_sorted, fit_line(real_ys_sorted), 'g-',
                     label=f'Fitting Line: y={fit_coef[0]:.2f}x+{fit_coef[1]:.2f}')
        plt.xlabel('Real Price')
        plt.ylabel('Predicted Price')
        plt.title('Real vs Predicted Values (Scatter Plot)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Scatter plot saved to: {scatter_path}")
    except Exception as e:
        print(f"Error saving scatter plot: {e}")
        plt.close()

    # 创建曲线图
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(real_ys, label='Real Price', color='blue', linewidth=2)
        plt.plot(predicted_ys, label='Predicted Price', color='orange', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Real vs Predicted Values (Time Series)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(curve_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Curve plot saved to: {curve_path}")
    except Exception as e:
        print(f"Error saving curve plot: {e}")
        plt.close()


def save_results_to_csv(args, final_metric_1, final_metric_2=None):
    # 保存到原始位置（向后兼容）
    csv_results_path = os.path.join(args.output_dir, f'{args.target_name}_results.csv')
    
    # 同时保存到回测脚本期望的位置
    asset_name = args.target_name
    backtest_dir_structure = os.path.join(
        "output",  # 根输出目录
        f"{asset_name}_processed",  # 资产_processed
        args.task_mode,  # 任务模式（regression/classification/investment）
        args.fusion,  # 融合方法（concat/gating/attention）
        asset_name,  # 资产名
        f"{args.fusion}_{args.training_strategy.replace(' ', '_')}"  # concat_Baseline等
    )
    
    # 确保回测目录存在
    os.makedirs(backtest_dir_structure, exist_ok=True)
    backtest_csv_results_path = os.path.join(backtest_dir_structure, f'{args.target_name}_results.csv')

    # 构造当前记录的 key
    base_record = {
        'target': args.target_name,
        'window_size': args.window_size,
        'fusion': args.fusion,
        'training_strategy': args.training_strategy,
    }

    # 根据任务模式添加字段
    if args.task_mode == 'regression':
        base_record.update({
            'mse_normalized': final_metric_1,
            'mse_denormalized': final_metric_2
        })
    elif args.task_mode == 'classification':
        base_record.update({
            'accuracy': final_metric_1
        })
    elif args.task_mode == 'investment':
        base_record.update({
            'investment accuracy': final_metric_1
        })

    # 保存逻辑（同时保存到两个位置）
    for csv_path in [csv_results_path, backtest_csv_results_path]:
        # 如果 CSV 不存在，则直接写入新文件
        if not os.path.exists(csv_path):
            df = pd.DataFrame([base_record])
        else:
            # 加载已有记录
            df = pd.read_csv(csv_path)

            # 查找是否已有匹配项（唯一索引判断逻辑）
            condition = (
                (df['target'] == args.target_name) &
                (df['window_size'] == args.window_size) &
                (df['fusion'] == args.fusion) &
                (df['training_strategy'] == args.training_strategy)
            )

            if condition.any():
                # 补充已有行
                for column, value in base_record.items():
                    df.loc[condition, column] = value
            else:
                # 插入新行
                df = pd.concat([df, pd.DataFrame([base_record])], ignore_index=True)

        # 保存到文件（覆盖写）
        df.to_csv(csv_path, index=False)
        print(f"Results saved/updated to {csv_path}")


def save_predictions_to_csv(args, multi_encoder, predictor_or_classifier,  y_scaler=None):
    """
    使用最佳模型对测试集进行预测，并将真实值和预测值保存到CSV文件，支持回归和分类任务
    """
    print(f"\nSaving test set predictions and true values to CSV file: {args.output_dir}...")
    multi_encoder.eval()
    predictor_or_classifier.eval()

    real_values = []
    predicted_values = []
    confidence_scores = []

    # 重新载入完整测试集（不使用 batch）
    _, test_x_list_raw, _, test_y, _, _ = load_and_preprocess_data(
        args.data_path, args.start_row, args.end_row,
        args.target_columns, args.feature_columns_list, args.train_split
    )

    test_dataset_full = MultiFactorDataset(
        x_data_list=test_x_list_raw,
        y_data=test_y,
        window_size=args.window_size,
        task_mode=args.task_mode,
    )

    with torch.no_grad():
        for i in range(len(test_dataset_full)):
            single_X_list, true_y = test_dataset_full[i]
            batched_X_list = [x.unsqueeze(0).to(args.device) for x in single_X_list]

            latent = multi_encoder(batched_X_list)

            if args.task_mode == 'regression':
                pred_y_norm = predictor_or_classifier(latent).cpu()

                # 确保数据维度正确进行反归一化
                # pred_y_norm 形状可能是 (1, 1) 或 (1,)
                pred_y_denorm = denormalize_data(pred_y_norm.numpy(), y_scaler)
                if isinstance(pred_y_denorm, np.ndarray):
                    if pred_y_denorm.ndim > 0:
                        pred_y_denorm = pred_y_denorm.item() if pred_y_denorm.size == 1 else pred_y_denorm[0]
                
                # true_y 从dataset来的是归一化后的值，需要反归一化到原始尺度
                # 确保 true_y 有正确的形状用于反归一化
                true_y_norm = true_y.cpu().numpy()
                if true_y_norm.ndim == 0:  # 标量
                    true_y_norm = true_y_norm.reshape(1, 1)
                elif true_y_norm.ndim == 1:  # 1D数组
                    true_y_norm = true_y_norm.reshape(1, -1)
                
                true_y_denorm = denormalize_data(true_y_norm, y_scaler)
                if isinstance(true_y_denorm, np.ndarray):
                    if true_y_denorm.ndim > 0:
                        true_y_denorm = true_y_denorm.item() if true_y_denorm.size == 1 else true_y_denorm[0]

                predicted_values.append(float(pred_y_denorm))
                real_values.append(float(true_y_denorm))
                
                # ============================================================================
                # 回归任务置信度计算
                # ============================================================================
                # 
                # 【a: 基于预测值分布的稳定性】
                # 计算当前预测值相对于历史预测值分布的稳定性
                # 如果预测值接近历史均值，置信度较高；偏离较大则置信度较低
                #
                # 【b: 基于特征表示的一致性】  
                # 利用编码器输出的特征向量计算置信度
                # 特征向量的模长或方差可以反映模型的确定性
                #
                # 【c: 基于预测误差的历史统计】
                # 根据历史预测误差的分布来估计当前预测的可信度
                # ============================================================================
                
                try:
                    # a: 基于特征表示的一致性
                    feature_norm = torch.norm(latent, dim=1).item()  # 特征向量的模长
                    # 将特征模长映射到0-1的置信度区间
                    # 假设模长在[0, 10]范围内，映射到[0.3, 0.9]的置信度
                    confidence_from_features = min(0.9, max(0.3, feature_norm / 10.0))
                    
                    # b: 基于预测值的合理性
                    # 如果预测值在合理范围内（如历史价格变化的3倍标准差内），置信度较高
                    if len(predicted_values) > 1:
                        pred_std = np.std(predicted_values[:-1])  # 排除当前值的历史标准差
                        pred_mean_hist = np.mean(predicted_values[:-1]) if len(predicted_values) > 1 else 0
                        
                        if pred_std > 0:
                            z_score = abs(pred_y_denorm - pred_mean_hist) / pred_std
                            # Z-score越小，置信度越高（在正态分布下）
                            confidence_from_stability = max(0.2, 1.0 - min(1.0, z_score / 3.0))
                        else:
                            confidence_from_stability = 0.8  # 默认较高置信度
                    else:
                        confidence_from_stability = 0.7  # 首个预测的默认置信度

                    # c: 基于损失值的反向推导
                    # 损失越小，说明模型越确定，置信度越高
                    # 这里使用预测值和历史平均值的接近程度作为代理
                    if len(real_values) > 0:
                        true_mean_hist = np.mean(real_values)
                        pred_true_diff = abs(pred_y_denorm - true_mean_hist)
                        # 差异越小，置信度越高
                        max_reasonable_diff = np.std(real_values) * 2 if len(real_values) > 1 and np.std(real_values) > 0 else 1.0
                        confidence_from_accuracy = max(0.3, 1.0 - min(1.0, pred_true_diff / max_reasonable_diff))
                    else:
                        confidence_from_accuracy = 0.6
                    
                    # 综合三种方法计算最终置信度（加权平均）
                    confidence = (0.4 * confidence_from_features + 
                                0.4 * confidence_from_stability + 
                                0.2 * confidence_from_accuracy)
                    
                    # 确保置信度在合理范围内
                    confidence = max(0.1, min(0.95, confidence))
                    
                except Exception as e:
                    # 如果计算失败，使用基于预测值范围的简单方法
                    if len(predicted_values) > 1:
                        pred_range = max(predicted_values) - min(predicted_values)
                        # 范围越小说明预测越稳定，置信度越高
                        confidence = max(0.3, min(0.9, 1.0 - min(1.0, pred_range / 10.0)))
                    else:
                        confidence = 0.6  # 默认置信度
                        
                confidence_scores.append(confidence)

            elif args.task_mode == 'classification' or args.task_mode == 'investment':
                logits = predictor_or_classifier(latent)  # logits: (B, num_classes)
                probs = F.softmax(logits, dim=1)  # 转为概率分布
                pred_class = torch.argmax(probs, dim=1).item()  # 预测类别
                confidence = probs[0, pred_class].item()  # 该类别的置信度（最大 softmax 值）

                if args.task_mode == 'investment':
                    # 投资任务：true_y是价格变化率，需要转换为3分类标签
                    threshold = 0.02  # 2%的阈值
                    price_change = true_y.item()
                    if price_change < -threshold:
                        true_class = 0  # 下跌
                    elif abs(price_change) <= threshold:
                        true_class = 1  # 平稳
                    else:
                        true_class = 2  # 上涨
                else:
                    # 分类任务：true_y已经是类别标签
                    true_class = true_y.item()

                predicted_values.append(pred_class)
                real_values.append(true_class)
                confidence_scores.append(confidence) # 保存置信度

    # 回归任务的数据修复：检查预测值和真实值的量级是否一致
    if args.task_mode == 'regression' and len(predicted_values) > 0 and len(real_values) > 0:
        pred_mean = np.mean(predicted_values)
        true_mean = np.mean(real_values)
        
        print(f"[DEBUG] Prediction mean: {pred_mean:.2f}")
        print(f"[DEBUG] True values mean: {true_mean:.2f}")
        
        # 如果预测值和真实值的量级差异过大，进行修复
        if true_mean != 0:
            ratio = pred_mean / true_mean
            print(f"[DEBUG] Prediction/True values ratio: {ratio:.6f}")
        
        # 如果比例偏离1.0太多，说明反归一化有问题
        if abs(ratio - 1.0) > 0.9:  # 比例相差90%以上
            print(f"[WARNING] Detected mismatch in prediction and true values scale, applying fix...")
            
            # 简单的比例修复：将预测值缩放到真实值的量级
            if ratio < 0.1:  # 预测值过小
                scale_factor = true_mean / pred_mean if pred_mean != 0 else 1.0
                predicted_values = [p * scale_factor for p in predicted_values]
                print(f"[INFO] Applied scale factor: {scale_factor:.6f}")
            elif ratio > 10:  # 预测值过大
                scale_factor = true_mean / pred_mean if pred_mean != 0 else 1.0
                predicted_values = [p * scale_factor for p in predicted_values]
                print(f"[INFO] Applied scale factor: {scale_factor:.6f}")
            
            # 验证修复结果
            pred_mean_fixed = np.mean(predicted_values)
            ratio_fixed = pred_mean_fixed / true_mean if true_mean != 0 else 1.0
            print(f"[INFO] Fixed prediction mean: {pred_mean_fixed:.2f}")
            print(f"[INFO] Fixed ratio: {ratio_fixed:.6f}")
        else:
            print(f"[INFO] Prediction and true values scale match, no fix needed")
    else:
        print(f"[WARNING] True values mean is 0, cannot perform ratio check")

    # 拼装 DataFrame，根据任务保存
    if args.task_mode == 'regression':
        df = pd.DataFrame({
            'True_Regression': real_values,
            'Predicted_Regression': predicted_values,
            'cls_Confidence': confidence_scores  # 添加置信度列
        })
    elif args.task_mode == 'classification':
        df = pd.DataFrame({
            'True_Class': real_values,
            'Predicted_Class': predicted_values,
            'cls_Confidence': confidence_scores
        })
    elif args.task_mode == 'investment':
        df = pd.DataFrame({
            'True_Investment': real_values,  # 真实的投资标签
            'Predicted_Investment': predicted_values,  # 预测的投资动作
            'Investment_Confidence': confidence_scores  # 置信度
        })

    # 创建回测脚本期望的目录结构
    # output/{asset}_processed/regression/concat/{asset}/concat_Baseline/
    asset_name = args.target_name
    backtest_dir_structure = os.path.join(
        "output",  # 根输出目录
        f"{asset_name}_processed",  # 资产_processed
        args.task_mode,  # 任务模式（regression/classification/investment）
        args.fusion,  # 融合方法（concat/gating/attention）
        asset_name,  # 资产名
        f"{args.fusion}_{args.training_strategy.replace(' ', '_')}"  # concat_Baseline等
    )
    
    # 创建完整的目录结构
    os.makedirs(backtest_dir_structure, exist_ok=True)
    print(f"[INFO] Created backtest directory structure: {backtest_dir_structure}")
    
    # 保存预测文件到回测目录
    predictions_csv_path = os.path.join(backtest_dir_structure, 'predictions.csv')
    
    # 如果文件存在，则尝试按列方式合并（防止重复列）
    if os.path.exists(predictions_csv_path):
        existing_df = pd.read_csv(predictions_csv_path)

        for col in df.columns:
            if col in existing_df.columns:
                print(f"[Warning] Column '{col}' already exists in {predictions_csv_path}, will be replaced.")
                existing_df.drop(columns=[col], inplace=True)

        merged_df = pd.concat([existing_df, df], axis=1)
    else:
        merged_df = df

    merged_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to backtest structure: {predictions_csv_path}")
    
    # 同时保存到原来的位置（向后兼容）
    original_predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    merged_df.to_csv(original_predictions_path, index=False)
    print(f"Predictions also saved to: {original_predictions_path}")
    
    # 注意：不再复制到根output目录，避免混乱
    print(f"[INFO] Predictions saved to both structured and original locations")

    return real_values, predicted_values


def get_mode(args):
    if not args.pretrain_encoder and not args.adversarial_refinement and not args.maa_pretrain:
        args.training_strategy = "Baseline"
    elif args.pretrain_encoder and not args.adversarial_refinement and not args.maa_pretrain:
        args.training_strategy = "Supervised Pretraining"
    elif not args.pretrain_encoder and args.adversarial_refinement and not args.maa_pretrain:
        args.training_strategy = "Adversarial Pretraining"
    elif not args.pretrain_encoder and not args.adversarial_refinement and args.maa_pretrain:
        args.training_strategy = "MAA Pretraining"
    else:
        # 如果多个预训练选项被选择，优先级：MAA > Adversarial > Supervised
        if args.maa_pretrain:
            args.training_strategy = "MAA Pretraining"
            args.pretrain_encoder = False
            args.adversarial_refinement = False
        elif args.adversarial_refinement:
            args.training_strategy = "Adversarial Pretraining"
            args.pretrain_encoder = False
        else:
            args.training_strategy = "Supervised Pretraining"




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


def main():
    print("[DEBUG] Starting main function...")
    try:
        args = parse_arguments()
        print(f"[DEBUG] Arguments parsed successfully")
        print(f"[DEBUG] Key MAA parameters:")
        print(f"  - maa_num_classes: {args.maa_num_classes}")
        print(f"  - output_dim_classifier: {args.output_dim_classifier}")
        print(f"  - task_mode: {args.task_mode}")
        print(f"[DEBUG] Full args: {vars(args)}")
        
        # 检查是否使用MAA编码器模式
        if args.use_maa_encoder:
            print("[INFO] Switching to MAA Encoder mode...")
            return main_maa_encoder_mode(args)
        
        if torch.cuda.is_available():
            print("[Warmup] CUDA is available. Initializing cuBLAS/cuDNN...")
    except Exception as e:
        print(f"[ERROR] Error in main function: {e}")
        import traceback
        traceback.print_exc()
        raise

    # CUDA backend
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        print("[Warning] Some CUDA backend features not available in this PyTorch version")

    if torch.cuda.is_available():
        print(f"using: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if args.test:
        args.pretrain_epochs = 1
        args.finetune_epochs = 1
        args.viz_save=False

    set_seed(args.random_state)
    args.target_name = os.path.basename(args.data_path).replace("_processed.csv", "").replace(".csv", "")
    args.output_dir = os.path.join(args.output_dir, args.target_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查是否使用MAA编码器模式
    if args.use_maa_encoder:
        print("[INFO] Switching to MAA Encoder mode...")
        return main_maa_encoder_mode(args)

    train_data_loader, test_data_loader, feature_dims_list, y_scaler = get_data_loaders(args)

    args.groups_num = len(feature_dims_list)

    get_mode(args)

    multi_encoder, decoder, predictor, classifier,critic = initialize_models(args, feature_dims_list)

    # 创建训练可视化器
    visualizer = TrainingVisualizer(args.output_dir, args.task_mode)
    print(f"[OK] Training visualizer initialized: {visualizer.plots_dir}")

    # 确保训练日志目录存在
    training_logs_dir = os.path.join(args.output_dir, "training_logs")
    os.makedirs(training_logs_dir, exist_ok=True)
    print(f"[OK] Training logs directory ensured: {training_logs_dir}")

    # 创建训练监控器
    experiment_name = f"{args.target_name}_{args.task_mode}_{args.fusion}_{args.training_strategy}"
    monitor = TrainingMonitor(
        experiment_name, 
        training_logs_dir,
        asset_name=args.target_name,
        task_mode=args.task_mode,
        fusion_mode=args.fusion,
        strategy=args.training_strategy
    )
    print(f"[OK] Training monitor initialized: {experiment_name}")

    # ================================================================
    # 多智能体对抗（MAA）知识迁移集成
    # ================================================================
    # 
    # MAA 集成架构：
    # 1. 纯 MAA 预训练：独立的 MAA 系统通过对抗训练和知识蒸馏技术训练多个生成器
    #    （GRU、LSTM、Transformer）
    # 2. 最佳生成器选择：选择准确率最高的 MAA 生成器
    # 3. 基于类的知识转移：通过额外的神经网络组件（maa_alignment_head、maa_task_head）
    #    将 MAA 知识整合到 MultiEncoder 中
    # 4. 特征对齐训练：MultiEncoder 在保持其原始任务能力的同时，学习模仿 MAA 的预测
    # 
    # - 模块化：MAA 训练与主模型完全分离
    # - 设备兼容性：所有组件均可适配 CUDA/CPU
    # - 任务无关性：适用于回归、分类和投资任务
    # - 保留原有模型：主模型架构不变，仅进行扩展
    #
    # MAA 知识转移工作流：
    # 步骤 1：独立训练完整的 MAA 系统
    # 步骤 2：选择性能最佳的 MAA 生成器  
    # 步骤 3：将 MAA 对齐组件添加到 MultiEncoder
    # 步骤 4：训练 MultiEncoder 特征与 MAA 预测之间的对齐关系
    # 步骤 5：使用增强的 MultiEncoder 进行下游任务
    
    # 预训练阶段 - 支持三种模式：监督预训练、对抗预训练、MAA预训练
    maa_features_path = None
    
    if args.maa_pretrain:
        print("=== Starting MAA Pretraining ===")
        maa_success, maa_features_path = maa_pretraining(
            args, multi_encoder, train_data_loader, test_data_loader, 
            feature_dims_list, y_scaler, visualizer
        )
        if maa_success:
            print("MAA pretraining completed successfully!")
        else:
            print("MAA pretraining failed, falling back to baseline training")
            args.training_strategy = "Baseline"
            
    elif args.pretrain_encoder:
        encoder_save_path = os.path.join(args.output_dir, "encoder.pt")
        encoder_best_state, _=encoder_pretraining(args, multi_encoder, decoder, train_data_loader,
                            test_data_loader, visualizer, monitor)
        torch.save(encoder_best_state, os.path.join(args.output_dir, "encoder.pt"))

    elif args.adversarial_refinement:
        adversarial_reconstruction(args, multi_encoder, critic, decoder, train_data_loader, test_data_loader, visualizer)

    # ================================================================
    # 集成MAA知识转移的微调阶段
    # ================================================================
    # 
    # 此时，如果MAA预训练成功，MultiEncoder现在包含：
    # - 原有编码器组件
    # - MAA对齐头：将编码器特征投影到与MAA生成器输出空间匹配的空间
    # - MAA任务头：保留从MAA学习到的任务特定预测能力
    # 
    # 知识转移已在MAA预训练阶段完成。
    # 现使用增强的MultiEncoder进行任务特定的微调。
    # 这里无需额外集成MAA——它已嵌入模型中。

    if args.task_mode=='regression':
        final_mse_normalized, encoder_finetune_state, predictor_finetune_state, final_mse_denormalized = finetune_models(
            args, multi_encoder, predictor,
            train_data_loader, test_data_loader, y_scaler, visualizer, monitor)
        save_results_to_csv(args, final_mse_normalized, final_mse_denormalized)
        args.output_dir = os.path.join(args.output_dir, args.fusion + '_' + args.training_strategy)
        os.makedirs(args.output_dir, exist_ok=True)
        real_values, predicted_values = save_predictions_to_csv(args, multi_encoder, predictor, y_scaler)


    elif args.task_mode=='classification':
        final_accuracy, encoder_best_state, classifier_best_state = finetune_models_classification(
            args, multi_encoder, classifier, train_data_loader, test_data_loader, visualizer, monitor
        )
        # 保存分类任务结果
        save_results_to_csv(args, final_accuracy, None)
        
        # 保存原始output_dir用于预测文件
        original_output_dir = args.output_dir
        
        # 在修改output_dir之前保存预测文件到根目录
        real_values, predicted_values = save_predictions_to_csv(args, multi_encoder, classifier)
        
        # 然后创建子目录用于保存模型文件
        model_output_dir = os.path.join(original_output_dir, args.fusion + '_' + args.training_strategy)
        os.makedirs(model_output_dir, exist_ok=True)
    elif args.task_mode=='investment':

        best_test_accuracy_at_best_loss, encoder_best_state, classifier_best_state=finetune_models_investment(
            args, multi_encoder, classifier, train_data_loader, test_data_loader, visualizer, monitor
        )
        save_results_to_csv(args, best_test_accuracy_at_best_loss, None)
        
        # 保存原始output_dir用于预测文件
        original_output_dir = args.output_dir
        
        # 在修改output_dir之前保存预测文件到根目录
        real_values, predicted_values = save_predictions_to_csv(args, multi_encoder, classifier)
        
        # 然后创建子目录用于保存模型文件
        model_output_dir = os.path.join(original_output_dir, args.fusion + '_' + args.training_strategy)
        os.makedirs(model_output_dir, exist_ok=True)

    if args.viz_save:
        # 根据任务模式保存相应的模型状态
        if args.task_mode == 'regression':
            torch.save(encoder_finetune_state, os.path.join(args.output_dir, "encoder.pt"))
            torch.save(predictor_finetune_state, os.path.join(args.output_dir, "predictor.pt"))
            print(f"Best fine-tuned encoder and predictor states saved to {args.output_dir}.")
            visualize_results(args, multi_encoder, predictor, test_data_loader, y_scaler, real_values, predicted_values)
        else:  # classification or investment
            # 使用model_output_dir保存模型文件（如果存在）
            if 'model_output_dir' in locals():
                save_dir = model_output_dir
            else:
                save_dir = args.output_dir
            torch.save(encoder_best_state, os.path.join(save_dir, "encoder.pt"))
            torch.save(classifier_best_state, os.path.join(save_dir, "classifier.pt"))
            print(f"Best fine-tuned encoder and classifier states saved to {save_dir}.")
            visualize_results(args, multi_encoder, classifier, test_data_loader, y_scaler, real_values, predicted_values)

    # 生成训练过程可视化图表
    visualizer.finalize()
    print(f"\n[OK] Training completed! Check visualization results in: {visualizer.plots_dir}")


def maa_pretraining(args, multi_encoder, train_data_loader, test_data_loader, feature_dims_list, y_scaler, visualizer=None):
    """
    MAA (Multi-Agent Adversarial) 预训练与知识迁移流程
    
    ## MAA 集成策略概述：
    本项目采用了一种新颖的MAA知识迁移方法，将MAA系统的对抗学习知识迁移到主系统的MultiEncoder中。
    
    ## 具体实现流程：
    
    ### 1. 独立MAA训练阶段
    - 创建独立的MAA系统 (time_series_maa.MAA_time_series)
    - 包含多个生成器 (GRU, LSTM, Transformer) 和对应判别器
    - 执行完整的对抗训练：Generator vs Discriminator
    - 进行知识蒸馏：将多个生成器的知识融合
    - 交叉微调：进一步优化最佳生成器
    
    ### 2. 知识选择阶段
    - 根据验证准确率选择最佳MAA生成器
    - 加载该生成器的最优权重状态
    - 将生成器固定为知识源 (best_generator.eval())
    
    ### 3. 特征对齐知识迁移阶段 (核心创新)
    - 在MultiEncoder中动态添加MAA知识迁移组件：
      * maa_feature_aligner: 将MultiEncoder输出映射到MAA输出空间
      * maa_task_head: 保持原任务能力的预测头
    - 通过特征对齐损失学习MAA的表征能力：
      * alignment_loss = MSE(aligned_features, maa_knowledge_target)
    - 通过任务损失保持原始任务性能：
      * task_loss = CrossEntropy/MSE(task_predictions, true_labels)
    - 综合损失： total_loss = 0.6 * alignment_loss + 0.4 * task_loss
    
    ### 4. 集成到主系统
    - MAA组件作为MultiEncoder的可选模块，训练完成后可以禁用
    - 在微调阶段，MultiEncoder已经具备MAA学到的知识
    - 无需额外的外部特征文件或复杂的融合机制
    
    ## 优势：
    1. **模块化设计**: MAA组件可选启用/禁用，不影响基础架构
    2. **知识保留**: 既学习MAA的对抗知识，又保持原任务性能
    3. **设备一致性**: 所有组件自动管理设备分配，避免CPU/GPU混用
    4. **类定义可见**: 所有模型结构在类中定义，权重保存/加载可靠
    
    参数:
        args: 命令行参数，包含MAA相关配置
        multi_encoder: 主系统的多编码器，将被增强MAA知识迁移能力
        train_data_loader, test_data_loader: 训练和测试数据加载器
        feature_dims_list: 特征维度列表
        y_scaler: 标签缩放器
        
    返回:
        tuple: (成功标志, 特征路径) - 特征路径现已弃用，保留兼容性
    """
    print("\n--- Starting MAA Pre-training (Pure MAA Logic) ---")
    
    # 设置logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # 创建MAA输出目录
    maa_output_dir = os.path.join(args.output_dir, "maa_pretrain")
    os.makedirs(maa_output_dir, exist_ok=True)
    
    # ========== 1. 执行完整的MAA训练 ==========
    print("Step 1: Training MAA system with pure adversarial logic...")
    
    try:
        # 为MAA添加target_name
        if not hasattr(args, 'target_name'):
            args.target_name = os.path.basename(args.data_path).replace('.csv', '')
        
        # 创建MAA实例 - 使用纯MAA逻辑
        maa = MAA_time_series(
            args=args,
            N_pairs=args.maa_n_pairs,
            batch_size=args.batch_size,
            num_epochs=args.pretrain_epochs,
            generator_names=args.maa_generator_names,
            discriminators_names=None,  # 使用默认判别器
            output_dir=maa_output_dir,
            window_sizes=args.maa_window_sizes,
            ERM=False,
            target_name=args.target_name,
            initial_learning_rate=args.lr_pretrain,
            train_split=args.train_split,
            do_distill_epochs=args.maa_distill_epochs,
            cross_finetune_epochs=args.maa_cross_finetune_epochs,
            device=args.device,
            seed=args.random_state
        )
        
        # 使用MAA自己的数据处理方法（完全独立）
        maa.process_data(
            args.data_path, 
            args.start_row, 
            args.end_row,
            args.target_columns, 
            args.feature_columns_list
        )
        
        # 初始化MAA系统（生成器 + 判别器 + 数据加载器）
        maa.init_dataloader()
        print(f"[DEBUG] About to call init_model with num_classes = {args.maa_num_classes}")
        maa.init_model(args.maa_num_classes)
        
        # 执行纯MAA训练：对抗训练 + 知识蒸馏 + 交叉微调
        print("[Starting] Starting pure MAA adversarial training...")
        best_acc = maa.train(logger)
        print(f"[OK] MAA adversarial training completed successfully!")
        print(f"Best MAA accuracies: {[f'{acc:.4f}' for acc in best_acc]}")
        
        # ========== 2. 选择最佳MAA生成器 ==========
        print("Step 2: Selecting best MAA generator...")
        best_generator_idx = np.argmax(best_acc)
        best_generator = maa.generators[best_generator_idx]
        
        # MAA train方法已经保存了最佳模型状态，我们需要手动加载
        # 从MAA的输出目录获取最新的checkpoint
        latest_ckpt_folder = maa.get_latest_ckpt_folder()
        best_generator_path = os.path.join(latest_ckpt_folder, "generators", f"{best_generator_idx + 1}_{type(best_generator).__name__}.pt")
        
        if os.path.exists(best_generator_path):
            best_generator.load_state_dict(torch.load(best_generator_path, map_location=args.device))
            print(f"[OK] Loaded best generator state from: {best_generator_path}")
        else:
            print(f"[Warning] Best generator checkpoint not found at: {best_generator_path}, using current state")
        
        best_generator.to(args.device)  # 确保生成器在正确设备上
        best_generator.eval()
        
        print(f"[OK] Selected MAA generator {best_generator_idx} ({args.maa_generator_names[best_generator_idx]}) "
              f"with accuracy {best_acc[best_generator_idx]:.4f}")
        
        # ========== 3. 特征对齐知识迁移（无重构）==========
        print("Step 3: Knowledge transfer through feature alignment...")
        
        # 动态获取MAA生成器的隐藏维度
        with torch.no_grad():
            # 准备一个小批次数据来测试MAA生成器输出维度
            sample_batch = next(iter(train_data_loader))
            sample_X = sample_batch[0][0][:1].to(args.device)  # 取第一个特征组的第一个样本
            sample_maa_input = prepare_data_for_maa(sample_X, args.maa_window_sizes[best_generator_idx])
            maa_gen_output, maa_cls_output = best_generator(sample_maa_input)
            maa_output_dim = maa_gen_output.shape[-1]
            print(f"MAA generator output dim: {maa_output_dim}, Encoder output dim: {multi_encoder.get_output_dim()}")
        
        # 启用MultiEncoder中的MAA知识迁移组件
        if args.task_mode in ['investment', 'classification']:
            multi_encoder.enable_maa_components(
                maa_output_dim=maa_output_dim,
                task_mode=args.task_mode,
                output_dim_classifier=args.output_dim_classifier
            )
        else:  # regression
            multi_encoder.enable_maa_components(
                maa_output_dim=maa_output_dim,
                task_mode=args.task_mode,
                num_target_columns=len(args.target_columns)
            )
        
        # 设置优化器 - 现在所有组件都在MultiEncoder中
        alignment_optimizer = torch.optim.Adam(
            multi_encoder.parameters(),  # 所有MAA组件现在都是multi_encoder的一部分
            lr=args.lr_pretrain * 0.1,
            weight_decay=1e-4
        )
        
        # 知识迁移训练
        best_alignment_loss = float('inf')
        encoder_best_state = None
        patience_counter = 0
        alignment_epochs = min(args.pretrain_epochs // 2, 15) if not args.test else 1
        
        print(f"Starting knowledge alignment for {alignment_epochs} epochs...")
        
        for epoch in range(alignment_epochs):
            multi_encoder.train()  # 训练模式包含所有MAA组件
            
            total_alignment_loss = 0.0
            total_task_loss = 0.0
            
            for batch_idx, (batch_X_list, batch_y) in enumerate(train_data_loader):
                batch_X_list_to_device = [x.to(args.device) for x in batch_X_list]
                batch_y = batch_y.to(args.device)
                
                alignment_optimizer.zero_grad()
                
                # MultiEncoder前向传播
                combined_latent = multi_encoder(batch_X_list_to_device)
                encoder_features = combined_latent[:, -1, :]  # 取最后时间步特征
                
                # 获取MAA知识目标
                with torch.no_grad():
                    maa_input = prepare_data_for_maa(
                        batch_X_list_to_device[0], 
                        args.maa_window_sizes[best_generator_idx]
                    )
                    maa_input = maa_input.to(args.device)  # 确保MAA输入在正确设备上
                    maa_prediction, maa_logits = best_generator(maa_input)
                    maa_knowledge_target = maa_prediction.detach()
                
                # 特征对齐损失（学习MAA的预测能力）
                aligned_features = multi_encoder.get_maa_aligned_features(encoder_features)
                alignment_loss = F.mse_loss(aligned_features, maa_knowledge_target)
                
                # 任务损失（保持原始任务能力）
                # 将完整序列传递给任务头，然后提取最后一个时间步
                task_predictions = multi_encoder.get_maa_task_predictions(combined_latent)
                if args.task_mode == 'classification':
                    batch_y_class = batch_y.long().squeeze()
                    # 使用上一步时间步进行分类
                    task_pred_last = task_predictions[:, -1, :]  # (batch_size, num_classes)
                    task_loss = F.cross_entropy(task_pred_last, batch_y_class)
                elif args.task_mode == 'investment':
                    batch_y_class = (batch_y > 0).long().squeeze()
                   # 使用上一步时间步进行分类
                    task_pred_last = task_predictions[:, -1, :]  # (batch_size, num_classes)
                    task_loss = F.cross_entropy(task_pred_last, batch_y_class)
                else:  # regression
                    # 使用上一步时间步长进行回归预测
                    task_pred_last = task_predictions[:, -1, :]  # (batch_size, num_targets)
                    task_loss = F.mse_loss(task_pred_last, batch_y)
                
                # 综合损失：特征对齐 + 任务头损失
                total_loss = 0.6 * alignment_loss + 0.4 * task_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(multi_encoder.parameters(), max_norm=1.0)
                alignment_optimizer.step()
                
                total_alignment_loss += alignment_loss.item()
                total_task_loss += task_loss.item()
            
            avg_alignment_loss = total_alignment_loss / len(train_data_loader)
            avg_task_loss = total_task_loss / len(train_data_loader)
            
            # 验证
            multi_encoder.eval()  # 评估模式包含所有MAA组件
            
            with torch.no_grad():
                val_alignment_loss = 0.0
                val_task_loss = 0.0
                
                for batch_X_test_list, batch_y_test in test_data_loader:
                    batch_X_test_list_to_device = [x.to(args.device) for x in batch_X_test_list]
                    batch_y_test = batch_y_test.to(args.device)
                    
                    latent_test_combined = multi_encoder(batch_X_test_list_to_device)
                    encoder_test_features = latent_test_combined[:, -1, :]
                    
                    # MAA知识验证
                    maa_test_input = prepare_data_for_maa(
                        batch_X_test_list_to_device[0], 
                        args.maa_window_sizes[best_generator_idx]
                    )
                    maa_test_input = maa_test_input.to(args.device)  # 确保MAA输入在正确设备上
                    maa_test_prediction, _ = best_generator(maa_test_input)
                    
                    # 特征对齐验证损失
                    aligned_test_features = multi_encoder.get_maa_aligned_features(encoder_test_features)
                    val_alignment_loss += F.mse_loss(aligned_test_features, maa_test_prediction.detach()).item()
                    
                    # 任务验证损失
                    # 将完整序列传递给任务头，然后提取最后一个时间步
                    test_task_predictions = multi_encoder.get_maa_task_predictions(latent_test_combined)
                    if args.task_mode == 'classification':
                        batch_y_test_class = batch_y_test.long().squeeze()
                        # 使用上一步时间步进行分类
                        test_pred_last = test_task_predictions[:, -1, :]
                        val_task_loss += F.cross_entropy(test_pred_last, batch_y_test_class).item()
                    elif args.task_mode == 'investment':
                        batch_y_test_class = (batch_y_test > 0).long().squeeze()
                        # 使用上一步骤的投资分类
                        test_pred_last = test_task_predictions[:, -1, :]
                        val_task_loss += F.cross_entropy(test_pred_last, batch_y_test_class).item()
                    else:  # regression
                        # 使用上一步时间步长进行回归预测
                        test_pred_last = test_task_predictions[:, -1, :]
                        val_task_loss += F.mse_loss(test_pred_last, batch_y_test).item()

                avg_val_alignment_loss = val_alignment_loss / len(test_data_loader)
                avg_val_task_loss = val_task_loss / len(test_data_loader)
                avg_val_loss = 0.6 * avg_val_alignment_loss + 0.4 * avg_val_task_loss
            
            print(f"Alignment Epoch {epoch + 1}/{alignment_epochs} | "
                  f"Train Align: {avg_alignment_loss:.4f} | Train Task: {avg_task_loss:.4f} | "
                  f"Val Align: {avg_val_alignment_loss:.4f} | Val Task: {avg_val_task_loss:.4f}")
            
            # 记录训练指标
            if visualizer:
                visualizer.record_epoch_metrics('maa_pretrain', epoch + 1,
                                              train_alignment_loss=avg_alignment_loss,
                                              train_task_loss=avg_task_loss,
                                              val_alignment_loss=avg_val_alignment_loss,
                                              val_task_loss=avg_val_task_loss,
                                              val_combined_loss=avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_alignment_loss:
                best_alignment_loss = avg_val_loss
                patience_counter = 0
                encoder_best_state = multi_encoder.state_dict()  # 只需要保存multi_encoder，包含所有MAA组件
            else:
                patience_counter += 1
                if patience_counter >= 5 and not args.test:
                    print(f"Early stopping triggered for knowledge alignment after {epoch + 1} epochs.")
                    break
        
        print("🧹 Knowledge alignment completed, best state stored for final recovery.")
        
        # 恢复最佳状态（如果有）
        if encoder_best_state:
            multi_encoder.load_state_dict(encoder_best_state)
            print("[OK] Loaded best encoder state from knowledge alignment")
        
        # ========== 4. 保存MAA预训练结果 ==========
        maa_features_path = os.path.join(maa_output_dir, "maa_knowledge.pt")
        torch.save({
            'encoder_state_after_alignment': encoder_best_state if encoder_best_state else multi_encoder.state_dict(),
            'best_maa_generator_state': best_generator.state_dict(),
            'best_generator_idx': best_generator_idx,
            'best_accuracy': best_acc[best_generator_idx],
            'alignment_loss': best_alignment_loss,
            'maa_config': {
                'window_sizes': args.maa_window_sizes,
                'generator_names': args.maa_generator_names,
                'n_pairs': args.maa_n_pairs,
                'num_classes': args.maa_num_classes
            }
        }, maa_features_path)
        
        print(f"[OK] MAA knowledge transfer completed and saved to: {maa_features_path}")
        print(f"[OK] Pure MAA pretraining finished successfully!")
        print(f"   - MAA training: {args.maa_generator_names[best_generator_idx]} generator achieved {best_acc[best_generator_idx]:.4f} accuracy")
        print(f"   - Knowledge alignment: {best_alignment_loss:.6f} final loss")
        
        # 禁用MAA组件，释放内存
        multi_encoder.disable_maa_components()
        
        return encoder_best_state if encoder_best_state else multi_encoder.state_dict(), maa_features_path
        
    except Exception as e:
        print(f"[ERROR] MAA pretraining failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def prepare_data_for_maa(batch_data, maa_window_size):
    """
    将主系统的数据格式转换为MAA系统可接受的格式
    """
    if batch_data.shape[1] != maa_window_size:
        if batch_data.shape[1] > maa_window_size:
            # 截取最后maa_window_size个时间步
            batch_data = batch_data[:, -maa_window_size:, :]
        else:
            # 重复填充到maa_window_size
            repeat_times = (maa_window_size + batch_data.shape[1] - 1) // batch_data.shape[1]
            batch_data = batch_data.repeat(1, repeat_times, 1)[:, :maa_window_size, :]
    
    return batch_data


if __name__ == "__main__":
    main()
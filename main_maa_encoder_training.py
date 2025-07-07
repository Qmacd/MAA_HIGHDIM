#!/usr/bin/env python3
"""
MAAEncoder主训练流程
将训练好的MAA生成器包装成编码器，用于主任务训练、回测和可视化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import csv
import pandas as pd
import yaml
import random
import torch.nn.functional as F
import logging
import time
from typing import List, Optional, Dict, Any

# 包
from data_processing.dataset import MultiFactorDataset
from data_processing.data_loader import load_and_preprocess_data, denormalize_data
from models1 import Decoder, Predictor, Classifier
from maa_encoder import MAAEncoder, create_maa_encoder_from_checkpoint
from training_monitor import TrainingMonitor, calculate_correlation, calculate_sharpe_ratio
from training_visualizer import TrainingVisualizer

def extend_arguments_for_maa_encoder(parser):
    """扩展参数解析器，添加MAA编码器相关参数"""
    
    # MAA编码器模式参数
    parser.add_argument('--use_maa_encoder', action='store_true', 
                       help='Use MAA encoder mode instead of regular MultiEncoder')
    parser.add_argument('--maa_checkpoint_dir', type=str, default=None,
                       help='Directory containing trained MAA generators')
    parser.add_argument('--maa_fusion', type=str, default='attention', 
                       choices=['concat', 'attention', 'gating'],
                       help='Fusion method for MAA encoder')
    parser.add_argument('--maa_target_dim', type=int, default=256,
                       help='Target encoder dimension for MAA encoder')
    parser.add_argument('--freeze_maa_generators', action='store_true', default=True,
                       help='Freeze MAA generator parameters during training')
    parser.add_argument('--maa_pretrain_epochs', type=int, default=5,
                       help='Number of epochs to pretrain task head with frozen MAA generators')
    
    return parser

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_maa_encoder_system(args, feature_dims_list, device):
    """创建MAA编码器系统"""
    print("|~| Creating MAA Encoder System...")
    
    if not args.maa_checkpoint_dir:
        raise ValueError("--maa_checkpoint_dir is required when using MAA encoder mode")
    
    if not os.path.exists(args.maa_checkpoint_dir):
        raise FileNotFoundError(f"MAA checkpoint directory not found: {args.maa_checkpoint_dir}")
    
    # 创建MAA编码器
    try:
        maa_encoder = create_maa_encoder_from_checkpoint(
            checkpoint_dir=args.maa_checkpoint_dir,
            feature_dims_list=feature_dims_list,
            fusion=args.maa_fusion,
            target_encoder_dim=args.maa_target_dim,
            freeze_generators=args.freeze_maa_generators,
            device=device
        )
    except Exception as e:
        print(f"× Failed to create MAA encoder: {e}")
        raise
    
    # 获取MAA编码器输出维度
    encoder_output_dim = maa_encoder.get_output_dim()
    
    # 加载配置文件
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 创建解码器（用于重构任务，可选）
    decoder_params = config["decoder"]
    total_feature_dim = sum(feature_dims_list)
    decoder = Decoder(
        input_dim=encoder_output_dim,
        output_dim=total_feature_dim,
        params={
            'd_model_decoder': decoder_params['d_model'],
            'nhead_decoder': decoder_params['nhead'],
            'num_layers_decoder': decoder_params['num_layers'],
            'dim_feedforward_decoder': decoder_params['dim_feedforward'],
            'dropout_decoder': decoder_params['dropout']
        }
    ).to(device)
    
    # 创建任务头（预测器或分类器）
    if args.task_mode == 'regression':
        predictor_params = config["predictor"]
        task_head = Predictor(
            input_dim=encoder_output_dim,
            output_dim=len(args.target_columns),
            params={
                'd_model_predictor': predictor_params['d_model'],
                'nhead_predictor': predictor_params['nhead'],
                'num_layers_predictor': predictor_params['num_layers'],
                'dim_feedforward_predictor': predictor_params['dim_feedforward'],
                'dropout_predictor': predictor_params['dropout']
            }
        ).to(device)
    elif args.task_mode in ['classification', 'investment']:
        classifier_params = config["classifier"]
        task_head = Classifier(
            input_dim_latent_representation=encoder_output_dim,
            num_classes=args.output_dim_classifier,
            params={
                'd_model_classifier': classifier_params['d_model'],
                'nhead_classifier': classifier_params['nhead'],
                'num_layers_classifier': classifier_params['num_layers'],
                'dim_feedforward_classifier': classifier_params['dim_feedforward'],
                'dropout_classifier': classifier_params['dropout']
            }
        ).to(device)
    else:
        raise ValueError(f"Unsupported task mode: {args.task_mode}")
    
    print(f"√ MAA Encoder System created:")
    print(f"   Encoder output dim: {encoder_output_dim}")
    print(f"   Task mode: {args.task_mode}")
    print(f"   MAA fusion: {args.maa_fusion}")
    print(f"   Total feature dim: {total_feature_dim}")
    
    return maa_encoder, decoder, task_head

def pretrain_task_head(args, maa_encoder, task_head, train_data_loader, val_data_loader, 
                      device, visualizer=None, monitor=None):
    """预训练任务头（冻结MAA编码器）"""
    print("\nO-O Pre-training Task Head (MAA Encoder Frozen)...")
    
    # 冻结MAA编码器
    for param in maa_encoder.parameters():
        param.requires_grad = False
    
    # 创建优化器（只优化任务头）
    optimizer = optim.Adam(task_head.parameters(), lr=args.lr_pretrain)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.7)
    
    # 损失函数
    if args.task_mode == 'regression':
        criterion = nn.MSELoss()
    elif args.task_mode in ['classification', 'investment']:
        criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.maa_pretrain_epochs):
        if monitor:
            monitor.start_epoch()
        
        # 训练阶段
        maa_encoder.eval()  # MAA编码器保持评估模式
        task_head.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (batch_X_list, batch_y_raw) in enumerate(train_data_loader):
            batch_X_list_to_device = [x.to(device) for x in batch_X_list]
            batch_y = batch_y_raw.to(device)
            
            optimizer.zero_grad()
            
            # MAA编码器前向传播（无梯度）
            with torch.no_grad():
                encoded_features = maa_encoder(batch_X_list_to_device)
            
            # 任务头前向传播
            if args.task_mode == 'regression':
                predictions = task_head(encoded_features)
                loss = criterion(predictions.squeeze(), batch_y.squeeze())
            elif args.task_mode in ['classification', 'investment']:
                logits = task_head(encoded_features)
                if args.task_mode == 'investment':
                    targets = (batch_y > 0).long().to(device)
                else:
                    targets = batch_y.long().to(device)
                
                loss = criterion(logits, targets.squeeze())
                
                # 计算准确率
                _, predicted = torch.max(logits, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_data_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        
        # 验证阶段
        val_loss, val_accuracy = evaluate_maa_encoder_system(
            maa_encoder, task_head, val_data_loader, criterion, args, device
        )
        
        scheduler.step(val_loss)
        
        # 记录指标
        if visualizer:
            if args.task_mode == 'regression':
                visualizer.record_epoch_metrics('maa_pretrain', epoch, 
                                               train_loss=avg_train_loss, val_loss=val_loss)
            else:
                visualizer.record_epoch_metrics('maa_pretrain', epoch,
                                               train_loss=avg_train_loss, val_loss=val_loss,
                                               train_accuracy=train_accuracy, val_accuracy=val_accuracy)
        
        if monitor:
            monitor.log_epoch_metrics('maa_pretrain', epoch, avg_train_loss, val_loss)
            monitor.end_epoch()
        
        print(f"Pretrain Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}", end="")
        if args.task_mode in ['classification', 'investment']:
            print(f", Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")
        else:
            print()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience_pretrain:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"√ Task Head Pre-training completed. Best Val Loss: {best_val_loss:.4f}")

def finetune_maa_encoder_system(args, maa_encoder, task_head, train_data_loader, val_data_loader,
                               device, visualizer=None, monitor=None):
    """微调整个MAAEncoder系统"""  
    print("\nO-O Fine-tuning MAA Encoder System...")
    
    # 解冻MAA编码器（如果需要）
    if not args.freeze_maa_generators:
        for param in maa_encoder.parameters():
            param.requires_grad = True
        print("||| MAA Encoder unfrozen for fine-tuning")
    else:
        print("||| MAA Encoder remains frozen during fine-tuning")
    
    # 创建优化器
    if args.freeze_maa_generators:
        optimizer = optim.Adam(task_head.parameters(), lr=args.lr_finetune)
    else:
        all_params = list(maa_encoder.parameters()) + list(task_head.parameters())
        optimizer = optim.Adam(all_params, lr=args.lr_finetune)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 损失函数
    if args.task_mode == 'regression':
        criterion = nn.MSELoss()
    elif args.task_mode in ['classification', 'investment']:
        criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.finetune_epochs):
        if monitor:
            monitor.start_epoch()
        
        # 训练阶段
        maa_encoder.train()
        task_head.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (batch_X_list, batch_y_raw) in enumerate(train_data_loader):
            batch_X_list_to_device = [x.to(device) for x in batch_X_list]
            batch_y = batch_y_raw.to(device)
            
            optimizer.zero_grad()
            
            # 完整前向传播
            encoded_features = maa_encoder(batch_X_list_to_device)
            
            if args.task_mode == 'regression':
                predictions = task_head(encoded_features)
                loss = criterion(predictions.squeeze(), batch_y.squeeze())
            elif args.task_mode in ['classification', 'investment']:
                logits = task_head(encoded_features)
                if args.task_mode == 'investment':
                    targets = (batch_y > 0).long().to(device)
                else:
                    targets = batch_y.long().to(device)
                
                loss = criterion(logits, targets.squeeze())
                
                # 计算准确率
                _, predicted = torch.max(logits, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_data_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        
        # 验证阶段
        val_loss, val_accuracy = evaluate_maa_encoder_system(
            maa_encoder, task_head, val_data_loader, criterion, args, device
        )
        
        scheduler.step(val_loss)
        
        # 记录指标
        if visualizer:
            if args.task_mode == 'regression':
                visualizer.record_epoch_metrics('maa_finetune', epoch,
                                               train_loss=avg_train_loss, val_loss=val_loss)
            else:
                visualizer.record_epoch_metrics('maa_finetune', epoch,
                                               train_loss=avg_train_loss, val_loss=val_loss,
                                               train_accuracy=train_accuracy, val_accuracy=val_accuracy)
        
        if monitor:
            monitor.log_epoch_metrics('maa_finetune', epoch, avg_train_loss, val_loss)
            monitor.end_epoch()
        
        print(f"Finetune Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}", end="")
        if args.task_mode in ['classification', 'investment']:
            print(f", Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")
        else:
            print()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience_finetune:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"√ Fine-tuning completed. Best Val Loss: {best_val_loss:.4f}")

def evaluate_maa_encoder_system(maa_encoder, task_head, data_loader, criterion, args, device):
    """评估MAAEncoder系统"""
    maa_encoder.eval()
    task_head.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X_list, batch_y_raw in data_loader:
            batch_X_list_to_device = [x.to(device) for x in batch_X_list]
            batch_y = batch_y_raw.to(device)
            
            # 前向传播
            encoded_features = maa_encoder(batch_X_list_to_device)
            
            if args.task_mode == 'regression':
                predictions = task_head(encoded_features)
                loss = criterion(predictions.squeeze(), batch_y.squeeze())
            elif args.task_mode in ['classification', 'investment']:
                logits = task_head(encoded_features)
                if args.task_mode == 'investment':
                    targets = (batch_y > 0).long().to(device)
                else:
                    targets = batch_y.long().to(device)
                
                loss = criterion(logits, targets.squeeze())
                
                # 计算准确率
                _, predicted = torch.max(logits, 1)
                total_samples += targets.size(0)
                total_correct += (predicted == targets.squeeze()).sum().item()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy

def predict_and_save_maa_encoder(args, maa_encoder, task_head, test_data_loader, y_scaler, device):
    """使用MAAEncoder系统进行预测并保存结果"""
    print("|}| Making predictions with MAA Encoder system...")
    
    maa_encoder.eval()
    task_head.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X_list, batch_y_raw in test_data_loader:
            batch_X_list_to_device = [x.to(device) for x in batch_X_list]
            batch_y = batch_y_raw.to(device)
            
            # 前向传播
            encoded_features = maa_encoder(batch_X_list_to_device)
            
            if args.task_mode == 'regression':
                predictions = task_head(encoded_features)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
            elif args.task_mode in ['classification', 'investment']:
                logits = task_head(encoded_features)
                probabilities = torch.softmax(logits, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
                
                if args.task_mode == 'investment':
                    targets = (batch_y > 0).long().to(device)
                else:
                    targets = batch_y.long().to(device)
                all_targets.append(targets.cpu().numpy())
    
    # 合并结果
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 保存预测结果
    if args.task_mode == 'regression':
        # 反归一化
        if y_scaler is not None:
            predictions_denorm = denormalize_data(predictions, y_scaler)
            targets_denorm = denormalize_data(targets, y_scaler)
        else:
            predictions_denorm = predictions
            targets_denorm = targets
        
        # 保存回归结果
        results_df = pd.DataFrame({
            'actual': targets_denorm.flatten(),
            'predicted': predictions_denorm.flatten()
        })
        
        # 计算指标
        mse = np.mean((predictions_denorm - targets_denorm) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_denorm - targets_denorm))
        
        results_path = os.path.join(args.output_dir, 'maa_encoder_predictions.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"|Stat| Regression Results:")
        print(f"   MSE: {mse:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE: {mae:.6f}")
        
    else:
        # 分类任务
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == targets.flatten())
        
        results_df = pd.DataFrame({
            'actual': targets.flatten(),
            'predicted': predicted_classes,
            'probability_0': predictions[:, 0] if predictions.shape[1] > 0 else 0,
            'probability_1': predictions[:, 1] if predictions.shape[1] > 1 else 0,
        })
        
        results_path = os.path.join(args.output_dir, 'maa_encoder_classifications.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"|Stat| Classification Results:")
        print(f"   Accuracy: {accuracy:.4f}")
    
    print(f"√ Results saved to: {results_path}")
    return results_df

def save_maa_encoder_models(args, maa_encoder, task_head, decoder=None):
    """保存MAAEncoder系统模型"""
    print("|#| Saving MAA Encoder models...")
    
    models_dir = os.path.join(args.output_dir, 'maa_encoder_models') 
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存MAA编码器
    maa_encoder_path = os.path.join(models_dir, 'maa_encoder.pth')
    torch.save({
        'model_state_dict': maa_encoder.state_dict(),
        'fusion_method': args.maa_fusion,
        'target_dim': args.maa_target_dim,
        'output_dim': maa_encoder.get_output_dim()
    }, maa_encoder_path)
    
    # 保存任务头
    task_head_path = os.path.join(models_dir, f'task_head_{args.task_mode}.pth')
    torch.save({
        'model_state_dict': task_head.state_dict(),
        'task_mode': args.task_mode,
        'input_dim': maa_encoder.get_output_dim()
    }, task_head_path)
    
    # 保存解码器（如果有）
    if decoder is not None:
        decoder_path = os.path.join(models_dir, 'decoder.pth')
        torch.save({
            'model_state_dict': decoder.state_dict(),
            'input_dim': maa_encoder.get_output_dim()
        }, decoder_path)
        print(f"   Decoder: {decoder_path}")
    
    print(f"   MAA Encoder: {maa_encoder_path}")
    print(f"   Task Head: {task_head_path}")
    print("√ Models saved successfully!")

def main_maa_encoder_mode(args):
    """MAAEncoder模式的主函数"""
    print("=" * 80)
    print("|*&*| MAA ENCODER MODE")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(args.random_state)
    
    # 设置输出目录
    args.target_name = os.path.basename(args.data_path).replace("_processed.csv", "").replace(".csv", "")
    args.output_dir = os.path.join(args.output_dir, f"{args.target_name}_maa_encoder")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'maa_encoder_training.log')),
            logging.StreamHandler()
        ]
    )
    
    # 加载数据
    print(f"|^| Loading data from {args.data_path}...")
    train_x_list_raw, test_x_list_raw, train_y, test_y, x_scalers, y_scaler = \
        load_and_preprocess_data(
            args.data_path, args.start_row, args.end_row,
            args.target_columns, args.feature_columns_list, args.train_split
        )
    
    feature_dims_list = [x.shape[1] for x in train_x_list_raw]
    print(f"|Stat| Feature dimensions per group: {feature_dims_list}")
    
    # 创建数据集
    train_dataset = MultiFactorDataset(
        x_data_list=train_x_list_raw, y_data=train_y, 
        window_size=args.window_size, task_mode=args.task_mode
    )
    test_dataset = MultiFactorDataset(
        x_data_list=test_x_list_raw, y_data=test_y, 
        window_size=args.window_size, task_mode=args.task_mode
    )
    
    def collate_fn(batch):
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch]
        xs_transposed = []
        num_encoders = len(xs[0])
        for i in range(num_encoders):
            xs_transposed.append(torch.stack([x[i] for x in xs], dim=0))
        ys = torch.stack(ys, dim=0)
        return xs_transposed, ys
    
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        drop_last=True, collate_fn=collate_fn
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        drop_last=False, collate_fn=collate_fn
    )
    
    print(f"|^| Dataset sizes: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"|+|  Using device: {device}")
    
    # 创建MAAEncoder系统
    maa_encoder, decoder, task_head = create_maa_encoder_system(args, feature_dims_list, device)
    
    # 创建训练监控器和可视化器
    experiment_name = f"{args.target_name}_maa_encoder_{args.maa_fusion}"
    monitor = TrainingMonitor(experiment_name, os.path.join(args.output_dir, "training_logs"))
    visualizer = TrainingVisualizer(args.output_dir, args.task_mode)
    
    # 分割验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_subset_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, 
        drop_last=True, collate_fn=collate_fn
    )
    val_subset_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, 
        drop_last=False, collate_fn=collate_fn
    )
    
    # 训练流程
    print("\n" + "="*60)
    print("|V| TRAINING PIPELINE")
    print("="*60)
    
    # 1. 预训练任务头
    if args.maa_pretrain_epochs > 0:
        pretrain_task_head(
            args, maa_encoder, task_head, train_subset_loader, val_subset_loader,
            device, visualizer, monitor
        )
    
    # 2. 微调整个系统
    if args.finetune_epochs > 0:
        finetune_maa_encoder_system(
            args, maa_encoder, task_head, train_subset_loader, val_subset_loader,
            device, visualizer, monitor
        )
    
    # 3. 最终预测
    results_df = predict_and_save_maa_encoder(
        args, maa_encoder, task_head, test_data_loader, y_scaler, device
    )
    
    # 4. 保存模型
    if args.viz_save:
        save_maa_encoder_models(args, maa_encoder, task_head, decoder)
    
    # 5. 生成可视化
    if args.viz_save:
        print("\n|Stat| Generating training visualizations...")
        visualizer.finalize()
        
        # 生成监控报告
        monitor.generate_summary_report()
    
    print("\n" + "="*60)
    print("√ MAA ENCODER MODE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return results_df

if __name__ == "__main__":
    # 这个脚本不应该直接运行，而是通过main1 maa.py调用
    print("This script should be imported and used through main1 maa.py with --use_maa_encoder flag")

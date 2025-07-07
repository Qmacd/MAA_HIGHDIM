#!/usr/bin/env python3
"""
扩展的主训练脚本，支持MAA编码器模式
在原有的main1 maa.py基础上新增MAA编码器训练模式，不修改原有功能
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
from tqdm import tqdm  # 添加tqdm进度条

# Import from local files
from data_processing.dataset import MultiFactorDataset
from data_processing.data_loader import load_and_preprocess_data, denormalize_data
from models1 import MultiEncoder, Decoder, Predictor, Classifier
from maa_encoder import MAAEncoder, create_maa_encoder_from_checkpoint
from training_monitor import TrainingMonitor
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
    
    return parser

def create_maa_encoder_system(args, feature_dims_list, device):
    """创建MAA编码器系统"""
    print("🔧 Creating MAA Encoder System...")
    
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
    
    # 创建模型参数字典（复用原有参数结构）
    model_params = {
        'd_model_predictor': args.d_model,
        'nhead_predictor': 4,
        'num_layers_predictor': 2,
        'dim_feedforward_predictor': args.d_model * 4,
        'dropout_predictor': 0.1,
        'd_model_classifier': args.d_model,
        'nhead_classifier': 4,
        'num_layers_classifier': 2,
        'dim_feedforward_classifier': args.d_model * 4,
        'dropout_classifier': 0.1,
    }
    
    # 创建预测器或分类器
    if args.task_mode == 'regression':
        predictor_or_classifier = Predictor(
            input_dim=encoder_output_dim,
            output_dim=len(args.target_columns),
            params=model_params
        ).to(device)
    elif args.task_mode in ['classification', 'investment']:
        predictor_or_classifier = Classifier(
            input_dim_latent_representation=encoder_output_dim,
            num_classes=args.output_dim_classifier,
            params=model_params
        ).to(device)
    else:
        raise ValueError(f"Unsupported task mode: {args.task_mode}")
    
    print(f"√ MAA Encoder System created:")
    print(f"   Encoder output dim: {encoder_output_dim}")
    print(f"   Task mode: {args.task_mode}")
    print(f"   MAA fusion: {args.maa_fusion}")
    
    return maa_encoder, predictor_or_classifier

def train_maa_encoder_system(args, maa_encoder, predictor_or_classifier, 
                           train_data_loader, test_data_loader, device):
    """训练MAA编码器系统"""
    print("|\| Starting MAA Encoder System Training...")
    
    # 创建训练监控器
    monitor = TrainingMonitor(f"maa_encoder_{args.maa_fusion}")
    
    # 创建可视化器
    visualizer = TrainingVisualizer(args.output_dir, args.task_mode)
    
    # 创建优化器
    if args.freeze_maa_generators:
        # 只优化预测器/分类器参数
        optimizer = optim.Adam(predictor_or_classifier.parameters(), lr=args.lr_finetune)
        print("|#| MAA generators frozen, only training predictor/classifier")
    else:
        # 优化所有参数
        all_params = list(maa_encoder.parameters()) + list(predictor_or_classifier.parameters())
        optimizer = optim.Adam(all_params, lr=args.lr_finetune)
        print("|#| Training both MAA encoder and predictor/classifier")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 损失函数
    if args.task_mode == 'regression':
        criterion = nn.MSELoss()
    elif args.task_mode in ['classification', 'investment']:
        criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"|Stat Training configuration:")
    print(f"   Epochs: {args.finetune_epochs}")
    print(f"   Learning rate: {args.lr_finetune}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Patience: {args.patience_finetune}")
    
    # 创建epoch进度条
    epoch_pbar = tqdm(range(args.finetune_epochs), desc="Training Epochs", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        monitor.start_epoch()
        
        # 训练阶段
        maa_encoder.train()
        predictor_or_classifier.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 创建训练批次进度条
        train_pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), 
                         desc=f"Epoch {epoch+1}/{args.finetune_epochs} - Training", 
                         leave=False, unit="batch")
        
        for batch_idx, (batch_X_list, batch_y_raw) in train_pbar:
            batch_X_list_to_device = [x.to(device) for x in batch_X_list]
            batch_y = batch_y_raw.to(device)
            
            optimizer.zero_grad()
            
            # MAA编码器前向传播
            encoded_features = maa_encoder(batch_X_list_to_device)
            
            # 预测器/分类器前向传播
            if args.task_mode == 'regression':
                predictions = predictor_or_classifier(encoded_features)
                loss = criterion(predictions.squeeze(), batch_y.squeeze())
            elif args.task_mode in ['classification', 'investment']:
                logits = predictor_or_classifier(encoded_features)
                if args.task_mode == 'investment':
                    # 投资任务：将价格变化转换为类别
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
            
            # 更新训练进度条
            current_train_loss = train_loss / (batch_idx + 1)
            if args.task_mode in ['classification', 'investment']:
                current_train_acc = train_correct / train_total if train_total > 0 else 0
                train_pbar.set_postfix({
                    'Loss': f'{current_train_loss:.4f}', 
                    'Acc': f'{current_train_acc:.4f}'
                })
            else:
                train_pbar.set_postfix({'Loss': f'{current_train_loss:.4f}'})
        
        # 验证阶段
        maa_encoder.eval()
        predictor_or_classifier.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        predictions_list = []
        targets_list = []
        
        # 创建验证批次进度条
        val_pbar = tqdm(test_data_loader, desc=f"Epoch {epoch+1}/{args.finetune_epochs} - Validation", 
                       leave=False, unit="batch")
        
        with torch.no_grad():
            for batch_X_list, batch_y_raw in val_pbar:
                batch_X_list_to_device = [x.to(device) for x in batch_X_list]
                batch_y = batch_y_raw.to(device)
                
                # MAA编码器前向传播
                encoded_features = maa_encoder(batch_X_list_to_device)
                
                # 预测器/分类器前向传播
                if args.task_mode == 'regression':
                    predictions = predictor_or_classifier(encoded_features)
                    loss = criterion(predictions.squeeze(), batch_y.squeeze())
                    
                    predictions_list.extend(predictions.squeeze().cpu().numpy())
                    targets_list.extend(batch_y.squeeze().cpu().numpy())
                    
                elif args.task_mode in ['classification', 'investment']:
                    logits = predictor_or_classifier(encoded_features)
                    if args.task_mode == 'investment':
                        targets = (batch_y > 0).long().to(device)
                    else:
                        targets = batch_y.long().to(device)
                    
                    loss = criterion(logits, targets.squeeze())
                    
                    # 计算准确率
                    _, predicted = torch.max(logits, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets.squeeze()).sum().item()
                    
                    predictions_list.extend(predicted.cpu().numpy())
                    targets_list.extend(targets.squeeze().cpu().numpy())
                
                val_loss += loss.item()
                
                # 更新验证进度条
                current_val_loss = val_loss / (len([x for x in val_pbar if x]))
                if args.task_mode in ['classification', 'investment']:
                    current_val_acc = val_correct / val_total if val_total > 0 else 0
                    val_pbar.set_postfix({
                        'Loss': f'{current_val_loss:.4f}', 
                        'Acc': f'{current_val_acc:.4f}'
                    })
                else:
                    val_pbar.set_postfix({'Loss': f'{current_val_loss:.4f}'})
        
        # 计算平均损失和指标
        avg_train_loss = train_loss / len(train_data_loader)
        avg_val_loss = val_loss / len(test_data_loader)
        
        # 记录指标
        monitor.log_loss('train', 'total_loss', avg_train_loss, epoch)
        monitor.log_loss('val', 'total_loss', avg_val_loss, epoch)
        
        visualizer.record_epoch_metrics('maa_finetune', epoch, 
                                       train_loss=avg_train_loss, val_loss=avg_val_loss)
        
        if args.task_mode == 'regression':
            # 计算相关系数
            if len(predictions_list) > 1:
                correlation = np.corrcoef(predictions_list, targets_list)[0, 1]
                if not np.isnan(correlation):
                    monitor.log_correlation('val', correlation, epoch)
                    visualizer.record_epoch_metrics('maa_finetune', epoch, correlation=correlation)
            
            epoch_pbar.set_postfix({
                'Train_Loss': f'{avg_train_loss:.4f}',
                'Val_Loss': f'{avg_val_loss:.4f}',
                'Correlation': f'{correlation:.4f}' if len(predictions_list) > 1 else 'N/A'
            })
            
            print(f"Epoch {epoch+1}/{args.finetune_epochs}: "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
        elif args.task_mode in ['classification', 'investment']:
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            
            monitor.log_accuracy('train', train_accuracy, epoch)
            monitor.log_accuracy('val', val_accuracy, epoch)
            
            visualizer.record_epoch_metrics('maa_finetune', epoch, 
                                           train_accuracy=train_accuracy, val_accuracy=val_accuracy)
            
            epoch_pbar.set_postfix({
                'Train_Loss': f'{avg_train_loss:.4f}',
                'Val_Loss': f'{avg_val_loss:.4f}',
                'Train_Acc': f'{train_accuracy:.4f}',
                'Val_Acc': f'{val_accuracy:.4f}'
            })
            
            print(f"Epoch {epoch+1}/{args.finetune_epochs}: "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'maa_encoder': maa_encoder.state_dict(),
                'predictor_classifier': predictor_or_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, os.path.join(args.output_dir, 'best_maa_encoder_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience_finetune:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        monitor.end_epoch(epoch)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f}s")
    
    # 关闭进度条
    epoch_pbar.close()
    
    # 生成训练报告和可视化
    monitor.generate_training_report()
    visualizer.finalize()
    
    print(f"√ MAA Encoder training completed!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    
    return maa_encoder, predictor_or_classifier

def save_maa_encoder_predictions(maa_encoder, predictor_or_classifier, test_data_loader, 
                                args, device, y_scaler=None):
    """保存MAA编码器的预测结果"""
    print("|:| Saving MAA encoder predictions...")
    
    maa_encoder.eval()
    predictor_or_classifier.eval()
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    # 创建预测进度条
    pred_pbar = tqdm(test_data_loader, desc="Generating Predictions", unit="batch")
    
    with torch.no_grad():
        for batch_X_list, batch_y_raw in pred_pbar:
            batch_X_list_to_device = [x.to(device) for x in batch_X_list]
            batch_y = batch_y_raw.to(device)
            
            # MAA编码器前向传播
            encoded_features = maa_encoder(batch_X_list_to_device)
            
            # 预测
            if args.task_mode == 'regression':
                predictions = predictor_or_classifier(encoded_features)
                pred_values = predictions.squeeze().cpu().numpy()
                target_values = batch_y.squeeze().cpu().numpy()
                
                # 反标准化
                if y_scaler is not None:
                    pred_values = y_scaler.inverse_transform(pred_values.reshape(-1, 1)).flatten()
                    target_values = y_scaler.inverse_transform(target_values.reshape(-1, 1)).flatten()
                
                all_predictions.extend(pred_values)
                all_targets.extend(target_values)
                
                # 简单的置信度计算（基于预测值的稳定性）
                confidence = np.ones_like(pred_values) * 0.7  # 默认置信度
                all_confidences.extend(confidence)
                
            elif args.task_mode in ['classification', 'investment']:
                logits = predictor_or_classifier(encoded_features)
                probs = F.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probs, dim=1)
                
                pred_values = predicted_classes.cpu().numpy()
                if args.task_mode == 'investment':
                    target_values = (batch_y > 0).long().cpu().numpy()
                else:
                    target_values = batch_y.long().cpu().numpy()
                
                all_predictions.extend(pred_values)
                all_targets.extend(target_values)
                
                # 置信度为最大概率值
                confidence = torch.max(probs, dim=1)[0].cpu().numpy()
                all_confidences.extend(confidence)
                
            # 更新进度条
            pred_pbar.set_postfix({
                'Predictions': len(all_predictions),
                'Targets': len(all_targets)
            })
    
    # 关闭进度条
    pred_pbar.close()
    
    # 创建预测结果DataFrame
    predictions_df = pd.DataFrame({
        'Predicted': all_predictions,
        'Actual': all_targets,
        'Confidence': all_confidences
    })
    
    if args.task_mode == 'investment':
        predictions_df['Predicted_Investment'] = all_predictions
        predictions_df['Investment_Confidence'] = all_confidences
    
    # 保存到CSV
    predictions_file = os.path.join(args.output_dir, 'maa_encoder_predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    
    print(f"√ MAA encoder predictions saved: {predictions_file}")
    print(f"   Total predictions: {len(all_predictions)}")
    
    return predictions_df

def main_maa_encoder_mode(args):
    """MAA编码器模式的主函数"""
    print("|V| MAA Encoder Training Mode")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    random.seed(args.random_state)
    
    # 加载和预处理数据
    print("|Stat| Loading and preprocessing data...")
    (train_data_X_list, train_data_y, test_data_X_list, test_data_y, 
     y_scaler, X_scalers) = load_and_preprocess_data(
        args.data_path, args.feature_columns_list, args.target_columns,
        args.window_size, args.train_split, args.random_state,
        args.start_row, args.end_row
    )
    
    # 获取特征维度
    feature_dims_list = [X.shape[2] for X in train_data_X_list]
    print(f"Feature dimensions: {feature_dims_list}")
    
    # 创建数据加载器
    train_dataset = MultiFactorDataset(train_data_X_list, train_data_y)
    test_dataset = MultiFactorDataset(test_data_X_list, test_data_y)
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建MAA编码器系统
    maa_encoder, predictor_or_classifier = create_maa_encoder_system(
        args, feature_dims_list, device
    )
    
    # 训练MAA编码器系统
    trained_maa_encoder, trained_predictor = train_maa_encoder_system(
        args, maa_encoder, predictor_or_classifier,
        train_data_loader, test_data_loader, device
    )
    
    # 保存预测结果
    predictions_df = save_maa_encoder_predictions(
        trained_maa_encoder, trained_predictor, test_data_loader,
        args, device, y_scaler
    )
    
    print("|H| MAA Encoder mode completed successfully!")
    
    return trained_maa_encoder, trained_predictor, predictions_df

if __name__ == "__main__":
    # 这里只是示例，实际使用时应该从main1 maa.py调用
    print("This is a module for MAA encoder functionality.")
    print("Please use main1 maa.py with --use_maa_encoder flag to run this mode.")

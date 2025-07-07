"""
Training monitoring and visualization system
Record all key metrics during training and generate charts
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import seaborn as sns
from collections import defaultdict
import torch
from tqdm import tqdm  # 添加tqdm进度条

class TrainingMonitor:
    """训练监控器 - 记录和可视化训练过程"""
    
    def __init__(self, experiment_name, save_dir="training_logs", asset_name=None, task_mode=None, fusion_mode=None, strategy=None):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, f"{experiment_name}_training_log.json")
        
        # 实验元信息
        self.asset_name = asset_name or "Unknown"
        self.task_mode = task_mode or "Unknown"
        self.fusion_mode = fusion_mode or "Unknown"
        self.strategy = strategy or "Unknown"
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化记录字典
        self.metrics = defaultdict(list)
        self.epoch_times = []
        self.learning_rates = []
        self.start_time = None
        
        # 设置matplotlib字体，解决中文显示问题
        self._use_english_labels = False
        try:
            # 尝试设置支持中文的字体
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            if os.name == 'nt':  # Windows系统
                fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
            else:  # Linux/Unix系统  
                fonts = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Liberation Sans']
            
            # 尝试每个字体
            for font in fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 测试中文字符渲染
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, '测试', fontsize=8)
                    fig.canvas.draw()
                    plt.close(fig)
                    
                    print(f"[INFO] Chinese font '{font}' successfully configured")
                    break
                except Exception as font_error:
                    print(f"[DEBUG] Font '{font}' failed: {font_error}")
                    continue
            else:
                # 所有中文字体都失败，使用英文
                raise Exception("No Chinese fonts available")
                
        except Exception as e:
            # 如果字体设置失败，使用默认字体并设置为英文
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[Warning] Chinese font setup failed: {e}, using English labels")
            self._use_english_labels = True
        
        print(f"[INFO] Training monitor started: {experiment_name}")
        print(f"[INFO] Log save path: {self.log_file}")
        print(f"[INFO] Experiment config: {self.asset_name}|{self.task_mode}|{self.fusion_mode}|{self.strategy}")
    
    def get_file_prefix(self):
        """生成带实验信息的文件前缀"""
        return f"{self.asset_name}_{self.task_mode}_{self.fusion_mode}_{self.strategy}"
    
    def start_epoch(self):
        """开始一个epoch"""
        self.start_time = datetime.now()
    
    def end_epoch(self, epoch, optimizer=None):
        """结束一个epoch"""
        if self.start_time:
            epoch_time = (datetime.now() - self.start_time).total_seconds()
            self.epoch_times.append(epoch_time)
            
        # 记录学习率
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
        
        # 保存日志
        self.save_log()
        
        # 每10个epoch生成一次图表
        if (epoch + 1) % 10 == 0:
            self.plot_training_curves()
    
    def log_loss(self, phase, loss_type, value, epoch):
        """记录损失"""
        key = f"{phase}_{loss_type}"
        self.metrics[key].append({'epoch': epoch, 'value': float(value)})
    
    def log_metric(self, phase, metric_name, value, epoch):
        """记录评估指标"""
        key = f"{phase}_{metric_name}"
        self.metrics[key].append({'epoch': epoch, 'value': float(value)})
    
    def log_maa_metrics(self, phase, generator_acc, discriminator_acc, epoch):
        """记录MAA特定指标"""
        self.log_metric(phase, 'generator_accuracy', generator_acc, epoch)
        self.log_metric(phase, 'discriminator_accuracy', discriminator_acc, epoch)
    
    def log_correlation(self, phase, correlation, epoch):
        """记录预测相关性"""
        self.log_metric(phase, 'correlation', correlation, epoch)
    
    def log_sharpe_ratio(self, phase, sharpe, epoch):
        """记录夏普比率"""
        self.log_metric(phase, 'sharpe_ratio', sharpe, epoch)
    
    def save_log(self):
        """保存日志到文件"""
        # 确保日志文件的目录存在
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        log_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': dict(self.metrics),
            'epoch_times': self.epoch_times,
            'learning_rates': self.learning_rates
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def plot_training_curves(self):
        """生成训练曲线图"""
        if not self.metrics:
            return
        
        # 创建图表目录
        plot_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # 创建实验特定的子目录
        exp_plot_dir = os.path.join(plot_dir, self.get_file_prefix())
        os.makedirs(exp_plot_dir, exist_ok=True)
        
        # 1. 损失曲线
        self._plot_losses(exp_plot_dir)
        
        # 2. 准确率/相关性曲线
        self._plot_accuracies(exp_plot_dir)
        
        # 3. MAA特定指标
        self._plot_maa_metrics(exp_plot_dir)
        
        # 4. 学习率和训练时间
        self._plot_training_stats(exp_plot_dir)
        
        # 5. 综合对比图
        self._plot_comprehensive_comparison(exp_plot_dir)
        
        print(f"[INFO] Training plots saved to: {exp_plot_dir}")
    
    def _plot_losses(self, plot_dir):
        """绘制损失曲线"""
        loss_keys = [k for k in self.metrics.keys() if 'loss' in k.lower()]
        if not loss_keys:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.experiment_name} - Loss Curves', fontsize=16, fontweight='bold')
        
        # 训练损失
        train_losses = [k for k in loss_keys if 'train' in k]
        if train_losses:
            ax = axes[0, 0]
            for loss_key in train_losses:
                data = self.metrics[loss_key]
                epochs = [d['epoch'] for d in data]
                values = [d['value'] for d in data]
                ax.plot(epochs, values, label=loss_key.replace('train_', ''), linewidth=2)
            ax.set_title('Training Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 验证损失
        val_losses = [k for k in loss_keys if 'val' in k]
        if val_losses:
            ax = axes[0, 1]
            for loss_key in val_losses:
                data = self.metrics[loss_key]
                epochs = [d['epoch'] for d in data]
                values = [d['value'] for d in data]
                ax.plot(epochs, values, label=loss_key.replace('val_', ''), linewidth=2)
            ax.set_title('Validation Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 训练vs验证对比
        if train_losses and val_losses:
            ax = axes[1, 0]
            # 选择主要损失进行对比
            main_train = next((k for k in train_losses if 'total' in k or 'main' in k), train_losses[0])
            main_val = next((k for k in val_losses if 'total' in k or 'main' in k), val_losses[0])
            
            train_data = self.metrics[main_train]
            val_data = self.metrics[main_val]
            
            train_epochs = [d['epoch'] for d in train_data]
            train_values = [d['value'] for d in train_data]
            val_epochs = [d['epoch'] for d in val_data]
            val_values = [d['value'] for d in val_data]
            
            ax.plot(train_epochs, train_values, label='Training', linewidth=2, color='blue')
            ax.plot(val_epochs, val_values, label='Validation', linewidth=2, color='red')
            ax.set_title('Training vs Validation Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 损失下降趋势
        ax = axes[1, 1]
        if train_losses:
            main_loss = train_losses[0]
            data = self.metrics[main_loss]
            epochs = [d['epoch'] for d in data]
            values = [d['value'] for d in data]
            
            # 计算移动平均
            if len(values) > 5:
                window = min(10, len(values) // 3)
                ma_values = pd.Series(values).rolling(window=window).mean()
                ax.plot(epochs, values, alpha=0.5, label='Raw', color='lightblue')
                ax.plot(epochs, ma_values, label=f'Moving Average({window})', linewidth=2, color='darkblue')
            else:
                ax.plot(epochs, values, label='Loss Trend', linewidth=2)
            
            ax.set_title('Loss Trend', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{self.get_file_prefix()}_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracies(self, plot_dir):
        """绘制准确率和相关性曲线"""
        acc_keys = [k for k in self.metrics.keys() if any(x in k.lower() for x in ['accuracy', 'correlation', 'sharpe'])]
        if not acc_keys:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.experiment_name} - Performance Metrics', fontsize=16, fontweight='bold')
        
        # 准确率
        accuracy_keys = [k for k in acc_keys if 'accuracy' in k]
        if accuracy_keys:
            ax = axes[0, 0]
            for acc_key in accuracy_keys:
                data = self.metrics[acc_key]
                epochs = [d['epoch'] for d in data]
                values = [d['value'] for d in data]
                ax.plot(epochs, values, label=acc_key, linewidth=2, marker='o', markersize=3)
            ax.set_title('Accuracy Changes', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # 相关性
        corr_keys = [k for k in acc_keys if 'correlation' in k]
        if corr_keys:
            ax = axes[0, 1]
            for corr_key in corr_keys:
                data = self.metrics[corr_key]
                epochs = [d['epoch'] for d in data]
                values = [d['value'] for d in data]
                ax.plot(epochs, values, label=corr_key, linewidth=2, marker='s', markersize=3)
            ax.set_title('Prediction Correlation', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Correlation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-1, 1])
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 夏普比率
        sharpe_keys = [k for k in acc_keys if 'sharpe' in k]
        if sharpe_keys:
            ax = axes[1, 0]
            for sharpe_key in sharpe_keys:
                data = self.metrics[sharpe_key]
                epochs = [d['epoch'] for d in data]
                values = [d['value'] for d in data]
                ax.plot(epochs, values, label=sharpe_key, linewidth=2, marker='^', markersize=3)
            ax.set_title('Sharpe Ratio', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 综合雷达图
        ax = axes[1, 1]
        self._plot_performance_radar(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{self.get_file_prefix()}_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_maa_metrics(self, plot_dir):
        """绘制MAA特定指标"""
        maa_keys = [k for k in self.metrics.keys() if 'generator' in k or 'discriminator' in k]
        if not maa_keys:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.experiment_name} - MAA Metrics', fontsize=16, fontweight='bold')
        
        # 生成器准确率
        gen_keys = [k for k in maa_keys if 'generator' in k]
        if gen_keys:
            ax = axes[0, 0]
            for gen_key in gen_keys:
                data = self.metrics[gen_key]
                epochs = [d['epoch'] for d in data]
                values = [d['value'] for d in data]
                ax.plot(epochs, values, label=gen_key, linewidth=2, marker='o', markersize=3)
            ax.set_title('Generator Accuracy', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Generator Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # 判别器准确率
        disc_keys = [k for k in maa_keys if 'discriminator' in k]
        if disc_keys:
            ax = axes[0, 1]
            for disc_key in disc_keys:
                data = self.metrics[disc_key]
                epochs = [d['epoch'] for d in data]
                values = [d['value'] for d in data]
                ax.plot(epochs, values, label=disc_key, linewidth=2, marker='s', markersize=3)
            ax.set_title('Discriminator Accuracy', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Discriminator Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # 生成器vs判别器对抗
        if gen_keys and disc_keys:
            ax = axes[1, 0]
            gen_data = self.metrics[gen_keys[0]]
            disc_data = self.metrics[disc_keys[0]]
            
            gen_epochs = [d['epoch'] for d in gen_data]
            gen_values = [d['value'] for d in gen_data]
            disc_epochs = [d['epoch'] for d in disc_data]
            disc_values = [d['value'] for d in disc_data]
            
            ax.plot(gen_epochs, gen_values, label='Generator', linewidth=2, color='blue')
            ax.plot(disc_epochs, disc_values, label='Discriminator', linewidth=2, color='red')
            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Balance Point')
            ax.set_title('Generator vs Discriminator', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # MAA收敛分析
        ax = axes[1, 1]
        if gen_keys and disc_keys:
            gen_data = self.metrics[gen_keys[0]]
            disc_data = self.metrics[disc_keys[0]]
            
            # 计算对抗平衡度 (接近0.5表示平衡)
            balance_scores = []
            for gen_point, disc_point in zip(gen_data, disc_data):
                if gen_point['epoch'] == disc_point['epoch']:
                    balance = abs(gen_point['value'] - 0.5) + abs(disc_point['value'] - 0.5)
                    balance_scores.append({'epoch': gen_point['epoch'], 'balance': balance})
            
            if balance_scores:
                epochs = [b['epoch'] for b in balance_scores]
                balances = [b['balance'] for b in balance_scores]
                ax.plot(epochs, balances, linewidth=2, color='purple', marker='o', markersize=3)
                ax.set_title('MAA Adversarial Balance', fontweight='bold')
                ax.set_xlabel('Epoch')
                # 根据字体支持情况选择标签
                if hasattr(self, '_use_english_labels') and self._use_english_labels:
                    ax.set_ylabel('Imbalance Score (lower is better)')
                else:
                    ax.set_ylabel('不平衡度 (越小越好)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{self.get_file_prefix()}_maa_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_stats(self, plot_dir):
        """绘制训练统计信息"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{self.experiment_name} - Training Statistics', fontsize=16, fontweight='bold')
        
        # 学习率变化
        ax1 = axes[0]
        if self.learning_rates and len(self.learning_rates) > 0:
            epochs = list(range(len(self.learning_rates)))
            ax1.plot(epochs, self.learning_rates, linewidth=2, color='orange', marker='o', markersize=3)
            ax1.set_title('Learning Rate Changes', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Learning Rate')
            ax1.grid(True, alpha=0.3)
            if max(self.learning_rates) / min(self.learning_rates) > 10:  # 大范围变化时使用对数坐标
                ax1.set_yscale('log')
        else:
            # 如果没有学习率数据，显示提示信息
            ax1.text(0.5, 0.5, 'No Learning Rate Data\nRecorded', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax1.set_title('Learning Rate Changes', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Learning Rate')
            ax1.grid(True, alpha=0.3)
        
        # 训练时间
        ax2 = axes[1]
        if self.epoch_times and len(self.epoch_times) > 0:
            epochs = list(range(len(self.epoch_times)))
            ax2.plot(epochs, self.epoch_times, linewidth=2, color='green', marker='s', markersize=3)
            ax2.set_title('Training Time per Epoch', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Time (seconds)')
            ax2.grid(True, alpha=0.3)
            
            # 添加平均时间线
            if len(self.epoch_times) > 1:
                avg_time = np.mean(self.epoch_times)
                ax2.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, 
                          label=f'Average: {avg_time:.1f}s')
                ax2.legend()
        else:
            # 如果没有时间数据，显示提示信息
            ax2.text(0.5, 0.5, 'No Training Time Data\nRecorded', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax2.set_title('Training Time per Epoch', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Time (seconds)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{self.get_file_prefix()}_training_stats.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, ax):
        """绘制性能雷达图"""
        try:
            # 获取最新的性能指标
            metrics_to_plot = ['correlation', 'sharpe_ratio', 'generator_accuracy', 'discriminator_accuracy']
            labels = ['Correlation', 'Sharpe Ratio', 'Generator Acc', 'Discriminator Acc']
            values = []
            available_metrics = []
            available_labels = []
            
            for i, metric in enumerate(metrics_to_plot):
                metric_keys = [k for k in self.metrics.keys() if metric in k and 'val' in k]
                if not metric_keys:
                    metric_keys = [k for k in self.metrics.keys() if metric in k]
                
                if metric_keys:
                    data = self.metrics[metric_keys[0]]
                    if data:
                        latest_value = data[-1]['value']
                        # 标准化到0-1范围
                        if metric == 'correlation':
                            normalized_value = (latest_value + 1) / 2  # -1到1映射到0-1
                        elif metric == 'sharpe_ratio':
                            normalized_value = max(0, min(1, (latest_value + 2) / 4))  # -2到2映射到0-1
                        else:
                            normalized_value = max(0, min(1, latest_value))
                        
                        values.append(normalized_value)
                        available_metrics.append(metric)
                        available_labels.append(labels[i])
            
            if len(values) >= 3:  # 至少需要3个指标才能画雷达图
                # 绘制雷达图
                angles = np.linspace(0, 2 * np.pi, len(available_labels), endpoint=False)
                values_closed = values + [values[0]]  # 闭合
                angles_closed = np.concatenate((angles, [angles[0]]))
                
                # 设置为极坐标
                ax = plt.subplot(111, projection='polar')
                ax.plot(angles_closed, values_closed, 'o-', linewidth=2, color='blue', markersize=6)
                ax.fill(angles_closed, values_closed, alpha=0.25, color='blue')
                ax.set_thetagrids(angles * 180/np.pi, available_labels)
                ax.set_ylim(0, 1)
                ax.set_title('Comprehensive Performance Radar', fontweight='bold', pad=20)
                ax.grid(True)
                
                # 添加数值标签
                for angle, value, label in zip(angles, values, available_labels):
                    ax.text(angle, value + 0.1, f'{value:.2f}', ha='center', va='center', fontsize=8)
                    
            elif len(values) > 0:
                # 如果指标不足3个，显示柱状图
                ax.bar(available_labels, values, color='skyblue', alpha=0.7)
                ax.set_ylim(0, 1)
                ax.set_title('Performance Metrics Comparison', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for i, (label, value) in enumerate(zip(available_labels, values)):
                    ax.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No performance data\nUnable to generate chart', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Comprehensive Performance Radar', fontweight='bold')
                
        except Exception as e:
            print(f"⚠️ Radar chart generation failed: {str(e)}")
            ax.text(0.5, 0.5, f'Radar chart generation failed\n{str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Comprehensive Performance Radar', fontweight='bold')
    
    def _plot_comprehensive_comparison(self, plot_dir):
        """绘制综合对比图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'{self.experiment_name} - Comprehensive Training Progress', fontsize=16, fontweight='bold')
        
        # 收集所有关键指标
        key_metrics = []
        
        # 主要损失
        train_loss_keys = [k for k in self.metrics.keys() if 'train' in k and 'loss' in k]
        if train_loss_keys:
            main_loss_key = train_loss_keys[0]
            data = self.metrics[main_loss_key]
            epochs = [d['epoch'] for d in data]
            values = [d['value'] for d in data]
            # 标准化损失（反转，因为损失越小越好）
            if values:
                max_val = max(values)
                if max_val > 0:
                    normalized_values = [1 - (v / max_val) for v in values]
                    ax.plot(epochs, normalized_values, label='Training Loss (Normalized)', linewidth=2)
        
        # 相关性
        corr_keys = [k for k in self.metrics.keys() if 'correlation' in k and 'val' in k]
        if not corr_keys:
            corr_keys = [k for k in self.metrics.keys() if 'correlation' in k]
        if corr_keys:
            data = self.metrics[corr_keys[0]]
            epochs = [d['epoch'] for d in data]
            values = [d['value'] for d in data]
            # 标准化相关性（-1到1映射到0-1）
            normalized_values = [(v + 1) / 2 for v in values]
            ax.plot(epochs, normalized_values, label='Prediction Correlation', linewidth=2)
        
        # 生成器准确率
        gen_keys = [k for k in self.metrics.keys() if 'generator_accuracy' in k]
        if gen_keys:
            data = self.metrics[gen_keys[0]]
            epochs = [d['epoch'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(epochs, values, label='Generator Accuracy', linewidth=2)
        
        # 验证准确率或MSE
        val_acc_keys = [k for k in self.metrics.keys() if 'val' in k and ('accuracy' in k or 'mse' in k)]
        if val_acc_keys:
            data = self.metrics[val_acc_keys[0]]
            epochs = [d['epoch'] for d in data]
            values = [d['value'] for d in data]
            if 'accuracy' in val_acc_keys[0]:
                ax.plot(epochs, values, label='Validation Accuracy', linewidth=2)
            else:  # MSE - 需要反转
                if values:
                    max_val = max(values)
                    if max_val > 0:
                        normalized_values = [1 - (v / max_val) for v in values]
                        ax.plot(epochs, normalized_values, label='Validation MSE (Normalized)', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 添加文本说明
        ax.text(0.02, 0.98, 'All metrics normalized to 0-1 range\nHigher values indicate better performance', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{self.get_file_prefix()}_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_training_report(self):
        """生成训练报告"""
        report_path = os.path.join(self.save_dir, f"{self.experiment_name}_training_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.experiment_name} 训练报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 训练概况
            f.write("## 训练概况\n\n")
            if self.epoch_times:
                f.write(f"- 训练轮数: {len(self.epoch_times)}\n")
                f.write(f"- 总训练时间: {sum(self.epoch_times):.1f} 秒\n")
                f.write(f"- 平均每轮时间: {np.mean(self.epoch_times):.1f} 秒\n")
            
            # 性能指标
            f.write("\n## 最终性能指标\n\n")
            for metric_name, data in self.metrics.items():
                if data:
                    latest_value = data[-1]['value']
                    f.write(f"- {metric_name}: {latest_value:.6f}\n")
            
            # 训练建议
            f.write("\n## 训练分析与建议\n\n")
            
            # 分析损失收敛
            train_loss_keys = [k for k in self.metrics.keys() if 'train' in k and 'loss' in k]
            if train_loss_keys:
                loss_data = self.metrics[train_loss_keys[0]]
                if len(loss_data) > 5:
                    recent_losses = [d['value'] for d in loss_data[-5:]]
                    early_losses = [d['value'] for d in loss_data[:5]]
                    improvement = (np.mean(early_losses) - np.mean(recent_losses)) / np.mean(early_losses)
                    
                    if improvement > 0.1:
                        f.write("✅ 损失收敛良好，模型正在学习\n")
                    elif improvement > 0.01:
                        f.write("⚠️ 损失下降缓慢，可能需要调整学习率\n")
                    else:
                        f.write("❌ 损失基本不变，建议检查模型架构或数据\n")
            
            # 分析相关性
            corr_keys = [k for k in self.metrics.keys() if 'correlation' in k]
            if corr_keys:
                corr_data = self.metrics[corr_keys[0]]
                if corr_data:
                    latest_corr = corr_data[-1]['value']
                    if latest_corr > 0.7:
                        f.write("✅ 预测相关性很高，模型效果优秀\n")
                    elif latest_corr > 0.3:
                        f.write("⚠️ 预测相关性中等，有优化空间\n")
                    else:
                        f.write("❌ 预测相关性较低，需要模型优化\n")
            
            # 分析MAA平衡
            gen_keys = [k for k in self.metrics.keys() if 'generator_accuracy' in k]
            disc_keys = [k for k in self.metrics.keys() if 'discriminator_accuracy' in k]
            if gen_keys and disc_keys:
                gen_acc = self.metrics[gen_keys[0]][-1]['value'] if self.metrics[gen_keys[0]] else 0
                disc_acc = self.metrics[disc_keys[0]][-1]['value'] if self.metrics[disc_keys[0]] else 0
                
                balance = abs(gen_acc - disc_acc)
                if balance < 0.1:
                    f.write("✅ MAA对抗平衡良好\n")
                elif balance < 0.3:
                    f.write("⚠️ MAA对抗略有不平衡\n")
                else:
                    f.write("❌ MAA对抗严重不平衡，需要调整\n")
        
        print(f"📋 训练报告已生成: {report_path}")
        return report_path


def calculate_correlation(pred, true):
    """计算预测与真实值的相关性"""
    if len(pred) != len(true) or len(pred) == 0:
        return 0.0
    
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    true_tensor = torch.tensor(true, dtype=torch.float32)
    
    # 移除NaN值
    mask = ~(torch.isnan(pred_tensor) | torch.isnan(true_tensor))
    if mask.sum() < 2:
        return 0.0
    
    pred_clean = pred_tensor[mask]
    true_clean = true_tensor[mask]
    
    # 计算皮尔逊相关系数
    pred_mean = pred_clean.mean()
    true_mean = true_clean.mean()
    
    numerator = ((pred_clean - pred_mean) * (true_clean - true_mean)).sum()
    pred_std = ((pred_clean - pred_mean) ** 2).sum().sqrt()
    true_std = ((true_clean - true_mean) ** 2).sum().sqrt()
    
    if pred_std == 0 or true_std == 0:
        return 0.0
    
    correlation = numerator / (pred_std * true_std)
    return float(correlation.item())


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """计算夏普比率"""
    if len(returns) == 0:
        return 0.0
    
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    mask = ~torch.isnan(returns_tensor)
    
    if mask.sum() < 2:
        return 0.0
    
    clean_returns = returns_tensor[mask]
    excess_returns = clean_returns - risk_free_rate
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # 年化
    return float(sharpe.item())


# 使用示例
if __name__ == "__main__":
    # 创建监控器
    monitor = TrainingMonitor("test_experiment")
    
    # 模拟训练过程
    for epoch in range(50):
        monitor.start_epoch()
        
        # 模拟各种指标
        train_loss = 1.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.1)
        val_loss = 1.2 * np.exp(-epoch * 0.08) + np.random.normal(0, 0.1)
        
        correlation = min(0.9, epoch * 0.02 + np.random.normal(0, 0.05))
        gen_acc = 0.5 + 0.3 * np.sin(epoch * 0.2) + np.random.normal(0, 0.02)
        disc_acc = 0.5 + 0.3 * np.cos(epoch * 0.2) + np.random.normal(0, 0.02)
        
        # 记录指标
        monitor.log_loss('train', 'total_loss', train_loss, epoch)
        monitor.log_loss('val', 'total_loss', val_loss, epoch)
        monitor.log_correlation('val', correlation, epoch)
        monitor.log_maa_metrics('train', gen_acc, disc_acc, epoch)
        
        monitor.end_epoch(epoch)
        
        # 模拟训练时间
        import time
        time.sleep(0.01)
    
    # 生成最终报告
    monitor.generate_training_report()
    print("[INFO] 训练监控示例完成！")

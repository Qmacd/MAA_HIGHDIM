# 设置matplotlib后端（必须在导入pyplot之前）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI依赖
import matplotlib.pyplot as plt

import numpy as np
import os
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm  # 添加tqdm进度条

class TrainingVisualizer:
    """训练过程可视化工具类"""
    
    def __init__(self, output_dir, task_mode='regression'):
        self.output_dir = output_dir
        self.task_mode = task_mode
        self.metrics = defaultdict(list)
        
        # 设置图表样式
        try:
            plt.style.use('seaborn')
        except OSError:
            try:
                plt.style.use('ggplot')
            except OSError:
                plt.style.use('default')
                print("Warning: Advanced styles not available, using default")
        
        try:
            import seaborn as sns
            sns.set_palette("husl")
        except ImportError:
            print("Warning: seaborn not available, using matplotlib defaults")
        
        # 创建可视化输出目录
        self.plots_dir = os.path.join(output_dir, 'training_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def record_epoch_metrics(self, phase, epoch, **metrics):
        """记录单个epoch的指标
        
        Args:
            phase: 训练阶段 ('pretrain', 'finetune', 'maa_pretrain', 'adversarial')
            epoch: 当前epoch
            **metrics: 指标字典，如 loss=0.5, accuracy=0.8, mse=0.3
        """
        for metric_name, value in metrics.items():
            # 将CUDA张量转换为CPU上的Python标量
            if hasattr(value, 'cpu'):  # 检查是否是torch张量
                value = value.cpu().item() if value.numel() == 1 else value.cpu().numpy()
            elif hasattr(value, 'numpy'):  # 检查是否是numpy数组
                value = value.item() if value.size == 1 else value
            # 如果是其他类型（如已经是标量），直接使用
            
            key = f"{phase}_{metric_name}"
            self.metrics[key].append({
                'epoch': epoch,
                'value': value
            })
    
    def plot_training_curves(self, phase, title_suffix=""):
        """绘制训练曲线
        
        Args:
            phase: 训练阶段
            title_suffix: 标题后缀
        """
        # 收集该阶段的所有指标
        phase_metrics = {}
        for key, values in self.metrics.items():
            if key.startswith(f"{phase}_"):
                metric_name = key.replace(f"{phase}_", "")
                phase_metrics[metric_name] = values
        
        if not phase_metrics:
            print(f"No metrics found for phase: {phase}")
            return
        
        # 创建子图
        n_metrics = len(phase_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'{phase.capitalize()} Training Curves {title_suffix}', fontsize=16)
        
        for idx, (metric_name, values) in enumerate(phase_metrics.items()):
            ax = axes[idx]
            
            epochs = [v['epoch'] for v in values]
            metric_values = [v['value'] for v in values]
            
            ax.plot(epochs, metric_values, 'o-', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'{metric_name.capitalize()} vs Epoch')
            ax.grid(True, alpha=0.3)
            
            # 添加最佳值标注
            if 'loss' in metric_name.lower() or 'mse' in metric_name.lower():
                best_idx = np.argmin(metric_values)
                best_value = metric_values[best_idx]
                best_epoch = epochs[best_idx]
                ax.annotate(f'Best: {best_value:.4f}', 
                           xy=(best_epoch, best_value), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            elif 'acc' in metric_name.lower():
                best_idx = np.argmax(metric_values)
                best_value = metric_values[best_idx]
                best_epoch = epochs[best_idx]
                ax.annotate(f'Best: {best_value:.4f}', 
                           xy=(best_epoch, best_value), 
                           xytext=(10, -10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.plots_dir, f'{phase}_training_curves.png')
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Training curves saved: {save_path}")
    
    def plot_comparison_curves(self, phases, metric_name, title=""):
        """绘制不同阶段的同一指标对比图
        
        Args:
            phases: 要对比的训练阶段列表
            metric_name: 指标名称
            title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, phase in enumerate(phases):
            key = f"{phase}_{metric_name}"
            if key in self.metrics:
                values = self.metrics[key]
                epochs = [v['epoch'] for v in values]
                metric_values = [v['value'] for v in values]
                
                plt.plot(epochs, metric_values, 'o-', 
                        color=colors[idx % len(colors)], 
                        label=f'{phase.capitalize()}',
                        linewidth=2, markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(title or f'{metric_name.capitalize()} Comparison Across Phases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        save_path = os.path.join(self.plots_dir, f'{metric_name}_comparison.png')
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Comparison plot saved: {save_path}")
    
    def plot_train_val_comparison(self, phase, metric_name):
        """绘制训练集和验证集的对比图
        
        Args:
            phase: 训练阶段
            metric_name: 基础指标名称（如 'loss', 'accuracy'）
        """
        train_key = f"{phase}_train_{metric_name}"
        val_key = f"{phase}_val_{metric_name}"
        
        if train_key not in self.metrics or val_key not in self.metrics:
            print(f"Missing data for train/val comparison: {train_key} or {val_key}")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 训练集数据
        train_values = self.metrics[train_key]
        train_epochs = [v['epoch'] for v in train_values]
        train_metrics = [v['value'] for v in train_values]
        
        # 验证集数据
        val_values = self.metrics[val_key]
        val_epochs = [v['epoch'] for v in val_values]
        val_metrics = [v['value'] for v in val_values]
        
        plt.plot(train_epochs, train_metrics, 'o-', color='blue', label='Training', linewidth=2)
        plt.plot(val_epochs, val_metrics, 'o-', color='red', label='Validation', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{phase.capitalize()} - {metric_name.capitalize()}: Train vs Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        save_path = os.path.join(self.plots_dir, f'{phase}_{metric_name}_train_val.png')
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Train/Val comparison saved: {save_path}")
    
    def save_metrics_to_csv(self):
        """将所有指标保存到CSV文件"""
        
        # 整理数据
        all_data = []
        for key, values in self.metrics.items():
            for v in values:
                all_data.append({
                    'metric': key,
                    'epoch': v['epoch'],
                    'value': v['value']
                })
        
        if all_data:
            import pandas as pd
            df = pd.DataFrame(all_data)
            csv_path = os.path.join(self.plots_dir, 'training_metrics.csv')
            # 确保目录存在
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"[OK] Training metrics saved to CSV: {csv_path}")
    
    def generate_summary_report(self):
        """生成训练总结报告"""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TRAINING SUMMARY REPORT")
        report_lines.append("=" * 60)
        
        # 按阶段整理指标
        phases = set()
        for key in self.metrics.keys():
            phase = key.split('_')[0]
            phases.add(phase)
        
        for phase in sorted(phases):
            report_lines.append(f"\n{phase.upper()} PHASE:")
            report_lines.append("-" * 30)
            
            phase_metrics = {}
            for key, values in self.metrics.items():
                if key.startswith(f"{phase}_"):
                    metric_name = key.replace(f"{phase}_", "")
                    if values:
                        final_value = values[-1]['value']
                        
                        # 计算最佳值
                        if 'loss' in metric_name.lower() or 'mse' in metric_name.lower():
                            best_value = min(v['value'] for v in values)
                            best_epoch = [v['epoch'] for v in values if v['value'] == best_value][0]
                        elif 'acc' in metric_name.lower():
                            best_value = max(v['value'] for v in values)
                            best_epoch = [v['epoch'] for v in values if v['value'] == best_value][0]
                        else:
                            best_value = final_value
                            best_epoch = values[-1]['epoch']
                        
                        phase_metrics[metric_name] = {
                            'final': final_value,
                            'best': best_value,
                            'best_epoch': best_epoch
                        }
            
            for metric_name, data in phase_metrics.items():
                report_lines.append(f"  {metric_name:15s}: Final={data['final']:.4f}, Best={data['best']:.4f} (Epoch {data['best_epoch']})")
        
        # 保存报告
        report_path = os.path.join(self.plots_dir, 'training_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[OK] Training summary report saved: {report_path}")
        
        # 同时打印到控制台
        print('\n'.join(report_lines))
    
    def finalize(self):
        """完成可视化，生成所有图表和报告"""
        
        # 生成各阶段的训练曲线
        phases = set()
        for key in self.metrics.keys():
            phase = key.split('_')[0]
            phases.add(phase)
        
        for phase in phases:
            self.plot_training_curves(phase)
        
        # 生成训练/验证对比图
        common_metrics = ['loss', 'mse', 'accuracy']
        for phase in phases:
            for metric in common_metrics:
                train_key = f"{phase}_train_{metric}"
                val_key = f"{phase}_val_{metric}"
                if train_key in self.metrics and val_key in self.metrics:
                    self.plot_train_val_comparison(phase, metric)
        
        # 保存CSV和生成报告
        self.save_metrics_to_csv()
        self.generate_summary_report()
        
        print(f"\n[OK] All training visualizations completed! Check: {self.plots_dir}")

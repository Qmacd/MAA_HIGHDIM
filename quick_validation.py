#!/usr/bin/env python3
"""
路径映射和输出结构快速验证
"""

import os
import sys
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_output():
    """创建模拟的完整输出结构来测试"""
    logger.info("创建模拟的完整输出结构...")
    
    asset_name = "Gold"
    base_dir = f"output/{asset_name}_processed"
    asset_dir = f"{base_dir}/{asset_name}"
    model_dir = f"{asset_dir}/concat_Baseline"
    training_logs_dir = f"{asset_dir}/training_logs"
    training_plots_dir = f"{asset_dir}/training_plots"
    
    # 创建目录结构
    for dir_path in [base_dir, asset_dir, model_dir, training_logs_dir, training_plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"  创建目录: {dir_path}")
    
    # 创建预测数据
    predictions_data = pd.DataFrame({
        'datetime': pd.date_range('2020-01-01', periods=100, freq='D'),
        'close': range(100, 200),
        'open': range(99, 199),
        'high': range(101, 201),
        'low': range(98, 198),
        'volume': [1000] * 100,
        'True_Class': [0, 1, 2] * 33 + [0],
        'Predicted_Class': [0, 1, 2] * 33 + [1],
        'Predicted_Probability_0': [0.3] * 100,
        'Predicted_Probability_1': [0.4] * 100,
        'Predicted_Probability_2': [0.3] * 100,
    })
    
    predictions_file = f"{asset_dir}/predictions.csv"
    predictions_data.to_csv(predictions_file, index=False)
    logger.info(f"  创建预测文件: {predictions_file}")
    
    # 创建结果文件
    results_data = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
        'value': [0.85, 0.82, 0.88, 0.85, '[[10,2,1],[1,15,2],[0,1,12]]']
    })
    
    results_file = f"{asset_dir}/{asset_name}_results.csv"
    results_data.to_csv(results_file, index=False)
    logger.info(f"  创建结果文件: {results_file}")
    
    # 创建可视化文件（模拟）
    viz_files = [
        f"{asset_dir}/real_vs_predicted_scatter.png",
        f"{asset_dir}/real_vs_predicted_curve.png"
    ]
    
    for viz_file in viz_files:
        with open(viz_file, 'w') as f:
            f.write("Mock PNG content")
        logger.info(f"  创建可视化文件: {viz_file}")
    
    # 创建训练日志
    training_log = {
        'experiment_name': f'{asset_name}_classification_concat_Baseline',
        'total_epochs': 5,
        'best_epoch': 4,
        'best_accuracy': 0.85,
        'final_loss': 0.32,
        'training_time': 120.5
    }
    
    log_file = f"{training_logs_dir}/{asset_name}_classification_concat_Baseline_training_log.json"
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"  创建训练日志: {log_file}")
    
    # 创建训练图表文件（模拟）
    plot_files = [
        f"{training_plots_dir}/finetune_accuracy_train_val.png",
        f"{training_plots_dir}/finetune_loss_train_val.png", 
        f"{training_plots_dir}/finetune_training_curves.png",
        f"{training_plots_dir}/training_metrics.csv",
        f"{training_plots_dir}/training_summary.txt"
    ]
    
    for plot_file in plot_files:
        if plot_file.endswith('.csv'):
            pd.DataFrame({
                'epoch': range(1, 6),
                'train_loss': [0.8, 0.6, 0.5, 0.4, 0.32],
                'val_loss': [0.7, 0.55, 0.48, 0.42, 0.35],
                'train_acc': [0.6, 0.7, 0.75, 0.8, 0.85],
                'val_acc': [0.65, 0.72, 0.78, 0.82, 0.83]
            }).to_csv(plot_file, index=False)
        elif plot_file.endswith('.txt'):
            with open(plot_file, 'w') as f:
                f.write("Training Summary\n")
                f.write("Best epoch: 4\n")
                f.write("Best accuracy: 0.85\n")
        else:
            with open(plot_file, 'w') as f:
                f.write("Mock PNG content")
        logger.info(f"  创建训练图表: {plot_file}")
    
    # 创建模型文件（模拟）
    model_files = [
        f"{model_dir}/encoder.pt",
        f"{model_dir}/classifier.pt"
    ]
    
    for model_file in model_files:
        with open(model_file, 'w') as f:
            f.write("Mock PyTorch model content")
        logger.info(f"  创建模型文件: {model_file}")
    
    logger.info("✅ 模拟输出结构创建完成")
    return predictions_file, results_file

def test_scanning():
    """测试扫描功能"""
    logger.info("测试扫描功能...")
    
    try:
        from complete_pipeline import PipelineConfig, BacktestRunner
        
        config = PipelineConfig()
        backtest_runner = BacktestRunner(config)
        
        experiments = backtest_runner.scan_experiment_results()
        
        if experiments:
            logger.info(f"✅ 扫描成功，找到 {len(experiments)} 个实验")
            for exp in experiments:
                logger.info(f"  实验: {exp['asset_name']}/{exp['experiment_type']}/{exp['fusion_mode']}/{exp['pretrain_type']}")
                logger.info(f"    预测文件: {exp['predictions_path']}")
                logger.info(f"    结果文件: {exp['results_path']}")
                
                # 验证文件存在
                pred_ok = os.path.exists(exp['predictions_path'])
                result_ok = os.path.exists(exp['results_path'])
                logger.info(f"    文件状态: 预测{'✅' if pred_ok else '❌'} 结果{'✅' if result_ok else '❌'}")
            
            return True, experiments
        else:
            logger.error("❌ 扫描失败，未找到实验")
            return False, []
    
    except Exception as e:
        logger.error(f"❌ 扫描异常: {e}")
        return False, []

def analyze_output_structure():
    """分析输出结构的完整性"""
    logger.info("分析输出结构完整性...")
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.error("❌ 输出目录不存在")
        return False
    
    structure_analysis = {}
    
    for root, dirs, files in os.walk(output_dir):
        rel_path = os.path.relpath(root, output_dir)
        if rel_path == ".":
            rel_path = "output/"
        else:
            rel_path = f"output/{rel_path}/"
        
        structure_analysis[rel_path] = {
            'dirs': dirs,
            'files': files,
            'file_count': len(files)
        }
    
    logger.info("📊 输出结构分析:")
    for path, info in structure_analysis.items():
        logger.info(f"  {path}")
        if info['files']:
            logger.info(f"    文件({info['file_count']}): {', '.join(info['files'][:3])}{'...' if len(info['files']) > 3 else ''}")
        if info['dirs']:
            logger.info(f"    子目录: {', '.join(info['dirs'])}")
    
    # 检查关键文件类型
    key_file_types = {
        '.csv': 0,
        '.png': 0,
        '.json': 0,
        '.pt': 0,
        '.txt': 0
    }
    
    for path, info in structure_analysis.items():
        for file in info['files']:
            for ext in key_file_types:
                if file.endswith(ext):
                    key_file_types[ext] += 1
    
    logger.info("📈 文件类型统计:")
    for ext, count in key_file_types.items():
        logger.info(f"  {ext}: {count} 个")
    
    # 检查是否有足够的文件类型
    expected_minimums = {'.csv': 2, '.png': 2, '.json': 1}  # 至少需要的文件数
    all_good = True
    
    for ext, min_count in expected_minimums.items():
        if key_file_types[ext] < min_count:
            logger.warning(f"⚠️ {ext} 文件数量不足: {key_file_types[ext]} < {min_count}")
            all_good = False
        else:
            logger.info(f"✅ {ext} 文件数量充足: {key_file_types[ext]} >= {min_count}")
    
    return all_good

def main():
    """主函数"""
    print("=" * 60)
    print("路径映射和输出结构快速验证")
    print("=" * 60)
    
    # 1. 创建模拟输出
    predictions_file, results_file = create_mock_output()
    
    # 2. 分析输出结构
    structure_ok = analyze_output_structure()
    
    # 3. 测试扫描功能
    scan_ok, experiments = test_scanning()
    
    # 4. 输出总结
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    
    results = [
        ("输出结构创建", True),
        ("结构完整性分析", structure_ok),
        ("扫描功能测试", scan_ok)
    ]
    
    success_count = sum(1 for name, success in results if success)
    total_count = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{name}: {status}")
    
    print(f"\n总体结果: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 路径映射验证成功！")
        print("\n关键发现:")
        print("✅ 输出结构：完整（包含预测、结果、图表、日志、模型）")
        print("✅ 扫描逻辑：正确识别实验配置和文件路径")
        print("✅ 文件组织：符合main1_maa.py输出 + complete_pipeline.py扫描逻辑")
        
        if experiments:
            print(f"\n📋 发现的实验:")
            for exp in experiments:
                print(f"  • {exp['asset_name']}: {exp['fusion_mode']}/{exp['pretrain_type']}")
    else:
        print("\n❌ 部分验证失败")
    
    return success_count == total_count

if __name__ == "__main__":
    main()

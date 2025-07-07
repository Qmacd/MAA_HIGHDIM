#!/usr/bin/env python3
"""
综合回测测试脚本
基于现有output文件夹中的所有模型结果进行组合投资回测
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加td目录到路径，以便导入mystrategy_paper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'td'))

try:
    from mystrategy_paper import run_backtest_for_experiment_group
except ImportError as e:
    print(f"[ERROR] 无法导入mystrategy_paper: {e}")
    print("请确保td/mystrategy_paper.py文件存在")
    sys.exit(1)

def scan_output_directory(output_dir):
    """扫描output目录，发现所有可用的实验结果"""
    experiments = []
    if not os.path.exists(output_dir):
        print(f"[ERROR] Output目录不存在: {output_dir}")
        return experiments
    
    print(f"[INFO] 扫描输出目录: {output_dir}")
    
    # 遍历所有资产文件夹
    for asset_folder in os.listdir(output_dir):
        if not asset_folder.endswith('_processed'):
            continue
            
        asset_name = asset_folder.replace('_processed', '')
        asset_path = os.path.join(output_dir, asset_folder)
        
        if not os.path.isdir(asset_path):
            continue
            
        print(f"[DEBUG] 处理资产: {asset_name}")
        
        # 遍历任务类型 (regression, classification, investment)
        for task_type in ['regression', 'classification', 'investment']:
            task_path = os.path.join(asset_path, task_type)
            if not os.path.exists(task_path):
                continue
                
            # 遍历融合模式 (concat, attention, gating)
            for fusion_mode in ['concat', 'attention', 'gating']:
                fusion_path = os.path.join(task_path, fusion_mode)
                if not os.path.exists(fusion_path):
                    continue
                    
                # 资产子目录
                asset_sub_path = os.path.join(fusion_path, asset_name)
                if not os.path.exists(asset_sub_path):
                    continue
                    
                # 遍历预训练类型
                for pretrain_folder in os.listdir(asset_sub_path):
                    pretrain_path = os.path.join(asset_sub_path, pretrain_folder)
                    if not os.path.isdir(pretrain_path):
                        continue
                    
                    # 检查是否有预测文件和结果文件
                    predictions_file = os.path.join(pretrain_path, 'predictions.csv')
                    results_file = os.path.join(pretrain_path, f'{asset_name}_results.csv')
                    
                    if os.path.exists(predictions_file) and os.path.exists(results_file):
                        # 解析预训练类型
                        if pretrain_folder.startswith(f'{fusion_mode}_'):
                            pretrain_type = pretrain_folder.replace(f'{fusion_mode}_', '')
                        else:
                            pretrain_type = pretrain_folder
                        
                        experiment = {
                            'asset_name': asset_name,
                            'experiment_type': task_type,
                            'fusion_mode': fusion_mode,
                            'pretrain_type': pretrain_type,
                            'predictions_path': predictions_file,
                            'results_path': results_file,
                            'asset_path': asset_path
                        }
                        
                        experiments.append(experiment)
                        print(f"[DEBUG] 发现实验: {asset_name}/{task_type}/{fusion_mode}/{pretrain_type}")
    
    return experiments

def group_experiments_by_configuration(experiments):
    """按实验配置分组实验"""
    grouped = {}
    
    for exp in experiments:
        # 创建配置键
        config_key = f"{exp['experiment_type']}_{exp['fusion_mode']}_{exp['pretrain_type']}"
        
        if config_key not in grouped:
            grouped[config_key] = {
                'experiment_type': exp['experiment_type'],
                'fusion_mode': exp['fusion_mode'],
                'pretrain_type': exp['pretrain_type'],
                'assets': []
            }
        
        grouped[config_key]['assets'].append({
            'asset_name': exp['asset_name'],
            'predictions_path': exp['predictions_path'],
            'results_path': exp['results_path']
        })
    
    return grouped

def create_multi_asset_experiment_config(group_config, experiment_name_prefix="multi_asset"):
    """为多资产组合创建实验配置"""
    
    # 构建实验名称
    experiment_name = f"{experiment_name_prefix}_{group_config['fusion_mode']}_{group_config['pretrain_type']}"
    
    # 创建虚拟的数据文件路径（基于第一个资产）
    if group_config['assets']:
        first_asset = group_config['assets'][0]
        data_file = f"data/multi_asset_{group_config['experiment_type']}.csv"
    else:
        data_file = "data/multi_asset_default.csv"
    
    experiment_config = {
        'experiment_name': experiment_name,
        'experiment_type': group_config['experiment_type'],
        'fusion_mode': group_config['fusion_mode'],
        'pretrain_type': group_config['pretrain_type'],
        'asset_group': 'multi_asset',
        'data_file': data_file,
        'multi_asset_data': group_config['assets']  # 包含所有资产的预测和结果路径
    }
    
    return experiment_config

def run_single_asset_backtests(experiments, output_base_dir):
    """运行单资产回测"""
    print("\n=== 单资产回测 ===")
    
    single_asset_results = []
    
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] 运行单资产回测: {exp['asset_name']}")
        
        # 构建数据文件路径（可能需要根据实际情况调整）
        data_file = f"data/test_{exp['asset_name'].lower()}.csv"
        if not os.path.exists(data_file):
            # 尝试其他可能的文件名
            alternative_paths = [
                f"data/{exp['asset_name']}.csv",
                f"data/{exp['asset_name'].lower()}.csv",
                "data/test_copper.csv"  # 默认备选
            ]
            
            data_file = None
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    data_file = alt_path
                    break
            
            if data_file is None:
                print(f"[WARNING] 找不到数据文件，使用默认路径: data/test_copper.csv")
                data_file = "data/test_copper.csv"
        
        experiment_config = {
            'experiment_name': f"single_{exp['asset_name']}_{exp['fusion_mode']}_{exp['pretrain_type']}",
            'experiment_type': exp['experiment_type'],
            'fusion_mode': exp['fusion_mode'],
            'pretrain_type': exp['pretrain_type'],
            'asset_group': 'single_asset',
            'data_file': data_file,
            'target_asset': exp['asset_name'],
            'predictions_path': exp['predictions_path'],
            'results_path': exp['results_path']
        }
        
        try:
            success = run_backtest_for_experiment_group(experiment_config)
            single_asset_results.append({
                'experiment_config': experiment_config,
                'success': success
            })
            
            if success:
                print(f"[SUCCESS] {exp['asset_name']} 回测成功")
            else:
                print(f"[FAILED] {exp['asset_name']} 回测失败")
                
        except Exception as e:
            print(f"[ERROR] {exp['asset_name']} 回测异常: {str(e)}")
            single_asset_results.append({
                'experiment_config': experiment_config,
                'success': False,
                'error': str(e)
            })
    
    return single_asset_results

def run_multi_asset_backtests(grouped_experiments, output_base_dir):
    """运行多资产组合回测"""
    print("\n=== 多资产组合回测 ===")
    
    multi_asset_results = []
    
    for config_key, group_config in grouped_experiments.items():
        if len(group_config['assets']) < 2:
            print(f"[SKIP] 配置 {config_key} 只有 {len(group_config['assets'])} 个资产，跳过组合回测")
            continue
            
        print(f"\n运行多资产组合回测: {config_key}")
        print(f"包含资产: {[asset['asset_name'] for asset in group_config['assets']]}")
        
        experiment_config = create_multi_asset_experiment_config(group_config)
        
        try:
            # 注意：这里需要修改mystrategy_paper.py来支持多资产配置
            # 目前先使用单资产的方式，但指定asset_group为相应的资产组
            
            # 根据资产类型确定asset_group
            asset_names = [asset['asset_name'] for asset in group_config['assets']]
            if len(asset_names) == 2:
                asset_group = 'dual_asset'
            elif len(asset_names) <= 5:
                asset_group = 'small_group'
            else:
                asset_group = 'large_group'
            
            experiment_config['asset_group'] = asset_group
            
            success = run_backtest_for_experiment_group(experiment_config)
            multi_asset_results.append({
                'experiment_config': experiment_config,
                'success': success
            })
            
            if success:
                print(f"[SUCCESS] {config_key} 组合回测成功")
            else:
                print(f"[FAILED] {config_key} 组合回测失败")
                
        except Exception as e:
            print(f"[ERROR] {config_key} 组合回测异常: {str(e)}")
            multi_asset_results.append({
                'experiment_config': experiment_config,
                'success': False,
                'error': str(e)
            })
    
    return multi_asset_results

def create_comprehensive_report(single_asset_results, multi_asset_results, output_dir):
    """创建综合回测报告"""
    report_path = os.path.join(output_dir, 'comprehensive_backtest_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 综合回测报告\n\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 单资产回测结果
        f.write("## 单资产回测结果\n\n")
        f.write("| 资产 | 任务类型 | 融合模式 | 预训练类型 | 状态 |\n")
        f.write("|------|----------|----------|------------|------|\n")
        
        for result in single_asset_results:
            config = result['experiment_config']
            status = "[SUCCESS] 成功" if result['success'] else "[FAILED] 失败"
            if 'error' in result:
                status += f" ({result['error'][:50]}...)"
            
            f.write(f"| {config.get('target_asset', 'N/A')} | {config['experiment_type']} | "
                   f"{config['fusion_mode']} | {config['pretrain_type']} | {status} |\n")
        
        # 多资产组合回测结果
        f.write("\n## 多资产组合回测结果\n\n")
        f.write("| 配置 | 资产数量 | 状态 |\n")
        f.write("|------|----------|------|\n")
        
        for result in multi_asset_results:
            config = result['experiment_config']
            asset_count = len(config.get('multi_asset_data', []))
            status = "[SUCCESS] 成功" if result['success'] else "[FAILED] 失败"
            if 'error' in result:
                status += f" ({result['error'][:50]}...)"
            
            config_name = f"{config['experiment_type']}_{config['fusion_mode']}_{config['pretrain_type']}"
            f.write(f"| {config_name} | {asset_count} | {status} |\n")
        
        # 统计信息
        total_single = len(single_asset_results)
        success_single = sum(1 for r in single_asset_results if r['success'])
        total_multi = len(multi_asset_results)
        success_multi = sum(1 for r in multi_asset_results if r['success'])
        
        f.write(f"\n## 统计信息\n\n")
        f.write(f"- 单资产回测: {success_single}/{total_single} 成功\n")
        f.write(f"- 多资产组合回测: {success_multi}/{total_multi} 成功\n")
        f.write(f"- 总体成功率: {(success_single + success_multi)/(total_single + total_multi)*100:.1f}%\n")
    
    print(f"\n综合回测报告已保存至: {report_path}")

def main():
    """主函数"""
    print("=== 综合回测测试 ===")
    print("基于现有output文件夹中的所有模型结果进行组合投资回测")
    
    # 设置路径
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    results_dir = script_dir / "comprehensive_backtest_results"
    results_dir.mkdir(exist_ok=True)
    
    # 1. 扫描output目录
    print("\n1. 扫描实验结果...")
    experiments = scan_output_directory(str(output_dir))
    
    if not experiments:
        print("[ERROR] 没有发现任何有效的实验结果！")
        print("请确保output文件夹中包含正确的实验结果文件")
        return
    
    print(f"发现 {len(experiments)} 个实验结果")
    
    # 按任务类型统计
    task_types = {}
    for exp in experiments:
        task_type = exp['experiment_type']
        if task_type not in task_types:
            task_types[task_type] = 0
        task_types[task_type] += 1
    
    print("任务类型分布:")
    for task_type, count in task_types.items():
        print(f"  {task_type}: {count} 个")
    
    # 2. 对实验进行分组
    print("\n2. 对实验进行分组...")
    grouped_experiments = group_experiments_by_configuration(experiments)
    
    print(f"发现 {len(grouped_experiments)} 个不同的实验配置")
    for config_key, group_config in grouped_experiments.items():
        print(f"  {config_key}: {len(group_config['assets'])} 个资产")
    
    # 3. 运行单资产回测
    print("\n3. 运行单资产回测...")
    single_asset_results = run_single_asset_backtests(experiments, str(results_dir))
    
    # 4. 运行多资产组合回测
    print("\n4. 运行多资产组合回测...")
    multi_asset_results = run_multi_asset_backtests(grouped_experiments, str(results_dir))
    
    # 5. 创建综合报告
    print("\n5. 创建综合报告...")
    create_comprehensive_report(single_asset_results, multi_asset_results, str(results_dir))
    
    # 6. 保存详细结果
    detailed_results = {
        'scan_timestamp': datetime.datetime.now().isoformat(),
        'total_experiments': len(experiments),
        'experiment_configurations': len(grouped_experiments),
        'single_asset_results': single_asset_results,
        'multi_asset_results': multi_asset_results,
        'summary': {
            'single_asset_success': sum(1 for r in single_asset_results if r['success']),
            'single_asset_total': len(single_asset_results),
            'multi_asset_success': sum(1 for r in multi_asset_results if r['success']),
            'multi_asset_total': len(multi_asset_results),
        }
    }
    
    results_file = results_dir / 'detailed_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存至: {results_file}")
    
    # 显示总结
    print("\n=== 回测总结 ===")
    print(f"单资产回测: {detailed_results['summary']['single_asset_success']}/{detailed_results['summary']['single_asset_total']} 成功")
    print(f"多资产组合回测: {detailed_results['summary']['multi_asset_success']}/{detailed_results['summary']['multi_asset_total']} 成功")
    
    total_success = detailed_results['summary']['single_asset_success'] + detailed_results['summary']['multi_asset_success']
    total_tests = detailed_results['summary']['single_asset_total'] + detailed_results['summary']['multi_asset_total']
    success_rate = total_success / total_tests * 100 if total_tests > 0 else 0
    print(f"总体成功率: {success_rate:.1f}%")

if __name__ == "__main__":
    main()

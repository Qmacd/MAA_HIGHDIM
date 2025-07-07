#!/usr/bin/env python3
"""
深度项目清理脚本
清理项目中的测试文件、临时文件和无关脚本，保持项目结构整洁
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime

def create_tmp_dir(base_path):
    """创建tmp目录"""
    tmp_dir = base_path / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir

def get_cleanup_rules():
    """定义清理规则"""
    
    # 核心流程文件 - 保留
    core_files = {
        # 主要训练和实验脚本
        'main_maa_encoder_training.py',
        'main_maa_encoder.py', 
        'main1_maa.py',
        'maa_encoder.py',
        'models1.py',
        'time_series_maa.py',
        
        # 核心流程脚本
        'complete_pipeline.py',
        'quick_start.py',
        'run_comprehensive_backtest.py',
        'run_all_experiments.ps1',
        
        # 配置文件
        'config.yaml',
        'pipeline_config.yaml',
        'requirements.txt',
        
        # 文档
        'README.md',
        'README_EXP.md',
        'CODE_ARCHITECTURE_SUMMARY.md',
        'PROJECT_STRUCTURE.md',
        
        # 重要的监控和可视化
        'training_monitor.py',
        'training_visualizer.py',
        'real_maa_accuracy.py',
    }
    
    # 需要移动到tmp的测试/诊断文件
    test_and_diagnostic_files = {
        # 清理脚本
        'deep_clean.py',
        'organize_project.py',
        
        # 诊断脚本
        'diagnose_original_script.ps1',
        'experiment_progress_monitor.py',
        'monitor_backtest.py',
        
        # 验证脚本
        'simple_backtest_validation.py',
        'validate_backtest_pipeline.py',
        
        # 多余的运行脚本
        'run_all_experiments_complete.ps1',
        'run_all_experiments_fixed.ps1', 
        'run_all_experiments_working.ps1',
        'run_backtest_experiments.ps1',
        'run_complete_pipeline.py',
        'run_complete_pipeline_fixed.ps1',
        'run_experiments_simple.ps1',
        
        # 报告和日志文件
        'BACKTEST_OPTIMIZATION_SUMMARY.md',
        'organization_report.md',
        'pipeline.log',
    }
    
    # 需要移动到tmp的目录
    directories_to_move = {
        'comprehensive_backtest_results',
    }
    
    # 需要清空的目录（保留目录结构但删除内容）
    directories_to_clean = {
        'output',
        'backtest_results', 
        'models',
        '__pycache__',
    }
    
    return {
        'core_files': core_files,
        'test_files': test_and_diagnostic_files,
        'move_directories': directories_to_move,
        'clean_directories': directories_to_clean
    }

def move_files_to_tmp(base_path, files_to_move, tmp_dir, dry_run=False):
    """移动文件到tmp目录"""
    moved_files = []
    
    for file_name in files_to_move:
        file_path = base_path / file_name
        if file_path.exists():
            if not dry_run:
                shutil.move(str(file_path), str(tmp_dir / file_name))
            moved_files.append(file_name)
            print(f"{'[DRY RUN] ' if dry_run else ''}移动文件: {file_name}")
    
    return moved_files

def move_directories_to_tmp(base_path, dirs_to_move, tmp_dir, dry_run=False):
    """移动目录到tmp"""
    moved_dirs = []
    
    for dir_name in dirs_to_move:
        dir_path = base_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            target_path = tmp_dir / dir_name
            if not dry_run:
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.move(str(dir_path), str(target_path))
            moved_dirs.append(dir_name)
            print(f"{'[DRY RUN] ' if dry_run else ''}移动目录: {dir_name}")
    
    return moved_dirs

def clean_directories(base_path, dirs_to_clean, dry_run=False):
    """清空指定目录的内容"""
    cleaned_dirs = []
    
    for dir_name in dirs_to_clean:
        dir_path = base_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            if not dry_run:
                # 删除目录内容但保留目录
                for item in dir_path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            # 统计清理的文件数量
            item_count = len(list(dir_path.iterdir())) if dir_path.exists() else 0
            if item_count > 0 or not dry_run:
                cleaned_dirs.append(f"{dir_name} ({item_count} items)")
                print(f"{'[DRY RUN] ' if dry_run else ''}清空目录: {dir_name} ({item_count} items)")
    
    return cleaned_dirs

def generate_cleanup_report(base_path, moved_files, moved_dirs, cleaned_dirs, dry_run=False):
    """生成清理报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'dry_run': dry_run,
        'summary': {
            'moved_files': len(moved_files),
            'moved_directories': len(moved_dirs), 
            'cleaned_directories': len(cleaned_dirs)
        },
        'details': {
            'moved_files': moved_files,
            'moved_directories': moved_dirs,
            'cleaned_directories': cleaned_dirs
        }
    }
    
    # 保存JSON报告
    report_file = base_path / f"cleanup_report{'_dry_run' if dry_run else ''}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    md_content = f"""# 项目清理报告

## 清理概要
- 清理时间: {report['timestamp']}
- 模式: {'预览模式 (DRY RUN)' if dry_run else '执行模式'}
- 移动文件数: {report['summary']['moved_files']}
- 移动目录数: {report['summary']['moved_directories']}
- 清空目录数: {report['summary']['cleaned_directories']}

## 移动到tmp的文件
{chr(10).join(f'- {f}' for f in moved_files)}

## 移动到tmp的目录
{chr(10).join(f'- {d}' for d in moved_dirs)}

## 清空的目录
{chr(10).join(f'- {d}' for d in cleaned_dirs)}

## 保留的核心文件结构
```
MAS_cls/
├── complete_pipeline.py          # 完整端到端流程
├── quick_start.py                # 快速启动工具
├── main_maa_encoder_training.py  # 主训练脚本
├── main_maa_encoder.py           # MAA编码器主脚本
├── main1_maa.py                  # MAA主脚本
├── maa_encoder.py                # MAA编码器实现
├── models1.py                    # 模型定义
├── time_series_maa.py            # 时间序列MAA
├── run_comprehensive_backtest.py # 综合回测
├── run_all_experiments.ps1       # 批量实验脚本
├── training_monitor.py           # 训练监控
├── training_visualizer.py        # 可视化工具
├── real_maa_accuracy.py          # 准确率计算
├── config.yaml                   # 配置文件
├── pipeline_config.yaml          # 流程配置
├── requirements.txt              # 依赖列表
├── README.md                     # 项目文档
├── data/                         # 数据目录
├── td/                           # 交易策略
├── output/                       # 输出目录(已清空)
├── backtest_results/             # 回测结果(已清空)
├── results/                      # 结果目录
├── models/                       # 模型目录(已清空)
└── tmp/                          # 临时文件
```
"""
    
    md_file = base_path / f"cleanup_report{'_dry_run' if dry_run else ''}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n清理报告已保存到: {md_file}")
    return report

def main():
    parser = argparse.ArgumentParser(description='深度清理项目结构')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际执行')
    parser.add_argument('--path', type=str, default='.', help='项目路径')
    
    args = parser.parse_args()
    
    base_path = Path(args.path).resolve()
    print(f"清理项目路径: {base_path}")
    
    if args.dry_run:
        print("=== 预览模式 (DRY RUN) ===")
    else:
        print("=== 执行模式 ===")
        confirm = input("确认要执行清理操作？(y/N): ")
        if confirm.lower() != 'y':
            print("取消清理操作")
            return
    
    # 创建tmp目录
    tmp_dir = create_tmp_dir(base_path)
    
    # 获取清理规则
    rules = get_cleanup_rules()
    
    # 执行清理
    moved_files = move_files_to_tmp(base_path, rules['test_files'], tmp_dir, args.dry_run)
    moved_dirs = move_directories_to_tmp(base_path, rules['move_directories'], tmp_dir, args.dry_run)
    cleaned_dirs = clean_directories(base_path, rules['clean_directories'], args.dry_run)
    
    # 生成报告
    report = generate_cleanup_report(base_path, moved_files, moved_dirs, cleaned_dirs, args.dry_run)
    
    print(f"\n=== 清理完成 ===")
    print(f"移动文件: {len(moved_files)}")
    print(f"移动目录: {len(moved_dirs)}")
    print(f"清空目录: {len(cleaned_dirs)}")

if __name__ == "__main__":
    main()

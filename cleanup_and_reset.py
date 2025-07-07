#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理和重置测试环境
清理所有测试输出，为重新测试准备干净环境
"""

import os
import shutil
import glob

def cleanup_test_outputs():
    """清理测试输出"""
    print("🧹 清理测试输出和实验结果...")
    
    # 要清理的目录和文件
    cleanup_items = [
        # 输出目录
        "output",
        
        # 回测结果  
        "backtest_results",
        "td/backtest_results",
        
        # 结果目录
        "results",
        
        # 模型目录
        "models",
        
        # 日志文件
        "pipeline.log",
        "log.txt",
        
        # 测试配置文件
        "test_*.yaml",
        "demo_*.yaml",
        
        # 临时文件
        "temp_*.py",
        "__pycache__",
        
        # 测试脚本(保留源码，只清理运行产生的文件)
        # 不清理 test_*.py 源文件，只清理输出
    ]
    
    cleaned_count = 0
    
    for item in cleanup_items:
        # 处理通配符
        if '*' in item:
            matches = glob.glob(item)
            for match in matches:
                if os.path.exists(match):
                    if os.path.isdir(match):
                        shutil.rmtree(match)
                        print(f"  删除目录: {match}")
                    else:
                        os.remove(match)
                        print(f"  删除文件: {match}")
                    cleaned_count += 1
        else:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"  删除目录: {item}")
                else:
                    os.remove(item)
                    print(f"  删除文件: {item}")
                cleaned_count += 1
    
    print(f"✅ 清理完成，删除了 {cleaned_count} 个项目")

def create_necessary_directories():
    """创建必要的目录"""
    print("\n📁 创建必要的目录结构...")
    
    directories = [
        "output",
        "results", 
        "backtest_results",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  创建目录: {directory}")

def verify_data_files():
    """验证数据文件存在"""
    print("\n🔍 验证数据文件...")
    
    # 检查训练数据
    data_dir = "data/processed"
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
        print(f"  找到 {len(data_files)} 个训练数据文件")
        for file in data_files[:5]:  # 只显示前5个
            print(f"    - {file}")
        if len(data_files) > 5:
            print(f"    ... 还有 {len(data_files) - 5} 个文件")
    else:
        print("  ❌ 训练数据目录不存在: data/processed")
        return False
    
    # 检查核心脚本
    core_scripts = [
        "main1_maa.py",
        "complete_pipeline.py", 
        "td/mystrategy_paper.py"
    ]
    
    missing_scripts = []
    for script in core_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script}")
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"  缺少关键脚本: {missing_scripts}")
        return False
    
    return True

def analyze_main1_output_structure():
    """分析main1_maa.py的输出结构"""
    print("\n🔍 分析main1_maa.py输出结构...")
    
    # 查看main1_maa.py中的输出路径设置
    try:
        with open("main1_maa.py", 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("  输出目录设置:")
        if "--output_dir" in content:
            print("    ✅ 支持 --output_dir 参数")
        
        if "predictions.csv" in content:
            print("    ✅ 生成 predictions.csv")
            
        if "_results.csv" in content:
            print("    ✅ 生成 {asset}_results.csv")
            
        if "args.fusion" in content and "args.training_strategy" in content:
            print("    ✅ 使用 fusion_strategy 子目录")
            
    except Exception as e:
        print(f"    ❌ 无法分析main1_maa.py: {e}")

def show_path_mapping():
    """显示路径映射关系"""
    print("\n📋 路径映射关系:")
    print("=" * 60)
    
    print("1. main1_maa.py 输出结构:")
    print("   output/")
    print("   └── {asset}_processed/")
    print("       ├── predictions.csv")
    print("       ├── {asset}_results.csv")
    print("       └── {fusion}_{strategy}/")
    print("           └── [模型文件]")
    
    print("\n2. complete_pipeline.py 扫描逻辑:")
    print("   - 在 output/{asset}_processed/ 下查找")
    print("   - 查找 predictions.csv 和 {asset}_results.csv")
    print("   - 从子目录名推断实验配置")
    
    print("\n3. mystrategy_paper.py 期望路径:")
    print("   - 优先使用 experiment_config['predictions_path']")
    print("   - 备选查找模式:")
    print("     * {fusion}_{pretrain}/predictions.csv")
    print("     * predictions.csv")
    print("     * {fusion}_{pretrain}/{asset}_results.csv")
    print("     * {asset}_results.csv")

def main():
    """主函数"""
    print("🔄 MAS-CLS 测试环境清理和重置")
    print("=" * 60)
    
    # 清理旧的输出
    cleanup_test_outputs()
    
    # 创建必要目录
    create_necessary_directories()
    
    # 验证数据文件
    if not verify_data_files():
        print("❌ 数据文件验证失败")
        return False
    
    # 分析输出结构
    analyze_main1_output_structure()
    
    # 显示路径映射
    show_path_mapping()
    
    print("\n✅ 环境清理和重置完成")
    print("现在可以运行端到端测试:")
    print("  python test_clean_pipeline.py")
    
    return True

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MAS-CLS 快速启动脚本
提供便捷的命令行接口来运行不同的任务
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

class QuickStart:
    """快速启动管理器"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        
    def run_command(self, cmd: list, description: str):
        """运行命令"""
        print(f"\n🚀 {description}")
        print(f"执行命令: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=self.script_dir, check=True)
            print(f"✅ {description} 完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} 失败: {e}")
            return False
    
    def run_experiments(self, assets=None, modes=None):
        """运行实验"""
        if assets:
            asset_groups = assets
        else:
            asset_groups = ["small_group"]  # 默认小规模测试
            
        cmd = ["python", "complete_pipeline.py", "--asset-groups"] + asset_groups
        return self.run_command(cmd, f"运行实验 - 资产组: {', '.join(asset_groups)}")
    
    def run_backtest_only(self):
        """只运行回测"""
        cmd = ["python", "complete_pipeline.py", "--skip-experiments"]
        return self.run_command(cmd, "运行回测（跳过实验训练）")
    
    def run_comprehensive_backtest(self):
        """运行综合回测"""
        cmd = ["python", "run_comprehensive_backtest.py"]
        return self.run_command(cmd, "运行综合回测")
    
    def run_single_experiment(self, asset="Copper", fusion="concat", exp_type="regression"):
        """运行单个实验"""
        data_file = f"data/processed/{asset}_processed.csv"
        cmd = [
            "python", "main_maa_encoder_training.py",
            "--data_file", data_file,
            "--fusion_mode", fusion,
            "--experiment_type", exp_type
        ]
        return self.run_command(cmd, f"单个实验: {asset}/{fusion}/{exp_type}")
    
    def list_available_assets(self):
        """列出可用资产"""
        print("\n📊 可用资产:")
        data_dir = self.script_dir / "data" / "processed"
        if data_dir.exists():
            for file in data_dir.glob("*_processed.csv"):
                asset = file.stem.replace("_processed", "")
                print(f"  - {asset}")
        else:
            print("  ❌ 数据目录不存在")
    
    def list_asset_groups(self):
        """列出资产组"""
        cmd = ["python", "complete_pipeline.py", "--list-groups"]
        return self.run_command(cmd, "列出可用资产组")
    
    def show_project_status(self):
        """显示项目状态"""
        print("\n📈 项目状态:")
        
        # 检查数据
        data_dir = self.script_dir / "data" / "processed"
        if data_dir.exists():
            data_files = list(data_dir.glob("*_processed.csv"))
            print(f"  📁 数据文件: {len(data_files)} 个")
        else:
            print("  ❌ 数据目录不存在")
        
        # 检查输出
        output_dir = self.script_dir / "output"
        if output_dir.exists():
            output_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            print(f"  📂 输出目录: {len(output_dirs)} 个")
        else:
            print("  📂 输出目录: 0 个")
        
        # 检查回测结果
        backtest_dir = self.script_dir / "comprehensive_backtest_results"
        if backtest_dir.exists():
            backtest_files = list(backtest_dir.glob("*.md")) + list(backtest_dir.glob("*.json"))
            print(f"  📊 回测报告: {len(backtest_files)} 个")
        else:
            print("  📊 回测报告: 0 个")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MAS-CLS 快速启动工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 运行实验
    exp_parser = subparsers.add_parser("experiment", help="运行实验")
    exp_parser.add_argument("--assets", nargs="+", default=["small_group"], 
                           help="资产组 (默认: small_group)")
    
    # 运行回测
    subparsers.add_parser("backtest", help="运行回测（跳过实验）")
    
    # 综合回测
    subparsers.add_parser("comprehensive", help="运行综合回测")
    
    # 单个实验
    single_parser = subparsers.add_parser("single", help="运行单个实验")
    single_parser.add_argument("--asset", default="Copper", help="资产名称")
    single_parser.add_argument("--fusion", default="concat", choices=["concat", "attention", "gating"])
    single_parser.add_argument("--type", default="regression", choices=["regression", "classification", "investment"])
    
    # 状态信息
    subparsers.add_parser("status", help="显示项目状态")
    subparsers.add_parser("assets", help="列出可用资产")
    subparsers.add_parser("groups", help="列出资产组")
    
    # 完整流程
    full_parser = subparsers.add_parser("full", help="运行完整流程（实验+回测）")
    full_parser.add_argument("--assets", nargs="+", default=["metals", "energy"], 
                            help="资产组")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    starter = QuickStart()
    
    if args.command == "experiment":
        starter.run_experiments(args.assets)
    elif args.command == "backtest":
        starter.run_backtest_only()
    elif args.command == "comprehensive":
        starter.run_comprehensive_backtest()
    elif args.command == "single":
        starter.run_single_experiment(args.asset, args.fusion, args.type)
    elif args.command == "status":
        starter.show_project_status()
    elif args.command == "assets":
        starter.list_available_assets()
    elif args.command == "groups":
        starter.list_asset_groups()
    elif args.command == "full":
        starter.run_experiments(args.assets)
        starter.run_comprehensive_backtest()
    
    print("\n✨ 任务完成！")

if __name__ == "__main__":
    main()

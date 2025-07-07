#!/usr/bin/env python3
"""
自动清理和重置脚本
"""

import os
import shutil
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_test_environment():
    """清理测试环境"""
    logger.info("开始清理测试环境...")
    
    dirs_to_clean = [
        "output",
        "results", 
        "backtest_results",
        "models",
        "td/backtest_results"
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.info(f"删除目录: {dir_path}")
            except Exception as e:
                logger.error(f"删除目录失败 {dir_path}: {e}")
        
        # 重新创建空目录
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 清理日志文件
    log_files = ["pipeline.log", "log.txt"]
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                logger.info(f"删除日志文件: {log_file}")
            except Exception as e:
                logger.error(f"删除日志文件失败 {log_file}: {e}")
    
    logger.info("测试环境清理完成")

if __name__ == "__main__":
    clean_test_environment()

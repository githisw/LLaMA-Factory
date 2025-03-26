#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSpeed-Ulysses增强专业版主入口

此文件提供了使用DeepSpeed-Ulysses进行训练的主入口。
"""

import os
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any

# 设置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导入训练管理器
from .training_manager import UlyssesTrainingManager

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSpeed-Ulysses增强专业版训练脚本")
    
    # 添加参数
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    parser.add_argument(
        "--adjust-lr",
        type=float,
        nargs='?',
        const=1.0,
        default=None,
        help="调整学习率的因子 (不提供值时默认为1.0)"
    )
    
    # 将未知参数传递给LLaMA-Factory
    args, unknown = parser.parse_known_args()
    
    # 设置环境变量
    if args.debug:
        os.environ["ULYSSES_DEBUG"] = "1"
    
    # 如果指定了adjust-lr，设置环境变量
    if args.adjust_lr is not None:
        os.environ["ULYSSES_ADJUST_LR"] = str(args.adjust_lr)
        
        # 从unknown中移除--adjust-lr参数及其值，避免HfArgumentParser报错
        i = 0
        while i < len(unknown):
            if unknown[i] == "--adjust-lr":
                # 移除--adjust-lr参数
                unknown.pop(i)
                # 如果下一个参数不是以--开头，则认为它是adjust-lr的值，也移除它
                if i < len(unknown) and not unknown[i].startswith("--"):
                    unknown.pop(i)
                else:
                    # 如果没有值，记录警告
                    logger.warning("--adjust-lr参数没有提供值，使用默认值1.0")
                    os.environ["ULYSSES_ADJUST_LR"] = "1.0"
            else:
                i += 1
    
    # 将未知参数添加回sys.argv
    sys.argv = [sys.argv[0]] + unknown
    
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建并初始化UlyssesTrainingManager
    manager = UlyssesTrainingManager()
    
    try:
        # 初始化训练环境
        manager.initialize()
        
        # 运行训练
        manager.run_training()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练出错: {e}", exc_info=True)
    finally:
        # 清理资源
        manager.cleanup()

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用DeepSpeed-Ulysses专业版进行训练的启动脚本

此脚本使用ulysses_enchanced_pro包进行训练，支持多种训练类型和分布式训练。
特点：
1. 模块化设计，代码结构清晰
2. 优化内存使用和训练效率
3. 支持DPO、SFT、PT、PPO、KTO和RM等多种训练类型
4. 支持单节点和多节点分布式训练
5. 优化了PyTorch 2.6反序列化问题
6. 支持通过--adjust-lr参数调整学习率
"""

import os
import sys
import logging
import argparse

# 设置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 确保可以导入LLaMA-Factory的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 解析命令行参数
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSpeed-Ulysses专业版训练脚本")
    
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
        
        # 从sys.argv中移除--adjust-lr参数及其值，避免HfArgumentParser报错
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] == "--adjust-lr":
                # 移除--adjust-lr参数
                sys.argv.pop(i)
                # 如果下一个参数不是以--开头，则认为它是adjust-lr的值，也移除它
                if i < len(sys.argv) and not sys.argv[i].startswith("--"):
                    sys.argv.pop(i)
                else:
                    # 如果没有值，记录警告
                    logger.warning("--adjust-lr参数没有提供值，使用默认值1.0")
                    os.environ["ULYSSES_ADJUST_LR"] = "1.0"
            else:
                i += 1
    
    return args

# 导入ulysses_enchanced_pro包
try:
    from ulysses_enchanced_pro import (
        UlyssesTrainingManager,
        __version__ as ulysses_version
    )
    logger.info(f"成功导入ulysses_enchanced_pro包，版本: {ulysses_version}")
except ImportError as e:
    logger.error(f"无法导入ulysses_enchanced_pro包: {e}")
    logger.error("请确保ulysses_enchanced_pro包已正确安装")
    sys.exit(1)

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

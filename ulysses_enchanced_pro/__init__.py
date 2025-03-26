"""
DeepSpeed-Ulysses增强专业版包

此包提供了DeepSpeed-Ulysses的增强专业版实现，用于优化大型语言模型的训练。
基于improved版本重构，采用更加模块化的设计。
"""

import os
import logging
import gc
import torch

# 设置日志
logger = logging.getLogger(__name__)

# 从子模块导入公共API
from .torch_compatibility import TorchCompatibilityModule
from .distributed_training import DistributedTrainingModule
from .trainer_patching import (
    TrainerPatchingModule,
    patch_dpo_trainer,
    patch_sft_trainer,
    patch_pt_trainer,
    patch_ppo_trainer,
    patch_kto_trainer,
    patch_rm_trainer
)
from .data_processing import DataProcessingModule
from .training_manager import UlyssesTrainingManager

# 设置环境变量以避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置PyTorch内存分配器选项
torch.backends.cuda.matmul.allow_tf32 = True  # 使用TF32精度，提高性能
torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试，提高性能
torch.backends.cudnn.deterministic = False  # 禁用确定性，提高性能

# 设置内存优化选项
USE_MEMORY_EFFICIENT_ATTENTION = True  # 使用内存高效的注意力机制
USE_MEMORY_EFFICIENT_PADDING = True  # 使用内存高效的padding处理
USE_DYNAMIC_BATCH_SIZE = True  # 使用动态批次大小
USE_AGGRESSIVE_MEMORY_MANAGEMENT = True  # 使用激进的内存管理

# 初始化时清理内存
gc.collect()
torch.cuda.empty_cache()

# 内存优化函数
def optimize_memory_usage(model=None, tokenizer=None, dataset=None, dataloader=None, debug=False):
    """
    优化内存使用
    
    Args:
        model: 模型
        tokenizer: tokenizer
        dataset: 数据集
        dataloader: 数据加载器
        debug: 是否启用调试模式
    """
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 优化模型
    if model is not None:
        # 使用梯度检查点
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info("内存使用已优化")

# 版本信息
__version__ = "0.3.0"

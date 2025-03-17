#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用DeepSpeed-Ulysses进行DPO训练的启动脚本
"""

import os
import sys
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, List

# 添加DeepSpeed的类到安全全局列表中，解决PyTorch 2.6的反序列化问题
try:
    # 导入所有可能需要的DeepSpeed类
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    
    # 添加到安全全局列表
    safe_classes = [
        ZeroStageEnum,
        LossScaler,
        DeepSpeedConfig,
        DeepSpeedZeroOptimizer_Stage3,
        ZeroParamStatus
    ]
    
    # 尝试导入更多可能需要的类
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        safe_classes.append(DeepSpeedZeroOptimizer)
    except ImportError:
        pass
    
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import CheckpointFunction
        safe_classes.append(CheckpointFunction)
    except ImportError:
        pass
    
    # 添加所有类到安全全局列表
    torch.serialization.add_safe_globals(safe_classes)
    print(f"Successfully added {len(safe_classes)} DeepSpeed classes to safe globals")
    
    # 设置torch.load的weights_only参数为False
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    print("Patched torch.load to use weights_only=False by default")
    
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not add DeepSpeed classes to safe globals: {e}")

# 确保可以导入LLaMA-Factory的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入Ulysses集成模块
from ulysses_integration import (
    patch_dpo_trainer,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size
)

# 导入LLaMA-Factory的模块
from src.llamafactory.train.dpo.trainer import CustomDPOTrainer
from src.llamafactory.train.dpo.workflow import run_dpo
from src.llamafactory.hparams import get_train_args, read_args
from src.llamafactory.extras import logging
from src.llamafactory.data import PairwiseDataCollatorWithPadding

logger = logging.get_logger(__name__)

# 修改数据收集器以支持序列长度分割
def patch_data_collator(collator_class):
    """
    修补数据收集器以支持序列长度分割
    
    Args:
        collator_class: 原始的数据收集器类
        
    Returns:
        修补后的数据收集器类
    """
    original_call = collator_class.__call__
    
    def patched_call(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 调用原始的__call__方法
        batch = original_call(self, features)
        
        # 检查是否已初始化序列并行
        try:
            seq_parallel_rank = get_sequence_parallel_rank()
            seq_parallel_world_size = get_sequence_parallel_world_size()
        except (ValueError, RuntimeError):
            # 序列并行未初始化，返回原始批次
            return batch
        
        # 对序列维度进行分割
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 2:
                # 获取序列长度
                seq_length = batch[key].size(1)
                
                # 计算每个GPU处理的序列长度
                sub_seq_length = seq_length // seq_parallel_world_size
                
                # 计算当前GPU处理的序列范围
                sub_seq_start = seq_parallel_rank * sub_seq_length
                sub_seq_end = (seq_parallel_rank + 1) * sub_seq_length
                
                # 分割序列
                batch[key] = batch[key][:, sub_seq_start:sub_seq_end]
        
        return batch
    
    # 替换__call__方法
    collator_class.__call__ = patched_call
    
    return collator_class

def main():
    """主函数"""
    # 解析命令行参数
    args = read_args()
    
    # 修补DPO训练器以支持DeepSpeed-Ulysses
    logger.info("Patching DPO trainer to support DeepSpeed-Ulysses...")
    patched_trainer = patch_dpo_trainer(CustomDPOTrainer)
    
    # 修补数据收集器以支持序列长度分割
    logger.info("Patching data collator to support sequence length splitting...")
    patched_collator = patch_data_collator(PairwiseDataCollatorWithPadding)
    
    # 保存原始的类
    original_trainer = sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer
    original_collator = sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding
    
    # 替换类
    sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer = patched_trainer
    sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding = patched_collator
    
    # 获取训练参数
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
    
    # 检查是否指定了DeepSpeed配置
    if training_args.deepspeed is None:
        logger.warning("DeepSpeed configuration not specified. Using default Ulysses configuration.")
        training_args.deepspeed = "examples/deepspeed/ds_z3_ulysses_config.json"
    
    # 检查是否是DPO训练
    if finetuning_args.stage != "dpo":
        logger.warning(f"Current stage is {finetuning_args.stage}, not dpo. Ulysses may not work properly.")
    
    try:
        # 导入并运行训练函数
        from src.llamafactory.train.tuner import run_exp
        run_exp(args)
    finally:
        # 恢复原始的类
        sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer = original_trainer
        sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding = original_collator

if __name__ == "__main__":
    main()

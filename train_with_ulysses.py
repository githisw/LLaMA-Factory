#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用DeepSpeed-Ulysses进行训练的启动脚本
支持DPO、SFT、PT、PPO、KTO和RM等多种训练类型
支持单节点和多节点分布式训练，优化了PyTorch 2.6反序列化问题
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
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    
    # 添加到安全全局列表
    safe_classes = [
        ZeroStageEnum,
        LossScaler,
        DeepSpeedConfig,
        DeepSpeedZeroOptimizer_Stage3,
        ZeroParamStatus,
        DeepSpeedEngine,
        get_fp32_state_dict_from_zero_checkpoint
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
    
    try:
        from deepspeed.runtime.pipe.engine import PipelineEngine
        safe_classes.append(PipelineEngine)
    except ImportError:
        pass
    
    try:
        from deepspeed.runtime.pipe.module import PipelineModule
        safe_classes.append(PipelineModule)
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
try:
    from ulysses_integration import (
        patch_dpo_trainer,
        patch_sft_trainer,
        patch_pt_trainer,
        patch_ppo_trainer,
        patch_kto_trainer,
        patch_rm_trainer,
        get_sequence_parallel_rank,
        get_sequence_parallel_world_size,
        initialize_sequence_parallel
    )
except ImportError:
    print("Warning: Could not import ulysses_integration module. Make sure it exists in the current directory.")
    sys.exit(1)

# 导入LLaMA-Factory的模块
from src.llamafactory.hparams import get_train_args, read_args
from src.llamafactory.extras import logging
from src.llamafactory.data import PairwiseDataCollatorWithPadding

# 根据训练类型导入相应的模块
from src.llamafactory.train.dpo.trainer import CustomDPOTrainer
from src.llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from src.llamafactory.train.pt.trainer import CustomTrainer
from src.llamafactory.train.ppo.trainer import CustomPPOTrainer
from src.llamafactory.train.kto.trainer import CustomKTOTrainer
from src.llamafactory.train.rm.trainer import PairwiseTrainer as CustomRMTrainer

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
    
    # 获取训练参数
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
    
    # 检查是否指定了DeepSpeed配置
    if training_args.deepspeed is None:
        logger.warning("DeepSpeed configuration not specified. Using default Ulysses configuration.")
        training_args.deepspeed = "examples/deepspeed/ds_z3_ulysses_config.json"
    
    # 修补数据收集器以支持序列长度分割
    logger.info("Patching data collator to support sequence length splitting...")
    patched_collator = patch_data_collator(PairwiseDataCollatorWithPadding)
    sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding = patched_collator
    original_collator = sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding
    
    # 根据训练类型修补相应的训练器
    stage = finetuning_args.stage
    if stage == "dpo":
        logger.info("Patching DPO trainer to support DeepSpeed-Ulysses...")
        patched_trainer = patch_dpo_trainer(CustomDPOTrainer)
        sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer = patched_trainer
        original_trainer = sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer
    elif stage == "sft":
        logger.info("Patching SFT trainer to support DeepSpeed-Ulysses...")
        patched_trainer = patch_sft_trainer(CustomSeq2SeqTrainer)
        sys.modules["src.llamafactory.train.sft.trainer"].CustomSeq2SeqTrainer = patched_trainer
        original_trainer = sys.modules["src.llamafactory.train.sft.trainer"].CustomSeq2SeqTrainer
    elif stage == "pt":
        logger.info("Patching PT trainer to support DeepSpeed-Ulysses...")
        patched_trainer = patch_pt_trainer(CustomTrainer)
        sys.modules["src.llamafactory.train.pt.trainer"].CustomTrainer = patched_trainer
        original_trainer = sys.modules["src.llamafactory.train.pt.trainer"].CustomTrainer
    elif stage == "ppo":
        logger.info("Patching PPO trainer to support DeepSpeed-Ulysses...")
        patched_trainer = patch_ppo_trainer(CustomPPOTrainer)
        sys.modules["src.llamafactory.train.ppo.trainer"].CustomPPOTrainer = patched_trainer
        original_trainer = sys.modules["src.llamafactory.train.ppo.trainer"].CustomPPOTrainer
    elif stage == "kto":
        logger.info("Patching KTO trainer to support DeepSpeed-Ulysses...")
        patched_trainer = patch_kto_trainer(CustomKTOTrainer)
        sys.modules["src.llamafactory.train.kto.trainer"].CustomKTOTrainer = patched_trainer
        original_trainer = sys.modules["src.llamafactory.train.kto.trainer"].CustomKTOTrainer
    elif stage == "rm":
        logger.info("Patching RM trainer to support DeepSpeed-Ulysses...")
        patched_trainer = patch_rm_trainer(CustomRMTrainer)
        sys.modules["src.llamafactory.train.rm.trainer"].PairwiseTrainer = patched_trainer
        original_trainer = sys.modules["src.llamafactory.train.rm.trainer"].PairwiseTrainer
    else:
        logger.warning(f"Unknown stage: {stage}. Ulysses may not work properly.")
        original_trainer = None
    
    # 手动初始化序列并行组（针对分布式环境优化）
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # 从DeepSpeed配置文件中读取序列并行大小
        import json
        with open(training_args.deepspeed, "r") as f:
            ds_config = json.load(f)
        
        sequence_parallel_size = ds_config.get("sequence_parallel_size", 2)
        logger.info(f"Initializing sequence parallel with size {sequence_parallel_size} for distributed training...")
        
        # 确保序列并行大小能被总GPU数量整除
        if world_size % sequence_parallel_size != 0:
            logger.warning(f"World size ({world_size}) is not divisible by sequence parallel size ({sequence_parallel_size}). Adjusting to {world_size}.")
            sequence_parallel_size = world_size
        
        # 初始化序列并行组
        initialize_sequence_parallel(world_size, rank, sequence_parallel_size)
        logger.info(f"Sequence parallel initialized with world_size={world_size}, rank={rank}, sequence_parallel_size={sequence_parallel_size}")
    
    try:
        # 导入并运行训练函数
        from src.llamafactory.train.tuner import run_exp
        run_exp(args)
    finally:
        # 恢复原始的类
        if stage == "dpo" and original_trainer is not None:
            sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer = original_trainer
        elif stage == "sft" and original_trainer is not None:
            sys.modules["src.llamafactory.train.sft.trainer"].CustomSeq2SeqTrainer = original_trainer
        elif stage == "pt" and original_trainer is not None:
            sys.modules["src.llamafactory.train.pt.trainer"].CustomTrainer = original_trainer
        elif stage == "ppo" and original_trainer is not None:
            sys.modules["src.llamafactory.train.ppo.trainer"].CustomPPOTrainer = original_trainer
        elif stage == "kto" and original_trainer is not None:
            sys.modules["src.llamafactory.train.kto.trainer"].CustomKTOTrainer = original_trainer
        elif stage == "rm" and original_trainer is not None:
            sys.modules["src.llamafactory.train.rm.trainer"].PairwiseTrainer = original_trainer
        
        # 恢复数据收集器
        sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding = original_collator

if __name__ == "__main__":
    main()

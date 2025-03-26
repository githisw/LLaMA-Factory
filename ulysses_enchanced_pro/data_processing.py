"""
数据处理模块

此模块处理数据相关的功能，包括数据整理器的修补和优化。
"""

import torch
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Union, Callable

# 设置日志
logger = logging.getLogger(__name__)

# 检查是否启用调试模式
DEBUG = os.environ.get("ULYSSES_DEBUG", "0") == "1"

# 从分布式训练模块导入
from .distributed_training import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size
)

class DataProcessingModule:
    """处理数据相关的功能"""
    
    @staticmethod
    def patch_data_collator_for_sequence_parallel(debug=False):
        """修补数据整理器以支持序列并行"""
        try:
            # 导入数据整理器类
            from src.llamafactory.data import PairwiseDataCollatorWithPadding
            
            logger.info("修补数据整理器以支持序列长度分割...")
            patched_collator = patch_data_collator(PairwiseDataCollatorWithPadding, debug=debug)
            sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding = patched_collator
            original_collator = sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding
            return original_collator
        except ImportError as e:
            logger.error(f"导入数据整理器类时出错: {e}")
            return None
    
    @staticmethod
    def restore_original_data_collator(original_collator):
        """恢复原始的数据整理器"""
        try:
            if original_collator is not None:
                sys.modules["src.llamafactory.data"].PairwiseDataCollatorWithPadding = original_collator
                logger.info("已恢复原始的数据整理器")
                return True
            return False
        except Exception as e:
            logger.error(f"恢复原始数据整理器时出错: {e}")
            return False
    
    @staticmethod
    def optimize_dataset(dataset, debug=False):
        """优化数据集，减少内存使用"""
        try:
            patched_dataset = patch_dataset_for_memory_efficiency(dataset.__class__, debug=debug)
            logger.info(f"已优化数据集: {type(dataset).__name__}")
            return patched_dataset
        except Exception as e:
            logger.error(f"优化数据集时出错: {e}")
            return None
    
    @staticmethod
    def optimize_dataloader(dataloader, debug=False):
        """优化数据加载器，减少内存使用"""
        try:
            patched_dataloader = patch_dataloader_for_memory_efficiency(dataloader.__class__, debug=debug)
            logger.info(f"已优化数据加载器: {type(dataloader).__name__}")
            return patched_dataloader
        except Exception as e:
            logger.error(f"优化数据加载器时出错: {e}")
            return None


# 修补数据收集器以支持序列长度分割 (优化版)
def patch_data_collator(collator_class, debug=False):
    """
    修补数据整理器以支持序列长度分割 (优化版)
    
    Args:
        collator_class: 原始的数据整理器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的数据整理器类
    """
    if collator_class is None:
        return None
    
    # 保存原始的__call__方法
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
        
        if debug:
            logger.info(f"修补数据整理器: seq_parallel_rank={seq_parallel_rank}, seq_parallel_world_size={seq_parallel_world_size}")
            logger.info(f"原始批次键: {batch.keys()}")
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    logger.info(f"原始批次[{key}].shape: {batch[key].shape}")
        
        # 检查是否是DPO训练
        is_dpo = any(key.startswith(("chosen_", "rejected_")) for key in batch.keys())
        
        # 对序列维度进行分割
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 2:
                # 获取序列长度
                seq_length = batch[key].size(1)
                
                # 计算每个GPU处理的序列长度
                # 使用向上取整，确保不会丢失序列信息
                sub_seq_length = (seq_length + seq_parallel_world_size - 1) // seq_parallel_world_size
                
                # 计算当前GPU处理的序列范围
                sub_seq_start = seq_parallel_rank * sub_seq_length
                sub_seq_end = min((seq_parallel_rank + 1) * sub_seq_length, seq_length)
                
                # 特殊处理DPO训练中的偏好对
                if is_dpo:
                    # 对于DPO训练，我们需要确保chosen和rejected样本的分割是一致的
                    if key.startswith("chosen_") or key.startswith("rejected_"):
                        # 提取基本键名（去掉chosen_或rejected_前缀）
                        base_key = key[8:] if key.startswith("chosen_") else key[9:]
                        
                        # 检查是否有对应的另一半
                        chosen_key = f"chosen_{base_key}"
                        rejected_key = f"rejected_{base_key}"
                        
                        # 如果两个键都存在，确保它们的分割是一致的
                        if chosen_key in batch and rejected_key in batch:
                            # 获取两个张量的序列长度
                            chosen_seq_length = batch[chosen_key].size(1)
                            rejected_seq_length = batch[rejected_key].size(1)
                            
                            # 使用较大的序列长度来计算子序列长度
                            max_seq_length = max(chosen_seq_length, rejected_seq_length)
                            sub_seq_length = (max_seq_length + seq_parallel_world_size - 1) // seq_parallel_world_size
                            
                            # 重新计算序列范围
                            sub_seq_start = seq_parallel_rank * sub_seq_length
                            sub_seq_end = min((seq_parallel_rank + 1) * sub_seq_length, seq_length)
                
                # 特殊处理不同类型的张量
                if "input_ids" in key or "token_type_ids" in key:
                    # 对于输入ID和token类型ID，我们需要确保分割不会破坏token的完整性
                    # 由于这些张量是整数类型，我们可以直接分割
                    pass
                elif "attention_mask" in key:
                    # 对于注意力掩码，我们需要确保分割后的掩码仍然有效
                    # 由于注意力掩码是布尔类型或整数类型，我们可以直接分割
                    pass
                elif "labels" in key:
                    # 对于标签，我们需要确保分割不会破坏标签的完整性
                    # 由于标签是整数类型，我们可以直接分割
                    pass
                elif "position_ids" in key:
                    # 对于位置ID，我们需要确保分割后的位置ID仍然有效
                    # 我们需要重新计算位置ID，使其从0开始
                    if sub_seq_start > 0:
                        # 分割位置ID
                        position_ids = batch[key][:, sub_seq_start:sub_seq_end]
                        # 重新计算位置ID，使其从0开始
                        position_ids = position_ids - sub_seq_start
                        batch[key] = position_ids
                        continue
                
                # 分割序列
                if sub_seq_end > sub_seq_start:  # 确保子序列长度大于0
                    batch[key] = batch[key][:, sub_seq_start:sub_seq_end]
                else:
                    # 如果子序列长度为0，创建一个空张量
                    batch[key] = torch.empty((batch[key].size(0), 0) + batch[key].shape[2:], 
                                            dtype=batch[key].dtype, 
                                            device=batch[key].device)
        
        if debug:
            logger.info(f"修补后的批次键: {batch.keys()}")
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    logger.info(f"修补后的批次[{key}].shape: {batch[key].shape}")
        
        return batch
    
    # 替换__call__方法
    collator_class.__call__ = patched_call
    
    return collator_class

# 修补数据集以优化内存使用
def patch_dataset_for_memory_efficiency(dataset_class, debug=False):
    """
    修补数据集以优化内存使用
    
    Args:
        dataset_class: 要修补的数据集类
        debug: 是否启用调试模式
        
    Returns:
        修补后的数据集类
    """
    if dataset_class is None:
        return None
    
    # 保存原始的__getitem__方法
    if hasattr(dataset_class, "__getitem__"):
        original_getitem = dataset_class.__getitem__
        
        def patched_getitem(self, index):
            """修补后的__getitem__方法"""
            # 调用原始的__getitem__方法
            item = original_getitem(self, index)
            
            # 如果item是字典且包含input_ids，转换为torch.int32以减少内存使用
            if isinstance(item, dict) and "input_ids" in item and isinstance(item["input_ids"], torch.Tensor):
                item["input_ids"] = item["input_ids"].to(torch.int32)
            
            # 如果item是字典且包含attention_mask，转换为torch.float16以减少内存使用
            if isinstance(item, dict) and "attention_mask" in item and isinstance(item["attention_mask"], torch.Tensor):
                item["attention_mask"] = item["attention_mask"].to(torch.float16)
            
            return item
        
        # 替换__getitem__方法
        dataset_class.__getitem__ = patched_getitem
    
    if debug:
        logger.info(f"修补数据集: {dataset_class.__name__}")
    
    return dataset_class

# 修补数据加载器以优化内存使用
def patch_dataloader_for_memory_efficiency(dataloader_class, debug=False):
    """
    修补数据加载器以优化内存使用
    
    Args:
        dataloader_class: 要修补的数据加载器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的数据加载器类
    """
    if dataloader_class is None:
        return None
    
    # 保存原始的__iter__方法
    if hasattr(dataloader_class, "__iter__"):
        original_iter = dataloader_class.__iter__
        
        def patched_iter(self):
            """修补后的__iter__方法"""
            # 调用原始的__iter__方法
            iterator = original_iter(self)
            
            # 包装迭代器，在每次迭代后清理内存
            for batch in iterator:
                yield batch
                
                # 每隔几个批次清理一次内存
                import random
                if random.random() < 0.1:
                    import gc
                    gc.collect()
        
        # 替换__iter__方法
        dataloader_class.__iter__ = patched_iter
    
    if debug:
        logger.info(f"修补数据加载器: {dataloader_class.__name__}")
    
    return dataloader_class

# 创建高效的注意力掩码
def create_efficient_attention_mask(input_ids, pad_token_id):
    """
    创建高效的注意力掩码
    
    Args:
        input_ids: 输入ID
        pad_token_id: 填充token ID
        
    Returns:
        attention_mask: 注意力掩码
    """
    # 创建一个全1的掩码
    attention_mask = torch.ones_like(input_ids, dtype=torch.float16)
    
    # 将pad_token_id对应的位置设为0
    attention_mask = torch.where(input_ids == pad_token_id, 0, attention_mask)
    
    return attention_mask

# 创建高效的位置ID
def create_efficient_position_ids(input_ids, pad_token_id, position_ids=None):
    """
    创建高效的位置ID
    
    Args:
        input_ids: 输入ID
        pad_token_id: 填充token ID
        position_ids: 预定义的位置ID
        
    Returns:
        position_ids: 位置ID
    """
    if position_ids is not None:
        return position_ids
    
    # 创建掩码，标记非填充位置
    mask = input_ids != pad_token_id
    
    # 创建位置ID
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    
    # 对于填充位置，将位置ID设为0
    position_ids = torch.where(mask, position_ids, torch.zeros_like(position_ids))
    
    return position_ids

# 动态填充批次
def dynamic_padding(batch, pad_token_id, max_length=None):
    """
    动态填充批次，减少不必要的填充
    
    Args:
        batch: 输入批次
        pad_token_id: 填充token ID
        max_length: 最大长度
        
    Returns:
        padded_batch: 填充后的批次
    """
    # 获取批次中最长序列的长度
    if max_length is None:
        max_length = max(len(x) for x in batch)
    
    # 创建填充后的批次
    padded_batch = []
    for seq in batch:
        # 计算需要填充的长度
        padding_length = max_length - len(seq)
        
        # 填充序列
        padded_seq = seq + [pad_token_id] * padding_length
        padded_batch.append(padded_seq)
    
    return padded_batch

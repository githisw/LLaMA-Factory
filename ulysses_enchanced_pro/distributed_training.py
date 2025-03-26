"""
分布式训练模块

此模块处理分布式训练相关的功能，包括序列并行初始化和通信。
"""

import torch
import torch.distributed as dist
from torch import Tensor
import logging
import json
import math
import os
from typing import Optional, Any, List, Tuple

# 设置日志
logger = logging.getLogger(__name__)

# 检查是否启用调试模式
DEBUG = os.environ.get("ULYSSES_DEBUG", "0") == "1"

# 全局变量，用于存储序列并行组
_SEQUENCE_PARALLEL_GROUP = None

class DistributedTrainingModule:
    """处理分布式训练相关的功能"""
    
    @staticmethod
    def initialize_sequence_parallel_from_config(deepspeed_config_path):
        """从DeepSpeed配置文件初始化序列并行组"""
        if not dist.is_initialized():
            logger.warning("分布式环境未初始化，跳过序列并行初始化")
            return False
        
        # 获取分布式环境信息
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # 从DeepSpeed配置文件中读取序列并行大小
        try:
            with open(deepspeed_config_path, "r") as f:
                ds_config = json.load(f)
            
            sequence_parallel_size = ds_config.get("sequence_parallel_size", 4)  # 默认使用4
            
            # 确保序列并行大小能被总GPU数量整除
            if world_size % sequence_parallel_size != 0:
                logger.warning(f"世界大小 ({world_size}) 不能被序列并行大小 ({sequence_parallel_size}) 整除。调整为 {world_size}。")
                sequence_parallel_size = world_size
            
            # 初始化序列并行组
            initialize_sequence_parallel(world_size, rank, sequence_parallel_size)
            logger.info(f"序列并行初始化完成: world_size={world_size}, rank={rank}, sequence_parallel_size={sequence_parallel_size}")
            
            return True
        except Exception as e:
            logger.error(f"初始化序列并行时出错: {e}")
            return False
    
    @staticmethod
    def get_sequence_parallel_group():
        """获取当前进程所属的序列并行组"""
        return get_sequence_parallel_group()
    
    @staticmethod
    def get_sequence_parallel_world_size():
        """获取序列并行组的大小"""
        return get_sequence_parallel_world_size()
    
    @staticmethod
    def get_sequence_parallel_rank():
        """获取当前进程在序列并行组中的rank"""
        return get_sequence_parallel_rank()


def initialize_sequence_parallel(
    world_size: int,
    rank: int,
    sequence_parallel_size: int,
    data_parallel_size: int = 1
) -> None:
    """
    初始化序列并行通信组
    
    Args:
        world_size: 总进程数
        rank: 当前进程的rank
        sequence_parallel_size: 序列并行度
        data_parallel_size: 数据并行度
    """
    global _SEQUENCE_PARALLEL_GROUP
    
    # 确保world_size可以被sequence_parallel_size整除
    assert world_size % sequence_parallel_size == 0, \
        f"World size ({world_size}) must be divisible by sequence parallel size ({sequence_parallel_size})"
    
    # 计算序列并行组的数量
    num_sequence_parallel_groups = world_size // sequence_parallel_size
    
    # 创建序列并行组
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
    
    logger.info(f"初始化序列并行组，大小为 {sequence_parallel_size}")

def get_sequence_parallel_group():
    """获取当前进程所属的序列并行组"""
    if _SEQUENCE_PARALLEL_GROUP is None:
        raise ValueError("序列并行组未初始化。请先调用initialize_sequence_parallel。")
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_world_size():
    """获取序列并行组的大小"""
    return dist.get_world_size(group=get_sequence_parallel_group())

def get_sequence_parallel_rank():
    """获取当前进程在序列并行组中的rank"""
    return dist.get_rank(group=get_sequence_parallel_group())

# 实现序列并行的All-to-All通信 (优化版)
class SeqAllToAll4D(torch.autograd.Function):
    """序列并行的All-to-All通信 (优化版)"""
    
    @staticmethod
    def forward(ctx, group, input_tensor, scatter_idx, gather_idx, padding_idx=None):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.padding_idx = padding_idx
        ctx.input_shape = input_tensor.shape
        
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        
        # 获取输入张量的形状
        input_shape = list(input_tensor.shape)
        scatter_dim_size = input_shape[scatter_idx]
        
        # 处理序列长度不能整除的情况
        remainder = scatter_dim_size % world_size
        if remainder != 0:
            # 计算需要添加的padding数量
            padding_size = world_size - remainder
            
            # 创建padding
            padding_shape = list(input_shape)
            padding_shape[scatter_idx] = padding_size
            
            # 如果提供了padding_idx，使用它创建padding；否则，使用零填充
            if padding_idx is not None:
                padding = torch.full(padding_shape, padding_idx, dtype=input_tensor.dtype, device=input_tensor.device)
            else:
                padding = torch.zeros(padding_shape, dtype=input_tensor.dtype, device=input_tensor.device)
            
            # 在scatter_idx维度上连接input_tensor和padding
            input_tensor = torch.cat([input_tensor, padding], dim=scatter_idx)
            
            # 更新input_shape和scatter_dim_size
            input_shape = list(input_tensor.shape)
            scatter_dim_size = input_shape[scatter_idx]
            
            # 保存padding信息，用于反向传播
            ctx.padded = True
            ctx.padding_size = padding_size
        else:
            ctx.padded = False
            ctx.padding_size = 0
        
        # 确保scatter_dim_size可以被world_size整除
        assert scatter_dim_size % world_size == 0, \
            f"Scatter dimension size ({scatter_dim_size}) must be divisible by world size ({world_size})"
        
        # 计算输出张量的形状
        output_shape = input_shape.copy()
        output_shape[scatter_idx] = scatter_dim_size // world_size
        output_shape[gather_idx] = input_shape[gather_idx] * world_size
        
        # 创建输出张量
        output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        # 执行all-to-all通信
        input_tensor_list = list(torch.chunk(input_tensor, world_size, dim=scatter_idx))
        output_tensor_list = list(torch.chunk(output_tensor, world_size, dim=gather_idx))
        
        # 使用异步通信提高效率
        handles = []
        for i in range(world_size):
            handle = dist.all_to_all_single(
                output_tensor_list[i],
                input_tensor_list[i],
                group=group,
                async_op=True
            )
            handles.append(handle)
        
        # 等待所有通信完成
        for handle in handles:
            handle.wait()
        
        # 拼接输出张量
        output_tensor = torch.cat(output_tensor_list, dim=gather_idx)
        
        # 释放不再需要的张量，减少内存使用
        del input_tensor_list
        del output_tensor_list
        
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        scatter_idx = ctx.scatter_idx
        gather_idx = ctx.gather_idx
        input_shape = ctx.input_shape
        
        # 反向传播时，scatter_idx和gather_idx互换
        grad_input = SeqAllToAll4D.apply(group, grad_output, gather_idx, scatter_idx, ctx.padding_idx)
        
        # 如果在前向传播中添加了padding，需要在反向传播中移除
        if ctx.padded:
            # 计算需要保留的大小
            keep_size = input_shape[scatter_idx]
            
            # 移除padding
            slices = [slice(None)] * len(input_shape)
            slices[scatter_idx] = slice(0, keep_size)
            grad_input = grad_input[tuple(slices)]
        
        # 返回与前向传播输入相同数量的梯度
        return (None, grad_input, None, None, None)

# 实现Ulysses注意力机制 (优化版)
class UlyssesAttention(torch.nn.Module):
    """
    Ulysses注意力机制 (优化版)
    
    Args:
        scatter_idx: all2all通信的scatter索引
        gather_idx: all2all通信的gather索引
        debug: 是否启用调试模式
    """
    
    def __init__(self, scatter_idx: int = 2, gather_idx: int = 1, debug: bool = False) -> None:
        super(UlyssesAttention, self).__init__()
        self.spg = get_sequence_parallel_group()
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.debug = debug
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> Tensor:
        """
        前向传播
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            attention_mask: 注意力掩码
            position_bias: 位置编码偏置
            dropout_p: dropout概率
            softmax_scale: softmax缩放因子
            causal: 是否使用因果掩码
            
        Returns:
            注意力输出
        """
        if self.debug:
            logger.info(f"UlyssesAttention.forward: query.shape={query.shape}, key.shape={key.shape}, value.shape={value.shape}")
            if attention_mask is not None:
                logger.info(f"UlyssesAttention.forward: attention_mask.shape={attention_mask.shape}")
            if position_bias is not None:
                logger.info(f"UlyssesAttention.forward: position_bias.shape={position_bias.shape}")
        
        # 获取序列并行组的大小和rank
        world_size = dist.get_world_size(group=self.spg)
        rank = dist.get_rank(group=self.spg)
        
        # 处理注意力掩码
        if attention_mask is not None:
            # 获取注意力掩码的形状
            mask_shape = attention_mask.shape
            
            # 计算每个GPU处理的序列长度
            seq_length = mask_shape[1]  # 假设注意力掩码的形状为 [batch_size, seq_length, ...]
            sub_seq_length = (seq_length + world_size - 1) // world_size  # 向上取整
            
            # 计算当前GPU处理的序列范围
            sub_seq_start = rank * sub_seq_length
            sub_seq_end = min((rank + 1) * sub_seq_length, seq_length)
            
            # 分割注意力掩码
            attention_mask = attention_mask[:, sub_seq_start:sub_seq_end]
            
            if self.debug:
                logger.info(f"UlyssesAttention.forward: split attention_mask.shape={attention_mask.shape}")
        
        # 处理位置编码偏置
        if position_bias is not None:
            # 获取位置编码偏置的形状
            bias_shape = position_bias.shape
            
            # 计算每个GPU处理的序列长度
            seq_length = bias_shape[2]  # 假设位置编码偏置的形状为 [batch_size, num_heads, seq_length, seq_length]
            sub_seq_length = (seq_length + world_size - 1) // world_size  # 向上取整
            
            # 计算当前GPU处理的序列范围
            sub_seq_start = rank * sub_seq_length
            sub_seq_end = min((rank + 1) * sub_seq_length, seq_length)
            
            # 分割位置编码偏置
            position_bias = position_bias[:, :, sub_seq_start:sub_seq_end, :]
            
            if self.debug:
                logger.info(f"UlyssesAttention.forward: split position_bias.shape={position_bias.shape}")
        
        # 执行序列并行的all-to-all通信
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx, 0)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx, 0)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx, 0)
        
        if self.debug:
            logger.info(f"UlyssesAttention.forward: after all2all: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
        
        # 使用FlashAttention计算注意力
        # 注意：FlashAttention不直接支持注意力掩码和位置编码偏置
        # 如果需要使用这些特性，可能需要修改FlashAttention的实现或使用其他注意力实现
        try:
            from flash_attn import flash_attn_func
            context_layer = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
            )
        except ImportError:
            # 如果没有FlashAttention，使用普通的注意力计算
            logger.warning("FlashAttention未安装，使用普通的注意力计算")
            
            # 计算注意力分数
            attention_scores = torch.matmul(q, k.transpose(-1, -2))
            
            # 应用缩放因子
            if softmax_scale is None:
                softmax_scale = 1.0 / math.sqrt(q.size(-1))
            attention_scores = attention_scores * softmax_scale
            
            # 应用注意力掩码
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # 应用位置编码偏置
            if position_bias is not None:
                attention_scores = attention_scores + position_bias
            
            # 应用因果掩码
            if causal:
                seq_len = q.size(-2)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                    diagonal=1
                )
                attention_scores.masked_fill_(causal_mask, float("-inf"))
            
            # 应用softmax
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            
            # 应用dropout
            if dropout_p > 0.0:
                attention_probs = torch.nn.functional.dropout(attention_probs, p=dropout_p)
            
            # 计算上下文层
            context_layer = torch.matmul(attention_probs, v)
        
        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]
        
        if self.debug:
            logger.info(f"UlyssesAttention.forward: context_layer.shape={context_layer.shape}")
        
        # 执行序列并行的all-to-all通信（反向）
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx, 0
        )
        
        if self.debug:
            logger.info(f"UlyssesAttention.forward: output.shape={output.shape}")
        
        return output

# 特殊的DPO注意力机制
class DPOUlyssesAttention(UlyssesAttention):
    """
    专门为DPO训练设计的Ulysses注意力机制
    
    处理chosen和rejected样本的特殊逻辑
    """
    
    def __init__(self, scatter_idx: int = 2, gather_idx: int = 1, debug: bool = False) -> None:
        super(DPOUlyssesAttention, self).__init__(scatter_idx, gather_idx, debug)
    
    def forward_chosen(self, *args, **kwargs):
        """处理chosen样本的前向传播"""
        return super().forward(*args, **kwargs)
    
    def forward_rejected(self, *args, **kwargs):
        """处理rejected样本的前向传播"""
        return super().forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """
        根据上下文决定使用哪种前向传播
        
        在DPO训练中，模型会分别处理chosen和rejected样本
        我们可以通过检查当前的计算图来确定是处理chosen还是rejected样本
        """
        # 获取当前的计算图名称（如果有）
        graph_name = ""
        if torch._C._get_tracing_state() is not None:
            graph_name = torch._C._get_tracing_state().graph().name()
        
        # 根据计算图名称决定使用哪种前向传播
        if "chosen" in graph_name.lower():
            return self.forward_chosen(*args, **kwargs)
        elif "rejected" in graph_name.lower():
            return self.forward_rejected(*args, **kwargs)
        else:
            # 如果无法确定，使用通用的前向传播
            return super().forward(*args, **kwargs)

def wrap_attention_with_ulysses_attention(model, is_dpo=False, debug=False):
    """
    将模型中的注意力模块替换为UlyssesAttention (优化版)
    
    Args:
        model: 要修改的模型
        is_dpo: 是否是DPO训练
        debug: 是否启用调试模式
        
    Returns:
        修改后的模型
    """
    # 获取序列并行组
    sequence_parallel_group = get_sequence_parallel_group()
    
    # 递归遍历模型的所有模块
    for name, module in model.named_children():
        # 检查是否是注意力模块（根据模型架构可能需要调整）
        if "attention" in name.lower() and hasattr(module, "forward"):
            # 只修改真正的注意力模块
            if "norm" not in name.lower() and "layernorm" not in name.lower():
                logger.info(f"包装注意力模块: {name}")
                
                # 创建Ulysses注意力模块
                if is_dpo:
                    module.dist_ulysser_attn = DPOUlyssesAttention(debug=debug)
                else:
                    module.dist_ulysser_attn = UlyssesAttention(debug=debug)
                
                # 保存原始的forward方法
                original_forward = module.forward
                
                # 定义新的forward方法
                def make_new_forward(orig_forward):
                    def new_forward(self, *args, **kwargs):
                        # 提取query, key, value参数
                        # 注意：这里的参数提取可能需要根据具体模型架构进行调整
                        if len(args) >= 3:
                            query, key, value = args[0], args[1], args[2]
                            args = args[3:]
                            
                            # 提取注意力掩码和位置编码偏置（如果有）
                            attention_mask = kwargs.get("attention_mask", None)
                            position_bias = kwargs.get("position_bias", None)
                            
                            # 如果args中包含注意力掩码或位置编码偏置，提取它们
                            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                                attention_mask = args[0]
                                args = args[1:]
                            
                            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                                position_bias = args[0]
                                args = args[1:]
                        else:
                            query = kwargs.pop("query", None)
                            key = kwargs.pop("key", None)
                            value = kwargs.pop("value", None)
                            attention_mask = kwargs.pop("attention_mask", None)
                            position_bias = kwargs.pop("position_bias", None)
                        
                        if query is not None and key is not None and value is not None:
                            # 使用Ulysses注意力
                            dropout_p = kwargs.pop("dropout_p", 0.0)
                            softmax_scale = kwargs.pop("softmax_scale", None)
                            causal = kwargs.pop("causal", False)
                            
                            return self.dist_ulysser_attn(
                                query, key, value, attention_mask, position_bias,
                                dropout_p, softmax_scale, causal, *args, **kwargs
                            )
                        else:
                            # 如果没有提供query, key, value，则使用原始的forward方法
                            return orig_forward(self, *args, **kwargs)
                    return new_forward
                
                # 替换forward方法
                module.forward = make_new_forward(original_forward).__get__(module)
        
        # 递归处理子模块
        wrap_attention_with_ulysses_attention(module, is_dpo, debug)
    
    return model

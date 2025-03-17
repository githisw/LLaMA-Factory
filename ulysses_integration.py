"""
DeepSpeed-Ulysses集成模块

此文件提供了将DeepSpeed-Ulysses集成到LLaMA-Factory的必要代码修改。
要启用DeepSpeed-Ulysses，您需要：
1. 将此文件放在项目根目录
2. 修改相关模型文件以使用UlyssesAttention
3. 初始化序列并行通信组
"""

import torch
import torch.distributed as dist
from torch import Tensor
from typing import Optional, Any
from flash_attn import flash_attn_func

# 全局变量，用于存储序列并行组
_SEQUENCE_PARALLEL_GROUP = None

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
    
    print(f"Initialized sequence parallel group with size {sequence_parallel_size}")

def get_sequence_parallel_group():
    """获取当前进程所属的序列并行组"""
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_world_size():
    """获取序列并行组的大小"""
    return dist.get_world_size(group=get_sequence_parallel_group())

def get_sequence_parallel_rank():
    """获取当前进程在序列并行组中的rank"""
    return dist.get_rank(group=get_sequence_parallel_group())

# 实现序列并行的All-to-All通信
class SeqAllToAll4D(torch.autograd.Function):
    """序列并行的All-to-All通信"""
    
    @staticmethod
    def forward(ctx, group, input_tensor, scatter_idx, gather_idx):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        
        world_size = dist.get_world_size(group=group)
        
        # 获取输入张量的形状
        input_shape = list(input_tensor.shape)
        scatter_dim_size = input_shape[scatter_idx]
        
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
        
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
        
        # 拼接输出张量
        output_tensor = torch.cat(output_tensor_list, dim=gather_idx)
        
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        scatter_idx = ctx.scatter_idx
        gather_idx = ctx.gather_idx
        
        # 反向传播时，scatter_idx和gather_idx互换
        return (None, 
                SeqAllToAll4D.apply(group, grad_output, gather_idx, scatter_idx),
                None, None)

# 实现Ulysses注意力机制
class UlyssesAttention(torch.nn.Module):
    """
    Ulysses注意力机制
    
    Args:
        scatter_idx: all2all通信的scatter索引
        gather_idx: all2all通信的gather索引
    """
    
    def __init__(self, scatter_idx: int = 2, gather_idx: int = 1) -> None:
        super(UlyssesAttention, self).__init__()
        self.spg = get_sequence_parallel_group()
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        *args: Any
    ) -> Tensor:
        # 执行序列并行的all-to-all通信
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)
        
        # 使用FlashAttention计算注意力
        context_layer = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
        )
        
        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]
        
        # 执行序列并行的all-to-all通信（反向）
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx
        )
        
        return output

def wrap_attention_with_ulysses_attention(model):
    """
    将模型中的注意力模块替换为UlyssesAttention
    
    Args:
        model: 要修改的模型
        
    Returns:
        修改后的模型
    """
    # 获取序列并行组
    sequence_parallel_group = get_sequence_parallel_group()
    if sequence_parallel_group is None:
        raise ValueError("Sequence parallel group not initialized. Call initialize_sequence_parallel first.")
    
    # 递归遍历模型的所有模块
    for name, module in model.named_children():
        # 检查是否是注意力模块（根据模型架构可能需要调整）
        if "attention" in name.lower() and hasattr(module, "forward"):
            # 只修改真正的注意力模块
            if "norm" not in name.lower() and "layernorm" not in name.lower():
                print(f"Wrapping attention module: {name}")
                
                # 创建Ulysses注意力模块
                module.dist_ulysser_attn = UlyssesAttention()
                
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
                        else:
                            query = kwargs.pop("query", None)
                            key = kwargs.pop("key", None)
                            value = kwargs.pop("value", None)
                        
                        if query is not None and key is not None and value is not None:
                            # 使用Ulysses注意力
                            dropout_p = kwargs.pop("dropout_p", 0.0)
                            softmax_scale = kwargs.pop("softmax_scale", None)
                            causal = kwargs.pop("causal", False)
                            
                            return self.dist_ulysser_attn(
                                query, key, value, dropout_p, softmax_scale, causal, *args
                            )
                        else:
                            # 如果没有提供query, key, value，则使用原始的forward方法
                            return orig_forward(self, *args, **kwargs)
                    return new_forward
                
                # 替换forward方法
                module.forward = make_new_forward(original_forward).__get__(module)
        
        # 递归处理子模块
        wrap_attention_with_ulysses_attention(module)
    
    return model

# 通用的训练器修补函数
def patch_trainer(trainer_class, has_ref_model=False):
    """
    通用的训练器修补函数，支持各种训练类型
    
    Args:
        trainer_class: 原始的训练器类
        has_ref_model: 训练器是否有参考模型（如DPO、PPO等）
        
    Returns:
        修补后的训练器类
    """
    original_init = trainer_class.__init__
    
    def patched_init(self, *args, **kwargs):
        # 调用原始的__init__方法
        original_init(self, *args, **kwargs)
        
        # 获取DeepSpeed配置
        if hasattr(self.args, "deepspeed") and self.args.deepspeed:
            import json
            with open(self.args.deepspeed, "r") as f:
                ds_config = json.load(f)
            
            # 检查是否启用了序列并行
            if "sequence_parallel_size" in ds_config:
                sequence_parallel_size = ds_config["sequence_parallel_size"]
                
                # 初始化序列并行组
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    initialize_sequence_parallel(world_size, rank, sequence_parallel_size)
                    
                    # 修改模型以使用Ulysses注意力
                    self.model = wrap_attention_with_ulysses_attention(self.model)
                    
                    # 如果有参考模型，也修改参考模型
                    if has_ref_model and hasattr(self, "ref_model") and self.ref_model is not None:
                        self.ref_model = wrap_attention_with_ulysses_attention(self.ref_model)
                    
                    # 对于PPO训练器，还需要修改奖励模型
                    if hasattr(self, "reward_model") and self.reward_model is not None:
                        self.reward_model = wrap_attention_with_ulysses_attention(self.reward_model)
    
    # 替换__init__方法
    trainer_class.__init__ = patched_init
    
    return trainer_class

# 修改DPO训练器以支持DeepSpeed-Ulysses
def patch_dpo_trainer(trainer_class):
    """
    修补DPO训练器以支持DeepSpeed-Ulysses
    
    Args:
        trainer_class: 原始的DPO训练器类
        
    Returns:
        修补后的DPO训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=True)

# 修改SFT训练器以支持DeepSpeed-Ulysses
def patch_sft_trainer(trainer_class):
    """
    修补SFT训练器以支持DeepSpeed-Ulysses
    
    Args:
        trainer_class: 原始的SFT训练器类
        
    Returns:
        修补后的SFT训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=False)

# 修改PT训练器以支持DeepSpeed-Ulysses
def patch_pt_trainer(trainer_class):
    """
    修补PT训练器以支持DeepSpeed-Ulysses
    
    Args:
        trainer_class: 原始的PT训练器类
        
    Returns:
        修补后的PT训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=False)

# 修改PPO训练器以支持DeepSpeed-Ulysses
def patch_ppo_trainer(trainer_class):
    """
    修补PPO训练器以支持DeepSpeed-Ulysses
    
    Args:
        trainer_class: 原始的PPO训练器类
        
    Returns:
        修补后的PPO训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=True)

# 修改KTO训练器以支持DeepSpeed-Ulysses
def patch_kto_trainer(trainer_class):
    """
    修补KTO训练器以支持DeepSpeed-Ulysses
    
    Args:
        trainer_class: 原始的KTO训练器类
        
    Returns:
        修补后的KTO训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=True)

# 修改RM训练器以支持DeepSpeed-Ulysses
def patch_rm_trainer(trainer_class):
    """
    修补RM训练器以支持DeepSpeed-Ulysses
    
    Args:
        trainer_class: 原始的RM训练器类
        
    Returns:
        修补后的RM训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=False)

# 使用示例
"""
from src.llamafactory.train.dpo.trainer import CustomDPOTrainer
from src.llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from src.llamafactory.train.pt.trainer import CustomTrainer
from src.llamafactory.train.ppo.trainer import CustomPPOTrainer
from ulysses_integration import patch_dpo_trainer, patch_sft_trainer, patch_pt_trainer, patch_ppo_trainer

# 修补各种训练器
CustomDPOTrainer = patch_dpo_trainer(CustomDPOTrainer)
CustomSeq2SeqTrainer = patch_sft_trainer(CustomSeq2SeqTrainer)
CustomTrainer = patch_pt_trainer(CustomTrainer)
CustomPPOTrainer = patch_ppo_trainer(CustomPPOTrainer)

# 然后正常使用修补后的训练器
"""

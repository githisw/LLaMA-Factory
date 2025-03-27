"""
训练器修补模块

此模块处理训练器修补相关的功能，支持不同类型的训练器。
"""

import torch
import logging
import sys
import os
from typing import Type, Dict, Any, Optional

# 设置日志
logger = logging.getLogger(__name__)

# 检查是否启用调试模式
DEBUG = os.environ.get("ULYSSES_DEBUG", "0") == "1"

# 从分布式训练模块导入
from .distributed_training import (
    wrap_attention_with_ulysses_attention,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank
)

# 梯度处理函数
def handle_gradients_func(trainer):
    """
    处理梯度，避免梯度消失和梯度爆炸
    
    Args:
        trainer: 训练器实例
    """
    # 获取所有参数的梯度
    gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    if not gradients:
        return
    
    # 获取当前训练步数和总步数
    current_step = trainer.state.global_step if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step') else 0
    total_steps = trainer.args.max_steps if hasattr(trainer, 'args') and hasattr(trainer.args, 'max_steps') else 1000
    
    # 计算梯度的平均值和标准差
    grad_mean = torch.mean(torch.stack([torch.mean(torch.abs(g)) for g in gradients]))
    grad_std = torch.std(torch.stack([torch.std(g) for g in gradients]))
    
    # 检查梯度是否过小（梯度消失）
    if grad_mean < 1e-3:  # 提高阈值，使检测更敏感
        logger.warning(f"检测到梯度可能消失: mean={grad_mean:.6f}, std={grad_std:.6f}")
        
        # 策略1: 添加梯度噪声，帮助模型跳出局部最小值
        # 在训练后期（最后10%的步骤）减少噪声
        if current_step > 0.9 * total_steps:
            noise_scale = 1e-5  # 训练后期使用非常小的噪声
            logger.info(f"训练后期（步骤 {current_step}/{total_steps}）：使用减小的噪声 {noise_scale}")
        else:
            noise_scale = 1e-4  # 降低噪声大小
        
        for p in trainer.model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * noise_scale)
        
        # 策略2: 梯度缩放，放大小梯度
        # 在训练后期减少缩放
        if current_step > 0.9 * total_steps:
            scale_factor = 1.1  # 训练后期使用更小的缩放因子
        else:
            scale_factor = 1.5  # 梯度缩放因子
            
        for p in trainer.model.parameters():
            if p.grad is not None:
                if torch.mean(torch.abs(p.grad)) < 1e-4:
                    p.grad.mul_(scale_factor)
        
        # 策略3: 如果有学习率调度器，临时增加学习率
        # 在训练后期不调整学习率
        if current_step <= 0.9 * total_steps and hasattr(trainer, 'optimizer') and hasattr(trainer.optimizer, 'param_groups'):
            for param_group in trainer.optimizer.param_groups:
                if 'lr' in param_group:
                    # 记录原始学习率
                    if not hasattr(trainer, '_original_lr'):
                        trainer._original_lr = param_group['lr']
                    
                    # 临时增加学习率
                    param_group['lr'] = param_group['lr'] * 1.2
                    logger.info(f"临时增加学习率至: {param_group['lr']:.8f}")
    else:
        # 如果梯度恢复正常，恢复原始学习率
        if hasattr(trainer, '_original_lr') and hasattr(trainer, 'optimizer'):
            for param_group in trainer.optimizer.param_groups:
                if 'lr' in param_group:
                    param_group['lr'] = trainer._original_lr
    
    # 策略4: 梯度中心化，减少梯度偏差
    for p in trainer.model.parameters():
        if p.grad is not None and p.grad.dim() > 1:
            p.grad.add_(-torch.mean(p.grad, dim=tuple(range(1, p.grad.dim())), keepdim=True))
    
    # 检查梯度是否过大（梯度爆炸）
    if grad_mean > 5.0:  # 降低阈值，使检测更敏感
        logger.warning(f"检测到梯度可能爆炸: mean={grad_mean:.6f}, std={grad_std:.6f}")
        # 裁剪梯度
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)

class TrainerPatchingModule:
    """处理训练器修补相关的功能"""
    
    @staticmethod
    def patch_trainer_by_stage(stage, debug=False):
        """根据训练阶段修补相应的训练器"""
        try:
            # 导入训练器类
            from src.llamafactory.train.dpo.trainer import CustomDPOTrainer
            from src.llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
            from src.llamafactory.train.pt.trainer import CustomTrainer
            from src.llamafactory.train.ppo.trainer import CustomPPOTrainer
            from src.llamafactory.train.kto.trainer import CustomKTOTrainer
            from src.llamafactory.train.rm.trainer import PairwiseTrainer as CustomRMTrainer
            
            if stage == "dpo":
                logger.info("修补DPO训练器以支持DeepSpeed-Ulysses...")
                patched_trainer = patch_dpo_trainer(CustomDPOTrainer, debug=debug)
                sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer = patched_trainer
                original_trainer = sys.modules["src.llamafactory.train.dpo.trainer"].CustomDPOTrainer
            elif stage == "sft":
                logger.info("修补SFT训练器以支持DeepSpeed-Ulysses...")
                patched_trainer = patch_sft_trainer(CustomSeq2SeqTrainer, debug=debug)
                sys.modules["src.llamafactory.train.sft.trainer"].CustomSeq2SeqTrainer = patched_trainer
                original_trainer = sys.modules["src.llamafactory.train.sft.trainer"].CustomSeq2SeqTrainer
            elif stage == "pt":
                logger.info("修补PT训练器以支持DeepSpeed-Ulysses...")
                patched_trainer = patch_pt_trainer(CustomTrainer, debug=debug)
                sys.modules["src.llamafactory.train.pt.trainer"].CustomTrainer = patched_trainer
                original_trainer = sys.modules["src.llamafactory.train.pt.trainer"].CustomTrainer
            elif stage == "ppo":
                logger.info("修补PPO训练器以支持DeepSpeed-Ulysses...")
                patched_trainer = patch_ppo_trainer(CustomPPOTrainer, debug=debug)
                sys.modules["src.llamafactory.train.ppo.trainer"].CustomPPOTrainer = patched_trainer
                original_trainer = sys.modules["src.llamafactory.train.ppo.trainer"].CustomPPOTrainer
            elif stage == "kto":
                logger.info("修补KTO训练器以支持DeepSpeed-Ulysses...")
                patched_trainer = patch_kto_trainer(CustomKTOTrainer, debug=debug)
                sys.modules["src.llamafactory.train.kto.trainer"].CustomKTOTrainer = patched_trainer
                original_trainer = sys.modules["src.llamafactory.train.kto.trainer"].CustomKTOTrainer
            elif stage == "rm":
                logger.info("修补RM训练器以支持DeepSpeed-Ulysses...")
                patched_trainer = patch_rm_trainer(CustomRMTrainer, debug=debug)
                sys.modules["src.llamafactory.train.rm.trainer"].PairwiseTrainer = patched_trainer
                original_trainer = sys.modules["src.llamafactory.train.rm.trainer"].PairwiseTrainer
            else:
                logger.warning(f"未知的训练阶段: {stage}。Ulysses可能无法正常工作。")
                original_trainer = None
            
            return original_trainer
        except ImportError as e:
            logger.error(f"导入训练器类时出错: {e}")
            return None
    
    @staticmethod
    def restore_original_trainer(stage, original_trainer):
        """恢复原始的训练器"""
        try:
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
            
            logger.info(f"已恢复原始的{stage}训练器")
            return True
        except Exception as e:
            logger.error(f"恢复原始训练器时出错: {e}")
            return False


# 通用的训练器修补函数 (优化版)
def patch_trainer(trainer_class, has_ref_model=False, debug=False):
    """
    通用的训练器修补函数，支持各种训练类型 (优化版)
    
    Args:
        trainer_class: 原始的训练器类
        has_ref_model: 训练器是否有参考模型（如DPO、PPO等）
        debug: 是否启用调试模式
        
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
                if torch.distributed.is_initialized():
                    # 修改模型以使用Ulysses注意力
                    is_dpo = "dpo" in trainer_class.__name__.lower()
                    self.model = wrap_attention_with_ulysses_attention(self.model, is_dpo, debug)
                    
                    # 如果有参考模型，也修改参考模型
                    if has_ref_model and hasattr(self, "ref_model") and self.ref_model is not None:
                        self.ref_model = wrap_attention_with_ulysses_attention(self.ref_model, is_dpo, debug)
                    
                    # 对于PPO训练器，还需要修改奖励模型
                    if hasattr(self, "reward_model") and self.reward_model is not None:
                        self.reward_model = wrap_attention_with_ulysses_attention(self.reward_model, False, debug)
                    
                    # 添加验证逻辑
                    if debug:
                        logger.info(f"修补训练器: {trainer_class.__name__}")
                        logger.info(f"模型: {type(self.model).__name__}")
                        if has_ref_model and hasattr(self, "ref_model") and self.ref_model is not None:
                            logger.info(f"参考模型: {type(self.ref_model).__name__}")
                        if hasattr(self, "reward_model") and self.reward_model is not None:
                            logger.info(f"奖励模型: {type(self.reward_model).__name__}")
        
        # 直接添加梯度处理方法到训练器实例
        if not hasattr(self, "handle_gradients"):
            self.handle_gradients = lambda: handle_gradients_func(self)
    
    # 替换__init__方法
    trainer_class.__init__ = patched_init
    
    # 如果是DPO训练器，还需要修改compute_loss方法
    if "dpo" in trainer_class.__name__.lower():
        if hasattr(trainer_class, "compute_loss"):
            original_compute_loss = trainer_class.compute_loss
            
            def patched_compute_loss(self, *args, **kwargs):
                # 在计算损失前添加验证逻辑
                if debug:
                    logger.info(f"计算DPO损失，参数: {args}")
                    logger.info(f"计算DPO损失，关键字参数: {kwargs}")
                
                # 调用原始的compute_loss方法
                loss = original_compute_loss(self, *args, **kwargs)
                
                # 在计算损失后添加验证逻辑
                if debug:
                    logger.info(f"DPO损失: {loss}")
                
                return loss
            
            # 替换compute_loss方法
            trainer_class.compute_loss = patched_compute_loss
    
    # 添加梯度处理方法
    def handle_gradients(self):
        """处理梯度，避免梯度消失和梯度爆炸"""
        # 获取所有参数的梯度
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients:
            return
        
        # 计算梯度的平均值和标准差
        grad_mean = torch.mean(torch.stack([torch.mean(torch.abs(g)) for g in gradients]))
        grad_std = torch.std(torch.stack([torch.std(g) for g in gradients]))
        
        # 检查梯度是否过小（梯度消失）
        if grad_mean < 1e-4:
            logger.warning(f"检测到梯度可能消失: mean={grad_mean:.6f}, std={grad_std:.6f}")
            # 添加梯度噪声，帮助模型跳出局部最小值
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * 1e-3)
        
        # 检查梯度是否过大（梯度爆炸）
        if grad_mean > 10.0:
            logger.warning(f"检测到梯度可能爆炸: mean={grad_mean:.6f}, std={grad_std:.6f}")
            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    
    # 添加清理内存的方法
    def cleanup_memory(self):
        """清理内存"""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # 添加方法
    trainer_class.handle_gradients = handle_gradients
    trainer_class.cleanup_memory = cleanup_memory
    
    return trainer_class

# 修改DPO训练器以支持DeepSpeed-Ulysses (优化版)
def patch_dpo_trainer(trainer_class, debug=False):
    """
    修补DPO训练器以支持DeepSpeed-Ulysses (优化版)
    
    Args:
        trainer_class: 原始的DPO训练器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的DPO训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=True, debug=debug)

# 修改SFT训练器以支持DeepSpeed-Ulysses (优化版)
def patch_sft_trainer(trainer_class, debug=False):
    """
    修补SFT训练器以支持DeepSpeed-Ulysses (优化版)
    
    Args:
        trainer_class: 原始的SFT训练器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的SFT训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=False, debug=debug)

# 修改PT训练器以支持DeepSpeed-Ulysses (优化版)
def patch_pt_trainer(trainer_class, debug=False):
    """
    修补PT训练器以支持DeepSpeed-Ulysses (优化版)
    
    Args:
        trainer_class: 原始的PT训练器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的PT训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=False, debug=debug)

# 修改PPO训练器以支持DeepSpeed-Ulysses (优化版)
def patch_ppo_trainer(trainer_class, debug=False):
    """
    修补PPO训练器以支持DeepSpeed-Ulysses (优化版)
    
    Args:
        trainer_class: 原始的PPO训练器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的PPO训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=True, debug=debug)

# 修改KTO训练器以支持DeepSpeed-Ulysses (优化版)
def patch_kto_trainer(trainer_class, debug=False):
    """
    修补KTO训练器以支持DeepSpeed-Ulysses (优化版)
    
    Args:
        trainer_class: 原始的KTO训练器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的KTO训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=True, debug=debug)

# 修改RM训练器以支持DeepSpeed-Ulysses (优化版)
def patch_rm_trainer(trainer_class, debug=False):
    """
    修补RM训练器以支持DeepSpeed-Ulysses (优化版)
    
    Args:
        trainer_class: 原始的RM训练器类
        debug: 是否启用调试模式
        
    Returns:
        修补后的RM训练器类
    """
    return patch_trainer(trainer_class, has_ref_model=False, debug=debug)

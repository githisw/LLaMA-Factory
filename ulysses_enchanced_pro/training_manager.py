"""
训练管理模块

此模块协调其他模块的工作并管理训练流程。
"""

import os
import sys
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple

# 设置日志
logger = logging.getLogger(__name__)

# 从其他模块导入
from .torch_compatibility import TorchCompatibilityModule
from .distributed_training import DistributedTrainingModule
from .trainer_patching import TrainerPatchingModule
from .data_processing import DataProcessingModule

class UlyssesTrainingManager:
    """管理Ulysses训练的主模块"""
    
    def __init__(self):
        """初始化UlyssesTrainingManager"""
        self.args = None
        self.model_args = None
        self.data_args = None
        self.training_args = None
        self.finetuning_args = None
        self.debug = os.environ.get("ULYSSES_DEBUG", "0") == "1"
        
        # 保存原始类，用于恢复
        self.original_trainers = {}
        self.original_collator = None
        
        # 初始化各个模块
        self.torch_compatibility = TorchCompatibilityModule()
        self.distributed_training = DistributedTrainingModule()
        self.trainer_patching = TrainerPatchingModule()
        self.data_processing = DataProcessingModule()
        
        if self.debug:
            logger.info("调试模式已启用")
    
    def initialize(self):
        """初始化训练环境"""
        # 修补PyTorch序列化
        self.torch_compatibility.patch_torch_serialization()
        
        # 解析命令行参数
        self.args = self._parse_args()
        
        # 获取训练参数
        self.model_args, self.data_args, self.training_args, self.finetuning_args, _ = self._get_train_args(self.args)
        
        # 检查是否指定了DeepSpeed配置
        if self.training_args.deepspeed is None:
            logger.warning("未指定DeepSpeed配置。使用改进的Ulysses配置。")
            self.training_args.deepspeed = "examples/deepspeed/ds_z3_ulysses_config_improved.json"
        
        # 修补数据整理器
        self.original_collator = self.data_processing.patch_data_collator_for_sequence_parallel(self.debug)
        
        # 根据训练类型修补相应的训练器
        stage = self.finetuning_args.stage
        self.original_trainers[stage] = self.trainer_patching.patch_trainer_by_stage(stage, self.debug)
        
        # 初始化序列并行组
        self.distributed_training.initialize_sequence_parallel_from_config(self.training_args.deepspeed)
        
        # 优化内存使用
        self.torch_compatibility.optimize_torch_memory()
        
        return self
    
    def run_training(self):
        """运行训练"""
        try:
            # 检查是否需要调整学习率
            adjust_lr = os.environ.get("ULYSSES_ADJUST_LR")
            if adjust_lr is not None:
                adjust_lr = float(adjust_lr)
                logger.info(f"调整学习率，因子: {adjust_lr}")
                
                # 调整学习率
                if hasattr(self.training_args, "learning_rate"):
                    original_lr = self.training_args.learning_rate
                    self.training_args.learning_rate = original_lr * adjust_lr
                    logger.info(f"学习率从 {original_lr} 调整为 {self.training_args.learning_rate}")
            
            # 添加梯度处理钩子
            self._add_gradient_hooks()
            
            # 导入并运行训练函数
            from src.llamafactory.train.tuner import run_exp
            run_exp(self.args)
        finally:
            # 恢复原始的类
            self.restore_original_classes()
    
    def restore_original_classes(self):
        """恢复原始的类"""
        # 恢复训练器
        stage = self.finetuning_args.stage if self.finetuning_args else None
        if stage in self.original_trainers and self.original_trainers[stage] is not None:
            self.trainer_patching.restore_original_trainer(stage, self.original_trainers[stage])
        
        # 恢复数据整理器
        if self.original_collator is not None:
            self.data_processing.restore_original_data_collator(self.original_collator)
        
        # 恢复torch.load
        self.torch_compatibility.restore_torch_load()
    
    def _parse_args(self):
        """解析命令行参数"""
        try:
            from src.llamafactory.hparams import read_args
            return read_args()
        except ImportError as e:
            logger.error(f"导入read_args时出错: {e}")
            sys.exit(1)
    
    def _get_train_args(self, args):
        """获取训练参数"""
        try:
            from src.llamafactory.hparams import get_train_args
            return get_train_args(args)
        except ImportError as e:
            logger.error(f"导入get_train_args时出错: {e}")
            sys.exit(1)
    
    def get_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            "debug": self.debug,
            "stage": self.finetuning_args.stage if self.finetuning_args else None,
            "deepspeed_config": self.training_args.deepspeed if self.training_args else None,
            "distributed": torch.distributed.is_initialized() if torch.distributed.is_available() else False,
            "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "sequence_parallel_initialized": self.distributed_training.get_sequence_parallel_group() is not None,
            "original_trainers": list(self.original_trainers.keys()),
            "original_collator": self.original_collator is not None,
        }
    
    def _add_gradient_hooks(self):
        """添加梯度处理钩子"""
        try:
            # 导入训练器类
            from transformers import Trainer
            
            # 保存原始的training_step方法
            if hasattr(Trainer, "training_step"):
                original_training_step = Trainer.training_step
                
                def patched_training_step(self, *args, **kwargs):
                    """修补后的training_step方法，在训练步骤中处理梯度"""
                    # 调用原始的training_step方法
                    loss = original_training_step(self, *args, **kwargs)
                    
                    # 处理梯度，避免梯度消失和梯度爆炸
                    if hasattr(self, "handle_gradients"):
                        self.handle_gradients()
                    
                    return loss
                
                # 替换training_step方法
                Trainer.training_step = patched_training_step
                logger.info("已添加梯度处理钩子到training_step方法")
            else:
                # 如果没有training_step方法，尝试直接添加handle_gradients方法
                logger.warning("未找到training_step方法，尝试直接添加handle_gradients方法")
                
                # 在训练器初始化时添加梯度处理
                original_init = Trainer.__init__
                
                def patched_init(self, *args, **kwargs):
                    # 调用原始的__init__方法
                    original_init(self, *args, **kwargs)
                    
                    # 添加梯度处理方法
                    if not hasattr(self, "handle_gradients"):
                        from ulysses_enchanced_pro.trainer_patching import patch_trainer
                        patch_trainer(self.__class__)
                
                # 替换__init__方法
                Trainer.__init__ = patched_init
                logger.info("已添加梯度处理钩子到__init__方法")
            
        except Exception as e:
            logger.error(f"添加梯度处理钩子时出错: {e}")
    
    def cleanup(self):
        """清理资源"""
        # 恢复原始的类
        self.restore_original_classes()
        
        # 清理内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("已清理资源")

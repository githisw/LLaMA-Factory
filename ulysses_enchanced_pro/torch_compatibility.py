"""
PyTorch兼容性模块

此模块处理PyTorch兼容性问题，特别是PyTorch 2.6的反序列化问题。
"""

import torch
import logging

# 设置日志
logger = logging.getLogger(__name__)

class TorchCompatibilityModule:
    """处理PyTorch兼容性问题的模块，特别是PyTorch 2.6的反序列化问题"""
    
    @staticmethod
    def patch_torch_serialization():
        """修补PyTorch序列化，解决PyTorch 2.6的反序列化问题"""
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
            logger.info(f"成功添加 {len(safe_classes)} 个DeepSpeed类到安全全局列表")
            
            # 设置torch.load的weights_only参数为False
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            torch.load = patched_torch_load
            logger.info("修补torch.load以默认使用weights_only=False")
            
            return True
        except (ImportError, AttributeError) as e:
            logger.warning(f"无法添加DeepSpeed类到安全全局列表: {e}")
            return False
    
    @staticmethod
    def restore_torch_load():
        """恢复原始的torch.load函数"""
        if hasattr(torch, "_original_load"):
            torch.load = torch._original_load
            logger.info("已恢复原始的torch.load函数")
            return True
        return False
    
    @staticmethod
    def optimize_torch_memory():
        """优化PyTorch内存使用"""
        # 设置PyTorch内存分配器选项
        torch.backends.cuda.matmul.allow_tf32 = True  # 使用TF32精度，提高性能
        torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试，提高性能
        torch.backends.cudnn.deterministic = False  # 禁用确定性，提高性能
        
        # 设置环境变量以避免内存碎片化
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # 清理内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("已优化PyTorch内存使用")
        return True

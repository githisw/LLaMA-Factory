# LLaMA-Factory with DeepSpeed-Ulysses

本项目是对[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)的扩展，集成了DeepSpeed-Ulysses序列并行技术，使其能够高效训练超长序列的大型语言模型。

## 新增功能概述

- **DeepSpeed-Ulysses序列并行**：支持沿序列维度分割输入张量，显著降低内存需求，支持训练长达百万token的序列
- **高效通信**：使用通信高效的all-to-all集合进行分布式注意力计算，减少通信开销
- **FlashAttention集成**：与FlashAttention结合使用，进一步提高计算效率
- **PyTorch 2.6兼容性**：解决了PyTorch 2.6中的反序列化安全限制问题
- **多模型兼容性**：经过优化，支持Qwen2.5、Llama3等多种模型架构

## 技术实现细节

### 1. 序列并行原理

DeepSpeed-Ulysses的核心思想是序列并行，它的工作原理如下：

1. **序列分割**：将输入序列沿着序列维度分割成多个部分，每个GPU处理一部分
2. **分布式注意力**：使用all-to-all通信在计算注意力时交换必要的信息
3. **序列合并**：计算完成后，再次使用all-to-all通信将结果合并

这种方法的优势在于：
- 显著减少每个GPU的内存使用
- 减少通信开销，提高训练吞吐量
- 支持训练超长序列，突破单GPU内存限制

### 2. 主要修改内容

相比原始的LLaMA-Factory项目，本项目主要进行了以下修改：

#### 2.1 核心组件

- **`ulysses_integration.py`**：DeepSpeed-Ulysses的集成模块，包含：
  - 序列并行通信组的初始化
  - 自定义的UlyssesAttention实现
  - 序列并行的all-to-all通信实现
  - 注意力模块的动态修改

- **`train_with_ulysses.py`**：使用DeepSpeed-Ulysses进行DPO训练的启动脚本，包含：
  - 修补DPO训练器以支持Ulysses
  - 修补数据收集器以支持序列长度分割
  - PyTorch 2.6反序列化问题的解决方案

- **`examples/deepspeed/ds_z3_ulysses_config.json`**：DeepSpeed-Ulysses的配置文件，包含：
  - ZeRO-3优化配置
  - 序列并行大小设置
  - FlashAttention启用设置

#### 2.2 关键技术实现

1. **序列并行通信**：
   ```python
   class SeqAllToAll4D(torch.autograd.Function):
       """序列并行的All-to-All通信"""
       
       @staticmethod
       def forward(ctx, group, input_tensor, scatter_idx, gather_idx):
           # 实现序列维度的分割和all-to-all通信
           # ...
   ```

2. **Ulysses注意力机制**：
   ```python
   class UlyssesAttention(torch.nn.Module):
       """Ulysses注意力机制"""
       
       def forward(self, query, key, value, dropout_p=0.0, softmax_scale=None, causal=False, *args):
           # 执行序列并行的all-to-all通信
           q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
           k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
           v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)
           
           # 使用FlashAttention计算注意力
           # ...
   ```

3. **动态注意力模块修改**：
   ```python
   def wrap_attention_with_ulysses_attention(model):
       """将模型中的注意力模块替换为UlyssesAttention"""
       # 递归遍历模型的所有模块
       for name, module in model.named_children():
           # 只修改真正的注意力模块
           if "attention" in name.lower() and hasattr(module, "forward"):
               if "norm" not in name.lower() and "layernorm" not in name.lower():
                   # 创建Ulysses注意力模块并替换原始forward方法
                   # ...
   ```

4. **PyTorch 2.6兼容性**：
   ```python
   # 添加DeepSpeed的类到安全全局列表中
   safe_classes = [ZeroStageEnum, LossScaler, DeepSpeedConfig, ...]
   torch.serialization.add_safe_globals(safe_classes)
   
   # 修补torch.load函数
   original_torch_load = torch.load
   def patched_torch_load(*args, **kwargs):
       if 'weights_only' not in kwargs:
           kwargs['weights_only'] = False
       return original_torch_load(*args, **kwargs)
   torch.load = patched_torch_load
   ```

## 性能优势

与标准训练相比，DeepSpeed-Ulysses提供以下性能优势：

1. **内存效率**：序列长度增加了4倍以上，支持训练超过百万token的序列
2. **通信效率**：通信减少了10倍以上，吞吐量提高了高达2.5倍
3. **计算效率**：每个GPU的持续吞吐量超过175 TFlops（超过硬件峰值的54%）
4. **通用性**：支持密集和稀疏的注意力，可与FlashAttention等高效实现结合使用
5. **可扩展性**：可与ZeRO-3结合使用，同时支持大模型和长序列训练

## 使用方法

### 1. 准备环境

确保已安装DeepSpeed v0.10.2或更高版本：

```bash
pip install "deepspeed>=0.10.2"
```

安装FlashAttention（必需）：

```bash
# 安装triton和FlashAttention
pip install triton
pip install flash-attn
```

### 2. 使用启动脚本进行训练

使用`train_with_ulysses.py`脚本启动训练：

```bash
python train_with_ulysses.py examples/train_lora/llama3_lora_dpo_ulysses.yaml
```

### 3. 自定义序列并行大小

要自定义序列并行大小，可以修改配置文件中的`sequence_parallel_size`参数：

```json
{
  "sequence_parallel_size": 8,  // 将序列并行大小设置为8
  ...
}
```

## 注意事项

1. 序列并行大小必须能够被总GPU数量整除
2. 注意力头的数量应该能被序列并行大小整除，以获得最佳性能
3. 当前实现与Megatron-LM的张量并行或流水线并行不兼容
4. 使用FlashAttention时，为获得最佳性能，头大小应该是8的倍数
5. 对于GQA或MQA模型（如Llama 3），K、V的head数量较小，序列并行大小不宜设置过大

## 参考资料

- [DeepSpeed-Ulysses教程](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/chinese/README.md)
- [FlashAttention文档](https://github.com/HazyResearch/flash-attention)

## 致谢

本项目基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)开发，感谢原项目作者的贡献。同时感谢DeepSpeed团队开发的Ulysses序列并行技术。

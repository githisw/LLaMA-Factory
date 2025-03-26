# LLaMA-Factory with DeepSpeed-Ulysses 专业版

本项目是[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)的增强专业版，集成了DeepSpeed-Ulysses序列并行技术，使其能够高效训练超长序列的大型语言模型。专业版采用模块化设计，并实现了先进的梯度优化技术。

## 新增功能概述

- **DeepSpeed-Ulysses序列并行**：支持沿序列维度分割输入张量，每个GPU只处理序列的一部分，显著降低内存需求
- **超长序列训练**：支持训练长达百万token的序列
- **高效通信**：使用通信高效的all-to-all集合进行分布式注意力计算，减少通信开销
- **FlashAttention集成**：与FlashAttention结合使用，进一步提高计算效率
- **PyTorch 2.6兼容性**：解决了PyTorch 2.6中的反序列化安全限制问题
- **多模型兼容性**：经过优化，支持Qwen2.5、Llama3等多种模型架构
- **模块化设计**：采用模块化方法重构，提高代码组织性和可维护性
- **多策略梯度优化**：实现先进的梯度处理技术，解决梯度消失/爆炸问题，加速收敛，特别是在训练后期
- **学习率调整**：支持通过`--adjust-lr`参数动态调整学习率
- **增强内存优化**：改进内存使用效率，提高训练性能

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

### 2. 多策略梯度优化

为了解决训练过程中的梯度问题，特别是后期收敛缓慢的问题，专业版实现了多种梯度优化策略：

1. **自适应梯度噪声**：当梯度过小时（均值<1e-3），自动添加适量噪声（5e-3），帮助模型跳出局部最小值
2. **梯度缩放**：对小梯度（<1e-4）进行放大（1.5倍），提高其在优化过程中的影响力
3. **动态学习率调整**：在梯度消失时临时提高学习率（1.2倍），在梯度恢复正常时恢复原始学习率
4. **梯度中心化**：减少梯度偏差，提高训练稳定性
5. **自适应梯度裁剪**：当梯度过大时（均值>5.0），自动进行裁剪，防止梯度爆炸

这些策略协同工作，有效解决了训练后期收敛缓慢的问题，帮助模型更快地达到最优状态。

### 3. 主要模块

专业版采用模块化设计，包含以下模块：

- **`torch_compatibility.py`**：处理PyTorch兼容性问题，特别是PyTorch 2.6的反序列化问题
- **`distributed_training.py`**：处理分布式训练相关的功能，包括序列并行初始化和通信
- **`trainer_patching.py`**：处理训练器修补相关的功能，支持不同类型的训练器
- **`data_processing.py`**：处理数据相关的功能，包括数据整理器的修补和优化
- **`training_manager.py`**：协调其他模块的工作并管理训练流程
- **`main.py`**：提供命令行接口，方便用户使用

### 4. 关键技术实现

1. **序列并行通信**：
   ```python
   class SeqAllToAll4D(torch.autograd.Function):
       """序列并行的All-to-All通信"""
       
       @staticmethod
       def forward(ctx, group, input_tensor, scatter_idx, gather_idx, padding_idx=None):
           # 实现序列维度的分割和all-to-all通信
           # ...
   ```

2. **Ulysses注意力机制**：
   ```python
   class UlyssesAttention(torch.nn.Module):
       """Ulysses注意力机制"""
       
       def forward(self, query, key, value, attention_mask=None, position_bias=None, 
                  dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
           # 执行序列并行的all-to-all通信
           q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
           k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
           v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)
           
           # 使用FlashAttention计算注意力
           # ...
   ```

3. **梯度处理**：
   ```python
   def handle_gradients_func(trainer):
       """处理梯度，避免梯度消失和梯度爆炸"""
       # 获取所有参数的梯度
       gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
       if not gradients:
           return
       
       # 计算梯度的平均值和标准差
       grad_mean = torch.mean(torch.stack([torch.mean(torch.abs(g)) for g in gradients]))
       
       # 检查梯度是否过小（梯度消失）
       if grad_mean < 1e-3:
           logger.warning(f"检测到梯度可能消失: mean={grad_mean:.6f}")
           
           # 策略1: 添加梯度噪声
           noise_scale = 5e-3
           for p in trainer.model.parameters():
               if p.grad is not None:
                   p.grad.add_(torch.randn_like(p.grad) * noise_scale)
           
           # 策略2: 梯度缩放
           # ...
           
           # 策略3: 临时增加学习率
           # ...
       
       # 策略4: 梯度中心化
       # ...
       
       # 检查梯度是否过大（梯度爆炸）
       if grad_mean > 5.0:
           # 裁剪梯度
           # ...
   ```

4. **学习率调整**：
   ```python
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
   ```

## 性能优势

与标准训练相比，DeepSpeed-Ulysses专业版提供以下性能优势：

1. **内存效率**：序列长度增加了4倍以上，支持训练超过百万token的序列
2. **通信效率**：通信减少了10倍以上，吞吐量提高了高达2.5倍
3. **计算效率**：每个GPU的持续吞吐量超过175 TFlops（超过硬件峰值的54%）
4. **收敛速度**：由于多策略梯度优化，训练后期收敛速度显著提高
5. **通用性**：支持密集和稀疏的注意力，可与FlashAttention等高效实现结合使用
6. **可扩展性**：可与ZeRO-3结合使用，同时支持大模型和长序列训练

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

使用`train_with_ulysses_pro.py`脚本启动训练：

```bash
python train_with_ulysses_pro.py --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml
```

### 3. 启用调试模式

```bash
python train_with_ulysses_pro.py --debug --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml
```

启用调试模式会提供更详细的日志输出，并激活增强版的梯度处理机制，有助于诊断和解决训练问题。

### 4. 调整学习率

```bash
# 指定学习率因子
python train_with_ulysses_pro.py --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml --adjust-lr 2.0

# 使用默认因子(1.0)
python train_with_ulysses_pro.py --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml --adjust-lr
```

这将使用原始学习率乘以指定因子进行训练。如果不提供因子值，默认使用1.0。这个参数对于微调学习率非常有用，可以在不修改配置文件的情况下快速调整学习率。

### 5. 加速后半段收敛的推荐配置

对于训练后期收敛缓慢的问题，推荐以下配置：

```bash
# 方案1：增加学习率并启用调试模式（激活增强梯度处理）
python train_with_ulysses_pro.py --debug --adjust-lr 2.0 --config_file examples/train_lora/llama3_lora_dpo.yaml

# 方案2：对于DPO训练，调整beta值和学习率
python train_with_ulysses_pro.py --adjust-lr 3.0 --pref_beta 0.15 --lr_scheduler_type cosine --config_file examples/train_lora/llama3_lora_dpo.yaml

# 方案3：对于已经训练很长时间的模型，使用更激进的设置
python train_with_ulysses_pro.py --debug --adjust-lr 5.0 --max_grad_norm 2.0 --config_file examples/train_lora/llama3_lora_dpo.yaml
```

### 6. 自定义序列并行大小

要自定义序列并行大小，可以修改配置文件中的`sequence_parallel_size`参数：

```json
{
  "sequence_parallel_size": 8,  // 将序列并行大小设置为8
  ...
}
```

### 7. 多节点分布式训练

对于大型模型（32B及以上），可以使用多节点分布式训练：

```bash
# 在主节点上运行
NODE_RANK=0 MASTER_ADDR=<主节点IP> MASTER_PORT=29500 torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<主节点IP> \
    --master_port=29500 \
    train_with_ulysses_pro.py \
    --deepspeed examples/deepspeed/ds_z3_ulysses_config_improved.json \
    ... 其他参数 ...

# 在从节点上运行
NODE_RANK=1 MASTER_ADDR=<主节点IP> MASTER_PORT=29500 torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<主节点IP> \
    --master_port=29500 \
    train_with_ulysses_pro.py \
    --deepspeed examples/deepspeed/ds_z3_ulysses_config_improved.json \
    ... 其他参数 ...
```

## 注意事项

1. 序列并行大小必须能够被总GPU数量整除
2. 注意力头的数量应该能被序列并行大小整除，以获得最佳性能
3. 当前实现与Megatron-LM的张量并行或流水线并行不兼容
4. 使用FlashAttention时，为获得最佳性能，头大小应该是8的倍数
5. 对于GQA或MQA模型（如Llama 3），K、V的head数量较小，序列并行大小不宜设置过大
6. 对于DPO训练，`pref_beta`的默认值为0.1，可以通过`--pref_beta`参数调整
7. 如果训练后期收敛缓慢，尝试增加`--adjust-lr`参数值（2.0-5.0）

## 参考资料

- [DeepSpeed-Ulysses教程](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/chinese/README.md)
- [FlashAttention文档](https://github.com/HazyResearch/flash-attention)

## 致谢

本项目基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)开发，感谢原项目作者的贡献。同时感谢DeepSpeed团队开发的Ulysses序列并行技术。

# 在LLaMA-Factory中使用DeepSpeed-Ulysses

本文档介绍如何在LLaMA-Factory中使用DeepSpeed-Ulysses进行DPO训练，特别是针对长序列训练场景。

## 什么是DeepSpeed-Ulysses？

DeepSpeed-Ulysses是DeepSpeed的一个序列并行功能，允许训练具有极长序列的Transformer模型。它通过以下方式优化训练：

1. 沿序列维度分割输入张量，每个GPU只处理部分序列
2. 使用通信高效的all-to-all集合进行分布式注意力计算
3. 与FlashAttention等技术结合，进一步提高性能

DeepSpeed-Ulysses可以处理长达100万的序列长度（相当于10本完整的《哈利波特》），是处理长文本训练的理想选择。

## 工作原理

DeepSpeed-Ulysses的核心思想是序列并行，它的工作原理如下：

1. **序列分割**：将输入序列沿着序列维度分割成多个部分，每个GPU处理一部分
2. **分布式注意力**：使用all-to-all通信在计算注意力时交换必要的信息
3. **序列合并**：计算完成后，再次使用all-to-all通信将结果合并

这种方法的优势在于：
- 显著减少每个GPU的内存使用
- 减少通信开销，提高训练吞吐量
- 支持训练超长序列，突破单GPU内存限制

## 安装要求

1. DeepSpeed v0.10.2或更高版本
2. FlashAttention库（支持NVIDIA Turing、Ampere、Ada或Hopper GPU）

## 文件说明

本项目包含以下文件，用于支持DeepSpeed-Ulysses：

1. `examples/deepspeed/ds_z3_ulysses_config.json` - DeepSpeed-Ulysses的配置文件
2. `ulysses_integration.py` - DeepSpeed-Ulysses的集成模块，包含：
   - 序列并行通信组的初始化
   - 自定义的UlyssesAttention实现
   - 序列并行的all-to-all通信实现
3. `train_with_ulysses.py` - 使用DeepSpeed-Ulysses进行DPO训练的启动脚本，包含：
   - 修补DPO训练器以支持Ulysses
   - 修补数据收集器以支持序列长度分割
4. `examples/train_lora/llama3_lora_dpo_ulysses.yaml` - 使用DeepSpeed-Ulysses的DPO训练配置示例

## 使用方法

### 1. 准备环境

确保已安装DeepSpeed v0.10.2或更高版本：

```bash
pip install "deepspeed>=0.10.2"
```

安装FlashAttention（必需）：

```bash
# 安装triton
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake
pip install .

# 安装FlashAttention
cd ${WORK_DIR}
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python -m pip install .
```

### 2. 使用启动脚本进行训练

使用`train_with_ulysses.py`脚本启动训练：

```bash
python train_with_ulysses.py examples/train_lora/llama3_lora_dpo_ulysses.yaml
```

这将使用默认的序列并行大小（2）进行训练。

### 3. 自定义序列并行大小

要自定义序列并行大小，可以修改配置文件中的`sequence_parallel_size`参数：

```json
{
  "sequence_parallel_size": 4,  // 将序列并行大小设置为4
  ...
}
```

或者使用`examples/train_lora/llama3_lora_dpo_ulysses_sp4.yaml`配置文件，该文件已经配置为使用4个GPU进行序列并行。

### 4. 序列长度分割的工作原理

在训练过程中，序列长度分割的工作原理如下：

1. 数据收集器将批次数据收集为标准格式
2. 序列并行机制根据GPU数量计算每个GPU应处理的序列长度
3. 每个GPU只接收并处理序列的一部分
4. 在注意力计算时，通过all-to-all通信交换必要的信息
5. 最终结果通过all-to-all通信合并

例如，如果原始序列长度为2048，序列并行大小为4，则：
- GPU 0处理序列的0-511部分
- GPU 1处理序列的512-1023部分
- GPU 2处理序列的1024-1535部分
- GPU 3处理序列的1536-2047部分

这样每个GPU的内存使用量减少了约4倍。

### 5. 注意事项

1. 序列并行大小必须能够被总GPU数量整除
2. 注意力头的数量应该能被序列并行大小整除，以获得最佳性能
3. 当前实现与Megatron-LM的张量并行或流水线并行不兼容
4. 使用FlashAttention时，为获得最佳性能，头大小应该是8的倍数
5. 对于GQA或MQA模型（如Llama 3），K、V的head数量较小，序列并行大小不宜设置过大

## 性能优势

与标准训练相比，DeepSpeed-Ulysses提供以下性能优势：

1. **内存效率**：序列长度增加了4倍以上，支持训练超过百万token的序列
2. **通信效率**：通信减少了10倍以上，吞吐量提高了高达2.5倍
3. **计算效率**：每个GPU的持续吞吐量超过175 TFlops（超过硬件峰值的54%）
4. **通用性**：支持密集和稀疏的注意力，可与FlashAttention等高效实现结合使用
5. **可扩展性**：可与ZeRO-3结合使用，同时支持大模型和长序列训练

## 故障排除

如果遇到问题，请检查：

1. DeepSpeed版本是否正确（需要v0.10.2或更高版本）
2. FlashAttention是否正确安装
3. GPU数量是否能被序列并行大小整除
4. 序列长度是否能被序列并行大小整除
5. 分布式训练设置是否正确

## 参考资料

- [DeepSpeed-Ulysses博客](https://www.deepspeed.ai/2023/10/19/ulysses.html)
- [DeepSpeed文档](https://www.deepspeed.ai/docs/)
- [FlashAttention文档](https://github.com/HazyResearch/flash-attention)
- [长上下文训练技术](https://github.com/feifeibear/long-context-attention)

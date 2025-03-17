# LLaMA-Factory with DeepSpeed-Ulysses

This project is an extension of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) that integrates DeepSpeed-Ulysses sequence parallel technology, enabling efficient training of large language models with ultra-long sequences.

## Overview of New Features

- **DeepSpeed-Ulysses Sequence Parallelism**: Supports splitting input tensors along the sequence dimension, with each GPU processing only a portion of the sequence, significantly reducing memory requirements
- **Ultra-long Sequence Training**: Supports training sequences up to 1 million tokens (equivalent to 10 complete Harry Potter books)
- **Efficient Communication**: Uses communication-efficient all-to-all collectives for distributed attention computation, reducing communication overhead
- **FlashAttention Integration**: Combined with FlashAttention for further computational efficiency
- **PyTorch 2.6 Compatibility**: Resolves deserialization security restriction issues in PyTorch 2.6
- **Multi-model Compatibility**: Optimized to support various model architectures including Qwen2.5, Llama3, and more

## Technical Implementation Details

### 1. Sequence Parallelism Principle

The core idea of DeepSpeed-Ulysses is sequence parallelism, which works as follows:

1. **Sequence Splitting**: The input sequence is split along the sequence dimension into multiple parts, with each GPU processing a portion
2. **Distributed Attention**: All-to-all communication is used to exchange necessary information when computing attention
3. **Sequence Merging**: After computation, all-to-all communication is used again to merge the results

The advantages of this approach include:
- Significantly reduced memory usage per GPU
- Reduced communication overhead, improving training throughput
- Support for ultra-long sequences, breaking through single GPU memory limitations

### 2. Main Modifications

Compared to the original LLaMA-Factory project, this project has made the following modifications:

#### 2.1 Core Components

- **`ulysses_integration.py`**: DeepSpeed-Ulysses integration module, including:
  - Initialization of sequence parallel communication groups
  - Custom UlyssesAttention implementation
  - Sequence parallel all-to-all communication implementation
  - Dynamic modification of attention modules

- **`train_with_ulysses.py`**: Launch script for DPO training with DeepSpeed-Ulysses, including:
  - Patching DPO trainer to support Ulysses
  - Patching data collectors to support sequence length splitting
  - Solutions for PyTorch 2.6 deserialization issues

- **`examples/deepspeed/ds_z3_ulysses_config.json`**: Configuration file for DeepSpeed-Ulysses, including:
  - ZeRO-3 optimization configuration
  - Sequence parallel size settings
  - FlashAttention enablement settings

#### 2.2 Key Technical Implementations

1. **Sequence Parallel Communication**:
   ```python
   class SeqAllToAll4D(torch.autograd.Function):
       """All-to-All communication for sequence parallelism"""
       
       @staticmethod
       def forward(ctx, group, input_tensor, scatter_idx, gather_idx):
           # Implementation of sequence dimension splitting and all-to-all communication
           # ...
   ```

2. **Ulysses Attention Mechanism**:
   ```python
   class UlyssesAttention(torch.nn.Module):
       """Ulysses attention mechanism"""
       
       def forward(self, query, key, value, dropout_p=0.0, softmax_scale=None, causal=False, *args):
           # Perform sequence parallel all-to-all communication
           q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
           k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
           v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)
           
           # Calculate attention using FlashAttention
           # ...
   ```

3. **Dynamic Attention Module Modification**:
   ```python
   def wrap_attention_with_ulysses_attention(model):
       """Replace attention modules in the model with UlyssesAttention"""
       # Recursively traverse all modules in the model
       for name, module in model.named_children():
           # Only modify actual attention modules
           if "attention" in name.lower() and hasattr(module, "forward"):
               if "norm" not in name.lower() and "layernorm" not in name.lower():
                   # Create Ulysses attention module and replace the original forward method
                   # ...
   ```

4. **PyTorch 2.6 Compatibility**:
   ```python
   # Add DeepSpeed classes to the safe global list
   safe_classes = [ZeroStageEnum, LossScaler, DeepSpeedConfig, ...]
   torch.serialization.add_safe_globals(safe_classes)
   
   # Patch torch.load function
   original_torch_load = torch.load
   def patched_torch_load(*args, **kwargs):
       if 'weights_only' not in kwargs:
           kwargs['weights_only'] = False
       return original_torch_load(*args, **kwargs)
   torch.load = patched_torch_load
   ```

## Performance Advantages

Compared to standard training, DeepSpeed-Ulysses provides the following performance advantages:

1. **Memory Efficiency**: Sequence length increased by more than 4x, supporting training of sequences over one million tokens
2. **Communication Efficiency**: Communication reduced by more than 10x, throughput improved by up to 2.5x
3. **Computational Efficiency**: Sustained throughput per GPU exceeding 175 TFlops (over 54% of hardware peak)
4. **Versatility**: Supports both dense and sparse attention, can be combined with efficient implementations like FlashAttention
5. **Scalability**: Can be combined with ZeRO-3, supporting both large model and long sequence training

## Usage

### 1. Environment Preparation

Ensure DeepSpeed v0.10.2 or higher is installed:

```bash
pip install "deepspeed>=0.10.2"
```

Install FlashAttention (required):

```bash
# Install triton
git clone -b legacy-backend https://github.com/openai/triton
cd triton/python/
pip install cmake
pip install .

# Install FlashAttention
cd ${WORK_DIR}
git clone -b v1.0.4 https://github.com/HazyResearch/flash-attention
cd flash-attention
python -m pip install .
```

### 2. Training with Launch Script

Use the `train_with_ulysses.py` script to start training:

```bash
python train_with_ulysses.py examples/train_lora/llama3_lora_dpo_ulysses.yaml
```

### 3. Customizing Sequence Parallel Size

To customize the sequence parallel size, modify the `sequence_parallel_size` parameter in the configuration file:

```json
{
  "sequence_parallel_size": 8,  // Set sequence parallel size to 8
  ...
}
```

## Notes

1. Sequence parallel size must be divisible by the total number of GPUs
2. The number of attention heads should be divisible by the sequence parallel size for optimal performance
3. The current implementation is not compatible with Megatron-LM's tensor parallelism or pipeline parallelism
4. When using FlashAttention, for best performance, the head size should be a multiple of 8
5. For GQA or MQA models (such as Llama 3), the number of K, V heads is smaller, so the sequence parallel size should not be set too large

## References

- [DeepSpeed-Ulysses Blog](https://www.deepspeed.ai/2023/10/19/ulysses.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
- [FlashAttention Documentation](https://github.com/HazyResearch/flash-attention)
- [Long Context Training Techniques](https://github.com/feifeibear/long-context-attention)

## Acknowledgements

This project is developed based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks to the original project authors for their contributions. Also thanks to the DeepSpeed team for developing the Ulysses sequence parallel technology.

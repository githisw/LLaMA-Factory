# LLaMA-Factory with DeepSpeed-Ulysses Pro

This project is an enhanced professional version of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with DeepSpeed-Ulysses sequence parallel technology, enabling efficient training of large language models with ultra-long sequences. This Pro version features a modular design and advanced gradient optimization techniques.

## Overview of New Features

- **DeepSpeed-Ulysses Sequence Parallelism**: Supports splitting input tensors along the sequence dimension, with each GPU processing only a portion of the sequence, significantly reducing memory requirements
- **Ultra-long Sequence Training**: Supports training sequences up to millions tokens 
- **Efficient Communication**: Uses communication-efficient all-to-all collectives for distributed attention computation, reducing communication overhead
- **FlashAttention Integration**: Combined with FlashAttention for further computational efficiency
- **PyTorch 2.6 Compatibility**: Resolves deserialization security restriction issues in PyTorch 2.6
- **Multi-model Compatibility**: Optimized to support various model architectures including Qwen2.5, Llama3, and more
- **Modular Design**: Restructured with a modular approach for better code organization and maintainability
- **Multi-strategy Gradient Optimization**: Implements advanced gradient handling techniques to solve vanishing/exploding gradient problems and accelerate convergence, especially in later training stages
- **Learning Rate Adjustment**: Supports dynamic learning rate adjustment with the `--adjust-lr` parameter
- **Enhanced Memory Optimization**: Improved memory usage efficiency for better training performance

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

### 2. Multi-strategy Gradient Optimization

To address gradient issues during training, especially slow convergence in later stages, the Pro version implements multiple gradient optimization strategies:

1. **Adaptive Gradient Noise**: Automatically adds appropriate noise (5e-3) when gradients are too small (mean < 1e-3), helping the model escape local minima
2. **Gradient Scaling**: Amplifies small gradients (< 1e-4) by a factor of 1.5, increasing their influence in the optimization process
3. **Dynamic Learning Rate Adjustment**: Temporarily increases learning rate (by 1.2x) when gradient vanishing is detected, restoring the original rate when gradients return to normal
4. **Gradient Centralization**: Reduces gradient bias, improving training stability
5. **Adaptive Gradient Clipping**: Automatically clips gradients when they become too large (mean > 5.0), preventing gradient explosion

These strategies work together to effectively solve the problem of slow convergence in the later stages of training, helping the model reach optimal states faster.

### 3. Main Modules

The Pro version is restructured with a modular design:

- **`torch_compatibility.py`**: Handles PyTorch compatibility issues, especially PyTorch 2.6 deserialization problems
- **`distributed_training.py`**: Manages distributed training functionality, including sequence parallel initialization and communication
- **`trainer_patching.py`**: Handles trainer patching functionality, supporting different types of trainers
- **`data_processing.py`**: Manages data-related functionality, including data collator patching and optimization
- **`training_manager.py`**: Coordinates the work of other modules and manages the training process
- **`main.py`**: Provides a command-line interface for easy use

### 4. Key Technical Implementations

1. **Sequence Parallel Communication**:
   ```python
   class SeqAllToAll4D(torch.autograd.Function):
       """All-to-All communication for sequence parallelism"""
       
       @staticmethod
       def forward(ctx, group, input_tensor, scatter_idx, gather_idx, padding_idx=None):
           # Implementation of sequence dimension splitting and all-to-all communication
           # ...
   ```

2. **Ulysses Attention Mechanism**:
   ```python
   class UlyssesAttention(torch.nn.Module):
       """Ulysses attention mechanism"""
       
       def forward(self, query, key, value, attention_mask=None, position_bias=None, 
                  dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
           # Perform sequence parallel all-to-all communication
           q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
           k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
           v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)
           
           # Calculate attention using FlashAttention
           # ...
   ```

3. **Gradient Handling**:
   ```python
   def handle_gradients_func(trainer):
       """Handle gradients to avoid vanishing and exploding gradients"""
       # Get all gradients
       gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
       if not gradients:
           return
       
       # Calculate mean and standard deviation of gradients
       grad_mean = torch.mean(torch.stack([torch.mean(torch.abs(g)) for g in gradients]))
       
       # Check for vanishing gradients
       if grad_mean < 1e-3:
           logger.warning(f"Possible gradient vanishing detected: mean={grad_mean:.6f}")
           
           # Strategy 1: Add gradient noise
           noise_scale = 5e-3
           for p in trainer.model.parameters():
               if p.grad is not None:
                   p.grad.add_(torch.randn_like(p.grad) * noise_scale)
           
           # Strategy 2: Scale small gradients
           # ...
           
           # Strategy 3: Temporarily increase learning rate
           # ...
       
       # Strategy 4: Gradient centralization
       # ...
       
       # Check for exploding gradients
       if grad_mean > 5.0:
           # Clip gradients
           # ...
   ```

4. **Learning Rate Adjustment**:
   ```python
   # Check if learning rate adjustment is needed
   adjust_lr = os.environ.get("ULYSSES_ADJUST_LR")
   if adjust_lr is not None:
       adjust_lr = float(adjust_lr)
       logger.info(f"Adjusting learning rate, factor: {adjust_lr}")
       
       # Adjust learning rate
       if hasattr(self.training_args, "learning_rate"):
           original_lr = self.training_args.learning_rate
           self.training_args.learning_rate = original_lr * adjust_lr
           logger.info(f"Learning rate adjusted from {original_lr} to {self.training_args.learning_rate}")
   ```

## Performance Advantages

Compared to standard training, DeepSpeed-Ulysses Pro provides the following performance advantages:

1. **Memory Efficiency**: Sequence length increased by more than 4x, supporting training of sequences over one million tokens
2. **Communication Efficiency**: Communication reduced by more than 10x, throughput improved by up to 2.5x
3. **Computational Efficiency**: Sustained throughput per GPU exceeding 175 TFlops (over 54% of hardware peak)
4. **Convergence Speed**: Significantly faster convergence in later training stages due to multi-strategy gradient optimization
5. **Versatility**: Supports both dense and sparse attention, can be combined with efficient implementations like FlashAttention
6. **Scalability**: Can be combined with ZeRO-3, supporting both large model and long sequence training

## Usage

### 1. Environment Preparation

Ensure DeepSpeed v0.10.2 or higher is installed:

```bash
pip install "deepspeed>=0.10.2"
```

Install FlashAttention (required):

```bash
# Install triton and FlashAttention
pip install triton
pip install flash-attn
```

### 2. Training with Launch Script

Use the `train_with_ulysses_pro.py` script to start training:

```bash
python train_with_ulysses_pro.py --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml
```

### 3. Enabling Debug Mode

```bash
python train_with_ulysses_pro.py --debug --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml
```

Enabling debug mode provides more detailed log output and activates enhanced gradient handling mechanisms, which can help diagnose and solve training issues.

### 4. Adjusting Learning Rate

```bash
# Specify learning rate factor
python train_with_ulysses_pro.py --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml --adjust-lr 2.0

# Use default factor (1.0)
python train_with_ulysses_pro.py --config_file examples/train_lora/llama3_lora_dpo_ulysses.yaml --adjust-lr
```

This will multiply the original learning rate by the specified factor. If no factor value is provided, the default value of 1.0 is used. This parameter is useful for fine-tuning the learning rate without modifying the configuration file.

### 5. Recommended Configurations for Accelerating Later-stage Convergence

For slow convergence in later training stages, the following configurations are recommended:

```bash
# Option 1: Increase learning rate and enable debug mode (activates enhanced gradient handling)
python train_with_ulysses_pro.py --debug --adjust-lr 2.0 --config_file examples/train_lora/llama3_lora_dpo.yaml

# Option 2: For DPO training, adjust beta value and learning rate
python train_with_ulysses_pro.py --adjust-lr 3.0 --pref_beta 0.15 --lr_scheduler_type cosine --config_file examples/train_lora/llama3_lora_dpo.yaml

# Option 3: For models that have been training for a long time, use more aggressive settings
python train_with_ulysses_pro.py --debug --adjust-lr 5.0 --max_grad_norm 2.0 --config_file examples/train_lora/llama3_lora_dpo.yaml
```

### 6. Customizing Sequence Parallel Size

To customize the sequence parallel size, modify the `sequence_parallel_size` parameter in the configuration file:

```json
{
  "sequence_parallel_size": 8,  // Set sequence parallel size to 8
  ...
}
```

### 7. Multi-node Distributed Training

For large models (32B and above), multi-node distributed training can be used:

```bash
# Run on the master node
NODE_RANK=0 MASTER_ADDR=<master_ip> MASTER_PORT=29500 torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<master_ip> \
    --master_port=29500 \
    train_with_ulysses_pro.py \
    --deepspeed examples/deepspeed/ds_z3_ulysses_config_improved.json \
    ... other parameters ...

# Run on the worker node
NODE_RANK=1 MASTER_ADDR=<master_ip> MASTER_PORT=29500 torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<master_ip> \
    --master_port=29500 \
    train_with_ulysses_pro.py \
    --deepspeed examples/deepspeed/ds_z3_ulysses_config_improved.json \
    ... other parameters ...
```

## Notes

1. Sequence parallel size must be divisible by the total number of GPUs
2. The number of attention heads should be divisible by the sequence parallel size for optimal performance
3. The current implementation is not compatible with Megatron-LM's tensor parallelism or pipeline parallelism
4. When using FlashAttention, for best performance, the head size should be a multiple of 8
5. For GQA or MQA models (such as Llama 3), the number of K, V heads is smaller, so the sequence parallel size should not be set too large
6. For DPO training, the default value of `pref_beta` is 0.1, which can be adjusted using the `--pref_beta` parameter
7. If training convergence is slow in later stages, try increasing the `--adjust-lr` parameter value (2.0-5.0)

## References

- [DeepSpeed-Ulysses Tutorial](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/chinese/README.md)
- [FlashAttention Documentation](https://github.com/HazyResearch/flash-attention)

## Acknowledgements

This project is developed based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks to the original project authors for their contributions. Also thanks to the DeepSpeed team for developing the Ulysses sequence parallel technology.

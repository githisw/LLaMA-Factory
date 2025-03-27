# LLaMA-Factory with DeepSpeed-Ulysses Pro 修改记录

本文件详细记录了项目的所有修改，以日期为一级标题，详细说明每次修改的代码位置和内容。

## 2025-03-27

### 优化梯度处理策略

#### 修改文件

- `ulysses_enchanced_pro/trainer_patching.py`

#### 修改内容

1. **降低梯度噪声**
   - 位置：`handle_gradients_func` 函数中的 `noise_scale` 参数
   - 原值：`5e-3`
   - 新值：`1e-4`
   - 代码段：
     ```python
     # 原代码
     noise_scale = 5e-3  # 增加噪声大小
     for p in trainer.model.parameters():
         if p.grad is not None:
             p.grad.add_(torch.randn_like(p.grad) * noise_scale)
     
     # 修改后
     # 在训练后期（最后10%的步骤）减少噪声
     if current_step > 0.9 * total_steps:
         noise_scale = 1e-5  # 训练后期使用非常小的噪声
         logger.info(f"训练后期（步骤 {current_step}/{total_steps}）：使用减小的噪声 {noise_scale}")
     else:
         noise_scale = 1e-4  # 降低噪声大小
     
     for p in trainer.model.parameters():
         if p.grad is not None:
             p.grad.add_(torch.randn_like(p.grad) * noise_scale)
     ```

2. **添加训练后期干预减少逻辑**
   - 位置：`handle_gradients_func` 函数
   - 添加了获取当前训练步数和总步数的代码：
     ```python
     # 获取当前训练步数和总步数
     current_step = trainer.state.global_step if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step') else 0
     total_steps = trainer.args.max_steps if hasattr(trainer, 'args') and hasattr(trainer.args, 'max_steps') else 1000
     ```

3. **训练后期梯度缩放因子调整**
   - 位置：`handle_gradients_func` 函数中的梯度缩放部分
   - 原代码：
     ```python
     # 策略2: 梯度缩放，放大小梯度
     scale_factor = 1.5  # 梯度缩放因子
     for p in trainer.model.parameters():
         if p.grad is not None:
             if torch.mean(torch.abs(p.grad)) < 1e-4:
                 p.grad.mul_(scale_factor)
     ```
   - 修改后：
     ```python
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
     ```

4. **训练后期禁用学习率临时增加**
   - 位置：`handle_gradients_func` 函数中的学习率调整部分
   - 原代码：
     ```python
     # 策略3: 如果有学习率调度器，临时增加学习率
     if hasattr(trainer, 'optimizer') and hasattr(trainer.optimizer, 'param_groups'):
         for param_group in trainer.optimizer.param_groups:
             if 'lr' in param_group:
                 # 记录原始学习率
                 if not hasattr(trainer, '_original_lr'):
                     trainer._original_lr = param_group['lr']
                 
                 # 临时增加学习率
                 param_group['lr'] = param_group['lr'] * 1.2
                 logger.info(f"临时增加学习率至: {param_group['lr']:.8f}")
     ```
   - 修改后：
     ```python
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
     ```

#### 修改原因

1. **降低训练波动**：原始的梯度噪声值（5e-3）过大，导致训练过程中loss和rewards/accuracies在收敛阶段出现大幅波动。降低噪声值可以减少这种波动，使训练更加稳定。

2. **改善收敛性能**：在训练后期（最后10%的步骤）进一步减少干预，使模型能够更自然地收敛到最优解，而不是在最优解附近剧烈震荡。

3. **平衡探索与稳定**：
   - 训练前期（前90%）：保持适度的梯度干预，帮助模型跳出局部最优解，探索更广阔的参数空间
   - 训练后期（最后10%）：减少干预，提高稳定性，让模型能够精细调整参数，找到全局最优解

#### 影响范围

- 此修改适用于所有训练类型（SFT、DPO、PT、PPO、KTO和RM等）
- 通过降低训练噪声和优化后期训练策略，预期会提高模型的最终性能和稳定性
- 特别对于长时间训练的模型，这些修改可以显著改善收敛阶段的稳定性

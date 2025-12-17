# Guide: Pruning ImageNet ResNet Models

This guide explains how to prune pretrained ImageNet ResNet models (from torchvision or timm) using network slimming with BatchNorm scale factors.

## Overview

The workflow:
1. **Transfer weights** from torchvision/timm ResNet to prunable architecture
2. **Fine-tune with sparsity** regularization (optional but recommended)
3. **Prune** the model based on BN scale factors
4. **Fine-tune** the pruned model to recover accuracy

## Step 1: Transfer Weights

Transfer weights from a pretrained torchvision/timm ResNet to the prunable architecture:

```bash
python transfer_imagenet.py \
    --source torchvision \
    --model-name resnet50 \
    --num-classes 1000 \
    --output transferred_resnet50.pth.tar
```

**Arguments:**
- `--source`: `torchvision` or `timm`
- `--model-name`: `resnet50`, `resnet101`, or `resnet152`
- `--num-classes`: Number of classes (1000 for ImageNet)
- `--output`: Output path for transferred model

## Step 2: Fine-tune with Sparsity Regularization (Recommended)

Fine-tune the transferred model with sparsity regularization to encourage sparse BN weights:

```bash
python train_imagenet.py \
    -sr \
    --s 0.0001 \
    --data /path/to/imagenet \
    --arch resnet50 \
    --num-classes 1000 \
    --resume transferred_resnet50.pth.tar \
    --epochs 30 \
    --lr 0.01 \
    --batch-size 256 \
    --save ./sparsity_trained
```

**Arguments:**
- `-sr`: Enable sparsity regularization
- `--s`: Sparsity strength (0.0001 is a good starting point)
- `--data`: Path to ImageNet dataset
- `--resume`: Path to transferred model
- `--epochs`: Number of fine-tuning epochs (20-30 usually sufficient)
- `--lr`: Learning rate (use lower LR like 0.01 for fine-tuning)

## Step 3: Prune the Model

Prune the model based on BatchNorm scale factors:

```bash
python prune_imagenet.py \
    --arch resnet50 \
    --num-classes 1000 \
    --percent 0.5 \
    --model ./sparsity_trained/model_best.pth.tar \
    --data /path/to/imagenet \
    --save ./pruned_model
```

**Arguments:**
- `--arch`: Architecture name
- `--percent`: Percentage of channels to prune (0.0-1.0)
  - `0.3` (30%) - Conservative
  - `0.5` (50%) - Moderate (recommended)
  - `0.7` (70%) - Aggressive
- `--model`: Path to sparsity-trained model
- `--data`: Path to ImageNet (for testing, optional)
- `--save`: Directory to save pruned model

**Output:**
- `pruned.pth.tar`: Pruned model with `cfg` and `state_dict`
- `prune.txt`: Configuration and statistics

## Step 4: Fine-tune the Pruned Model

Fine-tune the pruned model to recover accuracy:

```bash
python train_imagenet.py \
    --refine ./pruned_model/pruned.pth.tar \
    --data /path/to/imagenet \
    --arch resnet50 \
    --num-classes 1000 \
    --epochs 90 \
    --lr 0.1 \
    --batch-size 256 \
    --save ./final_model
```

**Arguments:**
- `--refine`: Path to pruned model
- `--epochs`: Number of epochs (90 is standard for ImageNet)
- `--lr`: Learning rate (0.1 is standard, will decay at epochs 30, 60)

## Complete Example Workflow

```bash
# 1. Transfer weights from torchvision ResNet-50
python transfer_imagenet.py \
    --source torchvision \
    --model-name resnet50 \
    --output transferred_resnet50.pth.tar

# 2. Fine-tune with sparsity (30 epochs)
python train_imagenet.py \
    -sr --s 0.0001 \
    --data /path/to/imagenet \
    --arch resnet50 \
    --resume transferred_resnet50.pth.tar \
    --epochs 30 \
    --lr 0.01 \
    --save ./sparsity_trained

# 3. Prune 50% of channels
python prune_imagenet.py \
    --arch resnet50 \
    --percent 0.5 \
    --model ./sparsity_trained/model_best.pth.tar \
    --save ./pruned_model

# 4. Fine-tune pruned model (90 epochs)
python train_imagenet.py \
    --refine ./pruned_model/pruned.pth.tar \
    --data /path/to/imagenet \
    --arch resnet50 \
    --epochs 90 \
    --save ./final_model
```

## Quick Pruning (Skip Sparsity Training)

If you want to prune immediately without sparsity training:

```bash
# 1. Transfer weights
python transfer_imagenet.py \
    --source torchvision \
    --model-name resnet50 \
    --output transferred_resnet50.pth.tar

# 2. Prune directly
python prune_imagenet.py \
    --arch resnet50 \
    --percent 0.5 \
    --model transferred_resnet50.pth.tar \
    --save ./pruned_model

# 3. Fine-tune
python train_imagenet.py \
    --refine ./pruned_model/pruned.pth.tar \
    --data /path/to/imagenet \
    --arch resnet50 \
    --epochs 90 \
    --save ./final_model
```

**Note:** Direct pruning may result in larger accuracy drop. Sparsity training is recommended.

## Pruning Percentages

Recommended pruning percentages for ImageNet ResNet:

| Architecture | Pruning % | Expected Accuracy Drop | Recovery After Fine-tuning |
|-------------|-----------|------------------------|---------------------------|
| ResNet-50   | 30%       | ~2-3%                  | Usually recovers          |
| ResNet-50   | 50%       | ~5-7%                  | Most recovers             |
| ResNet-50   | 70%       | ~10-15%                | May not fully recover     |
| ResNet-101  | 30%       | ~1-2%                  | Usually recovers          |
| ResNet-101  | 50%       | ~4-6%                  | Most recovers             |

Start with 30-50% pruning for best results.

## Understanding the Output

### Pruning Output

The pruning script prints:
- Layer-by-layer pruning statistics
- Total channels pruned
- Test accuracy after pruning (if ImageNet data provided)

### Saved Files

**`pruned.pth.tar`** contains:
- `cfg`: Channel configuration (list of channel counts per layer)
- `state_dict`: Pruned model weights

**`prune.txt`** contains:
- Configuration (channel counts)
- Number of parameters
- Test accuracy

## Tips

1. **Sparsity Training**: Always fine-tune with sparsity regularization before pruning for best results
2. **Gradual Pruning**: For aggressive pruning (70%+), consider iterative pruning:
   - Prune 30% → fine-tune → prune 30% more → fine-tune
3. **Learning Rate**: Use lower learning rates for fine-tuning (0.01) and standard rates for full training (0.1)
4. **Batch Size**: Adjust batch size based on GPU memory (256 is standard)
5. **Data Path**: Ensure ImageNet data is organized as:
   ```
   imagenet/
     train/
       n01440764/
       n01443537/
       ...
     val/
       n01440764/
       n01443537/
       ...
   ```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (e.g., 128 or 64)
- Use gradient accumulation if needed

### Low Accuracy After Pruning

- Use lower pruning percentage (30% instead of 50%)
- Ensure sparsity training was performed
- Fine-tune for more epochs

### Model Architecture Mismatch

- Ensure `--arch` matches the model you're pruning
- Check that the model was created with the same architecture

## Expected Results

For ResNet-50 on ImageNet with 50% pruning:

- **Before pruning**: ~76% top-1 accuracy, ~25M parameters
- **After pruning**: ~70% top-1 accuracy, ~12.5M parameters
- **After fine-tuning**: ~74-75% top-1 accuracy, ~12.5M parameters

The fine-tuning step is crucial for recovering accuracy!


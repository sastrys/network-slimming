# Guide: Pruning a Pretrained ResNet with Network Slimming

This guide explains how to prune a pretrained ResNet model using the network slimming technique.

## Important Note

**Network slimming works best when the model is trained with sparsity regularization.** If your pretrained model was NOT trained with sparsity regularization (`-sr` flag), the BatchNorm scaling factors may not be sparse, making pruning less effective.

## Option 1: Direct Pruning (Quick but may lose accuracy)

If you want to prune immediately without retraining:

```bash
python resprune.py \
    --dataset cifar10 \
    --depth 164 \
    --percent 0.4 \
    --model /path/to/your/pretrained_model.pth.tar \
    --save ./pruned_models
```

**Arguments:**
- `--dataset`: `cifar10` or `cifar100` (must match your pretrained model)
- `--depth`: ResNet depth (e.g., `164` for ResNet-164)
- `--percent`: Percentage of channels to prune (0.0-1.0). For ResNet, recommended:
  - `0.4` (40% pruning) - conservative
  - `0.5` (50% pruning) - moderate
  - `0.6` (60% pruning) - aggressive (may cause errors in some cases)
- `--model`: Path to your pretrained model checkpoint
- `--save`: Directory to save the pruned model

**Output:**
- `pruned.pth.tar`: The pruned model
- `prune.txt`: Configuration and statistics

**Warning:** Direct pruning without sparsity training may result in significant accuracy drop because the BatchNorm weights may not be sparse.

## Option 2: Fine-tune with Sparsity First (Recommended)

This is the recommended approach for best results:

### Step 1: Fine-tune with Sparsity Regularization

```bash
python main.py \
    -sr \
    --s 0.00001 \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --resume /path/to/your/pretrained_model.pth.tar \
    --epochs 40 \
    --lr 0.001 \
    --save ./sparsity_training
```

**Arguments:**
- `-sr` or `--sparsity-regularization`: Enable sparsity training
- `--s`: Sparsity regularization strength (for ResNet, use `0.00001`)
- `--resume`: Path to your pretrained model (loads weights and continues training)
- `--epochs`: Number of epochs for sparsity fine-tuning (20-40 epochs usually sufficient)
- `--lr`: Learning rate (use lower LR like 0.001 for fine-tuning)

This will:
1. Load your pretrained model
2. Apply L1 sparsity regularization to BatchNorm weights
3. Train for additional epochs to encourage sparsity
4. Save checkpoints in `./sparsity_training`

### Step 2: Prune the Sparsity-Trained Model

```bash
python resprune.py \
    --dataset cifar10 \
    --depth 164 \
    --percent 0.4 \
    --model ./sparsity_training/model_best.pth.tar \
    --save ./pruned_models
```

### Step 3: Fine-tune the Pruned Model

```bash
python main.py \
    --refine ./pruned_models/pruned.pth.tar \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --lr 0.1 \
    --save ./final_model
```

**Arguments:**
- `--refine`: Path to the pruned model
- `--epochs`: Number of fine-tuning epochs (160 is standard)
- `--lr`: Learning rate (0.1 is standard, will decay at 50% and 75% of epochs)

## Option 3: Train from Scratch with Sparsity (Best Results)

If you have time and want the best results:

### Step 1: Train with Sparsity from the Start

```bash
python main.py \
    -sr \
    --s 0.00001 \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --save ./sparsity_training
```

### Step 2: Prune

```bash
python resprune.py \
    --dataset cifar10 \
    --depth 164 \
    --percent 0.4 \
    --model ./sparsity_training/model_best.pth.tar \
    --save ./pruned_models
```

### Step 3: Fine-tune

```bash
python main.py \
    --refine ./pruned_models/pruned.pth.tar \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --save ./final_model
```

## Complete Example Workflow

Here's a complete example for pruning a CIFAR-10 ResNet-164:

```bash
# 1. Fine-tune pretrained model with sparsity (40 epochs)
python main.py \
    -sr \
    --s 0.00001 \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --resume ./pretrained/resnet164_cifar10.pth.tar \
    --epochs 40 \
    --lr 0.001 \
    --save ./sparsity_training

# 2. Prune 40% of channels
python resprune.py \
    --dataset cifar10 \
    --depth 164 \
    --percent 0.4 \
    --model ./sparsity_training/model_best.pth.tar \
    --save ./pruned_models

# 3. Fine-tune pruned model (160 epochs)
python main.py \
    --refine ./pruned_models/pruned.pth.tar \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --save ./final_model
```

## Understanding the Pruning Process

### What Happens During Pruning:

1. **Load Model**: Loads your pretrained model
2. **Collect BN Weights**: Gathers all BatchNorm scaling factors across the network
3. **Calculate Threshold**: Sorts all BN weights and finds the threshold for your pruning percentage
4. **Create Masks**: Creates binary masks for each BN layer (1 = keep, 0 = prune)
5. **Build New Architecture**: Creates a new model with reduced channels based on masks
6. **Copy Weights**: Selectively copies weights from old to new model
7. **Test**: Evaluates the pruned model (accuracy will drop initially)

### Output Files:

- `pruned.pth.tar`: Contains:
  - `cfg`: Channel configuration (list of channel counts per layer)
  - `state_dict`: Pruned model weights
- `prune.txt`: Text file with:
  - Configuration (channel counts)
  - Number of parameters
  - Test accuracy after pruning

## Recommended Pruning Percentages

Based on the README results:

| Architecture | Dataset | Recommended % | Notes |
|-------------|---------|---------------|-------|
| ResNet-164  | CIFAR-10 | 40% | Safe, good accuracy recovery |
| ResNet-164  | CIFAR-10 | 60% | Aggressive, may cause errors |
| ResNet-164  | CIFAR-100 | 40% | Safe |
| ResNet-164  | CIFAR-100 | 60% | May cause errors (use mask-impl) |

## Troubleshooting

### Error: "Some layers are all pruned"
- **Cause**: Too aggressive pruning (e.g., 60% on CIFAR-100)
- **Solution**: 
  - Use lower pruning percentage (40%)
  - Or use mask-based implementation: `mask-impl/prune_mask.py`

### Low Accuracy After Pruning
- **Cause**: Model wasn't trained with sparsity regularization
- **Solution**: Use Option 2 (fine-tune with sparsity first)

### Model Architecture Mismatch
- **Cause**: Wrong `--depth` or `--dataset` argument
- **Solution**: Ensure arguments match your pretrained model

## Using Mask-Based Implementation

If you encounter issues with the standard pruning, try the mask-based approach:

```bash
# Training with sparsity
python mask-impl/main_mask.py \
    -sr \
    --s 0.00001 \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --resume /path/to/pretrained_model.pth.tar \
    --epochs 40 \
    --save ./mask_training

# Pruning
python mask-impl/prune_mask.py \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --percent 0.4 \
    --model ./mask_training/model_best.pth.tar \
    --save ./mask_pruned

# Fine-tuning
python mask-impl/main_mask.py \
    --refine ./mask_pruned/pruned.pth.tar \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --save ./mask_final
```

## Expected Results

For ResNet-164 on CIFAR-10 with 40% pruning:
- **Before pruning**: ~94.75% accuracy, 1.71M parameters
- **After pruning**: ~94.58% accuracy, 1.45M parameters
- **After fine-tuning**: ~95.05% accuracy, 1.45M parameters

The fine-tuning step is crucial for recovering accuracy!


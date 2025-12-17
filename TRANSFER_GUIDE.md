# Guide: Transferring Weights from torchvision/timm ResNet to Prunable Architecture

This guide explains how to transfer weights from a pretrained ResNet (from torchvision or timm) to the prunable ResNet architecture in this repository.

## Important Limitations

⚠️ **Architecture Mismatch**: The prunable ResNet in this repo is designed for **CIFAR** (32x32 images), while torchvision/timm ResNets are designed for **ImageNet** (224x224 images). This creates several incompatibilities:

1. **Different conv1**: 
   - torchvision: 7x7 kernel, stride=2 (for 224x224 → 112x112)
   - prunable: 3x3 kernel, padding=1 (for 32x32 → 32x32)

2. **Different layer structure**:
   - torchvision: Standard ResNet with post-activation
   - prunable: Pre-activation ResNet with channel selection layers

3. **Different number of classes**:
   - torchvision: 1000 classes (ImageNet)
   - prunable: 10 or 100 classes (CIFAR)

4. **Different depth**:
   - torchvision: ResNet-18, 34, 50, 101, 152
   - prunable: ResNet-164 (9n+2 structure, e.g., n=18 → depth=164)

## What Can Be Transferred

✅ **Can be transferred:**
- Most BatchNorm weights (bn1, bn2, bn3) from bottleneck blocks
- Most convolution weights (conv1, conv2, conv3) from bottleneck blocks
- Downsample convolution weights

❌ **Cannot be transferred:**
- conv1 layer (different kernel size and stride)
- Final fc layer (different number of classes)
- Channel selection layers (don't exist in torchvision models - will be initialized to all 1s)

## Usage

### Basic Usage

```bash
python transfer_weights.py \
    --source torchvision \
    --model-name resnet50 \
    --dataset cifar10 \
    --depth 164 \
    --output transferred_resnet50.pth.tar
```

### Arguments

- `--source`: `torchvision` or `timm` (source of pretrained model)
- `--model-name`: Model name (e.g., `resnet50`, `resnet101`)
- `--dataset`: Target dataset (`cifar10` or `cifar100`)
- `--depth`: Depth of prunable ResNet (must be 9n+2, default: 164)
- `--output`: Output path for transferred model
- `--no-pretrained`: Load model without pretrained weights (for testing)

### Example: Transfer ResNet-50 from torchvision

```bash
python transfer_weights.py \
    --source torchvision \
    --model-name resnet50 \
    --dataset cifar10 \
    --depth 164 \
    --output transferred_resnet50_cifar10.pth.tar
```

### Example: Transfer ResNet-101 from timm

```bash
python transfer_weights.py \
    --source timm \
    --model-name resnet101 \
    --dataset cifar100 \
    --depth 164 \
    --output transferred_resnet101_cifar100.pth.tar
```

## Complete Workflow

### Step 1: Transfer Weights

```bash
python transfer_weights.py \
    --source torchvision \
    --model-name resnet50 \
    --dataset cifar10 \
    --depth 164 \
    --output transferred_model.pth.tar
```

### Step 2: Fine-tune on CIFAR (Required!)

Since many layers couldn't be transferred, you **must** fine-tune the model:

```bash
python main.py \
    --resume transferred_model.pth.tar \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --lr 0.1 \
    --save ./fine_tuned_model
```

### Step 3: Fine-tune with Sparsity Regularization

```bash
python main.py \
    -sr \
    --s 0.00001 \
    --resume ./fine_tuned_model/model_best.pth.tar \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 40 \
    --lr 0.001 \
    --save ./sparsity_trained_model
```

### Step 4: Prune

```bash
python resprune.py \
    --dataset cifar10 \
    --depth 164 \
    --percent 0.4 \
    --model ./sparsity_trained_model/model_best.pth.tar \
    --save ./pruned_model
```

### Step 5: Fine-tune Pruned Model

```bash
python main.py \
    --refine ./pruned_model/pruned.pth.tar \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --save ./final_model
```

## Understanding the Output

The script will print detailed information about what was transferred:

```
Mapping Summary:
  Mapped: 150 keys
  Skipped: 5 keys
  Unmapped: 10 keys
```

- **Mapped**: Successfully transferred layers
- **Skipped**: Layers that exist in both but shapes don't match
- **Unmapped**: Layers in prunable model that don't exist in source (will use random initialization)

## Alternative: Train from Scratch

If the architecture mismatch is too significant, consider training from scratch with sparsity:

```bash
# Train with sparsity from the start
python main.py \
    -sr \
    --s 0.00001 \
    --dataset cifar10 \
    --arch resnet \
    --depth 164 \
    --epochs 160 \
    --save ./sparsity_trained

# Then prune and fine-tune as usual
```

This often gives better results than transferring weights from ImageNet models.

## Troubleshooting

### Error: "Shapes don't match"

This is expected for:
- `conv1.weight`: Different kernel sizes (7x7 vs 3x3)
- `fc.weight`: Different number of classes (1000 vs 10/100)

These layers will use random initialization and need to be trained.

### Low Accuracy After Transfer

This is normal! The transferred model needs fine-tuning because:
1. Many layers couldn't be transferred
2. The model was trained on ImageNet, not CIFAR
3. Architecture differences require adaptation

**Solution**: Fine-tune for at least 160 epochs before pruning.

### Channel Selection Layers

Channel selection layers (`select.indexes`) don't exist in torchvision models. They will be initialized to all 1s (all channels active), which is correct for the initial state.

## Expected Results

After complete workflow (transfer → fine-tune → sparsity → prune → fine-tune):

- **Accuracy**: Should recover to ~94-95% on CIFAR-10 (similar to training from scratch)
- **Parameters**: Reduced by pruning percentage (e.g., 40% pruning → 60% of original parameters)
- **Speed**: Faster inference due to fewer channels

## Notes

1. **Transfer is partial**: Not all weights can be transferred due to architecture differences
2. **Fine-tuning is mandatory**: The model must be fine-tuned on CIFAR before pruning
3. **Sparsity training is recommended**: Fine-tune with sparsity regularization before pruning for best results
4. **Training from scratch may be better**: If you have time, training from scratch with sparsity often gives better results than transferring ImageNet weights


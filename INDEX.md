# Network Slimming Repository Index

This document provides a comprehensive index of the Network Slimming repository, a PyTorch implementation of channel pruning for deep neural networks.

## Repository Overview

This repository implements **Network Slimming** (ICCV 2017), a method for learning efficient convolutional networks through channel pruning. The technique uses BatchNorm scaling factors to identify and prune less important channels.

## Directory Structure

```
network-slimming/
├── main.py                 # Main training script
├── vggprune.py            # VGG network pruning script
├── resprune.py            # ResNet network pruning script
├── denseprune.py          # DenseNet network pruning script
├── models/                # Model architectures (with channel selection)
│   ├── __init__.py
│   ├── channel_selection.py  # Channel selection layer implementation
│   ├── vgg.py             # VGG architecture
│   ├── preresnet.py       # Pre-activation ResNet architecture
│   └── densenet.py        # DenseNet architecture
├── mask-impl/            # Alternative mask-based implementation
│   ├── main_mask.py       # Training with mask implementation
│   ├── prune_mask.py      # Pruning with mask implementation
│   └── models/            # Model architectures for mask implementation
│       ├── __init__.py
│       ├── vgg.py
│       ├── preresnet.py
│       └── densenet.py
└── README.md              # Project documentation
```

## Core Components

### 1. Training Scripts

#### `main.py`
- **Purpose**: Main training script for baseline and sparsity-regularized training
- **Key Features**:
  - Supports CIFAR-10 and CIFAR-100 datasets
  - Supports VGG, ResNet, and DenseNet architectures
  - Implements sparsity regularization via L1 penalty on BN weights
  - Supports fine-tuning pruned models
- **Key Functions**:
  - `updateBN()`: Adds L1 sparsity penalty to BatchNorm weight gradients
  - `train()`: Training loop with optional sparsity regularization
  - `test()`: Evaluation function
- **Usage**: See README.md for command-line examples

#### `mask-impl/main_mask.py`
- **Purpose**: Alternative training implementation using mask-based approach
- **Key Differences**:
  - Uses `BN_grad_zero()` to zero gradients of pruned channels
  - No channel selection layer needed
  - Handles fully pruned layers gracefully
- **Key Functions**:
  - `updateBN()`: L1 sparsity penalty (same as main.py)
  - `BN_grad_zero()`: Zeros gradients for channels with zero scaling factors

### 2. Pruning Scripts

#### `vggprune.py`
- **Purpose**: Prunes VGG networks
- **Process**:
  1. Loads trained model
  2. Collects all BN scaling factors
  3. Determines global threshold based on pruning percentage
  4. Creates masks for each BN layer
  5. Builds new pruned model architecture
  6. Copies weights from old to new model
- **Output**: `pruned.pth.tar` containing pruned model state and configuration

#### `resprune.py`
- **Purpose**: Prunes ResNet networks
- **Special Handling**:
  - Handles channel selection layers in residual blocks
  - Manages skip connections and downsampling convolutions
  - Tracks convolution count for proper weight copying

#### `denseprune.py`
- **Purpose**: Prunes DenseNet networks
- **Special Handling**:
  - Manages dense connections (channel concatenation)
  - Handles transition layers
  - Properly handles channel selection in dense blocks

#### `mask-impl/prune_mask.py`
- **Purpose**: Mask-based pruning (simpler approach)
- **Process**:
  - Sets BN scaling factors to zero for pruned channels
  - Saves model with zeroed scaling factors
  - No architecture reconstruction needed

### 3. Model Architectures

#### `models/channel_selection.py`
- **Class**: `channel_selection`
- **Purpose**: Selects active channels from BatchNorm output
- **Key Attributes**:
  - `indexes`: Parameter tensor (1 = keep channel, 0 = prune)
- **Usage**: Placed after BatchNorm layers in ResNet and DenseNet

#### `models/vgg.py`
- **Class**: `vgg`
- **Architectures**: VGG-11, VGG-13, VGG-16, VGG-19
- **Features**:
  - Configurable depth
  - Supports CIFAR-10/100
  - BatchNorm after each convolution
  - No channel selection layer (not needed for VGG)

#### `models/preresnet.py`
- **Class**: `resnet`
- **Architecture**: Pre-activation ResNet with bottleneck blocks
- **Features**:
  - Configurable depth (e.g., ResNet-164)
  - Uses channel selection layers
  - Handles skip connections properly
- **Key Components**:
  - `Bottleneck`: Residual block with channel selection
  - `_make_layer()`: Constructs ResNet layers

#### `models/densenet.py`
- **Class**: `densenet`
- **Architecture**: DenseNet with basic blocks
- **Features**:
  - Configurable depth (e.g., DenseNet-40)
  - Growth rate: 12
  - Uses channel selection in dense blocks and transitions
- **Key Components**:
  - `BasicBlock`: Dense block with channel selection
  - `Transition`: Transition layer with channel selection

### 4. Mask Implementation Models

The `mask-impl/models/` directory contains similar model architectures but without channel selection layers. These models use mask-based pruning where BN scaling factors are set to zero instead of physically removing channels.

## Workflow

### Standard Workflow (Channel Selection)

1. **Baseline Training**:
   ```bash
   python main.py --dataset cifar10 --arch vgg --depth 19
   ```

2. **Sparsity Training**:
   ```bash
   python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19
   ```

3. **Pruning**:
   ```bash
   python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 \
     --model [PATH] --save [DIRECTORY]
   ```

4. **Fine-tuning**:
   ```bash
   python main.py --refine [PRUNED_MODEL] --dataset cifar10 \
     --arch vgg --depth 19 --epochs 160
   ```

### Mask-Based Workflow

1. **Baseline Training**:
   ```bash
   python mask-impl/main_mask.py --dataset cifar100 --arch resnet --depth 164
   ```

2. **Sparsity Training**:
   ```bash
   python mask-impl/main_mask.py -sr --s 0.00001 --dataset cifar100 \
     --arch resnet --depth 164
   ```

3. **Pruning**:
   ```bash
   python mask-impl/prune_mask.py --dataset cifar100 --arch resnet \
     --depth 164 --percent 0.4 --model [PATH] --save [DIRECTORY]
   ```

4. **Fine-tuning**:
   ```bash
   python mask-impl/main_mask.py --refine [PRUNED_MODEL] \
     --dataset cifar100 --arch resnet --depth 164
   ```

## Key Concepts

### Channel Selection Layer
- Introduced for ResNet and DenseNet pruning
- Stores binary mask (`indexes`) indicating which channels to keep
- Placed after BatchNorm layers
- Reduces training overhead compared to mask-based approach

### Sparsity Regularization
- L1 penalty on BatchNorm scaling factors: `grad += s * sign(weight)`
- Encourages many scaling factors to approach zero
- Applied during training via `updateBN()` function

### Pruning Process
1. **Threshold Calculation**: Global threshold based on percentage of channels to prune
2. **Mask Creation**: Binary masks for each BN layer
3. **Architecture Reconstruction**: Build new model with reduced channels
4. **Weight Transfer**: Copy weights from old to new model, handling channel indices

### Mask-Based Approach
- Simpler implementation
- Sets BN scaling factors to zero (no architecture change)
- Zeroes gradients during training for pruned channels
- Handles fully pruned layers (outputs all zeros)

## File Dependencies

```
main.py
  └── models/ (vgg, preresnet, densenet, channel_selection)

vggprune.py
  └── models/ (vgg)

resprune.py
  └── models/ (preresnet, channel_selection)

denseprune.py
  └── models/ (densenet, channel_selection)

mask-impl/main_mask.py
  └── mask-impl/models/ (vgg, preresnet, densenet)

mask-impl/prune_mask.py
  └── mask-impl/models/ (vgg, preresnet, densenet)
```

## Configuration Files

Models save/load checkpoints containing:
- `epoch`: Training epoch
- `state_dict`: Model weights
- `best_prec1`: Best accuracy achieved
- `optimizer`: Optimizer state
- `cfg`: Channel configuration (for pruned models)

## Supported Architectures

| Architecture | Depths | Datasets | Pruning Script |
|-------------|--------|----------|----------------|
| VGG | 11, 13, 16, 19 | CIFAR-10, CIFAR-100 | `vggprune.py` |
| ResNet | 164 (9n+2) | CIFAR-10, CIFAR-100 | `resprune.py` |
| DenseNet | 40 (3n+4) | CIFAR-10, CIFAR-100 | `denseprune.py` |

## Notes

- **PyTorch Version**: Requires torch v0.3.1, torchvision v0.2.0 (legacy)
- **Channel Selection**: Used for ResNet and DenseNet, not needed for VGG
- **Mask Implementation**: Alternative approach that avoids architecture reconstruction
- **Pruning Percentages**: Varies by architecture (see README.md for recommended values)

## Quick Reference

### Training Arguments
- `--dataset`: `cifar10` or `cifar100`
- `--arch`: `vgg`, `resnet`, or `densenet`
- `--depth`: Network depth (19, 164, 40, etc.)
- `-sr` or `--sparsity-regularization`: Enable sparsity training
- `--s`: Sparsity regularization strength
- `--refine`: Path to pruned model for fine-tuning

### Pruning Arguments
- `--percent`: Percentage of channels to prune (0.0-1.0)
- `--model`: Path to trained model checkpoint
- `--save`: Directory to save pruned model


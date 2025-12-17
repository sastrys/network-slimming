"""
Script to transfer weights from torchvision/timm ResNet to prunable ResNet architecture.

This script handles:
1. Loading pretrained ResNet from torchvision or timm
2. Creating prunable ResNet architecture (with channel selection)
3. Mapping and transferring compatible weights
4. Handling architecture differences (ImageNet vs CIFAR, layer names, etc.)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from models import resnet as prunable_resnet
import argparse


def load_torchvision_resnet(model_name='resnet50', pretrained=True):
    """
    Load ResNet from torchvision.
    
    Args:
        model_name: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        pretrained: Whether to load pretrained weights
    
    Returns:
        torchvision ResNet model
    """
    model_dict = {
        'resnet18': tv_models.resnet18,
        'resnet34': tv_models.resnet34,
        'resnet50': tv_models.resnet50,
        'resnet101': tv_models.resnet101,
        'resnet152': tv_models.resnet152,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")
    
    model = model_dict[model_name](pretrained=pretrained)
    return model


def load_timm_resnet(model_name='resnet50', pretrained=True):
    """
    Load ResNet from timm.
    
    Args:
        model_name: Any timm ResNet model name
        pretrained: Whether to load pretrained weights
    
    Returns:
        timm ResNet model
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is not installed. Install with: pip install timm")
    
    model = timm.create_model(model_name, pretrained=pretrained)
    return model


def get_torchvision_state_dict(model):
    """Extract state dict from torchvision model."""
    return model.state_dict()


def map_torchvision_to_prunable(tv_state_dict, prunable_model, verbose=True):
    """
    Map torchvision ResNet weights to prunable ResNet architecture.
    
    Note: This is a partial mapping. Some layers won't match due to:
    - Different conv1 (torchvision: 7x7 stride=2, prunable: 3x3 padding=1)
    - Channel selection layers (don't exist in torchvision)
    - Different fc layer (ImageNet 1000 classes vs CIFAR 10/100)
    - Pre-activation vs post-activation structure
    
    Args:
        tv_state_dict: State dict from torchvision model
        prunable_model: Prunable ResNet model
        verbose: Print mapping details
    
    Returns:
        Mapped state dict for prunable model
    """
    prunable_state_dict = prunable_model.state_dict()
    mapped_state_dict = {}
    skipped_keys = []
    unmapped_keys = []
    
    # Helper to check if shapes match
    def shapes_match(key1, key2):
        return tv_state_dict[key1].shape == prunable_state_dict[key2].shape
    
    # Map conv1 (will likely fail due to different kernel sizes)
    if 'conv1.weight' in tv_state_dict and 'conv1.weight' in prunable_state_dict:
        tv_conv1 = tv_state_dict['conv1.weight']
        prunable_conv1 = prunable_state_dict['conv1.weight']
        
        if tv_conv1.shape == prunable_conv1.shape:
            mapped_state_dict['conv1.weight'] = tv_conv1.clone()
            if verbose:
                print(f"✓ Mapped conv1.weight: {tv_conv1.shape}")
        else:
            if verbose:
                print(f"✗ Skipped conv1.weight: shapes don't match "
                      f"(torchvision: {tv_conv1.shape}, prunable: {prunable_conv1.shape})")
            skipped_keys.append('conv1.weight')
    
    # Map BatchNorm layers from each bottleneck block
    # torchvision structure: layer1.0.bn1, layer1.0.bn2, layer1.0.bn3, etc.
    # prunable structure: layer1.0.bn1, layer1.0.bn2, layer1.0.bn3, etc.
    
    for layer_name in ['layer1', 'layer2', 'layer3']:
        # Count blocks in torchvision model
        tv_block_count = sum(1 for k in tv_state_dict.keys() if k.startswith(f'{layer_name}.0.'))
        if tv_block_count == 0:
            # Try to infer from layer structure
            tv_keys = [k for k in tv_state_dict.keys() if k.startswith(f'{layer_name}.')]
            if tv_keys:
                # Get unique block indices
                block_indices = set()
                for k in tv_keys:
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        block_indices.add(int(parts[1]))
                tv_block_count = len(block_indices)
        
        # Count blocks in prunable model
        prunable_keys = [k for k in prunable_state_dict.keys() if k.startswith(f'{layer_name}.')]
        prunable_block_indices = set()
        for k in prunable_keys:
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                prunable_block_indices.add(int(parts[1]))
        prunable_block_count = len(prunable_block_indices)
        
        if verbose:
            print(f"\n{layer_name}: torchvision has {tv_block_count} blocks, "
                  f"prunable has {prunable_block_count} blocks")
        
        # Map each block
        for block_idx in range(min(tv_block_count, prunable_block_count)):
            # Map bn1, bn2, bn3
            for bn_name in ['bn1', 'bn2', 'bn3']:
                tv_key = f'{layer_name}.{block_idx}.{bn_name}.weight'
                prunable_key = f'{layer_name}.{block_idx}.{bn_name}.weight'
                
                if tv_key in tv_state_dict and prunable_key in prunable_state_dict:
                    if shapes_match(tv_key, prunable_key):
                        mapped_state_dict[prunable_key] = tv_state_dict[tv_key].clone()
                        # Also map bias, running_mean, running_var
                        for suffix in ['bias', 'running_mean', 'running_var']:
                            tv_suffix_key = f'{layer_name}.{block_idx}.{bn_name}.{suffix}'
                            prunable_suffix_key = f'{layer_name}.{block_idx}.{bn_name}.{suffix}'
                            if tv_suffix_key in tv_state_dict and prunable_suffix_key in prunable_state_dict:
                                if shapes_match(tv_suffix_key, prunable_suffix_key):
                                    mapped_state_dict[prunable_suffix_key] = tv_state_dict[tv_suffix_key].clone()
                        if verbose:
                            print(f"  ✓ Mapped {prunable_key}: {tv_state_dict[tv_key].shape}")
                    else:
                        if verbose:
                            print(f"  ✗ Skipped {prunable_key}: shapes don't match")
                        skipped_keys.append(prunable_key)
            
            # Map conv1, conv2, conv3
            for conv_name in ['conv1', 'conv2', 'conv3']:
                tv_key = f'{layer_name}.{block_idx}.{conv_name}.weight'
                prunable_key = f'{layer_name}.{block_idx}.{conv_name}.weight'
                
                if tv_key in tv_state_dict and prunable_key in prunable_state_dict:
                    if shapes_match(tv_key, prunable_key):
                        mapped_state_dict[prunable_key] = tv_state_dict[tv_key].clone()
                        if verbose:
                            print(f"  ✓ Mapped {prunable_key}: {tv_state_dict[tv_key].shape}")
                    else:
                        if verbose:
                            print(f"  ✗ Skipped {prunable_key}: shapes don't match "
                                  f"(tv: {tv_state_dict[tv_key].shape}, "
                                  f"prunable: {prunable_state_dict[prunable_key].shape})")
                        skipped_keys.append(prunable_key)
            
            # Map downsample if exists
            tv_downsample_key = f'{layer_name}.{block_idx}.downsample.0.weight'
            prunable_downsample_key = f'{layer_name}.{block_idx}.downsample.0.weight'
            
            if tv_downsample_key in tv_state_dict and prunable_downsample_key in prunable_state_dict:
                if shapes_match(tv_downsample_key, prunable_downsample_key):
                    mapped_state_dict[prunable_downsample_key] = tv_state_dict[tv_downsample_key].clone()
                    if verbose:
                        print(f"  ✓ Mapped {prunable_downsample_key}")
                else:
                    skipped_keys.append(prunable_downsample_key)
    
    # Map final BN layer (if exists)
    if 'bn.weight' in tv_state_dict and 'bn.weight' in prunable_state_dict:
        if shapes_match('bn.weight', 'bn.weight'):
            mapped_state_dict['bn.weight'] = tv_state_dict['bn.weight'].clone()
            for suffix in ['bias', 'running_mean', 'running_var']:
                tv_key = f'bn.{suffix}'
                prunable_key = f'bn.{suffix}'
                if tv_key in tv_state_dict and prunable_key in prunable_state_dict:
                    if shapes_match(tv_key, prunable_key):
                        mapped_state_dict[prunable_key] = tv_state_dict[tv_key].clone()
            if verbose:
                print(f"\n✓ Mapped final bn layer")
        else:
            skipped_keys.append('bn.weight')
    
    # Channel selection layers: Initialize to all 1s (all channels active)
    for key in prunable_state_dict.keys():
        if 'select.indexes' in key:
            num_channels = prunable_state_dict[key].shape[0]
            mapped_state_dict[key] = torch.ones(num_channels)
            if verbose:
                print(f"✓ Initialized {key} to all 1s ({num_channels} channels)")
    
    # FC layer: Skip (different number of classes)
    if verbose:
        print(f"\n✗ Skipped fc layer (different number of classes)")
    
    # Report unmapped keys
    for key in prunable_state_dict.keys():
        if key not in mapped_state_dict:
            unmapped_keys.append(key)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Mapping Summary:")
        print(f"  Mapped: {len(mapped_state_dict)} keys")
        print(f"  Skipped: {len(skipped_keys)} keys")
        print(f"  Unmapped: {len(unmapped_keys)} keys")
        if unmapped_keys:
            print(f"\nUnmapped keys (will use random initialization):")
            for key in unmapped_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(unmapped_keys) > 10:
                print(f"  ... and {len(unmapped_keys) - 10} more")
    
    return mapped_state_dict, skipped_keys, unmapped_keys


def main():
    parser = argparse.ArgumentParser(description='Transfer weights from torchvision/timm ResNet to prunable ResNet')
    parser.add_argument('--source', type=str, default='torchvision',
                       choices=['torchvision', 'timm'],
                       help='Source of pretrained model')
    parser.add_argument('--model-name', type=str, default='resnet50',
                       help='Model name (e.g., resnet50, resnet101)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Target dataset')
    parser.add_argument('--depth', type=int, default=164,
                       help='Depth of prunable ResNet (must be 9n+2, e.g., 164)')
    parser.add_argument('--output', type=str, default='transferred_model.pth.tar',
                       help='Output path for transferred model')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Load model without pretrained weights (for testing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Weight Transfer: torchvision/timm → Prunable ResNet")
    print("="*60)
    
    # Load source model
    print(f"\n1. Loading {args.source} model: {args.model_name}")
    if args.source == 'torchvision':
        source_model = load_torchvision_resnet(args.model_name, pretrained=not args.no_pretrained)
    else:
        source_model = load_timm_resnet(args.model_name, pretrained=not args.no_pretrained)
    
    source_state_dict = get_torchvision_state_dict(source_model)
    print(f"   Loaded {len(source_state_dict)} parameters")
    
    # Create prunable model
    print(f"\n2. Creating prunable ResNet (depth={args.depth}, dataset={args.dataset})")
    prunable_model = prunable_resnet(depth=args.depth, dataset=args.dataset)
    print(f"   Created model with {sum(p.numel() for p in prunable_model.parameters())} parameters")
    
    # Map weights
    print(f"\n3. Mapping weights...")
    mapped_state_dict, skipped, unmapped = map_torchvision_to_prunable(
        source_state_dict, prunable_model, verbose=True
    )
    
    # Load mapped weights
    print(f"\n4. Loading mapped weights into prunable model...")
    prunable_model.load_state_dict(mapped_state_dict, strict=False)
    
    # Save model
    print(f"\n5. Saving transferred model to {args.output}")
    torch.save({
        'state_dict': prunable_model.state_dict(),
        'epoch': 0,
        'best_prec1': 0.0,
        'source': args.source,
        'source_model': args.model_name,
        'dataset': args.dataset,
        'depth': args.depth,
    }, args.output)
    
    print(f"\n✓ Transfer complete!")
    print(f"\nNote: Some layers were not transferred due to architecture differences.")
    print(f"      You should fine-tune this model before pruning for best results.")
    print(f"\nNext steps:")
    print(f"  1. Fine-tune with sparsity: python main.py -sr --s 0.00001 --resume {args.output} ...")
    print(f"  2. Prune: python resprune.py --model <sparsity_trained_model> ...")
    print(f"  3. Fine-tune pruned model: python main.py --refine <pruned_model> ...")


if __name__ == '__main__':
    main()


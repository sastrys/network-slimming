"""
Transfer weights from torchvision/timm ResNet to prunable ImageNet ResNet architecture.
This script handles ImageNet models (224x224, 1000 classes).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from models.preresnet_imagenet import resnet_imagenet
import argparse


def load_torchvision_resnet(model_name='resnet50', pretrained=True):
    """Load ResNet from torchvision."""
    model_dict = {
        'resnet50': tv_models.resnet50,
        'resnet101': tv_models.resnet101,
        'resnet152': tv_models.resnet152,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")
    
    model = model_dict[model_name](pretrained=pretrained)
    return model


def load_timm_resnet(model_name='resnet50', pretrained=True):
    """Load ResNet from timm."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is not installed. Install with: pip install timm")
    
    model = timm.create_model(model_name, pretrained=pretrained)
    return model


def map_weights(tv_state_dict, prunable_model, verbose=True):
    """
    Map torchvision ResNet weights to prunable ImageNet ResNet.
    
    Args:
        tv_state_dict: State dict from torchvision/timm model
        prunable_model: Prunable ResNet model
        verbose: Print mapping details
    
    Returns:
        Mapped state dict
    """
    prunable_state_dict = prunable_model.state_dict()
    mapped_state_dict = {}
    skipped_keys = []
    unmapped_keys = []
    
    def shapes_match(key1, key2):
        return tv_state_dict[key1].shape == prunable_state_dict[key2].shape
    
    # Map conv1, bn1
    if 'conv1.weight' in tv_state_dict and 'conv1.weight' in prunable_state_dict:
        if shapes_match('conv1.weight', 'conv1.weight'):
            mapped_state_dict['conv1.weight'] = tv_state_dict['conv1.weight'].clone()
            if verbose:
                print(f"✓ Mapped conv1.weight: {tv_state_dict['conv1.weight'].shape}")
        else:
            skipped_keys.append('conv1.weight')
    
    if 'bn1.weight' in tv_state_dict and 'bn1.weight' in prunable_state_dict:
        if shapes_match('bn1.weight', 'bn1.weight'):
            for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
                tv_key = f'bn1.{suffix}'
                prunable_key = f'bn1.{suffix}'
                if tv_key in tv_state_dict and prunable_key in prunable_state_dict:
                    if shapes_match(tv_key, prunable_key):
                        mapped_state_dict[prunable_key] = tv_state_dict[tv_key].clone()
            if verbose:
                print(f"✓ Mapped bn1")
        else:
            skipped_keys.append('bn1.weight')
    
    # Map layers
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        # Get block count from torchvision model
        tv_block_keys = [k for k in tv_state_dict.keys() if k.startswith(f'{layer_name}.')]
        tv_block_indices = set()
        for k in tv_block_keys:
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                tv_block_indices.add(int(parts[1]))
        tv_block_count = len(tv_block_indices) if tv_block_indices else 0
        
        # Get block count from prunable model
        prunable_block_keys = [k for k in prunable_state_dict.keys() if k.startswith(f'{layer_name}.')]
        prunable_block_indices = set()
        for k in prunable_block_keys:
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                prunable_block_indices.add(int(parts[1]))
        prunable_block_count = len(prunable_block_indices) if prunable_block_indices else 0
        
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
                        for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
                            tv_suffix_key = f'{layer_name}.{block_idx}.{bn_name}.{suffix}'
                            prunable_suffix_key = f'{layer_name}.{block_idx}.{bn_name}.{suffix}'
                            if tv_suffix_key in tv_state_dict and prunable_suffix_key in prunable_state_dict:
                                if shapes_match(tv_suffix_key, prunable_suffix_key):
                                    mapped_state_dict[prunable_suffix_key] = tv_state_dict[tv_suffix_key].clone()
                        if verbose:
                            print(f"  ✓ Mapped {prunable_key}")
                    else:
                        skipped_keys.append(prunable_key)
            
            # Map conv1, conv2, conv3
            for conv_name in ['conv1', 'conv2', 'conv3']:
                tv_key = f'{layer_name}.{block_idx}.{conv_name}.weight'
                prunable_key = f'{layer_name}.{block_idx}.{conv_name}.weight'
                
                if tv_key in tv_state_dict and prunable_key in prunable_state_dict:
                    if shapes_match(tv_key, prunable_key):
                        mapped_state_dict[prunable_key] = tv_state_dict[tv_key].clone()
                        if verbose:
                            print(f"  ✓ Mapped {prunable_key}")
                    else:
                        skipped_keys.append(prunable_key)
            
            # Map downsample
            tv_downsample_key = f'{layer_name}.{block_idx}.downsample.0.weight'
            prunable_downsample_key = f'{layer_name}.{block_idx}.downsample.0.weight'
            
            if tv_downsample_key in tv_state_dict and prunable_downsample_key in prunable_state_dict:
                if shapes_match(tv_downsample_key, prunable_downsample_key):
                    mapped_state_dict[prunable_downsample_key] = tv_state_dict[tv_downsample_key].clone()
                    if verbose:
                        print(f"  ✓ Mapped {prunable_downsample_key}")
                else:
                    skipped_keys.append(prunable_downsample_key)
    
    # Map final BN layer
    if 'bn.weight' in tv_state_dict and 'bn.weight' in prunable_state_dict:
        if shapes_match('bn.weight', 'bn.weight'):
            for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
                tv_key = f'bn.{suffix}'
                prunable_key = f'bn.{suffix}'
                if tv_key in tv_state_dict and prunable_key in prunable_state_dict:
                    if shapes_match(tv_key, prunable_key):
                        mapped_state_dict[prunable_key] = tv_state_dict[tv_key].clone()
            if verbose:
                print(f"\n✓ Mapped final bn layer")
        else:
            skipped_keys.append('bn.weight')
    
    # Map fc layer
    if 'fc.weight' in tv_state_dict and 'fc.weight' in prunable_state_dict:
        if shapes_match('fc.weight', 'fc.weight'):
            mapped_state_dict['fc.weight'] = tv_state_dict['fc.weight'].clone()
            mapped_state_dict['fc.bias'] = tv_state_dict['fc.bias'].clone()
            if verbose:
                print(f"✓ Mapped fc layer")
        else:
            skipped_keys.append('fc.weight')
    
    # Initialize channel selection layers to all 1s
    for key in prunable_state_dict.keys():
        if 'select.indexes' in key:
            num_channels = prunable_state_dict[key].shape[0]
            mapped_state_dict[key] = torch.ones(num_channels)
            if verbose:
                print(f"✓ Initialized {key} to all 1s ({num_channels} channels)")
    
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
            for key in unmapped_keys[:10]:
                print(f"  - {key}")
            if len(unmapped_keys) > 10:
                print(f"  ... and {len(unmapped_keys) - 10} more")
    
    return mapped_state_dict, skipped_keys, unmapped_keys


def main():
    parser = argparse.ArgumentParser(description='Transfer weights from torchvision/timm ResNet to prunable ImageNet ResNet')
    parser.add_argument('--source', type=str, default='torchvision',
                       choices=['torchvision', 'timm'],
                       help='Source of pretrained model')
    parser.add_argument('--model-name', type=str, default='resnet50',
                       help='Model name (resnet50, resnet101, resnet152)')
    parser.add_argument('--num-classes', type=int, default=1000,
                       help='Number of classes (default: 1000 for ImageNet)')
    parser.add_argument('--output', type=str, default='transferred_imagenet.pth.tar',
                       help='Output path for transferred model')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Load model without pretrained weights')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Weight Transfer: torchvision/timm → Prunable ImageNet ResNet")
    print("="*60)
    
    # Load source model
    print(f"\n1. Loading {args.source} model: {args.model_name}")
    if args.source == 'torchvision':
        source_model = load_torchvision_resnet(args.model_name, pretrained=not args.no_pretrained)
    else:
        source_model = load_timm_resnet(args.model_name, pretrained=not args.no_pretrained)
    
    source_state_dict = source_model.state_dict()
    print(f"   Loaded {len(source_state_dict)} parameters")
    
    # Create prunable model
    print(f"\n2. Creating prunable ImageNet ResNet: {args.model_name}")
    prunable_model = resnet_imagenet(arch=args.model_name, num_classes=args.num_classes)
    print(f"   Created model with {sum(p.numel() for p in prunable_model.parameters())} parameters")
    
    # Map weights
    print(f"\n3. Mapping weights...")
    mapped_state_dict, skipped, unmapped = map_weights(
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
        'num_classes': args.num_classes,
        'arch': args.model_name,
    }, args.output)
    
    print(f"\n✓ Transfer complete!")
    print(f"\nNext steps:")
    print(f"  1. Fine-tune with sparsity: python train_imagenet.py -sr --s 0.00001 --resume {args.output} ...")
    print(f"  2. Prune: python prune_imagenet.py --model <sparsity_trained_model> ...")
    print(f"  3. Fine-tune pruned model: python train_imagenet.py --refine <pruned_model> ...")


if __name__ == '__main__':
    main()


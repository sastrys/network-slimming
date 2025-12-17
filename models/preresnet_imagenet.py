"""
ImageNet-compatible prunable ResNet with channel selection layers.
This architecture matches torchvision ResNet structure for ImageNet (224x224, 1000 classes).
"""

from __future__ import absolute_import
import math
import torch.nn as nn
from .channel_selection import channel_selection


__all__ = ['resnet_imagenet']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


def resnet_imagenet(arch='resnet50', num_classes=1000, cfg=None, pretrained=False):
    """
    Create ImageNet-compatible prunable ResNet.
    
    Args:
        arch: Architecture name ('resnet50', 'resnet101', 'resnet152')
        num_classes: Number of classes (default: 1000 for ImageNet)
        cfg: Channel configuration (if None, uses default based on arch)
        pretrained: Not used, kept for compatibility
    
    Returns:
        Prunable ResNet model for ImageNet
    """
    # ResNet-50: [3, 4, 6, 3] blocks per layer
    # ResNet-101: [3, 4, 23, 3]
    # ResNet-152: [3, 8, 36, 3]
    
    if arch == 'resnet50':
        layers = [3, 4, 6, 3]
        default_planes = [64, 128, 256, 512]
    elif arch == 'resnet101':
        layers = [3, 4, 23, 3]
        default_planes = [64, 128, 256, 512]
    elif arch == 'resnet152':
        layers = [3, 8, 36, 3]
        default_planes = [64, 128, 256, 512]
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from resnet50, resnet101, resnet152")
    
    if cfg is None:
        # Default cfg: construct based on architecture
        # Each bottleneck block has 3 convs: [in_channels, mid_channels, mid_channels]
        cfg = []
        inplanes = 64  # After conv1 and maxpool
        
        for layer_idx, (num_blocks, planes) in enumerate(zip(layers, default_planes)):
            for block_idx in range(num_blocks):
                if block_idx == 0:
                    # First block: may have stride=2
                    stride = 2 if layer_idx > 0 else 1
                    # Input channels depend on previous layer
                    if layer_idx == 0:
                        in_channels = 64
                    else:
                        in_channels = default_planes[layer_idx - 1] * 4
                    mid_channels = planes
                else:
                    # Subsequent blocks: no stride, input = output of previous
                    stride = 1
                    in_channels = planes * 4
                    mid_channels = planes
                
                cfg.append([in_channels, mid_channels, mid_channels])
        
        # Add final layer channels
        cfg.append([default_planes[-1] * 4])
    
    model = ResNet_ImageNet(layers, default_planes, num_classes, cfg)
    return model


class ResNet_ImageNet(nn.Module):
    def __init__(self, layers, planes, num_classes=1000, cfg=None):
        super(ResNet_ImageNet, self).__init__()
        block = Bottleneck
        
        if cfg is None:
            raise ValueError("cfg must be provided")
        
        self.inplanes = 64
        
        # ImageNet-style conv1: 7x7, stride=2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build layers
        cfg_idx = 0
        self.layer1 = self._make_layer(block, planes[0], layers[0], 
                                      cfg[cfg_idx:cfg_idx+layers[0]*3], stride=1)
        cfg_idx += layers[0] * 3
        self.inplanes = planes[0] * block.expansion
        
        self.layer2 = self._make_layer(block, planes[1], layers[1], 
                                      cfg[cfg_idx:cfg_idx+layers[1]*3], stride=2)
        cfg_idx += layers[1] * 3
        self.inplanes = planes[1] * block.expansion
        
        self.layer3 = self._make_layer(block, planes[2], layers[2], 
                                      cfg[cfg_idx:cfg_idx+layers[2]*3], stride=2)
        cfg_idx += layers[2] * 3
        self.inplanes = planes[2] * block.expansion
        
        self.layer4 = self._make_layer(block, planes[3], layers[3], 
                                      cfg[cfg_idx:cfg_idx+layers[3]*3], stride=2)
        cfg_idx += layers[3] * 3
        self.inplanes = planes[3] * block.expansion
        
        # Final layers
        self.bn = nn.BatchNorm2d(planes[3] * block.expansion)
        self.select = channel_selection(planes[3] * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg[-1] if cfg else planes[3] * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers_list = []
        for i in range(blocks):
            layers_list.append(block(self.inplanes, planes, cfg[i*3:(i+1)*3], 
                                    stride if i == 0 else 1, downsample if i == 0 else None))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


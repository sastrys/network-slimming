"""
Prune ImageNet ResNet model using BatchNorm scale factors.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.preresnet_imagenet import resnet_imagenet
from models.channel_selection import channel_selection


parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet Pruning')
parser.add_argument('--data', type=str, default='./data/imagenet',
                    help='path to ImageNet data')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--arch', type=str, default='resnet50',
                    choices=['resnet50', 'resnet101', 'resnet152'],
                    help='architecture (default: resnet50)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes (default: 1000)')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

# Create model
model = resnet_imagenet(arch=args.arch, num_classes=args.num_classes)

if args.cuda:
    model.cuda()

# Load model
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        if 'cfg' in checkpoint:
            # Load pruned model
            model = resnet_imagenet(arch=args.arch, num_classes=args.num_classes, cfg=checkpoint['cfg'])
            if args.cuda:
                model.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        if 'best_prec1' in checkpoint:
            print("=> loaded checkpoint (epoch {}) Prec1: {:f}"
                  .format(checkpoint.get('epoch', 0), checkpoint['best_prec1']))
        else:
            print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

# Collect all BN weights
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

# Calculate threshold
y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

# Create masks
pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda() if args.cuda else weight_copy.gt(thre).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total
print(f'\nPre-processing Successful! Pruned {pruned_ratio*100:.2f}% of channels')

# Test model (optional - requires ImageNet data)
def test(model):
    if not os.path.exists(args.data):
        print(f"ImageNet data not found at {args.data}. Skipping test.")
        return 0.0
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, 'val'),
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                           ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    model.eval()
    correct = 0
    total_samples = 0
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_samples += len(target)
    
    acc = correct / float(total_samples)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, total_samples, 100. * acc))
    return acc

acc = test(model)

# Build new pruned model
print("Cfg:", cfg)
newmodel = resnet_imagenet(arch=args.arch, num_classes=args.num_classes, cfg=cfg)
if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))

# Copy weights
old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # Channel selection layer - don't prune BN, set channel selection indexes
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            # Regular BN - prune directly
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0:
            # First conv (conv1)
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

            # Check if this is the last conv in bottleneck (conv3)
            if conv_count % 3 != 1:
                w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue

        # Downsampling convolutions
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, 
          os.path.join(args.save, 'pruned.pth.tar'))

print(newmodel)
model = newmodel
test(model)


import torch

import numpy as np
import random

import os
import sys

sys.path.append("model")
import imgnet_resnet
import imgnet_resnet_supcon
import imgnet_mobilenet
import imgnet_vit

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.models as tmodels
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from typing import List

# -------- fix random seed 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

########################################################################################################
########################################################################################################
########################################################################################################

def get_model(args):
    if args.in_data == 'ImageNet':
        args.num_classes = 1000
    else:
        assert False, "Unknown dataset : {}".format(args.in_data)
    
    if args.arch == 'R50' and args.in_data == 'ImageNet':
        if args.supcon:
            net = imgnet_resnet_supcon.SupConResNet(num_classes=args.num_classes)
            cudnn.benchmark = True
        else:
            net = imgnet_resnet.__dict__['resnet50'](pretrained=True)
            cudnn.benchmark = True
    elif args.arch == 'MNet' and args.in_data == 'ImageNet':
        net = imgnet_mobilenet.__dict__['mobilenet_v2'](pretrained=True)
        cudnn.benchmark = True
    elif args.arch == 'ViT' and args.in_data == 'ImageNet':
        net = imgnet_vit.__dict__['vit_b_16'](pretrained=True)
        cudnn.benchmark = True
    else:
        assert False, "Unknown model : {}".format(args.arch)

    return net

########################################################################################################
########################################################################################################
########################################################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

########################################################################################################
########################################################################################################
########################################################################################################

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

########################################################################################################
########################################################################################################
########################################################################################################


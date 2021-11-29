#modules
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import math
from collections import OrderedDict
import re
from PIL import Image


import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models, transforms, datasets
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

#hyper-parameter
BATCH_size = 16
IMG_size = 256
LR=1e-4 
WEIGHT_decay=5e-4
MODEL_name = 'OUR_BEST_MODEL' #to stroe the hyperparameter
JITTER_strength = 0.2 
CROP_size = 224

#splitting the data into ratio(0.2)
def split_ids(path, ratio):
    ids = []
    train_ids = []
    val_ids = []
    
    lists = os.listdir(path)
    lists.sort()
    
    for i, label in enumerate(lists):
        ids.append([])
        for img_name in os.listdir(path+'/'+label):
            ids[i].append(label+'/'+img_name)

    for i in range(10):

        ids_array = np.array(ids[i])
        perm = np.random.permutation(np.arange(len(ids_array)))
        cut = int(ratio*len(ids_array))

        train_ids.append(ids_array[perm][cut:])
        val_ids.append(ids_array[perm][:cut])
        
    return train_ids, val_ids


train_path = "../input/state-farm-distracted-driver-detection/imgs/train"
train_ids, val_ids = split_ids(train_path, 0.2)

for i in range(10):
    print('found {} train, {} validation images for class {}'.format(len(train_ids[i]), len(val_ids[i]), i))

#Defininng loader
def one_image_loader(path):
    return Image.open(path).convert('RGB')

class Loader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split_type, ids=None, transform=None):
        self.impath = rootdir
        self.transform = transform
        self.loader = one_image_loader
            

        imnames = []
        imclasses = []
        
        for i in range(10):
            if(split_type == 'train'):
                for j in train_ids[i]:
                    imclasses.append(i)
                    imnames.append(j)
            else :
                for j in val_ids[i]:
                    imclasses.append(i)
                    imnames.append(j)

        self.imnames = imnames
        self.imclasses = imclasses
    
    def __getitem__(self, index):
        original_img = self.loader(os.path.join(self.impath, self.imnames[index]))
        img = self.transform(original_img)
        label = self.imclasses[index]
        return img, label
        
    def __len__(self):
        return len(self.imnames)

#loading
train_loader = torch.utils.data.DataLoader(
            Loader(train_path, 'train', train_ids,
                              transform=transforms.Compose([
                                  transforms.Resize((IMG_size,IMG_size)),
                                  transforms.ToTensor(),
                                                                            transforms.ColorJitter(
                                                brightness=JITTER_strength,
                                                contrast=JITTER_strength,
                                                saturation=JITTER_strength,
                                                hue = JITTER_strength),
                                            transforms.RandomResizedCrop(CROP_size),
                                  
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                batch_size= BATCH_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
print('train_loader done')
print('loaded {} train images'.format(len(train_loader.dataset)))

validation_loader = torch.utils.data.DataLoader(
            Loader(train_path, 'val', val_ids,
                               transform=transforms.Compose([
                                   transforms.Resize((IMG_size,IMG_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                               batch_size=BATCH_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
print('validation_loader done')
print('loaded {} validation images'.format(len(validation_loader.dataset)))

data_dict = {
    'train':train_loader,
    'validation': validation_loader
            }

#Test loader
class TestLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, transform=None):
        self.impath = rootdir
        self.transform = transform
        self.loader = one_image_loader
        self.imnames = os.listdir(rootdir)

    
    def __getitem__(self, index):
        original_img = self.loader(os.path.join(self.impath, self.imnames[index]))
        img = self.transform(original_img)

        return self.imnames[index],img
        
    def __len__(self):
        return len(self.imnames)
    
#loading the test data
test_path = "../input/state-farm-distracted-driver-detection/imgs/test"
test_loader = torch.utils.data.DataLoader(
             TestLoader(test_path,
                    transform=transforms.Compose([
                                   transforms.Resize((IMG_size,IMG_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                               batch_size=BATCH_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
print('test_loader done')
print('loaded {} test images'.format(len(test_loader.dataset)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#weight initialization functions
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)        
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:        
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:        
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine is not None:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)        

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)     

#blocks implementation
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512): 
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu: 
            add_block += [nn.ReLU()]
        if dropout: 
            add_block += [nn.Dropout(p=0.5)] 
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x    

#models(Resnet 18, 50, 101)
class Res18(nn.Module):
    def __init__(self, class_num):
        super(Res18, self).__init__()
        fea_dim = IMG_size
        model_ft = models.resnet18(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.fc_embed = nn.Linear(512, fea_dim)
        self.fc_embed.apply(weights_init_classifier)
        self.classifier = ClassBlock(512, class_num)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea =  x.view(x.size(0), -1)
        pred = self.classifier(fea)
        return pred

        
class Res50(nn.Module):
    def __init__(self, class_num):
        super(Res50, self).__init__()
        fea_dim = IMG_size       
        model_ft = models.resnet50(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()        
        self.model = model_ft
        self.fc_embed = nn.Linear(2048, fea_dim)
        self.fc_embed.apply(weights_init_classifier)
        self.classifier = ClassBlock(2048, class_num)
        self.classifier.apply(weights_init_classifier)        
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea =  x.view(x.size(0), -1)
        pred = self.classifier(fea)  
        return pred
        
class Res101(nn.Module):
    def __init__(self, class_num):
        super(Res101, self).__init__()
        fea_dim = IMG_size     
        model_ft = models.resnet101(pretrained=False)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()        
        self.model = model_ft
        self.fc_embed = nn.Linear(2048, fea_dim)
        self.fc_embed.apply(weights_init_classifier)
        self.classifier = ClassBlock(2048, class_num)
        self.classifier.apply(weights_init_classifier)        
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        fea =  x.view(x.size(0), -1)
        pred = self.classifier(fea)
        return pred

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet

from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(
            block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model

class SAM(torch.optim.Optimizer): # SAM optimizer from https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

model = Res50(10) #num classes : 10, Resnet50
model.eval()

parameters = filter(lambda p: p.requires_grad, model.parameters())
n_parameters = sum([p.data.nelement() for p in model.parameters()])
print('  + Number of params: {}'.format(n_parameters))

model.cuda()
criterion = nn.CrossEntropyLoss()
base_optimizer = optim.Adam
optimizer = SAM(model.parameters(), base_optimizer, lr=0, weight_decay=WEIGHT_decay)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=8, T_mult=2, eta_max=LR,  T_up=2, gamma=0.5)

training_losses = []
training_acc = []
valid_losses = []
valid_acc = []

def train_model(model, criterion, optimizer, num_epochs=3):
    best_acc = -1.0
    
    for epoch in range(num_epochs):            
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            if phase == 'train':
                for inputs, labels in data_dict[phase]:
                    
                    inputs = inputs.cuda()
                    labels = labels.cuda()
    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, preds = torch.max(outputs, 1)
 
                    running_loss += loss.item() * inputs.shape[0]
                    running_corrects += torch.sum(preds == labels.data)
            else:
                with torch.no_grad():
                    for inputs, labels in data_dict[phase]:
                        
                        inputs = inputs.cuda()
                        labels = labels.cuda()
    
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
            
                        _, preds = torch.max(outputs, 1)
 
                        running_loss += loss.item() *  inputs.shape[0]
                        running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (len(data_dict[phase].dataset))  
            epoch_acc = running_corrects.double() / (len(data_dict[phase].dataset)) * 100 # 0 ~ 100 (%)
            
            if phase =='train':
                training_losses.append(epoch_loss)
                training_acc.append(epoch_acc)
            else:
                valid_losses.append(epoch_loss)
                valid_acc.append(epoch_acc)
                
                if(epoch_acc > best_acc):
                    best_acc = epoch_acc
                    print('model achieved the best accuracy ({:.4f}%) - saving best checkpoint...'.format(best_acc))
                    ckpt = { 'classifier': model.state_dict(),
                             'optimizer' : optimizer.state_dict(),
                             'best_acc' : best_acc }
                    torch.save(ckpt, './'+MODEL_name+'_best')
                                         
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss,   epoch_acc))
        scheduler.step()    
        if(valid_acc[len(valid_acc)-1] < best_acc - 5) :
            print('It might seems that overfitting occurs. Stop training\n')
            break

training_losses = []
training_acc = []
valid_losses = []
valid_acc = []

def train_model_SAM(model, criterion, optimizer, num_epochs=3):
    best_acc = -1.0
    
    for epoch in range(num_epochs):            
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            if phase == 'train':
                for inputs, labels in data_dict[phase]:
                    
                    inputs = inputs.cuda()
                    labels = labels.cuda()
    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    
                    criterion(model(inputs), labels).backward()
                    optimizer.second_step(zero_grad=True)

                    _, preds = torch.max(outputs, 1)
 
                    running_loss += loss.item() * inputs.shape[0]
                    running_corrects += torch.sum(preds == labels.data)
            else:
                with torch.no_grad():
                    for inputs, labels in data_dict[phase]:
                        
                        inputs = inputs.cuda()
                        labels = labels.cuda()
    
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
            
                        _, preds = torch.max(outputs, 1)
 
                        running_loss += loss.item() *  inputs.shape[0]
                        running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (len(data_dict[phase].dataset))  
            epoch_acc = running_corrects.double() / (len(data_dict[phase].dataset)) * 100 # 0 ~ 100 (%)
            
            if phase =='train':
                training_losses.append(epoch_loss)
                training_acc.append(epoch_acc)
            else:
                valid_losses.append(epoch_loss)
                valid_acc.append(epoch_acc)
                
                if(epoch_acc > best_acc):
                    best_acc = epoch_acc
                    print('model achieved the best accuracy ({:.4f}%) - saving best checkpoint...'.format(best_acc))
                    ckpt = { 'classifier': model.state_dict(),
                             'optimizer' : optimizer.state_dict(),
                             'best_acc' : best_acc }
                    torch.save(ckpt, './'+MODEL_name+'_best')
                                         
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss,   epoch_acc))
        scheduler.step()    
        if(valid_acc[len(valid_acc)-1] < best_acc - 5) :
            print('It might seems that overfitting occurs. Stop training\n')
            break

train_model_SAM(model, criterion, optimizer, num_epochs=50)

#visualize the results
import matplotlib.pyplot as plt

plt.plot(training_losses, label='train loss')
plt.plot(valid_losses, label='valid loss')
plt.legend()
plt.show()

plt.plot(training_acc, label='train acc(%)')
plt.plot(valid_acc, label='valid acc(%)')
plt.legend()


#analyze the results
ckpt = torch.load('./'+MODEL_name+'_best')

#1. load the trained model
model.load_state_dict(ckpt['classifier'])
optimizer.load_state_dict(ckpt['optimizer'])
best_acc = ckpt['best_acc']
print('checkpoint is loaded !')
print('loaded model''s best accuracy : %.4f' % best_acc)

#2. Testing
print('testing the loaded model')
model.eval()
running_loss = 0.0
running_corrects = 0

with torch.no_grad():
    for inputs, labels in data_dict['validation']:
        
        inputs = inputs.cuda()
        labels = labels.cuda()
    
        outputs = model(inputs)
        loss = criterion(outputs, labels)
            
        _, preds = torch.max(outputs, 1)
 
        running_loss += loss.item() * inputs.shape[0]
        running_corrects += torch.sum(preds == labels.data)
        
test_loss = running_loss / len(data_dict['validation'].dataset)  
test_acc = running_corrects.double() /  len(data_dict['validation'].dataset) * 100 # 0 ~ 100 (%)
print('Test_loss : {:.4f}, Test accuracy : {:.4f}'.format(test_loss, test_acc))

#submission
import pandas as pd

model.eval()
np.set_printoptions(precision=10, suppress=True)

# (1) eval & write the test set's result
print('Writing start..')

f = open('/kaggle/working/submission.csv', 'w')
f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')

with torch.no_grad():
    for i, data in enumerate(test_loader):
        if not i%100 :
            print(i*100.0/len(test_loader), "% done.")
        name, img_data = data
    
        img_data = img_data.cuda()
        img_data = model(img_data)

        img_data = F.softmax(img_data, dim=1)
        img_data = img_data.detach().cpu().numpy()

        for j in range(img_data.shape[0]):
            f.write(name[j])
            for k in img_data[j]:
                f.write(',')
                f.write(str(k))
            f.write('\n')
f.close()

#(2) check the file
pd.read_csv('/kaggle/working/submission.csv')
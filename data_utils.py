import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
from torchvision import datasets, transforms, models
from baselines.ViT.ViT_LRP import vit_base_patch16_224
import clip
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from imblearn.over_sampling import RandomOverSampler 

import medmnist
from medmnist import INFO, Evaluator

class HamData(Dataset):
    def __init__(self, data_dir, img_name_list, label_list, transform=None):
        self.data_dir = data_dir
        self.img_name_list = img_name_list
        self.label_list = label_list
        self.transform = transform
        self.targets = np.array(label_list)
        
    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img_path = os.path.join(self.data_dir, img_name+'.jpg',)
        img = Image.open(img_path)
        label = self.label_list[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.label_list)

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test"
}

LABEL_FILES = {"places365":"data/categories_places365_clean.txt",
               "imagenet":"data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cub":"data/cub_classes.txt",
              "ham10000":"data/ham10000_classes.txt",
              "ham_new":"data/ham10000_classes.txt",
              "covid":"data/covid_classes.txt",
              "mnist":"data/mnist_classes.txt",
              "oct":"data/oct_classes.txt"}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_vit_ham10000_preprocess():
    preprocess = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()])
    
    return preprocess

def get_vit_mnist_preprocess():
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5]), transforms.Resize([224, 224])])
    
    return preprocess

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
        
    elif dataset_name =="ham_new_train":
        data = datasets.ImageFolder('./data/ham_new/train', preprocess)
    
    elif dataset_name =="ham_new_val":
        data = datasets.ImageFolder('./data/ham_new/test', preprocess)
        
    elif dataset_name =="covid_train":
        data = datasets.ImageFolder('./data/covid/train', preprocess)
    
    elif dataset_name =="covid_val":
        data = datasets.ImageFolder('./data/covid/test', preprocess)
        
    elif dataset_name =="mnist_train":
        data_flag = 'bloodmnist'
        download = True
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        data = DataClass(split='train', transform=preprocess, download=download)
    
    elif dataset_name =="mnist_val":
        data_flag = 'bloodmnist'
        download = True
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        data = DataClass(split='test', transform=preprocess, download=download)
    
    elif dataset_name =="oct_train":
        data = datasets.ImageFolder('./data/oct/train', preprocess)
    
    elif dataset_name =="oct_val":
        data = datasets.ImageFolder('./data/oct/test', preprocess)
    
    return data

def get_targets_only(dataset_name):
    if 'mnist' in dataset_name:
        pil_data = get_data(dataset_name)
        y = pil_data.labels
        y = y.reshape(-1)
        y = list(y)
        return y
    else:
        pil_data = get_data(dataset_name)
        return pil_data.targets

def get_target_model(target_name, device, dataset, num_classes):
    
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'vit':
        target_model = vit_base_patch16_224(pretrained=True, num_classes=num_classes).to(device)
        if dataset == 'ham_new':
            target_model.load_state_dict(torch.load("./backbone/ham/pretrained_ham10000_vit_base_patch16_224_epoch5_acc0.9913.pth"))
            preprocess = get_vit_ham10000_preprocess()
        elif dataset == 'covid':
            target_model.load_state_dict(torch.load("./backbone/covid/vit_base_patch16_224_epoch422_acc0.8162.pth"))
            preprocess = get_vit_ham10000_preprocess()
        elif dataset == 'mnist':
            target_model.load_state_dict(torch.load("./backbone/mnist/pretrained_mnist_vit_base_patch16_224_epoch62_acc0.9705.pth"))
            preprocess = get_vit_mnist_preprocess()
        elif dataset == 'oct':
            target_model.load_state_dict(torch.load("./backbone/oct/oct_vit_base_patch16_224_epoch19_acc0.9970.pth"))
            preprocess = get_vit_ham10000_preprocess()
        target_model.eval()    
    
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess
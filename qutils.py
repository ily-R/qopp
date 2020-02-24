import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def myloss(pred_label, true_label):
    loss_label = F.cross_entropy(pred_label, true_label)
    return loss_label


def transferedModel(classes_num):
    resnet = models.resnet18(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    input_features = resnet.fc.in_features
    resnet.fc = nn.Linear(input_features, classes_num)
    return resnet


def getDataloaders(batch_size, root='data', img_transforms=None, val_per=0.1, test_per=0.1):
    train_dataset = ImageFolder(root, transform=img_transforms)
    val_dataset = ImageFolder(root, transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]))
    val_size = int(len(train_dataset) * val_per)
    test_size = int(len(train_dataset) * test_per)

    torch.manual_seed(1)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_idx, valid_idx, test_idx = indices[val_size + test_size:], indices[:val_size], indices[
                                                                                         val_size:test_size + val_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              pin_memory=torch.cuda.is_available(), num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            pin_memory=torch.cuda.is_available(), num_workers=4)

    test_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=test_sampler,
                             pin_memory=torch.cuda.is_available(), num_workers=4)

    return train_loader, val_loader, test_loader

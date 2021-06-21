#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.datasets import CIFAR10

import resnet


def initSeed(seed):
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, optimizer, trainloader, device):
    model.train()
    trainloss = 0
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item() * inputs.size()[0]

    trainloss = trainloss / len(trainloader.dataset)
    return trainloss


def test(model, testloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / len(testloader.dataset)
    return acc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    initSeed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    normalize = Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    train_transform = Compose([
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    test_transform = Compose([
        ToTensor(),
        normalize
    ])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    model = resnet.resnet20(pretrained=None).to(device)
    
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.01)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    start = time.time()
    tloss_ = []
    acc_ = []
    for epoch in range(args.num_epochs):
        trainloss = train(model, optimizer, trainloader, device)
        accuracy = test(model, testloader, device)
        tloss_.append(trainloss)
        acc_.append(accuracy)
        print('epoch:{}, trainloss:{:.3f}, accuracy:{:.1f}%'.format(epoch + 1, trainloss, accuracy), end='\r')
        lr_scheduler.step()
    print('')
    print('epoch per time:{:.3f}s'.format((time.time() - start) / args.num_epochs))

    #torch.save(model.state_dict(), 'trained_resnet.pth')
    def save_var(filepath, **kwargs):
        torch.save(kwargs, filepath)

    save_var('output/trained.pth',
         state_dict=copy.deepcopy(model.state_dict()),
         trainloss=tloss_,
         accuracy=acc_
    )


if __name__ == '__main__':
    main()
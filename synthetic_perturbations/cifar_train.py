import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import random
import time
import argparse
import numpy as np
from sklearn.datasets import make_classification

from models.ResNet import ResNet18, ResNet50
from models.DenseNet import DenseNet121
from util import AverageMeter, cross_entropy, accuracy, comput_l2norm_lim, normalize_l2norm, adjust_learning_rate


parser = argparse.ArgumentParser(description='synthetic perturbations')
parser.add_argument('--dataset', type=str, default='c10', help='[c10, c100, svhn]')
parser.add_argument('--aug', action='store_true', default=False, help=' use data augmentation')
parser.add_argument('--eps', type=int, default=6, help='perturbation strength')

parser.add_argument('--epoch', type=int, default=100, help='running epochs')
parser.add_argument('--batchsize', type=int, default=128, help='batchsize')
parser.add_argument('--patchsize', type=int, default=8, help='size of patch')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

parser.add_argument('--model', type=str, default='resnet18', help='[vgg, resnet18, resnet50, densenet]')

parser.add_argument('--sess', type=str, default='default', help='session name for experiment')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--clean', action='store_true', default=False, help='use clean data')

args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

dataset = args.dataset

if(dataset == 'c10'):
    data_func = datasets.CIFAR10
elif(dataset == 'c100'):
    data_func = datasets.CIFAR100
elif(dataset == 'svhn'):
    data_func = datasets.SVHN

if(dataset == 'c100'):
    num_classes = 100
else:
    num_classes = 10

# Data
print('==> Preparing data..')

plain_transform =  transforms.Compose([
    transforms.ToTensor()
])

aug_transform =  transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_transform = test_transform = plain_transform

if(args.aug):
    train_transform = aug_transform


if(args.dataset == 'svhn'):
    train_dataset = data_func(root='../datasets', split='train', download=True, transform=train_transform)
else:
    train_dataset = data_func(root='../datasets', train=True, download=True, transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)

if(args.dataset == 'svhn'):
    test_dataset = data_func(root='../datasets', split='test', download=True, transform=test_transform)
else:
    test_dataset = data_func(root='../datasets', train=False, download=True, transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)



if(not args.clean):
    n = train_dataset.data.shape[0] 
    if(args.dataset == 'svhn'): # ensure we generate enough synthetic data
        n *= 2

    img_size = 32
    noise_frame_size = args.patchsize

    is_even = img_size % noise_frame_size  

    num_patch = img_size//noise_frame_size
    if(is_even > 0):
        num_patch += 1

    n_random_fea =  int((img_size/noise_frame_size)**2 * 3)

    # generate initial data points
    simple_data, simple_label = make_classification(n_samples=n, n_features=n_random_fea, n_classes=num_classes, n_informative=n_random_fea, n_redundant=0, n_repeated=0, class_sep=10., flip_y=0., n_clusters_per_class=1)
    simple_data = simple_data.reshape([simple_data.shape[0], num_patch, num_patch, 3])
    simple_data = simple_data.astype(np.float32)

    # duplicate each dimension to get 2-D patches
    simple_images = np.repeat(simple_data, noise_frame_size, 2) 
    simple_images = np.repeat(simple_images, noise_frame_size, 1)
    simple_data = simple_images[:, 0:img_size, 0:img_size, :]
    
    # project the synthetic images into a small L2 ball
    linf = args.eps/255.
    feature_dim = img_size**2 * 3
    l2norm_lim = comput_l2norm_lim(linf, feature_dim)
    simple_data = normalize_l2norm(simple_data, l2norm_lim)


    train_dataset.data = train_dataset.data.astype(np.float)/255.
    if(args.dataset == 'svhn'):
        train_dataset.data = np.transpose(train_dataset.data, [0, 2, 3, 1])
        arr_target = train_dataset.labels
    else:
        arr_target = np.array(train_dataset.targets)
    
    # add synthetic noises to original examples
    for label in range(num_classes):
        orig_data_idx = arr_target == label
        simple_data_idx = simple_label == label
        mini_simple_data = simple_data[simple_data_idx][0:int(sum(orig_data_idx))]
        train_dataset.data[orig_data_idx] += mini_simple_data

    train_dataset.data = np.clip((train_dataset.data*255), 0, 255).astype(np.uint8)
    if(args.dataset == 'svhn'):
        train_dataset.data = np.transpose(train_dataset.data, [0, 3, 1, 2])



if(args.model == 'resnet18'):
    model = ResNet18(num_classes = num_classes) 
elif(args.model == 'resnet50'):
    model = ResNet50(num_classes = num_classes) 
elif(args.model == 'vgg'):
    model = models.vgg11(num_classes = num_classes)
elif(args.model == 'densenet'):
    model = DenseNet121(num_classes = num_classes)

model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
test_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

for epoch in range(args.epoch):
    adjust_learning_rate(optimizer, args.lr, epoch, all_epoch=args.epoch)

    # Train
    model.train()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    time0 = time.time()

    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()

        model.zero_grad()
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
                
        
        _, predicted = torch.max(logits.data, 1)
        acc = (predicted == labels).sum().item()/labels.size(0)

        acc_meter.update(acc)
        loss_meter.update(loss.item())

    print('Epoch %d, '%epoch, "Train acc %.2f loss: %.2f" % (acc_meter.avg*100, loss_meter.avg), end=' ')

    # Eval
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            logits = model(images)
            test_loss = test_criterion(logits, labels)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    time1 = time.time()

    acc = correct / total
    print("Test acc %.2f loss: %.2f, epoch time: %ds" % (acc*100, test_loss.item(), time1-time0))



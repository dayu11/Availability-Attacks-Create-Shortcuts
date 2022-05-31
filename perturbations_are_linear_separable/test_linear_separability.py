import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse

import numpy as np
import random

def normalize_01(tsr):
    maxv = torch.max(tsr)
    minv = torch.min(tsr)
    return (tsr-minv)/(maxv-minv)

def train(train_data, train_targets, net, optimizer):


    optimizer.zero_grad()
    inputs, targets = train_data, train_targets
    def closure():
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        outputs = net(inputs)
        loss = loss_func(outputs, targets)

    train_loss = loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = predicted.eq(targets.data).float().cpu().sum()
    acc = 100.*float(correct)/float(total)
    return (train_loss, acc)

parser = argparse.ArgumentParser(description='Fit perturbations with simple models')
parser.add_argument('--perturbed_x_path', default='x_train_cifar10_ntga_cnn_best.npy', type=str, help='path of perturbed data')
parser.add_argument('--clean_x_path', default='x_train_cifar10.npy', type=str, help='path of clean data')
parser.add_argument('--label_path', default='y_train_cifar10.npy', type=str, help='path of labels')
parser.add_argument('--hidden_layers', default=0, type=int, help='number of hidden layers')

args = parser.parse_args()


perturbed_x = np.load(args.perturbed_x_path)
clean_x = np.load(args.clean_x_path)
labels = np.load(args.label_path)
if(len(labels.shape)>1): #one-hot format
    labels = np.argmax(labels, axis=1)

perturbations = perturbed_x - clean_x

perturbations = torch.tensor(perturbations, dtype=torch.float).cuda()
labels = torch.tensor(labels, dtype=torch.long).cuda()



loss_func = nn.CrossEntropyLoss()
train_data = normalize_01(perturbations)
train_targets = labels

num_classes = 10 # CIFAR-10 dataset

module_list = [nn.Flatten()]
input_dim = np.prod(train_data.shape[1:])

hidden_width = 30
for i in range(args.hidden_layers):
    module_list.append(nn.Linear(input_dim, hidden_width))
    module_list.append(nn.Tanh())
    input_dim = hidden_width

module_list += [nn.Linear(input_dim, num_classes)]

net = nn.Sequential(*module_list)
net = net.cuda()
optimizer = optim.LBFGS(net.parameters(), lr=0.5) 

for step in range(50):
    train_loss, train_acc = train(train_data, train_targets, net, optimizer)
print('training loss: %.3f'%train_loss, 'training acc: %.2f'%train_acc)


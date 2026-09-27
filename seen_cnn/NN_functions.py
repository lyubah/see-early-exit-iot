# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:43:42 2024

@author: dina.hussein
"""

#!/usr/bin/env python3

import argparse
import numpy as np
import statistics
from numpy.fft import fft
import torch.utils.data as dl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from torch.utils.data.sampler import SubsetRandomSampler
import math
import csv
from scipy import signal
from scipy.fft import fftshift
from torch.utils.data import DataLoader
from scipy import signal
from numpy import savetxt
from collections import Counter
import itertools
import pickle
import os
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

# Load necessary Pytorch packages
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np

from sklearn.model_selection import train_test_split
import sys
from random import randint
import copy
import numpy as np
np.random.seed(123)
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.vq import whiten, kmeans, vq
import pandas as pd
from scipy.spatial import distance

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from tqdm import tqdm
device = torch.device('cpu')

# Define the CNN architecture
class CNN1D(nn.Module):
    def __init__(self, in_channels, out_classes, input_sequence_length):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc1_input_size = 32 * input_sequence_length  #75 for shoaib, 192 for pamap
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, out_classes)

    def forward(self, x):
        # x has shape (batch_size, in_channels, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def adversarial_train(self, train_set, input_shape,device, CHANNEL_NB, num_activities, input_sequence_length, checkpoint_path="TrainingRes/model_advTrn_target", epochs=10, batch=32,
                            attack='PGD', epsilon=1e-1):
        tensor_data = []
        tensor_labs = []


        aug_data, aug_labels = np.zeros((0,) + input_shape[1:]), np.zeros((0,))

        for X, y in tqdm(train_set, desc="Adversarial augmentation"):
            tensor_data.append(X)
            tensor_labs.append(y)
            if attack == 'PGD':
                X_adv = self.projected_gradient_descent(X, epsilon, eps_iter=epsilon/10)
                X_adv = X_adv.cpu().numpy()
            else:
                raise ValueError("Attack not implemented yet.")

            aug_data = np.concatenate([aug_data, X_adv], axis=0)
            aug_labels = np.concatenate([aug_labels, y], axis=0)
        # dataset = mlp_dataset(aug_data, aug_labels)
        concatenated_data = torch.cat(tensor_data, dim=0)
        concatenated_labels = torch.cat(tensor_labs, dim=0)


        data_com = np.concatenate([concatenated_data, aug_data], axis=0)
        label_com = np.concatenate([concatenated_labels, aug_labels], axis=0)
        dataset = mlp_dataset(data_com, label_com)
        train_dl = dl.DataLoader(dataset, batch_size = batch, shuffle=True)
        
        # adv_dataset = torch.utils.data.TensorDataset(torch.from_numpy(aug_data).float(), torch.from_numpy(aug_labels).long())
        # combined_dataset = torch.utils.data.ConcatDataset([train_set, adv_dataset])
        # loader = torch.utils.data.DataLoader(combined_dataset, shuffle=True, batch_size=batch)

        cnn_model = CNN_train(train_dl, train_dl, device, epochs, CHANNEL_NB, num_activities, input_sequence_length)
        torch.save(cnn_model.state_dict(), checkpoint_path)
            
    def projected_gradient_descent(model_fn, x, eps, eps_iter=0.1, nb_iter=100, norm=2,
                                   clip_min=None, clip_max=None, y=None, targeted=False,
                                   rand_init=None, rand_minmax=0.3, sanity_checks=True):
        assert eps_iter <= eps, (eps_iter, eps)
        if norm == 1:
            raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                      " step for PGD when norm=1, because norm=1 FGM "
                                      " changes only one pixel at a time. We need "
                                      " to rigorously test a strong norm=1 PGD "
                                      "before enabling this feature.")
        if norm not in [np.inf, 2]:
            raise ValueError("Norm order must be either np.inf or 2.")

        asserts = []

        if clip_min is not None:
            asserts.append(torch.all(x >= clip_min))

        if clip_max is not None:
            asserts.append(torch.all(x <= clip_max))

        if rand_init:
            rand_minmax = eps
            eta = torch.FloatTensor(x.shape).uniform_(-rand_minmax, rand_minmax)
        else:
            eta = torch.zeros_like(x)

        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, min=clip_min, max=clip_max)

        if y is None:
            _, y = torch.max(model_fn(x), 1)

        i = 0
        while i < nb_iter:
            adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, norm, clip_min=clip_min,
                                         clip_max=clip_max, y=y, targeted=targeted)

            eta = adv_x - x
            eta = clip_eta(eta, norm, eps)
            adv_x = x + eta


            if clip_min is not None or clip_max is not None:
                adv_x = torch.clamp(adv_x, min=clip_min, max=clip_max)
            i += 1

        asserts.append(eps_iter <= eps)
        if norm == np.inf and clip_min is not None:
            asserts.append(eps + clip_min <= clip_max)

        if sanity_checks:
            assert all(asserts)
        return adv_x

def fast_gradient_method(model_fn, x, eps, norm, clip_min=None, clip_max=None, y=None,
                         targeted=False, sanity_checks=False):
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2.")

    asserts = []
    if clip_min is not None:
        asserts.append(torch.all(x >= clip_min))

    if clip_max is not None:
        asserts.append(torch.all(x <= clip_max))

    if y is None:
        _, y = torch.max(model_fn(x), 1)

    grad = compute_gradient(model_fn, x, y, targeted)

    optimal_perturbation = optimize_linear(grad, eps, norm)
    adv_x = x + optimal_perturbation

    if (clip_min is not None) or (clip_max is not None):
        assert clip_min is not None and clip_max is not None
        adv_x = torch.clamp(adv_x, min=clip_min, max=clip_max)

    if sanity_checks:
        assert all(asserts)
    return adv_x


def compute_gradient(model_fn, x, y, targeted):
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.tensor(x.clone().detach(),requires_grad = True)


    # Compute loss
    logits = model_fn(x)
    loss = loss_fn(logits, y)
    if targeted:
        loss = -loss  

    grad = torch.autograd.grad(loss, x)[0]
    return grad

def optimize_linear(grad, eps, norm=np.inf):
    axis = list(range(1, len(grad.size())))
    avoid_zero_div = 1e-12

    if norm == np.inf:
        optimal_perturbation = grad.sign()
        optimal_perturbation = optimal_perturbation.detach()
    elif norm == 1:
        abs_grad = grad.abs()
        sign = grad.sign()
        max_abs_grad, _ = abs_grad.max(dim=axis, keepdim=True)
        tied_for_max = abs_grad.eq(max_abs_grad).float()
        num_ties = tied_for_max.sum(dim=axis, keepdim=True)
        optimal_perturbation = sign * tied_for_max / (num_ties + avoid_zero_div)
    elif norm == 2:
        square = torch.clamp(torch.sum(grad**2, dim=axis, keepdim=True), min=avoid_zero_div)
        optimal_perturbation = grad / torch.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are currently implemented.")

    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation


def clip_eta(eta, norm, eps):
    """
    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param norm: Order of the norm (mimics Numpy).
                  Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, bound of the perturbation.
    """

    if norm not in [np.inf, 1, 2]:
        raise ValueError('norm must be np.inf, 1, or 2.')

    avoid_zero_div = 1e-12
    axis = list(range(1, len(eta.shape)))

    if norm == np.inf:
        eta = torch.clamp(eta, min=-eps, max=eps)
    else:
        if norm == 1:
            raise NotImplementedError("")

        elif norm == 2:
            norm = torch.sqrt(torch.clamp(torch.sum(eta ** 2, dim=axis, keepdim=True), min=avoid_zero_div))
        factor = torch.min(torch.tensor(1.0), eps / norm)
        eta = eta * factor

    return eta


def CNN_train(train_loader, test_loader, device, num_epochs, CHANNEL_NB, num_activities, input_sequence_length, model_type):
    if model_type=="classic":
        model = CNN1D(CHANNEL_NB, num_activities, input_sequence_length)

    elif model_type=="extended":
        model = CNN1D_extended(CHANNEL_NB, num_activities, input_sequence_length)
    
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for i, (batch_data, batch_labels) in enumerate(train_loader):
            # Zero the gradients, forward pass, and compute loss
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels.to(torch.int64))

            # Backward pass and update parameters
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        train_accuracy = 100 * correct / total
        print('Epoch %d, average loss: %.3f, train accuracy: %.2f%%' % (epoch+1, total_loss/len(train_loader), train_accuracy))
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (batch_data, batch_labels) in enumerate(test_loader):
                outputs = model(batch_data)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            test_accuracy = 100 * correct / total
            print('Epoch %d, test accuracy: %.2f%%' % (epoch+1, test_accuracy))


    return model

class CNN1D_extended(nn.Module):
    def __init__(self, in_channels, out_classes, input_sequence_length):
        super(CNN1D_extended, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)


        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc1_input_size = 32 * input_sequence_length  #75 for shoaib, 192 for pamap
        
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, out_classes)

    def forward(self, x):
        # x has shape (batch_size, in_channels, sequence_length)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def augment_data(channel_cluster_data, X_train, y_train, labels_to_aug, channels_to_aug):
    
    n_windows, n_c, n_data = X_train.shape
    X_train_aug = copy.deepcopy(X_train)
    y_train_aug = copy.deepcopy(y_train)
    
    dist = 'euclidean'
    
    for c in channels_to_aug:
        
        curr_channels_data = channel_cluster_data[c]
        curr_centroids = curr_channels_data.centroids
        curr_rep_windows = curr_channels_data.rep_windows[dist]
        curr_norm_data = curr_channels_data.normalization_param

        for l in labels_to_aug[c]:
            # find all indices with desired label
            curr_label_idx = np.where(y_train == l)[0]


            # initialize structure for storing results
            
            for w in curr_label_idx:
                window_data = copy.deepcopy(X_train[w,:,:])
                
                candidate_data = window_data[c, :]
                candidate_data = candidate_data.reshape(1, n_data)
                
                cluster, dis = vq(candidate_data/curr_norm_data, curr_centroids)
                
                # find cluster for current window
                window_data[c, :] = curr_rep_windows[cluster[0]]
                
                
                y_train_aug = np.append(y_train_aug, l)
                X_train_aug = np.vstack((X_train_aug, window_data.reshape(1, n_c, n_data)))
    
    return X_train_aug, y_train_aug



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
import matplotlib.pyplot as plt
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
from scipy.cluster.vq import kmeans, vq
#from TargetCnn import targetModel_cnn
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


def addlabels(x,y,h):
    for i in range(len(x)):
        #plt.text(i,y[i],y[i], ha = 'center')
        plt.text(i, y[i], y[i], ha = h)


def prepare_data(feature_matrix, label_vector,Label, test_split, bs,train_indices,test_indices):
    dataset = accel_dataset(feature_matrix, label_vector,Label)
    train, test = dataset.get_splits(test_split)
    train.indices = train_indices
    test.indices = test_indices

    train_dl = dl.DataLoader(train, batch_size = bs, shuffle=True)
    test_dl = dl.DataLoader(test, batch_size = bs, shuffle=False)
    # train_indices = train
    # test_indices = test

    return train_dl, test_dl

class mlp_dataset(dl.Dataset):
    def __init__(self, X, y):
        self.X = X # input accelerometer data
        self.X = self.X.astype('float32')
        self.y = y # output stretch data that we need
        self.y = self.y.astype(int)

    
    def __len__(self):
        return np.shape(self.X)[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, test_split):
        # determine sizes
        n_data = np.shape(self.X)[0]
        test_size = round(test_split * n_data)
        train_size = len(self.X) - test_size
        # calculate the split
        return dl.random_split(self, [train_size, test_size])
    
    

def prepare_data_mlp(feature_matrix, label_vector, test_split, bs,train_indices,test_indices):
    dataset = mlp_dataset(feature_matrix, label_vector)
    train, test = dataset.get_splits(test_split)
    train.indices = train_indices
    test.indices = test_indices

    train_dl = dl.DataLoader(train, batch_size = bs, shuffle=True)
    test_dl = dl.DataLoader(test, batch_size = bs, shuffle=False)
    # train_indices = train
    # test_indices = test

    return train_dl, test_dl 


def normalize_features(feature_matrix, feature_mean, feature_variance, norm_idx):
    
    feature_matrix_normalized = feature_matrix
    norm_fp = np.subtract(feature_matrix[:,0:norm_idx], feature_mean[:,0:norm_idx])
    norm_fp_var = np.divide(norm_fp, feature_variance[:,0:norm_idx])
    feature_matrix_normalized[:,0:norm_idx] = norm_fp_var
    return feature_matrix_normalized


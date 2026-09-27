"""
Created on Mon Feb  5 23:01:25 2024

@author: dina.hussein
"""
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
import seg_layout  # corrected triaxial segment layout (axis-major -> time-major)
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
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from tqdm import tqdm
device = torch.device('cpu')
from NN_functions import *
from utility_functions import *
from EENN_functions import * 
from Alex_Net_functions import *
#from AlexNetBasline import *

import time


# changed the way we load params, so I can pass params to this from run_experiments: 

def run_experiment(params):
    #print("made it to run_exp")
    
    # Unpack parameters
    dataset_name = params['dataset_name']
    model_type = params['model']
    is_training = params['is_training']
    num_epochs = params['num_epochs']
    num_exits = params['num_exits']
    thresholds = params.get('threshold', [])# Set default if not provided
    exit_placement = params.get('exit_placement', [])
    loss_weights = params.get('loss_weights', [])

    if model_type == "sensorAware":
        new_data_perc = params.get('new_data_perc', [])
    elif model_type == "AlexNetPartial":
        new_data_perc = params.get('new_data_perc', [])
        # new_data_perc = [40,40]
        # num_exits = 2
        # thresholds = [0.5, 0.5]
        # exit_placement = (2,4)
        # loss_weights = (1,1,1)
    

    # Construct paths based on the dataset name
    # exist_ok: parallel sweep workers create these at the same time
    output_path = os.path.join(dataset_name + "_outputs")
    os.makedirs(output_path, exist_ok=True)
    model_file_path = os.path.join(output_path, "models")
    os.makedirs(model_file_path, exist_ok=True)
    
    file_path = f'Datasets/{dataset_name}_dataLabels.pkl'
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    data = data_dict['data']
    labels_array = data_dict['labels']

    file_path = f'Datasets/{dataset_name}_specs.pkl'
    with open(file_path, 'rb') as file:
        dataSpecs = pickle.load(file)
    
    SEG_SIZE = dataSpecs['SEG_SIZE']
    CHANNEL_NB = dataSpecs['CHANNEL_NB']
    CLASS_NB = dataSpecs['CLASS_NB']
    n_window, n_channel, n_data = data.shape
    unique_numbers_set = set(labels_array)
    num_activities = len(unique_numbers_set)
    
    
    if dataset_name == "PAMAP2":
        input_sequence_length = 192 #98,97
        
        Aug_index = labels_array[(labels_array == 3)] #Augmenting R windows 
        aug_data = data[(labels_array == 3),:]
        labels_array = np.append(labels_array,Aug_index)
        data = np.append(data,aug_data,0)
        
        m,n = data.shape[::2]
        n_window, n_channel, n_data = data.shape
        
    elif dataset_name == "SelfRegulationSCP1":
        input_sequence_length = 112
        
    elif dataset_name == "Shoaib":
        if model_type=="classic":
            input_sequence_length = 75
        else:
            input_sequence_length = 72 #95, 95

        
    elif dataset_name == "ERing":
        input_sequence_length = 8 #90,90
        
    elif dataset_name == "Epilepsy": #w = 275, channels = 3, n_data = 206, classes = 4, classifier tr 90.91,te 94.55, clust:2,3,4
        if model_type=="classic":
            input_sequence_length = 25
        else:
            input_sequence_length = 24 #83,93
        
    elif dataset_name == "WESADchest": #w= 7421, channels = 5, n_data = 200, classes = 3 ,classifier tr 94,te 98, clust:4,6,8,10
        if model_type=="classic":
            input_sequence_length = 25
        else:
            input_sequence_length = 24 #97,98
    
    elif dataset_name == "EMGPhysical": #w= 782, channels = 8, n_data = 200, classes = 4 ,classifier tr 93,te 77
        if model_type=="classic":
            input_sequence_length = 25
        else:
            input_sequence_length = 24 #96,89
    
    save_on = is_training
    lst = list(range(0, n_window))
    X_train_ind, X_test_ind, l, ll = train_test_split(lst, labels_array, test_size = 0.40, random_state = 0)
    X_train = data[X_train_ind,:,:]
    X_test =  data[X_test_ind,:,:]
    y_train = labels_array[X_train_ind]
    y_test = labels_array[X_test_ind]
    # CHANGES MADE: COMMENT OUT THIS VALUE
    # num_epochs = 14 #ering15, emg,epilpsey 14
    
    # === CORRECTED TRIAXIAL SEGMENT LAYOUT (fix 2026-07-03) ===
    # Raw storage for the triaxial datasets (Shoaib, PAMAP2) is AXIS-MAJOR: each
    # sensor row packs [x_1..x_T, y_1..y_T, z_1..z_T] along the SEG_SIZE axis.
    # seg_layout.temporal_view regroups to real time on the last axis with the 3
    # axes promoted to channels, so the partial-sensing prefix slice
    # (extract_new_data -> data[:, :, :k]) reads early TIMESTEPS across ALL sensor
    # axes instead of a prefix of a single axis. No-op for non-triaxial datasets
    # (SEG_SIZE % 3 != 0 -> seg_layout returns the plain time-major swapaxes).
    data = seg_layout.temporal_view(data, dataset_name, verbose=False).transpose(0, 2, 1)  # (N, C*3, T) triaxial; (N, C, SEG) otherwise
    n_window, n_channel, n_data = data.shape
    # ===========================================================
    
    train_features, test_features = prepare_data_mlp(data, labels_array, 0.4, 32, X_train_ind, X_test_ind )
    
    file_name_model = f'{dataset_name}_1DCNN_{model_type}.ckpt'
    model_save_path = os.path.join(model_file_path, file_name_model)
# Goes in training mode 
    if (save_on == 1):
        
        if model_type=="classic":
        
            cnn_model = CNN_train(train_features, test_features, device, num_epochs, CHANNEL_NB, num_activities, input_sequence_length, model_type)
            torch.save(cnn_model.state_dict(),model_save_path)
        
        elif model_type=="extended":
            cnn_model = CNN_train(train_features, test_features, device, num_epochs, CHANNEL_NB, num_activities, input_sequence_length, model_type)
            torch.save(cnn_model.state_dict(),model_save_path)
            
     
        elif model_type=="EarlyExit":


            model = CNN1D_extended_EENN(n_channel, num_activities,
                                        input_sequence_length, thresholds,
                                        num_exits, is_training, n_data, exit_placement)
            
            
            # Loading the weights of the extended model to use for the early exit model
            # we set strict=False to load only the matching layers since EE model has extra parameters
            # extended_model = f'{dataset_name}_1DCNN_extended.ckpt'
            # saved_extended_path = os.path.join(model_file_path, extended_model)
            # pretrained_state_dict = torch.load(saved_extended_path)
            # model.load_state_dict(pretrained_state_dict, strict=False)

            # CHANGE MADE: added a way to save the results
            train_results = model.train_model(train_features, test_features,
                           loss_weights, num_epochs)         

            #torch.save(model.state_dict(),model_save_path)
            # keep only the final epoch's values
            final_epoch_results = train_results[-1]
            #torch.save(model.state_dict(),model_save_path)
            # switch to inference mode
            is_train = 0 
           

        elif model_type=="sensorAware":
            
            # thresholds = [0.90, 0.90, 0.9]  # Example thresholds
            # loss_weights = [1, 1, 1, 1]
            # exit_placement = [1, 2, 3]
            # new_data_perc = [10, 20, 30]
            
            # is_training = True
            
            # num_exits = 3
            
            
            model = CNN1D_extended_EENN_partialsampling(n_channel, num_activities,
                                        input_sequence_length, thresholds,
                                        num_exits, is_training, n_data, exit_placement, new_data_perc)
            pytorch_total_params = sum(p.numel() for p in model.parameters())

            train_results = model.train_model(train_features, test_features,
                           loss_weights, num_epochs)  
            
            #torch.save(model.state_dict(),model_save_path)
            if train_results == -1:
                return -1,(-1,-1)
            else:
                final_epoch_results = train_results[-1]
            is_train = 0 


        elif model_type=="AlexNetEENN":

            model = AlexNetEENN(n_channel, num_activities, input_sequence_length, thresholds, num_exits, is_training, n_data, exit_placement)
            
         
            train_results = model.train_model(train_features, test_features,
                           loss_weights, num_epochs)         

            #torch.save(model.state_dict(),model_save_path)
            # keep only the final epoch's values
            final_epoch_results = train_results[-1]
            #torch.save(model.state_dict(),model_save_path)
            # switch to inference mode
            
            is_train = 0 
            
            
        elif model_type=="AlexNetPartial":


            model = AlexNetPartial(n_channel, num_activities,
                                        input_sequence_length, thresholds,
                                        num_exits, is_training, n_data, exit_placement, new_data_perc)
            
           
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            
            if pytorch_total_params == 0:
                return -1,(-1,-1)

            train_results = model.train_model(train_features, test_features,
                           loss_weights, num_epochs)  
            
            #torch.save(model.state_dict(),model_save_path)
            if train_results == -1:
                return -1,(-1,-1)
            else:
                final_epoch_results = train_results[-1]
            is_train = 0 
        
        
        elif model_type=="AlexNetBaseline":
            model = AlexNetBaseline(n_channel, num_activities, input_sequence_length)

            accuracy = AlexNetBaseline.train_and_evaluate_model( model,train_features, test_features, num_epochs=10,  lr=0.001)
            print(f"Final Test Accuracy: {accuracy:.2f}%")


#NOTE MAJOR ADJUSTMENT : Instead of having else for train or not, we want to regradless, for the experiment purposes run  this to get data. So id we train, 
# We still want to run this 
# Goes into infer
    is_training = 0
    if model_type=="classic":

        cnn_model = CNN1D(CHANNEL_NB, num_activities, input_sequence_length)
        cnn_model.load_state_dict(torch.load(model_save_path))
        cnn_model.eval()
        print("loaded model")

    elif model_type=="extended":
        cnn_model = CNN1D_extended(CHANNEL_NB, num_activities, input_sequence_length)
        cnn_model.load_state_dict(torch.load(model_save_path))
        cnn_model.eval()
        print("loaded model")

 # CHANGES MADE: you can add a simple if and else for returning a statement. If its not training it will simply return whatever infrence
 # has, if it is training, it will return train_model_results here, and these inference values. The way I unpach
 # these values here AKA train_results, ( inference_test_results, train_test_track) is how we are goinf to unpack them in 
 # run experiments

    elif model_type=="EarlyExit":
        #we will load the early exit network here
        # re-initializing the model here would discard the trained weights, so it is not done
        #cnn_model = CNN1D_extended_EENN(n_channel, num_activities,
         #                           input_sequence_length, thresholds,
          #                          num_exits, is_training, n_data, exit_placement)
        model.is_training = 0
        inference_test_results, train_test_track = model.inference(train_features, test_features)
        
        
        return final_epoch_results, (inference_test_results, train_test_track)

    elif model_type=="sensorAware":
        #we will load the early exit + partial sampling  here

        #cnn_model = CNN1D_extended_EENN_partialsampling(n_channel, num_activities,
                                     #input_sequence_length, thresholds,
                                    # num_exits, is_training, n_data, exit_placement, new_data_perc)
        # cnn_model.load_state_dict(torch.load(model_save_path)) 
        # we want to return this list! 
        model.is_training = 0
        inference_test_results, train_test_track = model.inference(train_features, test_features)

        return final_epoch_results, (inference_test_results, train_test_track)
    
    
    elif model_type=="AlexNetEENN":
       
        model.is_training = 0
        inference_test_results, train_test_track = model.inference(train_features, test_features)
        
        return final_epoch_results, (inference_test_results, train_test_track)


    elif model_type=="AlexNetPartial":
        model.is_training = 0
        inference_test_results, train_test_track = model.inference(train_features, test_features)
        return final_epoch_results, (inference_test_results, train_test_track)


    elif model_type=="AlexNetBaseline":
        model = AlexNetBaseline(n_channel, num_activities, input_sequence_length)

        accuracy = AlexNetBaseline.train_and_evaluate_model( model,train_features, test_features, num_epochs=10,  lr=0.001)
        print(f"Final Test Accuracy: {accuracy:.2f}%")


  
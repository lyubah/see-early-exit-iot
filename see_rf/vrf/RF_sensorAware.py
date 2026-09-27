import argparse
import numpy as np
import statistics
from numpy.fft import fft
import math
import csv
from scipy import signal
from scipy.fft import fftshift
from numpy import savetxt
from collections import Counter
import itertools
import os
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
import matplotlib

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np
import sys
from random import randint
import copy
import numpy as np
np.random.seed(123)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utility_functions import *
from collections import OrderedDict
import time
import joblib


def main(args):
    dataset_name = args.dataset_name
    model_type = args.model_type
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
        Aug_index = labels_array[(labels_array == 3)] #Augmenting R windows 
        aug_data = data[(labels_array == 3),:]
        labels_array = np.append(labels_array,Aug_index)
        data = np.append(data,aug_data,0)
        
        m,n = data.shape[::2]
        n_window, n_channel, n_data = data.shape
        n_trees = 85
    elif args.dataset_name == "SelfRegulationSCP1":
        n_trees = 10

    elif args.dataset_name == "Shoaib":
        n_trees = 100
        
    elif args.dataset_name == "Epilepsy": 
        n_trees = 95

    elif args.dataset_name == "WESADchest": 
        n_trees = 1

    elif args.dataset_name == "EMGPhysical":
        n_trees = 50


        
    
    lst = list(range(0, n_window))

    X_train_ind, X_test_ind, l, ll = train_test_split(lst, labels_array, test_size = 0.40, random_state = 0)
    X_train = data[X_train_ind,:,:]
    X_test =  data[X_test_ind,:,:]
    y_train = labels_array[X_train_ind]
    y_test = labels_array[X_test_ind]

    
    
    if model_type=="classic":
        input_data_RF_train = copy.deepcopy(X_train)
        train_flatten = input_data_RF_train.reshape(len(y_train), n_channel*n_data)
        
        input_data_RF_test = copy.deepcopy(X_test)
        test_flatten = input_data_RF_test.reshape(len(y_test), n_channel*n_data)
        
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        clf.fit(train_flatten, y_train)
        n_params = sum(tree.tree_.node_count for tree in clf.estimators_) * 5
        
        # Save the model to a file
        joblib.dump(clf, f"{dataset_name}_RF_{model_type}_model.joblib")
        
        
        # Make predictions
        y_pred = clf.predict(test_flatten)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy*100)
        
    elif  model_type=="RF_sensorAware":
        flog_clusters = open(f"{dataset_name}_RF_sensorAware_classification.csv","w", newline='')
        flog_writer = csv.writer(flog_clusters)
        csv_header =['Portion', 'Model_Size','Accuracy', 'Classic_Acc', 'Org_n_trees']
        flog_writer.writerow(csv_header)
        
        input_data_RF_train = copy.deepcopy(X_train)
        train_flatten = input_data_RF_train.reshape(len(y_train), n_channel*n_data)
        
        input_data_RF_test = copy.deepcopy(X_test)
        test_flatten = input_data_RF_test.reshape(len(y_test), n_channel*n_data)
        
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        clf.fit(train_flatten, y_train)

        
        y_pred = clf.predict(test_flatten)
        acc_default = accuracy_score(y_test, y_pred)
        print("Classic Accuracy:", acc_default*100)

        Model_sizes = [i * 20 for i in range(1, int(n_trees / 20) + 1)]
        
        proportions = [0.2, 0.5,0.6, 0.7, 0.8, 1.0]
        

        for proportion in proportions:
            for k in range (0,len(Model_sizes)):
                # subset of the training data according to the proportion
                num_samples = int(train_flatten.shape[1] * proportion)
                train_subset = train_flatten[:,:num_samples]
                test_subset = test_flatten[:,:num_samples]
            
                # Train a random forest classifier on the subset
                if Model_sizes:
                    clf = RandomForestClassifier(n_estimators=Model_sizes[k], random_state=42)
                else: 
                    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    
                clf.fit(train_subset, y_train)
                joblib.dump(clf, f"{dataset_name}_RF_{model_type}_model_in{proportion*100}_size{Model_sizes[k]}.joblib")

                # Make predictions on the test data
                y_pred = clf.predict(test_subset)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                if Model_sizes:
    
                    print(f"Accuracy with {proportion * 100}% of the data, model size {Model_sizes[k]}: {accuracy*100}")
                    curr_row = ((proportion* 100,)+ (Model_sizes[k],)+ (accuracy*100,)+ (acc_default*100,)+(n_trees,))
                    flog_writer.writerow(curr_row)

                else:
                    print(f"Accuracy with {proportion * 100}% of the data, model size {n_trees}: {accuracy*100}")
                    curr_row = ((proportion* 100,)+ (n_trees,)+ (accuracy*100,)+ (acc_default*100,)+(n_trees,))
                    flog_writer.writerow(curr_row)

    
                # # Check if accuracy meets the threshold
                # if accuracy >= (acc_default-0.1):
                #     print("Achieved desired accuracy. No need for more data.")
                #     break
        z= 0
    
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Epilepsy', required=False)
    parser.add_argument('--model_type', type=str, help="model", required=False, default="classic")
    args = parser.parse_args()
    main(args)
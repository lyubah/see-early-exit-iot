import os
# Disable CUDA before importing torch to avoid library loading errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from tqdm import tqdm
from NN_functions import *
from utility_functions import *
from EENN_functions import * 
from collections import OrderedDict
import time
import concurrent.futures
device = torch.device('cpu')

torch.set_num_threads(1)

def max_entropy(num_classes, first_class_prob=0.7):
    
    probabilities = [first_class_prob] + [(1 - first_class_prob) / (num_classes - 1)] * (num_classes - 1)
    
    entropy_sum = sum(p * math.log(p) for p in probabilities)
    max_entropy = - entropy_sum
    
    return max_entropy


def func_threshold_combinations(num_exits, dataset_max_entropy):
    # # Create array of thresholds from 0.2 to dataset_max_entropy with increment of 0.2
    # thresholds = [i * 0.2 for i in range(1, int(dataset_max_entropy / 0.2) + 1)]
    num_bins = 5
    bin_width = dataset_max_entropy / num_bins
    thresholds = [(i + 1) * bin_width for i in range(num_bins)]
    
    combinations_set = set() 
    combinations = []
    for combo in itertools.product(thresholds, repeat=num_exits):
        # Ensure the first exit value is the maximum of all
        combo_list = list(combo)
        max_threshold = max(combo_list)
        combo_list[0] = max_threshold
        combinations_set.add(tuple(combo_list)) 
        #combinations.append(combo_list)
    combinations = [list(combo) for combo in combinations_set]
    
    return combinations

def get_percentage_accuracy(actual_labels, output_labels):
    correct_predictions = (actual_labels == output_labels).sum()
    total_predictions = len(actual_labels)
    accuracy_percentage = (correct_predictions / total_predictions) * 100
    return float(np.asarray(accuracy_percentage).ravel()[0])  # DataFrame inputs give a 1-element Series

def process_combination(combination, test_df, dataset_max_entropy, num_layers, model_type, columns_to_select, n_window):
    
    if model_type=="sensorAware" or model_type == "AlexNetPartial":
        matching_rows_df = test_df.loc[(test_df[columns_to_select[3]] == combination[columns_to_select[3]]) &
                                            (test_df[columns_to_select[4]] == combination[columns_to_select[4]]) &
                                            (test_df[columns_to_select[5]] == combination[columns_to_select[5]]) &
                                            (test_df[columns_to_select[6]] == combination[columns_to_select[6]]) &
                                            (test_df[columns_to_select[7]] == combination[columns_to_select[7]]) &
                                            (test_df[columns_to_select[8]] == combination[columns_to_select[8]]) &
                                            (test_df[columns_to_select[9]] == combination[columns_to_select[9]]) &
                                            (test_df[columns_to_select[10]] == combination[columns_to_select[10]]) &
                                            (test_df[columns_to_select[11]] == combination[columns_to_select[11]]) &
                                            (test_df[columns_to_select[12]] == combination[columns_to_select[12]]) &
                                            (test_df[columns_to_select[13]] == combination[columns_to_select[13]]) &
                                            (test_df[columns_to_select[14]] == combination[columns_to_select[14]]) &
                                            (test_df[columns_to_select[15]] == combination[columns_to_select[15]]) &
                                            (test_df[columns_to_select[16]] == combination[columns_to_select[16]]) &
                                            (test_df[columns_to_select[17]] == combination[columns_to_select[17]])]
    else:
        matching_rows_df = test_df.loc[(test_df[columns_to_select[3]] == combination[columns_to_select[3]]) &
                                                (test_df[columns_to_select[4]] == combination[columns_to_select[4]]) &
                                                (test_df[columns_to_select[5]] == combination[columns_to_select[5]]) &
                                                (test_df[columns_to_select[6]] == combination[columns_to_select[6]]) &
                                                (test_df[columns_to_select[7]] == combination[columns_to_select[7]]) &
                                                (test_df[columns_to_select[8]] == combination[columns_to_select[8]]) &
                                                (test_df[columns_to_select[9]] == combination[columns_to_select[9]]) &
                                                (test_df[columns_to_select[10]] == combination[columns_to_select[10]]) &
                                                (test_df[columns_to_select[11]] == combination[columns_to_select[11]]) &
                                                (test_df[columns_to_select[12]] == combination[columns_to_select[12]])]

    matching_rows_df.reset_index(drop=True, inplace=True)
    # Some configs in the train CSV have no matching per-window rows in the test CSV
    # (e.g. a config that failed to build or train); skip them instead of crashing on .iloc[0].
    if matching_rows_df.empty:
        return []

    exit_numbers_array = np.full(len(matching_rows_df), num_layers, dtype=int)
    exit_threshold_array = np.full(len(matching_rows_df), -1, dtype=float)


    threshold_combinations = func_threshold_combinations(matching_rows_df['Num_exits'].iloc[0], dataset_max_entropy)
    number_exits = matching_rows_df['Num_exits'].iloc[0]
    
    comb_copy = copy.deepcopy(combination)
    #print (number_exits)
    output_data_config = []
    
    # for combin in threshold_combinations:
    #     odata = process_threshold_combination(combination, combin, matching_rows_df, dataset_max_entropy, 
    #                                           num_layers, model_type,comb_copy, number_exits, n_window)
    #     output_data_config.extend(odata)

    
    # Threads, not processes: a process pool per config forks inside the outer thread pool
    # (deadlock-prone) and, where workers are spawned (macOS/Windows), re-imports torch in
    # every worker. Each task gets its own copy of comb_copy because
    # process_threshold_combination writes that task's thresholds into it.
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures_result = {executor.submit(process_threshold_combination, combination, combin, matching_rows_df, 
                                           dataset_max_entropy, num_layers, model_type, copy.deepcopy(comb_copy),
                                           number_exits, n_window)
                   for combin in threshold_combinations}

        for future in concurrent.futures.as_completed(futures_result):
            output_data_config.extend(future.result())
    return output_data_config
        
def process_threshold_combination(combination, combin, matching_rows_df, dataset_max_entropy, num_layers, model_type, comb_copy, number_exits, n_window):
        
    #for combin in threshold_combinations:
    exit_numbers_array = np.full(len(matching_rows_df), num_layers, dtype=int)
    exit_threshold_array = np.full(len(matching_rows_df), -1, dtype=float)

    exit_used = []
    for i in range(1, num_layers):
        column_name = "Exit {} layer".format(i)
        if column_name in matching_rows_df.columns:
            if (matching_rows_df[column_name].iloc[0] == i).any():
                exit_used.append(i)
    for ind in range(0,len(combin)):
        #if matching_rows_df['Exit {ind} confidence'] <= combin[ind]:
        
        column_name = 'Exit ' + str(exit_used[ind]) + ' confidence'
        col_name_threshold = 'Exit_' + str(exit_used[ind]) + '_threshold'
        comb_copy[col_name_threshold] = combin[ind]
        exit_taken = matching_rows_df[matching_rows_df[column_name] <= combin[ind]]
        if not exit_taken.empty:
            for index in exit_taken.index:
                if exit_numbers_array[index] == num_layers:
                    exit_numbers_array[index] = exit_used[ind]
                    exit_threshold_array[index] = combin[ind]
                    
                    
        output_arr = np.full(len(matching_rows_df), 0, dtype=float)
        train_output_arr = []
        test_output_arr = []
        train_actual_arr = []
        test_actual_arr = []

        
        for index, exit_num in enumerate(exit_numbers_array):
            column_name = 'Exit ' + str(exit_num) + ' output'
            output_arr[index] = matching_rows_df[column_name].iloc[index]
            if matching_rows_df['is_train'].iloc[index] == 1:
                train_output_arr.append(matching_rows_df[column_name].iloc[index])
                train_actual_arr.append(matching_rows_df['label'].iloc[index])
            else: 
                test_output_arr.append(matching_rows_df[column_name].iloc[index])
                test_actual_arr.append(matching_rows_df['label'].iloc[index])
                

        Accuracy_configuration = get_percentage_accuracy(matching_rows_df['label'], output_arr)
        Train_acc = get_percentage_accuracy(pd.DataFrame(train_actual_arr), pd.DataFrame(train_output_arr))
        Test_acc = get_percentage_accuracy(pd.DataFrame(test_actual_arr), pd.DataFrame(test_output_arr))
        Total_energy = 0
        for index, exit_num in enumerate(exit_numbers_array):
            output_row = OrderedDict()
            output_row.update(combination)
            output_row['Number_exits'] = number_exits
            output_row['Exit_taken'] = exit_num
            output_row['Threshold'] = exit_threshold_array[index]
            column_name = 'Exit_' + str(exit_num) + '_percentage'
            if model_type=="sensorAware" or model_type=="AlexNetPartial" :
                output_row['Percentage'] = matching_rows_df[column_name].iloc[index]
                energy = matching_rows_df[column_name].iloc[index]/100
                output_row['Energy'] = energy
                Total_energy = Total_energy + energy

            
            output_row['Output'] = output_arr[index]
            output_row['Actual'] = matching_rows_df['label'].iloc[index]

            output_row['Total_acc_configuration'] = Accuracy_configuration
            output_row['Train_acc_configuration'] = float(Train_acc)
            output_row['Test_acc_configuration'] = float(Test_acc)
            output_row['Is_train'] = matching_rows_df['is_train'].iloc[index]


            #output_data.append(output_row)
        output_data_config = []

        output_row_config = OrderedDict()
        output_row_config.update(comb_copy)
        output_row_config['Number_exits'] = number_exits
        output_row_config['Total_acc_configuration'] = Accuracy_configuration
        output_row_config['Train_acc_configuration'] = float(Train_acc)
        output_row_config['Test_acc_configuration'] = float(Test_acc)
        if model_type=="sensorAware" or model_type=="AlexNetPartial" :
            output_row_config['Total_E_SensorAware'] = Total_energy
            output_row_config['Total_E_Default'] = n_window
            output_row_config['E_ratio'] = Total_energy / n_window
            output_row_config['Number_windows'] = n_window

        output_data_config.append(output_row_config)
        
    return output_data_config
def main(args):
    dataset_name = args.dataset_name
    num_epochs = args.num_epochs
    model_type = args.model_type
    results_dir = args.results_dir
    
    # Check if train_file and test_file are provided directly
    if hasattr(args, 'train_file') and args.train_file and hasattr(args, 'test_file') and args.test_file:
        train_results_path = args.train_file
        test_results_path = args.test_file
    else:
        # Use the old method of constructing filenames
#Look at git version of files names and adjust for that     
        if model_type=="EarlyExit":
            train_results_filename = "train_accuracy_dataset_{}_epoch_{}__max_exits3_EENN_parallel_batchMarch19.csv".format(dataset_name, num_epochs)
            test_results_filename = "test_accuracy_dataset_{}_epoch_{}__max_exits3_EENN_parallel_batchMarch19.csv".format(dataset_name, num_epochs)


        elif model_type=="sensorAware":
            train_results_filename = "train_accuracy_dataset_{}_epoch_{}__max_exits3_SEENN_multi_parallel_batches_Nov27.csv".format(dataset_name, num_epochs)
            test_results_filename = "test_accuracy_dataset_{}_epoch_{}__max_exits3_SEENN_multi_parallel_batches_Nov27.csv".format(dataset_name, num_epochs)
            
        elif model_type == "AlexNetPartial":
            train_results_filename = "train_accuracy_dataset_{}_epoch_{}__max_exits3_AlexNet_SEENN_multi_parallel_batches_March17.csv".format(dataset_name, num_epochs)
            test_results_filename = "test_accuracy_dataset_{}_epoch_{}__max_exits3_AlexNet_SEENN_multi_parallel_batches_March17.csv".format(dataset_name, num_epochs)
            
            
        elif model_type == "AlexNetEENN":
            train_results_filename = "train_accuracy_dataset_{}_epoch_{}__max_exits3_AlexEENN_parallel_batch.csv".format(dataset_name, num_epochs)
            test_results_filename = "test_accuracy_dataset_{}_epoch_{}__max_exits3_AlexEENN_parallel_batch.csv".format(dataset_name, num_epochs)

        # Construct full file paths
        train_results_path = os.path.join(results_dir, train_results_filename)
        test_results_path = os.path.join(results_dir, test_results_filename)
    
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
    
    dataset_max_entropy = max_entropy(num_activities)
    num_layers = 5
        
    train_df = pd.read_csv(train_results_path)
    test_df = pd.read_csv(test_results_path)
    
    columns_to_exclude = [f"Exit_{i}_Final_training_accuracy" for i in range(1, num_layers+1)] + \
                     [f"Exit_{i}_Final_test_accuracy" for i in range(1, num_layers+1)] + \
                     ["Final_training_accuracy", "Final_test_accuracy"]
    
    columns_to_select = [col for col in train_df.columns if col not in columns_to_exclude]
    unique_train_combinations = train_df[columns_to_select].drop_duplicates()
    
    exit_numbers = range(1, num_layers+1) 
    mapping_dict = {}
    
    first_write = 0
    for exit_num in exit_numbers:
        column_name_train_file = "Exit_{}_loss_weight".format(exit_num)
        column_name_test_file = "Loss_weight_{}".format(exit_num)
        mapping_dict[column_name_test_file] = column_name_train_file
        
        column_name_train_file_threshold = "Exit_{}_threshold".format(exit_num)
        column_name_test_file_threshold = "Exit {} threshold".format(exit_num)
        mapping_dict[column_name_test_file_threshold] = column_name_train_file_threshold


        if model_type=="sensorAware" or model_type=="AlexNetPartial" :
            column_name_train_file_percentage = "Exit_{}_percentage".format(exit_num)
            column_name_test_file_percentage = "Exit {} percentage".format(exit_num)
            mapping_dict[column_name_test_file_percentage] = column_name_train_file_percentage

    
    test_df = test_df.rename(columns=mapping_dict)
    output_data = []
    output_data_config = []
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for k, combination in unique_train_combinations.iterrows():
            futures.append(executor.submit(process_combination, combination, test_df, dataset_max_entropy, num_layers, model_type, columns_to_select, n_window))

        for future in concurrent.futures.as_completed(futures):
            output_data_config.extend(future.result())

    file_name_config = f"{dataset_name}_output_config_analysis_{model_type}_epochs_{num_epochs}_ends.csv"
    output_config_df = pd.DataFrame(output_data_config)
    output_config_df.to_csv(file_name_config, index=False)


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str, help="Dataset name", required=False, default="Epilepsy")
  parser.add_argument('--model_type', type=str, help="model", required=False, default="EarlyExit")
  parser.add_argument('--num_epochs', type=int, default=20)
  parser.add_argument('--results_dir', type=str, help="Directory containing results CSV files", required=False, default="results/combined")
  parser.add_argument('--train_file', type=str, help="Direct path to train results CSV file (overrides results_dir and filename construction)", required=False, default=None)
  parser.add_argument('--test_file', type=str, help="Direct path to test results CSV file (overrides results_dir and filename construction)", required=False, default=None)
  args = parser.parse_args()
  
  start_time = time.time()
  main(args)
  end_time = time.time()
  total_time = end_time - start_time
  print('Total execution time is', total_time, 'seconds' )
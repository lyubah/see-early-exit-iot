import concurrent.futures
import itertools
import csv
import torch
from eval_config import run_and_record_experiment
import pandas as pd
import numpy as np
import argparse
import logging
import pprint
import ast
import time
import datetime
import math
import itertools


pp = pprint.PrettyPrinter(indent=2)

device = torch.device('cpu')
torch.set_num_threads(1)


def max_entropy(num_classes, first_class_prob=0.7):
    
    probabilities = [first_class_prob] + [(1 - first_class_prob) / (num_classes - 1)] * (num_classes - 1)
    
    entropy_sum = sum(p * math.log(p) for p in probabilities)
    max_entropy = - entropy_sum
    
    return max_entropy


def func_threshold_combinations(num_exits, dataset_max_entropy):
    # print("made it to func combo")
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


def generate_all_configurations(dataset_name):
  exits = [3,4] # we will work with 3 exits or 4 exits
  depths = [100, 200, 500, 800]
  l=[]
  for depth in depths:
    for num_exits in exits:
        if dataset_name == 'Shoaib':
            n_classes = 7
        if dataset_name == 'Epilepsy':
            n_classes = 4
        if dataset_name == 'EMGPhysical':
            n_classes = 4
        if dataset_name == 'SelfRegulationSCP1':
            n_classes = 2
        if dataset_name == 'WESADchest':
            n_classes = 3
        if dataset_name == 'PAMAP2':
            n_classes = 5
        if dataset_name == 'AI4I':
            n_classes = 2
        if dataset_name == 'OCCUPANCY':
            n_classes = 2


        entropy_max = max_entropy(n_classes)  
        th_combinations = func_threshold_combinations(num_exits, entropy_max)
    
        if num_exits == 4:
            proportions = [0.2, 0.6, 0.8, 1, ]
            tree_splits = [0.5, 0.7, 0.85, 1]

        if num_exits == 3:
            proportions = [0.5, 0.8, 1]
            tree_splits = [0.57, 0.8, 1]
        

        dictionary = {'Dataset': dataset_name, 'Num_exits': num_exits, 'Tree_depth': depth, 'Tree_splits': tree_splits, 'Proportions': proportions, 'th': th_combinations}
        l.append(dictionary)

  return l


# The function that will prepare the header
def prepare_header(num_exits):
    header = ['dataset', 'num_exits', 'max_depth', 'tree splits', 'data percentages']


    # Adding accuracy for Train, Validation, and Test (at each exit)
    for i in range(num_exits):
        header.append(f'T_acc_{i+1}')

    for i in range(num_exits):
        header.append(f'V_acc_{i+1}')

    for i in range(num_exits):
        header.append(f'Test_acc_{i+1}')

    # Adding exit thresholds (E_TH)
    for i in range(num_exits):
        header.append(f'E_TH{i+1}')

    # Adding accuracy at each exit
    for i in range(num_exits):
        header.append(f'T_acc_exit_{i+1}')
    for i in range(num_exits):
        header.append(f'V_acc_exit_{i+1}')
    for i in range(num_exits):
        header.append(f'Test_acc_exit_{i+1}')

    # Adding percentage taken at each exit
    for i in range(num_exits):
        header.append(f'T_perc_taken_{i+1}')
    for i in range(num_exits):
        header.append(f'V_perc_taken_{i+1}')
    for i in range(num_exits):
        header.append(f'Test_perc_taken_{i+1}')

    # Adding entire data percentages
    for i in range(num_exits):
        header.append(f'entire_data_perc_taken_{i+1}')
    
    header.append("Total_train_accuracy")
    header.append("Total_validation_accuracy")
    header.append("Total_test_accuracy")

    return header


# the function that will write the results to the csv file
def write_to_csv(config_results, header, file):  

    writer = csv.writer(file)
    for config_result in config_results:
        row = [config_result[key] for key in header]
        writer.writerow(row)


# the parallel code
def parallel_code (csv_file, datasest_name):

    header = prepare_header(4)
    with open(csv_file, mode='w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(header)


      # Generate all configurations upfront
      all_configs = generate_all_configurations(datasest_name)
      
      #ADDED THIS AS TESTING CONDITION
      # all_configs =  all_configs[:2]
      
      logging.info(f"Total configurations generated: {len(all_configs)}")

      # Use ProcessPoolExecutor for CPU-bound tasks

      num_configs = len(all_configs)
      BATCH_SIZE = 8
      num_batches = int(np.ceil(num_configs/BATCH_SIZE))

      for b in range(0, num_batches):
          batch_start = b*BATCH_SIZE
          batch_end = (b+1)*BATCH_SIZE

          if(batch_end > num_configs):
              batch_end = num_configs
          print(time.time(), "Batch ", b, "/", num_batches)

          batch_results = []
          
          with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor: 
              futures = {executor.submit(run_and_record_experiment, config): config for config in all_configs[batch_start:batch_end]} 

              for future in concurrent.futures.as_completed(futures):
                  try:
                      result = future.result()
                      batch_results.append(result)

                  except Exception as e:
                      logging.error(f"Error processing config: {e}")

          # Write in batches to reduce I/O overhead
          for br in batch_results:
            write_to_csv(br, header, f)

          f.flush()


    logging.info("All experiments completed and results written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Epilepsy')
    args = parser.parse_args()
    start_time = time.time()
    parallel_code(f"RF_results_{args.dataset_name}.csv", args.dataset_name)
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")



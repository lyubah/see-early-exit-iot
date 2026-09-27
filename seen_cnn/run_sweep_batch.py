"""
Stage 1 worker: train and evaluate one batch of SEEN sweep configurations.

Reads a JSON list of configs (written by generate_sweep_jobs.py), trains each one with
Sensor_aware_early_exit.run_experiment (8 processes in parallel), and writes two CSVs:
    train_accuracy_dataset_<ds>_..._batch<id>.csv  one row per config (per-exit accuracies)
    test_accuracy_dataset_<ds>_..._batch<id>.csv   one row per window (per-exit output,
                                                   confidence, data %), used by analyze_sweep.py
Run from seen_cnn/ so the relative Datasets/ path resolves.
"""
import concurrent.futures
import itertools
import csv
import torch
import Sensor_aware_early_exit
import pandas as pd
import numpy as np
import argparse
import logging
import pprint
import ast
import time
import datetime
import datetime
import random
import json
import os 


# Setup pretty printer
pp = pprint.PrettyPrinter(indent=2)

device = torch.device('cpu')
torch.set_num_threads(1)


def prepare_headers_train(num_layers):
    header = ['Dataset', 'Model', 'Num_epochs']
    for i in range(1, num_layers + 1):
        header.extend([
            f'Exit_{i}_loss_weight',
            f'Exit_{i}_threshold',
            f'Exit_{i}_percentage',
            f'Exit_{i}_Final_training_accuracy',
            f'Exit_{i}_Final_test_accuracy'
        ])
    header.extend(['Final_training_accuracy', 'Final_test_accuracy'])
    return header

def prepare_headers(exp_params):
    #initial header
    header = ['Dataset', 'Model', 'Num_epochs', 'Num_exits']
    
    # put headers corresponding to the exits
    for i in range(exp_params):
        header.append(f'Loss_weight_{i+1}')
    for i in range(1, exp_params + 1):
        header.extend([
            f'Exit {i} layer',
            f'Exit {i} output',
            f'Exit {i} threshold',
            f'Exit {i} confidence',
            f'Exit {i} is confident',
            f'Exit {i} percentage'
        ])
    header.extend(['is_train', 'label'])
    return header

# -------------------------------
# CSV Writing Functions
# -------------------------------
def write_training_results_to_csv(exp_params, final_training_result, writer, num_layers):
    
    experiment_failed = (
        final_training_result == -1
        or not isinstance(final_training_result, dict)
        or 'train_accuracy' not in final_training_result
        or 'test_accuracy' not in final_training_result
    )

    row = {
        'Dataset': exp_params['dataset_name'],
        'Model': exp_params['model'],
        'Num_epochs': exp_params['num_epochs']
    }
    num_exits = exp_params['num_exits']
    new_data_perc = exp_params.get('new_data_perc', [])
    cumulative_data_used = 0
    sum_new_data = float(np.sum(new_data_perc)) if new_data_perc else 0
    leftover_data = max(0, 100 - sum_new_data)
    
    for i in range(1, num_layers + 1):
        row[f'Exit_{i}_loss_weight'] = -1
        row[f'Exit_{i}_threshold'] = -1
        row[f'Exit_{i}_percentage'] = -1
        row[f'Exit_{i}_Final_training_accuracy'] = -1
        row[f'Exit_{i}_Final_test_accuracy'] = -1
    
    if experiment_failed:
        for exit_idx in range(num_exits):
            exit_layer_id = exp_params['exit_placement'][exit_idx]
            row[f'Exit_{exit_layer_id}_loss_weight'] = exp_params['loss_weights'][exit_idx]
            row[f'Exit_{exit_layer_id}_threshold'] = exp_params['threshold'][exit_idx]
            row[f'Exit_{exit_layer_id}_percentage'] = -50
            row[f'Exit_{exit_layer_id}_Final_training_accuracy'] = -50
            row[f'Exit_{exit_layer_id}_Final_test_accuracy'] = -50

        final_exit_idx = num_layers
        row[f'Exit_{final_exit_idx}_loss_weight'] = exp_params['loss_weights'][-1]
        row[f'Exit_{final_exit_idx}_threshold'] = -1
        row[f'Exit_{final_exit_idx}_percentage'] = -50
        row[f'Exit_{final_exit_idx}_Final_training_accuracy'] = -50
        row[f'Exit_{final_exit_idx}_Final_test_accuracy'] = -50
        row['Final_training_accuracy'] = -50
        row['Final_test_accuracy'] = -50
        writer.writerow(row)
        return

    train_acc_list = final_training_result['train_accuracy']
    test_acc_list = final_training_result['test_accuracy']
    for exit_idx in range(num_exits):
        exit_layer_id = exp_params['exit_placement'][exit_idx]
        row[f'Exit_{exit_layer_id}_loss_weight'] = exp_params['loss_weights'][exit_idx]
        row[f'Exit_{exit_layer_id}_threshold'] = exp_params['threshold'][exit_idx]
        increment = new_data_perc[exit_idx] if exit_idx < len(new_data_perc) else 0
        row[f'Exit_{exit_layer_id}_percentage'] = cumulative_data_used + increment
        cumulative_data_used = row[f'Exit_{exit_layer_id}_percentage']
        row[f'Exit_{exit_layer_id}_Final_training_accuracy'] = train_acc_list[exit_idx]
        row[f'Exit_{exit_layer_id}_Final_test_accuracy'] = test_acc_list[exit_idx]
    
    final_exit_idx = num_layers
    row[f'Exit_{final_exit_idx}_loss_weight'] = exp_params['loss_weights'][-1]
    row[f'Exit_{final_exit_idx}_threshold'] = -1
    row[f'Exit_{final_exit_idx}_percentage'] = cumulative_data_used + leftover_data
    row[f'Exit_{final_exit_idx}_Final_training_accuracy'] = train_acc_list[-1]
    row[f'Exit_{final_exit_idx}_Final_test_accuracy'] = test_acc_list[-1]
    row['Final_training_accuracy'] = train_acc_list[-1]
    row['Final_test_accuracy'] = test_acc_list[-1]
    writer.writerow(row)

def write_results_to_csv(results, train_test_track, exp_params, writer, num_layers):
    
    num_exits = exp_params['num_exits']
    used_exits = set(exp_params['exit_placement'])

    if results == -1 or train_test_track == -1:
        row = {
            'Dataset': exp_params['dataset_name'],
            'Model': exp_params['model'],
            'Num_epochs': exp_params['num_epochs'],
            'Num_exits': exp_params['num_exits']
        }
        for j in range(1, num_layers + 1):
            row[f'Loss_weight_{j}'] = exp_params['loss_weights'][j - 1] if j <= len(exp_params['loss_weights']) else -1
        for exit_idx in range(1, num_layers + 1):
            val = -50 if exit_idx in used_exits else -1
            row[f'Exit {exit_idx} layer'] = val
            row[f'Exit {exit_idx} output'] = val
            row[f'Exit {exit_idx} threshold'] = val
            row[f'Exit {exit_idx} confidence'] = val
            row[f'Exit {exit_idx} is confident'] = val
            row[f'Exit {exit_idx} percentage'] = val
        row['is_train'] = -50
        row['label'] = -50
        writer.writerow(row)
        return
    
    # Iterate through each batch of results
    for batch_index, (batch_results, batch_train_test) in enumerate(zip(results, train_test_track)):
        batch_size = len(batch_train_test['label'])
        
        for i in range(batch_size):
            # Initialize a row for each sample in the batch
            row = {
                'Dataset': exp_params['dataset_name'],
                'Model': exp_params['model'],
                'Num_epochs': exp_params['num_epochs'],
                'Num_exits': exp_params['num_exits'],
            }
            
            # Initialize loss weights to -1
            for j in range(num_layers):
                row[f'Loss_weight_{j+1}'] = -1

            # # Populate loss weights for the exits used in the experiment
            # for exit_index in range(num_exits):
            #     exit_layer_id = exp_params['exit_placement'][exit_index]

            for exit_index in range(num_exits):
                exit_layer_id = exp_params['exit_placement'][exit_index]
                row[f'Loss_weight_{exit_layer_id}'] = exp_params['loss_weights'][exit_index]
           
            # Ensure the final exit always gets the final loss weight
            final_loss_key = f'Loss_weight_{num_layers}'
            row[final_loss_key] = exp_params['loss_weights'][-1]
                

            # Add exit-specific results to the row
            for exit_index, exit_results in enumerate(batch_results):
                row.update({
                    f'Exit {exit_index+1} layer': exit_results['exit_point'][i],
                    f'Exit {exit_index+1} output': exit_results['output'][i],
                    f'Exit {exit_index+1} threshold': exit_results['threshold_value'][i],
                    f'Exit {exit_index+1} confidence': exit_results['confidence'][i],
                    f'Exit {exit_index+1} is confident': exit_results['is_confident'][i],
                    f'Exit {exit_index +1} percentage':  exit_results['partial_percentage'][i],

                })

            row['is_train'] = batch_train_test['train_test'][i]
            row['label'] = batch_train_test['label'][i] 
            
            
            # Write the row to the CSV
            writer.writerow(row)


def run_and_record_experiment(exp_config):
    # Exceptions propagate to the caller, which logs them and skips the config.
    final_training_result, (inference_test_results, train_test) = Sensor_aware_early_exit.run_experiment(exp_config)
    return {
        'final_training_result': final_training_result,
        'inference_test_results': inference_test_results,
        'train_test_track': train_test,
        'exp_params': exp_config
    }


def sanitize_config(cfg, default_dataset, default_model):
    # ── guarantees the two keys are present ───────────────────────────
    cfg.setdefault('dataset_name', default_dataset)
    cfg.setdefault('model',        default_model)

    # ── cast everything to the right Python types ─────────────────────
    cfg["exit_placement"] = tuple(int(x) for x in cfg["exit_placement"])
    cfg["loss_weights"]   = tuple(int(x) for x in cfg["loss_weights"])
    cfg["is_training"]    = bool(cfg["is_training"])
    cfg["num_epochs"]     = int(cfg["num_epochs"])
    cfg["num_exits"]      = int(cfg["num_exits"])
    cfg["threshold"]      = [float(x) for x in cfg["threshold"]]
    cfg["new_data_perc"]  = [int(x)   for x in cfg["new_data_perc"]]
    return cfg


def test_loss_weights_with_different_num_exits(num_layers, max_exits, dataset, model, num_epochs, filename_train, file_name, configs=None
 ):
    headers_train = prepare_headers_train(num_layers)
    header = prepare_headers(num_layers)

    with open(filename_train, mode='w', newline='') as file_train, open(file_name, mode='w', newline='') as file_test:
        writer_train = csv.DictWriter(file_train, fieldnames=headers_train)
        writer_train.writeheader()

        writer_inference = csv.DictWriter(file_test, fieldnames=header)
        writer_inference.writeheader()


        all_configs = configs
        
        logging.info(f"Total configurations generated: {len(all_configs)}")

        # Use ProcessPoolExecutor for CPU-bound tasks

        num_configs = len(all_configs)
        BATCH_SIZE = 60
        num_batches = int(np.ceil(num_configs/BATCH_SIZE))

        for b in range(0, num_batches):
            batch_start = b*BATCH_SIZE
            batch_end = (b+1)*BATCH_SIZE

            if(batch_end > num_configs):
                batch_end = num_configs
            print("Batch ", b, "/", num_batches)

            batch_results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers= 8) as executor:
                futures = {executor.submit(run_and_record_experiment, config): config for config in all_configs[batch_start:batch_end]}

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)

                    except Exception as e:
                        logging.error(f"Error processing config: {e}")

            # Write in batches to reduce I/O overhead
            for br in batch_results:
                write_training_results_to_csv(br['exp_params'], br['final_training_result'], writer_train, num_layers)
                write_results_to_csv(br['inference_test_results'], br['train_test_track'], br['exp_params'], writer_inference, num_layers)
            file_train.flush()
            file_test.flush()

    logging.info("All experiments completed and results written")
    
    
def main():
    parser = argparse.ArgumentParser(description="Run one batch of SEEN sweep configs")
    parser.add_argument(
        '--config_file', type=str, required=True,
        help="Path to JSON list of experiment configs"
    )
    parser.add_argument(
        '--batch_id', type=int, default=0,
        help="Tag to embed in CSV filenames"
    )
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help="Directory where this batch will store CSVs"
    )
    args = parser.parse_args()

    # load & sanitize all of the configs
    with open(args.config_file) as cf:
        raw_configs = json.load(cf)
    # pull dataset/model/num_epochs/max_exits from the *first* entry
    dataset    = raw_configs[0]['dataset_name']
    model      = raw_configs[0]['model']
    num_epochs = int(raw_configs[0]['num_epochs'])
    max_exits  = int(raw_configs[0]['num_exits'])

    # make sure everything is the right Python type
    configs = [sanitize_config(cfg, dataset, model) for cfg in raw_configs]


    # prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    LOG_FILE = os.path.join(
        args.output_dir,
        f"log_run_{time.strftime('%Y%m%d-%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

    num_layers = 5  # both backbones have five conv blocks

    # compose CSV paths
    train_csv = os.path.join(
        args.output_dir,
        f"train_accuracy_dataset_{dataset}"
        f"_epoch_{num_epochs}__max_exits{max_exits}"
        f"_batch{args.batch_id}.csv"
    )
    test_csv = os.path.join(
        args.output_dir,
        f"test_accuracy_dataset_{dataset}"
        f"_epoch_{num_epochs}__max_exits{max_exits}"
        f"_batch{args.batch_id}.csv"
    )

    # run!
    t0 = time.time()
    test_loss_weights_with_different_num_exits(
        num_layers,
        max_exits,
        dataset,
        model,
        num_epochs,
        train_csv,
        test_csv,
        configs=configs
    )
    logging.info("Total wall-time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()



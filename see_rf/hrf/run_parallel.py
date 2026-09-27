import concurrent.futures
import logging
import time
import numpy as np
import os
from generateConfiguration import generate_configurations_for_datasets
from SaveData2 import SaveToCSV
from updateClassWeight_lastTree2 import Run_orchestrator 
import argparse 

# --- Setup logging ---
os.makedirs("LogFolder", exist_ok=True)
LOG_FILE = f"LogFolder/parallelized_experiment_run_{time.strftime('%Y%m%d-%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
def run_and_record_experiment(config, config_index_in_batch):
    """
    Runs a single experiment for a given configuration and dataset.
    This function will be executed in parallel processes.
    It calls the Run_orchestrator and returns its results.
    """
    dataset_name = config['dataset_name']
    split_points = config['split_points']
    max_depth = config['max_depth']
    n_estimators = config['n_estimators']

    # Use a unique index that spans all configurations, not just the batch
    unique_config_idx = config_index_in_batch 

    # It's good practice to get a logger specific to the process
    # This prevents multiprocessing issues with shared logging handlers if not configured carefully.
    # For simple logging setup as above, this might not be strictly necessary, but good for robustness.
    process_logger = logging.getLogger(f"Process-{os.getpid()}")
    process_logger.setLevel(logging.INFO) # Ensure logger level is set for this process
    
    process_logger.info(f"Starting experiment for Dataset: {dataset_name}, Config Index: {unique_config_idx}")
    process_logger.info(f"  Split Points: {split_points}")
    process_logger.info(f"  Max Depth: {max_depth}")
    process_logger.info(f"  Number of Trees: {n_estimators}")

    try:
        # Run_orchestrator trains, evaluates, and returns one row per threshold combination
        experiment_results_rows = Run_orchestrator(
            dataset_name=dataset_name,
            max_depth=max_depth,
            overall_n_estimators = n_estimators,
            split_points=split_points,
            config_index=unique_config_idx,
            logger=process_logger # Pass the logger instance to the orchestrator
        )
        process_logger.info(f"Finished experiment for Dataset: {dataset_name}, Config Index: {unique_config_idx}\n")
        return {'dataset_name': dataset_name, 'rows': experiment_results_rows}

    except Exception as e:
        process_logger.error(f"Error processing Dataset: {dataset_name}, Config Index: {unique_config_idx}: {e}", exc_info=True)
        return {'dataset_name': dataset_name, 'rows': [], 'error': str(e)}


if __name__ == '__main__':
    logging.info("Starting parallel experiment run.")

    # 1. Generate all configurations for all datasets
    num_configurations_per_dataset = 100
  
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Epilepsy')
    args = parser.parse_args()
    
    datasetnames = [name.strip() for name in  args.dataset_name.split(',')]
    
    all_configs_with_datasets = generate_configurations_for_datasets(datasetnames , num_configurations=num_configurations_per_dataset )
    logging.info(f"Total configurations generated: {len(all_configs_with_datasets)}")

    # Group configurations by dataset for easier processing and saving
    configs_by_dataset = {}
    for i, config in enumerate(all_configs_with_datasets):
        dataset_name = config['dataset_name']
        if dataset_name not in configs_by_dataset:
            configs_by_dataset[dataset_name] = []
        # Store config along with its global index for consistent tracking
        configs_by_dataset[dataset_name].append((config, i + 1)) 
    
    overall_start_time = time.time()

    # Iterate through each dataset and run its configurations in parallel
    for dataset_name, configs_data in configs_by_dataset.items():
        logging.info(f"\n--- Running experiments for Dataset: {dataset_name} ---")
        
        # Prepare a list of (config, global_index) tuples for the current dataset
        configs_for_current_dataset = [item[0] for item in configs_data]
        global_indices_for_current_dataset = [item[1] for item in configs_data]

        num_configs_dataset = len(configs_for_current_dataset)
        BATCH_SIZE = 100  # configs submitted to the process pool at a time
        num_batches = int(np.ceil(num_configs_dataset / BATCH_SIZE))

        # Initialize a SaveToCSV object for the current dataset in the main process
        dataset_saver = SaveToCSV(DataSet_Name=dataset_name)

        for b in range(num_batches):
            batch_start = b * BATCH_SIZE
            batch_end = min((b + 1) * BATCH_SIZE, num_configs_dataset)
            
            logging.info(f"Processing Batch {b+1}/{num_batches} for {dataset_name} (Configs {batch_start+1} to {batch_end})")

            batch_configs = configs_for_current_dataset[batch_start:batch_end]
            batch_global_indices = global_indices_for_current_dataset[batch_start:batch_end]

            batch_results = []
            # Use ProcessPoolExecutor for CPU-bound tasks
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                futures = {executor.submit(run_and_record_experiment, config, global_idx): (config, global_idx) 
                           for config, global_idx in zip(batch_configs, batch_global_indices)}

                for future in concurrent.futures.as_completed(futures):
                    original_config, original_global_idx = futures[future]
                    try:
                        result = future.result() # This will now receive the dictionary {'dataset_name': ..., 'rows': ...}
                        if result: # Check if result is not None (in case of unexpected errors)
                            batch_results.append(result)
                        else:
                            logging.warning(f"Received None result for config (Dataset: {original_config['dataset_name']}, Global Index: {original_global_idx}).")
                    except Exception as e:
                        logging.error(f"Error retrieving result for config (Dataset: {original_config['dataset_name']}, Global Index: {original_global_idx}): {e}", exc_info=True)

            # Write results for the current batch to the dataset's buffer
            for result_dict in batch_results: # Renamed 'result' to 'result_dict' for clarity
                if 'rows' in result_dict and result_dict['rows']:
                    for row in result_dict['rows']:
                        dataset_saver.append_completed_config_row(row)
                elif 'error' in result_dict:
                    logging.error(f"Skipping saving for errored config (Dataset: {result_dict['dataset_name']}): {result_dict['error']}")
            
        # After all batches for a dataset are processed, write its collected data to file
        dataset_saver.write_all_combined_metrics_to_file()
        logging.info(f"Completed all configurations for Dataset: {dataset_name}")

    overall_end_time = time.time()
    logging.info(f"All experiments completed across all datasets. Total duration: {overall_end_time - overall_start_time:.2f} seconds.")
    
    
    
    
    


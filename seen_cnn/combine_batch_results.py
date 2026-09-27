#!/usr/bin/env python3
"""
Combine the per-batch train/test CSVs written by run_sweep_batch.py into one train
and one test CSV per dataset, named the way analyze_sweep.py expects.

    results/<dataset>/<model>/batch_*/{train,test}_accuracy_dataset_*.csv
        -> results/combined/{train,test}_accuracy_dataset_<dataset>_epoch_<N>__max_exits3_<suffix>.csv
"""

import os
import glob
import pandas as pd
import argparse
from pathlib import Path

def find_batch_directories(base_path, dataset_name, model_type="AlexNetPartial"):
    """
    Find all batch directories for a given dataset.
    
    Args:
        base_path: Base path to results directory
        dataset_name: Name of the dataset (PAMAP2, Shoaib, WESADchest)
        model_type: Model type (default: AlexNetPartial)
    
    Returns:
        List of batch directory paths
    """
    pattern = os.path.join(base_path, dataset_name, model_type, "batch_*")
    batch_dirs = sorted(glob.glob(pattern))
    return [d for d in batch_dirs if os.path.isdir(d)]

def find_csv_files(batch_dir, file_type="train"):
    """
    Find all CSV files of a given type in a batch directory.
    This will find files with any max_exits value (max_exits2, max_exits3, etc.)
    
    Args:
        batch_dir: Path to batch directory
        file_type: "train" or "test"
    
    Returns:
        List of CSV file paths
    """
    pattern = os.path.join(batch_dir, f"{file_type}_accuracy_dataset_*.csv")
    csv_files = glob.glob(pattern)
    return sorted(csv_files)

def combine_csv_files(csv_files, output_file):
    """
    Combine multiple CSV files into one.
    
    Args:
        csv_files: List of CSV file paths to combine
        output_file: Output file path
    
    Returns:
        Combined DataFrame
    """
    if not csv_files:
        print(f"Warning: No CSV files found to combine for {output_file}")
        return None
    
    print(f"Combining {len(csv_files)} files:")
    for f in csv_files:
        print(f"  - {f}")
    
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            dataframes.append(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue
    
    if not dataframes:
        print(f"Error: No valid dataframes to combine for {output_file}")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"  Combined total: {len(combined_df)} rows")
    
    # Remove duplicates if any (based on all columns)
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    if len(combined_df) < initial_len:
        print(f"  Removed {initial_len - len(combined_df)} duplicate rows")
    
    # Save combined file
    combined_df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}\n")
    
    return combined_df

def get_epoch_from_filename(filename):
    """Extract epoch number from filename."""
    import re
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main(args):
    base_path = args.base_path
    datasets = args.datasets
    model_type = args.model_type
    num_epochs = args.num_epochs
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Base path: {base_path}")
    print(f"Model type: {model_type}")
    print(f"Output directory: {output_dir}\n")
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}\n")
        
        # Find all batch directories
        batch_dirs = find_batch_directories(base_path, dataset_name, model_type)
        
        if not batch_dirs:
            print(f"Warning: No batch directories found for {dataset_name}")
            continue
        
        print(f"Found {len(batch_dirs)} batch directories:")
        for bd in batch_dirs:
            print(f"  - {bd}")
        print()
        
        # Collect all train and test CSV files
        train_csv_files = []
        test_csv_files = []
        
        for batch_dir in batch_dirs:
            train_files = find_csv_files(batch_dir, "train")
            test_files = find_csv_files(batch_dir, "test")
            
            train_csv_files.extend(train_files)
            test_csv_files.extend(test_files)
        
        # Determine epoch number if not provided
        if num_epochs is None:
            if train_csv_files:
                epoch = get_epoch_from_filename(train_csv_files[0])
                if epoch:
                    num_epochs = epoch
                    print(f"Detected epoch number: {num_epochs}\n")
                else:
                    print("Warning: Could not detect epoch number from filename. Using default: 20")
                    num_epochs = 20
            else:
                num_epochs = 20
        
        # Filename suffix based on model type
        if model_type == "AlexNetPartial":
            filename_suffix = "AlexNet_SEENN_multi_parallel_batches_March17.csv"
        elif model_type == "sensorAware":
            filename_suffix = "SEENN_multi_parallel_batches_Nov27.csv"
        else:
            # Default fallback
            filename_suffix = "SEENN_multi_parallel_batches_March17.csv"
        
        # Combine train files
        if train_csv_files:
            train_output = os.path.join(
                output_dir,
                f"train_accuracy_dataset_{dataset_name}_epoch_{num_epochs}__max_exits3_{filename_suffix}"
            )
            print(f"Combining train files...")
            combine_csv_files(train_csv_files, train_output)
        else:
            print(f"Warning: No train CSV files found for {dataset_name}")
        
        # Combine test files
        if test_csv_files:
            test_output = os.path.join(
                output_dir,
                f"test_accuracy_dataset_{dataset_name}_epoch_{num_epochs}__max_exits3_{filename_suffix}"
            )
            print(f"Combining test files...")
            combine_csv_files(test_csv_files, test_output)
        else:
            print(f"Warning: No test CSV files found for {dataset_name}")
    
    print(f"\n{'='*60}")
    print("Combination complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine batch CSV results for analysis")
    parser.add_argument(
        '--base_path',
        type=str,
        default='results',
        help='Base path to results directory'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['Epilepsy'],
        help='List of dataset names to process'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='sensorAware',
        help='Model type (default: sensorAware)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Number of epochs (will be detected from filenames if not provided)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/combined',
        help='Output directory for combined CSV files'
    )
    
    args = parser.parse_args()
    main(args)


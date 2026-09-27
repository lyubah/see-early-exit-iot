#!/usr/bin/env python3
"""
Stage 1 of the SEEN pipeline: build the design-space sweep and split it into batch jobs.

1) Generates every (num_exits, exit_placement, loss_weights, new_data_perc) configuration
2) Shuffles them so heavy and light configs are spread across jobs
3) Splits them into chunks of --configs_per_job and writes each chunk to JSON
4) Writes one SLURM script per chunk that runs run_sweep_batch.py on it, and submits it
   (pass --no_submit to only write the files, e.g. to run a chunk by hand without SLURM)

Example:
    python generate_sweep_jobs.py --model sensorAware --dataset Epilepsy --configs_per_job 50 --no_submit
    python run_sweep_batch.py --config_file jobs/Epilepsy_sensorAware/configs_job_1.json \
        --batch_id 1 --output_dir results/Epilepsy/sensorAware/batch_1
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import textwrap
import time

import numpy as np


# Defaults: adjust if you ever need a different backbone
NUM_LAYERS = 5
MAX_EXITS  = 3
NUM_EPOCHS = 20


def make_json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)  # JSON doesn't allow tuples
    return obj


def chunk_list(lst, size):
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def partial_sampling_percentage_combinations(num_exits):
    valid_percentages = [20, 30, 40, 50]
    combinations = []
    for combo in itertools.product(valid_percentages, repeat=num_exits):
        if sum(combo) <= 80:
            combinations.append(list(combo))
    # logging.debug(f"Generated {len(combinations)} percentage combinations for {num_exits} exits")
    return combinations


def generate_loss_weight_combinations(num_exits):
    base_weights = np.arange(1, 5, 1)  # Range from 1 to 4
    all_combinations = list(itertools.product(base_weights, repeat=num_exits))
    descending_combinations = [combo for combo in all_combinations if list(combo) == sorted(combo, reverse=True)]    
    return descending_combinations


def generate_all_configurations(num_layers, max_exits, dataset, model, num_epochs):
    """Generate all possible experiment configurations upfront"""
    all_configs = []
    
    for num_exits_run in range(1, max_exits + 1):
        # Generate exit placements
        exit_placements_combinations = list(itertools.combinations(range(1, num_layers), num_exits_run))
        
        for exit_placements in exit_placements_combinations:
            # Generate loss weights
            loss_weight_combinations = generate_loss_weight_combinations(num_exits_run + 1)
            
            for loss_weights in loss_weight_combinations:
                # Generate percentage combinations
                percentage_combinations = partial_sampling_percentage_combinations(num_exits_run)
                
                for new_data_perc in percentage_combinations:
                    config = {
                        'dataset_name': dataset,
                        'model': model,
                        'is_training': True,
                        'num_epochs': num_epochs,
                        'num_exits': num_exits_run,
                        'threshold': [0.5] * num_exits_run,
                        'exit_placement': exit_placements,
                        'loss_weights': loss_weights,
                        'new_data_perc': new_data_perc
                    }
                    all_configs.append(config)
    
    return all_configs


def main():
    p = argparse.ArgumentParser(description="Generate the SEEN design-space sweep as SLURM batch jobs")
    p.add_argument('--model',            required=True, choices=['sensorAware', 'AlexNetPartial'],
                   help="sensorAware = CNN1D SEEN, AlexNetPartial = AlexNet SEEN")
    p.add_argument('--dataset',          default='Epilepsy',
                   help="Dataset name (expects Datasets/<name>_dataLabels.pkl)")
    p.add_argument('--configs_per_job',  type=int, required=True,
                   help="How many configs each SLURM job should run")
    p.add_argument('--NUM_EPOCHS', type=int, default=NUM_EPOCHS)
    p.add_argument('--partition', default=None, help="SLURM partition (omit to use the cluster default)")
    p.add_argument('--setup_cmd', default="",
                   help="Shell line run before the job, e.g. 'module load anaconda3 && conda activate seen'")
    p.add_argument('--no_submit', action='store_true',
                   help="Write the config JSONs and SLURM scripts but do not call sbatch")
    args = p.parse_args()

    # 1) generate every possible config
    print("Generating full configuration grid…")
    all_cfgs = generate_all_configurations(
        NUM_LAYERS,
        MAX_EXITS,
        args.dataset,
        args.model,
        args.NUM_EPOCHS
    )
    total = len(all_cfgs)
    print(f"  → {total} configs generated")

    # 2) shuffle to mix heavy/light
    random.shuffle(all_cfgs)

    # 3) split into chunks
    chunks = chunk_list(all_cfgs, args.configs_per_job)
    print(f"Splitting into {len(chunks)} jobs, ~{args.configs_per_job} configs each")

    # prepare directories (combine_batch_results.py reads results/<dataset>/<model>/batch_*)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    jobs_dir = os.path.join("jobs", f"{args.dataset}_{args.model}")
    results_dir = os.path.join("results", args.dataset, args.model)
    logs_dir = os.path.join(jobs_dir, "logs")
    for d in (jobs_dir, results_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    partition_line = f"#SBATCH --partition={args.partition}" if args.partition else ""

    for i, chunk in enumerate(chunks, start=1):
        cfg_path = os.path.abspath(os.path.join(jobs_dir, f"configs_job_{i}.json"))
        with open(cfg_path, 'w') as cf:
            json.dump(chunk, cf, default=make_json_safe)

        job_out = os.path.abspath(os.path.join(results_dir, f"batch_{i}"))
        os.makedirs(job_out, exist_ok=True)

        slurm_script = os.path.join(jobs_dir, f"slurm_job_{i}.sh")
        script = textwrap.dedent(f"""\
        #!/bin/bash
        {partition_line}
        #SBATCH --job-name={args.dataset}_{args.model}_{i}
        #SBATCH --output={os.path.abspath(logs_dir)}/slurm_%x_%j.out
        #SBATCH --error={os.path.abspath(logs_dir)}/slurm_%x_%j.err
        #SBATCH --time=08:00:00
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=8
        #SBATCH --mem=50G

        {args.setup_cmd}
        cd {os.path.abspath(os.getcwd())}

        srun python3 {os.path.join(code_dir, "run_sweep_batch.py")} \\
            --config_file {cfg_path} \\
            --batch_id {i} \\
            --output_dir {job_out}

        echo "Completed job on node $HOSTNAME"
        """)
        with open(slurm_script, 'w') as sf:
            sf.write(script)

        if args.no_submit:
            continue
        res = subprocess.check_output(['sbatch', slurm_script]).decode().strip()
        job_id = res.split()[-1]
        print(f"[{i}/{len(chunks)}] Submitted SLURM job {job_id} for {len(chunk)} configs")
        time.sleep(3)

    if args.no_submit:
        print(f"Wrote {len(chunks)} config chunks + SLURM scripts to {jobs_dir}/ (not submitted).")
    else:
        print("All jobs submitted. Monitor via `squeue -u $USER`.")


if __name__ == "__main__":
    main()

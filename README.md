# Sensor-Aware Early Exit (SEE) for Time-Series Classification on IoT Devices

On battery-powered wearables, **sensing** often costs more energy than computing. A
normal classifier waits for a full window of sensor data before it predicts. The
models in this repo predict from a **prefix of the window** instead. Each early exit
sees only the first *p*% of the samples. When an exit is confident enough (its
prediction entropy is below a threshold), the device stops sensing for that window. The
later exits, and the rest of the window, are only used for the hard cases.

The repo contains the core code for two families of these models:

| Family | Models | Folder |
|---|---|---|
| **SEE-RF**: random forests with exits | **SEE-vRF**: each tree is split by depth into segments, and each deeper segment may split on a longer prefix of the window. **SEE-hRF**: a cascade of sub-forests, one per exit, each trained on a longer prefix, with class weights carried from one stage to the next. | [`see_rf/`](see_rf) |
| **SEEN**: CNNs with exits | A 1-D CNN (`CNN1D`) and an AlexNet-style 1-D CNN, 5 conv blocks each, with exit branches after chosen blocks. Each exit sees only its share of the window, and later data is fed in through extra input blocks. | [`seen_cnn/`](seen_cnn) |

Both families include a **design-space sweep** (exit placement, data percentages,
loss weights, and entropy thresholds). The sweep reports each configuration's accuracy
together with its **sensing ratio** (`E_ratio`, the fraction of the window actually
sensed). Both also include **board scripts** that measure per-window latency and, with
an INA219 power sensor on a Raspberry Pi, energy.

## Repository layout

```
Datasets/                 preprocessed windows (<name>_dataLabels.pkl) + specs; Epilepsy is bundled
see_rf/
  vrf/                    SEE-vRF
    RF_sensorAware.py       accuracy vs. data fraction and forest size (scikit-learn RFs)
    rf_with_exits/          RandomForest / DecisionTree with exits + run_sweep.py (design-space sweep)
  hrf/                    SEE-hRF
    Seq_RandomForest.py     class-weighted sequential random forest
    updateClassWeight_lastTree2.py  staged training + exits for one configuration
    run_parallel.py         design-space sweep over random configurations
  hardware/               Raspberry Pi scripts: train_and_save.py -> Main.py (+ data_logger.py)
seen_cnn/
  NN_functions.py, EENN_functions.py, Alex_Net_functions.py   models + training
  Sensor_aware_early_exit.py                                  train/evaluate one configuration
  generate_sweep_jobs.py -> run_sweep_batch.py -> combine_batch_results.py -> analyze_sweep.py
  train_and_save_best.py -> Main_board_cnn.py -> process_power_energy.py
  *_timers.py, data_logger.py                                 timed model copies + power logger
```

Every script is run **from its own folder**. They read `Datasets/` by relative path, and
each folder has a `Datasets` symlink to the top-level one.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.10-3.12, CPU-only PyTorch is enough
```

Only the Epilepsy dataset (1.3 MB) is committed. For the other datasets, see
[Datasets/DATA_PROVENANCE.md](Datasets/DATA_PROVENANCE.md).

## Quick start (Epilepsy, a few minutes on a laptop)

**SEE-RF**

```bash
(cd see_rf/vrf && python RF_sensorAware.py --dataset_name Epilepsy --model_type RF_sensorAware)
(cd see_rf/hrf && python run_parallel.py --dataset_name Epilepsy)            # -> RecordsFolder/Epilepsy_records.csv
(cd see_rf/vrf/rf_with_exits && python run_sweep.py --dataset_name Epilepsy) # -> RF_results_Epilepsy.csv
```

**SEEN (CNN)**: the full pipeline, from sweep to board measurement:

```bash
cd seen_cnn

# 1. Build the sweep (3,120 configs per dataset/backbone) and run a batch.
#    On a SLURM cluster, drop --no_submit and the jobs are submitted for you.
python generate_sweep_jobs.py --model sensorAware --dataset Epilepsy --configs_per_job 50 --no_submit
python run_sweep_batch.py --config_file jobs/Epilepsy_sensorAware/configs_job_1.json \
    --batch_id 1 --output_dir results/Epilepsy/sensorAware/batch_1

# 2. Merge the batches, then score every config over a grid of entropy thresholds
python combine_batch_results.py --datasets Epilepsy --model_type sensorAware
python analyze_sweep.py --dataset_name Epilepsy --model_type sensorAware --num_epochs 20
#    -> Epilepsy_output_config_analysis_sensorAware_epochs_20_ends.csv
#       (train/test/total accuracy and E_ratio for every config x threshold combination)

# 3. Train the chosen config and save its weights (the exit config goes into the file name)
python train_and_save_best.py --dataset_name Epilepsy --backbone CNN1D --variant SEEN \
    --thresholds 0.8 0.8 --num_exits 2 --exit_placement 2 4 --new_data_perc 20 40 --loss_weights 2 1 1

# 4. Measure it (on the board, add --with_power to log INA219 power), then compute energy
python Main_board_cnn.py --dataset_name Epilepsy --backbone CNN1D --variant SEEN \
    --model_ckpt "ckpts/Epilepsy_CNN1D_SEEN_thresholds[0.8, 0.8]_exit_placement[2, 4]_new_data_perc[20, 40].ckpt" \
    --outdir board_results
python process_power_energy.py --results_dir board_results
```

`--model_type AlexNetPartial` / `--backbone AlexNet` selects the AlexNet backbone.
`--variant Baseline` runs the same backbone without the exit branches, for comparison.
To pick a configuration in step 3, load the analysis CSV and trade accuracy against
`E_ratio`. For example, among configs within one point of the best accuracy, take the
one with the lowest `E_ratio`:

```python
import pandas as pd
df = pd.read_csv("Epilepsy_output_config_analysis_sensorAware_epochs_20_ends.csv")
near_best = df[df.Test_acc_configuration >= df.Test_acc_configuration.max() - 1]
print(near_best.nsmallest(5, "E_ratio"))
```

## Notes

- The full sweeps were run on a SLURM cluster (8 CPU cores and about 50 GB per job).
  The quick start runs one small batch.
- `E_ratio` and the energy savings derived from it measure how much of each window was
  sensed. Measured energy comes from the board run (`Main_board_cnn.py` /
  `see_rf/hardware/Main.py` + `data_logger.py` + `process_power_energy.py`). See
  [see_rf/hardware/HARDWARE.md](see_rf/hardware/HARDWARE.md).
- Data splits are by window: 60/40 train/test for the CNN pipeline and
  `RF_sensorAware.py`, and 60/20/20 train/validation/test for the SEE-vRF sweep, SEE-hRF,
  and the Pi RF scripts.
- This is research code from a multi-author lab project. See the papers for the full
  author list.

## Papers

- *Energy-Efficient Time Series Applications on IoT Devices via Sensor-Aware Early-Exit
  Random Forest Architectures* (SEE-RF)
- *Energy-Efficient Time Series Applications on IoT Devices with Sensor-Aware Early-Exit
  Classifiers* (SEEN, CNN)

## License

MIT (see [LICENSE](LICENSE)). The datasets keep their original licenses.

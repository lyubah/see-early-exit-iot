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

It also contains the rest of the evaluation:
- **Design-space sweep** (both families): exit placement, data percentages, loss weights,
  and entropy thresholds. It reports each configuration's accuracy together with its
  **sensing ratio** (`E_ratio`, the fraction of the window actually sensed).
- **On-device measurement** ([`hardware/`](hardware)): per-window latency on a Raspberry Pi.
  With an INA219 power sensor it also gives energy per window; a model-size script
  covers memory.
- **Comparison baselines** ([`baselines/`](baselines)):
  - early-classification methods: static truncation, TEASER, Non-Myopic
  - compute-saving methods: Adaptive RF (EDEN), BranchyNet
  - full-window reference models: GB, RF, CNN

  They run over 5 seeds with a shared split. [ENERGY_METHODS.md](baselines/ENERGY_METHODS.md)
  puts every method on one total-energy axis.

## Repository layout

```
Datasets/        preprocessed windows (<name>_dataLabels.pkl) + specs; Epilepsy is bundled
see_rf/
  vrf/             SEE-vRF: RF_sensorAware.py (accuracy vs. data fraction and forest size),
                   rf_with_exits/ (forest with exits + run_sweep.py design-space sweep)
  hrf/             SEE-hRF: Seq_RandomForest.py, updateClassWeight_lastTree2.py (staged
                   training + exits), run_parallel.py (design-space sweep)
seen_cnn/
  NN_functions.py, EENN_functions.py, Alex_Net_functions.py   networks + training
  Sensor_aware_early_exit.py                                  train/evaluate one configuration
  generate_sweep_jobs.py -> run_sweep_batch.py -> combine_batch_results.py -> analyze_sweep.py
  train_and_save_best.py                                      train the chosen config -> .ckpt
hardware/
  rf/              SEE-vRF board run (+ full_window_baseline/ with the BMI160 sensor driver)
  cnn/             CNN board run (Main_board_cnn.py + timed model classes)
  data_logger.py, process_power_energy.py, compute_energy.py, model_size_table.py
baselines/         multi-seed comparison baselines (RF- and CNN-based) + SLURM scripts
```

Every script is run **from its own folder**. They read `Datasets/` by relative path, and
each folder has a `Datasets` symlink to the top-level one.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # Python 3.10-3.12, CPU-only PyTorch is enough
pip install -r baselines/requirements.txt       # only for the comparison baselines
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

# 3. Train the chosen config and save its weights (the exit config goes into the file name),
#    plus the same backbone without exits (Baseline) to compare against
for v in SEEN Baseline; do
  python train_and_save_best.py --dataset_name Epilepsy --backbone CNN1D --variant $v \
      --thresholds 0.8 0.8 --num_exits 2 --exit_placement 2 4 --new_data_perc 20 40 --loss_weights 2 1 1
done

# 4. Measure it (on the board, add --with_power to log INA219 power), energy, and model size
cd ../hardware/cnn
python Main_board_cnn.py --dataset_name Epilepsy --backbone CNN1D --variant SEEN \
    --model_ckpt "../../seen_cnn/ckpts/Epilepsy_CNN1D_SEEN_thresholds[0.8, 0.8]_exit_placement[2, 4]_new_data_perc[20, 40].ckpt" \
    --outdir board_results
python ../process_power_energy.py --results_dir board_results
python ../model_size_table.py
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

**Baselines**

```bash
cd baselines
python run_teaser_multiseed.py --dataset Epilepsy --seeds 42      # likewise for truncation / nonmyopic / eden
python aggregate.py && python make_paper_table.py                 # -> results/FINAL_acc_energy_table.md
```

See [baselines/README.md](baselines/README.md) for every method and the cluster runs, and
[hardware/README.md](hardware/README.md) for the board setup.

## Notes

- The full sweeps and baselines were run on a SLURM cluster. The quick start runs one
  small batch.
- `E_ratio` and the energy savings derived from it measure how much of each window was
  sensed. Measured energy comes from the board runs in `hardware/`.
- Data splits are by window:
  - 60/40 train/test for the CNN pipeline, `RF_sensorAware.py`, and the multi-seed baselines.
  - 60/20/20 train/validation/test for the SEE-vRF sweep, SEE-hRF, the Pi RF scripts, and
    the GB baseline.
- This is research code from a multi-author lab project. See the papers for the full
  author list.

## Papers

- *Energy-Efficient Time Series Applications on IoT Devices via Sensor-Aware Early-Exit
  Random Forest Architectures* (SEE-RF)
- *Energy-Efficient Time Series Applications on IoT Devices with Sensor-Aware Early-Exit
  Classifiers* (SEEN, CNN)

## License

MIT (see [LICENSE](LICENSE)). The datasets keep their original licenses.

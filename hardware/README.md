# On-device measurement

Everything that runs on the board or turns board logs into numbers:
- per-window **latency**
- **energy** from an INA219 power sensor
- **model size**

The early-exit board runs (RF and CNN) write one row per window:

```
t_start, t1, t2, t3, t4, total, true_label, prediction, correctness, exit_taken, data%
```

The full-window baseline writes `t_start, t_end, total, true_label, prediction,
correctness`. `process_power_energy.py` reads both formats.

## Rig
- **Board:** Raspberry Pi (a Zero 2W is enough for the forests; the AlexNet-sized CNNs
  need a Pi 4/5 or a Jetson-class board), Python 3.
- **Power:** Texas Instruments **INA219** current/power monitor on I2C bus 1, 0.1 Ω shunt
  (`pip install pi-ina219`).
- **Sensor:** Bosch **BMI160** accelerometer/gyroscope on I2C (address 0x69), used by
  the full-window baseline to power the sensor for real acquisition time.

## Layout

| Path | What it does |
|---|---|
| `data_logger.py` | Samples INA219 power as fast as it can into `<dataset>_shared_Power.csv` (`time s, power mW`). Run it next to a board run. |
| `process_power_energy.py` | Pairs each run's latency CSV with its power log and integrates power over each window's `[t_start, t_start + total]`, giving mJ per window and one summary row per run. |
| `compute_energy.py` | The same integration for a single power log (optionally per window with `--latency_csv`). |
| `model_size_table.py` | Model size in KB (fp32 parameters) of each trained CNN checkpoint, and the extra memory the early exits add over the plain backbone. |
| `rf/train_and_save.py` | Trains the SEE-vRF forest (`see_rf/vrf/rf_with_exits`) on a 60/20/20 split and pickles it with the test windows. |
| `rf/Main.py`, `rf/Training_Inference.py` | Runs the forest exit by exit on each test window. It stops at the first exit whose entropy is under its threshold. |
| `rf/DataLoad.py` | Loads a dataset, splits it, and flattens triaxial windows in time order. |
| `rf/full_window_baseline/` | Full-window baseline (`run_baseline.py`): switches the BMI160 on, waits one window of real acquisition time, then classifies. Includes the sensor driver (`sensor_control.py`) and `run_all.sh`, which loops over the datasets with the logger running. |
| `cnn/Main_board_cnn.py` | CNN1D / AlexNet board run for the Baseline, EE, and SEEN variants. The exit config is read from the checkpoint file name. |
| `cnn/*_timers.py` | Copies of the SEEN model classes with per-exit timing. They build on the networks in `seen_cnn/`. |

## Runs

```bash
# SEE-vRF with exits
cd hardware/rf
python train_and_save.py --dataset_name Epilepsy --n_est 83 --max_depth 17 \
       --tree_splits 0.5 1 --proportions 0.25 1
python3 ../data_logger.py Epilepsy &                 # on the Pi: start power logging
python3 Main.py --dataset_name Epilepsy --num_exits 2 --proportions 0.25 1 --th_combination 0.4
#   -> Epilepsy_accuracy_results.csv  (Main.py stops the logger when it finishes)

# CNN (checkpoints come from seen_cnn/train_and_save_best.py)
cd hardware/cnn
python3 Main_board_cnn.py --dataset_name Epilepsy --backbone CNN1D --variant SEEN \
    --model_ckpt "../../seen_cnn/ckpts/<checkpoint>.ckpt" --outdir board_results --with_power
#   --with_power starts ../data_logger.py itself (Pi only); rename its
#   Epilepsy_shared_Power.csv to Epilepsy_CNN1D_SEEN_power.csv if you run several models

# Full-window baseline with the real sensor (expects saved_pkl/<ds>_trained_model.pkl,
# a scikit-learn forest trained on the full window, plus the test-set .npy files
# in the layout train_and_save.py writes)
cd hardware/rf/full_window_baseline && bash run_all.sh

# Energy and model size
python3 hardware/process_power_energy.py --results_dir hardware/cnn/board_results
python3 hardware/model_size_table.py --models seen_cnn/ckpts
```

Without a power log, `process_power_energy.py` still reports latency and accuracy. You
can also pass `--active_power_mw` to estimate energy from latency.

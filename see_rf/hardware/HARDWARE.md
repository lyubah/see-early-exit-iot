# On-device measurement (Raspberry Pi + INA219)

These scripts measure per-window latency and energy for the SEE-vRF forest on a
Raspberry Pi. The CNN models use the same rig and CSV format through
[`seen_cnn/Main_board_cnn.py`](../../seen_cnn/Main_board_cnn.py).

## Rig
- **Board:** Raspberry Pi (Zero 2W for the RF models; the AlexNet-sized CNNs need a
  Pi 4/5 or a Jetson-class board), Python 3.
- **Power:** Texas Instruments **INA219** current/power monitor on I2C **bus 1** with a
  **0.1 Ω** shunt (`pip install pi-ina219`).

## Scripts
| File | Role |
|---|---|
| `train_and_save.py` | Trains the SEE-vRF forest (`see_rf/vrf/rf_with_exits`) on a 60/20/20 split and pickles it with the test windows. |
| `Main.py` | Loads the pickled forest and classifies each test window exit by exit, stopping at the first exit whose entropy is under its threshold. Writes `<dataset>_accuracy_results.csv` with per-exit timestamps. |
| `Training_Inference.py` | The per-window exit loop used by `Main.py`. |
| `DataLoad.py` | Loads a dataset, makes the split, and flattens triaxial windows in time order. |
| `data_logger.py` | Samples INA219 power as fast as it can into `<dataset>_shared_Power.csv` (runs alongside inference). |

## Typical run
```bash
# Anywhere (trains the model):
python train_and_save.py --dataset_name Epilepsy --n_est 83 --max_depth 17 \
       --tree_splits 0.5 1 --proportions 0.25 1

# On the Pi, in the same folder as the saved model files:
python data_logger.py Epilepsy &          # start power logging
python Main.py --dataset_name Epilepsy --num_exits 2 --proportions 0.25 1 --th_combination 0.4
# -> Epilepsy_accuracy_results.csv (t_start, t1..t4, total, true_label, prediction,
#    correctness, exit_taken, data%) + Epilepsy_shared_Power.csv (time, power mW)
```

Energy per window is the integral of power over the window's
`[t_start, t_start + total]` interval. `seen_cnn/process_power_energy.py` does this
pairing and integration for a whole folder of runs.

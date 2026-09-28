#!/bin/bash
# Full-window RF baseline on the board, one dataset after another: start the INA219
# power logger in the background, run the baseline (run_baseline.py stops the logger
# when it finishes), then let the board cool down before the next dataset.

datasets=(
  Epilepsy
  Shoaib
  EMGPhysical
  SelfRegulationSCP1
  WESADchest
  PAMAP2
)

LOGGER="$(dirname "$0")/../../data_logger.py"

for d in "${datasets[@]}"; do
  echo "Running run_baseline.py for $d"

  python3 "$LOGGER" "$d" &
  python3 run_baseline.py --dataset_name "$d"

  echo "Finished $d. Sleeping 300 seconds..."
  sleep 300
done

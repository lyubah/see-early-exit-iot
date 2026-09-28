# Comparison baselines

Multi-seed runners for the methods SEE-RF and SEEN are compared against. Every
baseline is run over **5 seeds** (42–46) on all six datasets and reports accuracy,
weighted precision / recall / F1, and energy savings as `mean (std)`.

| Baseline | What it does | Savings type | Runner |
|---|---|---|---|
| Static truncation (RF) | Classifies from only the first P% of the window (P = 10…100); P = 100 is the full-window RF | sensing | `run_truncation_multiseed.py` |
| TEASER | RF slave classifiers + one-class-SVM master that decides when to stop | sensing | `run_teaser_multiseed.py` |
| Non-Myopic | tslearn's `NonMyopicEarlyClassifier` (RF base, fixed time cost `alpha = c / T`) | sensing | `run_nonmyopic_multiseed.py` |
| Adaptive RF (EDEN) | Reads the full window and stops evaluating trees early | compute | `run_eden_multiseed.py` |
| Gradient boosting | Full-window GB (single seed) | none | `run_simple_gb_baseline.py` |
| Baseline / truncation CNN | The CNN1D backbone on the first P% of the window (P = 100 is the plain CNN) | sensing | `run_baseline_cnn_multiseed.py` |
| BranchyNet | CNN1D with depth early exits on the full window | compute | `run_branchynet_cnn_multiseed.py` |
| TEASER-CNN, Non-Myopic-CNN | TEASER and Non-Myopic with the CNN1D backbone as the base classifier | sensing | `run_teaser_cnn_multiseed.py`, `run_nonmyopic_cnn_multiseed.py` |

**Protocol** (shared through `mseed_common.py`):
- One 60/40 train/test split by window per seed, identical for every method.
- The seed sets both the split and the model's random state.
- RF-based methods use 50 trees, max depth 30.
- Weighted P/R/F1 (so weighted recall equals accuracy); std is the sample std (ddof = 1).

**Energy.** Sensing-saving and compute-saving methods are put on one total-energy
axis, where sensing is 70% of the budget. See [ENERGY_METHODS.md](ENERGY_METHODS.md).

## Layout

```
mseed_common.py            seeds, shared split, metric aggregation, table formatting
run_*_multiseed.py         one runner per baseline (writes results/ or results_cnn/)
run_simple_gb_baseline.py  full-window gradient boosting
cnn_base.py                scikit-learn wrapper around the CNN1D backbone
aggregate.py, make_paper_table.py            RF results  -> mean(std) tables (CSV / TeX / Markdown)
aggregate_cnn.py, make_cnn_paper_table.py    CNN results -> mean(std) tables
methods/                   shared helpers: data loading + triaxial layout (baseline_data.py,
                           seg_layout.py), TEASER (run_teaser_benchmark.py), EDEN
                           (run_eden_baseline.py), Non-Myopic settings (nonmyopic_config.py)
slurm/                     SLURM jobs + submit_all.sh / submit_all_cnn.sh + _env.sh
ENERGY_METHODS.md          how energy savings are computed for each method
```

## Run

```bash
pip install -r requirements.txt            # on top of the repo-level requirements.txt

# one dataset, one seed (a few minutes on a laptop)
python run_truncation_multiseed.py --dataset Epilepsy --seeds 42
python run_teaser_multiseed.py     --dataset Epilepsy --seeds 42
python run_nonmyopic_multiseed.py  --dataset Epilepsy --seeds 42
python run_eden_multiseed.py       --dataset Epilepsy --seeds 42
python aggregate.py && python make_paper_table.py     # -> results/FINAL_acc_energy_table.{csv,md,tex}

python run_baseline_cnn_multiseed.py   --dataset Epilepsy --seeds 42 --epochs 20
python run_branchynet_cnn_multiseed.py --dataset Epilepsy --seeds 42 --epochs 20
python aggregate_cnn.py && python make_cnn_paper_table.py   # -> results_cnn/CNN_PAPER_TABLE.{csv,md}

# all datasets x 5 seeds on a SLURM cluster (set the conda env names in slurm/_env.sh)
bash slurm/submit_all.sh
bash slurm/submit_all_cnn.sh
```

Non-Myopic-CNN trains one CNN per timestep, so it is by far the slowest baseline.
Run it on a cluster.

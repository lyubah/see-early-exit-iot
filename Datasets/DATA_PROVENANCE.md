# Datasets

Every dataset is stored as a preprocessed, fixed-window pickle:

- `<name>_dataLabels.pkl`: `{"data": float array (N windows, C channels, T samples), "labels": (N,) int array}`
- `<name>_specs.pkl`: `{"SEG_SIZE": T, "CHANNEL_NB": C, ...}`

Only **Epilepsy** (1.3 MB) is committed. To run another dataset, preprocess it into the
same format and put `<name>_dataLabels.pkl` in this folder. The `*_specs.pkl` files for
all six datasets are already here.

| Dataset (pickle name) | Source | Original reference | Windows | Channels x samples | Classes |
|---|---|---|---|---|---|
| Epilepsy | UEA/UCR archive | Villar et al., 2016 | 275 | 3 x 206 | 4 |
| PAMAP2 | UCI PAMAP2 | Reiss & Stricker, ISWC 2012 | 2048 | 3 x 1536 (3 axes x 512) | 5 |
| Shoaib | Physical Activity Recognition | Shoaib et al., *Sensors* 2014 | 3150 | 5 x 600 (3 axes x 200) | 7 |
| WESADchest | WESAD (chest) | Schmidt et al., ICMI 2018 | 7421 | 5 x 200 | 3 |
| SelfRegulationSCP1 | UEA/UCR archive | Birbaumer et al., *Nature* 2001 | 561 | 6 x 896 | 2 |
| EMGPhysical | UCI EMG Physical Action | Theodoridis, 2011 | 782 | 8 x 200 | 4 |

For the triaxial datasets (PAMAP2, Shoaib), the three axes are packed inside the sample
axis. `seg_layout.py` detects that layout and regroups it into time order, so a "first
*p*% of the window" prefix really is the earliest *p*% of time across all axes.

Each dataset keeps the license of its original source.

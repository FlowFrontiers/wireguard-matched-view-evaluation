# WireGuard Matched-View Evaluation

Self-contained code for matched-view WireGuard traffic-classification
evaluation.

The implementation provides the public-data build, validation, feature
construction, five frozen model-training configurations, and analysis outputs.
Evaluation tables, confidence intervals, and figures are generated from stored
prediction artifacts.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

The lock file pins the environment used for artifact generation, including
TensorFlow/Keras, scikit-learn, and XGBoost.

## Public inputs

Download the dataset from <https://doi.org/10.5281/zenodo.18945858> and place
the four released Parquet files at:

```text
data/raw/
├── session1/
│   ├── session1_flows.parquet
│   └── session1_packet_matches.parquet
└── session2/
    ├── session2_flows.parquet
    └── session2_packet_matches.parquet
```

No CSV conversion and no PCAP files are required.

## Data and feature contract

```bash
wgme build-data
wgme validate-data
wgme audit-features
```

`build-data` reassigns all public packet matches to released flows and rebuilds
both observation views under one code path. It excludes flows without matched
outer packets, applies the frozen 200-flow class threshold, and produces the
226,281-pair canonical dataset. Full-flow statistics use consecutive PIATs,
sample standard deviation (`ddof=1`), initiator-relative direction, and NaN
rates for non-positive duration.

The early-flow representation uses the first 50 source-selected matched packet
pairs. Each view preserves its own observed ordering. Valid direction is mapped
from raw `{0,1}` to `{-1,+1}` and zero is reserved for padding. Size and PIAT are
transformed with `log1p`.

Generated artifacts are written to `data/processed/` and are protected by
input/output SHA-256 hashes plus executable invariants. Existing outputs are
never overwritten unless `--force` is given.

## Development checks

```bash
pytest
ruff check .
```

## Model training

Each command uses the inner view as its source and writes predictions for the
matched outer view under `outputs/runs/<model-id>/`. The classical models fit
all source rows. CNN1D first reserves a stratified 10% inner-view validation
subset, then augments only the remaining training rows. Its direction channel
and padded timesteps are never perturbed.

```bash
wgme train-model --model rf_matched_flow_stats
wgme train-model --model xgboost_matched_flow_stats
wgme train-model --model rf_flattened_splt
wgme train-model --model xgboost_flattened_splt
wgme train-model --model cnn1d_sequential_splt --device gpu
```

Runs require a clean committed revision and refuse to overwrite existing
artifacts unless `--force` is supplied. Validate any completed run with:

```bash
wgme validate-run outputs/runs/rf_matched_flow_stats
```

## Complete reproduction

Run a non-executing preflight, then execute the five models in isolated
processes and generate every analysis artifact:

```bash
wgme run-all --preflight
wgme run-all --device gpu
```

Isolation prevents TensorFlow and XGBoost runtime libraries from sharing one
process. Completed valid runs are reused, so the command can resume an
interrupted campaign. Analysis outputs are written under `outputs/analysis/`:

- `metrics.csv` and `bootstrap_intervals.csv` contain the result table;
- `per_class_metrics.csv`, `cnn1d_confused_pairs.csv`, and
  `confusion_matrices.npz` provide class-level diagnostics;
- `figures/` contains PNG and PDF plots;
- `latex/` contains a generated table fragment and result macros.

The four confusion-derived metrics use 1,000 paired pair-level bootstrap
replicates at 95% confidence. Macro one-vs-rest average precision is a point
estimate because it depends on row-level probabilities and is not included in
the confidence-interval table.

Validate the complete analysis against the stored model predictions with:

```bash
wgme validate-analysis
```

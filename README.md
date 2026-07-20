# Matched-View Cross-Domain Evaluation of WireGuard VPN Traffic Classification Using Early-Flow Fingerprints

Reproducibility artifact accompanying the manuscript by Yasameen Sajid
Razooqi and Adrian Pekar.

It provides the public-data build, validation, feature
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

## Released evaluation evidence

The definitive campaign evidence is committed under `artifacts/fcf7f2d/`,
separate from the ignored `outputs/` directory used by fresh reproductions. It
contains the five prediction sets and run manifests, all analysis tables,
confidence intervals, diagnostics, figures, LaTeX fragments, the complete
CNN1D model, and the campaign log and exit status.

Validate this evidence directly after setup; the public dataset is not needed
for this command:

```bash
wgme validate-release artifacts/fcf7f2d
```

### Campaign provenance

The committed manifests record the original campaign revision
`fcf7f2d4fbcf607b303133a9ebde600f23c41745`.  The public repository uses a
clean two-commit history rather than retaining the intermediate development
history.  Its annotated tag `fcf7f2d` identifies a campaign-source commit whose
Git tree is byte-identical to that original revision; both have tree SHA-1
`dd755bbdb420ad524aea7c6f4acde6e36e3b958c`.  The different commit SHA is a
consequence of rewriting the parent history and commit message, not a change to
the campaign source.  The following commit adds the validated evidence and the
release-validation layer.  Thus `git checkout fcf7f2d` recovers the exact source
tree used for the definitive campaign, while `main` contains that source plus
the released evidence.

The four fitted RF/XGBoost model files are deliberately omitted from Git
because they total approximately 1.75 GB. Their exact SHA-256 hashes and byte
sizes remain bound in both the run manifests and `release_manifest.json`.
Predictions, probabilities, point metrics, confusion matrices, and downstream
tables are retained and revalidated from content. The omitted estimators can be
regenerated from the public inputs with the complete-reproduction command
below.

`validate-release` permits only these four declared omissions. By contrast,
`validate-run` remains strict and requires every artifact from a locally
generated run. Bootstrap replay uses a narrow absolute tolerance only for
summary values because NumPy's seeded multinomial sampler is not bit-identical
across ARM and x86; point metrics and artifact hashes remain strictly checked.

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

Fresh reproduction outputs never overwrite the committed evidence under
`artifacts/fcf7f2d/`.

## Licensing

Source code and documentation are released under the MIT License in
[`LICENSE`](LICENSE).  The generated evaluation evidence committed under
`artifacts/` is released under the Creative Commons Attribution 4.0
International License (CC BY 4.0), as specified in
[`artifacts/LICENSE`](artifacts/LICENSE).

The upstream flow and packet-match dataset is not redistributed by this
repository.  It remains subject to the license and attribution requirements of
its Zenodo release and should be cited through the dataset article and archive
listed in [Public inputs](#public-inputs).

# Subject-aware NIRS modeling

Code for the two-stage subject-aware NIRS pipeline used in the thesis:

1. **Stage 1 — representation learning:** learn a subject-level representation from DTOF measurements.
2. **Stage 2 — absolute prediction:** combine DRS measurements with the Stage 1 representation to predict absolute HC or StO₂.

The main thesis configuration is a **32-dimensional Stage 1 representation** and a **gated residual Stage 2 fusion model**. A 16D sensitivity experiment and an optional FiLM comparison are retained as YAML configurations. Unless stated otherwise, run every command below from the repository root (the directory containing `pyproject.toml`).

## Repository layout

```text
configs/                         Reproducible Stage 1/Stage 2 YAML configurations
data/README.md                   Private-data placement instructions
notebooks/01_... to 07_...       Ordered thesis analyses and figures
scripts/train_stage1.py          Train one Stage 1 LOSO fold
scripts/export_subject_features.py  Export one fold's full-OP features
scripts/run_stage1_loso.sh       Train/export all 154 Stage 1 folds
scripts/train_stage2.py          Stage 2 entry point driven by YAML
scripts/run_stage2_loso.sh       Stage 2 wrapper (154 folds by default)
scripts/smoke_test.py            Data-free model construction/forward check
src/subject_nirs/stage1/         Stage 1 implementation and analyses
src/subject_nirs/stage2/         Stage 2 implementation and analyses
tests/                           Fast configuration and resume tests
artifacts/                       Local checkpoints/predictions (not committed)
cache/                           Rebuildable preprocessing caches (not committed)
results/figures/                 Curated thesis PNGs (may be committed)
results/tables/                  Rebuildable tables (not committed)
```

The old `Result_plot/`, `tpbm_prediction/`, exploratory manifold/evaluation programs, and unused `blending.py` are intentionally excluded.

## 1. Environment setup

Python 3.10 or newer is required.

```bash
cd /path/to/final_refactored
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,test]"
```

An existing Conda environment can be used instead:

```bash
conda activate YOUR_ENV
cd /path/to/final_refactored
python -m pip install -e ".[notebook,test]"
```

Check PyTorch/CUDA and run the data-free tests:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python scripts/smoke_test.py
pytest -q
```

The shell wrappers use the active environment's `python`. To force a particular interpreter without editing them:

```bash
export SUBJECT_NIRS_PYTHON=/path/to/conda/env/bin/python
```

## 2. Private data placement

Datasets, metadata, caches, checkpoints, and predictions are excluded from Git. Use this local layout:

```text
data/
├── raw/
│   ├── stage1/
│   │   ├── dataset_dtof.mat
│   │   ├── dataset_OP.mat
│   │   └── temp_metadata_v2.csv
│   └── stage2/
│       └── NIRS_Absolute_Dataset_grid.mat
└── NIRS_Absolute_Val_Dataset_new.mat
```

`NIRS_Absolute_Val_Dataset_new.mat` is used only by notebook 07. It is read directly in batches and does not require the old cache format. Large files may be symlinked instead of copied:

```bash
ln -s /external/path/dataset_dtof.mat data/raw/stage1/dataset_dtof.mat
```

## 3. Stage 1 — DTOF representation learning

### 3.1 Train one fold

The test subject ID is zero-based (`0`–`153`). This trains fold 0 with the main 32D configuration:

```bash
python scripts/train_stage1.py \
  --config configs/stage1_32.yaml \
  --data_dir data/raw/stage1 \
  --output_dir artifacts/stage1/0/relat_cons_32 \
  --test_id 0 \
  --amp \
  --skip_tsne
```

Important outputs are `best_model.pt`, `evaluation_metrics.json`, `run_config.json`, `subject_features_eval_grid.npz`, and `training_history.png` under the fold directory. `--amp` enables CUDA mixed precision; remove it on CPU. `--skip_tsne` avoids a costly visualization not needed for the main results.

### 3.2 Export one fold's full-OP features

Stage 2 consumes `subject_features_all_op.npz`, not only the evaluation-grid NPZ:

```bash
python scripts/export_subject_features.py \
  --checkpoint artifacts/stage1/0/relat_cons_32/best_model.pt \
  --data_dir data/raw/stage1 \
  --save_path artifacts/stage1/0/relat_cons_32/subject_features_all_op.npz \
  --amp
```

The checkpoint fold, directory number, and exported feature fold must match. The exported `latent_raw` array is the default Stage 2 feature.

### 3.3 Run all 154 folds

```bash
bash scripts/run_stage1_loso.sh
```

The wrapper loops over test IDs `0..153`; each fold is trained and then exported. To run 16D, change both `CONFIG` and `OUTPUT_DIR` in the wrapper from `stage1_32`/`relat_cons_32` to `stage1_16`/`relat_cons_16`. Stage 1 does not automatically skip completed folds; when restarting, change the loop range (for example `{37..153}`) or run only the required folds individually.

## 4. Stage 2 — absolute HC/StO₂ prediction

YAML values override defaults in `src/subject_nirs/stage2/config.py`. Thus `target_mode`, `experiment_mode`, and `fusion_method` are not fixed by their dataclass defaults when a config is passed.

### 4.1 Required order

1. Complete all Stage 1 folds and export every `subject_features_all_op.npz`.
2. Train the DRS-only HC and/or StO₂ baselines.
3. Train residual (main) or FiLM (optional) fusion from those checkpoints.
4. Run notebooks 04–07.

Stage 2 reads `data/raw/stage2/NIRS_Absolute_Dataset_grid.mat`. Rebuildable preprocessing and compact subsets are stored under `cache/stage2/`.

### 4.2 Train DRS-only baselines

```bash
bash scripts/run_stage2_loso.sh configs/stage2/baseline_hc.yaml
bash scripts/run_stage2_loso.sh configs/stage2/baseline_sto2.yaml
```

Equivalent direct calls:

```bash
python scripts/train_stage2.py --config configs/stage2/baseline_hc.yaml
python scripts/train_stage2.py --config configs/stage2/baseline_sto2.yaml
```

Outputs go to `artifacts/stage2/hc_baseline_scratch_main/` and `artifacts/stage2/sto2_baseline_scratch_main/`. These configs enable `save_predictions: true` because notebook 04 needs prediction NPZs in addition to `loso_results.csv`.

### 4.3 Train the main residual models

```bash
bash scripts/run_stage2_loso.sh configs/stage2/residual_hc.yaml
bash scripts/run_stage2_loso.sh configs/stage2/residual_sto2.yaml
```

Calling the wrapper without an argument defaults to HC residual. Outputs are:

```text
artifacts/stage2/hc_fusion_dtof_actual_pretrained_finetune_residual_gated_main/
artifacts/stage2/sto2_fusion_dtof_actual_pretrained_finetune_residual_gated_main/
```

### 4.4 Optional FiLM comparison

```bash
bash scripts/run_stage2_loso.sh configs/stage2/film_hc.yaml
```

FiLM is retained as a comparison; residual is the main model. To add StO₂ FiLM, copy `film_hc.yaml`, change `target_mode` and the baseline checkpoint path, and use a distinct suffix.

### 4.5 Use 16D Stage 1 features

Copy the relevant residual/FiLM YAML and change both fields:

```yaml
structure_feature_dim: 16
structure_feature_path: ./artifacts/stage1/{test_id}/relat_cons_16/subject_features_all_op.npz
```

### 4.6 Stop and resume safely

All supplied Stage 2 configs use `resume: true`. On startup the runner audits the output directory. A fold is skipped only when it has both a valid global `test_id` row in `loso_results.csv` and the matching checkpoint. An interrupted fold is rerun, while completed folds retain their global subject mapping. Re-execute the same command after interruption; do not manually renumber folds. Use `resume: false` only for an intentional new run/output directory.

## 5. Notebooks and thesis analyses

Start Jupyter or VS Code from the project root:

```bash
cd /path/to/final_refactored
python -m jupyter lab
# or: code .
```

Select the same environment as the notebook kernel, open a file under `notebooks/`, and choose **Run All**. The notebooks support a kernel working directory of either the root or `notebooks/` and locate the root with `pyproject.toml`.

### 01 — `01_stage1_retrieval.ipynb`

- **Purpose:** Stage 1 retrieval figure over 154 LOSO folds.
- **Reads:** Stage 1 checkpoints and DTOF data. Set `EXPERIMENT='relat_cons_16'` for 16D.
- **Computes:** Top-1 retrieval, within-subject distance, nearest-other distance, and margin.
- **Writes:** fold metrics under `results/tables/stage1/full_grid_<experiment>/` and the curated PNG under `results/figures/stage1/`.
- **Resume:** existing per-fold evaluation JSON files are reused.

### 02 — `02_stage1_op_robustness_cka.ipynb`

- **Purpose:** representation stability across optical-property changes using linear CKA.
- **Reads:** Stage 1 checkpoints and `dataset_dtof.mat`.
- **Computes:** full OP-pair CKA, transition summaries, and conditional effects.
- **Writes:** rebuildable tables/cache under `results/tables/stage1/cka_<experiment>/` plus PNGs.
- **Resume:** valid fold caches are reused.

### 03 — `03_stage1_structure_probe.ipynb`

- **Purpose:** test whether held-out embeddings contain structural thickness information.
- **Reads:** all `subject_features_all_op.npz` and `data/raw/stage1/temp_metadata_v2.csv`.
- **Computes:** nested ridge probing; scaling and alpha selection use training subjects only.
- **Writes:** held-out predictions, metrics, and probe PNGs under `results/tables/stage1/structure_probe_32/`.

### 04 — `04_stage2_baseline_evaluation.ipynb`

- **Purpose:** evaluate DRS-only HC and StO₂ baselines.
- **Reads:** both baseline directories, `loso_results.csv`, and prediction NPZ files.
- **Computes:** subject-level calibration, RMSE, MAE, and signed bias.
- **Writes:** tables under `results/tables/stage2/baseline/` and final baseline PNGs.

### 05 — `05_stage2_fusion_comparison.ipynb`

- **Purpose:** compare DRS-only, residual, and available FiLM experiments on matching folds.
- **Reads:** directories in the notebook's `EXPERIMENTS` dictionary; baseline plus one fusion run is required.
- **Computes:** paired ΔRMSE (fusion minus baseline), Wilcoxon tests, bootstrap intervals, and effects by baseline difficulty.
- **Writes:** comparison PNGs and rebuildable tables under `results/tables/stage2/fusion_comparison/`; PDFs are disabled.
- **Extend:** add more experiment-directory/label pairs to `EXPERIMENTS`.

### 06 — `06_stage2_subject_failure.ipynb`

- **Purpose:** generate thesis subject-failure Figure 2.
- **Reads:** HC baseline performance and Stage 2 validation cache.
- **Computes:** intensity-balanced DRS signatures and contrasts for accurate, over-estimated, and under-estimated subjects.
- **Writes:** a rebuildable signature NPZ under `cache/stage2/subject_failure/` and the final PNG under `results/figures/stage2/`.

### 07 — `07_unseen_scattering_validation.ipynb`

- **Purpose:** evaluate frozen baselines on the separate unseen-scattering MAT file.
- **Parameters:** `TARGET='hc'` or `'sto2'`; `FOLDS=None` for all; `DEVICE='auto'`, `'cpu'`, `'cuda'`, or e.g. `'cuda:1'`.
- **Reads:** `data/NIRS_Absolute_Val_Dataset_new.mat` directly and the target's baseline checkpoints.
- **Writes:** resumable metrics under `artifacts/stage2/unseen_scattering/<target>/` and a two-panel violin/box/fold-point PNG under `results/figures/stage2/unseen_scattering/`.
- **Resume:** completed fold JSONs are reused. `FOLDS='1,3,10-20'` runs a subset. Predictions are off by default.

## 6. Python module reference

Most users should call the scripts/notebooks. These modules contain their implementation:

| Module | Function |
|---|---|
| `stage1/config.py` | CLI/YAML parsing and validation |
| `stage1/data.py` | DTOF/OP loading, split, preprocessing, grouped sampling |
| `stage1/model.py` | DTOF subject encoder |
| `stage1/losses.py` | contrastive and compactness losses |
| `stage1/training.py` | training, model selection, early stopping |
| `stage1/metrics.py` | representation/gallery metrics |
| `stage1/runner.py` | complete one-fold workflow |
| `stage1/exporter.py` | full-OP feature export |
| `stage1/evaluation.py` | notebook 01 retrieval evaluation |
| `stage1/cka.py`, `cka_plotting.py` | notebook 02 CKA compute/plots |
| `stage1/probe.py` | notebook 03 nested ridge probe |
| `stage1/visualization.py`, `utils.py` | plots and shared helpers |
| `stage2/config.py` | defaults, YAML overrides, validation, run naming |
| `stage2/cache.py`, `compact_cache.py` | HDF5 preprocessing and compact caches |
| `stage2/datasets.py`, `sampling.py` | iterable data and deterministic sampling |
| `stage2/model.py` | baseline, gated residual, and FiLM models |
| `stage2/latent.py` | DTOF/metadata feature loading and controls |
| `stage2/engine.py`, `metrics.py` | training/evaluation/prediction and metrics |
| `stage2/loso.py` | 154-fold orchestration and resume audit |
| `stage2/baseline_analysis.py` | notebook 04 analyses |
| `stage2/comparison.py` | notebook 05 comparisons |
| `stage2/subject_failure.py` | notebook 06 analysis |
| `stage2/unseen_scattering.py` | notebook 07 direct-MAT inference |
| `stage2/unseen_scattering_plot.py` | notebook 07 violin plots |
| `stage2/utils.py` | serialization and shared helpers |

Examples of CLI help:

```bash
python -m subject_nirs.stage1.evaluation --help
python -m subject_nirs.stage1.cka --help
python -m subject_nirs.stage1.probe --help
python -m subject_nirs.stage2.comparison --help
python -m subject_nirs.stage2.unseen_scattering --help
```

## 7. What belongs on GitHub?

Commit:

- `README.md`, `.gitignore`, `pyproject.toml`;
- `configs/`, `scripts/`, `src/`, `tests/`, `notebooks/`;
- `data/README.md`, `results/README.md`;
- selected final PNGs under `results/figures/` that help readers understand the results.

Do **not** commit:

- real datasets/metadata under `data/`;
- `artifacts/` checkpoints, prediction NPZs, or logs;
- `cache/` preprocessing/signature caches;
- rebuildable CSV/JSON/HDF5 files under `results/tables/`;
- PDF duplicates, `.history/`, notebook checkpoints, `.venv`, `__pycache__`, or editor settings.

Verify the exclusions before the first commit:

```bash
git status --short --ignored
git check-ignore -v data/NIRS_Absolute_Val_Dataset_new.mat
git check-ignore -v artifacts/stage1/0/relat_cons_32/best_model.pt
git check-ignore -v cache/stage2/preprocessed_cache_grid_50k
```

## 8. First GitHub push

Create an empty GitHub repository without a remote README/`.gitignore`, then run from `final_refactored`:

```bash
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
git commit -m "Initial refactored thesis pipeline"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Use `git add .` only after confirming from `git status --short --ignored` that data, artifacts, caches, and table outputs remain ignored. If `origin` already exists:

```bash
git remote set-url origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
```

Later updates:

```bash
git status
git add README.md configs scripts src tests notebooks results/figures
git diff --cached
git commit -m "Describe the change"
git push
```

Never use `git add -f` for ignored data/artifacts. To unstage an accidentally added large file without deleting it:

```bash
git restore --staged path/to/file
```

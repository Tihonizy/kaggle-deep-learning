# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

PyTorch solutions to Kaggle computer vision competitions, written as Jupyter notebooks. There is no package,
build, linter or test suite: all code lives in `.ipynb` files, one folder per competition. Each folder has its own
`README.md` describing every notebook and the project result; the root `README.md` (plus its Russian copy
`README.ru.md`) has a summary table and the repo layout.

## Setup and data

```bash
pip install -r requirements.txt
kaggle competitions download -c digit-recognizer -p digit-recognizer/data
kaggle competitions download -c tpu-getting-started -p petals-to-the-metal/data
```

Notebooks unzip `data/<competition>.zip` themselves on first run. `data/`, `weights/`, `outputs/`, model files
(`*.pth`, `*.pt`, `*.ckpt`, `*.npy`) and `submission*.csv` are git-ignored. RSNA Knee notebooks run only on Kaggle
(GPU, data from `/kaggle/input`); the other two projects run locally and on Kaggle.

To execute a notebook headlessly: `jupyter nbconvert --to notebook --execute --inplace <path>.ipynb`
(run from inside the project folder — paths like `data/` are relative to it).

## Notebook conventions

- Names are `NN_short_description.ipynb`, `NN` = order written; each notebook builds on the previous one, so the
  folder reads as a progression of experiments. Kaggle-only notebooks get a `_kaggle` suffix.
- Folder is named after the competition slug, or the title when the slug is generic
  (`petals-to-the-metal` = competition `tpu-getting-started`).
- Notebooks that run in both places start with an **Environment** cell holding a single `ON_KAGGLE = False` flag.
  It sets `DATA_DIR`, `OUT_DIR`, `NUM_WORKERS` (and for Petals `FLOWERS_DIR` plus `pip install tfrecord`).
  Keep it that way — there are no separate Kaggle copies of notebooks (earlier separate GPU/TPU copies were removed).
- `NUM_WORKERS = 0` locally on purpose: on macOS, DataLoader workers cannot see classes defined in a notebook.
- Device is picked as MPS → CUDA → CPU; local development is on Apple Silicon (MPS), so avoid CUDA-only code paths
  unless guarded.
- Notebooks are self-contained (config, dataset, model, train loop, inference, submission in one file); code is not
  shared across notebooks via modules. Exception: RSNA uses `%%writefile knee_dataset.py` (git-ignored) so
  DataLoader workers can import the dataset.
- Markdown cells explain each step and what was tried/failed; update the project `README.md` notebook table and
  the root README result/layout when adding or renaming notebooks (the root README layout currently lags the
  petals renames: `02_baseline_cnn_large`, `03_cnn_ensemble_tta`).

## Project specifics

- **digit-recognizer**: 28x28 MNIST CSVs, custom CNNs (BatchNorm/residual/SE), `StratifiedKFold` with `N_SPLITS`,
  fold ensemble + TTA in `03`.
- **petals-to-the-metal**: 104-class flowers, macro F1, heavy class imbalance. `.tfrec` files are converted once to
  `.jpg` + `.csv` (via `tfrecord`, no TensorFlow) and read lazily; best epoch chosen by validation macro F1.
  `03_cnn_ensemble_tta.ipynb` is work in progress — its plan cell lists the intended upgrades (ensemble of fine-tuned
  timm models + custom CNN, TTA, 331px, stratified K-fold on train+val, AMP).
- **rsna-knee-abnormality-detection**: 12-label knee MRI, macro ROC AUC. Pipeline is split across notebooks:
  `03_build_cache_kaggle` builds a per-study uint8 slice cache on a Kaggle CPU session, `05_pack_cache_kaggle`
  packs it, `02_report_labels_mil` trains (DINOv2 + per-finding attention, labels extracted from multilingual
  reports), `04_kaggle_submission` does inference. Code in `02` is commented in Russian.

## Git

Conventional-commit style with project scope, e.g. `feat(petals): ...`, `fix(digit-recognizer): ...`,
`docs(knee-mri): ...`, `exp(...)` for experiment runs.

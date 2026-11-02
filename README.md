# Kaggle Deep Learning

PyTorch solutions to Kaggle computer vision competitions. Each competition has its own
folder, and notebooks are numbered in the order they were built, so the progression of
each approach (what was tried, what failed and why) can be followed from start to finish.

[Русская версия](README.ru.md) · Kaggle profile: [tihon2010](https://www.kaggle.com/tihon2010)

## Projects

| Project | Task | Metric | Approach | Result |
|---|---|---|---|---|
| [Digit Recognizer](digit-recognizer) | Classify 28x28 handwritten digits into 10 classes | Accuracy | Custom CNNs with BatchNorm, residual and squeeze-and-excitation blocks, 5-fold CV, fold ensemble with test-time augmentation | 0.99692 public leaderboard accuracy (0.9966 mean validation accuracy over 5 folds) |
| [Petals to the Metal](petals-to-the-metal) | Classify 224x224 flower photos into 104 classes | Macro F1 | Custom CNNs trained from scratch (0.86M and 4.8M parameters) with augmentation and cosine schedule, best epoch by validation F1; the same notebook runs locally, on a Kaggle GPU and on a Kaggle TPU (PyTorch/XLA) for the submission | 0.668 public leaderboard macro F1 (baseline) |
| [RSNA Knee Abnormality Detection](rsna-knee-abnormality-detection) | Score 12 findings (ligament and meniscus tears, osteoarthritis, effusion and others) for each knee MRI study | Macro ROC AUC over 12 labels | Labels derived from multilingual radiology reports, multi-series DICOM preprocessing, DINOv2 encoder with per-finding attention over slice windows | 0.69103 private leaderboard macro ROC AUC |

## Repository layout

```
.
├── digit-recognizer/
│   ├── 01_baseline_cnn.ipynb
│   ├── 02_modern_cnn_cv.ipynb
│   ├── 03_cnn_ensemble_tta.ipynb
│   └── README.md
├── petals-to-the-metal/
│   ├── 01_baseline_cnn.ipynb
│   ├── 01_baseline_cnn_large.ipynb
│   └── README.md
├── rsna-knee-abnormality-detection/
│   ├── 01_baseline_dinov2.ipynb
│   ├── 02_report_labels_mil.ipynb
│   ├── 03_build_cache_kaggle.ipynb
│   ├── 04_kaggle_submission.ipynb
│   ├── 05_pack_cache_kaggle.ipynb
│   └── README.md
├── requirements.txt
└── LICENSE
```

Conventions:

- One folder per competition, named after the Kaggle competition slug, or after the competition title when the
  slug is generic (Petals to the Metal is `tpu-getting-started`).
- Notebooks are named `NN_short_description.ipynb`, where `NN` is the order in which they were written. A notebook
  that has to run on Kaggle detects it and switches paths and device itself, so there are no separate Kaggle copies.
  Notebooks that only make sense on Kaggle get a `_kaggle` suffix, for example `03_build_cache_kaggle.ipynb`.
- Competition data goes to `<project>/data/` and model weights to `<project>/weights/`. Both are excluded from git.

## Running the notebooks

```bash
pip install -r requirements.txt
```

Digit Recognizer runs locally. Download the data with the Kaggle CLI into the project folder:

```bash
kaggle competitions download -c digit-recognizer -p digit-recognizer/data
```

Petals to the Metal runs locally as well. The same notebooks are uploaded to Kaggle to make the submission, where
they run on a GPU or a TPU:

```bash
kaggle competitions download -c tpu-getting-started -p petals-to-the-metal/data
```

RSNA Knee Abnormality Detection is a code competition with about 4,400 MRI studies, so its
notebooks are written to run on Kaggle with a GPU and read the data from `/kaggle/input`.

## License

[MIT](LICENSE)

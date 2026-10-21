# Kaggle Deep Learning

PyTorch solutions to Kaggle computer vision competitions. Each competition has its own
folder, and notebooks are numbered in the order they were built, so the progression of
each approach (what was tried, what failed and why) can be followed from start to finish.

[Русская версия](README.ru.md)

## Projects

| Project | Task | Metric | Approach | Result |
|---|---|---|---|---|
| [Digit Recognizer](digit-recognizer) | Classify 28x28 handwritten digits into 10 classes | Accuracy | Custom CNNs with BatchNorm, residual and squeeze-and-excitation blocks, 5-fold CV, fold ensemble with test-time augmentation | 0.9966 mean validation accuracy over 5 folds |
| [RSNA Knee Abnormality Detection](rsna-knee-abnormality-detection) | Score 12 findings (ligament and meniscus tears, osteoarthritis, effusion and others) for each knee MRI study | Macro ROC AUC over 12 labels | Labels derived from multilingual radiology reports, multi-series DICOM preprocessing, DINOv2 encoder with per-finding attention over slice windows | In progress |

## Repository layout

```
.
├── digit-recognizer/
│   ├── 01_baseline_cnn.ipynb
│   ├── 02_modern_cnn_cv.ipynb
│   ├── 03_cnn_ensemble_tta.ipynb
│   └── README.md
├── rsna-knee-abnormality-detection/
│   ├── 01_baseline_dinov2.ipynb
│   ├── 02_report_labels_mil.ipynb
│   └── README.md
├── requirements.txt
└── LICENSE
```

Conventions:

- One folder per competition, named after the Kaggle competition slug.
- Notebooks are named `NN_short_description.ipynb`, where `NN` is the order in which they were written.
- Competition data goes to `<project>/data/` and model weights to `<project>/weights/`. Both are excluded from git.

## Running the notebooks

```bash
pip install -r requirements.txt
```

Digit Recognizer runs locally. Download the data with the Kaggle CLI into the project folder:

```bash
kaggle competitions download -c digit-recognizer -p digit-recognizer/data
```

RSNA Knee Abnormality Detection is a code competition with about 4,400 MRI studies, so its
notebooks are written to run on Kaggle with a GPU and read the data from `/kaggle/input`.

## License

[MIT](LICENSE)

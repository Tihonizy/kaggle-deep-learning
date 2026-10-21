# Digit Recognizer

Kaggle competition: [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer).
The task is to classify 28x28 grayscale images of handwritten digits (MNIST) into 10 classes.
The training set has 42,000 labelled images and the test set 28,000. The metric is accuracy.

## Notebooks

| Notebook | What it adds | Validation |
|---|---|---|
| `01_baseline_cnn.ipynb` | Two convolutional blocks with max pooling, Adam, 5 epochs on the full training set | none, train accuracy 0.986 |
| `02_modern_cnn_cv.ipynb` | Deeper CNN (three blocks with BatchNorm, SiLU and dropout, global average pooling), rotation and affine augmentation, AdamW with cosine schedule, label smoothing, 5-fold stratified CV, test prediction averaged over folds | 0.9966 mean best-epoch accuracy over 5 folds |
| `03_cnn_ensemble_tta.ipynb` | Residual blocks with squeeze-and-excitation and GeM pooling, elastic augmentation, ensemble of two CNN families, 5-view test-time augmentation (original, two rotations, zoom in, zoom out). Pretrained timm ConvNeXt and EfficientNet were also tried | 0.9962 and 0.9961 mean best-epoch accuracy per model family |

Validation accuracy is reported for the best epoch of each fold, so it is slightly optimistic.

## Running

```bash
kaggle competitions download -c digit-recognizer -p data
```

The notebooks unzip `data/digit-recognizer.zip` into `data/digit_recognizer/`, save fold weights
to `weights/` and write `submission.csv` next to the notebook. None of these are tracked by git.

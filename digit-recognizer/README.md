# Digit Recognizer

Kaggle competition: [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer).
The task is to classify 28x28 grayscale images of handwritten digits (MNIST) into 10 classes.
The training set has 42,000 labelled images and the test set 28,000. The metric is accuracy.

## Notebooks

| Notebook | What it adds |
|---|---|
| `01_baseline_cnn.ipynb` | Two convolutional blocks with max pooling, Adam, 5 epochs on the full training set |
| `02_modern_cnn_cv.ipynb` | Deeper CNN (three blocks with BatchNorm, SiLU and dropout, global average pooling), rotation and affine augmentation, AdamW with cosine schedule, label smoothing, 5-fold stratified CV, test prediction averaged over folds |
| `03_cnn_ensemble_tta.ipynb` | Residual blocks with squeeze-and-excitation and GeM pooling, elastic augmentation, ensemble of two 5-fold runs, 5-view test-time augmentation (original, two rotations, zoom in, zoom out). Pretrained timm ConvNeXt and EfficientNet were also tried |

## Result

0.99692 accuracy on the Kaggle public leaderboard.

## Running

```bash
kaggle competitions download -c digit-recognizer -p data
```

The notebooks unzip `data/digit-recognizer.zip` into `data/digit_recognizer/`, save fold weights
to `weights/` and write `submission.csv` next to the notebook. None of these are tracked by git.

On Kaggle: set `ON_KAGGLE = True` in the first cell and add the competition data. The notebook then reads it from
`/kaggle/input` and writes the weights and `submission.csv` to `/kaggle/working`.

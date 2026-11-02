# Petals to the Metal

Kaggle competition: [Petals to the Metal - Flower Classification on TPU](https://www.kaggle.com/competitions/tpu-getting-started).
The task is to classify flower photos into 104 classes. The data comes as TFRecord files in four resolutions
(192, 224, 331 and 512 px); these notebooks use 224x224. The training set has 12,753 images, the validation set 3,712
and the test set 7,382. The classes are heavily imbalanced (18 to 782 training images per class), and the metric is
macro F1.

## Notebooks

| Notebook | What it adds |
|---|---|
| `01_baseline_cnn.ipynb` | TFRecords converted to `.jpg` files and a `.csv` with labels, lazy `Dataset`, light augmentation (random resized crop, flip, rotation, color jitter), custom CNN from scratch (four conv blocks with BatchNorm and SiLU, global average pooling, 0.86M parameters), AdamW with cosine schedule, 50 epochs, best epoch by validation macro F1 |
| `01_baseline_cnn_kaggle_gpu.ipynb` | The same notebook adapted to a Kaggle GPU to make the submission |
| `01_baseline_cnn_kaggle_tpu.ipynb` | The same notebook adapted to a Kaggle TPU (PyTorch/XLA on one TPU core) |
| `01_baseline_cnn_large.ipynb` | The same pipeline with a larger CNN: five conv blocks instead of four, channels 32-64-128-256-512 instead of 32-64-128-192, dropout 0.2 before the linear layer, 4.8M parameters instead of 0.86M. The model class takes `widths`, `convs_per_block` and `dropout`, so its size changes without editing the class |
| `01_baseline_cnn_large_kaggle_gpu.ipynb` | `01_baseline_cnn_large.ipynb` adapted to a Kaggle GPU |
| `01_baseline_cnn_large_kaggle_tpu.ipynb` | `01_baseline_cnn_large.ipynb` adapted to a Kaggle TPU |

## Result

0.668 macro F1 on the Kaggle public leaderboard.

## Running

Locally:

```bash
kaggle competitions download -c tpu-getting-started -p data
```

`01_baseline_cnn.ipynb` unzips `data/tpu-getting-started.zip` into `data/tpu-getting-started/`, converts the 224x224
TFRecords into `data/flowers-224/` and writes `submission.csv` next to the notebook. None of these are tracked by git.

On Kaggle: create a notebook from `01_baseline_cnn_kaggle_gpu.ipynb` (or `01_baseline_cnn_large_kaggle_gpu.ipynb`), add the competition data, choose a GPU
accelerator (T4 x2 or P100) and turn the internet on (the notebook installs `tfrecord`). The internet toggle needs a
phone-verified Kaggle account. The `_kaggle_tpu` notebooks work the same way with a TPU accelerator, but TPU
sessions often wait in a long queue. Every Kaggle notebook reads the data from `/kaggle/input` and
writes `/kaggle/working/submission.csv`.

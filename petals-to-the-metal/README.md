# Petals to the Metal

Kaggle competition: [Petals to the Metal - Flower Classification on TPU](https://www.kaggle.com/competitions/tpu-getting-started).
The task is to classify flower photos into 104 classes. The data comes as TFRecord files in four resolutions
(192, 224, 331 and 512 px); these notebooks use 224x224. The training set has 12,753 images, the validation set 3,712
and the test set 7,382. The classes are heavily imbalanced (18 to 782 training images per class), and the metric is
macro F1.

## Notebooks

| Notebook | What it adds | Validation |
|---|---|---|
| `01_baseline_cnn.ipynb` | TFRecords converted to `.jpg` files and a `.csv` with labels, lazy `Dataset`, light augmentation (random resized crop, flip, rotation, color jitter), custom CNN from scratch (four conv blocks with BatchNorm and SiLU, global average pooling, 0.86M parameters), AdamW with cosine schedule, 50 epochs, best epoch by validation macro F1 | 0.644 macro F1, 0.697 accuracy |
| `01_baseline_cnn_large.ipynb` | The same pipeline with a larger CNN: five conv blocks instead of four, channels 32-64-128-256-512 instead of 32-64-128-192, dropout 0.2 before the linear layer, 4.8M parameters instead of 0.86M. The model class takes `widths`, `convs_per_block` and `dropout`, so its size changes without editing the class | not run yet |

Balanced class weights in the loss were also tried with the baseline and lowered validation macro F1 from 0.64 to
0.42, so the final run uses an unweighted loss.

On Kaggle the baseline scored 0.649 macro F1 on the public leaderboard. A run with most of the augmentation removed
scored 0.668 (validation macro F1 0.687 instead of 0.648): the small model underfits even the augmented training set
(train macro F1 0.74), so heavier augmentation only makes it worse. The larger model is the next step to check this.

Validation scores are reported for the best epoch, so they are slightly optimistic.

## Running

Locally:

```bash
kaggle competitions download -c tpu-getting-started -p data
```

`01_baseline_cnn.ipynb` unzips `data/tpu-getting-started.zip` into `data/tpu-getting-started/`, converts the 224x224
TFRecords into `data/flowers-224/` and writes `submission.csv` next to the notebook. None of these are tracked by git.

On Kaggle: upload the same notebook, add the competition data, choose a GPU accelerator (T4 x2 or P100) or a TPU and
turn the internet on (the notebook installs `tfrecord`). The internet toggle needs a phone-verified Kaggle account.
The notebook sees `/kaggle/input` and switches to it, and on a TPU it uses PyTorch/XLA on one core. A GPU is enough for
this model; TPU sessions often wait in a long queue. On Kaggle the notebook writes `/kaggle/working/submission.csv`.

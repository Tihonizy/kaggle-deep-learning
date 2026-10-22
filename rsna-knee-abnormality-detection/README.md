# RSNA Knee Abnormality Detection

Kaggle code competition: RSNA Knee Abnormality Detection. Each knee MRI study has to be scored
for 12 findings: ACL and MCL injury, medial and lateral meniscal tear, osteoarthritis in the
medial, lateral and patellofemoral compartments, joint effusion, synovitis, Baker's cyst, bone
contusion and fracture. The metric is the unweighted mean of the 12 per-label ROC AUCs.

## Data

- 4,407 training studies with 24,371 DICOM series, 5.5 series per study on average
  (sagittal, coronal and axial planes, with and without fat suppression), 18 to 35 slices per series in most cases.
- Only 58 studies have the 12 labels filled in. Every study has the free-text radiology report,
  written in about nine languages (English, Spanish, Turkish, Greek, German, Dutch, French and others).
- The test set is hidden; the notebook is re-run on it at submission time.

## Notebooks

### 01_baseline_dinov2.ipynb

First attempt: three slices from the middle of the study as a pseudo-RGB image, DINOv2 ViT-B from
timm with the last two blocks unfrozen, BCE loss, 5-fold GroupKFold.

Validation macro AUC stayed at 0.5 to 0.56. The cause was in the labels, not the model:
missing labels were replaced with zeros, which turned 4,349 of 4,407 studies into negatives for
every finding. Two smaller problems added to it: slices were collected from all series of a study
at once and sorted by the z coordinate, which is not the slice axis for sagittal series, and each
epoch re-read the DICOM files from disk, which made a full run take hours.

### 02_report_labels_mil.ipynb

Second version, rebuilt around the findings above. The code is commented in Russian.

- **Labels.** A rule-based extractor reads each report clause by clause in nine languages and
  produces a graded score (asserted, uncertain, not mentioned, negated) and a confidence that is
  used as a sample weight. Public tables of LLM-read report labels are picked up automatically
  when attached. The 58 annotated studies are held out from training and used as the main validation metric.
- **Series selection.** Six slots per study (three planes, with and without fat suppression). A
  missing slot is filled with the other series of the same plane.
- **Preprocessing.** Slices are ordered by position along the slice normal, reoriented to a common
  orientation from `ImageOrientationPatient`, normalised by the 1st and 99th percentile of the series,
  cropped to 140 mm using `PixelSpacing` and resized to 224 px. Four windows of three adjacent slices
  are taken from each series between 8% and 92% of the stack.
- **Cache.** Every selected slice is decoded once into a uint8 array in memory, so training epochs do not touch the disk.
- **Model.** DINOv2 ViT-S encodes each window. Window embeddings get slot and position embeddings,
  pass through one transformer layer, and each of the 12 findings pools them with its own attention.
- **Training.** Weighted BCE plus a pairwise ranking loss, separate learning rates for the encoder
  and the head, OneCycle schedule, EMA of weights, mixed precision, affine and intensity augmentation,
  random left-right knee mirroring. Folds are grouped by a hash of the report text.
- **Inference.** Mirror test-time augmentation and a rank average over folds.

### 03_build_cache_kaggle.ipynb

Builds the training cache on a Kaggle CPU session, which does not use the weekly GPU quota, and
saves it as a dataset: about 15 GB instead of 570 GB of DICOM. It reuses the preprocessing functions
of notebook 02, and notebook 02 checks that the cache was built with the same settings.

## Where to run

Notebook 02 detects its environment:

| Mode | Input | What happens |
|---|---|---|
| Kaggle GPU from scratch | competition data | reads DICOM, builds the cache (about 1 hour), trains, writes `submission.csv` |
| Kaggle GPU with cache | competition data and the cache dataset from notebook 03 | loads the cache in minutes, then trains and predicts |
| Mac (Apple MPS) or any machine | contents of the cache dataset in `data/` next to the notebook | skips the DICOM cells, trains on the GPU through MPS, saves fold weights to `outputs/weights` |

The submission always runs on Kaggle with `RUN_TRAINING = False`, reading fold weights from a dataset.

```bash
pip install kaggle
kaggle datasets download -d <user>/rsna-knee-cache-224 -p rsna-knee-abnormality-detection/data --unzip
```

Results will be added after the full Kaggle run.

## Credits

The report label extractor is taken from the public Kaggle notebook `rsna-knee-baseline-v1`.
Several preprocessing and modelling ideas (held-out annotated studies, wide slice windows,
canonical orientation, mirror augmentation, per-finding attention pooling, slot filling) come from
the public notebook `rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` by goodpjw2008.

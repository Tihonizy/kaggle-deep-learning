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
timm with the last two blocks unfrozen, BCE loss, 5-fold GroupKFold. It now reads the same raw cache as
the second version: the sagittal series of each study and three slices around the centre of the stack.

Validation macro AUC stayed at 0.5 to 0.56. The cause was in the labels, not the model:
missing labels were replaced with zeros, which turned 4,349 of 4,407 studies into negatives for
every finding. Two smaller problems added to it: slices were collected from all series of a study
at once and sorted by the z coordinate, which is not the slice axis for sagittal series, and each
epoch re-read the DICOM files from disk, which made a full run take hours.

### 02_report_labels_mil.ipynb

Second version, rebuilt around the findings above, training on a Mac through Apple MPS from the raw cache
built by `03_build_cache_kaggle.ipynb`. The code is commented in Russian.

- **Labels.** A rule-based extractor reads each report clause by clause in nine languages and
  produces a graded score (asserted, uncertain, not mentioned, negated) and a confidence that is
  used as a sample weight. Public tables of LLM-read report labels are picked up automatically
  when attached. The 58 annotated studies are held out from training and used as the main validation metric.
- **Series selection.** Six slots per study (three planes, with and without fat suppression). A
  missing slot is filled with the other series of the same plane.
- **Preprocessing.** Slices are ordered by position along the slice normal, reoriented to a common
  orientation from `ImageOrientationPatient`, normalised by the 0.5 and 99.5 percentiles of the series,
  cropped to 140 mm using `PixelSpacing` and resized to 224 px. This part runs once, in the cache builder.
- **Dataset.** A standard PyTorch `Dataset` loads one study file and does everything else on the fly:
  fills missing slots, mirrors the knee, picks windows of three adjacent slices between 8% and 92% of the
  stack (two random windows per series in training, four fixed ones in evaluation), applies augmentation
  and ImageNet normalisation. A `DataLoader` with worker processes prepares the next batch while the GPU
  computes the current one.
- **Model.** DINOv2 ViT-S encodes each window. Window embeddings get slot and position embeddings,
  pass through one transformer layer, and each of the 12 findings pools them with its own attention.
- **Training.** Weighted BCE plus a pairwise ranking loss, separate learning rates for the encoder
  and the head, OneCycle schedule, EMA of weights, mixed precision. Folds are grouped by a hash of the
  report text.
- **Inference.** Mirror test-time augmentation and a rank average over folds.

### 03_build_cache_kaggle.ipynb

Builds a raw per-study cache on a Kaggle CPU session, which does not use the weekly GPU quota. For each
study it keeps one series per plane and fat suppression combination, all its slices (up to 32) in
anatomical order, reoriented, cropped to 140 mm, resized to 224 px and scaled to uint8, and saves them as
`studies/<StudyInstanceUID>.npz`. Window selection, slot filling, normalization and augmentation are left
to the `Dataset` class of the training notebooks. The output limit of 20 GB per run is met by building the
cache in two parts of about 11 GB each.

### 05_pack_cache_kaggle.ipynb

Packs one part of the cache into a single uncompressed tar file saved as `rsna_raw_partK.pack` (Kaggle unpacks `.tar` and `.zip` when creating a dataset), because Kaggle does
not always build a download archive for a dataset with thousands of small files. It checks that every
study listed in the index is present before packing.

### 04_kaggle_submission.ipynb

Inference only. Reads the fold weights from attached datasets, takes the preprocessing settings from
the checkpoints, converts the test DICOM files with the same functions as the cache builder, feeds them
through the same `Dataset` class and writes `submission.csv` as a rank average of all models.

## Workflow

| Step | Notebook | Where |
|---|---|---|
| 1. Build the cache | `03_build_cache_kaggle.ipynb` | Kaggle, CPU |
| 2. Pack each part of the cache into one file | `05_pack_cache_kaggle.ipynb` | Kaggle, CPU |
| 3. Download both `.pack` files and unpack them into `data/raw/` (`tar -xf rsna_raw_part0.pack -C data/raw`) | browser | local machine |
| 4. Train | `02_report_labels_mil.ipynb`, weights go to `outputs/weights` | Mac, Apple MPS |
| 5. Upload `outputs/weights` as a dataset and submit | `04_kaggle_submission.ipynb` | Kaggle, GPU |

Results will be added after the full Kaggle run.

## Credits

The report label extractor is taken from the public Kaggle notebook `rsna-knee-baseline-v1`.
Several preprocessing and modelling ideas (held-out annotated studies, wide slice windows,
canonical orientation, mirror augmentation, per-finding attention pooling, slot filling) come from
the public notebook `rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` by goodpjw2008.

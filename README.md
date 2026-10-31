# Credit Card Fraud Detection: Decision Trees, SVM and a Duplicate-Row Audit

A scikit-learn study of fraud detection on a highly imbalanced transaction dataset. It compares a Decision Tree, an RBF-kernel SVM and a class-weighted Decision Tree, and checks whether the evaluation is trustworthy by auditing exact duplicate rows and measuring train-test overlap.

All numbers below were produced by `credit_card_fraud_detection.py` on the full dataset. Where a statement is an interpretation rather than a measurement, it is labelled as such.

## Overview

- Three models compared on one stratified held-out test set per experimental setup, using fraud-class precision, recall, F1, Average Precision (AP) and confusion matrices.
- Decision Tree depth chosen by five-fold stratified cross-validation on the training data only.
- Threshold analysis using out-of-fold predictions on the training data only.
- Exact duplicate rows audited. Two setups compared: **Setup A** keeps duplicates; **Setup B** removes them before splitting. Setup B is the primary evaluation and Setup A a sensitivity analysis.

## Problem statement and financial relevance

Card fraud detection is a classification problem in which the rare class is the costly one. A bank loses money on fraud it misses, but every alert also has a cost: analyst time, declined legitimate payments and customer friction. A useful evaluation therefore has to show both sides of that trade-off, not a single headline score.

This project does not estimate monetary costs. It reports how many fraud cases each model catches (recall) and how many of its alerts are real (precision), so the trade-off is visible.

## Dataset

| Property | Value (measured) |
|---|---|
| File | `creditcard.csv` (not included in this repository) |
| Rows / columns | 284,807 / 31 |
| Columns | `Time`, `V1`-`V28`, `Amount`, `Class` |
| Fraud (`Class = 1`) | 492 (0.173%) |
| Missing values | 0 |
| Exact duplicate rows | 1,081 extra copies in 773 groups (largest group: 18 identical rows) |
| Duplicates that are fraud | 19 of the 1,081 |
| Groups with identical features but different labels | 0 |

Source: `[ADD THE LINK AND CITATION OF THE PAGE YOU DOWNLOADED THE DATASET FROM]`

The feature columns `V1`-`V28` have generic names, so they cannot be interpreted as named financial variables. `Time` and `Amount` are the only features with recognisable names. The project uses all 30 feature columns exactly as provided.

## Methodology and workflow

1. **Load and check.** Confirm shape and missing values; audit exact duplicates (count, class split, group sizes, label conflicts).
2. **Build two datasets.** Setup A is the file as downloaded. Setup B is `drop_duplicates()` on all columns, applied **before any split**, so every later step (cross-validation, threshold analysis, test evaluation) uses the same deduplicated rows.
3. **Split.** Stratified 80/20 train/test split, `random_state=42`, separately for each setup.
4. **Measure train-test overlap.** An exact join of every test row against the training rows on all 30 feature columns. This counts test rows that have an identical twin in training and whether the twin has the same label.
5. **Choose tree depth.** Five-fold stratified, shuffled cross-validation on the training data only, candidate depths 3, 5, 7, 10, 15, scored by mean fraud-class F1.
6. **Train three models** on the full training set.
7. **Threshold analysis (training data only).** Out-of-fold probabilities from the selected tree; pick the threshold with the best F1 (midpoint of the interval of equivalent thresholds); compare it with the default 0.5 using a paired per-fold F1 difference. The tuned threshold is called "supported" only if the average fold gain exceeds twice its standard error. This is a deliberately simple rule of thumb, not a formal test.
8. **Final evaluation.** Every model is scored once on the same test rows. The code checks that precision, recall and F1 agree with the confusion-matrix counts.
9. **Overlap-adjusted metrics.** The same predictions re-scored on only those test rows with no exact twin in training.

Fixed seeds (`random_state=42`) are used everywhere.

## Models

| Model | Configuration |
|---|---|
| Decision Tree | `max_depth` from cross-validation (5 in both setups), `random_state=42` |
| SVM | `StandardScaler` + `SVC(kernel="rbf", C=1.0)` in a `Pipeline`, other settings at scikit-learn defaults, not tuned |
| Class-weighted Decision Tree | Same depth as the tree, `class_weight="balanced"` |

The scaler sits inside the Pipeline, so it is fitted on training data only. The class-weighted tree reuses the depth selected for the unweighted tree, so the only intended difference between the two trees is the class weighting.

## Evaluation metrics

- **Precision**: of the transactions flagged as fraud, the share that really are fraud.
- **Recall**: of the real fraud, the share that was flagged.
- **F1**: harmonic mean of precision and recall at one threshold. It ignores true negatives and weights precision and recall equally.
- **Average Precision (AP)**: summarises the precision-recall curve over all thresholds, so it measures how well a model ranks fraud above normal transactions. It is computed from continuous scores (tree probabilities, SVM decision function), never from hard predictions. A model with no skill has AP equal to the fraud rate (about 0.0017).
- **Confusion matrix**: TP, FP, FN, TN counts behind every number above.
- **Accuracy** is reported for completeness only. A model that flags nothing as fraud scores 99.827% (Setup A) or 99.833% (Setup B).

Precision-recall curves are used instead of ROC curves because the false-positive rate divides by more than 56,000 normal test transactions, which makes large numbers of false alarms look small.

## Setup A versus Setup B

| | Setup A (duplicates kept) | Setup B (duplicates removed, primary) |
|---|---|---|
| Rows | 284,807 | 283,726 |
| Fraud (rate) | 492 (0.1727%) | 473 (0.1667%) |
| Training rows / fraud | 227,845 / 394 | 226,980 / 378 |
| Test rows / fraud | 56,962 / 98 | 56,746 / 95 |
| Test rows with an exact twin in training | 344 (8 fraud, 336 normal) | 0 |
| Twin label conflicts | 0 | 0 |
| Selected tree depth | 5 | 5 |

Removing duplicates deleted 19 fraud rows (3.9% of all fraud) but 1,062 normal rows (0.37% of normal). The two setups use different random splits, so their test numbers are two separate reports, not a paired experiment.

## Results

### Setup B (primary): duplicates removed before the split

| Model | Precision | Recall | F1 | AP | Accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| Decision Tree (depth 5) | 0.8816 | 0.7053 | 0.7836 | 0.6416 | 0.9993 | 67 | 9 | 28 | 56,642 |
| SVM (RBF, C=1.0) | 0.9821 | 0.5789 | 0.7285 | 0.7700 | 0.9993 | 55 | 1 | 40 | 56,650 |
| Class-weighted Tree (depth 5) | 0.0388 | 0.8526 | 0.0743 | 0.4502 | 0.9644 | 81 | 2,004 | 14 | 54,647 |

### Setup A (sensitivity analysis): duplicates kept

| Model | Precision | Recall | F1 | AP | Accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| Decision Tree (depth 5) | 0.8941 | 0.7755 | 0.8306 | 0.7402 | 0.9995 | 76 | 9 | 22 | 56,855 |
| SVM (RBF, C=1.0) | 0.9577 | 0.6939 | 0.8047 | 0.8552 | 0.9994 | 68 | 3 | 30 | 56,861 |
| Class-weighted Tree (depth 5) | 0.0474 | 0.8776 | 0.0899 | 0.4498 | 0.9694 | 86 | 1,729 | 12 | 55,135 |

### Setup A re-scored on test rows with no exact twin in training (56,618 rows, 90 fraud)

| Model | Precision (all test) | Precision (no twin) | Recall (all test) | Recall (no twin) | F1 (all test) | F1 (no twin) |
|---|---|---|---|---|---|---|
| Decision Tree | 0.8941 | 0.8831 | 0.7755 | 0.7556 | 0.8306 | 0.8144 |
| SVM | 0.9577 | 0.9524 | 0.6939 | 0.6667 | 0.8047 | 0.7843 |
| Class-weighted Tree | 0.0474 | 0.0434 | 0.8776 | 0.8667 | 0.0899 | 0.0827 |

In Setup B this table is identical to the all-test results, because no test row has a twin.

### Cross-validation for tree depth (mean fraud-class F1, training data only)

| Depth | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|
| Setup A | 0.7778 | **0.8327** | 0.8302 | 0.8112 | 0.7859 |
| Setup B | 0.7785 | **0.8324** | 0.8127 | 0.8030 | 0.7712 |

### Threshold analysis (decision tree)

| Setup | Chosen threshold | Validation F1 at 0.5 | Validation F1 at chosen | Paired fold gain (standard error) | Supported? | Test predictions changed |
|---|---|---|---|---|---|---|
| A | 0.2316 | 0.8329 | 0.8331 | +0.0003 (0.0033) | No | 7 (5 real fraud) |
| B | 0.5833 | 0.8291 | 0.8326 | +0.0035 (0.0023) | No | 0 |

Validation here means out-of-fold predictions on the training data; the test set was not used. Because the validation evidence did not support the tuned thresholds, **the default 0.5 threshold is the headline result in both setups.** For transparency: in Setup A the tuned threshold gave TP 81, FP 11, FN 17 (precision 0.8804, recall 0.8265, F1 0.8526) against the default's F1 of 0.8306. That difference comes from 7 changed predictions out of 56,962 and is not presented as an improvement.

### Figures

The script saves six figures per setup: `figures/B_deduplicated/` for Setup B (primary) and `figures/A_keep_duplicates/` for Setup A.

| File | Content |
|---|---|
| `01_class_distribution.png` | Normal vs fraud counts |
| `02_cv_depth_selection.png` | Cross-validated F1 by tree depth |
| `03_threshold_tuning_validation.png` | Precision, recall, F1 against threshold on out-of-fold training predictions |
| `04_precision_recall_curves.png` | Test-set precision-recall curves with AP |
| `05_confusion_matrices.png` | Confusion matrices on the same test rows |
| `06_precision_recall_f1_comparison.png` | Precision, recall and F1 side by side |

#### Setup B (primary): duplicates removed before the split

![Class distribution, Setup B](figures/B_deduplicated/01_class_distribution.png)

![Cross-validated F1 by tree depth, Setup B](figures/B_deduplicated/02_cv_depth_selection.png)

![Threshold analysis on out-of-fold training predictions, Setup B](figures/B_deduplicated/03_threshold_tuning_validation.png)

![Precision-recall curves on the test set, Setup B](figures/B_deduplicated/04_precision_recall_curves.png)

![Confusion matrices on the test set, Setup B](figures/B_deduplicated/05_confusion_matrices.png)

![Precision, recall and F1 by model, Setup B](figures/B_deduplicated/06_precision_recall_f1_comparison.png)

<details>
<summary><strong>Setup A (sensitivity analysis): duplicates kept</strong></summary>

![Class distribution, Setup A](figures/A_keep_duplicates/01_class_distribution.png)

![Cross-validated F1 by tree depth, Setup A](figures/A_keep_duplicates/02_cv_depth_selection.png)

![Threshold analysis on out-of-fold training predictions, Setup A](figures/A_keep_duplicates/03_threshold_tuning_validation.png)

![Precision-recall curves on the test set, Setup A](figures/A_keep_duplicates/04_precision_recall_curves.png)

![Confusion matrices on the test set, Setup A](figures/A_keep_duplicates/05_confusion_matrices.png)

![Precision, recall and F1 by model, Setup A](figures/A_keep_duplicates/06_precision_recall_f1_comparison.png)

</details>

## Key findings

**Measured**

- In Setup B the decision tree has the highest F1 (0.7836), the SVM the highest precision (0.9821, from a single false positive) and the highest AP (0.7700), and the class-weighted tree the highest recall (0.8526).
- The class-weighted tree gains recall by flagging far more transactions: 2,004 false positives in Setup B (precision 0.0388, so about 96% of its alerts are false), against 9 for the unweighted tree. Its AP (0.4502) is also lower than the unweighted tree's (0.6416).
- The SVM has a higher AP than the tree in both setups (0.7700 vs 0.6416 in B; 0.8552 vs 0.7402 in A) but a lower F1 at its default cut-off.
- In Setup A, 344 test rows (0.6%) had an exact twin in training, including 8 of the 98 test fraud cases. All four sets of predictions (the three models and the tuned-threshold tree) caught all 8, as implied by the printed recall figures.
- Scoring Setup A only on rows without a twin lowered the tree's F1 from 0.8306 to 0.8144 and the SVM's from 0.8047 to 0.7843.
- Setup B's scores are lower than Setup A's (tree F1 0.7836 vs 0.8306; SVM 0.7285 vs 0.8047).

**Interpretation**

- Duplicate rows produced a measurable but modest inflation in Setup A. The A-to-B gap is larger than the twin-adjusted gap, but the setups use different random splits, different test rows (98 vs 95 fraud cases) and different training data, so the gap cannot be attributed to leakage alone.
- The tree and the SVM sit at different operating points: the SVM is cautious (high precision, lower recall), the tree more balanced. Which is preferable depends on the relative cost of missed fraud and false alerts, which this project does not model.
- Higher recall alone does not make the class-weighted tree better; it moved along the precision-recall trade-off to a point with very low precision.
- The two setups share most of their rows, so agreement between them (for example the SVM's higher AP) is not independent confirmation.

## Limitations

- **Few fraud cases in the test set** (95 in Setup B). One fraud case moves recall by about one percentage point, and the SVM's precision of 0.9821 rests on one false positive. No confidence intervals are reported, so small differences between models should not be read as real.
- **Random split, not time-ordered.** Real systems predict future transactions from past ones. `Time` is also used as a model feature, and its effect was not examined.
- **Only exact duplicates were checked.** Near-duplicates were not examined. The `V1`-`V28` features are provided pre-transformed, so any preprocessing done before the data reached this project cannot be audited here.
- **Decision tree scores are coarse.** The depth-5 tree outputs 10 distinct score values on the test set (the SVM about 56,000), so its AP comes from a coarse precision-recall curve.
- **The SVM and the class-weighted tree were not tuned.** The SVM uses default `C` and `gamma`, and the weighted tree reuses the unweighted tree's depth. Comparisons across models are therefore not tuned-versus-tuned.
- **F1-based choices ignore real costs.** Depth and threshold were chosen by F1, which treats a missed fraud and a false alert as equally costly.
- **The threshold analysis covers the decision tree only**, and the "supported" rule is a rough heuristic.
- **Not implemented:** resampling methods, ensembles, time-based validation, bootstrap confidence intervals, cost-based thresholds.

## Project structure

```
credit_card_fraud_detection.py   full pipeline (both setups)
requirements.txt                 dependencies
README.md
.gitignore
figures/
  A_keep_duplicates/             plots for Setup A
  B_deduplicated/                plots for Setup B (primary)
results/
  duplicate_audit.md             duplicate counts and A-vs-B dataset table
  setup_comparison.md / .csv     test results for both setups
  A_keep_duplicates/             results.md, CV, threshold and overlap tables
  B_deduplicated/                results.md, CV, threshold and overlap tables
```

`creditcard.csv` must be downloaded separately and placed next to the script.

## How to run

```bash
pip install -r requirements.txt

# both setups (default); expects creditcard.csv in the current folder
python credit_card_fraud_detection.py

# other file location
python credit_card_fraud_detection.py path/to/creditcard.csv

# one setup only: "keep" (Setup A) or "dedup" (Setup B)
python credit_card_fraud_detection.py path/to/creditcard.csv dedup
```

The RBF SVM is the slowest step. For a quick test, set `RUN_SVM = False` near the top of the script.

The main result tables are also written to `results/`. Library versions for each run are recorded in `results/<setup>/results.md`.

## Libraries and dependencies

- Python (developed with 3.13.9)
- numpy, pandas, matplotlib, scikit-learn (listed in `requirements.txt`; the exact versions used in each run are recorded in `results/<setup>/results.md`)

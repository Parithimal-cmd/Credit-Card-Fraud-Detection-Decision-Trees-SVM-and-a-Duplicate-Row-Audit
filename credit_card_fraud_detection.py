# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 22:29:44 2026

@author: parit
"""

# ============================================================
# CREDIT CARD FRAUD DETECTION: DECISION TREES AND SVM  (audited version)
# ============================================================
#
# Research question:
#   On a highly imbalanced fraud dataset, how do a Decision Tree, an
#   RBF-kernel SVM and a class-weighted Decision Tree compare when judged
#   by fraud-class precision, recall, F1 and Average Precision (AP), and
#   what does moving the decision threshold do to false alarms vs missed fraud?
#
# What the audit changed (tags used in the comments below):
#   [UNCHANGED] same logic as the previous version
#   [CHANGED]   modified during the methodological audit (reason given)
#   [NEW]       added during the audit
#
# The two experimental setups (both are run by default):
#   A_keep_duplicates : the dataset exactly as downloaded (284,807 rows)
#   B_deduplicated    : exact duplicate rows removed BEFORE the train/test split
#   Within each setup, every model sees the same training rows and is scored on
#   the same test rows. The two setups use different rows, so their test numbers
#   are not a paired comparison (see the twin diagnostic below for that).
#
# How to run:
#   python credit_card_fraud_detection.py                          (creditcard.csv next to the script, both setups)
#   python credit_card_fraud_detection.py path/to/file.csv         (other location)
#   python credit_card_fraud_detection.py path/to/file.csv dedup   (only setup B; use "keep" for only A)
#
# Outputs:
#   figures/<setup>/*.png   and   results/<setup>/*  (tables + results.md)
#   results/duplicate_audit.md and results/setup_comparison.md (copy numbers into the README from these)
# ============================================================
 
 
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
 
import os
import sys
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
 
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    StratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)
 
 
# ============================================================
# 1b. SETTINGS  (nothing here changed from the previous version except SETUP)
# ============================================================
 
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "creditcard.csv"
SETUP = sys.argv[2] if len(sys.argv) > 2 else "both"   # [NEW] "keep", "dedup" or "both"
RANDOM_STATE = 42            # [UNCHANGED]
TEST_SIZE = 0.2              # [UNCHANGED]
TREE_DEPTHS = [3, 5, 7, 10, 15]   # [UNCHANGED]
N_FOLDS = 5                  # [UNCHANGED]
RUN_SVM = True               # set False for a quick test run; the RBF SVM is the slowest step
DEFAULT_THRESHOLD = 0.5      # the usual default for tree probabilities
 
if SETUP not in ("keep", "dedup", "both"):
    raise SystemExit('SETUP must be "keep", "dedup" or "both"')
 
 
def save_fig(path):
    """Save the current figure and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
 
 
def df_to_markdown(frame, float_cols=(), digits=4):
    """Tiny markdown-table writer so we do not need the 'tabulate' package."""
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            val = row[col]
            cells.append(f"{val:.{digits}f}" if col in float_cols else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
 
 
# ============================================================
# 2. DUPLICATE AUDIT  [NEW]
# ============================================================
# Question: are there exact duplicate rows, and which classes do they belong to?
# We only MEASURE here. Nothing is removed in this function.
 
def audit_duplicates(df):
    feature_cols = [c for c in df.columns if c != "Class"]
 
    extra_copy = df.duplicated(keep="first")            # 2nd, 3rd, ... copy of a row (this is the "1,081" count)
    n_extra = int(extra_copy.sum())
    n_extra_fraud = int(df.loc[extra_copy, "Class"].sum())
 
    in_group = df.duplicated(keep=False)                 # every row that has at least one identical twin
    group_sizes = df[in_group].groupby(list(df.columns), dropna=False).size()
 
    # Identical features but DIFFERENT labels: not counted by df.duplicated() because Class differs.
    feat_dup_rows = df[df.duplicated(subset=feature_cols, keep=False)]
    labels_per_group = feat_dup_rows.groupby(feature_cols, dropna=False)["Class"].nunique()
    n_conflicting_groups = int((labels_per_group > 1).sum())
 
    return {
        "n_rows": len(df),
        "n_fraud": int(df["Class"].sum()),
        "n_extra_copies": n_extra,
        "n_extra_copies_fraud": n_extra_fraud,
        "n_rows_in_duplicate_groups": int(in_group.sum()),
        "n_duplicate_groups": int(len(group_sizes)),
        "max_group_size": int(group_sizes.max()) if len(group_sizes) else 1,
        "n_conflicting_label_groups": n_conflicting_groups,
    }
 
 
def describe_setup(name, data):
    """One row of the A-versus-B dataset table."""
    return {
        "Setup": name,
        "Rows": len(data),
        "Fraud": int(data["Class"].sum()),
        "Normal": int((data["Class"] == 0).sum()),
        "Fraud rate (%)": round(100 * data["Class"].mean(), 4),
    }
 
 
# ============================================================
# 3. TRAIN-TEST "TWIN" DETECTOR  [NEW]
# ============================================================
# Question: how many TEST rows have an exact copy in the TRAINING set?
# Method: an exact join on all 30 feature columns (not a hash, so no collisions).
# Returns two True/False arrays aligned with the test rows:
#   has_twin       : a training row with identical features exists
#   has_twin_same  : ... and it also has the same label
 
def find_train_test_twins(X_train, y_train, X_test, y_test):
    cols = list(X_train.columns)
    test_feat = X_test.reset_index(drop=True)
 
    train_feat = X_train.drop_duplicates().assign(_twin=1)
    merged = test_feat.merge(train_feat, on=cols, how="left")
    assert len(merged) == len(test_feat), "join changed the number of test rows"
    has_twin = merged["_twin"].notna().to_numpy()
 
    train_feat_lab = X_train.assign(Class=y_train.values).drop_duplicates().assign(_twin=1)
    test_feat_lab = test_feat.assign(Class=y_test.values)
    merged2 = test_feat_lab.merge(train_feat_lab, on=cols + ["Class"], how="left")
    assert len(merged2) == len(test_feat_lab), "join changed the number of test rows"
    has_twin_same = merged2["_twin"].notna().to_numpy()
 
    return has_twin, has_twin_same
 
 
# ============================================================
# 4. METRIC HELPERS
# ============================================================
 
def evaluate(name, threshold_note, y_true, y_pred, y_score):
    """Fraud-class metrics (class 1) + confusion-matrix counts for one model."""
    y_true_arr = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true_arr, y_pred, zero_division=0)
    recall = recall_score(y_true_arr, y_pred)
    f1 = f1_score(y_true_arr, y_pred)
 
    # [NEW] Built-in consistency checks: the reported numbers must agree with the confusion matrix.
    assert tp + fn == int(np.sum(y_true_arr == 1)), "TP + FN must equal the number of real fraud cases"
    assert tp + fp + fn + tn == len(y_true_arr), "confusion matrix must cover every test row"
    assert np.isclose(recall, tp / (tp + fn))
    assert np.isclose(precision, tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    assert np.isclose(f1, 2 * tp / (2 * tp + fp + fn))
 
    n_scores = len(np.unique(y_score))
    print(f"  [{name}] distinct score values used for AP: {n_scores}")
 
    return {
        "Model": name,
        "Threshold": threshold_note,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AP": average_precision_score(y_true_arr, y_score),   # computed from continuous scores, never from hard labels
        "Accuracy": accuracy_score(y_true_arr, y_pred),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
 
 
# ============================================================
# 5. ONE FULL EXPERIMENT  (everything that used to be sections 3-11)
# ============================================================
# Called once per setup. Split -> twin check -> CV depth -> models -> threshold -> test evaluation -> plots.
 
def run_setup(data, setup_name):
    fig_dir = os.path.join("figures", setup_name)
    res_dir = os.path.join("results", setup_name)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
 
    print("\n" + "=" * 70)
    print(f"SETUP: {setup_name}")
    print("=" * 70)
 
    fraud_rate = data["Class"].mean()
    print("Rows:", len(data), "| fraud:", int(data["Class"].sum()), f"| fraud rate: {fraud_rate * 100:.4f}%")
    print(f"Accuracy of a model that predicts 'normal' every time: {(1 - fraud_rate) * 100:.3f}%")
 
    # --- Visualization 1: class distribution [UNCHANGED apart from the folder] ---
    class_counts = data["Class"].value_counts().reindex([0, 1])
    plt.figure(figsize=(7, 4))
    bars = plt.bar(["Normal", "Fraud"], class_counts.values)
    plt.title(f"Distribution of Normal and Fraudulent Transactions\n({setup_name})")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Transactions")
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{count:,}\n({count / len(data) * 100:.3f}%)", ha="center", va="bottom")
    plt.ylim(0, class_counts.max() * 1.18)
    save_fig(os.path.join(fig_dir, "01_class_distribution.png"))
 
    # --- Features, target, split [UNCHANGED] ---
    X = data.drop("Class", axis=1)
    y = data["Class"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Train size:", len(y_train), "| fraud in train:", int(y_train.sum()))
    print("Test size: ", len(y_test), "| fraud in test: ", int(y_test.sum()))
 
    # --- [NEW] How many test rows have an exact twin in the training set? ---
    has_twin, has_twin_same = find_train_test_twins(X_train, y_train, X_test, y_test)
    y_test_arr = y_test.to_numpy()
    twin_info = {
        "test_rows_with_twin": int(has_twin.sum()),
        "test_fraud_with_twin": int((has_twin & (y_test_arr == 1)).sum()),
        "test_normal_with_twin": int((has_twin & (y_test_arr == 0)).sum()),
        "twin_label_conflicts": int((has_twin & ~has_twin_same).sum()),
    }
    print("\nTRAIN-TEST OVERLAP (exact feature match):", twin_info)
 
    # --- Depth selection with 5-fold CV on training data only [UNCHANGED] ---
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
 
    print("\nCROSS-VALIDATION (training data only)")
    cv_rows = []
    for depth in TREE_DEPTHS:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_rows.append({"Depth": depth, "Mean CV F1": scores.mean(), "Std CV F1": scores.std()})
        print(f"Depth {depth:>2}: fold F1 = {np.round(scores, 4)} | mean = {scores.mean():.4f}")
 
    cv_df = pd.DataFrame(cv_rows)
    best_depth = int(cv_df.loc[cv_df["Mean CV F1"].idxmax(), "Depth"])
    print("Selected depth:", best_depth, "| best mean CV F1:", round(cv_df["Mean CV F1"].max(), 4))
    cv_df.to_csv(os.path.join(res_dir, "cv_depth_results.csv"), index=False)
 
    plt.figure(figsize=(7, 4))
    plt.errorbar(cv_df["Depth"], cv_df["Mean CV F1"], yerr=cv_df["Std CV F1"], marker="o", capsize=4)
    plt.axvline(best_depth, linestyle="--", color="grey", label=f"Selected depth = {best_depth}")
    plt.title(f"Decision Tree depth selection (5-fold CV on training data)\n({setup_name})")
    plt.xlabel("max_depth")
    plt.ylabel("Mean fraud-class F1 across folds (bars = 1 std)")
    plt.legend()
    save_fig(os.path.join(fig_dir, "02_cv_depth_selection.png"))
 
    # --- Train the models on the full training set [UNCHANGED] ---
    final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
    final_tree.fit(X_train, y_train)
 
    svm_model = None
    if RUN_SVM:
        svm_model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0))])
        svm_model.fit(X_train, y_train)
 
    balanced_tree = DecisionTreeClassifier(max_depth=best_depth, class_weight="balanced",
                                           random_state=RANDOM_STATE)
    balanced_tree.fit(X_train, y_train)
 
    # ------------------------------------------------------------------
    # Threshold selection, training data only
    # ------------------------------------------------------------------
    # Out-of-fold (OOF) probabilities: each training row is scored by a tree that did not see it.
    oof_proba = cross_val_predict(
        DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE),
        X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
 
    prec_v, rec_v, thr_v = precision_recall_curve(y_train, oof_proba)
    f1_v = 2 * prec_v[:-1] * rec_v[:-1] / np.maximum(prec_v[:-1] + rec_v[:-1], 1e-12)
    best_idx = int(np.argmax(f1_v))
 
    # [CHANGED] The previous version used the OOF score itself (e.g. 0.2632) as the threshold.
    # A tree only produces a few distinct scores, and the final tree (fitted on all training data)
    # has slightly different leaf scores than the fold trees, so a threshold sitting exactly on one
    # fold-tree score is arbitrary. Every threshold between the previous OOF score and this one
    # gives IDENTICAL out-of-fold predictions, so we take the midpoint of that interval.
    if best_idx > 0:
        tuned_threshold = float((thr_v[best_idx] + thr_v[best_idx - 1]) / 2)
    else:
        tuned_threshold = float(thr_v[best_idx])
 
    n_unique_oof_scores = len(np.unique(oof_proba))
    threshold_is_degenerate = n_unique_oof_scores < 5
 
    # [NEW] How much better is the tuned threshold than the default ON VALIDATION DATA?
    val_f1_default = f1_score(y_train, (oof_proba >= DEFAULT_THRESHOLD).astype(int))
    val_f1_tuned = f1_score(y_train, (oof_proba >= tuned_threshold).astype(int))
 
    # Paired per-fold comparison (same folds as above because the splitter has a fixed seed).
    fold_diffs = []
    for _, val_idx in cv.split(X_train, y_train):
        yv = y_train.iloc[val_idx]
        pv = oof_proba[val_idx]
        fold_diffs.append(f1_score(yv, (pv >= tuned_threshold).astype(int))
                          - f1_score(yv, (pv >= DEFAULT_THRESHOLD).astype(int)))
    mean_diff = float(np.mean(fold_diffs))
    se_diff = float(np.std(fold_diffs, ddof=1) / np.sqrt(len(fold_diffs)))
    # Rough rule of thumb: only call the tuned threshold "supported" if the average fold gain is
    # clearly larger than its own standard error. This is deliberately simple, not a formal test.
    threshold_supported = (mean_diff > 2 * se_diff) and (mean_diff > 0)
 
    print("\nTHRESHOLD TUNING (validation = out-of-fold predictions on training data)")
    print(f"Distinct validation probability values: {n_unique_oof_scores}")
    if threshold_is_degenerate:
        print("WARNING: the tree outputs very few distinct probabilities, so moving the threshold "
              "changes almost nothing.")
    print(f"Chosen threshold (midpoint rule): {tuned_threshold:.4f}")
    print(f"Validation F1 at default {DEFAULT_THRESHOLD}: {val_f1_default:.4f} | at chosen threshold: {val_f1_tuned:.4f}")
    print(f"Paired per-fold F1 gain (tuned - default): mean {mean_diff:+.4f}, standard error {se_diff:.4f}, "
          f"folds {np.round(fold_diffs, 4)}")
    print("Validation evidence supports the tuned threshold:", threshold_supported)
 
    sens_rows = []
    for t in [0.1, 0.25, DEFAULT_THRESHOLD, 0.75, 0.9, tuned_threshold]:
        pred_v = (oof_proba >= t).astype(int)
        tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_train, pred_v, labels=[0, 1]).ravel()
        sens_rows.append({
            "Threshold": round(t, 4),
            "Precision": precision_score(y_train, pred_v, zero_division=0),
            "Recall": recall_score(y_train, pred_v),
            "F1": f1_score(y_train, pred_v),
            "False positives": int(fp_v),
            "Missed fraud (FN)": int(fn_v),
        })
    sens_df = pd.DataFrame(sens_rows).drop_duplicates("Threshold").sort_values("Threshold")
    print("\nEffect of the threshold on validation data:")
    print(sens_df.to_string(index=False))
    sens_df.to_csv(os.path.join(res_dir, "threshold_sensitivity_validation.csv"), index=False)
 
    plt.figure(figsize=(7, 4))
    plt.plot(thr_v, prec_v[:-1], label="Precision")
    plt.plot(thr_v, rec_v[:-1], label="Recall")
    plt.plot(thr_v, f1_v, label="F1")
    plt.axvline(tuned_threshold, linestyle="--", color="grey", label=f"Chosen = {tuned_threshold:.3f}")
    plt.title(f"Threshold vs precision / recall / F1\n(validation = out-of-fold predictions, training data only; {setup_name})")
    plt.xlabel("Decision threshold on fraud probability")
    plt.ylabel("Score")
    plt.legend()
    save_fig(os.path.join(fig_dir, "03_threshold_tuning_validation.png"))
 
    # ------------------------------------------------------------------
    # Final evaluation on the test set: used once, same rows for every model
    # ------------------------------------------------------------------
    print("\nFINAL TEST-SET EVALUATION")
    tree_pred = final_tree.predict(X_test)
    tree_score = final_tree.predict_proba(X_test)[:, 1]
    balanced_pred = balanced_tree.predict(X_test)
    balanced_score = balanced_tree.predict_proba(X_test)[:, 1]
 
    # The tuned threshold is applied to the SAME tree scores with ">=" (the same rule used when choosing it).
    tuned_pred = (tree_score >= tuned_threshold).astype(int)
 
    tuned_note = f"{tuned_threshold:.3f} (validation-selected; " + \
                 ("supported" if threshold_supported else "NOT supported by validation") + ")"
 
    tree_name = f"Decision Tree (depth {best_depth})"
    rows = [evaluate(tree_name, "default (0.5)", y_test, tree_pred, tree_score)]
    operating_points = {tree_name: (tree_pred, tree_score)}
    preds_for_plots = [tree_pred]
    preds_for_overlap = {tree_name: tree_pred}
 
    if RUN_SVM:
        svm_pred = svm_model.predict(X_test)
        svm_score = svm_model.decision_function(X_test)     # continuous margin score: correct input for AP / PR curve
        rows.append(evaluate("SVM (RBF, C=1.0)", "default (score > 0)", y_test, svm_pred, svm_score))
        operating_points["SVM (RBF, C=1.0)"] = (svm_pred, svm_score)
        preds_for_plots.append(svm_pred)
        preds_for_overlap["SVM (RBF, C=1.0)"] = svm_pred
 
    bal_name = f"Class-weighted Tree (depth {best_depth})"
    rows.append(evaluate(bal_name, "default (0.5)", y_test, balanced_pred, balanced_score))
    operating_points[bal_name] = (balanced_pred, balanced_score)
    preds_for_plots.append(balanced_pred)
    preds_for_overlap[bal_name] = balanced_pred
 
    n_base_models = len(rows)
 
    tuned_name = "Decision Tree + tuned threshold"
    rows.append(evaluate(tuned_name, tuned_note, y_test, tuned_pred, tree_score))
    preds_for_plots.append(tuned_pred)
    preds_for_overlap[tuned_name] = tuned_pred
 
    results = pd.DataFrame(rows)
    results.insert(0, "Setup", setup_name)
    print(results.drop(columns="Setup").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDetailed classification report, final Decision Tree:")
    print(classification_report(y_test, tree_pred, digits=4))
    results.to_csv(os.path.join(res_dir, "model_comparison_test.csv"), index=False)
 
    # [NEW] Same predictions, scored on all test rows vs only rows WITHOUT an exact twin in training.
    clean = ~has_twin
    overlap_rows = []
    for model_name, pred in preds_for_overlap.items():
        overlap_rows.append({
            "Model": model_name,
            "Precision (all test)": precision_score(y_test_arr, pred, zero_division=0),
            "Precision (no twin)": precision_score(y_test_arr[clean], pred[clean], zero_division=0),
            "Recall (all test)": recall_score(y_test_arr, pred),
            "Recall (no twin)": recall_score(y_test_arr[clean], pred[clean]),
            "F1 (all test)": f1_score(y_test_arr, pred),
            "F1 (no twin)": f1_score(y_test_arr[clean], pred[clean]),
        })
    overlap_df = pd.DataFrame(overlap_rows)
    n_fraud_clean = int(y_test_arr[clean].sum())
    print(f"\nMetrics on test rows with NO exact twin in training ({int(clean.sum())} rows, {n_fraud_clean} fraud cases):")
    print(overlap_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    overlap_df.to_csv(os.path.join(res_dir, "twin_adjusted_metrics.csv"), index=False)
 
    # ------------------------------------------------------------------
    # Plots for the test-set results [UNCHANGED apart from titles and folder]
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 5.5))
    for name, (pred, score) in operating_points.items():
        p, r, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)
        line, = plt.plot(r, p, drawstyle="steps-post", label=f"{name}  (AP = {ap:.3f})")
        plt.scatter(recall_score(y_test, pred), precision_score(y_test, pred, zero_division=0),
                    color=line.get_color(), s=60, zorder=3, edgecolor="black")
    plt.scatter(recall_score(y_test, tuned_pred), precision_score(y_test, tuned_pred, zero_division=0),
                marker="*", s=200, color="gold", edgecolor="black", zorder=4,
                label=f"Tree at tuned threshold {tuned_threshold:.3f}")
    plt.axhline(y_test.mean(), linestyle=":", color="grey", label=f"No-skill baseline = {y_test.mean():.4f}")
    plt.title(f"Precision-Recall curves on the test set\n(dots = default-threshold operating points; {setup_name})")
    plt.xlabel("Recall (share of fraud caught)")
    plt.ylabel("Precision (share of alerts that are real fraud)")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower left", fontsize=8)
    save_fig(os.path.join(fig_dir, "04_precision_recall_curves.png"))
 
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.2 * n_plots, 4))
    for ax, title, pred in zip(np.atleast_1d(axes), results["Model"], preds_for_plots):
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Normal", "Fraud"],
                                                values_format="d", ax=ax, colorbar=False)
        ax.set_title(title, fontsize=9)
    plt.suptitle(f"Confusion matrices on the same test set ({setup_name})")
    save_fig(os.path.join(fig_dir, "05_confusion_matrices.png"))
 
    x_pos = np.arange(len(results))
    width = 0.25
    plt.figure(figsize=(max(8, 2.2 * len(results)), 4.5))
    plt.bar(x_pos - width, results["Precision"], width, label="Precision")
    plt.bar(x_pos, results["Recall"], width, label="Recall")
    plt.bar(x_pos + width, results["F1"], width, label="F1")
    plt.xticks(x_pos, [m.replace(" + ", "\n+ ").replace(" (", "\n(") for m in results["Model"]], fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Fraud-class score")
    plt.title(f"Precision, recall and F1 for the fraud class (test set; {setup_name})")
    plt.legend()
    save_fig(os.path.join(fig_dir, "06_precision_recall_f1_comparison.png"))
 
    # ------------------------------------------------------------------
    # results.md for this setup
    # ------------------------------------------------------------------
    metric_cols = ["Precision", "Recall", "F1", "AP", "Accuracy"]
    base = results.iloc[:n_base_models]
    tuned_row = results.iloc[n_base_models]
    tree_row = results.iloc[0]
    n_changed = int((tuned_pred != tree_pred).sum())
    n_changed_fraud = int(((tuned_pred != tree_pred) & (y_test_arr == 1)).sum())
 
    observations = [
        f"- Highest AP among the three models: **{base.loc[base['AP'].idxmax(), 'Model']}** (AP = {base['AP'].max():.4f}).",
        f"- Highest fraud F1 at default thresholds: **{base.loc[base['F1'].idxmax(), 'Model']}** (F1 = {base['F1'].max():.4f}).",
        f"- Highest fraud recall at default thresholds: **{base.loc[base['Recall'].idxmax(), 'Model']}** "
        f"(recall = {base['Recall'].max():.4f}) with {int(base.loc[base['Recall'].idxmax(), 'FP'])} false positives.",
        f"- Tuned vs default threshold for the same tree: FP {int(tree_row['FP'])} -> {int(tuned_row['FP'])}, "
        f"FN {int(tree_row['FN'])} -> {int(tuned_row['FN'])}, F1 {tree_row['F1']:.4f} -> {tuned_row['F1']:.4f}. "
        f"This comes from only {n_changed} changed predictions ({n_changed_fraud} of them real fraud).",
        f"- On validation data F1 was {val_f1_default:.4f} at the default vs {val_f1_tuned:.4f} at the chosen threshold "
        f"(paired per-fold gain {mean_diff:+.4f}, standard error {se_diff:.4f}). The validation evidence "
        f"{'supports' if threshold_supported else 'does NOT support'} preferring the tuned threshold, "
        f"so the default-threshold rows are the headline result.",
        f"- Test set: {len(y_test):,} transactions, {int(y_test.sum())} fraud cases; one extra missed fraud "
        f"changes recall by about {100 / int(y_test.sum()):.1f} percentage points.",
        f"- Test rows with an exact feature twin in training: {twin_info['test_rows_with_twin']} "
        f"({twin_info['test_fraud_with_twin']} fraud, {twin_info['test_normal_with_twin']} normal; "
        f"{twin_info['twin_label_conflicts']} twins with a different label).",
    ]
    if threshold_is_degenerate:
        observations.append("- NOTE: the tree outputs very few distinct probabilities, so threshold tuning is not informative.")
 
    with open(os.path.join(res_dir, "results.md"), "w") as f:
        f.write(f"# Results for setup {setup_name} (generated by credit_card_fraud_detection.py)\n\n")
        f.write(f"- Data file: `{DATA_PATH}` | rows used: {len(data):,} | fraud: {int(data['Class'].sum())} "
                f"({fraud_rate * 100:.4f}%)\n")
        f.write(f"- Split: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} stratified, random_state={RANDOM_STATE}\n")
        f.write(f"- Train fraud cases: {int(y_train.sum())} | test fraud cases: {int(y_test.sum())}\n")
        f.write(f"- Selected tree depth (5-fold CV, mean F1): {best_depth}\n")
        f.write(f"- Chosen threshold (validation = out-of-fold training predictions): {tuned_threshold:.4f}\n")
        f.write(f"- Accuracy of an 'always normal' model on the test set: {(1 - y_test.mean()) * 100:.3f}%\n")
        f.write(f"- Train-test overlap (exact feature match): {twin_info}\n")
        f.write(f"- Library versions: scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}\n\n")
        f.write("## Test-set comparison (fraud class)\n\n")
        f.write(df_to_markdown(results.drop(columns="Setup"), float_cols=metric_cols) + "\n\n")
        f.write(f"## Same predictions, all test rows vs rows with no exact twin in training "
                f"({int(clean.sum())} rows, {n_fraud_clean} fraud)\n\n")
        f.write(df_to_markdown(overlap_df, float_cols=[c for c in overlap_df.columns if c != "Model"]) + "\n\n")
        f.write("## Cross-validation for tree depth (training data only)\n\n")
        f.write(df_to_markdown(cv_df, float_cols=["Mean CV F1", "Std CV F1"]) + "\n\n")
        f.write("## Threshold sensitivity (validation data, tree)\n\n")
        f.write(df_to_markdown(sens_df, float_cols=["Precision", "Recall", "F1"]) + "\n\n")
        f.write("## Automatically generated observations\n\n")
        f.write("\n".join(observations) + "\n")
 
    print("\n" + "\n".join(observations))
    return results, twin_info
 
 
# ============================================================
# 6. MAIN: load data, audit duplicates, build the setups, run them
# ============================================================
 
def main():
    os.makedirs("results", exist_ok=True)
 
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Missing values in total:", int(df.isna().sum().sum()))
    print("\nClass counts:")
    print(df["Class"].value_counts().sort_index())
 
    # [NEW] Duplicate audit on the full file
    audit = audit_duplicates(df)
    print("\nDUPLICATE AUDIT (full file)")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    print("  P(a duplicated PAIR ends up split across train and test under a random 80/20 split) "
          f"~ {2 * (1 - TEST_SIZE) * TEST_SIZE:.2f}")
 
    # [NEW] Setup B: remove exact duplicate rows BEFORE any split. Every later step uses these rows.
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    setups = {"A_keep_duplicates": df, "B_deduplicated": df_dedup}
    table = pd.DataFrame([describe_setup(n, d) for n, d in setups.items()])
    print("\nDATASET FOR EACH SETUP")
    print(table.to_string(index=False))
 
    with open(os.path.join("results", "duplicate_audit.md"), "w") as f:
        f.write("# Duplicate-row audit (generated by credit_card_fraud_detection.py)\n\n")
        f.write("\n".join(f"- {k}: {v}" for k, v in audit.items()) + "\n\n")
        f.write(df_to_markdown(table) + "\n")
 
    to_run = {"keep": ["A_keep_duplicates"], "dedup": ["B_deduplicated"],
              "both": ["A_keep_duplicates", "B_deduplicated"]}[SETUP]
 
    all_results = []
    for name in to_run:
        res, _ = run_setup(setups[name], name)
        all_results.append(res)
 
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join("results", "setup_comparison.csv"), index=False)
    with open(os.path.join("results", "setup_comparison.md"), "w") as f:
        f.write("# Test-set results by setup (generated by credit_card_fraud_detection.py)\n\n")
        f.write("Different setups use different rows, so compare them as two reports, not as a paired experiment.\n\n")
        f.write(df_to_markdown(combined, float_cols=["Precision", "Recall", "F1", "AP", "Accuracy"]) + "\n")
 
    print("\nDone. Figures in 'figures/<setup>/', tables in 'results/'.")
 
 
if __name__ == "__main__":
    main()

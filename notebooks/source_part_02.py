
# %% [Cell 32]
# ============================================================
# PHASE 5.2 — STATISTICAL ROBUSTNESS ANALYSIS
# McNemar exact test + FDR correction
# ============================================================

import os
import math
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import binomtest
from sklearn.metrics import f1_score


# ============================================================
# 1. Paths
# ============================================================

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR /
    "full_robustness_predictions.csv"
)

RESULTS_PATH = (
    OUTPUT_DIR /
    "full_robustness_results.csv"
)

# ============================================================
# 2. Load predictions
# ============================================================

pred_df = pd.read_csv(
    PREDICTIONS_PATH
)

results_df = pd.read_csv(
    RESULTS_PATH
)

print("=" * 75)
print("STATISTICAL ROBUSTNESS ANALYSIS")
print("=" * 75)

print("Prediction rows:", len(pred_df))
print("Result rows:", len(results_df))

print("\nModels:")
print(pred_df["model"].unique())

print("\nSeeds:")
print(sorted(pred_df["seed"].unique()))

print("\nTransformations:")
print(pred_df["transformation"].unique())


# ============================================================
# 3. Helper: Benjamini-Hochberg FDR
# ============================================================

def benjamini_hochberg(
    p_values,
    alpha=0.05
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p_values)

    order = np.argsort(p_values)

    ranked = p_values[order]

    q_values_sorted = (
        ranked *
        n /
        np.arange(1, n + 1)
    )

    # Enforce monotonicity from right to left
    q_values_sorted = np.minimum.accumulate(
        q_values_sorted[::-1]
    )[::-1]

    q_values_sorted = np.clip(
        q_values_sorted,
        0,
        1
    )

    q_values = np.empty(
        n,
        dtype=float
    )

    q_values[order] = q_values_sorted

    significant = (
        q_values <= alpha
    )

    return q_values, significant


# ============================================================
# 4. Exact McNemar p-value
# ============================================================

def mcnemar_exact_pvalue(
    b,
    c
):
    """
    b = Original correct, transformed wrong
    c = Original wrong, transformed correct

    Under H0:
        b ~ Binomial(b+c, 0.5)
    """

    discordant = b + c

    if discordant == 0:
        return 1.0

    return binomtest(
        b,
        n=discordant,
        p=0.5,
        alternative="two-sided"
    ).pvalue


# ============================================================
# 5. Statistical comparison
# ============================================================

TRANSFORMATIONS = [
    t
    for t in pred_df["transformation"].unique()
    if t != "Original"
]

MODELS = sorted(
    pred_df["model"].unique()
)

SEEDS = sorted(
    pred_df["seed"].unique()
)


statistical_results = []


for model in MODELS:

    for seed in SEEDS:

        subset = pred_df[
            (pred_df["model"] == model) &
            (pred_df["seed"] == seed)
        ].copy()

        # ----------------------------------------------------
        # Original predictions
        # ----------------------------------------------------

        original = subset[
            subset["transformation"] == "Original"
        ][
            [
                "original_image_id",
                "true_label",
                "pred_label"
            ]
        ].copy()

        original = original.rename(
            columns={
                "pred_label":
                    "original_pred"
            }
        )

        original["original_correct"] = (
            original["original_pred"] ==
            original["true_label"]
        )

        original_f1 = f1_score(
            original["true_label"],
            original["original_pred"],
            average="macro"
        )

        # ----------------------------------------------------
        # Each transformation
        # ----------------------------------------------------

        for transformation in TRANSFORMATIONS:

            transformed = subset[
                subset["transformation"] ==
                transformation
            ][
                [
                    "original_image_id",
                    "true_label",
                    "pred_label"
                ]
            ].copy()

            transformed = transformed.rename(
                columns={
                    "pred_label":
                        "transformed_pred"
                }
            )

            merged = original.merge(
                transformed,
                on="original_image_id",
                how="inner",
                suffixes=(
                    "_orig",
                    "_trans"
                ),
                validate="one_to_one"
            )

            # Safety checks
            assert len(merged) == 128, (
                f"Expected 128 paired leaves, "
                f"got {len(merged)} for "
                f"{model} / {seed} / "
                f"{transformation}"
            )

            # ------------------------------------------------
            # Correctness
            # ------------------------------------------------

            original_correct = (
                merged["original_correct"]
            ).to_numpy()

            transformed_correct = (
                merged["transformed_pred"] ==
                merged["true_label_trans"]
            ).to_numpy()

            # ------------------------------------------------
            # Discordant pairs
            # ------------------------------------------------

            # Original correct -> transformed wrong
            b = int(
                np.sum(
                    original_correct &
                    ~transformed_correct
                )
            )

            # Original wrong -> transformed correct
            c = int(
                np.sum(
                    ~original_correct &
                    transformed_correct
                )
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            original_accuracy = (
                original_correct.mean()
            )

            transformed_accuracy = (
                transformed_correct.mean()
            )

            accuracy_delta = (
                transformed_accuracy -
                original_accuracy
            )

            consistency = np.mean(
                merged["original_pred"].to_numpy()
                ==
                merged["transformed_pred"].to_numpy()
            )

            transformed_f1 = f1_score(
                merged["true_label_trans"],
                merged["transformed_pred"],
                average="macro"
            )

            delta_f1 = (
                transformed_f1 -
                original_f1
            )

            p_value = (
                mcnemar_exact_pvalue(
                    b,
                    c
                )
            )

            statistical_results.append({

                "model":
                    model,

                "seed":
                    seed,

                "transformation":
                    transformation,

                "n_pairs":
                    len(merged),

                "original_accuracy":
                    original_accuracy,

                "transformed_accuracy":
                    transformed_accuracy,

                "accuracy_delta":
                    accuracy_delta,

                "original_macro_f1":
                    original_f1,

                "transformed_macro_f1":
                    transformed_f1,

                "delta_macro_f1":
                    delta_f1,

                "prediction_consistency":
                    consistency,

                "original_correct":
                    int(
                        original_correct.sum()
                    ),

                "transformed_correct":
                    int(
                        transformed_correct.sum()
                    ),

                "correct_to_wrong":
                    b,

                "wrong_to_correct":
                    c,

                "discordant_pairs":
                    b + c,

                "mcnemar_p":
                    p_value
            })


stats_df = pd.DataFrame(
    statistical_results
)


# ============================================================
# 6. Multiple-comparison correction
# ============================================================

q_values, significant = (
    benjamini_hochberg(
        stats_df["mcnemar_p"].to_numpy(),
        alpha=0.05
    )
)

stats_df["fdr_q"] = q_values
stats_df["significant_fdr_05"] = significant


# ============================================================
# 7. Sort by statistical evidence
# ============================================================

stats_df = stats_df.sort_values(
    [
        "fdr_q",
        "model",
        "seed",
        "transformation"
    ]
).reset_index(drop=True)


# ============================================================
# 8. Significant results
# ============================================================

significant_df = stats_df[
    stats_df["significant_fdr_05"]
].copy()

print("\n" + "=" * 75)
print("SIGNIFICANT PAIRED CHANGES AFTER FDR CORRECTION")
print("=" * 75)

print(
    f"Significant comparisons: "
    f"{len(significant_df)} / {len(stats_df)}"
)

if len(significant_df) > 0:

    display(
        significant_df[
            [
                "model",
                "seed",
                "transformation",
                "original_accuracy",
                "transformed_accuracy",
                "accuracy_delta",
                "delta_macro_f1",
                "correct_to_wrong",
                "wrong_to_correct",
                "prediction_consistency",
                "mcnemar_p",
                "fdr_q"
            ]
        ].head(50)
    )

else:

    print(
        "No comparison survived FDR correction."
    )


# ============================================================
# 9. Strongest degradations
# ============================================================

print("\n" + "=" * 75)
print("LARGEST MACRO-F1 DEGRADATIONS")
print("=" * 75)

worst = (
    stats_df
    .sort_values(
        "delta_macro_f1"
    )
    .head(20)
)

display(
    worst[
        [
            "model",
            "seed",
            "transformation",
            "delta_macro_f1",
            "accuracy_delta",
            "prediction_consistency",
            "correct_to_wrong",
            "wrong_to_correct",
            "fdr_q"
        ]
    ]
)


# ============================================================
# 10. Largest improvements
# ============================================================

print("\n" + "=" * 75)
print("LARGEST MACRO-F1 IMPROVEMENTS")
print("=" * 75)

best = (
    stats_df
    .sort_values(
        "delta_macro_f1",
        ascending=False
    )
    .head(20)
)

display(
    best[
        [
            "model",
            "seed",
            "transformation",
            "delta_macro_f1",
            "accuracy_delta",
            "prediction_consistency",
            "correct_to_wrong",
            "wrong_to_correct",
            "fdr_q"
        ]
    ]
)


# ============================================================
# 11. Aggregate transformation-level statistics
# ============================================================

aggregate = (
    stats_df
    .groupby(
        [
            "model",
            "transformation"
        ],
        as_index=False
    )
    .agg({

        "accuracy_delta":
            ["mean", "std"],

        "delta_macro_f1":
            ["mean", "std"],

        "prediction_consistency":
            ["mean", "std"],

        "correct_to_wrong":
            ["mean", "std"],

        "wrong_to_correct":
            ["mean", "std"],

        "mcnemar_p":
            ["mean"],

        "fdr_q":
            ["min"]
    })
)

aggregate.columns = [
    "_".join(col).strip("_")
    if isinstance(col, tuple)
    else col
    for col in aggregate.columns
]


# ============================================================
# 12. Save
# ============================================================

stats_path = (
    OUTPUT_DIR /
    "robustness_statistical_analysis.csv"
)

significant_path = (
    OUTPUT_DIR /
    "robustness_significant_results_fdr.csv"
)

aggregate_path = (
    OUTPUT_DIR /
    "robustness_statistical_aggregate.csv"
)

stats_df.to_csv(
    stats_path,
    index=False
)

significant_df.to_csv(
    significant_path,
    index=False
)

aggregate.to_csv(
    aggregate_path,
    index=False
)


# ============================================================
# 13. Final summary
# ============================================================

print("\n" + "=" * 75)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"Total paired comparisons: "
    f"{len(stats_df)}"
)

print(
    f"FDR-significant: "
    f"{len(significant_df)}"
)

print("\nSaved:")
print(stats_path)
print(significant_path)
print(aggregate_path)


# %% [Cell 33]
# ============================================================
# CACHE VALIDATION TARGET TRANSFORMATIONS
# Only 508 images
# ============================================================

import shutil
import time
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

VAL_ROBUSTNESS_ROOT = Path(
    "/content/brinjal_val_robustness"
)

if VAL_ROBUSTNESS_ROOT.exists():
    shutil.rmtree(VAL_ROBUSTNESS_ROOT)

VAL_ROBUSTNESS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_TRANSFORMS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]

val_target_df = df[
    (df["data_split"] == "val") &
    (df["preprocessing_technique"].isin(TARGET_TRANSFORMS))
].copy()

print("Validation images to cache:", len(val_target_df))

start = time.time()

for _, row in val_target_df.iterrows():

    src = DRIVE_ROOT / row["filename"]

    relative_path = Path(row["filename"])

    dst = (
        VAL_ROBUSTNESS_ROOT /
        relative_path
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(src, dst)

elapsed = time.time() - start

local_count = sum(
    1
    for p in VAL_ROBUSTNESS_ROOT.rglob("*")
    if p.is_file()
)

print("\n✅ Validation robustness cache complete")
print(f"Copied: {len(val_target_df)}")
print(f"Local files: {local_count}")
print(f"Time: {elapsed:.1f} sec")

# %% [Cell 34]
# ============================================================
# CACHE TRAIN AUGMENTATION VARIANTS
# Original + Grayscale + Rotate + Brightness
# ============================================================

import shutil
import time
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

TRAIN_AUG_ROOT = Path(
    "/content/brinjal_train_augmentation"
)

if TRAIN_AUG_ROOT.exists():
    shutil.rmtree(TRAIN_AUG_ROOT)

TRAIN_AUG_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_AUGS = [
    "Grayscale",
    "Rotate",
    "Brightness"
]

train_aug_df = df[
    (df["data_split"] == "train") &
    (
        df["preprocessing_technique"].isin(
            TRAIN_AUGS
        )
    )
].copy()

print(
    "Train augmentation images to cache:",
    len(train_aug_df)
)

start = time.time()

for _, row in train_aug_df.iterrows():

    src = DRIVE_ROOT / row["filename"]

    relative_path = Path(row["filename"])

    dst = (
        TRAIN_AUG_ROOT /
        relative_path
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        src,
        dst
    )

elapsed = time.time() - start

local_count = sum(
    1
    for p in TRAIN_AUG_ROOT.rglob("*")
    if p.is_file()
)

print("\n✅ Train augmentation cache complete")
print(f"Copied: {len(train_aug_df)}")
print(f"Local files: {local_count}")
print(f"Time: {elapsed:.1f} sec")

print("\nExpected:")
print("595 Grayscale")
print("595 Rotate")
print("595 Brightness")
print("Total = 1785")

# %% [Cell 35]
# ============================================================
# CACHE GAUSSIAN BLUR TRAINING TRANSFORMATION
# Used for the unrelated augmentation control
# ============================================================

import shutil
import time
from pathlib import Path
import pandas as pd

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

TRAIN_BLUR_ROOT = Path(
    "/content/brinjal_train_gaussian_blur"
)

if TRAIN_BLUR_ROOT.exists():
    shutil.rmtree(TRAIN_BLUR_ROOT)

TRAIN_BLUR_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# Ensure metadata is loaded
METADATA_PATH = DRIVE_ROOT / "metadata.csv"
df = pd.read_csv(METADATA_PATH)

BLUR_TRANSFORM = "Gaussian_Blur"

train_blur_df = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == BLUR_TRANSFORM)
].copy()

print(
    "Gaussian Blur training images:",
    len(train_blur_df)
)

assert len(train_blur_df) == 595, (
    f"Expected 595 Gaussian Blur images, "
    f"got {len(train_blur_df)}"
)

start = time.time()

for _, row in train_blur_df.iterrows():

    src = DRIVE_ROOT / row["filename"]

    relative_path = Path(row["filename"])

    dst = (
        TRAIN_BLUR_ROOT /
        relative_path
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        src,
        dst
    )

elapsed = time.time() - start

local_count = sum(
    1
    for p in TRAIN_BLUR_ROOT.rglob("*")
    if p.is_file()
)

print("\n✅ Gaussian Blur cache complete")
print("Expected:", len(train_blur_df))
print("Copied:", local_count)
print(f"Time: {elapsed:.1f} sec")

assert local_count == 595

# %% [Cell 36]
# ============================================================
# STEP 1B — CACHE ONLY THE 595 ORIGINAL TRAINING IMAGES
#
# This recreates the exact local cache expected by the
# existing final training pipeline.
#
# NO TRAINING
# ============================================================

import shutil
import time
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

METADATA_PATH = DRIVE_ROOT / "metadata.csv"

LOCAL_ROOT = Path(
    "/content/brinjal_original"
)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

df_local = pd.read_csv(
    METADATA_PATH
)

# ------------------------------------------------------------
# Select ONLY the 595 original training images
# ------------------------------------------------------------

train_original_cache_df = df_local[
    (df_local["data_split"] == "train") &
    (df_local["preprocessing_technique"] == "Original")
].copy()

print(
    "Original training images to cache:",
    len(train_original_cache_df)
)

assert len(train_original_cache_df) == 595

# ------------------------------------------------------------
# Recreate local cache
# ------------------------------------------------------------

if LOCAL_ROOT.exists():
    shutil.rmtree(
        LOCAL_ROOT
    )

LOCAL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Copy
# ------------------------------------------------------------

start = time.time()

for _, row in train_original_cache_df.iterrows():

    src = (
        DRIVE_ROOT /
        row["filename"]
    )

    relative_path = Path(
        row["filename"]
    )

    dst = (
        LOCAL_ROOT /
        relative_path
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        src,
        dst
    )

elapsed = time.time() - start

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

local_files = [
    p
    for p in LOCAL_ROOT.rglob("*")
    if p.is_file()
]

print("\n" + "=" * 70)
print("ORIGINAL TRAIN CACHE")
print("=" * 70)

print(
    "Expected:",
    595
)

print(
    "Copied  :",
    len(local_files)
)

print(
    f"Time    : {elapsed:.1f} sec"
)

assert len(local_files) == 595

print(
    "\n✅ Original training cache is ready."
)

print(
    "✅ NO TRAINING WAS PERFORMED."
)

# %% [Cell 37]
# ============================================================
# STEP 1B — FINAL CONTROL DATASET VERIFICATION
# NO TRAINING
# ============================================================

import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

META_PATH = BASE_DIR / "metadata.csv"

ORIGINAL_DIR = Path(
    "/content/brinjal_original"
)

BLUR_DIR = Path(
    "/content/brinjal_train_gaussian_blur"
)

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

metadata = pd.read_csv(
    META_PATH
)

CLASS_NAMES = sorted(
    metadata["class_label"].unique()
)

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

# ------------------------------------------------------------
# Metadata lookup
# ------------------------------------------------------------

meta_lookup = {}

for _, row in metadata.iterrows():

    filename = str(row["filename"])
    basename = os.path.basename(filename)

    meta_lookup[filename] = row
    meta_lookup[basename] = row


def get_metadata(path):

    path = Path(path)

    for candidate in [
        path.name,
        path.as_posix(),
        str(path)
    ]:

        if candidate in meta_lookup:
            return meta_lookup[candidate]

    return None


# ============================================================
# 1. ORIGINAL TRAINING
# ============================================================

original_rows = []

for path in ORIGINAL_DIR.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:
        continue

    row = get_metadata(path)

    if row is None:
        continue

    if str(row["data_split"]) != "train":
        continue

    if str(
        row["preprocessing_technique"]
    ).lower() != "original":
        continue

    original_rows.append({

        "path": str(path),

        "label":
            CLASS_TO_ID[
                row["class_label"]
            ],

        "class_name":
            row["class_label"],

        "original_image_id":
            row["original_image_id"],

        "transformation":
            "Original"

    })


train_original_control = pd.DataFrame(
    original_rows
)


# ============================================================
# 2. GAUSSIAN BLUR TRAINING
# ============================================================

blur_rows = []

for path in BLUR_DIR.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:
        continue

    row = get_metadata(path)

    if row is None:
        continue

    if str(row["data_split"]) != "train":
        continue

    if str(
        row["preprocessing_technique"]
    ) != "Gaussian_Blur":
        continue

    blur_rows.append({

        "path": str(path),

        "label":
            CLASS_TO_ID[
                row["class_label"]
            ],

        "class_name":
            row["class_label"],

        "original_image_id":
            row["original_image_id"],

        "transformation":
            "Gaussian_Blur"

    })


train_blur = pd.DataFrame(
    blur_rows
)


# ============================================================
# 3. BASIC CHECKS
# ============================================================

assert len(train_original_control) == 595
assert len(train_blur) == 595

# Duplicate control = same original images twice
train_duplicate_control = pd.concat(
    [
        train_original_control.copy(),
        train_original_control.copy()
    ],
    ignore_index=True
)

# Blur control = original + Gaussian Blur
train_blur_control = pd.concat(
    [
        train_original_control.copy(),
        train_blur.copy()
    ],
    ignore_index=True
)

assert len(train_duplicate_control) == 1190
assert len(train_blur_control) == 1190


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

original_counts = (
    train_original_control[
        "class_name"
    ]
    .value_counts()
    .sort_index()
)

duplicate_counts = (
    train_duplicate_control[
        "class_name"
    ]
    .value_counts()
    .sort_index()
)

blur_counts = (
    train_blur_control[
        "class_name"
    ]
    .value_counts()
    .sort_index()
)

expected_counts = (
    original_counts * 2
)

pd.testing.assert_series_equal(
    duplicate_counts,
    expected_counts,
    check_names=False
)

pd.testing.assert_series_equal(
    blur_counts,
    expected_counts,
    check_names=False
)


# ============================================================
# 5. LEAF CHECK FOR BLUR
# ============================================================

assert (
    train_blur[
        "original_image_id"
    ].nunique()
    == 595
)

assert (
    train_blur[
        "original_image_id"
    ].duplicated()
    .sum()
    == 0
)


# ============================================================
# 6. DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("STEP 1B — CONTROL DATASET VERIFICATION")
print("=" * 80)

print("\nClass mapping:")
print(CLASS_TO_ID)

print("\nOriginal baseline:")
print(
    len(train_original_control)
)

print(original_counts)

print("\nDuplicate-Original control:")
print(
    len(train_duplicate_control)
)

print(duplicate_counts)

print("\nGaussian-Blur control:")
print(
    len(train_blur_control)
)

print(blur_counts)

print("\n" + "=" * 80)
print("✅ ALL CONTROL DATASETS VERIFIED")
print("✅ 595 original samples")
print("✅ 1190 samples in each control")
print("✅ Exact class balance preserved")
print("✅ 595 unique leaves in Gaussian Blur")
print("✅ NO TRAINING PERFORMED")
print("=" * 80)

# %% [Cell 38]
# ============================================================
# FIX — ORIGINAL VALIDATION SET DIRECTLY FROM METADATA
#
# No recaching required.
# No training.
# ============================================================

from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

META_PATH = (
    DATASET_ROOT /
    "metadata.csv"
)

metadata = pd.read_csv(
    META_PATH
)

CLASS_NAMES = sorted(
    metadata["class_label"].unique()
)

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(
        CLASS_NAMES
    )
}

# ------------------------------------------------------------
# Original validation images only
# ------------------------------------------------------------

val_original = metadata[
    (metadata["data_split"] == "val") &
    (
        metadata["preprocessing_technique"]
        == "Original"
    )
].copy()

# Build actual image paths directly from metadata
val_original["path"] = (
    val_original["filename"]
    .apply(
        lambda x:
            str(DATASET_ROOT / x)
    )
)

# Rename / construct columns expected by training code
val_original["label"] = (
    val_original["class_label"]
    .map(CLASS_TO_ID)
)

val_original["class_name"] = (
    val_original["class_label"]
)

val_original["original_image_id"] = (
    val_original["original_image_id"]
)

val_original["transformation"] = (
    "Original"
)

# Keep only required columns
val_original = val_original[
    [
        "path",
        "label",
        "class_name",
        "original_image_id",
        "transformation"
    ]
].reset_index(drop=True)

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

print("=" * 80)
print("ORIGINAL VALIDATION SET")
print("=" * 80)

print(
    "Validation samples:",
    len(val_original)
)

print(
    "\nClass distribution:"
)

print(
    val_original[
        "class_name"
    ].value_counts()
)

# File existence check
missing = [
    p
    for p in val_original["path"]
    if not Path(p).exists()
]

print(
    "\nMissing image files:",
    len(missing)
)

assert len(val_original) == 127
assert len(missing) == 0

print(
    "\n✅ 127 original validation images ready."
)

print(
    "✅ No recaching required."
)

print(
    "✅ No training performed."
)

# %% [Cell 39]
# ============================================================
# STEP 1C
# GAUSSIAN-BLUR CONTROL | SEED 42
#
# This reproduces the FINAL PYTORCH training protocol
# used in the original ablation experiment.
#
# IMPORTANT:
# - No existing experiment is overwritten.
# - Locked test set is NOT used.
# - Checkpoint selection uses ORIGINAL validation loss only.
# ============================================================

import os
import gc
import random
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# ============================================================
# 1. CONFIG — MATCH ORIGINAL FINAL PYTORCH EXPERIMENT
# ============================================================

SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

MAX_EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 1e-4

LR_FACTOR = 0.5
LR_PATIENCE = 2
MIN_LR = 1e-7

EARLY_STOP_PATIENCE = 4


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

META_PATH = (
    BASE_DIR / "metadata.csv"
)

ORIGINAL_DIR = Path(
    "/content/brinjal_original"
)

BLUR_DIR = Path(
    "/content/brinjal_train_gaussian_blur"
)

CONTROL_DIR = (
    BASE_DIR /
    "augmentation_ablation_controls_pytorch"
)

CONTROL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ============================================================
# 4. METADATA
# ============================================================

metadata = pd.read_csv(
    META_PATH
)

CLASS_NAMES = sorted(
    metadata["class_label"].unique()
)

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(
        CLASS_NAMES
    )
}

print("Classes:")
print(CLASS_TO_ID)


# ============================================================
# 5. METADATA LOOKUP
# ============================================================

meta_lookup = {}

for _, row in metadata.iterrows():

    filename = str(
        row["filename"]
    )

    basename = os.path.basename(
        filename
    )

    meta_lookup[filename] = row
    meta_lookup[basename] = row


def get_metadata(path):

    path = Path(path)

    for candidate in [
        path.name,
        path.as_posix(),
        str(path)
    ]:

        if candidate in meta_lookup:
            return meta_lookup[candidate]

    return None


# ============================================================
# 6. COLLECT ORIGINAL TRAINING IMAGES
# ============================================================

def collect_original_train():

    rows = []

    for path in ORIGINAL_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        row = get_metadata(path)

        if row is None:
            continue

        if row["data_split"] != "train":
            continue

        if (
            str(row["preprocessing_technique"])
            .lower()
            != "original"
        ):
            continue

        rows.append({

            "path":
                str(path),

            "label":
                CLASS_TO_ID[
                    row["class_label"]
                ],

            "class_name":
                row["class_label"],

            "original_image_id":
                row["original_image_id"],

            "transformation":
                "Original"

        })

    return pd.DataFrame(rows)


train_original = (
    collect_original_train()
)

assert len(train_original) == 595


# ============================================================
# 7. COLLECT GAUSSIAN-BLUR TRAINING IMAGES
# ============================================================

def collect_blur_train():

    rows = []

    for path in BLUR_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        row = get_metadata(path)

        if row is None:
            continue

        if row["data_split"] != "train":
            continue

        if (
            str(
                row["preprocessing_technique"]
            )
            != "Gaussian_Blur"
        ):
            continue

        rows.append({

            "path":
                str(path),

            "label":
                CLASS_TO_ID[
                    row["class_label"]
                ],

            "class_name":
                row["class_label"],

            "original_image_id":
                row["original_image_id"],

            "transformation":
                "Gaussian_Blur"

        })

    return pd.DataFrame(rows)


train_blur = (
    collect_blur_train()
)

assert len(train_blur) == 595


# ============================================================
# 8. BUILD CONTROL TRAINING SET
# ============================================================

train_df = pd.concat(
    [
        train_original,
        train_blur
    ],
    ignore_index=True
)

assert len(train_df) == 1190

train_df = (
    train_df
    .sample(
        frac=1.0,
        random_state=SEED
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nTraining samples:",
    len(train_df)
)

print(
    train_df["class_name"]
    .value_counts()
)


# ============================================================
# 9. ORIGINAL VALIDATION SET
# ============================================================

val_original = metadata[
    (metadata["data_split"] == "val") &
    (
        metadata["preprocessing_technique"]
        == "Original"
    )
].copy()

val_original["path"] = (
    val_original["filename"]
    .apply(
        lambda x:
            str(BASE_DIR / x)
    )
)

val_original["label"] = (
    val_original["class_label"]
    .map(CLASS_TO_ID)
)

val_original["class_name"] = (
    val_original["class_label"]
)

val_original["transformation"] = (
    "Original"
)

val_original = val_original[
    [
        "path",
        "label",
        "class_name",
        "original_image_id",
        "transformation"
    ]
].reset_index(drop=True)

assert len(val_original) == 127

print(
    "Original validation:",
    len(val_original)
)


# ============================================================
# 10. IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ============================================================
# 11. DATASET
# ============================================================

class BrinjalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["path"]
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(
                image
            )

        label = int(
            row["label"]
        )

        return image, label


# ============================================================
# 12. DATALOADER
# ============================================================

def make_loader(
    dataframe,
    shuffle,
    seed
):

    dataset = BrinjalDataset(
        dataframe,
        transform=image_transform
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        generator=generator,

        persistent_workers=False
    )


train_loader = make_loader(
    train_df,
    shuffle=True,
    seed=SEED
)

val_loader = make_loader(
    val_original,
    shuffle=False,
    seed=SEED
)


# ============================================================
# 13. MODEL
# ============================================================

def build_model():

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model = models.mobilenet_v3_small(
        weights=weights
    )

    in_features = (
        model.classifier[3].in_features
    )

    model.classifier[3] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    for param in model.parameters():

        param.requires_grad = True

    return model


# ============================================================
# 14. EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    y_true = []
    y_pred = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            total_samples += (
                images.size(0)
            )

            preds = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    return {

        "loss":
            total_loss /
            total_samples,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "macro_precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "macro_recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred
            ).tolist()
    }


# ============================================================
# 15. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nDevice:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 16. MODEL / LOSS / OPTIMIZER
# ============================================================

model = build_model().to(
    device
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR
    )
)


# ============================================================
# 17. OUTPUT
# ============================================================

RUN_NAME = (
    "gaussian_blur_control_seed42"
)

CHECKPOINT_PATH = (
    CONTROL_DIR /
    f"{RUN_NAME}.pt"
)

HISTORY_PATH = (
    CONTROL_DIR /
    f"{RUN_NAME}_history.csv"
)

RESULT_PATH = (
    CONTROL_DIR /
    f"{RUN_NAME}_validation.csv"
)


# ============================================================
# 18. TRAIN
# ============================================================

best_val_loss = float(
    "inf"
)

best_epoch = 0

epochs_without_improvement = 0

history = []

print("\n" + "=" * 90)
print(
    "GAUSSIAN-BLUR CONTROL | SEED 42"
)
print("=" * 90)

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item()
            * images.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += (
            labels.size(0)
        )

    train_loss = (
        train_loss_sum /
        train_total
    )

    train_acc = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # ORIGINAL VALIDATION
    # --------------------------------------------------------

    val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device
    )

    val_loss = (
        val_metrics["loss"]
    )

    val_acc = (
        val_metrics["accuracy"]
    )

    val_f1 = (
        val_metrics["macro_f1"]
    )

    val_balacc = (
        val_metrics["balanced_accuracy"]
    )


    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # CHECKPOINT BY ORIGINAL VAL LOSS
    # --------------------------------------------------------

    if (
        val_loss <
        best_val_loss
    ):

        best_val_loss = val_loss

        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "strategy":
                    "Gaussian_Blur_Control",

                "augmentation":
                    "Gaussian_Blur",

                "seed":
                    SEED,

                "epoch":
                    epoch,

                "best_val_loss":
                    best_val_loss
            },

            CHECKPOINT_PATH
        )

        marker = "✅ BEST"

    else:

        epochs_without_improvement += 1

        marker = ""


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_acc,

        "val_macro_f1":
            val_f1,

        "val_balanced_accuracy":
            val_balacc,

        "learning_rate":
            current_lr

    })


    print(

        f"Epoch {epoch:02d}/20 | "

        f"Train Loss {train_loss:.4f} | "

        f"Train Acc {train_acc:.4f} | "

        f"Orig Val Loss {val_loss:.4f} | "

        f"Orig Val F1 {val_f1:.4f} | "

        f"Orig Val Acc {val_acc:.4f} | "

        f"LR {current_lr:.6f} "

        f"{marker}"

    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOP_PATIENCE
    ):

        print(
            "\n⏹ Early stopping"
        )

        break


# ============================================================
# 19. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 20. LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# 21. FINAL ORIGINAL VALIDATION METRICS
# ============================================================

final_metrics = evaluate_model(
    model,
    val_loader,
    criterion,
    device
)

result = {

    "strategy":
        "Gaussian_Blur_Control",

    "seed":
        SEED,

    "training_samples":
        len(train_df),

    "best_epoch":
        best_epoch,

    "best_val_loss":
        best_val_loss,

    "original_val_accuracy":
        final_metrics["accuracy"],

    "original_val_macro_f1":
        final_metrics["macro_f1"],

    "original_val_balanced_accuracy":
        final_metrics[
            "balanced_accuracy"
        ],

    "checkpoint":
        str(CHECKPOINT_PATH)

}

pd.DataFrame(
    [result]
).to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print(
    "✅ GAUSSIAN-BLUR CONTROL SEED 42 COMPLETE"
)
print("=" * 90)

print(
    "Training samples:",
    len(train_df)
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best original val loss:",
    f"{best_val_loss:.6f}"
)

print(
    "Original val Macro-F1:",
    f"{final_metrics['macro_f1']:.4f}"
)

print(
    "Original val Accuracy:",
    f"{final_metrics['accuracy']:.4f}"
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "\n⚠️ LOCKED TEST WAS NOT USED."
)

print("=" * 90)

# %% [Cell 40]
# ============================================================
# DUPLICATE-ORIGINAL CONTROL | SEED 42
#
# 595 Original + 595 duplicated Original = 1190 samples
#
# IMPORTANT:
# - Same MobileNetV3-Small
# - Same training recipe
# - Same original validation
# - Checkpoint selected by original validation loss
# - LOCKED TEST NOT USED
# ============================================================

SEED = 42

seed_everything(SEED)

# ------------------------------------------------------------
# TRAINING DATA
# ------------------------------------------------------------

duplicate_train_df = pd.concat(
    [
        train_original.copy(),
        train_original.copy()
    ],
    ignore_index=True
)

assert len(duplicate_train_df) == 1190

duplicate_train_df = (
    duplicate_train_df
    .sample(
        frac=1.0,
        random_state=SEED
    )
    .reset_index(drop=True)
)

print("=" * 90)
print("DUPLICATE-ORIGINAL CONTROL | SEED 42")
print("=" * 90)

print(
    "Training samples:",
    len(duplicate_train_df)
)

print(
    duplicate_train_df[
        "class_name"
    ].value_counts()
)

# ------------------------------------------------------------
# DATALOADER
# ------------------------------------------------------------

duplicate_train_loader = make_loader(
    duplicate_train_df,
    shuffle=True,
    seed=SEED
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = build_model().to(
    device
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR
    )
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

RUN_NAME = (
    "duplicate_original_control_seed42"
)

CHECKPOINT_PATH = (
    CONTROL_DIR /
    f"{RUN_NAME}.pt"
)

HISTORY_PATH = (
    CONTROL_DIR /
    f"{RUN_NAME}_history.csv"
)

RESULT_PATH = (
    CONTROL_DIR /
    f"{RUN_NAME}_validation.csv"
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

best_val_loss = float("inf")

best_epoch = 0

epochs_without_improvement = 0

history = []

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for images, labels in duplicate_train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item()
            * images.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += (
            labels.size(0)
        )

    train_loss = (
        train_loss_sum /
        train_total
    )

    train_acc = (
        train_correct /
        train_total
    )

    # --------------------------------------------------------
    # ORIGINAL VALIDATION
    # --------------------------------------------------------

    val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device
    )

    val_loss = val_metrics["loss"]
    val_acc = val_metrics["accuracy"]
    val_f1 = val_metrics["macro_f1"]
    val_balacc = val_metrics["balanced_accuracy"]

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "strategy":
                    "Duplicate_Original_Control",

                "augmentation":
                    "Duplicate_Original",

                "seed":
                    SEED,

                "epoch":
                    epoch,

                "best_val_loss":
                    best_val_loss
            },
            CHECKPOINT_PATH
        )

        marker = "✅ BEST"

    else:

        epochs_without_improvement += 1
        marker = ""

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_acc,

        "val_macro_f1":
            val_f1,

        "val_balanced_accuracy":
            val_balacc,

        "learning_rate":
            current_lr

    })

    print(

        f"Epoch {epoch:02d}/20 | "

        f"Train Loss {train_loss:.4f} | "

        f"Train Acc {train_acc:.4f} | "

        f"Orig Val Loss {val_loss:.4f} | "

        f"Orig Val F1 {val_f1:.4f} | "

        f"Orig Val Acc {val_acc:.4f} | "

        f"LR {current_lr:.6f} "

        f"{marker}"

    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOP_PATIENCE
    ):

        print(
            "\n⏹ Early stopping"
        )

        break

# ------------------------------------------------------------
# SAVE HISTORY
# ------------------------------------------------------------

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

# ------------------------------------------------------------
# LOAD BEST CHECKPOINT
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

final_metrics = evaluate_model(
    model,
    val_loader,
    criterion,
    device
)

result = {

    "strategy":
        "Duplicate_Original_Control",

    "seed":
        SEED,

    "training_samples":
        len(duplicate_train_df),

    "best_epoch":
        best_epoch,

    "best_val_loss":
        best_val_loss,

    "original_val_accuracy":
        final_metrics["accuracy"],

    "original_val_macro_f1":
        final_metrics["macro_f1"],

    "original_val_balanced_accuracy":
        final_metrics["balanced_accuracy"],

    "checkpoint":
        str(CHECKPOINT_PATH)
}

pd.DataFrame(
    [result]
).to_csv(
    RESULT_PATH,
    index=False
)

print("\n" + "=" * 90)
print("✅ DUPLICATE-ORIGINAL CONTROL SEED 42 COMPLETE")
print("=" * 90)

print(
    "Training samples:",
    len(duplicate_train_df)
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best original val loss:",
    f"{best_val_loss:.6f}"
)

print(
    "Original val Macro-F1:",
    f"{final_metrics['macro_f1']:.4f}"
)

print(
    "Original val Accuracy:",
    f"{final_metrics['accuracy']:.4f}"
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "\n⚠️ LOCKED TEST WAS NOT USED."
)

print("=" * 90)

# %% [Cell 41]
# ============================================================
# DUPLICATE-ORIGINAL CONTROL | SEED 1337
# ============================================================

SEED = 1337
seed_everything(SEED)

print("=" * 90)
print("DUPLICATE-ORIGINAL CONTROL | SEED 1337")
print("=" * 90)

# Same 595 original training images duplicated once
duplicate_train_df = pd.concat(
    [train_original.copy(), train_original.copy()],
    ignore_index=True
)

assert len(train_original) == 595
assert len(duplicate_train_df) == 1190

# Shuffle, but do NOT create any new visual variation
duplicate_train_df = duplicate_train_df.sample(
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)

print(f"Training samples: {len(duplicate_train_df)}")
print(duplicate_train_df["class_name"].value_counts())

# Same training recipe as previous control
train_loader = make_loader(
    duplicate_train_df,
    shuffle=True,
    seed=SEED
)

val_loader = make_loader(
    val_original,
    shuffle=False,
    seed=SEED
)

model = build_model()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
history = []

checkpoint_path = os.path.join(
    CONTROL_DIR,
    "duplicate_original_control_seed1337.pt"
)

history_path = os.path.join(
    CONTROL_DIR,
    "duplicate_original_control_seed1337_history.csv"
)

val_path = os.path.join(
    CONTROL_DIR,
    "duplicate_original_control_seed1337_validation.csv"
)

for epoch in range(1, 21):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device
    )

    val_loss = val_metrics["loss"]
    val_f1 = val_metrics["macro_f1"]
    val_acc = val_metrics["accuracy"]

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    is_best = val_loss < best_val_loss

    if is_best:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "strategy": "Duplicate_Original_Control",
                "augmentation": "Duplicate_Original"
            },
            checkpoint_path
        )

        best_flag = " ✅ BEST"
    else:
        patience_counter += 1
        best_flag = ""

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "learning_rate": current_lr
    })

    print(
        f"Epoch {epoch:02d}/20 | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Orig Val Loss {val_loss:.4f} | "
        f"Orig Val F1 {val_f1:.4f} | "
        f"Orig Val Acc {val_acc:.4f} | "
        f"LR {current_lr:.6f}"
        f"{best_flag}"
    )

    if patience_counter >= 4:
        print("\n⏹ Early stopping")
        break

# Save history
history_df = pd.DataFrame(history)
history_df.to_csv(history_path, index=False)

# Load best checkpoint and re-evaluate validation
best_checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(best_checkpoint["model_state_dict"])

best_val_metrics = evaluate_model(
    model,
    val_loader,
    criterion,
    device
)

pd.DataFrame([{
    "strategy": "Duplicate_Original_Control",
    "seed": SEED,
    "training_samples": len(duplicate_train_df),
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "original_val_accuracy": best_val_metrics["accuracy"],
    "original_val_macro_f1": best_val_metrics["macro_f1"],
    "original_val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
    "checkpoint": str(checkpoint_path)
}]).to_csv(
    val_path,
    index=False
)

print("\n" + "=" * 90)
print("✅ DUPLICATE-ORIGINAL CONTROL SEED 1337 COMPLETE")
print("=" * 90)
print(f"Training samples: {len(duplicate_train_df)}")
print(f"Best epoch: {best_epoch}")
print(f"Best original val loss: {best_val_loss:.6f}")
print(f"Original val Macro-F1: {best_val_metrics['macro_f1']:.4f}")
print(f"Original val Accuracy: {best_val_metrics['accuracy']:.4f}")
print(f"Checkpoint: {checkpoint_path}")
print("\n⚠️ LOCKED TEST WAS NOT USED.")
print("=" * 90)

# %% [Cell 42]
# ============================================================
# DUPLICATE-ORIGINAL CONTROL | SEED 2026
# ============================================================

SEED = 2026
seed_everything(SEED)

print("=" * 90)
print("DUPLICATE-ORIGINAL CONTROL | SEED 2026")
print("=" * 90)

# Same 595 original training images duplicated once
duplicate_train_df = pd.concat(
    [train_original.copy(), train_original.copy()],
    ignore_index=True
)

assert len(train_original) == 595
assert len(duplicate_train_df) == 1190

# Shuffle deterministically
duplicate_train_df = duplicate_train_df.sample(
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)

print(f"Training samples: {len(duplicate_train_df)}")
print(duplicate_train_df["class_name"].value_counts())

# Same loaders / pipeline (passing required SEED parameter)
train_loader = make_loader(
    duplicate_train_df,
    shuffle=True,
    seed=SEED
)

val_loader = make_loader(
    val_original,
    shuffle=False,
    seed=SEED
)

# Same model and training recipe
model = build_model()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

history = []

checkpoint_path = os.path.join(
    CONTROL_DIR,
    "duplicate_original_control_seed2026.pt"
)

history_path = os.path.join(
    CONTROL_DIR,
    "duplicate_original_control_seed2026_history.csv"
)

val_path = os.path.join(
    CONTROL_DIR,
    "duplicate_original_control_seed2026_validation.csv"
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
for epoch in range(1, 21):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Original validation ONLY (passing required device parameter)
    val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device
    )

    val_loss = val_metrics["loss"]
    val_f1 = val_metrics["macro_f1"]
    val_acc = val_metrics["accuracy"]

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # Checkpoint selection by ORIGINAL validation loss
    # --------------------------------------------------------
    is_best = val_loss < best_val_loss

    if is_best:

        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "strategy": "Duplicate_Original_Control",
                "augmentation": "Duplicate_Original"
            },
            checkpoint_path
        )

        best_flag = " ✅ BEST"

    else:

        patience_counter += 1
        best_flag = ""

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_macro_f1": val_f1,
        "val_accuracy": val_acc,
        "lr": current_lr
    })

    print(
        f"Epoch {epoch:02d}/20 | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Orig Val Loss {val_loss:.4f} | "
        f"Orig Val F1 {val_f1:.4f} | "
        f"Orig Val Acc {val_acc:.4f} | "
        f"LR {current_lr:.6f}"
        f"{best_flag}"
    )

    if patience_counter >= 4:
        print("\n⏹ Early stopping")
        break

# ------------------------------------------------------------
# Save training history
# ------------------------------------------------------------
history_df = pd.DataFrame(history)
history_df.to_csv(
    history_path,
    index=False
)

# ------------------------------------------------------------
# Reload BEST checkpoint and evaluate validation
# ------------------------------------------------------------
best_checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

best_val_metrics = evaluate_model(
    model,
    val_loader,
    criterion,
    device
)

pd.DataFrame([{
    "strategy": "Duplicate_Original_Control",
    "seed": SEED,
    "training_samples": len(duplicate_train_df),
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "original_val_accuracy": best_val_metrics["accuracy"],
    "original_val_macro_f1": best_val_metrics["macro_f1"],
    "original_val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
    "checkpoint": str(checkpoint_path)
}]).to_csv(
    val_path,
    index=False
)

print("\n" + "=" * 90)
print("✅ DUPLICATE-ORIGINAL CONTROL SEED 2026 COMPLETE")
print("=" * 90)
print(f"Training samples: {len(duplicate_train_df)}")
print(f"Best epoch: {best_epoch}")
print(f"Best original val loss: {best_val_loss:.6f}")
print(f"Original val Macro-F1: {best_val_metrics['macro_f1']:.4f}")
print(f"Original val Accuracy: {best_val_metrics['accuracy']:.4f}")
print(f"Checkpoint: {checkpoint_path}")
print("\n⚠️ LOCKED TEST WAS NOT USED.")
print("=" * 90)

# %% [Cell 43]
# ============================================================
# GAUSSIAN BLUR CONTROL
# SEEDS: 42, 1337, 2026
# ============================================================

SEEDS = [42, 1337, 2026]
SKIP_EXISTING = True

print("=" * 90)
print("GAUSSIAN BLUR CONTROL | ALL SEEDS")
print("=" * 90)

# ------------------------------------------------------------
# 1) Prepare Gaussian Blur dataframe
# ------------------------------------------------------------

assert len(train_original) == 595

if "gaussian_blur_train_df" in globals():
    gaussian_df = gaussian_blur_train_df.copy()

elif "gaussian_train_df" in globals():
    gaussian_df = gaussian_train_df.copy()

else:
    gaussian_dir = "/content/brinjal_train_gaussian_blur"

    assert os.path.isdir(gaussian_dir), (
        f"Gaussian Blur cache not found: {gaussian_dir}"
    )

    # Detect image path column
    possible_path_cols = [
        "filepath",
        "file_path",
        "path",
        "image_path",
        "filename"
    ]

    path_col = None

    for c in possible_path_cols:
        if c in train_original.columns:
            path_col = c
            break

    assert path_col is not None, (
        f"Could not identify image path column. "
        f"Available columns: {list(train_original.columns)}"
    )

    gaussian_files = []

    for root, _, files in os.walk(gaussian_dir):
        for fname in files:
            if fname.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                gaussian_files.append(
                    os.path.join(root, fname)
                )

    assert len(gaussian_files) == 595, (
        f"Expected 595 Gaussian Blur images, "
        f"found {len(gaussian_files)}"
    )

    # Robust lookup using the stem/ID of the original leaf
    gaussian_lookup = {}
    for p in gaussian_files:
        base = os.path.basename(p)
        # Extract ID e.g., "Phomopsis_Blight_052" from "Phomopsis_Blight_052_Gaussian_Blur.jpg"
        stem_id = base.split("_Gaussian_Blur")[0].split("_Original")[0]
        gaussian_lookup[stem_id] = p

    rows = []

    for _, row in train_original.iterrows():
        original_path = row[path_col]
        basename = os.path.basename(str(original_path))
        original_stem = basename.split("_Original")[0]

        blur_path = gaussian_lookup.get(original_stem)

        if blur_path is None:
            # Fallback search
            candidates = [
                p for p in gaussian_files
                if original_stem in os.path.basename(p)
            ]
            if len(candidates) == 1:
                blur_path = candidates[0]

        assert blur_path is not None, (
            f"Gaussian counterpart not found for {basename} (stem: {original_stem})"
        )

        new_row = row.copy()
        new_row[path_col] = blur_path

        if "transformation" in new_row.index:
            new_row["transformation"] = "Gaussian_Blur"

        rows.append(new_row)

    gaussian_df = pd.DataFrame(rows).reset_index(drop=True)

assert len(gaussian_df) == 595

print(f"Original images    : {len(train_original)}")
print(f"Gaussian Blur      : {len(gaussian_df)}")

# ------------------------------------------------------------
# 2) Build equal-size control dataset
# ------------------------------------------------------------

gaussian_control_train_df = pd.concat(
    [
        train_original.copy(),
        gaussian_df.copy()
    ],
    ignore_index=True
)

assert len(gaussian_control_train_df) == 1190

class_counts = gaussian_control_train_df["class_name"].value_counts()

assert class_counts["Healthy_Leaves"] == 560
assert class_counts["Phomopsis_Blight"] == 350
assert class_counts["Little_Leaf"] == 280

print("\nCombined training set:")
print(gaussian_control_train_df["class_name"].value_counts())

# ------------------------------------------------------------
# 3) Validation loader
# ------------------------------------------------------------

val_loader = make_loader(
    val_original,
    shuffle=False,
    seed=42
)

# ------------------------------------------------------------
# 4) Output directory
# ------------------------------------------------------------

os.makedirs(
    CONTROL_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# 5) Run all seeds
# ------------------------------------------------------------

all_results = []

for SEED in SEEDS:

    print("\n\n" + "=" * 90)
    print(f"GAUSSIAN BLUR CONTROL | SEED {SEED}")
    print("=" * 90)

    checkpoint_path = os.path.join(
        CONTROL_DIR,
        f"gaussian_blur_control_seed{SEED}.pt"
    )

    history_path = os.path.join(
        CONTROL_DIR,
        f"gaussian_blur_control_seed{SEED}_history.csv"
    )

    val_path = os.path.join(
        CONTROL_DIR,
        f"gaussian_blur_control_seed{SEED}_validation.csv"
    )

    # --------------------------------------------------------
    # Skip completed seed
    # --------------------------------------------------------

    if SKIP_EXISTING and os.path.exists(checkpoint_path):
        print(
            f"⏭️ Seed {SEED} already has a checkpoint."
        )
        print(
            f"Checkpoint: {checkpoint_path}"
        )
        print(
            "Skipping this seed."
        )

        if os.path.exists(val_path):
            existing_result = pd.read_csv(val_path)
            if len(existing_result) > 0:
                all_results.append(existing_result.iloc[0].to_dict())
        continue

    # --------------------------------------------------------
    # Set seed
    # --------------------------------------------------------

    seed_everything(SEED)

    # Deterministic shuffle
    seed_train_df = gaussian_control_train_df.sample(
        frac=1.0,
        random_state=SEED
    ).reset_index(drop=True)

    assert len(seed_train_df) == 1190

    print(f"Training samples: {len(seed_train_df)}")
    print(
        seed_train_df["class_name"].value_counts()
    )

    # --------------------------------------------------------
    # Training loader
    # --------------------------------------------------------

    train_loader = make_loader(
        seed_train_df,
        shuffle=True,
        seed=SEED
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, 21):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )
            preds = outputs.argmax(dim=1)
            correct += (
                preds == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ----------------------------------------------------
        # Original validation only
        # ----------------------------------------------------

        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_loss)
        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Best checkpoint = minimum original validation loss
        # ----------------------------------------------------

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "seed":
                        SEED,
                    "best_epoch":
                        best_epoch,
                    "best_val_loss":
                        best_val_loss,
                    "training_type":
                        "gaussian_blur_control"
                },
                checkpoint_path
            )
            best_flag = " ✅ BEST"
        else:
            patience_counter += 1
            best_flag = ""

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "lr": current_lr
            }
        )

        print(
            f"Epoch {epoch:02d}/20 | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Orig Val Loss {val_loss:.4f} | "
            f"Orig Val F1 {val_f1:.4f} | "
            f"Orig Val Acc {val_acc:.4f} | "
            f"LR {current_lr:.6f}{best_flag}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("\n⏹ Early stopping")
            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    # --------------------------------------------------------
    # Reload best checkpoint
    # --------------------------------------------------------

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    best_val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device
    )

    result = {
        "strategy": "Gaussian_Blur_Control",
        "seed": SEED,
        "training_samples": len(seed_train_df),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "original_val_accuracy": best_val_metrics["accuracy"],
        "original_val_macro_f1": best_val_metrics["macro_f1"],
        "original_val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
        "checkpoint": str(checkpoint_path)
    }

    pd.DataFrame([result]).to_csv(val_path, index=False)
    all_results.append(result)

    print("\n" + "=" * 90)
    print(f"✅ GAUSSIAN BLUR CONTROL SEED {SEED} COMPLETE")
    print("=" * 90)

# ------------------------------------------------------------
# 6) Save combined summary
# ------------------------------------------------------------

if len(all_results) > 0:
    summary_df = pd.DataFrame(all_results).sort_values("seed")
    summary_path = os.path.join(
        CONTROL_DIR,
        "gaussian_blur_control_all_seeds_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print("\n\n" + "=" * 90)
    print("✅ ALL GAUSSIAN BLUR CONTROL SEEDS COMPLETE")
    print("=" * 90)
    display(summary_df)


# %% [Cell 44]
# ============================================================
# LOCKED TEST EVALUATION
# DUPLICATE-ORIGINAL + GAUSSIAN BLUR CONTROLS
# ALL 3 SEEDS | ALL 16 CONDITIONS
# ============================================================

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ------------------------------------------------------------
# 1) Paths
# ------------------------------------------------------------

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

CONTROL_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_controls_pytorch"
)

TEST_CACHE = "/content/brinjal_test_all"

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

OUTPUT_DIR = os.path.join(
    CONTROL_DIR,
    "locked_test_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 90)
print("LOCKED TEST EVALUATION | CONTROL MODELS")
print("=" * 90)

# ------------------------------------------------------------
# 2) Verify paths
# ------------------------------------------------------------

assert os.path.exists(METADATA_PATH), (
    f"Metadata not found:\n{METADATA_PATH}"
)

assert os.path.isdir(TEST_CACHE), (
    f"Test cache not found:\n{TEST_CACHE}"
)

# ------------------------------------------------------------
# 3) Load metadata
# ------------------------------------------------------------

metadata = pd.read_csv(
    METADATA_PATH
)

print("\nMetadata shape:", metadata.shape)
print("Metadata columns:")
print(list(metadata.columns))

# ------------------------------------------------------------
# 4) Flexible column detection
# ------------------------------------------------------------

def find_column(df, candidates, required=True):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find one of {candidates} "
            f"in columns: {list(df.columns)}"
        )

    return None


filename_col = find_column(
    metadata,
    [
        "filename",
        "file_name",
        "image_name",
        "image_filename"
    ]
)

class_col = find_column(
    metadata,
    [
        "class_name",
        "class",
        "label",
        "disease_class"
    ]
)

transformation_col = find_column(
    metadata,
    [
        "transformation",
        "transform",
        "augmentation"
    ]
)

split_col = find_column(
    metadata,
    [
        "split",
        "dataset_split",
        "subset"
    ]
)

leaf_col = find_column(
    metadata,
    [
        "leaf_id",
        "original_leaf_id",
        "leaf",
        "image_id"
    ],
    required=False
)

print("\nDetected columns:")
print("Filename      :", filename_col)
print("Class         :", class_col)
print("Transformation:", transformation_col)
print("Split         :", split_col)
print("Leaf ID       :", leaf_col)

# ------------------------------------------------------------
# 5) Filter locked TEST split
# ------------------------------------------------------------

test_metadata = metadata[
    metadata[split_col].astype(str).str.lower() == "test"
].copy()

print(
    "\nLocked test metadata rows:",
    len(test_metadata)
)

assert len(test_metadata) == 2048, (
    f"Expected 2048 test images, "
    f"found {len(test_metadata)}"
)

# ------------------------------------------------------------
# 6) Scan cached test images
# ------------------------------------------------------------

cached_files = []

for root, _, files in os.walk(TEST_CACHE):

    for fname in files:

        if fname.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            cached_files.append(
                os.path.join(root, fname)
            )

print(
    "Cached test files found:",
    len(cached_files)
)

assert len(cached_files) >= 2048

# ------------------------------------------------------------
# 7) Build filename -> filepath lookup
# ------------------------------------------------------------

file_lookup = {
    os.path.basename(path): path
    for path in cached_files
}

# ------------------------------------------------------------
# 8) Match metadata rows to cached images
# ------------------------------------------------------------

matched_rows = []
missing_files = []

for _, row in test_metadata.iterrows():

    filename = str(
        row[filename_col]
    )

    path = file_lookup.get(
        os.path.basename(filename)
    )

    if path is None:

        # fallback: stem matching
        stem = os.path.splitext(
            os.path.basename(filename)
        )[0]

        candidates = [
            p
            for p in cached_files
            if stem == os.path.splitext(
                os.path.basename(p)
            )[0]
        ]

        if len(candidates) == 1:
            path = candidates[0]

    if path is None:

        missing_files.append(
            filename
        )

    else:

        item = row.to_dict()

        item["filepath"] = path

        matched_rows.append(
            item
        )

assert len(missing_files) == 0, (
    f"Missing cached files: "
    f"{len(missing_files)}\n"
    f"{missing_files[:10]}"
)

test_df = pd.DataFrame(
    matched_rows
).reset_index(drop=True)

assert len(test_df) == 2048

print(
    "\n✅ Metadata ↔ cached-file matching complete."
)

print(
    "Matched:",
    len(test_df)
)

# ------------------------------------------------------------
# 9) Verify transformations
# ------------------------------------------------------------

TRANSFORMATIONS = [
    "Original",
    "Grayscale",
    "CLAHE",
    "Gamma",
    "HSV",
    "Brightness",
    "Contrast",
    "Sharpen",
    "Gaussian_Blur",
    "Median_Blur",
    "Bilateral",
    "TopHat",
    "BlackHat",
    "Rotate",
    "Flip",
    "Unsharp_Mask"
]

print("\nTransformation counts:")

print(
    test_df[transformation_col]
    .value_counts()
)

missing_transformations = [
    t
    for t in TRANSFORMATIONS
    if t not in set(
        test_df[transformation_col]
    )
]

assert not missing_transformations, (
    "Missing transformations: "
    f"{missing_transformations}"
)

# ------------------------------------------------------------
# 10) Class mapping
# ------------------------------------------------------------

CLASS_TO_IDX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

IDX_TO_CLASS = {
    v: k
    for k, v in CLASS_TO_IDX.items()
}

test_df["label"] = (
    test_df[class_col]
    .map(CLASS_TO_IDX)
)

assert not test_df["label"].isna().any()

test_df["label"] = (
    test_df["label"]
    .astype(int)
)

# ------------------------------------------------------------
# 11) Image transform
# ------------------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# ------------------------------------------------------------
# 12) Dataset
# ------------------------------------------------------------

class LockedTestDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(
            row["label"]
        )

        return (
            image,
            label,
            idx
        )

# ------------------------------------------------------------
# 13) DataLoader
# ------------------------------------------------------------

def make_locked_loader(df):

    dataset = LockedTestDataset(
        df,
        transform=IMAGE_TRANSFORM
    )

    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

# ------------------------------------------------------------
# 14) Checkpoint list
# ------------------------------------------------------------

checkpoint_specs = []

for strategy in [
    "Duplicate_Original",
    "Gaussian_Blur"
]:

    strategy_prefix = (
        "duplicate_original_control"
        if strategy == "Duplicate_Original"
        else "gaussian_blur_control"
    )

    for seed in [
        42,
        1337,
        2026
    ]:

        checkpoint = os.path.join(
            CONTROL_DIR,
            f"{strategy_prefix}_seed{seed}.pt"
        )

        assert os.path.exists(
            checkpoint
        ), (
            f"Checkpoint not found:\n"
            f"{checkpoint}"
        )

        checkpoint_specs.append(
            {
                "strategy": strategy,
                "seed": seed,
                "checkpoint": checkpoint
            }
        )

print("\nCheckpoints:")
for spec in checkpoint_specs:
    print(
        spec["strategy"],
        "| Seed",
        spec["seed"]
    )

# ------------------------------------------------------------
# 15) Device
# ------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

# ------------------------------------------------------------
# 16) Evaluation function
# ------------------------------------------------------------

def evaluate_checkpoint_on_dataframe(
    model,
    dataframe,
    strategy,
    seed,
    transformation_name
):

    loader = make_locked_loader(
        dataframe
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_indices = []

    with torch.no_grad():

        for images, labels, indices in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            preds = outputs.argmax(
                dim=1
            ).cpu().numpy()

            labels = labels.numpy()
            indices = indices.numpy()

            all_preds.extend(
                preds.tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

            all_indices.extend(
                indices.tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_preds
    )

    return {
        "strategy": strategy,
        "seed": seed,
        "transformation": transformation_name,
        "n": len(all_labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "predictions": np.asarray(
            all_preds
        ),
        "labels": np.asarray(
            all_labels
        )
    }

# ------------------------------------------------------------
# 17) Run locked-test evaluation
# ------------------------------------------------------------

results = []
prediction_records = []

for spec in checkpoint_specs:

    strategy = spec["strategy"]
    seed = spec["seed"]
    checkpoint_path = spec["checkpoint"]

    print("\n" + "=" * 90)
    print(
        f"{strategy} | SEED {seed}"
    )
    print("=" * 90)

    # Build fresh model
    model = build_model(
        "mobilenet_v3_small"
    )

    model = model.to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    for transformation_name in TRANSFORMATIONS:

        condition_df = test_df[
            test_df[
                transformation_col
            ].astype(str)
            == transformation_name
        ].copy()

        # Every transformation should contain
        # exactly the same 128 test leaves.
        expected_n = 128

        assert len(condition_df) == expected_n, (
            f"{transformation_name}: "
            f"expected {expected_n}, "
            f"found {len(condition_df)}"
        )

        evaluation = (
            evaluate_checkpoint_on_dataframe(
                model=model,
                dataframe=condition_df,
                strategy=strategy,
                seed=seed,
                transformation_name=(
                    transformation_name
                )
            )
        )

        results.append(
            {
                "strategy":
                    strategy,
                "seed":
                    seed,
                "transformation":
                    transformation_name,
                "n":
                    evaluation["n"],
                "accuracy":
                    evaluation["accuracy"],
                "macro_f1":
                    evaluation["macro_f1"],
                "balanced_accuracy":
                    evaluation[
                        "balanced_accuracy"
                    ]
            }
        )

        # ----------------------------------------------------
        # Save individual predictions
        # ----------------------------------------------------

        condition_df = (
            condition_df.reset_index(
                drop=True
            )
        )

        for i in range(
            len(evaluation["predictions"])
        ):

            row = condition_df.iloc[i]

            record = {
                "strategy":
                    strategy,
                "seed":
                    seed,
                "transformation":
                    transformation_name,
                "filename":
                    row[filename_col],
                "filepath":
                    row["filepath"],
                "true_label":
                    int(
                        evaluation[
                            "labels"
                        ][i]
                    ),
                "pred_label":
                    int(
                        evaluation[
                            "predictions"
                        ][i]
                    ),
                "true_class":
                    IDX_TO_CLASS[
                        int(
                            evaluation[
                                "labels"
                            ][i]
                        )
                    ],
                "pred_class":
                    IDX_TO_CLASS[
                        int(
                            evaluation[
                                "predictions"
                            ][i]
                        )
                    ],
                "correct":
                    int(
                        evaluation[
                            "labels"
                        ][i]
                    )
                    ==
                    int(
                        evaluation[
                            "predictions"
                        ][i]
                    )
            }

            if leaf_col is not None:

                record["leaf_id"] = (
                    row[leaf_col]
                )

            prediction_records.append(
                record
            )

        print(
            f"{transformation_name:15s} | "
            f"Acc={evaluation['accuracy']:.4f} | "
            f"Macro-F1={evaluation['macro_f1']:.4f} | "
            f"BalAcc={evaluation['balanced_accuracy']:.4f}"
        )

# ------------------------------------------------------------
# 18) Save metrics
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

predictions_df = pd.DataFrame(
    prediction_records
)

metrics_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_metrics.csv"
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_predictions.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ------------------------------------------------------------
# 19) Aggregate across seeds
# ------------------------------------------------------------

aggregate_df = (
    results_df
    .groupby(
        ["strategy", "transformation"],
        as_index=False
    )
    .agg(
        accuracy_mean=(
            "accuracy",
            "mean"
        ),
        accuracy_sd=(
            "accuracy",
            "std"
        ),
        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),
        macro_f1_sd=(
            "macro_f1",
            "std"
        ),
        balanced_accuracy_mean=(
            "balanced_accuracy",
            "mean"
        ),
        balanced_accuracy_sd=(
            "balanced_accuracy",
            "std"
        )
    )
)

aggregate_df["accuracy_sd"] = (
    aggregate_df["accuracy_sd"]
    .fillna(0)
)

aggregate_df["macro_f1_sd"] = (
    aggregate_df["macro_f1_sd"]
    .fillna(0)
)

aggregate_df["balanced_accuracy_sd"] = (
    aggregate_df[
        "balanced_accuracy_sd"
    ]
    .fillna(0)
)

aggregate_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_aggregate.csv"
)

aggregate_df.to_csv(
    aggregate_path,
    index=False
)

# ------------------------------------------------------------
# 20) Wide Macro-F1 table
# ------------------------------------------------------------

f1_wide = aggregate_df.pivot(
    index="strategy",
    columns="transformation",
    values="macro_f1_mean"
)

f1_wide_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_macro_f1_wide.csv"
)

f1_wide.to_csv(
    f1_wide_path
)

# ------------------------------------------------------------
# 21) Target transformations summary
# ------------------------------------------------------------

target_transformations = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness",
    "Gaussian_Blur"
]

target_summary = aggregate_df[
    aggregate_df[
        "transformation"
    ].isin(
        target_transformations
    )
].copy()

target_summary_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_target_summary.csv"
)

target_summary.to_csv(
    target_summary_path,
    index=False
)

# ------------------------------------------------------------
# 22) Final report
# ------------------------------------------------------------

print("\n\n" + "=" * 90)
print("✅ LOCKED TEST CONTROL EVALUATION COMPLETE")
print("=" * 90)

print(
    "\nMetrics:",
    metrics_path
)

print(
    "Predictions:",
    predictions_path
)

print(
    "Aggregate:",
    aggregate_path
)

print(
    "Macro-F1 wide:",
    f1_wide_path
)

print(
    "Target summary:",
    target_summary_path
)

print("\nTarget-condition Macro-F1:")
display(
    target_summary.sort_values(
        [
            "strategy",
            "transformation"
        ]
    )
)

print("\n" + "=" * 90)
print(
    "⚠️ LOCKED TEST WAS USED FOR FINAL INFERENCE ONLY."
)
print(
    "⚠️ NO MODEL SELECTION OR TRAINING WAS PERFORMED."
)
print("=" * 90)

# %% [Cell 45]
import os
import shutil
import time
from pathlib import Path
import pandas as pd

DRIVE_ROOT = Path("/content/drive/MyDrive/Brinjal_Final_Preprocessed")
TEST_CACHE = Path("/content/brinjal_test_all")
METADATA_PATH = DRIVE_ROOT / "metadata.csv"

# 1. Clean up existing cache directory if any
if TEST_CACHE.exists():
    shutil.rmtree(TEST_CACHE)
TEST_CACHE.mkdir(parents=True, exist_ok=True)

# 2. Load metadata
df_meta = pd.read_csv(METADATA_PATH)
test_metadata_df = df_meta[df_meta["data_split"] == "test"].copy()

print(f"Total test images to copy: {len(test_metadata_df)}")

# 3. Copy test images
start_time = time.time()
for idx, row in test_metadata_df.iterrows():
    src = DRIVE_ROOT / row["filename"]
    dst = TEST_CACHE / Path(row["filename"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

elapsed = time.time() - start_time
local_count = sum(1 for p in TEST_CACHE.rglob("*") if p.is_file())

print(f"\n✅ Test cache copied successfully!")
print(f"Expected: {len(test_metadata_df)}")
print(f"Copied: {local_count}")
print(f"Time elapsed: {elapsed:.1f} seconds")

# %% [Cell 46]
# ============================================================
# LOCKED TEST EVALUATION
# CONTROL MODELS
# Duplicate-Original + Gaussian Blur
# Seeds: 42, 1337, 2026
# All 16 test transformations
#
# IMPORTANT:
# - Uses the real metadata column names of this dataset
# - Reads images directly from Google Drive
# - Does NOT retrain anything
# - Does NOT use test data for model selection
# - Evaluates only saved checkpoints on the locked test set
# ============================================================

import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

# ============================================================
# 1. PATHS
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

CONTROL_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_controls_pytorch"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

RAW_DIR = os.path.join(
    ROOT,
    "Raw"
)

AUGMENTED_DIR = os.path.join(
    ROOT,
    "Augmented"
)

OUTPUT_DIR = os.path.join(
    CONTROL_DIR,
    "locked_test_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 100)
print("LOCKED TEST EVALUATION | CONTROL MODELS")
print("=" * 100)

# ============================================================
# 2. BASIC CHECKS
# ============================================================

assert os.path.exists(
    METADATA_PATH
), f"Metadata not found:\n{METADATA_PATH}"

assert os.path.isdir(
    RAW_DIR
), f"Raw directory not found:\n{RAW_DIR}"

assert os.path.isdir(
    AUGMENTED_DIR
), f"Augmented directory not found:\n{AUGMENTED_DIR}"

print("✅ Metadata found")
print("✅ Raw directory found")
print("✅ Augmented directory found")

# ============================================================
# 3. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "\nMetadata shape:",
    metadata.shape
)

print(
    "Metadata columns:",
    list(metadata.columns)
)

# EXACT columns of your dataset
filename_col = "filename"
class_col = "class_label"
leaf_col = "original_image_id"
transformation_col = "preprocessing_technique"
split_col = "data_split"

required_columns = [
    filename_col,
    class_col,
    leaf_col,
    transformation_col,
    split_col
]

missing_columns = [
    c for c in required_columns
    if c not in metadata.columns
]

assert len(missing_columns) == 0, (
    f"Missing required columns: {missing_columns}"
)

# ============================================================
# 4. EXPECTED TRANSFORMATIONS
# ============================================================

TRANSFORMATIONS = [
    "Original",
    "Grayscale",
    "CLAHE",
    "Gamma",
    "HSV",
    "Brightness",
    "Contrast",
    "Sharpen",
    "Gaussian_Blur",
    "Median_Blur",
    "Bilateral",
    "TopHat",
    "BlackHat",
    "Rotate",
    "Flip",
    "Unsharp_Mask"
]

print("\nTransformation counts in full metadata:")
print(
    metadata[
        transformation_col
    ].value_counts()
)

# Make sure all expected transformations exist
metadata_transformations = set(
    metadata[
        transformation_col
    ].astype(str)
)

missing_transformations = [
    t
    for t in TRANSFORMATIONS
    if t not in metadata_transformations
]

assert not missing_transformations, (
    "Missing transformations: "
    f"{missing_transformations}"
)

# ============================================================
# 5. LOCKED TEST METADATA
# ============================================================

test_metadata = metadata[
    metadata[split_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("test")
].copy()

print(
    "\nLocked test metadata rows:",
    len(test_metadata)
)

assert len(test_metadata) == 2048, (
    f"Expected 2048 test images, "
    f"found {len(test_metadata)}"
)

# Every transformation must have 128 test images
test_transform_counts = (
    test_metadata[
        transformation_col
    ]
    .value_counts()
)

print("\nTest transformation counts:")
print(test_transform_counts)

for transformation in TRANSFORMATIONS:

    count = int(
        test_transform_counts.get(
            transformation,
            0
        )
    )

    assert count == 128, (
        f"{transformation}: "
        f"expected 128 test images, "
        f"found {count}"
    )

print(
    "\n✅ All 16 transformations have exactly 128 test images."
)

# ============================================================
# 6. CLASS MAPPING
# ============================================================

CLASS_TO_IDX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

IDX_TO_CLASS = {
    v: k
    for k, v in CLASS_TO_IDX.items()
}

print("\nClass distribution in locked test:")
print(
    test_metadata[
        class_col
    ].value_counts()
)

unexpected_classes = set(
    test_metadata[
        class_col
    ].astype(str)
) - set(CLASS_TO_IDX.keys())

assert not unexpected_classes, (
    f"Unexpected class labels: "
    f"{unexpected_classes}"
)

# ============================================================
# 7. SCAN ACTUAL DRIVE DATASET
# ============================================================

print("\nScanning Drive image files...")

all_image_files = []

for base_dir in [
    RAW_DIR,
    AUGMENTED_DIR
]:

    for root, _, files in os.walk(
        base_dir
    ):

        for fname in files:

            if fname.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            ):

                all_image_files.append(
                    os.path.join(
                        root,
                        fname
                    )
                )

print(
    "Total image files found:",
    len(all_image_files)
)

assert len(all_image_files) == 13600, (
    f"Expected 13600 images, "
    f"found {len(all_image_files)}"
)

# ============================================================
# 8. BUILD FILENAME LOOKUP
# ============================================================

file_lookup = {}

duplicate_filenames = []

for path in all_image_files:

    fname = os.path.basename(
        path
    )

    if fname in file_lookup:

        duplicate_filenames.append(
            fname
        )

    else:

        file_lookup[fname] = path

assert len(duplicate_filenames) == 0, (
    "Duplicate filenames detected. "
    f"Examples: {duplicate_filenames[:10]}"
)

print(
    "Unique filenames:",
    len(file_lookup)
)

assert len(file_lookup) == 13600

# ============================================================
# 9. MATCH TEST METADATA TO REAL FILES
# ============================================================

matched_rows = []
missing_files = []

for _, row in test_metadata.iterrows():

    filename = os.path.basename(
        str(row[filename_col])
    )

    filepath = file_lookup.get(
        filename
    )

    if filepath is None:

        missing_files.append(
            filename
        )

    else:

        item = row.to_dict()

        item["filepath"] = filepath

        matched_rows.append(
            item
        )

print(
    "\nMissing locked-test files:",
    len(missing_files)
)

assert len(missing_files) == 0, (
    "Some locked-test images could not be matched.\n"
    f"Examples:\n{missing_files[:20]}"
)

test_df = pd.DataFrame(
    matched_rows
).reset_index(
    drop=True
)

assert len(test_df) == 2048

# ============================================================
# 10. ADD NUMERIC LABELS
# ============================================================

test_df["label"] = (
    test_df[class_col]
    .map(CLASS_TO_IDX)
)

assert not test_df["label"].isna().any()

test_df["label"] = (
    test_df["label"].astype(int)
)

# ============================================================
# 11. IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)

# ============================================================
# 12. LOCKED TEST DATASET
# ============================================================

class LockedTestDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[
            idx
        ]

        image = Image.open(
            row["filepath"]
        ).convert(
            "RGB"
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        label = int(
            row["label"]
        )

        return (
            image,
            label,
            idx
        )

# ============================================================
# 13. DATA LOADER
# ============================================================

def make_locked_loader(
    dataframe
):

    dataset = LockedTestDataset(
        dataframe,
        transform=IMAGE_TRANSFORM
    )

    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

# ============================================================
# 14. CHECKPOINTS
# ============================================================

checkpoint_specs = []

for seed in [
    42,
    1337,
    2026
]:

    duplicate_checkpoint = os.path.join(
        CONTROL_DIR,
        f"duplicate_original_control_seed{seed}.pt"
    )

    gaussian_checkpoint = os.path.join(
        CONTROL_DIR,
        f"gaussian_blur_control_seed{seed}.pt"
    )

    assert os.path.exists(
        duplicate_checkpoint
    ), (
        f"Duplicate-Original checkpoint missing:\n"
        f"{duplicate_checkpoint}"
    )

    assert os.path.exists(
        gaussian_checkpoint
    ), (
        f"Gaussian Blur checkpoint missing:\n"
        f"{gaussian_checkpoint}"
    )

    checkpoint_specs.append(
        {
            "strategy":
                "Duplicate_Original",
            "seed":
                seed,
            "checkpoint":
                duplicate_checkpoint
        }
    )

    checkpoint_specs.append(
        {
            "strategy":
                "Gaussian_Blur",
            "seed":
                seed,
            "checkpoint":
                gaussian_checkpoint
        }
    )

print(
    "\n✅ All six control checkpoints found."
)

for spec in checkpoint_specs:

    print(
        f"{spec['strategy']:20s} | "
        f"Seed {spec['seed']}"
    )

# ============================================================
# 15. DEVICE
# ============================================================

assert "build_model" in globals(), (
    "build_model() is not available in the "
    "current notebook runtime."
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nDevice:",
    device
)

# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================

def evaluate_condition(
    model,
    condition_df
):

    loader = make_locked_loader(
        condition_df
    )

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels, _ in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            predictions = (
                outputs
                .argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            labels = (
                labels
                .numpy()
            )

            all_predictions.extend(
                predictions.tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    all_predictions = np.asarray(
        all_predictions,
        dtype=int
    )

    all_labels = np.asarray(
        all_labels,
        dtype=int
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions
        )
    )

    return {
        "predictions":
            all_predictions,
        "labels":
            all_labels,
        "accuracy":
            accuracy,
        "macro_f1":
            macro_f1,
        "balanced_accuracy":
            balanced_accuracy
    }

# ============================================================
# 17. RUN ALL CONTROL CHECKPOINTS
# ============================================================

results = []
prediction_records = []

for spec in checkpoint_specs:

    strategy = spec["strategy"]
    seed = spec["seed"]
    checkpoint_path = spec["checkpoint"]

    print("\n")
    print("=" * 100)
    print(f"{strategy} | SEED {seed}")
    print("=" * 100)

    # --------------------------------------------------------
    # Fresh MobileNetV3-Small model
    # build_model() takes NO arguments in this notebook
    # --------------------------------------------------------

    model = build_model()
    model = model.to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Load saved best checkpoint
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Evaluate all 16 transformations
    # --------------------------------------------------------

    for transformation in TRANSFORMATIONS:

        condition_df = test_df[
            test_df[transformation_col]
            .astype(str)
            .eq(transformation)
        ].copy().reset_index(drop=True)

        assert len(condition_df) == 128, (
            f"{strategy} seed {seed} "
            f"{transformation}: "
            f"expected 128 images, "
            f"found {len(condition_df)}"
        )

        evaluation = evaluate_condition(
            model,
            condition_df
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        results.append({
            "strategy": strategy,
            "seed": seed,
            "transformation": transformation,
            "n": 128,
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "balanced_accuracy":
                evaluation["balanced_accuracy"]
        })

        # ----------------------------------------------------
        # Individual predictions
        # ----------------------------------------------------

        predictions = evaluation["predictions"]
        labels = evaluation["labels"]

        for i in range(len(condition_df)):

            row = condition_df.iloc[i]

            true_label = int(labels[i])
            pred_label = int(predictions[i])

            prediction_records.append({
                "strategy": strategy,
                "seed": seed,
                "transformation": transformation,
                "filename": row[filename_col],
                "filepath": row["filepath"],
                "original_image_id": row[leaf_col],
                "true_label": true_label,
                "pred_label": pred_label,
                "true_class":
                    IDX_TO_CLASS[true_label],
                "pred_class":
                    IDX_TO_CLASS[pred_label],
                "correct":
                    int(true_label == pred_label)
            })

        print(
            f"{transformation:15s} | "
            f"Acc = {evaluation['accuracy']:.4f} | "
            f"Macro-F1 = {evaluation['macro_f1']:.4f} | "
            f"BalAcc = {evaluation['balanced_accuracy']:.4f}"
        )

# ============================================================
# SAVE RAW METRICS
# ============================================================

results_df = pd.DataFrame(results)
predictions_df = pd.DataFrame(prediction_records)

metrics_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_metrics.csv"
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_predictions.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# AGGREGATE ACROSS 3 SEEDS
# ============================================================

aggregate_df = (
    results_df
    .groupby(
        ["strategy", "transformation"],
        as_index=False
    )
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        balanced_accuracy_mean=(
            "balanced_accuracy",
            "mean"
        ),
        balanced_accuracy_sd=(
            "balanced_accuracy",
            "std"
        )
    )
)

aggregate_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_aggregate.csv"
)

aggregate_df.to_csv(
    aggregate_path,
    index=False
)

# ============================================================
# TARGET-FOCUSED SUMMARY
# ============================================================

TARGET_SUMMARY_TRANSFORMATIONS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness",
    "Gaussian_Blur"
]

target_summary = aggregate_df[
    aggregate_df["transformation"].isin(
        TARGET_SUMMARY_TRANSFORMATIONS
    )
].copy()

target_summary = (
    target_summary
    .sort_values(
        ["strategy", "transformation"]
    )
    .reset_index(drop=True)
)

target_summary_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_target_summary.csv"
)

target_summary.to_csv(
    target_summary_path,
    index=False
)

# ============================================================
# WIDE MACRO-F1 TABLE
# ============================================================

macro_f1_wide = aggregate_df.pivot(
    index="strategy",
    columns="transformation",
    values="macro_f1_mean"
)

macro_f1_wide_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_macro_f1_wide.csv"
)

macro_f1_wide.to_csv(
    macro_f1_wide_path
)

# ============================================================
# WIDE ACCURACY TABLE
# ============================================================

accuracy_wide = aggregate_df.pivot(
    index="strategy",
    columns="transformation",
    values="accuracy_mean"
)

accuracy_wide_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_accuracy_wide.csv"
)

accuracy_wide.to_csv(
    accuracy_wide_path
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n\n")
print("=" * 100)
print("✅ LOCKED TEST CONTROL EVALUATION COMPLETE")
print("=" * 100)

print("\nTarget-condition summary:")
display(target_summary)

print("\nMacro-F1 across all 16 transformations:")
display(macro_f1_wide)

print("\nSaved files:")
print(metrics_path)
print(predictions_path)
print(aggregate_path)
print(target_summary_path)
print(macro_f1_wide_path)
print(accuracy_wide_path)

print("\n" + "=" * 100)
print("⚠️ TEST SET USED FOR INFERENCE ONLY.")
print("⚠️ NO TRAINING OR MODEL SELECTION USED THE TEST SET.")
print("=" * 100)
# ============================================================
# 18. SAVE RAW METRICS
# ============================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.DataFrame(
    prediction_records
)

metrics_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_metrics.csv"
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_predictions.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# 19. AGGREGATE ACROSS 3 SEEDS
# ============================================================

aggregate_df = (
    results_df
    .groupby(
        [
            "strategy",
            "transformation"
        ],
        as_index=False
    )
    .agg(
        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_sd=(
            "accuracy",
            "std"
        ),

        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),

        macro_f1_sd=(
            "macro_f1",
            "std"
        ),

        balanced_accuracy_mean=(
            "balanced_accuracy",
            "mean"
        ),

        balanced_accuracy_sd=(
            "balanced_accuracy",
            "std"
        )
    )
)

# ============================================================
# 20. SAVE AGGREGATE
# ============================================================

aggregate_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_aggregate.csv"
)

aggregate_df.to_csv(
    aggregate_path,
    index=False
)

# ============================================================
# 21. TARGET-FOCUSED SUMMARY
# ============================================================

TARGET_SUMMARY_TRANSFORMATIONS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness",
    "Gaussian_Blur"
]

target_summary = aggregate_df[
    aggregate_df[
        "transformation"
    ].isin(
        TARGET_SUMMARY_TRANSFORMATIONS
    )
].copy()

target_summary = (
    target_summary
    .sort_values(
        [
            "strategy",
            "transformation"
        ]
    )
    .reset_index(
        drop=True
    )
)

target_summary_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_target_summary.csv"
)

target_summary.to_csv(
    target_summary_path,
    index=False
)

# ============================================================
# 22. WIDE MACRO-F1 TABLE
# ============================================================

macro_f1_wide = (
    aggregate_df
    .pivot(
        index="strategy",
        columns="transformation",
        values="macro_f1_mean"
    )
)

macro_f1_wide_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_macro_f1_wide.csv"
)

macro_f1_wide.to_csv(
    macro_f1_wide_path
)

# ============================================================
# 23. WIDE ACCURACY TABLE
# ============================================================

accuracy_wide = (
    aggregate_df
    .pivot(
        index="strategy",
        columns="transformation",
        values="accuracy_mean"
    )
)

accuracy_wide_path = os.path.join(
    OUTPUT_DIR,
    "control_locked_test_accuracy_wide.csv"
)

accuracy_wide.to_csv(
    accuracy_wide_path
)

# ============================================================
# 24. FINAL REPORT
# ============================================================

print("\n\n")
print("=" * 100)
print("✅ LOCKED TEST CONTROL EVALUATION COMPLETE")
print("=" * 100)

print(
    "\nTarget-condition summary:"
)

display(
    target_summary
)

print(
    "\n\nMacro-F1 across all 16 transformations:"
)

display(
    macro_f1_wide
)

print(
    "\nSaved files:"
)

print(
    f"1. {metrics_path}"
)

print(
    f"2. {predictions_path}"
)

print(
    f"3. {aggregate_path}"
)

print(
    f"4. {target_summary_path}"
)

print(
    f"5. {macro_f1_wide_path}"
)

print(
    f"6. {accuracy_wide_path}"
)

print("\n")
print("=" * 100)
print(
    "⚠️ TEST SET WAS USED FOR INFERENCE ONLY."
)
print(
    "⚠️ NO TRAINING OR MODEL SELECTION USED THE TEST SET."
)
print("=" * 100)

# %% [Cell 47]
# ============================================================
# CONTROL STATISTICAL ANALYSIS
# Exact McNemar + Benjamini-Hochberg FDR
#
# Baseline/Targeted predictions:
#   y_true / y_pred / correct
#
# Control predictions:
#   true_label / pred_label / correct
#
# Comparisons:
#   1) Baseline vs Duplicate-Original
#   2) Baseline vs Gaussian-Blur
#   3) Targeted vs Duplicate-Original
#   4) Targeted vs Gaussian-Blur
#
# Targets:
#   Grayscale
#   Rotate
#   Brightness
#
# Seeds:
#   42, 1337, 2026
#
# McNemar tests paired prediction correctness,
# NOT Macro-F1 significance.
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import binomtest

# ============================================================
# 1. PATHS
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

BASELINE_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "final_locked_test"
)

CONTROL_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_controls_pytorch",
    "locked_test_evaluation"
)

BASELINE_PRED_PATH = os.path.join(
    BASELINE_DIR,
    "final_test_predictions.csv"
)

CONTROL_PRED_PATH = os.path.join(
    CONTROL_DIR,
    "control_locked_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    CONTROL_DIR,
    "statistical_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 100)
print("CONTROL STATISTICAL ANALYSIS")
print("Exact McNemar + Benjamini-Hochberg FDR")
print("=" * 100)

# ============================================================
# 2. VERIFY FILES
# ============================================================

assert os.path.exists(BASELINE_PRED_PATH), (
    f"Baseline prediction file not found:\n"
    f"{BASELINE_PRED_PATH}"
)

assert os.path.exists(CONTROL_PRED_PATH), (
    f"Control prediction file not found:\n"
    f"{CONTROL_PRED_PATH}"
)

print("✅ Baseline prediction file found")
print("✅ Control prediction file found")

# ============================================================
# 3. LOAD
# ============================================================

baseline_df = pd.read_csv(
    BASELINE_PRED_PATH
)

control_df = pd.read_csv(
    CONTROL_PRED_PATH
)

print("\nBaseline shape:", baseline_df.shape)
print("Control shape :", control_df.shape)

print("\nBaseline columns:")
print(list(baseline_df.columns))

print("\nControl columns:")
print(list(control_df.columns))

# ============================================================
# 4. ROBUST COLUMN STANDARDIZATION
# ============================================================

def standardize_prediction_columns(
    df,
    dataset_name
):

    df = df.copy()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    if "seed" not in df.columns:

        for candidate in [
            "random_seed",
            "model_seed"
        ]:

            if candidate in df.columns:

                df["seed"] = df[
                    candidate
                ]

                break

    # --------------------------------------------------------
    # Transformation
    # --------------------------------------------------------

    if "transformation" not in df.columns:

        for candidate in [
            "test_transformation",
            "preprocessing_technique",
            "transform"
        ]:

            if candidate in df.columns:

                df["transformation"] = df[
                    candidate
                ]

                break

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    if "filename" not in df.columns:

        for candidate in [
            "file_name",
            "image_name",
            "image_filename"
        ]:

            if candidate in df.columns:

                df["filename"] = df[
                    candidate
                ]

                break

    # --------------------------------------------------------
    # Prediction
    # Supports:
    #   y_pred
    #   pred_label
    #   prediction
    #   pred
    # --------------------------------------------------------

    if "pred_label" not in df.columns:

        if "y_pred" in df.columns:

            df["pred_label"] = df[
                "y_pred"
            ]

        else:

            for candidate in [
                "prediction",
                "pred",
                "predicted_label"
            ]:

                if candidate in df.columns:

                    df["pred_label"] = df[
                        candidate
                    ]

                    break

    # --------------------------------------------------------
    # True label
    # Supports:
    #   y_true
    #   true_label
    #   label
    #   target
    # --------------------------------------------------------

    if "true_label" not in df.columns:

        if "y_true" in df.columns:

            df["true_label"] = df[
                "y_true"
            ]

        else:

            for candidate in [
                "label",
                "target",
                "ground_truth"
            ]:

                if candidate in df.columns:

                    df["true_label"] = df[
                        candidate
                    ]

                    break

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    if "correct" not in df.columns:

        assert (
            "true_label" in df.columns
            and
            "pred_label" in df.columns
        ), (
            f"{dataset_name}: cannot create "
            f"'correct' because true/pred columns "
            f"are missing."
        )

        df["correct"] = (
            pd.to_numeric(
                df["true_label"]
            )
            ==
            pd.to_numeric(
                df["pred_label"]
            )
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "strategy",
        "seed",
        "transformation",
        "filename",
        "pred_label",
        "true_label",
        "correct"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    assert not missing, (
        f"{dataset_name}: missing columns "
        f"{missing}\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # Normalize values
    # --------------------------------------------------------

    df["seed"] = pd.to_numeric(
        df["seed"]
    ).astype(int)

    df["strategy"] = (
        df["strategy"]
        .astype(str)
        .str.strip()
    )

    df["transformation"] = (
        df["transformation"]
        .astype(str)
        .str.strip()
    )

    df["filename"] = (
        df["filename"]
        .astype(str)
        .map(os.path.basename)
    )

    df["pred_label"] = pd.to_numeric(
        df["pred_label"]
    ).astype(int)

    df["true_label"] = pd.to_numeric(
        df["true_label"]
    ).astype(int)

    # force boolean
    df["correct"] = (
        df["correct"]
        .astype(bool)
    )

    return df


baseline_df = standardize_prediction_columns(
    baseline_df,
    "BASELINE"
)

control_df = standardize_prediction_columns(
    control_df,
    "CONTROL"
)

print("\n✅ Prediction columns standardized.")

print("\nBaseline standardized columns:")
print(list(baseline_df.columns))

print("\nControl standardized columns:")
print(list(control_df.columns))

# ============================================================
# 5. CHECK STRATEGIES
# ============================================================

print("\nBaseline strategies:")
print(
    baseline_df[
        "strategy"
    ].value_counts()
)

print("\nControl strategies:")
print(
    control_df[
        "strategy"
    ].value_counts()
)

# ============================================================
# 6. EXPECTED EXPERIMENT
# ============================================================

SEEDS = [
    42,
    1337,
    2026
]

TARGETS = [
    "Grayscale",
    "Rotate",
    "Brightness"
]

CONTROL_DUPLICATE = (
    "Duplicate_Original"
)

CONTROL_GAUSSIAN = (
    "Gaussian_Blur"
)

TARGETED_STRATEGIES = {
    "Grayscale":
        "Grayscale",

    "Rotate":
        "Rotate",

    "Brightness":
        "Brightness"
}

assert (
    CONTROL_DUPLICATE
    in set(control_df["strategy"])
), "Duplicate_Original strategy missing."

assert (
    CONTROL_GAUSSIAN
    in set(control_df["strategy"])
), "Gaussian_Blur strategy missing."

# ============================================================
# 7. CHECK ALL EXPECTED PAIRS
# ============================================================

print("\nChecking prediction counts...")

for seed in SEEDS:

    for target in TARGETS:

        # Baseline
        baseline_subset = baseline_df[
            (baseline_df["strategy"] == "Baseline")
            &
            (baseline_df["seed"] == seed)
            &
            (
                baseline_df["transformation"]
                == target
            )
        ]

        assert len(baseline_subset) == 128, (
            f"Baseline | seed {seed} | "
            f"{target}: expected 128, "
            f"found {len(baseline_subset)}"
        )

        # Targeted
        targeted_strategy = (
            TARGETED_STRATEGIES[target]
        )

        targeted_subset = baseline_df[
            (baseline_df["strategy"]
             == targeted_strategy)
            &
            (baseline_df["seed"]
             == seed)
            &
            (
                baseline_df["transformation"]
                == target
            )
        ]

        assert len(targeted_subset) == 128, (
            f"Targeted {target} | seed {seed}: "
            f"expected 128, "
            f"found {len(targeted_subset)}"
        )

        # Duplicate
        duplicate_subset = control_df[
            (control_df["strategy"]
             == CONTROL_DUPLICATE)
            &
            (control_df["seed"]
             == seed)
            &
            (
                control_df["transformation"]
                == target
            )
        ]

        assert len(duplicate_subset) == 128, (
            f"Duplicate | seed {seed} | "
            f"{target}: expected 128, "
            f"found {len(duplicate_subset)}"
        )

        # Gaussian
        gaussian_subset = control_df[
            (control_df["strategy"]
             == CONTROL_GAUSSIAN)
            &
            (control_df["seed"]
             == seed)
            &
            (
                control_df["transformation"]
                == target
            )
        ]

        assert len(gaussian_subset) == 128, (
            f"Gaussian | seed {seed} | "
            f"{target}: expected 128, "
            f"found {len(gaussian_subset)}"
        )

print(
    "✅ All required prediction groups contain 128 leaves."
)

# ============================================================
# 8. EXACT MCNEMAR
# ============================================================

def exact_mcnemar(
    model_a_correct,
    model_b_correct
):

    a = np.asarray(
        model_a_correct,
        dtype=bool
    )

    b = np.asarray(
        model_b_correct,
        dtype=bool
    )

    assert len(a) == len(b)

    both_correct = (
        a & b
    ).sum()

    both_wrong = (
        (~a) & (~b)
    ).sum()

    a_correct_b_wrong = (
        a & (~b)
    ).sum()

    a_wrong_b_correct = (
        (~a) & b
    ).sum()

    discordant = (
        a_correct_b_wrong
        +
        a_wrong_b_correct
    )

    if discordant == 0:

        p_value = 1.0

    else:

        p_value = binomtest(
            k=int(
                min(
                    a_correct_b_wrong,
                    a_wrong_b_correct
                )
            ),
            n=int(
                discordant
            ),
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return {
        "n":
            int(len(a)),

        "both_correct":
            int(both_correct),

        "both_wrong":
            int(both_wrong),

        "a_correct_b_wrong":
            int(a_correct_b_wrong),

        "a_wrong_b_correct":
            int(a_wrong_b_correct),

        "discordant":
            int(discordant),

        "p_value":
            float(p_value)
    }

# ============================================================
# 9. COMPARISON HELPER
# ============================================================

def extract_subset(
    df,
    strategy,
    seed,
    transformation
):

    return df[
        (df["strategy"] == strategy)
        &
        (df["seed"] == seed)
        &
        (
            df["transformation"]
            == transformation
        )
    ].copy()


def paired_mcnemar(
    df_a,
    df_b
):

    a = df_a[
        [
            "filename",
            "correct"
        ]
    ].rename(
        columns={
            "correct":
                "a_correct"
        }
    )

    b = df_b[
        [
            "filename",
            "correct"
        ]
    ].rename(
        columns={
            "correct":
                "b_correct"
        }
    )

    paired = pd.merge(
        a,
        b,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    assert len(paired) == 128, (
        "Paired leaves are not exactly 128. "
        f"Found {len(paired)}."
    )

    return exact_mcnemar(
        paired["a_correct"].values,
        paired["b_correct"].values
    )

# ============================================================
# 10. RUN ALL 36 TESTS
# ============================================================

comparison_results = []

for target in TARGETS:

    targeted_strategy = (
        TARGETED_STRATEGIES[target]
    )

    for seed in SEEDS:

        baseline_subset = extract_subset(
            baseline_df,
            "Baseline",
            seed,
            target
        )

        targeted_subset = extract_subset(
            baseline_df,
            targeted_strategy,
            seed,
            target
        )

        duplicate_subset = extract_subset(
            control_df,
            CONTROL_DUPLICATE,
            seed,
            target
        )

        gaussian_subset = extract_subset(
            control_df,
            CONTROL_GAUSSIAN,
            seed,
            target
        )

        pair_specs = [
            (
                "Baseline_vs_Duplicate",
                "Baseline",
                CONTROL_DUPLICATE,
                baseline_subset,
                duplicate_subset
            ),

            (
                "Baseline_vs_Gaussian",
                "Baseline",
                CONTROL_GAUSSIAN,
                baseline_subset,
                gaussian_subset
            ),

            (
                "Targeted_vs_Duplicate",
                targeted_strategy,
                CONTROL_DUPLICATE,
                targeted_subset,
                duplicate_subset
            ),

            (
                "Targeted_vs_Gaussian",
                targeted_strategy,
                CONTROL_GAUSSIAN,
                targeted_subset,
                gaussian_subset
            )
        ]

        for (
            comparison,
            model_a,
            model_b,
            df_a,
            df_b
        ) in pair_specs:

            result = paired_mcnemar(
                df_a,
                df_b
            )

            comparison_results.append(
                {
                    "comparison":
                        comparison,

                    "target_transformation":
                        target,

                    "seed":
                        seed,

                    "model_a":
                        model_a,

                    "model_b":
                        model_b,

                    "n":
                        result["n"],

                    "both_correct":
                        result["both_correct"],

                    "both_wrong":
                        result["both_wrong"],

                    "a_correct_b_wrong":
                        result[
                            "a_correct_b_wrong"
                        ],

                    "a_wrong_b_correct":
                        result[
                            "a_wrong_b_correct"
                        ],

                    "discordant":
                        result[
                            "discordant"
                        ],

                    "p_value":
                        result[
                            "p_value"
                        ]
                }
            )

mcnemar_df = pd.DataFrame(
    comparison_results
)

assert len(
    mcnemar_df
) == 36

# ============================================================
# 11. BENJAMINI-HOCHBERG
# ============================================================

def benjamini_hochberg(
    p_values
):

    p = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p)

    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    q_ranked = (
        ranked
        * n
        / np.arange(
            1,
            n + 1
        )
    )

    q_ranked = np.minimum.accumulate(
        q_ranked[::-1]
    )[::-1]

    q_ranked = np.clip(
        q_ranked,
        0,
        1
    )

    q = np.empty_like(
        q_ranked
    )

    q[order] = q_ranked

    return q


mcnemar_df["q_value"] = (
    benjamini_hochberg(
        mcnemar_df["p_value"].values
    )
)

mcnemar_df["significant_fdr_0.05"] = (
    mcnemar_df["q_value"] < 0.05
)

# ============================================================
# 12. DIRECTION OF CHANGE
# ============================================================

mcnemar_df[
    "net_correctness_change"
] = (
    mcnemar_df[
        "a_wrong_b_correct"
    ]
    -
    mcnemar_df[
        "a_correct_b_wrong"
    ]
)

mcnemar_df[
    "preferred_model"
] = np.where(
    mcnemar_df[
        "net_correctness_change"
    ] > 0,
    mcnemar_df["model_b"],
    np.where(
        mcnemar_df[
            "net_correctness_change"
        ] < 0,
        mcnemar_df["model_a"],
        "Tie"
    )
)

# ============================================================
# 13. SAVE FULL RESULTS
# ============================================================

full_results_path = os.path.join(
    OUTPUT_DIR,
    "control_mcnemar_all_comparisons.csv"
)

mcnemar_df.to_csv(
    full_results_path,
    index=False
)

# ============================================================
# 14. SUMMARY
# ============================================================

summary_df = (
    mcnemar_df
    .groupby(
        [
            "comparison",
            "target_transformation"
        ],
        as_index=False
    )
    .agg(
        significant_tests=(
            "significant_fdr_0.05",
            "sum"
        ),

        total_tests=(
            "seed",
            "count"
        ),

        min_q=(
            "q_value",
            "min"
        ),

        median_q=(
            "q_value",
            "median"
        ),

        mean_net_correctness_change=(
            "net_correctness_change",
            "mean"
        ),

        mean_a_correct_b_wrong=(
            "a_correct_b_wrong",
            "mean"
        ),

        mean_a_wrong_b_correct=(
            "a_wrong_b_correct",
            "mean"
        )
    )
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "control_mcnemar_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

# ============================================================
# 15. PRINT MAIN RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("EXACT McNEMAR RESULTS")
print("=" * 100)

display(
    mcnemar_df[
        [
            "comparison",
            "target_transformation",
            "seed",
            "a_correct_b_wrong",
            "a_wrong_b_correct",
            "p_value",
            "q_value",
            "significant_fdr_0.05",
            "preferred_model"
        ]
    ]
    .sort_values(
        [
            "comparison",
            "target_transformation",
            "seed"
        ]
    )
    .reset_index(drop=True)
)

print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

display(
    summary_df.sort_values(
        [
            "comparison",
            "target_transformation"
        ]
    ).reset_index(
        drop=True
    )
)

# ============================================================
# 16. TARGET-FOCUSED TABLE
# ============================================================

target_table = mcnemar_df[
    mcnemar_df[
        "target_transformation"
    ].isin(TARGETS)
].copy()

target_table_path = os.path.join(
    OUTPUT_DIR,
    "control_mcnemar_target_table.csv"
)

target_table.to_csv(
    target_table_path,
    index=False
)

# ============================================================
# 17. FINAL
# ============================================================

print("\n")
print("=" * 100)
print("✅ CONTROL STATISTICAL ANALYSIS COMPLETE")
print("=" * 100)

print(
    "\nTotal McNemar tests:",
    len(mcnemar_df)
)

print(
    "FDR-significant tests:",
    int(
        mcnemar_df[
            "significant_fdr_0.05"
        ].sum()
    )
)

print(
    "\nFull results:",
    full_results_path
)

print(
    "Summary:",
    summary_path
)

print(
    "Target table:",
    target_table_path
)

print("\n⚠️ McNemar tests prediction correctness.")
print("⚠️ It does not directly test Macro-F1.")
print("=" * 100)

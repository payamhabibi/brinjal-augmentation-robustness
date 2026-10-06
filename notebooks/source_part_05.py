
# %% [Cell 67]
# ============================================================
# MODEL-vs-MODEL ROBUSTNESS STATISTICAL COMPARISON
#
# 4 CNN architectures
# 3 random seeds
# 16 test conditions
#
# Exact McNemar test + Benjamini-Hochberg FDR correction
#
# TOTAL TESTS:
#   6 model pairs × 3 seeds × 16 transformations = 288
#
# NO TRAINING
# NO CHECKPOINT SELECTION
# LOCKED TEST PREDICTIONS ONLY
# ============================================================

import os
import itertools
import numpy as np
import pandas as pd

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. PATHS
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

PREDICTION_PATH = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "final_locked_test",
    "final_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "final_locked_test",
    "model_vs_model_statistical_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

assert os.path.exists(
    PREDICTION_PATH
), (
    "Prediction file not found:\n"
    f"{PREDICTION_PATH}"
)


# ============================================================
# 2. LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTION_PATH
)

print("=" * 110)
print("MODEL-vs-MODEL ROBUSTNESS STATISTICAL ANALYSIS")
print("=" * 110)

print("\nPrediction file:")
print(PREDICTION_PATH)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. AUTO-DETECT IMPORTANT COLUMN NAMES
# ============================================================

def detect_column(
    dataframe,
    candidates,
    column_description
):
    for col in candidates:
        if col in dataframe.columns:
            return col

    raise ValueError(
        f"\nCould not detect {column_description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {dataframe.columns.tolist()}"
    )


MODEL_COL = detect_column(
    df,
    [
        "model",
        "model_name",
        "architecture",
        "architecture_name"
    ],
    "model"
)

SEED_COL = detect_column(
    df,
    [
        "seed",
        "random_seed"
    ],
    "seed"
)

TRANSFORMATION_COL = detect_column(
    df,
    [
        "transformation",
        "transform",
        "condition"
    ],
    "transformation"
)

IMAGE_ID_COL = detect_column(
    df,
    [
        "original_image_id",
        "image_id",
        "sample_id",
        "leaf_id"
    ],
    "image/leaf ID"
)

Y_TRUE_COL = detect_column(
    df,
    [
        "y_true",
        "true_label",
        "label"
    ],
    "ground-truth label"
)

Y_PRED_COL = detect_column(
    df,
    [
        "y_pred",
        "prediction",
        "predicted_label"
    ],
    "prediction"
)


print("\nDetected columns:")
print("MODEL        :", MODEL_COL)
print("SEED         :", SEED_COL)
print("TRANSFORM    :", TRANSFORMATION_COL)
print("IMAGE ID     :", IMAGE_ID_COL)
print("Y TRUE       :", Y_TRUE_COL)
print("Y PRED       :", Y_PRED_COL)


# ============================================================
# 4. EXPECTED MODELS / SEEDS / TRANSFORMATIONS
# ============================================================

EXPECTED_MODELS = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18"
]

SEEDS = [
    42,
    1337,
    2026
]

EXPECTED_TRANSFORMATIONS = [
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


# ============================================================
# 5. DISPLAY AVAILABLE VALUES
# ============================================================

print("\n")
print("=" * 110)
print("AVAILABLE VALUES")
print("=" * 110)

available_models = sorted(
    df[MODEL_COL].astype(str).unique()
)

available_seeds = sorted(
    df[SEED_COL].unique()
)

available_transformations = sorted(
    df[TRANSFORMATION_COL].astype(str).unique()
)

print("\nModels:")
print(available_models)

print("\nSeeds:")
print(available_seeds)

print("\nTransformations:")
print(available_transformations)


# ============================================================
# 6. VERIFY EXPECTED VALUES
# ============================================================

missing_models = [
    x for x in EXPECTED_MODELS
    if x not in available_models
]

missing_seeds = [
    x for x in SEEDS
    if x not in available_seeds
]

missing_transformations = [
    x for x in EXPECTED_TRANSFORMATIONS
    if x not in available_transformations
]

if missing_models:
    raise ValueError(
        f"Missing expected models: {missing_models}"
    )

if missing_seeds:
    raise ValueError(
        f"Missing expected seeds: {missing_seeds}"
    )

if missing_transformations:
    raise ValueError(
        f"Missing expected transformations: "
        f"{missing_transformations}"
    )


# ============================================================
# 7. KEEP ONLY THE FOUR CNN MODELS
# ============================================================

df = df[
    df[MODEL_COL].isin(
        EXPECTED_MODELS
    )
].copy()


# ============================================================
# 8. GROUP SIZE VERIFICATION
#
# Expected:
# 128 leaves for every
# model × seed × transformation
# ============================================================

print("\n")
print("=" * 110)
print("GROUP SIZE VERIFICATION")
print("=" * 110)

group_size_records = []

for model in EXPECTED_MODELS:

    for seed in SEEDS:

        for transformation in EXPECTED_TRANSFORMATIONS:

            group = df[
                (df[MODEL_COL] == model)
                &
                (df[SEED_COL] == seed)
                &
                (
                    df[TRANSFORMATION_COL]
                    == transformation
                )
            ]

            n = len(group)

            group_size_records.append({

                "model": model,
                "seed": seed,
                "transformation": transformation,
                "n": n
            })

            print(
                f"{model:20s} | "
                f"seed={seed:4d} | "
                f"{transformation:15s} | "
                f"n={n}"
            )

            assert n == 128, (
                "\nUnexpected group size!\n"
                f"Model: {model}\n"
                f"Seed: {seed}\n"
                f"Transformation: {transformation}\n"
                f"Expected: 128\n"
                f"Observed: {n}"
            )


group_size_df = pd.DataFrame(
    group_size_records
)


# ============================================================
# 9. MODEL PAIRS
# ============================================================

MODEL_PAIRS = list(
    itertools.combinations(
        EXPECTED_MODELS,
        2
    )
)

print("\n")
print("=" * 110)
print("MODEL PAIRS")
print("=" * 110)

for i, pair in enumerate(
    MODEL_PAIRS,
    start=1
):
    print(
        f"{i}. {pair[0]} vs {pair[1]}"
    )

print(
    f"\nTotal model pairs: {len(MODEL_PAIRS)}"
)

print(
    "Total planned tests:",
    len(MODEL_PAIRS)
    * len(SEEDS)
    * len(EXPECTED_TRANSFORMATIONS)
)


# ============================================================
# 10. EXACT McNEMAR TEST
# ============================================================

results = []


for model_a, model_b in MODEL_PAIRS:

    for seed in SEEDS:

        for transformation in EXPECTED_TRANSFORMATIONS:

            # ------------------------------------------------
            # Extract model A
            # ------------------------------------------------

            a_df = df[
                (df[MODEL_COL] == model_a)
                &
                (df[SEED_COL] == seed)
                &
                (
                    df[TRANSFORMATION_COL]
                    == transformation
                )
            ].copy()

            # ------------------------------------------------
            # Extract model B
            # ------------------------------------------------

            b_df = df[
                (df[MODEL_COL] == model_b)
                &
                (df[SEED_COL] == seed)
                &
                (
                    df[TRANSFORMATION_COL]
                    == transformation
                )
            ].copy()

            # ------------------------------------------------
            # Pair SAME leaf/image
            # ------------------------------------------------

            merged = a_df.merge(
                b_df,
                on=IMAGE_ID_COL,
                suffixes=(
                    "_model_a",
                    "_model_b"
                ),
                how="inner"
            )

            # ------------------------------------------------
            # Verify exactly 128 paired samples
            # ------------------------------------------------

            assert len(merged) == 128, (
                "\nPairing failure!\n"
                f"Model A: {model_a}\n"
                f"Model B: {model_b}\n"
                f"Seed: {seed}\n"
                f"Transformation: {transformation}\n"
                f"Pairs found: {len(merged)}"
            )

            # ------------------------------------------------
            # Verify ground truth matches
            # ------------------------------------------------

            y_true_a = merged[
                f"{Y_TRUE_COL}_model_a"
            ].to_numpy()

            y_true_b = merged[
                f"{Y_TRUE_COL}_model_b"
            ].to_numpy()

            assert np.array_equal(
                y_true_a,
                y_true_b
            ), (
                "\nGround-truth mismatch!\n"
                f"{model_a} vs {model_b}\n"
                f"seed={seed}\n"
                f"transformation={transformation}"
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            y_pred_a = merged[
                f"{Y_PRED_COL}_model_a"
            ].to_numpy()

            y_pred_b = merged[
                f"{Y_PRED_COL}_model_b"
            ].to_numpy()

            # ------------------------------------------------
            # Correct / incorrect
            # ------------------------------------------------

            correct_a = (
                y_pred_a == y_true_a
            )

            correct_b = (
                y_pred_b == y_true_b
            )

            # ------------------------------------------------
            # McNemar discordant cells
            #
            # a:
            # Model A correct
            # Model B wrong
            #
            # b:
            # Model A wrong
            # Model B correct
            # ------------------------------------------------

            a = int(
                np.sum(
                    correct_a
                    &
                    (~correct_b)
                )
            )

            b = int(
                np.sum(
                    (~correct_a)
                    &
                    correct_b
                )
            )

            discordant = a + b

            # ------------------------------------------------
            # Exact two-sided McNemar
            # ------------------------------------------------

            if discordant == 0:

                p_value = 1.0

            else:

                p_value = binomtest(
                    k=min(a, b),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided"
                ).pvalue

            # ------------------------------------------------
            # Accuracy of both models
            # ------------------------------------------------

            accuracy_a = (
                correct_a.mean()
            )

            accuracy_b = (
                correct_b.mean()
            )

            accuracy_delta_pp = (
                accuracy_a
                -
                accuracy_b
            ) * 100.0

            # ------------------------------------------------
            # Direction
            # ------------------------------------------------

            if a > b:

                preferred = model_a

            elif b > a:

                preferred = model_b

            else:

                preferred = "Tie"

            results.append({

                "model_a": model_a,
                "model_b": model_b,

                "seed": seed,

                "transformation":
                    transformation,

                "n":
                    len(merged),

                "model_a_correct_model_b_wrong":
                    a,

                "model_a_wrong_model_b_correct":
                    b,

                "discordant_pairs":
                    discordant,

                "model_a_accuracy":
                    accuracy_a,

                "model_b_accuracy":
                    accuracy_b,

                "accuracy_delta_pp_a_minus_b":
                    accuracy_delta_pp,

                "p_value":
                    p_value,

                "preferred_model":
                    preferred
            })


# ============================================================
# 11. RESULTS DATAFRAME
# ============================================================

stats_df = pd.DataFrame(
    results
)


# ============================================================
# 12. BENJAMINI-HOCHBERG FDR
#
# Correction across ALL 288 tests
# ============================================================

assert len(stats_df) == 288, (
    f"Expected 288 tests, got "
    f"{len(stats_df)}"
)

reject, q_values, _, _ = multipletests(
    stats_df["p_value"].to_numpy(),
    alpha=0.05,
    method="fdr_bh"
)

stats_df["fdr_q"] = q_values

stats_df["significant_after_fdr"] = (
    reject
)


# ============================================================
# 13. SORT RESULTS
# ============================================================

stats_df = (
    stats_df
    .sort_values(
        [
            "model_a",
            "model_b",
            "transformation",
            "seed"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. SUMMARY BY MODEL PAIR × TRANSFORMATION
# ============================================================

pair_transformation_summary = []


for model_a, model_b in MODEL_PAIRS:

    for transformation in EXPECTED_TRANSFORMATIONS:

        subset = stats_df[
            (stats_df["model_a"] == model_a)
            &
            (stats_df["model_b"] == model_b)
            &
            (
                stats_df["transformation"]
                == transformation
            )
        ].copy()

        assert len(subset) == 3

        total_a_wins = subset[
            "model_a_correct_model_b_wrong"
        ].sum()

        total_b_wins = subset[
            "model_a_wrong_model_b_correct"
        ].sum()

        if total_a_wins > total_b_wins:

            overall_preferred = model_a

        elif total_b_wins > total_a_wins:

            overall_preferred = model_b

        else:

            overall_preferred = "Tie"

        pair_transformation_summary.append({

            "model_a":
                model_a,

            "model_b":
                model_b,

            "transformation":
                transformation,

            "n_seeds":
                len(subset),

            "significant_seeds":
                int(
                    subset[
                        "significant_after_fdr"
                    ].sum()
                ),

            "mean_fdr_q":
                subset["fdr_q"].mean(),

            "min_fdr_q":
                subset["fdr_q"].min(),

            "max_fdr_q":
                subset["fdr_q"].max(),

            "model_a_wins_mean":
                subset[
                    "model_a_correct_model_b_wrong"
                ].mean(),

            "model_b_wins_mean":
                subset[
                    "model_a_wrong_model_b_correct"
                ].mean(),

            "mean_discordant_pairs":
                subset[
                    "discordant_pairs"
                ].mean(),

            "mean_accuracy_delta_pp":
                subset[
                    "accuracy_delta_pp_a_minus_b"
                ].mean(),

            "overall_preferred":
                overall_preferred
        })


pair_transformation_summary_df = pd.DataFrame(
    pair_transformation_summary
)


# ============================================================
# 15. SUMMARY BY MODEL PAIR
# ============================================================

pair_summary = []

for model_a, model_b in MODEL_PAIRS:

    subset = stats_df[
        (stats_df["model_a"] == model_a)
        &
        (stats_df["model_b"] == model_b)
    ].copy()

    total_a_wins = subset[
        "model_a_correct_model_b_wrong"
    ].sum()

    total_b_wins = subset[
        "model_a_wrong_model_b_correct"
    ].sum()

    if total_a_wins > total_b_wins:

        overall_preferred = model_a

    elif total_b_wins > total_a_wins:

        overall_preferred = model_b

    else:

        overall_preferred = "Tie"

    pair_summary.append({

        "model_a":
            model_a,

        "model_b":
            model_b,

        "total_tests":
            len(subset),

        "significant_tests":
            int(
                subset[
                    "significant_after_fdr"
                ].sum()
            ),

        "percentage_significant":
            (
                subset[
                    "significant_after_fdr"
                ].mean()
                * 100
            ),

        "mean_fdr_q":
            subset["fdr_q"].mean(),

        "min_fdr_q":
            subset["fdr_q"].min(),

        "total_a_wins":
            total_a_wins,

        "total_b_wins":
            total_b_wins,

        "mean_discordant_pairs":
            subset[
                "discordant_pairs"
            ].mean(),

        "mean_accuracy_delta_pp_a_minus_b":
            subset[
                "accuracy_delta_pp_a_minus_b"
            ].mean(),

        "overall_preferred":
            overall_preferred
    })


pair_summary_df = pd.DataFrame(
    pair_summary
)


# ============================================================
# 16. SUMMARY BY TRANSFORMATION
# ============================================================

transformation_summary = []

for transformation in EXPECTED_TRANSFORMATIONS:

    subset = stats_df[
        stats_df["transformation"]
        == transformation
    ].copy()

    transformation_summary.append({

        "transformation":
            transformation,

        "total_tests":
            len(subset),

        "significant_tests":
            int(
                subset[
                    "significant_after_fdr"
                ].sum()
            ),

        "percentage_significant":
            (
                subset[
                    "significant_after_fdr"
                ].mean()
                * 100
            ),

        "mean_fdr_q":
            subset["fdr_q"].mean(),

        "min_fdr_q":
            subset["fdr_q"].min(),

        "mean_discordant_pairs":
            subset[
                "discordant_pairs"
            ].mean(),

        "mean_abs_accuracy_delta_pp":
            subset[
                "accuracy_delta_pp_a_minus_b"
            ].abs().mean()
    })


transformation_summary_df = pd.DataFrame(
    transformation_summary
)


# ============================================================
# 17. GLOBAL SIGNIFICANCE SUMMARY
# ============================================================

global_summary = pd.DataFrame({

    "metric": [

        "Total pairwise tests",

        "FDR-significant tests",

        "Percentage FDR-significant tests",

        "Number of model pairs",

        "Number of seeds",

        "Number of transformations",

        "Test type",

        "Multiple-comparison correction"
    ],

    "value": [

        len(stats_df),

        int(
            stats_df[
                "significant_after_fdr"
            ].sum()
        ),

        (
            stats_df[
                "significant_after_fdr"
            ].mean()
            * 100
        ),

        len(MODEL_PAIRS),

        len(SEEDS),

        len(EXPECTED_TRANSFORMATIONS),

        "Exact two-sided McNemar",

        "Benjamini-Hochberg FDR"
    ]
})


# ============================================================
# 18. SAVE ALL OUTPUTS
# ============================================================

ALL_TESTS_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_all_288_tests.csv"
)

PAIR_TRANSFORM_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_pair_transformation_summary.csv"
)

PAIR_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_pair_summary.csv"
)

TRANSFORMATION_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_transformation_summary.csv"
)

GLOBAL_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_global_summary.csv"
)

GROUP_SIZE_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_group_size_verification.csv"
)


stats_df.to_csv(
    ALL_TESTS_PATH,
    index=False
)

pair_transformation_summary_df.to_csv(
    PAIR_TRANSFORM_PATH,
    index=False
)

pair_summary_df.to_csv(
    PAIR_SUMMARY_PATH,
    index=False
)

transformation_summary_df.to_csv(
    TRANSFORMATION_SUMMARY_PATH,
    index=False
)

global_summary.to_csv(
    GLOBAL_SUMMARY_PATH,
    index=False
)

group_size_df.to_csv(
    GROUP_SIZE_PATH,
    index=False
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 110)
print("GLOBAL SUMMARY")
print("=" * 110)

display(
    global_summary
)


print("\n")
print("=" * 110)
print("MODEL-PAIR SUMMARY")
print("=" * 110)

display(
    pair_summary_df
)


print("\n")
print("=" * 110)
print("TRANSFORMATION SUMMARY")
print("=" * 110)

display(
    transformation_summary_df
)


print("\n")
print("=" * 110)
print(
    "PAIR × TRANSFORMATION SUMMARY"
)
print("=" * 110)

display(
    pair_transformation_summary_df
)


print("\n")
print("=" * 110)
print("ALL 288 TESTS")
print("=" * 110)

display(
    stats_df[
        [
            "model_a",
            "model_b",
            "seed",
            "transformation",
            "model_a_correct_model_b_wrong",
            "model_a_wrong_model_b_correct",
            "discordant_pairs",
            "model_a_accuracy",
            "model_b_accuracy",
            "accuracy_delta_pp_a_minus_b",
            "p_value",
            "fdr_q",
            "significant_after_fdr",
            "preferred_model"
        ]
    ]
)


# ============================================================
# 20. SHOW ONLY SIGNIFICANT RESULTS
# ============================================================

significant_df = stats_df[
    stats_df[
        "significant_after_fdr"
    ]
].copy()

print("\n")
print("=" * 110)
print(
    "FDR-SIGNIFICANT MODEL-vs-MODEL COMPARISONS"
)
print("=" * 110)

if len(significant_df) == 0:

    print(
        "No comparison remained significant "
        "after Benjamini-Hochberg FDR correction."
    )

else:

    display(
        significant_df[
            [
                "model_a",
                "model_b",
                "seed",
                "transformation",
                "model_a_correct_model_b_wrong",
                "model_a_wrong_model_b_correct",
                "discordant_pairs",
                "model_a_accuracy",
                "model_b_accuracy",
                "accuracy_delta_pp_a_minus_b",
                "p_value",
                "fdr_q",
                "preferred_model"
            ]
        ]
    )


# ============================================================
# 21. FINAL CHECKS
# ============================================================

assert len(stats_df) == 288

assert (
    stats_df[
        "significant_after_fdr"
    ].dtype == bool
)

assert np.all(
    stats_df["fdr_q"]
    >=
    stats_df["p_value"]
)


# Every model pair must have:
# 3 seeds × 16 transformations = 48 tests

pair_counts = (
    stats_df
    .groupby(
        [
            "model_a",
            "model_b"
        ]
    )
    .size()
)

assert np.all(
    pair_counts.values == 48
)


print("\n")
print("=" * 110)
print("SAVED FILES")
print("=" * 110)

print(
    ALL_TESTS_PATH
)

print(
    PAIR_TRANSFORM_PATH
)

print(
    PAIR_SUMMARY_PATH
)

print(
    TRANSFORMATION_SUMMARY_PATH
)

print(
    GLOBAL_SUMMARY_PATH
)

print(
    GROUP_SIZE_PATH
)


print("\n")
print("=" * 110)
print("✅ MODEL-vs-MODEL STATISTICAL ANALYSIS COMPLETE")
print("✅ 4 CNNs")
print("✅ 6 MODEL PAIRS")
print("✅ 3 SEEDS")
print("✅ 16 TRANSFORMATIONS")
print("✅ 288 EXACT McNEMAR TESTS")
print("✅ BENJAMINI-HOCHBERG FDR")
print("✅ SAME 128 TEST LEAVES PAIRED")
print("✅ NO TRAINING")
print("✅ NO CHECKPOINT SELECTION")
print("=" * 110)


# %% [Cell 68]
import os
import pandas as pd

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

print("=" * 100)
print("SEARCHING FOR MODEL-LEVEL PREDICTION FILES")
print("=" * 100)

candidates = []

for root, dirs, files in os.walk(ROOT):

    for file in files:

        if not file.lower().endswith(".csv"):
            continue

        path = os.path.join(root, file)

        try:
            sample = pd.read_csv(
                path,
                nrows=10
            )

            cols = set(sample.columns)

            # Files useful for model-vs-model analysis
            has_prediction_columns = (
                {"seed", "transformation", "original_image_id", "y_true", "y_pred"}
                .issubset(cols)
            )

            if has_prediction_columns:

                # read only columns needed for diagnosis
                full = pd.read_csv(
                    path,
                    usecols=[
                        c for c in [
                            "strategy",
                            "model",
                            "model_name",
                            "architecture",
                            "architecture_name",
                            "seed",
                            "transformation",
                            "original_image_id",
                            "y_true",
                            "y_pred"
                        ]
                        if c in cols
                    ]
                )

                models = []

                for c in [
                    "model",
                    "model_name",
                    "architecture",
                    "architecture_name",
                    "strategy"
                ]:

                    if c in full.columns:

                        vals = (
                            full[c]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        if len(vals) > 1:
                            models = vals
                            break

                candidates.append({
                    "path": path,
                    "rows": len(pd.read_csv(path, usecols=["seed"])),
                    "columns": list(cols),
                    "detected_values": models
                })

        except Exception:
            pass


print("\nFOUND CANDIDATE FILES:\n")

if not candidates:

    print(
        "❌ No suitable prediction CSV was found."
    )

else:

    for i, item in enumerate(
        candidates,
        start=1
    ):

        print("=" * 100)
        print(f"[{i}]")
        print("Path:")
        print(item["path"])
        print("Rows:")
        print(item["rows"])
        print("Detected model/strategy values:")
        print(item["detected_values"])
        print("Columns:")
        print(item["columns"])

# %% [Cell 69]
import os

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

MODEL_NAMES = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18",
]

print("=" * 100)
print("SEARCHING FOR MODEL CHECKPOINTS")
print("=" * 100)

found = []

for root, dirs, files in os.walk(ROOT):

    for file in files:

        lower = file.lower()

        if not (
            lower.endswith(".pt")
            or lower.endswith(".pth")
            or lower.endswith(".ckpt")
        ):
            continue

        path = os.path.join(root, file)

        name = file.lower()

        matched_models = []

        for model in MODEL_NAMES:

            aliases = [
                model.lower(),
                model.lower().replace("-", "_"),
                model.lower().replace("-", ""),
                model.lower().replace(" ", "_"),
                model.lower().replace(" ", ""),
            ]

            if any(alias in name for alias in aliases):
                matched_models.append(model)

        # Also inspect parent folders
        parent_text = root.lower()

        for model in MODEL_NAMES:

            aliases = [
                model.lower(),
                model.lower().replace("-", "_"),
                model.lower().replace("-", ""),
                model.lower().replace(" ", "_"),
                model.lower().replace(" ", ""),
            ]

            if any(alias in parent_text for alias in aliases):
                if model not in matched_models:
                    matched_models.append(model)

        if matched_models:

            found.append({
                "model": matched_models,
                "path": path
            })


print("\n")

if not found:

    print("❌ No model-specific checkpoints found.")

else:

    for i, item in enumerate(found, 1):

        print("=" * 100)
        print(f"[{i}] Model candidate: {item['model']}")
        print(item["path"])

# %% [Cell 70]
# ============================================================
# MODEL-LEVEL LOCKED TEST INFERENCE
#
# 4 CNN architectures
# 3 seeds
# 16 transformations
# 128 test leaves
#
# TOTAL EXPECTED PREDICTIONS:
# 4 × 3 × 16 × 128 = 24,576
#
# NO TRAINING
# NO VALIDATION
# NO CHECKPOINT SELECTION
# LOCKED TEST INFERENCE ONLY
# ============================================================

import os
import gc
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
from torchvision import transforms

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "model_level_locked_test"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "model_level_locked_test_predictions.csv"
)

BATCH_SIZE = 64
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 110)
print("MODEL-LEVEL LOCKED TEST INFERENCE")
print("=" * 110)

print("PyTorch :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("Device :", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 2. CHECK METADATA
# ============================================================

assert os.path.exists(
    METADATA_PATH
), (
    f"Metadata not found:\n{METADATA_PATH}"
)

metadata = pd.read_csv(
    METADATA_PATH
)

print("\nMetadata shape:")
print(metadata.shape)

print("\nMetadata columns:")
print(metadata.columns.tolist())


# ============================================================
# 3. AUTOMATICALLY DETECT METADATA COLUMNS
# ============================================================

def find_column(
    dataframe,
    candidates,
    description,
    required=True
):

    for candidate in candidates:

        if candidate in dataframe.columns:
            return candidate

    if required:

        raise ValueError(
            f"\nCould not detect {description} column.\n"
            f"Expected one of:\n{candidates}\n\n"
            f"Available columns:\n"
            f"{dataframe.columns.tolist()}"
        )

    return None


FILENAME_COL = find_column(
    metadata,
    [
        "filename",
        "Filename",
        "file_name",
        "FileName"
    ],
    "filename"
)

SPLIT_COL = find_column(
    metadata,
    [
        "split",
        "Split",
        "data_split",
        "dataset_split",
        "Dataset_Split"
    ],
    "split"
)

TRANSFORMATION_COL = find_column(
    metadata,
    [
        "transformation",
        "Transformation",
        "transform",
        "Transform"
    ],
    "transformation"
)

ORIGINAL_ID_COL = find_column(
    metadata,
    [
        "original_image_id",
        "Original_Image_ID",
        "original_leaf_id",
        "Original_Leaf_ID",
        "leaf_id",
        "Leaf_ID",
        "original_id",
        "Original_ID"
    ],
    "original leaf/image ID"
)

CLASS_COL = find_column(
    metadata,
    [
        "class",
        "Class",
        "label",
        "Label",
        "class_name",
        "Class_Name"
    ],
    "class label",
    required=False
)


print("\nDetected metadata columns:")
print("Filename       :", FILENAME_COL)
print("Split          :", SPLIT_COL)
print("Transformation :", TRANSFORMATION_COL)
print("Original ID    :", ORIGINAL_ID_COL)
print("Class          :", CLASS_COL)


# ============================================================
# 4. EXPECTED TRANSFORMATIONS
# ============================================================

EXPECTED_TRANSFORMATIONS = [
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


# ============================================================
# 5. FILTER TEST SET
# ============================================================

metadata["_split_normalized"] = (
    metadata[SPLIT_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

test_meta = metadata[
    metadata["_split_normalized"] == "test"
].copy()

if len(test_meta) == 0:

    raise ValueError(
        "No rows found for split='test'.\n"
        "Available split values:\n"
        f"{metadata[SPLIT_COL].unique().tolist()}"
    )


print("\nTest metadata rows:")
print(len(test_meta))


# ============================================================
# 6. VERIFY TEST STRUCTURE
# ============================================================

print("\nTransformation counts:")

print(
    test_meta[
        TRANSFORMATION_COL
    ].value_counts()
    .reindex(
        EXPECTED_TRANSFORMATIONS
    )
)


missing_transformations = [
    t for t in EXPECTED_TRANSFORMATIONS
    if t not in test_meta[
        TRANSFORMATION_COL
    ].astype(str).unique()
]

if missing_transformations:

    raise ValueError(
        "Missing transformations:\n"
        f"{missing_transformations}"
    )


# 16 transformations × 128 test leaves
assert len(test_meta) == 2048, (
    f"Expected 2048 test rows, "
    f"got {len(test_meta)}"
)


num_test_leaves = (
    test_meta[
        ORIGINAL_ID_COL
    ].nunique()
)

print(
    "\nUnique test leaves:",
    num_test_leaves
)

assert num_test_leaves == 128, (
    f"Expected 128 test leaves, "
    f"got {num_test_leaves}"
)


# Each leaf must have exactly 16 transformations
leaf_transform_counts = (
    test_meta
    .groupby(ORIGINAL_ID_COL)[
        TRANSFORMATION_COL
    ]
    .nunique()
)

assert (
    leaf_transform_counts == 16
).all(), (
    "At least one leaf does not contain "
    "all 16 transformations."
)


# ============================================================
# 7. BUILD IMAGE INDEX
# ============================================================

print("\n")
print("=" * 110)
print("BUILDING IMAGE INDEX")
print("=" * 110)

image_paths = {}

for subdir in [
    "Raw",
    "Augmented"
]:

    folder = os.path.join(
        ROOT,
        subdir
    )

    if not os.path.exists(folder):
        continue

    for current_root, dirs, files in os.walk(
        folder
    ):

        for filename in files:

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                continue

            full_path = os.path.join(
                current_root,
                filename
            )

            if filename in image_paths:

                raise RuntimeError(
                    "Duplicate filename detected:\n"
                    f"{filename}\n"
                    f"{image_paths[filename]}\n"
                    f"{full_path}"
                )

            image_paths[
                filename
            ] = full_path


print(
    "Indexed images:",
    len(image_paths)
)


# ============================================================
# 8. VERIFY ALL TEST FILES EXIST
# ============================================================

missing_files = []

for filename in test_meta[
    FILENAME_COL
].astype(str):

    if filename not in image_paths:

        missing_files.append(
            filename
        )


if missing_files:

    print("\nFirst missing files:")

    for f in missing_files[:20]:
        print(f)

    raise FileNotFoundError(
        f"\nMissing {len(missing_files)} test images."
    )


print(
    "All test images found:",
    len(test_meta)
)


# ============================================================
# 9. SORT TEST SET
# ============================================================

transformation_order = {
    name: i
    for i, name in enumerate(
        EXPECTED_TRANSFORMATIONS
    )
}

test_meta["_transform_order"] = (
    test_meta[
        TRANSFORMATION_COL
    ]
    .map(transformation_order)
)

test_meta = (
    test_meta
    .sort_values(
        [
            ORIGINAL_ID_COL,
            "_transform_order"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 10. IMAGE PREPROCESSING
#
# Same base preprocessing:
# RGB → 224×224 → Tensor → ImageNet normalization
# ============================================================

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]

base_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ============================================================
# 11. DATASET
# ============================================================

class LockedTestDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        image_index,
        filename_col,
        transform
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_index = image_index
        self.filename_col = filename_col
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        filename = str(
            row[self.filename_col]
        )

        path = self.image_index[
            filename
        ]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return (
            image,
            index
        )


# ============================================================
# 12. MODEL BUILDERS
# ============================================================

NUM_CLASSES = 3


def build_model(
    model_name
):

    if model_name == "MobileNetV2":

         model = models.mobilenet_v2(
            weights=None
        )

         model.classifier[1] = nn.Linear(
            model.last_channel,
            NUM_CLASSES
        )

    elif model_name == "MobileNetV3-Small":

        model = models.mobilenet_v3_small(
            weights=None
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            NUM_CLASSES
        )

    elif model_name == "EfficientNet-B0":

        model = models.efficientnet_b0(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )

    elif model_name == "ResNet18":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# 13. CHECKPOINT PATHS
# ============================================================

CHECKPOINTS = {

    "MobileNetV2": {

        42:
        os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed2026.pth"
        )
    },

    "MobileNetV3-Small": {

        42:
        os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed2026.pth"
        )
    },

    "EfficientNet-B0": {

        42:
        os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed2026.pth"
        )
    },

    "ResNet18": {

        42:
        os.path.join(
            ROOT,
            "baseline_resnet18_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_resnet18_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_resnet18_seed2026.pth"
        )
    }
}


MODELS = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18"
]

SEEDS = [
    42,
    1337,
    2026
]


# ============================================================
# 14. VERIFY CHECKPOINTS
# ============================================================

print("\n")
print("=" * 110)
print("CHECKPOINT VERIFICATION")
print("=" * 110)

for model_name in MODELS:

    for seed in SEEDS:

        path = CHECKPOINTS[
            model_name
        ][seed]

        print(
            f"{model_name:20s} | "
            f"seed={seed:4d} | "
            f"{os.path.basename(path)}"
        )

        assert os.path.exists(
            path
        ), (
            f"Checkpoint not found:\n{path}"
        )


# ============================================================
# 15. CHECKPOINT LOADER
# ============================================================

def extract_state_dict(
    checkpoint
):

    # Case 1: raw state_dict
    if isinstance(
        checkpoint,
        OrderedDict
    ):
        return checkpoint

    if isinstance(
        checkpoint,
        dict
    ):

        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights"
        ]

        for key in candidate_keys:

            if key in checkpoint:

                candidate = checkpoint[key]

                if isinstance(
                    candidate,
                    (dict, OrderedDict)
                ):
                    return candidate

    raise ValueError(
        "Could not extract state_dict "
        "from checkpoint."
    )


def clean_state_dict(
    state_dict
):

    cleaned = OrderedDict()

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "model."
        ):

            new_key = new_key[
                len("model.") :
            ]

        cleaned[new_key] = value

    return cleaned


# ============================================================
# 16. LABEL DECODING
# ============================================================

CLASS_TO_INDEX = {

    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

INDEX_TO_CLASS = {
    0: "Healthy_Leaves",
    1: "Little_Leaf",
    2: "Phomopsis_Blight"
}


def get_true_label(
    row
):

    # Preferred: numeric y_true if present
    if "y_true" in row.index:

        value = row["y_true"]

        if pd.notna(value):

            try:
                return int(value)
            except Exception:
                pass

    # Otherwise use class label
    if CLASS_COL is not None:

        value = str(
            row[CLASS_COL]
        ).strip()

        if value in CLASS_TO_INDEX:

            return CLASS_TO_INDEX[
                value
            ]

    raise ValueError(
        "Could not determine ground-truth "
        "class for row."
    )


test_meta["y_true_numeric"] = (
    test_meta
    .apply(
        get_true_label,
        axis=1
    )
)


# ============================================================
# 17. INFERENCE FUNCTION
# ============================================================

@torch.inference_mode()
def run_inference(
    model,
    dataframe,
    model_name,
    seed,
    transformation
):

    dataset = LockedTestDataset(
        dataframe=dataframe,
        image_index=image_paths,
        filename_col=FILENAME_COL,
        transform=base_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    all_predictions = []
    all_probabilities = []

    model.eval()

    for images, indices in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            images
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.append(
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    assert len(
        all_predictions
    ) == len(dataframe)

    assert (
        all_probabilities.shape
        ==
        (len(dataframe), 3)
    )

    result = dataframe.copy()

    result["model"] = model_name
    result["seed"] = seed
    result["transformation"] = transformation

    result["y_pred"] = np.asarray(
        all_predictions,
        dtype=int
    )

    result["prob_healthy"] = (
        all_probabilities[:, 0]
    )

    result["prob_little_leaf"] = (
        all_probabilities[:, 1]
    )

    result["prob_phomopsis"] = (
        all_probabilities[:, 2]
    )

    result["correct"] = (
        result["y_pred"].values
        ==
        result["y_true_numeric"].values
    )

    return result


# ============================================================
# 18. RUN ALL 192 INFERENCE GROUPS
#
# 4 models × 3 seeds × 16 transformations
# ============================================================

all_results = []

total_runs = (
    len(MODELS)
    * len(SEEDS)
    * len(EXPECTED_TRANSFORMATIONS)
)

run_counter = 0


for model_name in MODELS:

    for seed in SEEDS:

        checkpoint_path = CHECKPOINTS[
            model_name
        ][seed]

        print("\n")
        print("=" * 110)
        print(
            f"LOADING: {model_name} | seed={seed}"
        )
        print(
            os.path.basename(
                checkpoint_path
            )
        )
        print("=" * 110)

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = build_model(
            model_name
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        state_dict = extract_state_dict(
            checkpoint
        )

        state_dict = clean_state_dict(
            state_dict
        )

        missing_keys, unexpected_keys = (
            model.load_state_dict(
                state_dict,
                strict=False
            )
        )

        if missing_keys:

            print(
                "\n⚠ Missing keys:"
            )

            print(
                missing_keys[:20]
            )

        if unexpected_keys:

            print(
                "\n⚠ Unexpected keys:"
            )

            print(
                unexpected_keys[:20]
            )

        # If the checkpoint is incompatible,
        # stop instead of silently producing bad predictions.
        if len(missing_keys) > 0:

            raise RuntimeError(
                f"\nCheckpoint loading produced "
                f"missing keys for "
                f"{model_name}, seed={seed}:\n"
                f"{missing_keys}"
            )

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        model = model.to(
            DEVICE
        )

        model.eval()

        # ----------------------------------------------------
        # Run each transformation
        # ----------------------------------------------------

        for transformation in EXPECTED_TRANSFORMATIONS:

            run_counter += 1

            print(
                f"\n[{run_counter}/{total_runs}] "
                f"{model_name} | "
                f"seed={seed} | "
                f"{transformation}"
            )

            subset = test_meta[
                test_meta[
                    TRANSFORMATION_COL
                ].astype(str)
                == transformation
            ].copy()

            # Sort by original leaf ID so
            # identical pairing is maintained
            subset = (
                subset
                .sort_values(
                    ORIGINAL_ID_COL
                )
                .reset_index(
                    drop=True
                )
            )

            assert len(subset) == 128, (
                f"Expected 128 rows for "
                f"{transformation}, got "
                f"{len(subset)}"
            )

            result = run_inference(
                model=model,
                dataframe=subset,
                model_name=model_name,
                seed=seed,
                transformation=transformation
            )

            # Keep only useful output columns
            result = result[
                [
                    "model",
                    "seed",
                    TRANSFORMATION_COL,
                    ORIGINAL_ID_COL,
                    FILENAME_COL,
                    "y_true_numeric",
                    "y_pred",
                    "correct",
                    "prob_healthy",
                    "prob_little_leaf",
                    "prob_phomopsis"
                ]
            ].copy()

            result = result.rename(
                columns={
                    TRANSFORMATION_COL:
                        "transformation",

                    ORIGINAL_ID_COL:
                        "original_image_id",

                    FILENAME_COL:
                        "filename",

                    "y_true_numeric":
                        "y_true"
                }
            )

            all_results.append(
                result
            )

        # ----------------------------------------------------
        # Free model memory
        # ----------------------------------------------------

        del model
        del checkpoint
        del state_dict

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 19. CONCATENATE
# ============================================================

final_predictions = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# 20. FINAL SHAPE ASSERTION
# ============================================================

EXPECTED_ROWS = (
    4
    * 3
    * 16
    * 128
)

print("\n")
print("=" * 110)
print("FINAL PREDICTION FILE")
print("=" * 110)

print(
    "Expected rows:",
    EXPECTED_ROWS
)

print(
    "Actual rows:",
    len(final_predictions)
)

assert (
    len(final_predictions)
    ==
    EXPECTED_ROWS
), (
    f"Expected {EXPECTED_ROWS} "
    f"rows, got {len(final_predictions)}"
)


# ============================================================
# 21. VERIFY GROUP COUNTS
# ============================================================

group_counts = (
    final_predictions
    .groupby(
        [
            "model",
            "seed",
            "transformation"
        ]
    )
    .size()
    .reset_index(
        name="n"
    )
)

assert len(group_counts) == (
    4 * 3 * 16
)

assert (
    group_counts["n"] == 128
).all()


# ============================================================
# 22. VERIFY SAME TEST LEAVES ACROSS MODELS
# ============================================================

reference_ids = set(
    test_meta[
        ORIGINAL_ID_COL
    ]
    .astype(str)
    .unique()
)

for model_name in MODELS:

    model_ids = set(
        final_predictions[
            final_predictions["model"]
            == model_name
        ][
            "original_image_id"
        ]
        .astype(str)
        .unique()
    )

    assert (
        model_ids
        ==
        reference_ids
    ), (
        f"Leaf IDs mismatch for "
        f"{model_name}"
    )


# ============================================================
# 23. CLEAN UP DATA TYPES
# ============================================================

final_predictions["seed"] = (
    final_predictions["seed"]
    .astype(int)
)

final_predictions["y_true"] = (
    final_predictions["y_true"]
    .astype(int)
)

final_predictions["y_pred"] = (
    final_predictions["y_pred"]
    .astype(int)
)

final_predictions["correct"] = (
    final_predictions["correct"]
    .astype(bool)
)


# ============================================================
# 24. SAVE
# ============================================================

final_predictions.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 25. VERIFY SAVED FILE
# ============================================================

saved = pd.read_csv(
    OUTPUT_PATH
)

print("\nSaved file:")
print(OUTPUT_PATH)

print("\nSaved shape:")
print(saved.shape)

assert saved.shape[0] == 24576


# ============================================================
# 26. DISPLAY SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("MODEL COUNTS")
print("=" * 110)

print(
    final_predictions[
        "model"
    ].value_counts()
)


print("\n")
print("=" * 110)
print("TRANSFORMATION COUNTS")
print("=" * 110)

print(
    final_predictions[
        "transformation"
    ].value_counts()
)


print("\n")
print("=" * 110)
print("MEAN ACCURACY BY MODEL")
print("=" * 110)

model_accuracy = (
    final_predictions
    .groupby(
        "model"
    )["correct"]
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)

print(
    model_accuracy
)


print("\n")
print("=" * 110)
print("MEAN ACCURACY BY MODEL × TRANSFORMATION")
print("=" * 110)

model_transform_accuracy = (
    final_predictions
    .groupby(
        [
            "model",
            "transformation"
        ]
    )["correct"]
    .mean()
    .mul(100)
    .reset_index()
)

display(
    model_transform_accuracy
)


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 110)
print("✅ MODEL-LEVEL INFERENCE COMPLETE")
print("=" * 110)

print("Models              : 4")
print("Seeds               : 3")
print("Transformations     : 16")
print("Test leaves         : 128")
print("Expected predictions : 24,576")
print(
    "Actual predictions   :",
    len(final_predictions)
)

print("\nPrediction file:")
print(OUTPUT_PATH)

print(
    "\nNext step:"
)

print(
    "Run Exact McNemar + Benjamini-Hochberg FDR "
    "on this prediction file."
)

print("=" * 110)


# %% [Cell 71]
# ============================================================
# MODEL-LEVEL LOCKED TEST INFERENCE — FINAL CORRECTED VERSION
#
# Dataset metadata columns:
#   filename
#   class_label
#   original_image_id
#   preprocessing_technique
#   data_split
#   device_used
#   lighting_condition
#
# 4 CNN architectures
# 3 seeds
# 16 transformations
# 128 test leaves
#
# Expected:
# 4 × 3 × 16 × 128 = 24,576 predictions
#
# NO TRAINING
# NO VALIDATION
# NO CHECKPOINT SELECTION
# LOCKED TEST INFERENCE ONLY
# ============================================================

import os
import gc
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
from torchvision import transforms

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "model_level_locked_test"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "model_level_locked_test_predictions.csv"
)

BATCH_SIZE = 64
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("MODEL-LEVEL LOCKED TEST INFERENCE")
print("=" * 110)

print("PyTorch     :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("Device      :", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU         :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 2. LOAD METADATA
# ============================================================

assert os.path.exists(
    METADATA_PATH
), f"Metadata not found:\n{METADATA_PATH}"

metadata = pd.read_csv(
    METADATA_PATH
)

print("\nMetadata shape:")
print(metadata.shape)

print("\nMetadata columns:")
print(metadata.columns.tolist())


# ============================================================
# 3. REAL METADATA COLUMN NAMES
# ============================================================

FILENAME_COL = "filename"
CLASS_COL = "class_label"
ORIGINAL_ID_COL = "original_image_id"
TRANSFORMATION_COL = "preprocessing_technique"
SPLIT_COL = "data_split"


REQUIRED_COLUMNS = [
    FILENAME_COL,
    CLASS_COL,
    ORIGINAL_ID_COL,
    TRANSFORMATION_COL,
    SPLIT_COL
]

for col in REQUIRED_COLUMNS:

    assert col in metadata.columns, (
        f"\nRequired metadata column missing: {col}\n"
        f"Available columns:\n{metadata.columns.tolist()}"
    )

print("\nUsing metadata columns:")
print("Filename       :", FILENAME_COL)
print("Class          :", CLASS_COL)
print("Original ID    :", ORIGINAL_ID_COL)
print("Transformation :", TRANSFORMATION_COL)
print("Split          :", SPLIT_COL)


# ============================================================
# 4. NORMALIZE STRING FIELDS
# ============================================================

for col in [
    FILENAME_COL,
    CLASS_COL,
    ORIGINAL_ID_COL,
    TRANSFORMATION_COL,
    SPLIT_COL
]:

    metadata[col] = (
        metadata[col]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 5. EXPECTED TRANSFORMATIONS
# ============================================================

EXPECTED_TRANSFORMATIONS = [
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


# ============================================================
# 6. CHECK ACTUAL DATASET VOCABULARY
# ============================================================

print("\n")
print("=" * 110)
print("DATASET VOCABULARY CHECK")
print("=" * 110)

print("\nActual split values:")
print(
    metadata[SPLIT_COL]
    .value_counts(dropna=False)
)

print("\nActual transformation values:")
print(
    metadata[TRANSFORMATION_COL]
    .value_counts(dropna=False)
)

print("\nActual class values:")
print(
    metadata[CLASS_COL]
    .value_counts(dropna=False)
)


actual_transformations = set(
    metadata[
        TRANSFORMATION_COL
    ].unique()
)

expected_transformations = set(
    EXPECTED_TRANSFORMATIONS
)


missing_transformations = (
    expected_transformations
    -
    actual_transformations
)

unexpected_transformations = (
    actual_transformations
    -
    expected_transformations
)

if missing_transformations:

    raise ValueError(
        "\nMissing expected transformations:\n"
        f"{sorted(missing_transformations)}\n\n"
        "Actual transformations:\n"
        f"{sorted(actual_transformations)}"
    )

if unexpected_transformations:

    print(
        "\n⚠ Additional transformation values found:"
    )

    print(
        sorted(
            unexpected_transformations
        )
    )

    raise ValueError(
        "\nUnexpected transformation vocabulary detected. "
        "Please verify before inference."
    )


# ============================================================
# 7. NORMALIZE SPLIT
# ============================================================

metadata["_split_normalized"] = (
    metadata[SPLIT_COL]
    .str.lower()
)


# ============================================================
# 8. EXTRACT TEST SET
# ============================================================

test_meta = metadata[
    metadata["_split_normalized"] == "test"
].copy()

if len(test_meta) == 0:

    raise ValueError(
        "\nNo rows with data_split='test' found.\n"
        f"Actual values:\n"
        f"{metadata[SPLIT_COL].unique().tolist()}"
    )

print("\n")
print("=" * 110)
print("TEST SET VERIFICATION")
print("=" * 110)

print(
    "\nTest rows:",
    len(test_meta)
)

print(
    "Unique test leaves:",
    test_meta[
        ORIGINAL_ID_COL
    ].nunique()
)


# Expected:
# 128 leaves × 16 transformations = 2048
assert len(test_meta) == 2048, (
    f"\nExpected 2048 test rows, "
    f"got {len(test_meta)}"
)

assert (
    test_meta[
        ORIGINAL_ID_COL
    ].nunique()
    == 128
), (
    "Expected 128 unique test leaves."
)


# ============================================================
# 9. VERIFY EACH LEAF HAS ALL 16 TRANSFORMATIONS
# ============================================================

leaf_transform_counts = (
    test_meta
    .groupby(
        ORIGINAL_ID_COL
    )[TRANSFORMATION_COL]
    .nunique()
)

if not (
    leaf_transform_counts == 16
).all():

    bad_leaves = leaf_transform_counts[
        leaf_transform_counts != 16
    ]

    print(
        "\nProblematic leaves:"
    )

    print(
        bad_leaves
    )

    raise ValueError(
        "Not every test leaf has all 16 transformations."
    )


# ============================================================
# 10. CLASS LABEL MAPPING
# ============================================================

CLASS_TO_INDEX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

INDEX_TO_CLASS = {
    0: "Healthy_Leaves",
    1: "Little_Leaf",
    2: "Phomopsis_Blight"
}


actual_classes = set(
    test_meta[
        CLASS_COL
    ].unique()
)

expected_classes = set(
    CLASS_TO_INDEX.keys()
)

if actual_classes != expected_classes:

    raise ValueError(
        "\nUnexpected class labels.\n"
        f"Expected: {sorted(expected_classes)}\n"
        f"Actual:   {sorted(actual_classes)}"
    )


test_meta["y_true"] = (
    test_meta[CLASS_COL]
    .map(CLASS_TO_INDEX)
    .astype(int)
)


# ============================================================
# 11. BUILD ROBUST IMAGE INDEX
#
# metadata filename contains relative paths such as:
#   Raw/Healthy_Leaves/...
#   Augmented/Healthy_Leaves/...
#
# We index BOTH:
#   1) relative path from ROOT
#   2) basename
# ============================================================

print("\n")
print("=" * 110)
print("BUILDING ROBUST IMAGE INDEX")
print("=" * 110)

image_paths = {}

for subdir in [
    "Raw",
    "Augmented"
]:

    folder = os.path.join(
        ROOT,
        subdir
    )

    if not os.path.exists(folder):
        continue

    for current_root, dirs, files in os.walk(folder):

        for filename in files:

            if not filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            ):
                continue

            full_path = os.path.abspath(
                os.path.join(
                    current_root,
                    filename
                )
            )

            # ------------------------------------------------
            # Relative path from dataset root
            # Example:
            # Raw/Healthy_Leaves/file.jpg
            # ------------------------------------------------

            relative_path = os.path.relpath(
                full_path,
                ROOT
            )

            # Normalize path separators
            relative_path = (
                relative_path
                .replace("\\", "/")
            )

            # Store relative path
            image_paths[
                relative_path
            ] = full_path

            # Also store basename as fallback
            basename = os.path.basename(
                full_path
            )

            # Only assign basename when unique
            if basename not in image_paths:

                image_paths[
                    basename
                ] = full_path


print(
    "Indexed path keys:",
    len(image_paths)
)


# ============================================================
# 12. ROBUST IMAGE PATH RESOLVER
# ============================================================

def resolve_image_path(
    filename
):

    filename = str(
        filename
    ).strip()

    # Normalize separators
    normalized = (
        filename
        .replace("\\", "/")
        .lstrip("./")
    )

    # --------------------------------------------------------
    # Method 1:
    # Exact relative-path lookup
    # --------------------------------------------------------

    if normalized in image_paths:

        return image_paths[
            normalized
        ]

    # --------------------------------------------------------
    # Method 2:
    # Direct path from ROOT
    # --------------------------------------------------------

    direct_path = os.path.join(
        ROOT,
        normalized
    )

    if os.path.isfile(
        direct_path
    ):

        return os.path.abspath(
            direct_path
        )

    # --------------------------------------------------------
    # Method 3:
    # Basename fallback
    # --------------------------------------------------------

    basename = os.path.basename(
        normalized
    )

    if basename in image_paths:

        return image_paths[
            basename
        ]

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    return None


# ============================================================
# 13. VERIFY ALL TEST FILES EXIST
# ============================================================

missing_files = []

resolved_paths = {}

for filename in test_meta[
    FILENAME_COL
].astype(str):

    resolved = resolve_image_path(
        filename
    )

    if resolved is None:

        missing_files.append(
            filename
        )

    else:

        resolved_paths[
            filename
        ] = resolved


if missing_files:

    print(
        "\nFirst missing files:"
    )

    for filename in missing_files[:20]:
        print(filename)

    raise FileNotFoundError(
        f"\nMissing {len(missing_files)} test images."
    )


print(
    "All test images resolved:",
    len(resolved_paths)
)


# ============================================================
# 14. OPTIONAL SANITY CHECK
# ============================================================

print("\nExample resolved paths:")

for filename in list(
    resolved_paths.keys()
)[:5]:

    print(
        f"{filename}"
        f"  -->  "
        f"{resolved_paths[filename]}"
    )


# ============================================================
# IMPORTANT:
# Update Dataset __getitem__ to use the resolver
# ============================================================

class LockedTestDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        image_index,
        filename_col,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.image_index = image_index
        self.filename_col = filename_col
        self.transform = transform

    def __len__(
        self
    ):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        filename = str(
            row[
                self.filename_col
            ]
        ).strip()

        # Resolve exact dataset path
        path = resolve_image_path(
            filename
        )

        if path is None:

            raise FileNotFoundError(
                f"\nImage could not be resolved:\n"
                f"{filename}"
            )

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return (
            image,
            index
        )
# ============================================================
# 13. TRANSFORMATION ORDER
# ============================================================

transformation_order = {
    name: i
    for i, name in enumerate(
        EXPECTED_TRANSFORMATIONS
    )
}

test_meta["_transform_order"] = (
    test_meta[
        TRANSFORMATION_COL
    ].map(
        transformation_order
    )
)

test_meta = (
    test_meta
    .sort_values(
        [
            ORIGINAL_ID_COL,
            "_transform_order"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. IMAGE PREPROCESSING
#
# Same inference preprocessing:
# RGB → Resize 224×224 → Tensor → ImageNet normalization
# ============================================================

base_transform = transforms.Compose([
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


# ============================================================
# 15. DATASET CLASS
# ============================================================

class LockedTestDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        image_index,
        filename_col,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.image_index = image_index
        self.filename_col = filename_col
        self.transform = transform

    def __len__(
        self
    ):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        filename = str(
            row[
                self.filename_col
            ]
        )

        path = self.image_index[
            filename
        ]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return (
            image,
            index
        )


# ============================================================
# 16. MODEL BUILDERS
# ============================================================

NUM_CLASSES = 3


def build_model(
    model_name
):

    if model_name == "MobileNetV2":

        model = models.mobilenet_v2(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.last_channel,
            NUM_CLASSES
        )

    elif model_name == "MobileNetV3-Small":

        model = models.mobilenet_v3_small(
            weights=None
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            NUM_CLASSES
        )

    elif model_name == "EfficientNet-B0":

        model = models.efficientnet_b0(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )

    elif model_name == "ResNet18":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ============================================================
# 17. CHECKPOINT PATHS
# ============================================================

CHECKPOINTS = {

    "MobileNetV2": {

        42:
        os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed2026.pth"
        )
    },

    "MobileNetV3-Small": {

        42:
        os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed2026.pth"
        )
    },

    "EfficientNet-B0": {

        42:
        os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed2026.pth"
        )
    },

    "ResNet18": {

        42:
        os.path.join(
            ROOT,
            "baseline_resnet18_seed42.pth"
        ),

        1337:
        os.path.join(
            ROOT,
            "baseline_resnet18_seed1337.pth"
        ),

        2026:
        os.path.join(
            ROOT,
            "baseline_resnet18_seed2026.pth"
        )
    }
}


MODELS = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18"
]

SEEDS = [
    42,
    1337,
    2026
]


# ============================================================
# 18. VERIFY CHECKPOINTS
# ============================================================

print("\n")
print("=" * 110)
print("CHECKPOINT VERIFICATION")
print("=" * 110)

for model_name in MODELS:

    for seed in SEEDS:

        path = CHECKPOINTS[
            model_name
        ][seed]

        print(
            f"{model_name:20s} | "
            f"seed={seed:4d} | "
            f"{os.path.basename(path)}"
        )

        assert os.path.exists(
            path
        ), (
            f"\nCheckpoint not found:\n{path}"
        )


# ============================================================
# 19. CHECKPOINT HELPERS
# ============================================================

def extract_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        OrderedDict
    ):

        return checkpoint

    if isinstance(
        checkpoint,
        dict
    ):

        possible_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights"
        ]

        for key in possible_keys:

            if key not in checkpoint:
                continue

            candidate = checkpoint[
                key
            ]

            if isinstance(
                candidate,
                (
                    dict,
                    OrderedDict
                )
            ):

                return candidate

    raise ValueError(
        "\nCould not extract state_dict "
        "from checkpoint."
    )


def clean_state_dict(
    state_dict
):

    cleaned = OrderedDict()

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "model."
        ):

            new_key = new_key[
                len("model.") :
            ]

        cleaned[
            new_key
        ] = value

    return cleaned


# ============================================================
# 20. INFERENCE FUNCTION
# ============================================================

@torch.inference_mode()
def run_inference(
    model,
    dataframe
):

    dataset = LockedTestDataset(
        dataframe=dataframe,
        image_index=image_paths,
        filename_col=FILENAME_COL,
        transform=base_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    predictions = []
    probabilities = []

    model.eval()

    for images, indices in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            images
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        predictions.extend(
            preds
            .cpu()
            .numpy()
            .tolist()
        )

        probabilities.append(
            probs
            .cpu()
            .numpy()
        )

    probabilities = np.concatenate(
        probabilities,
        axis=0
    )

    assert len(
        predictions
    ) == len(dataframe)

    assert probabilities.shape == (
        len(dataframe),
        3
    )

    return (
        np.asarray(
            predictions,
            dtype=int
        ),
        probabilities
    )


# ============================================================
# 21. RUN ALL INFERENCE
#
# 4 × 3 × 16 = 192 inference groups
# ============================================================

all_results = []

total_runs = (
    len(MODELS)
    * len(SEEDS)
    * len(EXPECTED_TRANSFORMATIONS)
)

run_counter = 0


for model_name in MODELS:

    for seed in SEEDS:

        checkpoint_path = (
            CHECKPOINTS[
                model_name
            ][seed]
        )

        print("\n")
        print("=" * 110)
        print(
            f"LOADING {model_name} | SEED {seed}"
        )
        print(
            os.path.basename(
                checkpoint_path
            )
        )
        print("=" * 110)

        # ----------------------------------------------------
        # Build architecture
        # ----------------------------------------------------

        model = build_model(
            model_name
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        state_dict = extract_state_dict(
            checkpoint
        )

        state_dict = clean_state_dict(
            state_dict
        )

        load_result = (
            model.load_state_dict(
                state_dict,
                strict=False
            )
        )

        missing_keys = (
            load_result.missing_keys
        )

        unexpected_keys = (
            load_result.unexpected_keys
        )

        if missing_keys:

            raise RuntimeError(
                f"\nMissing checkpoint keys for "
                f"{model_name}, seed={seed}:\n"
                f"{missing_keys}"
            )

        if unexpected_keys:

            print(
                "\n⚠ Unexpected checkpoint keys:"
            )

            print(
                unexpected_keys[:20]
            )

        model = model.to(
            DEVICE
        )

        model.eval()

        # ----------------------------------------------------
        # Every transformation
        # ----------------------------------------------------

        for transformation in (
            EXPECTED_TRANSFORMATIONS
        ):

            run_counter += 1

            print(
                f"[{run_counter}/{total_runs}] "
                f"{model_name} | "
                f"seed={seed} | "
                f"{transformation}"
            )

            subset = test_meta[
                test_meta[
                    TRANSFORMATION_COL
                ]
                ==
                transformation
            ].copy()

            # Stable leaf ordering
            subset = (
                subset
                .sort_values(
                    ORIGINAL_ID_COL
                )
                .reset_index(
                    drop=True
                )
            )

            assert len(
                subset
            ) == 128, (
                f"\nExpected 128 rows for "
                f"{transformation}, got "
                f"{len(subset)}"
            )

            # Inference
            predictions, probabilities = (
                run_inference(
                    model=model,
                    dataframe=subset
                )
            )

            # ------------------------------------------------
            # Build result
            # ------------------------------------------------

            result = pd.DataFrame({

                "model":
                    model_name,

                "seed":
                    seed,

                "transformation":
                    transformation,

                "original_image_id":
                    subset[
                        ORIGINAL_ID_COL
                    ].values,

                "filename":
                    subset[
                        FILENAME_COL
                    ].values,

                "y_true":
                    subset[
                        "y_true"
                    ].values,

                "y_pred":
                    predictions,

                "correct":
                    (
                        predictions
                        ==
                        subset[
                            "y_true"
                        ].values
                    ),

                "prob_healthy":
                    probabilities[:, 0],

                "prob_little_leaf":
                    probabilities[:, 1],

                "prob_phomopsis":
                    probabilities[:, 2]
            })

            all_results.append(
                result
            )

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del model
        del checkpoint
        del state_dict

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 22. CONCATENATE ALL RESULTS
# ============================================================

final_predictions = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# 23. EXPECTED SHAPE
# ============================================================

EXPECTED_ROWS = (
    4
    * 3
    * 16
    * 128
)

print("\n")
print("=" * 110)
print("FINAL PREDICTION DATASET")
print("=" * 110)

print(
    "Expected rows :",
    EXPECTED_ROWS
)

print(
    "Actual rows   :",
    len(final_predictions)
)

assert (
    len(final_predictions)
    ==
    EXPECTED_ROWS
), (
    f"\nExpected {EXPECTED_ROWS} rows, "
    f"got {len(final_predictions)}"
)


# ============================================================
# 24. VERIFY GROUP COUNTS
# ============================================================

group_counts = (
    final_predictions
    .groupby(
        [
            "model",
            "seed",
            "transformation"
        ]
    )
    .size()
    .reset_index(
        name="n"
    )
)

assert (
    len(group_counts)
    ==
    4 * 3 * 16
)

assert (
    group_counts["n"] == 128
).all()


# ============================================================
# 25. VERIFY SAME 128 LEAVES FOR ALL MODELS
# ============================================================

reference_ids = set(
    test_meta[
        ORIGINAL_ID_COL
    ]
    .astype(str)
    .unique()
)

for model_name in MODELS:

    model_ids = set(
        final_predictions[
            final_predictions["model"]
            ==
            model_name
        ][
            "original_image_id"
        ]
        .astype(str)
        .unique()
    )

    assert (
        model_ids
        ==
        reference_ids
    ), (
        f"\nTest leaf IDs mismatch "
        f"for {model_name}"
    )


# ============================================================
# 26. VERIFY SEEDS / MODELS / TRANSFORMATIONS
# ============================================================

assert set(
    final_predictions[
        "model"
    ].unique()
) == set(MODELS)

assert set(
    final_predictions[
        "seed"
    ].unique()
) == set(SEEDS)

assert set(
    final_predictions[
        "transformation"
    ].unique()
) == set(
    EXPECTED_TRANSFORMATIONS
)


# ============================================================
# 27. SAVE FINAL FILE
# ============================================================

final_predictions.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 28. RELOAD AND VERIFY
# ============================================================

saved = pd.read_csv(
    OUTPUT_PATH
)

assert len(saved) == 24576

print("\n")
print("=" * 110)
print("SAVED FILE")
print("=" * 110)

print(
    OUTPUT_PATH
)

print(
    "\nShape:",
    saved.shape
)


# ============================================================
# 29. MODEL SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("MEAN ACCURACY BY MODEL")
print("=" * 110)

model_accuracy = (
    final_predictions
    .groupby(
        "model"
    )["correct"]
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)

print(
    model_accuracy
)


# ============================================================
# 30. ACCURACY BY MODEL × TRANSFORMATION
# ============================================================

accuracy_table = (
    final_predictions
    .groupby(
        [
            "model",
            "transformation"
        ]
    )["correct"]
    .mean()
    .mul(100)
    .reset_index()
)

print("\n")
print("=" * 110)
print("ACCURACY BY MODEL × TRANSFORMATION")
print("=" * 110)

display(
    accuracy_table
)


# ============================================================
# 31. FINAL CHECK
# ============================================================

print("\n")
print("=" * 110)
print("✅ COMPLETE")
print("=" * 110)

print(
    "Models             : 4"
)

print(
    "Seeds              : 3"
)

print(
    "Transformations    : 16"
)

print(
    "Test leaves        : 128"
)

print(
    "Expected rows      : 24,576"
)

print(
    "Actual rows        :",
    len(final_predictions)
)

print(
    "\nPrediction file:"
)

print(
    OUTPUT_PATH
)

print(
    "\nReady for:"
)

print(
    "Exact McNemar + Benjamini-Hochberg FDR"
)

print("=" * 110)

# %% [Cell 72]
# ============================================================
# MODEL-vs-MODEL ROBUSTNESS STATISTICAL ANALYSIS
# FINAL VERSION
#
# INPUT:
#   4 models × 3 seeds × 16 transformations × 128 leaves
#   = 24,576 locked-test predictions
#
# TEST:
#   Exact two-sided McNemar test
#
# MULTIPLE-COMPARISON CORRECTION:
#   Benjamini-Hochberg FDR
#
# TOTAL TESTS:
#   6 model pairs × 3 seeds × 16 transformations
#   = 288
#
# NO TRAINING
# NO INFERENCE
# LOCKED TEST PREDICTIONS ONLY
# ============================================================


import os
import itertools
import numpy as np
import pandas as pd

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. PATHS
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

INPUT_PATH = os.path.join(
    ROOT,
    "model_level_locked_test",
    "model_level_locked_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "model_level_locked_test",
    "statistical_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. CHECK INPUT
# ============================================================

assert os.path.exists(
    INPUT_PATH
), (
    "\nPrediction file not found:\n"
    f"{INPUT_PATH}"
)


print("=" * 110)
print("MODEL-vs-MODEL ROBUSTNESS STATISTICAL ANALYSIS")
print("=" * 110)

print("\nInput file:")
print(INPUT_PATH)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_PATH
)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "model",
    "seed",
    "transformation",
    "original_image_id",
    "filename",
    "y_true",
    "y_pred",
    "correct"
]

missing_columns = [
    col
    for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        f"{missing_columns}\n\n"
        "Available columns:\n"
        f"{df.columns.tolist()}"
    )


# ============================================================
# 5. EXPECTED EXPERIMENTAL DESIGN
# ============================================================

MODELS = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18"
]

SEEDS = [
    42,
    1337,
    2026
]

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


# ============================================================
# 6. BASIC DATASET VERIFICATION
# ============================================================

print("\n")
print("=" * 110)
print("BASIC DATASET VERIFICATION")
print("=" * 110)

print("\nExpected rows:")
print(4 * 3 * 16 * 128)

print("\nActual rows:")
print(len(df))

assert len(df) == 24576, (
    f"\nExpected 24,576 rows, "
    f"got {len(df)}"
)

print("\nModels found:")
print(
    sorted(
        df["model"].unique()
    )
)

print("\nSeeds found:")
print(
    sorted(
        df["seed"].unique()
    )
)

print("\nTransformations found:")
print(
    sorted(
        df["transformation"].unique()
    )
)


# Exact vocabulary checks
assert set(
    df["model"].unique()
) == set(MODELS), (
    "\nModel vocabulary mismatch."
)

assert set(
    df["seed"].unique()
) == set(SEEDS), (
    "\nSeed vocabulary mismatch."
)

assert set(
    df["transformation"].unique()
) == set(TRANSFORMATIONS), (
    "\nTransformation vocabulary mismatch."
)


# ============================================================
# 7. VERIFY EVERY MODEL × SEED × TRANSFORMATION HAS 128
# ============================================================

print("\n")
print("=" * 110)
print("GROUP-SIZE VERIFICATION")
print("=" * 110)

group_counts = (
    df
    .groupby(
        [
            "model",
            "seed",
            "transformation"
        ]
    )
    .size()
    .reset_index(
        name="n"
    )
)

print(
    group_counts.head()
)

assert len(group_counts) == (
    4 * 3 * 16
)

assert (
    group_counts["n"] == 128
).all(), (
    "\nAt least one model × seed × transformation "
    "group does not contain exactly 128 leaves."
)

print(
    "\n✅ All 192 groups contain exactly 128 leaves."
)


# ============================================================
# 8. MODEL PAIRS
# ============================================================

MODEL_PAIRS = list(
    itertools.combinations(
        MODELS,
        2
    )
)

print("\n")
print("=" * 110)
print("MODEL PAIRS")
print("=" * 110)

for i, (model_a, model_b) in enumerate(
    MODEL_PAIRS,
    start=1
):

    print(
        f"{i}. {model_a} vs {model_b}"
    )

print(
    f"\nNumber of model pairs: {len(MODEL_PAIRS)}"
)

EXPECTED_TESTS = (
    len(MODEL_PAIRS)
    * len(SEEDS)
    * len(TRANSFORMATIONS)
)

print(
    "Expected statistical tests:",
    EXPECTED_TESTS
)

assert EXPECTED_TESTS == 288


# ============================================================
# 9. EXACT McNEMAR
# ============================================================

results = []


for model_a, model_b in MODEL_PAIRS:

    for seed in SEEDS:

        for transformation in TRANSFORMATIONS:

            # ------------------------------------------------
            # Select model A
            # ------------------------------------------------

            a_df = df[
                (df["model"] == model_a)
                &
                (df["seed"] == seed)
                &
                (
                    df["transformation"]
                    == transformation
                )
            ].copy()

            # ------------------------------------------------
            # Select model B
            # ------------------------------------------------

            b_df = df[
                (df["model"] == model_b)
                &
                (df["seed"] == seed)
                &
                (
                    df["transformation"]
                    == transformation
                )
            ].copy()

            assert len(a_df) == 128
            assert len(b_df) == 128

            # ------------------------------------------------
            # Pair the SAME leaves
            # ------------------------------------------------

            merged = a_df.merge(
                b_df,
                on="original_image_id",
                suffixes=(
                    "_a",
                    "_b"
                ),
                how="inner"
            )

            assert len(merged) == 128, (
                "\nPairing failure:\n"
                f"{model_a} vs {model_b}\n"
                f"seed={seed}\n"
                f"transformation={transformation}\n"
                f"pairs={len(merged)}"
            )

            # ------------------------------------------------
            # Verify same ground truth
            # ------------------------------------------------

            y_true_a = (
                merged["y_true_a"]
                .to_numpy()
            )

            y_true_b = (
                merged["y_true_b"]
                .to_numpy()
            )

            assert np.array_equal(
                y_true_a,
                y_true_b
            ), (
                "\nGround-truth labels differ "
                "between the two models.\n"
                f"{model_a} vs {model_b}\n"
                f"seed={seed}\n"
                f"transformation={transformation}"
            )

            # ------------------------------------------------
            # Correctness
            # ------------------------------------------------

            correct_a = (
                merged["y_pred_a"].to_numpy()
                ==
                y_true_a
            )

            correct_b = (
                merged["y_pred_b"].to_numpy()
                ==
                y_true_b
            )

            # ------------------------------------------------
            # McNemar discordant cells
            #
            # n01:
            # A wrong, B correct
            #
            # n10:
            # A correct, B wrong
            # ------------------------------------------------

            n10 = int(
                np.sum(
                    correct_a
                    &
                    (~correct_b)
                )
            )

            n01 = int(
                np.sum(
                    (~correct_a)
                    &
                    correct_b
                )
            )

            discordant = n10 + n01

            # ------------------------------------------------
            # Exact two-sided McNemar
            #
            # Conditional on discordant pairs:
            # Binomial(discordant, 0.5)
            # ------------------------------------------------

            if discordant == 0:

                p_value = 1.0

            else:

                p_value = binomtest(
                    k=min(
                        n10,
                        n01
                    ),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided"
                ).pvalue

            # ------------------------------------------------
            # Accuracy
            # ------------------------------------------------

            accuracy_a = (
                correct_a.mean()
            )

            accuracy_b = (
                correct_b.mean()
            )

            accuracy_delta_pp = (
                accuracy_a
                -
                accuracy_b
            ) * 100.0

            # ------------------------------------------------
            # Direction
            # ------------------------------------------------

            if n10 > n01:

                preferred = model_a

            elif n01 > n10:

                preferred = model_b

            else:

                preferred = "Tie"

            results.append({

                "model_a":
                    model_a,

                "model_b":
                    model_b,

                "seed":
                    seed,

                "transformation":
                    transformation,

                "n":
                    128,

                "model_a_correct_model_b_wrong":
                    n10,

                "model_a_wrong_model_b_correct":
                    n01,

                "discordant_pairs":
                    discordant,

                "model_a_accuracy":
                    accuracy_a,

                "model_b_accuracy":
                    accuracy_b,

                "accuracy_delta_pp_a_minus_b":
                    accuracy_delta_pp,

                "p_value":
                    p_value,

                "preferred_model":
                    preferred
            })


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

stats_df = pd.DataFrame(
    results
)

assert len(stats_df) == 288


# ============================================================
# 11. BENJAMINI-HOCHBERG FDR
#
# IMPORTANT:
# Correction is applied across ALL 288 tests.
# ============================================================

reject, q_values, _, _ = multipletests(
    stats_df["p_value"].to_numpy(),
    alpha=0.05,
    method="fdr_bh"
)

stats_df["fdr_q"] = q_values

stats_df["significant_after_fdr"] = (
    reject.astype(bool)
)


# ============================================================
# 12. SORT
# ============================================================

stats_df = (
    stats_df
    .sort_values(
        [
            "model_a",
            "model_b",
            "transformation",
            "seed"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 13. SUMMARY:
# MODEL PAIR × TRANSFORMATION
# ============================================================

pair_transformation_rows = []


for model_a, model_b in MODEL_PAIRS:

    for transformation in TRANSFORMATIONS:

        subset = stats_df[
            (stats_df["model_a"] == model_a)
            &
            (stats_df["model_b"] == model_b)
            &
            (
                stats_df["transformation"]
                == transformation
            )
        ].copy()

        assert len(subset) == 3

        a_wins_total = subset[
            "model_a_correct_model_b_wrong"
        ].sum()

        b_wins_total = subset[
            "model_a_wrong_model_b_correct"
        ].sum()

        if a_wins_total > b_wins_total:

            overall_preferred = model_a

        elif b_wins_total > a_wins_total:

            overall_preferred = model_b

        else:

            overall_preferred = "Tie"

        pair_transformation_rows.append({

            "model_a":
                model_a,

            "model_b":
                model_b,

            "transformation":
                transformation,

            "n_seeds":
                3,

            "significant_seeds":
                int(
                    subset[
                        "significant_after_fdr"
                    ].sum()
                ),

            "mean_fdr_q":
                subset[
                    "fdr_q"
                ].mean(),

            "min_fdr_q":
                subset[
                    "fdr_q"
                ].min(),

            "max_fdr_q":
                subset[
                    "fdr_q"
                ].max(),

            "model_a_wins_mean":
                subset[
                    "model_a_correct_model_b_wrong"
                ].mean(),

            "model_b_wins_mean":
                subset[
                    "model_a_wrong_model_b_correct"
                ].mean(),

            "mean_discordant_pairs":
                subset[
                    "discordant_pairs"
                ].mean(),

            "mean_accuracy_delta_pp":
                subset[
                    "accuracy_delta_pp_a_minus_b"
                ].mean(),

            "overall_preferred":
                overall_preferred
        })


pair_transformation_summary = pd.DataFrame(
    pair_transformation_rows
)


# ============================================================
# 14. SUMMARY:
# MODEL PAIR
# ============================================================

pair_summary_rows = []


for model_a, model_b in MODEL_PAIRS:

    subset = stats_df[
        (stats_df["model_a"] == model_a)
        &
        (stats_df["model_b"] == model_b)
    ].copy()

    a_wins_total = subset[
        "model_a_correct_model_b_wrong"
    ].sum()

    b_wins_total = subset[
        "model_a_wrong_model_b_correct"
    ].sum()

    if a_wins_total > b_wins_total:

        overall_preferred = model_a

    elif b_wins_total > a_wins_total:

        overall_preferred = model_b

    else:

        overall_preferred = "Tie"

    pair_summary_rows.append({

        "model_a":
            model_a,

        "model_b":
            model_b,

        "total_tests":
            len(subset),

        "fdr_significant_tests":
            int(
                subset[
                    "significant_after_fdr"
                ].sum()
            ),

        "percentage_fdr_significant":
            subset[
                "significant_after_fdr"
            ].mean()
            * 100,

        "mean_fdr_q":
            subset[
                "fdr_q"
            ].mean(),

        "min_fdr_q":
            subset[
                "fdr_q"
            ].min(),

        "total_model_a_wins":
            int(a_wins_total),

        "total_model_b_wins":
            int(b_wins_total),

        "mean_discordant_pairs":
            subset[
                "discordant_pairs"
            ].mean(),

        "mean_accuracy_delta_pp_a_minus_b":
            subset[
                "accuracy_delta_pp_a_minus_b"
            ].mean(),

        "overall_preferred":
            overall_preferred
    })


pair_summary = pd.DataFrame(
    pair_summary_rows
)


# ============================================================
# 15. SUMMARY:
# TRANSFORMATION
# ============================================================

transformation_summary_rows = []


for transformation in TRANSFORMATIONS:

    subset = stats_df[
        stats_df[
            "transformation"
        ]
        ==
        transformation
    ].copy()

    transformation_summary_rows.append({

        "transformation":
            transformation,

        "total_tests":
            len(subset),

        "fdr_significant_tests":
            int(
                subset[
                    "significant_after_fdr"
                ].sum()
            ),

        "percentage_fdr_significant":
            subset[
                "significant_after_fdr"
            ].mean()
            * 100,

        "mean_fdr_q":
            subset[
                "fdr_q"
            ].mean(),

        "min_fdr_q":
            subset[
                "fdr_q"
            ].min(),

        "mean_discordant_pairs":
            subset[
                "discordant_pairs"
            ].mean(),

        "mean_abs_accuracy_delta_pp":
            subset[
                "accuracy_delta_pp_a_minus_b"
            ].abs().mean()
    })


transformation_summary = pd.DataFrame(
    transformation_summary_rows
)


# ============================================================
# 16. SIGNIFICANT TESTS ONLY
# ============================================================

significant_results = stats_df[
    stats_df[
        "significant_after_fdr"
    ]
].copy()


# ============================================================
# 17. GLOBAL SUMMARY
# ============================================================

global_summary = pd.DataFrame({

    "metric": [

        "Number of models",

        "Number of model pairs",

        "Number of seeds",

        "Number of transformations",

        "Number of test leaves",

        "Total McNemar tests",

        "FDR-significant tests",

        "Percentage FDR-significant",

        "Statistical test",

        "Multiple-comparison correction",

        "Alpha"
    ],

    "value": [

        4,

        6,

        3,

        16,

        128,

        288,

        int(
            stats_df[
                "significant_after_fdr"
            ].sum()
        ),

        stats_df[
            "significant_after_fdr"
        ].mean() * 100,

        "Exact two-sided McNemar",

        "Benjamini-Hochberg FDR",

        0.05
    ]
})


# ============================================================
# 18. SAVE OUTPUTS
# ============================================================

ALL_TESTS_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_all_288_tests.csv"
)

PAIR_TRANSFORMATION_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_pair_transformation_summary.csv"
)

PAIR_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_pair_summary.csv"
)

TRANSFORMATION_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_transformation_summary.csv"
)

SIGNIFICANT_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_fdr_significant_only.csv"
)

GLOBAL_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "model_vs_model_mcnemar_global_summary.csv"
)


stats_df.to_csv(
    ALL_TESTS_PATH,
    index=False
)

pair_transformation_summary.to_csv(
    PAIR_TRANSFORMATION_PATH,
    index=False
)

pair_summary.to_csv(
    PAIR_SUMMARY_PATH,
    index=False
)

transformation_summary.to_csv(
    TRANSFORMATION_SUMMARY_PATH,
    index=False
)

significant_results.to_csv(
    SIGNIFICANT_PATH,
    index=False
)

global_summary.to_csv(
    GLOBAL_SUMMARY_PATH,
    index=False
)


# ============================================================
# 19. PRINT GLOBAL SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("GLOBAL SUMMARY")
print("=" * 110)

display(
    global_summary
)


# ============================================================
# 20. PRINT MODEL-PAIR SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("MODEL-PAIR SUMMARY")
print("=" * 110)

display(
    pair_summary
)


# ============================================================
# 21. PRINT TRANSFORMATION SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("TRANSFORMATION SUMMARY")
print("=" * 110)

display(
    transformation_summary
)


# ============================================================
# 22. PRINT MODEL PAIR × TRANSFORMATION SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("MODEL PAIR × TRANSFORMATION SUMMARY")
print("=" * 110)

display(
    pair_transformation_summary
)


# ============================================================
# 23. PRINT SIGNIFICANT RESULTS
# ============================================================

print("\n")
print("=" * 110)
print("FDR-SIGNIFICANT COMPARISONS")
print("=" * 110)

if len(significant_results) == 0:

    print(
        "No model-vs-model comparison remained "
        "significant after BH-FDR correction."
    )

else:

    display(
        significant_results[
            [
                "model_a",
                "model_b",
                "seed",
                "transformation",
                "model_a_correct_model_b_wrong",
                "model_a_wrong_model_b_correct",
                "discordant_pairs",
                "model_a_accuracy",
                "model_b_accuracy",
                "accuracy_delta_pp_a_minus_b",
                "p_value",
                "fdr_q",
                "preferred_model"
            ]
        ]
        .sort_values(
            "fdr_q"
        )
    )


# ============================================================
# 24. FINAL VALIDATION
# ============================================================

assert len(stats_df) == 288

assert (
    stats_df[
        "significant_after_fdr"
    ].dtype
    ==
    bool
)

assert (
    stats_df["fdr_q"] >=
    stats_df["p_value"]
).all()

# Every model pair:
# 3 seeds × 16 transformations = 48 tests

pair_test_counts = (
    stats_df
    .groupby(
        [
            "model_a",
            "model_b"
        ]
    )
    .size()
)

assert (
    pair_test_counts == 48
).all()


# ============================================================
# 25. FINAL OUTPUT PATHS
# ============================================================

print("\n")
print("=" * 110)
print("SAVED OUTPUT FILES")
print("=" * 110)

print("\n1. All 288 tests:")
print(ALL_TESTS_PATH)

print("\n2. Pair × transformation summary:")
print(PAIR_TRANSFORMATION_PATH)

print("\n3. Model-pair summary:")
print(PAIR_SUMMARY_PATH)

print("\n4. Transformation summary:")
print(TRANSFORMATION_SUMMARY_PATH)

print("\n5. FDR-significant tests only:")
print(SIGNIFICANT_PATH)

print("\n6. Global summary:")
print(GLOBAL_SUMMARY_PATH)


# ============================================================
# 26. COMPLETION
# ============================================================

print("\n")
print("=" * 110)
print("✅ MODEL-vs-MODEL STATISTICAL ANALYSIS COMPLETE")
print("=" * 110)

print("Models            :", 4)
print("Model pairs       :", 6)
print("Seeds             :", 3)
print("Transformations   :", 16)
print("Test leaves       :", 128)
print("Total tests       :", 288)

print(
    "FDR-significant   :",
    int(
        stats_df[
            "significant_after_fdr"
        ].sum()
    )
)

print("\n✅ Exact two-sided McNemar")
print("✅ Benjamini-Hochberg FDR")
print("✅ Same 128 leaves paired")
print("✅ No training")
print("✅ No inference")
print("=" * 110)

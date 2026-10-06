
# %% [Cell 73]
# ============================================================
# UNSEEN ANGLE / INTENSITY ROBUSTNESS TEST
# FINAL REVIEWER-FACING EXPERIMENT
#
# PURPOSE:
#   Evaluate model robustness to transformation parameters
#   that were NOT used in the released transformation set.
#
# DESIGN:
#   4 CNN models
#   × 3 seeds
#   × 2 unseen rotation angles
#   × 2 unseen brightness factors
#   × 128 original test leaves
#
# TOTAL:
#   4 × 3 × 4 × 128 = 6,144 predictions
#
# ADDITIONAL:
#   Exact McNemar:
#       Original vs each unseen condition
#
#   4 models × 3 seeds × 4 unseen conditions
#   = 48 paired tests
#
#   Benjamini-Hochberg FDR across all 48 tests.
#
# IMPORTANT:
#   No training.
#   No checkpoint selection.
#   Locked test inference only.
# ============================================================


import os
import gc
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

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

NUM_CLASSES = 3

BATCH_SIZE = 64
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "unseen_parameter_robustness"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "unseen_parameter_locked_test_predictions.csv"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "unseen_parameter_metrics_by_seed.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "unseen_parameter_target_summary.csv"
)

MCNEMAR_ALL_PATH = os.path.join(
    OUTPUT_DIR,
    "unseen_parameter_mcnemar_all_48_tests.csv"
)

MCNEMAR_SIG_PATH = os.path.join(
    OUTPUT_DIR,
    "unseen_parameter_mcnemar_fdr_significant_only.csv"
)

MCNEMAR_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "unseen_parameter_mcnemar_summary.csv"
)

PARAMETER_PATH = os.path.join(
    OUTPUT_DIR,
    "recovered_original_parameters_and_unseen_parameters.csv"
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 110)
print("UNSEEN ANGLE / INTENSITY ROBUSTNESS TEST")
print("=" * 110)

print(
    "PyTorch     :",
    torch.__version__
)

print(
    "Device      :",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU         :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "\nMetadata shape:",
    metadata.shape
)

required_columns = [
    "filename",
    "class_label",
    "original_image_id",
    "preprocessing_technique",
    "data_split"
]

for col in required_columns:

    assert col in metadata.columns, (
        f"Missing metadata column: {col}"
    )


# ============================================================
# 5. NORMALIZE
# ============================================================

for col in required_columns:

    metadata[col] = (
        metadata[col]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 6. CLASS MAPPING
# ============================================================

CLASS_TO_INDEX = {

    "Healthy_Leaves": 0,

    "Little_Leaf": 1,

    "Phomopsis_Blight": 2
}


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

            relative_path = os.path.relpath(
                full_path,
                ROOT
            ).replace(
                "\\",
                "/"
            )

            image_paths[
                relative_path
            ] = full_path

            basename = os.path.basename(
                full_path
            )

            if basename not in image_paths:

                image_paths[
                    basename
                ] = full_path


def resolve_path(
    filename
):

    filename = (
        str(filename)
        .strip()
        .replace("\\", "/")
    )

    filename = filename.lstrip("./")

    if filename in image_paths:

        return image_paths[
            filename
        ]

    direct = os.path.join(
        ROOT,
        filename
    )

    if os.path.isfile(
        direct
    ):

        return os.path.abspath(
            direct
        )

    basename = os.path.basename(
        filename
    )

    if basename in image_paths:

        return image_paths[
            basename
        ]

    return None


print(
    "Indexed image paths:",
    len(image_paths)
)


# ============================================================
# 8. ORIGINAL TEST LEAVES ONLY
# ============================================================

test_original = metadata[
    (
        metadata["data_split"]
        .str.lower()
        == "test"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        ==
        "Original"
    )
].copy()


print("\n")
print("=" * 110)
print("ORIGINAL TEST LEAF SET")
print("=" * 110)

print(
    "Rows:",
    len(test_original)
)

print(
    "Unique leaves:",
    test_original[
        "original_image_id"
    ].nunique()
)

assert len(test_original) == 128
assert (
    test_original[
        "original_image_id"
    ].nunique()
    == 128
)


# ============================================================
# 9. GET ORIGINAL / ROTATE / BRIGHTNESS PAIRS
# ============================================================

test_all = metadata[
    metadata["data_split"]
    .str.lower()
    ==
    "test"
].copy()


def get_variant(
    leaf_id,
    transformation
):

    rows = test_all[
        (
            test_all[
                "original_image_id"
            ]
            ==
            leaf_id
        )
        &
        (
            test_all[
                "preprocessing_technique"
            ]
            ==
            transformation
        )
    ]

    assert len(rows) == 1, (
        f"Expected exactly one row for "
        f"{leaf_id} / {transformation}, "
        f"got {len(rows)}"
    )

    filename = rows.iloc[0][
        "filename"
    ]

    return resolve_path(
        filename
    )


# ============================================================
# 10. IMAGE LOADER
# ============================================================

def read_rgb(
    path
):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# 11. RECOVER EXISTING ROTATION ANGLE
#
# We estimate the parameter used in the existing Rotate
# transformation from original/rotated pairs.
#
# This is NOT based on model performance.
# ============================================================

print("\n")
print("=" * 110)
print("RECOVERING EXISTING ROTATION PARAMETER")
print("=" * 110)


def rotation_similarity(
    original_gray,
    rotated_gray,
    angle
):

    h, w = original_gray.shape

    center = (
        w / 2.0,
        h / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    warped = cv2.warpAffine(
        original_gray,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Exclude black border regions
    mask = (
        (warped > 8)
        &
        (rotated_gray > 8)
    )

    if mask.sum() < 100:

        return -1e9

    a = warped[
        mask
    ].astype(
        np.float32
    )

    b = rotated_gray[
        mask
    ].astype(
        np.float32
    )

    a -= a.mean()
    b -= b.mean()

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:

        return -1e9

    return float(
        np.dot(a, b)
        /
        denominator
    )


# Use a fixed subset for parameter recovery
sample_leaves = (
    test_original[
        "original_image_id"
    ]
    .drop_duplicates()
    .tolist()
)

sample_leaves = sample_leaves[
    : min(
        12,
        len(sample_leaves)
    )
]


estimated_angles = []


for leaf_id in sample_leaves:

    original_path = resolve_path(
        test_original[
            test_original[
                "original_image_id"
            ]
            ==
            leaf_id
        ].iloc[0]["filename"]
    )

    rotate_path = get_variant(
        leaf_id,
        "Rotate"
    )

    original = cv2.imread(
        original_path,
        cv2.IMREAD_GRAYSCALE
    )

    rotated = cv2.imread(
        rotate_path,
        cv2.IMREAD_GRAYSCALE
    )

    original = cv2.resize(
        original,
        (160, 160)
    )

    rotated = cv2.resize(
        rotated,
        (160, 160)
    )

    best_angle = None
    best_score = -1e9

    # Coarse sweep
    for angle in np.arange(
        -90,
        90.1,
        2.0
    ):

        score = rotation_similarity(
            original,
            rotated,
            float(angle)
        )

        if score > best_score:

            best_score = score
            best_angle = float(
                angle
            )

    # Fine sweep around coarse winner
    fine_start = best_angle - 2.0
    fine_end = best_angle + 2.0

    for angle in np.arange(
        fine_start,
        fine_end + 0.001,
        0.25
    ):

        score = rotation_similarity(
            original,
            rotated,
            float(angle)
        )

        if score > best_score:

            best_score = score
            best_angle = float(
                angle
            )

    estimated_angles.append(
        best_angle
    )


estimated_original_rotate_angle = float(
    np.median(
        estimated_angles
    )
)

print(
    "Estimated existing Rotate angle:",
    estimated_original_rotate_angle
)

print(
    "Per-sample estimates:",
    [
        round(x, 2)
        for x in estimated_angles
    ]
)


# ============================================================
# 12. RECOVER EXISTING BRIGHTNESS PARAMETER
#
# We estimate the multiplicative brightness factor.
# ============================================================

print("\n")
print("=" * 110)
print("RECOVERING EXISTING BRIGHTNESS PARAMETER")
print("=" * 110)


estimated_brightness_factors = []
estimated_brightness_intercepts = []


for leaf_id in sample_leaves:

    original_path = resolve_path(
        test_original[
            test_original[
                "original_image_id"
            ]
            ==
            leaf_id
        ].iloc[0]["filename"]
    )

    brightness_path = get_variant(
        leaf_id,
        "Brightness"
    )

    original = read_rgb(
        original_path
    )

    brightness = read_rgb(
        brightness_path
    )

    original = cv2.resize(
        original,
        (160, 160)
    )

    brightness = cv2.resize(
        brightness,
        (160, 160)
    )

    x = (
        cv2.cvtColor(
            original,
            cv2.COLOR_RGB2GRAY
        )
        .astype(
            np.float32
        )
        .reshape(-1)
    )

    y = (
        cv2.cvtColor(
            brightness,
            cv2.COLOR_RGB2GRAY
        )
        .astype(
            np.float32
        )
        .reshape(-1)
    )

    valid = (
        (x > 20)
        &
        (x < 235)
        &
        (y > 5)
        &
        (y < 250)
    )

    x_valid = x[
        valid
    ]

    y_valid = y[
        valid
    ]

    if len(x_valid) < 100:

        continue

    slope, intercept = np.polyfit(
        x_valid,
        y_valid,
        1
    )

    estimated_brightness_factors.append(
        float(slope)
    )

    estimated_brightness_intercepts.append(
        float(intercept)
    )


existing_brightness_factor = float(
    np.median(
        estimated_brightness_factors
    )
)

existing_brightness_intercept = float(
    np.median(
        estimated_brightness_intercepts
    )
)

print(
    "Estimated existing brightness factor:",
    existing_brightness_factor
)

print(
    "Estimated brightness intercept:",
    existing_brightness_intercept
)

print(
    "Per-sample factors:",
    [
        round(x, 3)
        for x in estimated_brightness_factors
    ]
)


# ============================================================
# 13. SELECT UNSEEN PARAMETERS
#
# We explicitly avoid parameters close to the recovered
# dataset parameters.
# ============================================================

candidate_angles = [
    -60,
    -45,
    -30,
    -20,
    -15,
    15,
    20,
    30,
    45,
    60
]

candidate_brightness = [
    0.55,
    0.60,
    0.70,
    0.80,
    1.20,
    1.30,
    1.40,
    1.45
]


# Minimum distance from existing dataset parameter
ANGLE_MIN_DISTANCE = 10.0
BRIGHTNESS_MIN_DISTANCE = 0.10


valid_angles = [

    angle

    for angle in candidate_angles

    if abs(
        angle
        -
        estimated_original_rotate_angle
    )
    >=
    ANGLE_MIN_DISTANCE
]


valid_brightness = [

    factor

    for factor in candidate_brightness

    if abs(
        factor
        -
        existing_brightness_factor
    )
    >=
    BRIGHTNESS_MIN_DISTANCE
]


# Prefer one negative and one positive angle
negative_angles = [
    x for x in valid_angles
    if x < 0
]

positive_angles = [
    x for x in valid_angles
    if x > 0
]


if negative_angles:

    unseen_angle_1 = max(
        negative_angles,
        key=lambda x: -abs(x)
    )

else:

    unseen_angle_1 = valid_angles[0]


if positive_angles:

    unseen_angle_2 = min(
        positive_angles,
        key=lambda x: abs(x)
    )

else:

    unseen_angle_2 = valid_angles[1]


# Choose separated brightness factors:
# one darkening and one brightening
darkening_factors = [
    x for x in valid_brightness
    if x < 1.0
]

brightening_factors = [
    x for x in valid_brightness
    if x > 1.0
]


if not darkening_factors:

    raise RuntimeError(
        "Could not find an unseen darkening factor."
    )

if not brightening_factors:

    raise RuntimeError(
        "Could not find an unseen brightening factor."
    )


unseen_brightness_1 = max(
    darkening_factors
)

unseen_brightness_2 = min(
    brightening_factors
)


UNSEEN_CONDITIONS = OrderedDict({

    f"Rotate_Unseen_{unseen_angle_1:+g}deg":
        {
            "type": "rotate",
            "value": unseen_angle_1
        },

    f"Rotate_Unseen_{unseen_angle_2:+g}deg":
        {
            "type": "rotate",
            "value": unseen_angle_2
        },

    f"Brightness_Unseen_{unseen_brightness_1:.2f}x":
        {
            "type": "brightness",
            "value": unseen_brightness_1
        },

    f"Brightness_Unseen_{unseen_brightness_2:.2f}x":
        {
            "type": "brightness",
            "value": unseen_brightness_2
        }
})


print("\n")
print("=" * 110)
print("UNSEEN PARAMETERS SELECTED")
print("=" * 110)

print(
    "\nExisting Rotate:",
    estimated_original_rotate_angle,
    "degrees"
)

print(
    "Existing Brightness:",
    round(
        existing_brightness_factor,
        4
    ),
    "x"
)

print("\nUnseen conditions:")

for name, info in (
    UNSEEN_CONDITIONS.items()
):

    print(
        f"{name:40s}"
        f" -> {info}"
    )


# ============================================================
# 14. SAVE PARAMETER AUDIT
# ============================================================

parameter_rows = [

    {
        "parameter_type":
            "existing_rotate",

        "estimated_value":
            estimated_original_rotate_angle,

        "unit":
            "degrees",

        "source":
            "recovered_from_dataset_pairs"
    },

    {
        "parameter_type":
            "existing_brightness",

        "estimated_value":
            existing_brightness_factor,

        "unit":
            "multiplicative_factor",

        "source":
            "recovered_from_dataset_pairs"
    }
]


for name, info in (
    UNSEEN_CONDITIONS.items()
):

    parameter_rows.append({

        "parameter_type":
            name,

        "estimated_value":
            info["value"],

        "unit":
            (
                "degrees"
                if info["type"] == "rotate"
                else
                "multiplicative_factor"
            ),

        "source":
            "new_test_time_parameter"
    })


pd.DataFrame(
    parameter_rows
).to_csv(
    PARAMETER_PATH,
    index=False
)


# ============================================================
# 15. IMAGE TRANSFORMATION FUNCTIONS
# ============================================================

def apply_unseen_rotate(
    image_rgb,
    angle
):

    h, w = image_rgb.shape[:2]

    center = (
        w / 2.0,
        h / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0
        )
    )

    return rotated


def apply_unseen_brightness(
    image_rgb,
    factor
):

    image_float = (
        image_rgb
        .astype(
            np.float32
        )
        *
        factor
    )

    image_float = np.clip(
        image_float,
        0,
        255
    )

    return image_float.astype(
        np.uint8
    )


def apply_condition(
    image_rgb,
    condition_type,
    value
):

    if condition_type == "rotate":

        return apply_unseen_rotate(
            image_rgb,
            value
        )

    elif condition_type == "brightness":

        return apply_unseen_brightness(
            image_rgb,
            value
        )

    raise ValueError(
        f"Unknown condition type: "
        f"{condition_type}"
    )


# ============================================================
# 16. IMAGE PREPROCESSING
# ============================================================

base_transform = transforms.Compose([

    transforms.ToPILImage(),

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
# 17. DATASET
# ============================================================

class UnseenTestDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        condition_type,
        condition_value
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.condition_type = (
            condition_type
        )

        self.condition_value = (
            condition_value
        )

    def __len__(self):

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

        filename = (
            row["filename"]
        )

        path = resolve_path(
            filename
        )

        if path is None:

            raise FileNotFoundError(
                f"Image not found:\n"
                f"{filename}"
            )

        image_rgb = read_rgb(
            path
        )

        transformed = apply_condition(
            image_rgb,
            self.condition_type,
            self.condition_value
        )

        image_tensor = (
            base_transform(
                transformed
            )
        )

        y_true = int(
            row["y_true"]
        )

        return (
            image_tensor,
            y_true,
            index
        )


# ============================================================
# 18. MODEL BUILDERS
# ============================================================

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

        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights"
        ]:

            if key in checkpoint:

                candidate = (
                    checkpoint[key]
                )

                if isinstance(
                    candidate,
                    (
                        dict,
                        OrderedDict
                    )
                ):

                    return candidate

    raise ValueError(
        "Unable to extract checkpoint state_dict."
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
# 20. LOAD / CHECKPOINT VALIDATION
# ============================================================

for model_name in MODELS:

    for seed in SEEDS:

        path = CHECKPOINTS[
            model_name
        ][seed]

        assert os.path.exists(
            path
        ), (
            f"Missing checkpoint:\n{path}"
        )


# ============================================================
# 21. ORIGINAL TEST DATAFRAME
# ============================================================

test_original["y_true"] = (
    test_original[
        "class_label"
    ].map(
        CLASS_TO_INDEX
    ).astype(int)
)

test_original = (
    test_original
    .sort_values(
        "original_image_id"
    )
    .reset_index(
        drop=True
    )
)

assert len(test_original) == 128


# ============================================================
# 22. INFERENCE FUNCTION
# ============================================================

@torch.inference_mode()
def run_model_inference(
    model,
    dataframe,
    condition_type,
    condition_value
):

    dataset = UnseenTestDataset(
        dataframe=dataframe,
        condition_type=condition_type,
        condition_value=condition_value
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for images, y_true, indices in loader:

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

        all_true.extend(
            y_true.numpy().tolist()
        )

        all_pred.extend(
            preds
            .cpu()
            .numpy()
            .tolist()
        )

        all_prob.append(
            probs
            .cpu()
            .numpy()
        )

    all_prob = np.concatenate(
        all_prob,
        axis=0
    )

    return (
        np.asarray(
            all_true,
            dtype=int
        ),

        np.asarray(
            all_pred,
            dtype=int
        ),

        all_prob
    )


# ============================================================
# 23. RUN ALL UNSEEN CONDITIONS
#
# 4 models × 3 seeds × 4 conditions
# = 48 inference groups
# ============================================================

all_prediction_rows = []
all_metric_rows = []

total_runs = (
    len(MODELS)
    * len(SEEDS)
    * len(
        UNSEEN_CONDITIONS
    )
)

counter = 0


for model_name in MODELS:

    for seed in SEEDS:

        print("\n")
        print("=" * 110)
        print(
            f"MODEL: {model_name} | "
            f"SEED: {seed}"
        )
        print("=" * 110)

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        model = build_model(
            model_name
        )

        checkpoint = torch.load(
            CHECKPOINTS[
                model_name
            ][seed],
            map_location="cpu",
            weights_only=False
        )

        state_dict = clean_state_dict(
            extract_state_dict(
                checkpoint
            )
        )

        load_result = (
            model.load_state_dict(
                state_dict,
                strict=False
            )
        )

        if load_result.missing_keys:

            raise RuntimeError(
                f"Missing checkpoint keys:\n"
                f"{load_result.missing_keys}"
            )

        model = model.to(
            DEVICE
        )

        model.eval()

        # ----------------------------------------------------
        # Every unseen condition
        # ----------------------------------------------------

        for condition_name, condition_info in (
            UNSEEN_CONDITIONS.items()
        ):

            counter += 1

            print(
                f"[{counter}/{total_runs}] "
                f"{condition_name}"
            )

            condition_type = (
                condition_info["type"]
            )

            condition_value = (
                condition_info["value"]
            )

            y_true, y_pred, probs = (
                run_model_inference(
                    model=model,
                    dataframe=test_original,
                    condition_type=condition_type,
                    condition_value=condition_value
                )
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            macro_f1 = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            )

            all_metric_rows.append({

                "model":
                    model_name,

                "seed":
                    seed,

                "condition":
                    condition_name,

                "condition_type":
                    condition_type,

                "parameter":
                    condition_value,

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,

                "balanced_accuracy":
                    balanced_accuracy
            })

            # ------------------------------------------------
            # Prediction rows
            # ------------------------------------------------

            for i in range(
                len(test_original)
            ):

                row = test_original.iloc[
                    i
                ]

                all_prediction_rows.append({

                    "model":
                        model_name,

                    "seed":
                        seed,

                    "condition":
                        condition_name,

                    "condition_type":
                        condition_type,

                    "parameter":
                        condition_value,

                    "original_image_id":
                        row[
                            "original_image_id"
                        ],

                    "filename":
                        row[
                            "filename"
                        ],

                    "y_true":
                        int(
                            y_true[i]
                        ),

                    "y_pred":
                        int(
                            y_pred[i]
                        ),

                    "correct":
                        bool(
                            y_true[i]
                            ==
                            y_pred[i]
                        ),

                    "prob_healthy":
                        float(
                            probs[i, 0]
                        ),

                    "prob_little_leaf":
                        float(
                            probs[i, 1]
                        ),

                    "prob_phomopsis":
                        float(
                            probs[i, 2]
                        )
                })

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del model
        del checkpoint
        del state_dict

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 24. DATAFRAMES
# ============================================================

prediction_df = pd.DataFrame(
    all_prediction_rows
)

metrics_df = pd.DataFrame(
    all_metric_rows
)

print("\n")
print("=" * 110)
print("FINAL UNSEEN DATASET")
print("=" * 110)

print(
    "Expected predictions:",
    4 * 3 * 4 * 128
)

print(
    "Actual predictions:",
    len(prediction_df)
)

assert len(
    prediction_df
) == 6144

assert len(
    metrics_df
) == 48


# ============================================================
# 25. SAVE RAW RESULTS
# ============================================================

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# 26. SUMMARY ACROSS SEEDS
# ============================================================

summary_df = (
    metrics_df
    .groupby(
        [
            "model",
            "condition",
            "condition_type",
            "parameter"
        ]
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
    .reset_index()
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# 27. PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("UNSEEN PARAMETER PERFORMANCE")
print("=" * 110)

display(
    summary_df
    .sort_values(
        [
            "model",
            "condition"
        ]
    )
)


# ============================================================
# 28. EXACT McNEMAR:
# ORIGINAL vs EACH UNSEEN CONDITION
#
# 4 models × 3 seeds × 4 conditions = 48 tests
# ============================================================

print("\n")
print("=" * 110)
print("EXACT McNEMAR: ORIGINAL vs UNSEEN CONDITIONS")
print("=" * 110)


mcnemar_rows = []


for model_name in MODELS:

    for seed in SEEDS:

        # ----------------------------------------------------
        # Load original prediction from existing locked file
        # ----------------------------------------------------

        # Existing original baseline predictions
        # are stored in model_level_locked_test.
        # We load from the previously generated prediction file.
        original_predictions_path = os.path.join(
            ROOT,
            "model_level_locked_test",
            "model_level_locked_test_predictions.csv"
        )

        original_df = pd.read_csv(
            original_predictions_path
        )

        original_df = original_df[
            (
                original_df["model"]
                ==
                model_name
            )
            &
            (
                original_df["seed"]
                ==
                seed
            )
            &
            (
                original_df["transformation"]
                ==
                "Original"
            )
        ].copy()

        assert len(
            original_df
        ) == 128

        # ----------------------------------------------------
        # Unseen conditions
        # ----------------------------------------------------

        for condition_name in (
            UNSEEN_CONDITIONS.keys()
        ):

            unseen_df = prediction_df[
                (
                    prediction_df["model"]
                    ==
                    model_name
                )
                &
                (
                    prediction_df["seed"]
                    ==
                    seed
                )
                &
                (
                    prediction_df["condition"]
                    ==
                    condition_name
                )
            ].copy()

            assert len(
                unseen_df
            ) == 128

            # ------------------------------------------------
            # Pair same leaves
            # ------------------------------------------------

            merged = original_df.merge(
                unseen_df,
                on="original_image_id",
                suffixes=(
                    "_original",
                    "_unseen"
                ),
                how="inner"
            )

            assert len(
                merged
            ) == 128

            # ------------------------------------------------
            # Ground truth consistency
            # ------------------------------------------------

            assert np.array_equal(
                merged[
                    "y_true_original"
                ].to_numpy(),

                merged[
                    "y_true_unseen"
                ].to_numpy()
            )

            y_true = merged[
                "y_true_original"
            ].to_numpy()

            original_pred = merged[
                "y_pred_original"
            ].to_numpy()

            unseen_pred = merged[
                "y_pred_unseen"
            ].to_numpy()

            original_correct = (
                original_pred
                ==
                y_true
            )

            unseen_correct = (
                unseen_pred
                ==
                y_true
            )

            # Original correct / unseen wrong
            original_correct_unseen_wrong = int(
                np.sum(
                    original_correct
                    &
                    (~unseen_correct)
                )
            )

            # Original wrong / unseen correct
            original_wrong_unseen_correct = int(
                np.sum(
                    (~original_correct)
                    &
                    unseen_correct
                )
            )

            discordant = (
                original_correct_unseen_wrong
                +
                original_wrong_unseen_correct
            )

            if discordant == 0:

                p_value = 1.0

            else:

                p_value = binomtest(
                    k=min(
                        original_correct_unseen_wrong,
                        original_wrong_unseen_correct
                    ),

                    n=discordant,

                    p=0.5,

                    alternative="two-sided"
                ).pvalue

            original_acc = (
                original_correct.mean()
            )

            unseen_acc = (
                unseen_correct.mean()
            )

            delta_pp = (
                unseen_acc
                -
                original_acc
            ) * 100.0

            mcnemar_rows.append({

                "model":
                    model_name,

                "seed":
                    seed,

                "condition":
                    condition_name,

                "original_correct_unseen_wrong":
                    original_correct_unseen_wrong,

                "original_wrong_unseen_correct":
                    original_wrong_unseen_correct,

                "discordant_pairs":
                    discordant,

                "original_accuracy":
                    original_acc,

                "unseen_accuracy":
                    unseen_acc,

                "accuracy_delta_pp_unseen_minus_original":
                    delta_pp,

                "p_value":
                    p_value
            })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)


# ============================================================
# 29. BH-FDR ON ALL 48 TESTS
# ============================================================

assert len(
    mcnemar_df
) == 48


reject, q_values, _, _ = multipletests(
    mcnemar_df[
        "p_value"
    ].to_numpy(),

    alpha=0.05,

    method="fdr_bh"
)


mcnemar_df[
    "fdr_q"
] = q_values

mcnemar_df[
    "significant_after_fdr"
] = (
    reject.astype(bool)
)


mcnemar_df = (
    mcnemar_df
    .sort_values(
        "fdr_q"
    )
    .reset_index(
        drop=True
    )
)


mcnemar_df.to_csv(
    MCNEMAR_ALL_PATH,
    index=False
)


# ============================================================
# 30. SIGNIFICANT ONLY
# ============================================================

significant_mcnemar = (
    mcnemar_df[
        mcnemar_df[
            "significant_after_fdr"
        ]
    ]
    .copy()
)


significant_mcnemar.to_csv(
    MCNEMAR_SIG_PATH,
    index=False
)


# ============================================================
# 31. MCNEMAR SUMMARY BY CONDITION
# ============================================================

mcnemar_summary = (
    mcnemar_df
    .groupby(
        "condition"
    )
    .agg(

        total_tests=(
            "p_value",
            "size"
        ),

        fdr_significant_tests=(
            "significant_after_fdr",
            "sum"
        ),

        mean_fdr_q=(
            "fdr_q",
            "mean"
        ),

        min_fdr_q=(
            "fdr_q",
            "min"
        ),

        mean_accuracy_delta_pp=(
            "accuracy_delta_pp_unseen_minus_original",
            "mean"
        ),

        mean_discordant_pairs=(
            "discordant_pairs",
            "mean"
        )
    )
    .reset_index()
)


mcnemar_summary.to_csv(
    MCNEMAR_SUMMARY_PATH,
    index=False
)


# ============================================================
# 32. DISPLAY MCNEMAR
# ============================================================

print("\n")
print("=" * 110)
print("MCNEMAR SUMMARY")
print("=" * 110)

display(
    mcnemar_summary
)


print("\n")
print("=" * 110)
print("FDR-SIGNIFICANT ORIGINAL vs UNSEEN COMPARISONS")
print("=" * 110)

if len(
    significant_mcnemar
) == 0:

    print(
        "No comparison remained significant "
        "after BH-FDR."
    )

else:

    display(
        significant_mcnemar
    )


# ============================================================
# 33. FINAL VALIDATION
# ============================================================

assert (
    len(prediction_df)
    ==
    6144
)

assert (
    len(metrics_df)
    ==
    48
)

assert (
    len(mcnemar_df)
    ==
    48
)


# Every model × seed × condition has 128
group_counts = (
    prediction_df
    .groupby(
        [
            "model",
            "seed",
            "condition"
        ]
    )
    .size()
)

assert (
    group_counts == 128
).all()


# ============================================================
# 34. OUTPUT PATHS
# ============================================================

print("\n")
print("=" * 110)
print("SAVED OUTPUTS")
print("=" * 110)

print(
    "\nParameter recovery:"
)

print(
    PARAMETER_PATH
)

print(
    "\nPredictions:"
)

print(
    PREDICTIONS_PATH
)

print(
    "\nMetrics:"
)

print(
    METRICS_PATH
)

print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)

print(
    "\nMcNemar — all 48:"
)

print(
    MCNEMAR_ALL_PATH
)

print(
    "\nMcNemar — significant only:"
)

print(
    MCNEMAR_SIG_PATH
)

print(
    "\nMcNemar summary:"
)

print(
    MCNEMAR_SUMMARY_PATH
)


# ============================================================
# 35. FINAL
# ============================================================

print("\n")
print("=" * 110)
print("✅ UNSEEN PARAMETER ROBUSTNESS TEST COMPLETE")
print("=" * 110)

print(
    "Models             : 4"
)

print(
    "Seeds              : 3"
)

print(
    "Unseen conditions  : 4"
)

print(
    "Test leaves        : 128"
)

print(
    "Total predictions  :",
    len(prediction_df)
)

print(
    "McNemar tests      :",
    len(mcnemar_df)
)

print(
    "FDR-significant    :",
    int(
        mcnemar_df[
            "significant_after_fdr"
        ].sum()
    )
)

print("\n✅ No training")
print("✅ No checkpoint selection")
print("✅ New transformation parameters only")
print("✅ Exact McNemar")
print("✅ Benjamini-Hochberg FDR")
print("=" * 110)

# %% [Cell 74]
# ============================================================
# FINAL STRONG UNSEEN-PARAMETER ROBUSTNESS TEST
#
# PURPOSE:
#   Test genuinely unseen and clearly separated transformation
#   parameters on the locked original test leaves.
#
# CONDITIONS:
#   Rotate  -45°
#   Rotate  +45°
#   Brightness 0.60x
#   Brightness 1.60x
#
# DESIGN:
#   4 models × 3 seeds × 4 unseen conditions × 128 leaves
#   = 6,144 predictions
#
# STATISTICS:
#   Exact two-sided McNemar
#   BH-FDR across all 48 tests
#
# NO TRAINING
# NO MODEL SELECTION
# LOCKED TEST INFERENCE ONLY
# ============================================================

import os
import gc
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

ORIGINAL_PREDICTIONS_PATH = os.path.join(
    ROOT,
    "model_level_locked_test",
    "model_level_locked_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "unseen_parameter_robustness_STRONG"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Strong unseen parameters
# ------------------------------------------------------------

UNSEEN_CONDITIONS = OrderedDict({

    "Rotate_Unseen_-45deg": {
        "type": "rotate",
        "value": -45.0
    },

    "Rotate_Unseen_+45deg": {
        "type": "rotate",
        "value": 45.0
    },

    "Brightness_Unseen_0.60x": {
        "type": "brightness",
        "value": 0.60
    },

    "Brightness_Unseen_1.60x": {
        "type": "brightness",
        "value": 1.60
    }
})


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

NUM_CLASSES = 3

BATCH_SIZE = 64
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

PARAMETER_AUDIT_PATH = os.path.join(
    OUTPUT_DIR,
    "strong_unseen_parameter_audit.csv"
)

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "strong_unseen_locked_test_predictions.csv"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "strong_unseen_metrics_by_seed.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "strong_unseen_target_summary.csv"
)

MCNEMAR_ALL_PATH = os.path.join(
    OUTPUT_DIR,
    "strong_unseen_mcnemar_all_48_tests.csv"
)

MCNEMAR_SIG_PATH = os.path.join(
    OUTPUT_DIR,
    "strong_unseen_mcnemar_fdr_significant_only.csv"
)

MCNEMAR_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "strong_unseen_mcnemar_summary.csv"
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 110)
print("FINAL STRONG UNSEEN-PARAMETER ROBUSTNESS TEST")
print("=" * 110)

print(
    "PyTorch :",
    torch.__version__
)

print(
    "Device  :",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU     :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. LOAD METADATA
# ============================================================

assert os.path.exists(
    METADATA_PATH
), f"Metadata not found:\n{METADATA_PATH}"

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "\nMetadata shape:",
    metadata.shape
)


REQUIRED_COLUMNS = [
    "filename",
    "class_label",
    "original_image_id",
    "preprocessing_technique",
    "data_split"
]

for col in REQUIRED_COLUMNS:

    assert col in metadata.columns, (
        f"Missing metadata column: {col}"
    )


for col in REQUIRED_COLUMNS:

    metadata[col] = (
        metadata[col]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 5. CLASS MAPPING
# ============================================================

CLASS_TO_INDEX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}


# ============================================================
# 6. BUILD IMAGE INDEX
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

            relative_path = os.path.relpath(
                full_path,
                ROOT
            ).replace(
                "\\",
                "/"
            )

            image_paths[
                relative_path
            ] = full_path

            basename = os.path.basename(
                full_path
            )

            if basename not in image_paths:

                image_paths[
                    basename
                ] = full_path


def resolve_path(
    filename
):

    filename = (
        str(filename)
        .strip()
        .replace("\\", "/")
    )

    filename = filename.lstrip("./")

    if filename in image_paths:

        return image_paths[
            filename
        ]

    direct = os.path.join(
        ROOT,
        filename
    )

    if os.path.isfile(
        direct
    ):

        return os.path.abspath(
            direct
        )

    basename = os.path.basename(
        filename
    )

    if basename in image_paths:

        return image_paths[
            basename
        ]

    return None


print(
    "Indexed image keys:",
    len(image_paths)
)


# ============================================================
# 7. LOCKED ORIGINAL TEST LEAVES
# ============================================================

test_original = metadata[
    (
        metadata["data_split"]
        .str.lower()
        ==
        "test"
    )
    &
    (
        metadata["preprocessing_technique"]
        ==
        "Original"
    )
].copy()


assert len(
    test_original
) == 128

assert (
    test_original[
        "original_image_id"
    ].nunique()
    == 128
)


test_original["y_true"] = (
    test_original[
        "class_label"
    ]
    .map(
        CLASS_TO_INDEX
    )
    .astype(int)
)

test_original = (
    test_original
    .sort_values(
        "original_image_id"
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 110)
print("ORIGINAL TEST SET")
print("=" * 110)

print(
    "Original test leaves:",
    len(test_original)
)

print(
    "Unique leaves:",
    test_original[
        "original_image_id"
    ].nunique()
)


# ============================================================
# 8. VERIFY ALL ORIGINAL IMAGES EXIST
# ============================================================

for filename in test_original[
    "filename"
]:

    assert resolve_path(
        filename
    ) is not None, (
        f"Original test image not found:\n"
        f"{filename}"
    )


# ============================================================
# 9. PARAMETER AUDIT
# ============================================================

parameter_audit = pd.DataFrame([

    {
        "condition":
            "Dataset_Rotate_Existing",

        "parameter":
            25.0,

        "unit":
            "degrees",

        "status":
            "estimated previously from dataset pairs"
    },

    {
        "condition":
            "Dataset_Brightness_Existing",

        "parameter":
            1.2709,

        "unit":
            "multiplicative_factor",

        "status":
            "estimated previously from dataset pairs"
    },

    {
        "condition":
            "Rotate_Unseen_-45deg",

        "parameter":
            -45.0,

        "unit":
            "degrees",

        "status":
            "new strong unseen condition"
    },

    {
        "condition":
            "Rotate_Unseen_+45deg",

        "parameter":
            45.0,

        "unit":
            "degrees",

        "status":
            "new strong unseen condition"
    },

    {
        "condition":
            "Brightness_Unseen_0.60x",

        "parameter":
            0.60,

        "unit":
            "multiplicative_factor",

        "status":
            "new strong unseen condition"
    },

    {
        "condition":
            "Brightness_Unseen_1.60x",

        "parameter":
            1.60,

        "unit":
            "multiplicative_factor",

        "status":
            "new strong unseen condition"
    }

])

parameter_audit.to_csv(
    PARAMETER_AUDIT_PATH,
    index=False
)

print("\n")
print("=" * 110)
print("STRONG UNSEEN PARAMETERS")
print("=" * 110)

for name, info in (
    UNSEEN_CONDITIONS.items()
):

    print(
        f"{name:35s}"
        f" -> {info['value']}"
    )


# ============================================================
# 10. TRANSFORMATION FUNCTIONS
# ============================================================

def read_rgb(
    path
):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read:\n{path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


def apply_rotate(
    image_rgb,
    angle
):

    h, w = image_rgb.shape[:2]

    center = (
        w / 2.0,
        h / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0
        )
    )


def apply_brightness(
    image_rgb,
    factor
):

    result = (
        image_rgb
        .astype(np.float32)
        *
        factor
    )

    result = np.clip(
        result,
        0,
        255
    )

    return result.astype(
        np.uint8
    )


def apply_condition(
    image_rgb,
    condition_type,
    value
):

    if condition_type == "rotate":

        return apply_rotate(
            image_rgb,
            value
        )

    if condition_type == "brightness":

        return apply_brightness(
            image_rgb,
            value
        )

    raise ValueError(
        f"Unknown condition: "
        f"{condition_type}"
    )


# ============================================================
# 11. MODEL INPUT TRANSFORM
# ============================================================

input_transform = transforms.Compose([

    transforms.ToPILImage(),

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
# 12. DATASET
# ============================================================

class StrongUnseenDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        condition_type,
        condition_value
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.condition_type = (
            condition_type
        )

        self.condition_value = (
            condition_value
        )

    def __len__(self):

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

        path = resolve_path(
            row["filename"]
        )

        if path is None:

            raise FileNotFoundError(
                f"Cannot resolve:\n"
                f"{row['filename']}"
            )

        image = read_rgb(
            path
        )

        image = apply_condition(
            image,
            self.condition_type,
            self.condition_value
        )

        tensor = input_transform(
            image
        )

        return (
            tensor,
            int(row["y_true"]),
            index
        )


# ============================================================
# 13. MODEL BUILDERS
# ============================================================

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
# 14. CHECKPOINTS
# ============================================================

CHECKPOINTS = {

    "MobileNetV2": {

        42: os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed42.pth"
        ),

        1337: os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed1337.pth"
        ),

        2026: os.path.join(
            ROOT,
            "baseline_mobilenetv2_seed2026.pth"
        )
    },

    "MobileNetV3-Small": {

        42: os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed42.pth"
        ),

        1337: os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed1337.pth"
        ),

        2026: os.path.join(
            ROOT,
            "baseline_mobilenetv3_small_seed2026.pth"
        )
    },

    "EfficientNet-B0": {

        42: os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed42.pth"
        ),

        1337: os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed1337.pth"
        ),

        2026: os.path.join(
            ROOT,
            "baseline_efficientnet_b0_seed2026.pth"
        )
    },

    "ResNet18": {

        42: os.path.join(
            ROOT,
            "baseline_resnet18_seed42.pth"
        ),

        1337: os.path.join(
            ROOT,
            "baseline_resnet18_seed1337.pth"
        ),

        2026: os.path.join(
            ROOT,
            "baseline_resnet18_seed2026.pth"
        )
    }
}


for model_name in MODELS:

    for seed in SEEDS:

        path = CHECKPOINTS[
            model_name
        ][seed]

        assert os.path.exists(
            path
        ), (
            f"Checkpoint missing:\n{path}"
        )


# ============================================================
# 15. CHECKPOINT HELPERS
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

        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights"
        ]:

            if key in checkpoint:

                value = checkpoint[
                    key
                ]

                if isinstance(
                    value,
                    (
                        dict,
                        OrderedDict
                    )
                ):

                    return value

    raise ValueError(
        "Could not extract state_dict."
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
# 16. INFERENCE
# ============================================================

@torch.inference_mode()
def run_inference(
    model,
    dataframe,
    condition_type,
    condition_value
):

    dataset = StrongUnseenDataset(
        dataframe=dataframe,
        condition_type=condition_type,
        condition_value=condition_value
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    true_labels = []
    predictions = []
    probabilities = []

    for images, y_true, indices in loader:

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

        true_labels.extend(
            y_true
            .numpy()
            .tolist()
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

    return (
        np.asarray(
            true_labels,
            dtype=int
        ),

        np.asarray(
            predictions,
            dtype=int
        ),

        probabilities
    )


# ============================================================
# 17. RUN 48 INFERENCE GROUPS
# ============================================================

all_prediction_rows = []
all_metric_rows = []

total_runs = (
    len(MODELS)
    *
    len(SEEDS)
    *
    len(UNSEEN_CONDITIONS)
)

counter = 0


for model_name in MODELS:

    for seed in SEEDS:

        print("\n")
        print("=" * 110)
        print(
            f"{model_name} | seed={seed}"
        )
        print("=" * 110)

        model = build_model(
            model_name
        )

        checkpoint = torch.load(
            CHECKPOINTS[
                model_name
            ][seed],
            map_location="cpu",
            weights_only=False
        )

        state_dict = clean_state_dict(
            extract_state_dict(
                checkpoint
            )
        )

        load_result = (
            model.load_state_dict(
                state_dict,
                strict=False
            )
        )

        if load_result.missing_keys:

            raise RuntimeError(
                f"Missing checkpoint keys:\n"
                f"{load_result.missing_keys}"
            )

        model = model.to(
            DEVICE
        )

        model.eval()

        for condition_name, condition_info in (
            UNSEEN_CONDITIONS.items()
        ):

            counter += 1

            print(
                f"[{counter}/{total_runs}] "
                f"{condition_name}"
            )

            y_true, y_pred, probs = (
                run_inference(
                    model=model,
                    dataframe=test_original,
                    condition_type=condition_info[
                        "type"
                    ],
                    condition_value=condition_info[
                        "value"
                    ]
                )
            )

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            acc = accuracy_score(
                y_true,
                y_pred
            )

            macro_f1 = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )

            bal_acc = (
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            )

            all_metric_rows.append({

                "model":
                    model_name,

                "seed":
                    seed,

                "condition":
                    condition_name,

                "condition_type":
                    condition_info[
                        "type"
                    ],

                "parameter":
                    condition_info[
                        "value"
                    ],

                "accuracy":
                    acc,

                "macro_f1":
                    macro_f1,

                "balanced_accuracy":
                    bal_acc
            })

            # -----------------------------------------------
            # Prediction rows
            # -----------------------------------------------

            for i in range(
                len(test_original)
            ):

                row = test_original.iloc[
                    i
                ]

                all_prediction_rows.append({

                    "model":
                        model_name,

                    "seed":
                        seed,

                    "condition":
                        condition_name,

                    "condition_type":
                        condition_info[
                            "type"
                        ],

                    "parameter":
                        condition_info[
                            "value"
                        ],

                    "original_image_id":
                        row[
                            "original_image_id"
                        ],

                    "filename":
                        row[
                            "filename"
                        ],

                    "y_true":
                        int(
                            y_true[i]
                        ),

                    "y_pred":
                        int(
                            y_pred[i]
                        ),

                    "correct":
                        bool(
                            y_true[i]
                            ==
                            y_pred[i]
                        ),

                    "prob_healthy":
                        float(
                            probs[i, 0]
                        ),

                    "prob_little_leaf":
                        float(
                            probs[i, 1]
                        ),

                    "prob_phomopsis":
                        float(
                            probs[i, 2]
                        )
                })


        del model
        del checkpoint
        del state_dict

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 18. DATAFRAMES
# ============================================================

prediction_df = pd.DataFrame(
    all_prediction_rows
)

metrics_df = pd.DataFrame(
    all_metric_rows
)


assert len(
    prediction_df
) == 6144

assert len(
    metrics_df
) == 48


print("\n")
print("=" * 110)
print("STRONG UNSEEN DATASET")
print("=" * 110)

print(
    "Expected predictions:",
    6144
)

print(
    "Actual predictions:",
    len(prediction_df)
)


# ============================================================
# 19. SAVE PREDICTIONS / METRICS
# ============================================================

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# 20. SUMMARY ACROSS SEEDS
# ============================================================

summary_df = (
    metrics_df
    .groupby(
        [
            "model",
            "condition",
            "condition_type",
            "parameter"
        ]
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
    .reset_index()
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


print("\n")
print("=" * 110)
print("STRONG UNSEEN PERFORMANCE")
print("=" * 110)

display(
    summary_df
    .sort_values(
        [
            "model",
            "condition"
        ]
    )
)


# ============================================================
# 21. LOAD EXISTING ORIGINAL PREDICTIONS
# ============================================================

assert os.path.exists(
    ORIGINAL_PREDICTIONS_PATH
), (
    "Existing model-level original predictions not found:\n"
    f"{ORIGINAL_PREDICTIONS_PATH}"
)

original_predictions = pd.read_csv(
    ORIGINAL_PREDICTIONS_PATH
)

original_predictions = original_predictions[
    original_predictions[
        "transformation"
    ]
    ==
    "Original"
].copy()


# ============================================================
# 22. VERIFY ORIGINAL PREDICTIONS
# ============================================================

for model_name in MODELS:

    for seed in SEEDS:

        subset = original_predictions[
            (
                original_predictions["model"]
                ==
                model_name
            )
            &
            (
                original_predictions["seed"]
                ==
                seed
            )
        ]

        assert len(subset) == 128


# ============================================================
# 23. EXACT McNEMAR:
# ORIGINAL vs STRONG UNSEEN
#
# 4 models × 3 seeds × 4 conditions = 48 tests
# ============================================================

mcnemar_rows = []


for model_name in MODELS:

    for seed in SEEDS:

        original_subset = (
            original_predictions[
                (
                    original_predictions["model"]
                    ==
                    model_name
                )
                &
                (
                    original_predictions["seed"]
                    ==
                    seed
                )
            ]
            .copy()
        )

        for condition_name in (
            UNSEEN_CONDITIONS.keys()
        ):

            unseen_subset = prediction_df[
                (
                    prediction_df["model"]
                    ==
                    model_name
                )
                &
                (
                    prediction_df["seed"]
                    ==
                    seed
                )
                &
                (
                    prediction_df["condition"]
                    ==
                    condition_name
                )
            ].copy()

            assert len(
                unseen_subset
            ) == 128

            merged = original_subset.merge(
                unseen_subset,
                on="original_image_id",
                suffixes=(
                    "_original",
                    "_unseen"
                ),
                how="inner"
            )

            assert len(
                merged
            ) == 128

            # -----------------------------------------------
            # Same true labels
            # -----------------------------------------------

            assert np.array_equal(
                merged[
                    "y_true_original"
                ].to_numpy(),

                merged[
                    "y_true_unseen"
                ].to_numpy()
            )

            y_true = (
                merged[
                    "y_true_original"
                ].to_numpy()
            )

            original_pred = (
                merged[
                    "y_pred_original"
                ].to_numpy()
            )

            unseen_pred = (
                merged[
                    "y_pred_unseen"
                ].to_numpy()
            )

            original_correct = (
                original_pred
                ==
                y_true
            )

            unseen_correct = (
                unseen_pred
                ==
                y_true
            )

            n10 = int(
                np.sum(
                    original_correct
                    &
                    (~unseen_correct)
                )
            )

            n01 = int(
                np.sum(
                    (~original_correct)
                    &
                    unseen_correct
                )
            )

            discordant = n10 + n01

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

            original_accuracy = (
                original_correct.mean()
            )

            unseen_accuracy = (
                unseen_correct.mean()
            )

            delta_pp = (
                unseen_accuracy
                -
                original_accuracy
            ) * 100.0

            mcnemar_rows.append({

                "model":
                    model_name,

                "seed":
                    seed,

                "condition":
                    condition_name,

                "parameter":
                    UNSEEN_CONDITIONS[
                        condition_name
                    ]["value"],

                "original_correct_unseen_wrong":
                    n10,

                "original_wrong_unseen_correct":
                    n01,

                "discordant_pairs":
                    discordant,

                "original_accuracy":
                    original_accuracy,

                "unseen_accuracy":
                    unseen_accuracy,

                "accuracy_delta_pp_unseen_minus_original":
                    delta_pp,

                "p_value":
                    p_value
            })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)

assert len(
    mcnemar_df
) == 48


# ============================================================
# 24. BH-FDR
# ============================================================

reject, q_values, _, _ = multipletests(

    mcnemar_df[
        "p_value"
    ].to_numpy(),

    alpha=0.05,

    method="fdr_bh"
)


mcnemar_df[
    "fdr_q"
] = q_values

mcnemar_df[
    "significant_after_fdr"
] = (
    reject.astype(bool)
)


mcnemar_df = (
    mcnemar_df
    .sort_values(
        "fdr_q"
    )
    .reset_index(
        drop=True
    )
)


mcnemar_df.to_csv(
    MCNEMAR_ALL_PATH,
    index=False
)


# ============================================================
# 25. SIGNIFICANT ONLY
# ============================================================

significant_mcnemar = (
    mcnemar_df[
        mcnemar_df[
            "significant_after_fdr"
        ]
    ]
    .copy()
)

significant_mcnemar.to_csv(
    MCNEMAR_SIG_PATH,
    index=False
)


# ============================================================
# 26. SUMMARY
# ============================================================

mcnemar_summary = (
    mcnemar_df
    .groupby(
        "condition"
    )
    .agg(

        total_tests=(
            "p_value",
            "size"
        ),

        fdr_significant_tests=(
            "significant_after_fdr",
            "sum"
        ),

        mean_fdr_q=(
            "fdr_q",
            "mean"
        ),

        min_fdr_q=(
            "fdr_q",
            "min"
        ),

        mean_accuracy_delta_pp=(
            "accuracy_delta_pp_unseen_minus_original",
            "mean"
        ),

        mean_discordant_pairs=(
            "discordant_pairs",
            "mean"
        )
    )
    .reset_index()
)


mcnemar_summary.to_csv(
    MCNEMAR_SUMMARY_PATH,
    index=False
)


# ============================================================
# 27. PRINT MCNEMAR RESULTS
# ============================================================

print("\n")
print("=" * 110)
print("STRONG UNSEEN McNEMAR SUMMARY")
print("=" * 110)

display(
    mcnemar_summary
)


print("\n")
print("=" * 110)
print("FDR-SIGNIFICANT COMPARISONS")
print("=" * 110)

if len(
    significant_mcnemar
) == 0:

    print(
        "No comparison remained significant "
        "after BH-FDR."
    )

else:

    display(
        significant_mcnemar
    )


# ============================================================
# 28. FINAL VALIDATION
# ============================================================

assert (
    len(prediction_df)
    ==
    6144
)

assert (
    len(metrics_df)
    ==
    48
)

assert (
    len(mcnemar_df)
    ==
    48
)

group_counts = (
    prediction_df
    .groupby(
        [
            "model",
            "seed",
            "condition"
        ]
    )
    .size()
)

assert (
    group_counts == 128
).all()


# ============================================================
# 29. OUTPUT PATHS
# ============================================================

print("\n")
print("=" * 110)
print("SAVED OUTPUTS")
print("=" * 110)

print(
    "\nParameter audit:"
)

print(
    PARAMETER_AUDIT_PATH
)

print(
    "\nPredictions:"
)

print(
    PREDICTIONS_PATH
)

print(
    "\nMetrics:"
)

print(
    METRICS_PATH
)

print(
    "\nSummary:"
)

print(
    SUMMARY_PATH
)

print(
    "\nAll 48 McNemar tests:"
)

print(
    MCNEMAR_ALL_PATH
)

print(
    "\nFDR-significant only:"
)

print(
    MCNEMAR_SIG_PATH
)

print(
    "\nMcNemar summary:"
)

print(
    MCNEMAR_SUMMARY_PATH
)


# ============================================================
# 30. FINAL
# ============================================================

print("\n")
print("=" * 110)
print("✅ STRONG UNSEEN-PARAMETER TEST COMPLETE")
print("=" * 110)

print(
    "Models            :",
    4
)

print(
    "Seeds             :",
    3
)

print(
    "Unseen conditions :",
    4
)

print(
    "Test leaves       :",
    128
)

print(
    "Total predictions :",
    len(prediction_df)
)

print(
    "McNemar tests     :",
    len(mcnemar_df)
)

print(
    "FDR-significant   :",
    int(
        mcnemar_df[
            "significant_after_fdr"
        ].sum()
    )
)

print("\n✅ No training")
print("✅ No checkpoint selection")
print("✅ Strongly separated unseen parameters")
print("✅ Exact two-sided McNemar")
print("✅ Benjamini-Hochberg FDR")
print("=" * 110)

# %% [Cell 75]
# ============================================================
# A1 — RESNET18 TARGETED AUGMENTATION + CONTROLS
#
# EXACT DESIGN:
#   Baseline          = Original
#   Grayscale         = Original + Grayscale
#   Rotate            = Original + Rotate
#   Brightness        = Original + Brightness
#   Duplicate-Control = Original + duplicated Original
#   Gaussian-Control  = Original + Gaussian_Blur
#
# 3 SEEDS:
#   42, 1337, 2026
#
# TOTAL TRAINING RUNS:
#   6 strategies  3 seeds = 18
#
# TEST CONDITIONS:
#   Original
#   Grayscale
#   Rotate
#   Brightness
#
# MODEL:
#   ResNet18 ImageNet-pretrained
#
# CHECKPOINT SELECTION:
#   Original validation loss ONLY
#
# STATISTICS:
#   Exact two-sided McNemar
#   Benjamini-Hochberg FDR
#
# NO TEST-SET MODEL SELECTION
# ============================================================

import os
import gc
import copy
import random
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "resnet18_targeted_ablation_controls"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SEEDS = [
    42,
    1337,
    2026
]

NUM_CLASSES = 3

BATCH_SIZE = 32

MAX_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LR_FACTOR = 0.5

LR_PATIENCE = 2

MIN_LR = 1e-7

EARLY_STOP_PATIENCE = 4

NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. STRATEGIES
# ============================================================

STRATEGIES = OrderedDict({

    "Baseline": {
        "train_transforms": [
            "Original"
        ]
    },

    "Grayscale": {
        "train_transforms": [
            "Original",
            "Grayscale"
        ]
    },

    "Rotate": {
        "train_transforms": [
            "Original",
            "Rotate"
        ]
    },

    "Brightness": {
        "train_transforms": [
            "Original",
            "Brightness"
        ]
    },

    "Duplicate_Original": {
        "train_transforms": [
            "Original",
            "Original"
        ]
    },

    "Gaussian_Control": {
        "train_transforms": [
            "Original",
            "Gaussian_Blur"
        ]
    }
})


TEST_TRANSFORMS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]


# ============================================================
# 3. OUTPUT FILES
# ============================================================

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

TRAIN_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet18_targeted_training_log.csv"
)

VAL_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet18_targeted_validation_metrics.csv"
)

TEST_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet18_targeted_locked_test_metrics.csv"
)

TEST_PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet18_targeted_locked_test_predictions.csv"
)

TEST_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet18_targeted_locked_test_summary.csv"
)

MCNEMAR_ALL_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet18_targeted_mcnemar_all.csv"
)

MCNEMAR_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet18_targeted_mcnemar_summary.csv"
)

# ============================================================
# 4. HEADER
# ============================================================

print("=" * 110)
print("A1 — RESNET18 TARGETED AUGMENTATION + CONTROLS")
print("=" * 110)

print(
    "PyTorch     :",
    torch.__version__
)

print(
    "Torchvision :",
    __import__(
        "torchvision"
    ).__version__
)

print(
    "Device      :",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU         :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

         torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 6. LOAD METADATA
# ============================================================

assert os.path.exists(
    METADATA_PATH
), (
    f"Metadata not found:\n"
    f"{METADATA_PATH}"
)

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "\nMetadata shape:",
    metadata.shape
)

print(
    "Columns:",
    metadata.columns.tolist()
)


REQUIRED_COLUMNS = [
    "filename",
    "class_label",
    "original_image_id",
    "preprocessing_technique",
    "data_split"
]

for column in REQUIRED_COLUMNS:

    assert column in metadata.columns, (
        f"Missing metadata column: {column}"
    )

for column in REQUIRED_COLUMNS:

    metadata[column] = (
        metadata[column]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 7. CLASS MAPPING
# ============================================================

CLASS_TO_INDEX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}


metadata["class_index"] = (
    metadata[
        "class_label"
    ]
    .map(
        CLASS_TO_INDEX
    )
)

assert not metadata[
    "class_index"
].isna().any()


# ============================================================
# 8. DATASET VOCABULARY
# ============================================================

print("\n")
print("=" * 110)
print("DATASET VOCABULARY")
print("=" * 110)

print(
    metadata[
        "data_split"
    ].value_counts()
)

print()

print(
    metadata[
        "preprocessing_technique"
    ].value_counts()
)


# ============================================================
# 9. BUILD IMAGE INDEX
# ============================================================

print("\n")
print("=" * 110)
print("BUILDING IMAGE INDEX")
print("=" * 110)

image_index = {}

for subfolder in [
    "Raw",
    "Augmented"
]:

    folder = os.path.join(
        ROOT,
        subfolder
    )

    if not os.path.exists(
        folder
    ):

        continue

    for current_root, _, files in os.walk(
        folder
    ):

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

            relative_path = os.path.relpath(
                full_path,
                ROOT
            ).replace(
                "\\",
                "/"
            )

            image_index[
                relative_path
            ] = full_path

            image_index[
                filename
            ] = full_path

print(
    "Indexed files:",
    len(image_index)
)


def resolve_image_path(
    filename
):

    filename = (
        str(filename)
        .strip()
        .replace("\\", "/")
    )

    filename = filename.lstrip(
        "./"
    )

    if filename in image_index:

        return image_index[
            filename
        ]

    direct = os.path.join(
        ROOT,
        filename
    )

    if os.path.isfile(
        direct
    ):

        return os.path.abspath(
            direct
        )

    basename = os.path.basename(
        filename
    )

    if basename in image_index:

        return image_index[
            basename
        ]

    return None


# ============================================================
# 10. SPLIT VERIFICATION
# ============================================================

print("\n")
print("=" * 110)
print("SPLIT VERIFICATION")
print("=" * 110)

for split_name in [
    "train",
    "val",
    "test"
]:

    split_df = metadata[
        metadata[
            "data_split"
        ].str.lower()
        ==
        split_name
    ]

    print(
        split_name,
        "rows:",
        len(split_df),
        "| unique leaves:",
        split_df[
            "original_image_id"
        ].nunique()
    )


# ============================================================
# 11. ORIGINAL TRAIN / VAL / TEST
# ============================================================

train_original = metadata[
    (
        metadata[
            "data_split"
        ].str.lower()
        ==
        "train"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        ==
        "Original"
    )
].copy()

val_original = metadata[
    (
        metadata[
            "data_split"
        ].str.lower()
        ==
        "val"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        ==
        "Original"
    )
].copy()

test_original = metadata[
    (
        metadata[
            "data_split"
        ].str.lower()
        ==
        "test"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        ==
        "Original"
    )
].copy()


assert len(
    train_original
) == 595

assert len(
    val_original
) == 127

assert len(
    test_original
) == 128


print("\n")
print("=" * 110)
print("ORIGINAL LEAF COUNTS")
print("=" * 110)

print(
    "Train:",
    len(train_original)
)

print(
    "Validation:",
    len(val_original)
)

print(
    "Test:",
    len(test_original)
)


# ============================================================
# 12. VERIFY NO SPLIT LEAKAGE
# ============================================================

train_ids = set(
    train_original[
        "original_image_id"
    ]
)

val_ids = set(
    val_original[
        "original_image_id"
    ]
)

test_ids = set(
    test_original[
        "original_image_id"
    ]
)

assert (
    train_ids
    .isdisjoint(
        val_ids
    )
)

assert (
    train_ids
    .isdisjoint(
        test_ids
    )
)

assert (
    val_ids
    .isdisjoint(
        test_ids
    )
)


# ============================================================
# 13. PREPARE ALL TRAINING ROWS
# ============================================================

def get_split_transform(
    split_name,
    transformation
):

    return metadata[
        (
            metadata[
                "data_split"
            ].str.lower()
            ==
            split_name
        )
        &
        (
            metadata[
                "preprocessing_technique"
            ]
            ==
            transformation
        )
    ].copy()


TRAIN_BY_TRANSFORM = {}
VAL_BY_TRANSFORM = {}
TEST_BY_TRANSFORM = {}

for transformation in [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness",
    "Gaussian_Blur"
]:

    TRAIN_BY_TRANSFORM[
        transformation
    ] = get_split_transform(
        "train",
        transformation
    )

    VAL_BY_TRANSFORM[
        transformation
    ] = get_split_transform(
         "val",
         transformation
    )

    TEST_BY_TRANSFORM[
        transformation
    ] = get_split_transform(
         "test",
         transformation
    )


for transformation in [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness",
    "Gaussian_Blur"
]:

    assert len(
        TRAIN_BY_TRANSFORM[
            transformation
        ]
    ) == 595

    assert len(
        VAL_BY_TRANSFORM[
            transformation
        ]
    ) == 127

    assert len(
        TEST_BY_TRANSFORM[
            transformation
        ]
    ) == 128


# ============================================================
# 14. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

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

eval_transform = transforms.Compose([

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

class MetadataImageDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

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

        path = resolve_image_path(
            row[
                "filename"
            ]
        )

        if path is None:

             raise FileNotFoundError(
                 f"Image not found:\n"
                 f"{row['filename']}"
             )

        image = Image.open(
            path
        ).convert(
             "RGB"
        )

        image = self.transform(
             image
        )

        label = int(
            row[
                "class_index"
            ]
        )

        return (
            image,
            label,
            row[
                "original_image_id"
            ],
            row[
                "filename"
            ]
        )


# ============================================================
# 16. BUILD TRAINING DATA
# ============================================================

def build_training_dataframe(
    strategy_name
):

    strategy = STRATEGIES[
        strategy_name
    ]

    transformations = strategy[
        "train_transforms"
    ]

    parts = []

    for transformation in transformations:

        part = TRAIN_BY_TRANSFORM[
            transformation
        ].copy()

        parts.append(
            part
        )

    result = pd.concat(
        parts,
        ignore_index=True
    )

    return result


# ============================================================
# 17. BUILD VALIDATION DATA
# ============================================================

def build_validation_dataframe(
    transformation
):

    return VAL_BY_TRANSFORM[
        transformation
    ].copy()


# ============================================================
# 18. MODEL
# ============================================================

def build_resnet18():

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# 19. LOSS / METRICS
# ============================================================

criterion = nn.CrossEntropyLoss()


def calculate_metrics(
    y_true,
    y_pred
):

    return {

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
            )
    }


# ============================================================
# 20. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    running_loss = 0.0

    y_true = []

    y_pred = []

    total = 0

    for images, labels, _, _ in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(
            0
        )

        running_loss += (
            loss.item()
            *
            batch_size
        )

        total += batch_size

        preds = torch.argmax(
            logits,
            dim=1
        )

        y_true.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            preds
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics[
        "loss"
    ] = (
        running_loss
        /
        total
    )

    return metrics


# ============================================================
# 21. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    running_loss = 0.0

    total = 0

    y_true = []

    y_pred = []

    for images, labels, _, _ in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        batch_size = labels.size(
            0
        )

        running_loss += (
            loss.item()
            *
            batch_size
        )

        total += batch_size

        preds = torch.argmax(
            logits,
            dim=1
        )

        y_true.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            preds
            .cpu()
            .numpy()
            .tolist()
        )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics[
        "loss"
    ] = (
        running_loss
        /
        total
    )

    return metrics


# ============================================================
# 22. TRAINING LOOP
# ============================================================

training_rows = []
validation_rows = []
best_checkpoints = {}

for strategy_name in STRATEGIES:

    print("\n")
    print("=" * 110)
    print(
        f"STRATEGY: {strategy_name}"
    )
    print("=" * 110)

    train_df = build_training_dataframe(
        strategy_name
    )

    print(
        "Training images:",
        len(train_df)
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if strategy_name == "Baseline":

        assert len(
            train_df
        ) == 595

    else:

        assert len(
            train_df
        ) == 1190


    for seed in SEEDS:

        seed_everything(
            seed
        )

        print("\n")
        print(
            "-" * 100
        )

        print(
            f"{strategy_name} | seed={seed}"
        )

        # ----------------------------------------------------
        # TRAIN DATASET
        # ----------------------------------------------------

        train_dataset = (
            MetadataImageDataset(
                train_df,
                train_transform
            )
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        # ----------------------------------------------------
        # ORIGINAL VALIDATION
        #
        # IMPORTANT:
        # checkpoint selection is based ONLY on
        # original validation loss.
        # ----------------------------------------------------

        original_val_dataset = (
            MetadataImageDataset(
                VAL_BY_TRANSFORM[
                    "Original"
                ],
                eval_transform
            )
        )

        original_val_loader = DataLoader(
            original_val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        # ----------------------------------------------------
        # TARGET VALIDATION LOADERS
        # Used for reporting only.
        # ----------------------------------------------------

        validation_loaders = {}

        for transformation in [
            "Original",
            "Grayscale",
            "Rotate",
            "Brightness"
        ]:

            val_dataset = (
                MetadataImageDataset(
                    VAL_BY_TRANSFORM[
                        transformation
                    ],
                    eval_transform
                )
            )

            validation_loaders[
                transformation
            ] = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available()
            )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = build_resnet18()

        model = model.to(
            DEVICE
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR
        )

        best_val_loss = float(
            "inf"
        )

        best_epoch = None

        best_state = None

        epochs_without_improvement = 0

        # ----------------------------------------------------
        # EPOCHS
        # ----------------------------------------------------

        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer
            )

            original_val_metrics = evaluate(
                model,
                original_val_loader
            )

            scheduler.step(
                original_val_metrics[
                    "loss"
                ]
            )

            current_lr = (
                optimizer.param_groups[0][
                    "lr"
                ]
            )

            row = {

                "strategy":
                    strategy_name,

                "seed":
                    seed,

                "epoch":
                    epoch,

                "learning_rate":
                    current_lr,

                "train_loss":
                    train_metrics[
                        "loss"
                    ],

                "train_accuracy":
                    train_metrics[
                        "accuracy"
                    ],

                "train_macro_f1":
                    train_metrics[
                        "macro_f1"
                    ],

                "train_balanced_accuracy":
                    train_metrics[
                        "balanced_accuracy"
                    ],

                "original_val_loss":
                    original_val_metrics[
                        "loss"
                    ],

                "original_val_accuracy":
                    original_val_metrics[
                        "accuracy"
                    ],

                "original_val_macro_f1":
                    original_val_metrics[
                        "macro_f1"
                    ],

                "original_val_balanced_accuracy":
                    original_val_metrics[
                        "balanced_accuracy"
                    ]
            }

            training_rows.append(
                row
            )

            print(
                f"Epoch {epoch:02d} | "
                f"Train F1={train_metrics['macro_f1']:.4f} | "
                f"Orig Val Loss={original_val_metrics['loss']:.4f} | "
                f"Orig Val F1={original_val_metrics['macro_f1']:.4f} | "
                f"LR={current_lr:.2e}"
            )

            # ------------------------------------------------
            # CHECKPOINT SELECTION
            # ------------------------------------------------

            if (
                original_val_metrics[
                    "loss"
                ]
                <
                best_val_loss
            ):

                best_val_loss = (
                    original_val_metrics[
                        "loss"
                    ]
                )

                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >=
                EARLY_STOP_PATIENCE
            ):

                print(
                    f"Early stopping at epoch "
                    f"{epoch}"
                )

                break

        # ----------------------------------------------------
        # RESTORE BEST CHECKPOINT
        # ----------------------------------------------------

        model.load_state_dict(
            best_state
        )

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            (
                f"resnet18_"
                f"{strategy_name.lower()}_"
                f"seed{seed}_"
                f"best.pth"
            )
        )

        torch.save(
            {
                "model_name":
                    "ResNet18",

                "strategy":
                    strategy_name,

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "best_original_val_loss":
                    best_val_loss,

                "state_dict":
                    model.state_dict()
            },
            checkpoint_path
        )

        best_checkpoints[
            (
                strategy_name,
                seed
            )
        ] = checkpoint_path

        print(
            f"BEST epoch={best_epoch} | "
            f"Original Val Loss={best_val_loss:.6f}"
        )

        # ----------------------------------------------------
        # VALIDATION METRICS AT BEST CHECKPOINT
        # ----------------------------------------------------

        for transformation in [
            "Original",
            "Grayscale",
            "Rotate",
            "Brightness"
        ]:

            val_metrics = evaluate(
                model,
                validation_loaders[
                    transformation
                ]
            )

            validation_rows.append({

                "strategy":
                    strategy_name,

                "seed":
                    seed,

                "best_epoch":
                    best_epoch,

                "validation_transformation":
                    transformation,

                "loss":
                    val_metrics[
                        "loss"
                    ],

                "accuracy":
                    val_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    val_metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    val_metrics[
                        "balanced_accuracy"
                    ]
            })

        del model
        del train_loader
        del original_val_loader
        del validation_loaders

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 23. SAVE TRAINING / VALIDATION LOGS
# ============================================================

training_df = pd.DataFrame(
    training_rows
)

validation_df = pd.DataFrame(
    validation_rows
)

training_df.to_csv(
    TRAIN_LOG_PATH,
    index=False
)

validation_df.to_csv(
    VAL_METRICS_PATH,
    index=False
)


print("\n")
print("=" * 110)
print("TRAINING COMPLETE")
print("=" * 110)

print(
    "Training log:",
    TRAIN_LOG_PATH
)

print(
    "Validation metrics:",
    VAL_METRICS_PATH
)


# ============================================================
# 24. VALIDATION SUMMARY
# ============================================================

validation_summary = (
    validation_df
    .groupby(
        [
            "strategy",
            "validation_transformation"
        ]
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
    .reset_index()
)

print("\n")
print("=" * 110)
print("VALIDATION SUMMARY")
print("=" * 110)

display(
    validation_summary
)


# ============================================================
# 25. LOCKED TEST INFERENCE
# ============================================================

test_metrics_rows = []

test_prediction_rows = []

print("\n")
print("=" * 110)
print("LOCKED TEST INFERENCE")
print("=" * 110)


for strategy_name in STRATEGIES:

    for seed in SEEDS:

        checkpoint_path = (
            best_checkpoints[
                (
                    strategy_name,
                    seed
                )
            ]
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        model = build_resnet18()

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

        model = model.to(
            DEVICE
        )

        model.eval()

        for transformation in TEST_TRANSFORMS:

            test_dataset = (
                MetadataImageDataset(
                    TEST_BY_TRANSFORM[
                        transformation
                    ],
                    eval_transform
                )
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available()
            )

            y_true = []

            y_pred = []

            probabilities = []

            ids = []

            filenames = []

            with torch.no_grad():

                for (
                    images,
                    labels,
                    original_ids,
                    batch_filenames
                ) in test_loader:

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

                    y_true.extend(
                        labels
                        .numpy()
                        .tolist()
                    )

                    y_pred.extend(
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

                    ids.extend(
                        list(
                            original_ids
                        )
                    )

                    filenames.extend(
                        list(
                            batch_filenames
                        )
                    )

            probabilities = np.concatenate(
                probabilities,
                axis=0
            )

            y_true_np = np.asarray(
                y_true,
                dtype=int
            )

            y_pred_np = np.asarray(
                y_pred,
                dtype=int
            )

            metrics = calculate_metrics(
                y_true_np,
                y_pred_np
            )

            test_metrics_rows.append({

                "model":
                    "ResNet18",

                "strategy":
                    strategy_name,

                "seed":
                    seed,

                "transformation":
                    transformation,

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    metrics[
                        "balanced_accuracy"
                    ]
            })

            for i in range(
                len(y_true_np)
            ):

                test_prediction_rows.append({

                    "model":
                        "ResNet18",

                    "strategy":
                        strategy_name,

                    "seed":
                        seed,

                    "transformation":
                        transformation,

                    "original_image_id":
                        ids[i],

                    "filename":
                        filenames[i],

                    "y_true":
                        int(
                            y_true_np[i]
                        ),

                    "y_pred":
                        int(
                            y_pred_np[i]
                        ),

                    "correct":
                        bool(
                            y_true_np[i]
                            ==
                            y_pred_np[i]
                        ),

                    "prob_healthy":
                        float(
                            probabilities[
                                i,
                                0
                            ]
                        ),

                    "prob_little_leaf":
                        float(
                            probabilities[
                                i,
                                1
                            ]
                        ),

                    "prob_phomopsis":
                        float(
                            probabilities[
                                i,
                                2
                            ]
                        )
                })


        del model
        del checkpoint

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 26. SAVE TEST RESULTS
# ============================================================

test_metrics_df = pd.DataFrame(
    test_metrics_rows
)

test_predictions_df = pd.DataFrame(
    test_prediction_rows
)


assert len(
    test_metrics_df
) == (
    6
    *
    3
    *
    4
)

assert len(
    test_predictions_df
) == (
    6
    *
    3
    *
    4
    *
    128
)


test_metrics_df.to_csv(
    TEST_METRICS_PATH,
    index=False
)

test_predictions_df.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 27. TEST SUMMARY
# ============================================================

test_summary = (
    test_metrics_df
    .groupby(
        [
            "strategy",
            "transformation"
        ]
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
    .reset_index()
)

test_summary.to_csv(
    TEST_SUMMARY_PATH,
    index=False
)


print("\n")
print("=" * 110)
print("LOCKED TEST SUMMARY")
print("=" * 110)

display(
    test_summary
)


# ============================================================
# 28. McNEMAR TARGETED VS BASELINE
#
# For each targeted strategy:
#   Grayscale
#   Rotate
#   Brightness
#
# Compare against Baseline on the SAME test leaves.
#
# 3 targeted strategies × 3 seeds = 9 tests
# ============================================================

baseline_predictions = (
    test_predictions_df[
        test_predictions_df[
            "strategy"
        ]
        ==
        "Baseline"
    ]
    .copy()
)


mcnemar_rows = []


for targeted_strategy in [
    "Grayscale",
    "Rotate",
    "Brightness"
]:

    for seed in SEEDS:

        for transformation in [
            "Original",
            targeted_strategy
        ]:

            baseline = (
                baseline_predictions[
                    (
                        baseline_predictions[
                            "seed"
                        ]
                        ==
                        seed
                    )
                    &
                    (
                        baseline_predictions[
                            "transformation"
                        ]
                        ==
                        transformation
                    )
                ]
                .copy()
            )

            targeted = (
                test_predictions_df[
                    (
                        test_predictions_df[
                            "strategy"
                        ]
                        ==
                        targeted_strategy
                    )
                    &
                    (
                        test_predictions_df[
                            "seed"
                        ]
                        ==
                        seed
                    )
                    &
                    (
                        test_predictions_df[
                            "transformation"
                        ]
                        ==
                        transformation
                    )
                ]
                .copy()
            )

            merged = baseline.merge(
                targeted,
                on="original_image_id",
                suffixes=(
                    "_baseline",
                    "_targeted"
                )
            )

            assert len(
                merged
            ) == 128

            y_true = merged[
                "y_true_baseline"
            ].to_numpy()

            baseline_correct = (
                merged[
                    "y_pred_baseline"
                ].to_numpy()
                ==
                y_true
            )

            targeted_correct = (
                merged[
                    "y_pred_targeted"
                ].to_numpy()
                ==
                y_true
            )

            n10 = int(
                np.sum(
                    baseline_correct
                    &
                    (~targeted_correct)
                )
            )

            n01 = int(
                np.sum(
                    (~baseline_correct)
                    &
                    targeted_correct
                )
            )

            discordant = (
                n10
                +
                n01
            )

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

            baseline_acc = (
                baseline_correct.mean()
            )

            targeted_acc = (
                targeted_correct.mean()
            )

            mcnemar_rows.append({

                "targeted_strategy":
                    targeted_strategy,

                "seed":
                    seed,

                "transformation":
                    transformation,

                "baseline_correct_targeted_wrong":
                    n10,

                "baseline_wrong_targeted_correct":
                    n01,

                "discordant_pairs":
                    discordant,

                "baseline_accuracy":
                    baseline_acc,

                "targeted_accuracy":
                    targeted_acc,

                "accuracy_delta_pp":
                    (
                        targeted_acc
                        -
                        baseline_acc
                    )
                    *
                    100.0,

                "p_value":
                    p_value
            })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)


# ============================================================
# 29. BH-FDR
# ============================================================

reject, q_values, _, _ = multipletests(

    mcnemar_df[
        "p_value"
    ].to_numpy(),

    alpha=0.05,

    method="fdr_bh"
)

mcnemar_df[
    "fdr_q"
] = q_values

mcnemar_df[
    "significant_after_fdr"
] = (
    reject.astype(bool)
)


mcnemar_df.to_csv(
    MCNEMAR_ALL_PATH,
    index=False
)


# ============================================================
# 30. McNEMAR SUMMARY
# ============================================================

mcnemar_summary = (
    mcnemar_df
    .groupby(
        [
            "targeted_strategy",
            "transformation"
        ]
    )
    .agg(

        tests=(
            "p_value",
            "size"
        ),

        fdr_significant=(
            "significant_after_fdr",
            "sum"
        ),

        mean_q=(
            "fdr_q",
            "mean"
        ),

        min_q=(
            "fdr_q",
            "min"
        ),

        mean_accuracy_delta_pp=(
            "accuracy_delta_pp",
            "mean"
        )
    )
    .reset_index()
)

mcnemar_summary.to_csv(
    MCNEMAR_SUMMARY_PATH,
    index=False
)


# ============================================================
# 31. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 110)
print("McNEMAR — TARGETED VS BASELINE")
print("=" * 110)

display(
    mcnemar_df
    .sort_values(
        "fdr_q"
    )
)


print("\n")
print("=" * 110)
print("McNEMAR SUMMARY")
print("=" * 110)

display(
    mcnemar_summary
)


# ============================================================
# 32. FINAL SANITY CHECKS
# ============================================================

assert (
    len(
        test_predictions_df
    )
    ==
    9216
)

# 6 strategies × 3 seeds × 4 transformations × 128
#
# = 9,216 predictions


group_sizes = (
    test_predictions_df
    .groupby(
        [
            "strategy",
            "seed",
            "transformation"
        ]
    )
    .size()
)

assert (
    group_sizes == 128
).all()


print("\n")
print("=" * 110)
print("OUTPUT FILES")
print("=" * 110)

print(
    "\nTraining log:"
)

print(
    TRAIN_LOG_PATH
)

print(
    "\nValidation metrics:"
)

print(
    VAL_METRICS_PATH
)

print(
    "\nTest metrics:"
)

print(
    TEST_METRICS_PATH
)

print(
    "\nTest predictions:"
)

print(
    TEST_PREDICTIONS_PATH
)

print(
    "\nTest summary:"
)

print(
    TEST_SUMMARY_PATH
)

print(
    "\nMcNemar all tests:"
)

print(
    MCNEMAR_ALL_PATH
)

print(
    "\nMcNemar summary:"
)

print(
    MCNEMAR_SUMMARY_PATH
)


print("\n")
print("=" * 110)
print("✅ A1 COMPLETE")
print("=" * 110)

print(
    "Model              : ResNet18"
)

print(
    "Seeds              :",
    len(SEEDS)
)

print(
    "Strategies         :",
    len(STRATEGIES)
)

print(
    "Training runs      :",
    len(
        STRATEGIES
    )
    *
    len(SEEDS)
)

print(
    "Test conditions    :",
    len(TEST_TRANSFORMS)
)

print(
    "Test predictions   :",
    len(
        test_predictions_df
    )
)

print(
    "McNemar tests      :",
    len(
        mcnemar_df
    )
)

print("=" * 110)


# %% [Cell 76]
import os

root = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        name = f.lower()
        if any(x in name for x in [
            "mnv3",
            "mobilenet",
            "targeted",
            "ablation",
            "augmentation"
        ]):
            print(os.path.join(dirpath, f))

# %% [Cell 77]
# ============================================================
# A3 — UNSEEN TRANSFORMATION SEVERITY
# COMPLETE EXPERIMENT
#
# 1) Train 6 targeted MobileNetV3-Small models:
#       Rotate      × seeds 42, 1337, 2026
#       Brightness  × seeds 42, 1337, 2026
#
# 2) Use existing baseline MobileNetV3-Small checkpoints
#
# 3) Evaluate BOTH baseline and targeted models on:
#       Rotate:     -45°, +45°
#       Brightness: 0.60, 1.60
#
# 4) Report:
#       - Accuracy
#       - Macro-F1
#       - Balanced Accuracy
#       - Mean ± SD
#       - Targeted - Baseline deltas
#       - Exact McNemar
#       - BH-FDR
#
# IMPORTANT:
# - No test data used for checkpoint selection
# - Validation = ORIGINAL validation set only
# - Training = Original + corresponding targeted transformation
# - 595 + 595 = 1190 training images
# ===========================================================

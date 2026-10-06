
# %% [Cell 78]
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
# ============================================================

import os
import gc
import copy
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

from scipy.stats import binomtest


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

# Existing baseline checkpoints
BASELINE_CHECKPOINTS = {

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
}


# ------------------------------------------------------------
# A3 output folder
# ------------------------------------------------------------

A3_ROOT = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "A3_unseen_severity"
)

CHECKPOINT_DIR = os.path.join(
    A3_ROOT,
    "checkpoints"
)

HISTORY_DIR = os.path.join(
    A3_ROOT,
    "training_histories"
)

os.makedirs(
    A3_ROOT,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    HISTORY_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

MODEL_NAME = "MobileNetV3-Small"

SEEDS = [
    42,
    1337,
    2026
]

TARGETED_STRATEGIES = [
    "Rotate",
    "Brightness"
]

IMAGE_SIZE = 224

BATCH_SIZE = 32

MAX_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LR_FACTOR = 0.5

LR_PATIENCE = 2

MIN_LR = 1e-7

EARLY_STOPPING_PATIENCE = 4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ImageNet normalization
MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
]


print("=" * 100)
print("A3 — UNSEEN TRANSFORMATION SEVERITY")
print("=" * 100)

print(
    f"Model      : {MODEL_NAME}"
)

print(
    f"Device     : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU        : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Seeds      : {SEEDS}"
)

print(
    f"Strategies : {TARGETED_STRATEGIES}"
)

print(
    f"Output     : {A3_ROOT}"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. LOAD METADATA
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

print("\nMetadata shape:")
print(
    metadata.shape
)

print("\nColumns:")
print(
    list(metadata.columns)
)


# ============================================================
# 4. CLASS MAPPING
# ============================================================

CLASS_TO_IDX = {

    "Healthy_Leaves": 0,

    "Little_Leaf": 1,

    "Phomopsis_Blight": 2
}


unique_classes = sorted(
    metadata[
        "class_label"
    ].dropna().unique().tolist()
)

print(
    "\nMetadata classes:"
)

print(
    unique_classes
)


unknown_classes = (
    set(unique_classes)
    -
    set(CLASS_TO_IDX.keys())
)

if unknown_classes:

    raise RuntimeError(
        f"Unexpected class labels found: "
        f"{unknown_classes}"
    )


metadata["label"] = (
    metadata["class_label"]
    .map(CLASS_TO_IDX)
    .astype(int)
)


# ============================================================
# 5. ROBUST IMAGE PATH INDEX
# ============================================================

print(
    "\nBuilding image index..."
)

image_index = {}


for base_dir in [
    os.path.join(ROOT, "Raw"),
    os.path.join(ROOT, "Augmented")
]:

    if not os.path.exists(base_dir):

        continue

    for dirpath, _, filenames in os.walk(
        base_dir
    ):

        for filename in filenames:

            if not filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                )
            ):

                continue

            full_path = os.path.join(
                dirpath,
                filename
            )

            image_index[
                filename
            ] = full_path

            image_index[
                filename.lower()
            ] = full_path


print(
    f"Indexed keys: "
    f"{len(image_index)}"
)


# ============================================================
# 6. PATH RESOLVER
# ============================================================

def resolve_image_path(
    filename
):

    filename = str(
        filename
    )

    # Exact
    if filename in image_index:

        return image_index[
            filename
        ]

    # Lowercase
    lower = filename.lower()

    if lower in image_index:

        return image_index[
            lower
        ]

    # Basename
    basename = os.path.basename(
        filename
    )

    if basename in image_index:

        return image_index[
            basename
        ]

    basename_lower = (
        basename.lower()
    )

    if basename_lower in image_index:

        return image_index[
            basename_lower
        ]

    return None


metadata["image_path"] = (
    metadata["filename"]
    .apply(
        resolve_image_path
    )
)


missing = metadata[
    metadata["image_path"].isna()
]


print(
    f"Resolved paths: "
    f"{metadata['image_path'].notna().sum()}"
)

print(
    f"Missing paths : "
    f"{len(missing)}"
)


if len(missing) > 0:

    print(
        "\nFirst missing files:"
    )

    print(
        missing[
            "filename"
        ].head(20).tolist()
    )

    raise RuntimeError(
        "Some metadata files could not be resolved."
    )


print(
    "✓ All metadata files resolved."
)


# ============================================================
# 7. BUILD SPLITS
# ============================================================

train_original = metadata[
    (
        metadata[
            "data_split"
        ]
        == "train"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        == "Original"
    )
].copy()


val_original = metadata[
    (
        metadata[
            "data_split"
        ]
        == "val"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        == "Original"
    )
].copy()


test_original = metadata[
    (
        metadata[
            "data_split"
        ]
        == "test"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        == "Original"
    )
].copy()


print("\n")
print("=" * 100)
print("SPLITS")
print("=" * 100)

print(
    f"Original train : {len(train_original)}"
)

print(
    f"Original val   : {len(val_original)}"
)

print(
    f"Original test  : {len(test_original)}"
)


assert len(train_original) == 595

assert len(val_original) == 127

assert len(test_original) == 128


# ============================================================
# 8. VERIFY TARGET TRAINING TRANSFORMATIONS
# ============================================================

training_transform_counts = (
    metadata[
        metadata[
            "data_split"
        ]
        == "train"
    ]
    [
        "preprocessing_technique"
    ]
    .value_counts()
)


print("\nTraining transformation counts:")

print(
    training_transform_counts
)


for transformation in [
    "Rotate",
    "Brightness"
]:

    count = (
        metadata[
            (
                metadata[
                    "data_split"
                ]
                == "train"
            )
            &
            (
                metadata[
                    "preprocessing_technique"
                ]
                == transformation
            )
        ]
        .shape[0]
    )

    assert count == 595, (
        f"{transformation} "
        f"training count should be 595, "
        f"got {count}"
    )


# ============================================================
# 9. BUILD TARGETED TRAINING DATA
# ============================================================

def build_targeted_training_df(
    strategy
):

    assert strategy in [
        "Rotate",
        "Brightness"
    ]

    target_df = metadata[
        (
            metadata[
                "data_split"
            ]
            == "train"
        )
        &
        (
            metadata[
                "preprocessing_technique"
            ]
            == strategy
        )
    ].copy()


    # Ensure exact same original leaves
    original_ids = set(
        train_original[
            "original_image_id"
        ]
    )

    target_ids = set(
        target_df[
            "original_image_id"
        ]
    )

    assert (
        original_ids
        ==
        target_ids
    ), (
        f"{strategy}: "
        "target transformation does not "
        "match the 595 original training leaves."
    )


    combined = pd.concat(
        [
            train_original,
            target_df
        ],
        ignore_index=True
    )


    assert len(
        combined
    ) == 1190


    return combined


# ============================================================
# 10. DATASET
# ============================================================

class BrinjalDataset(
    Dataset
):

    def __init__(
        self,
        dataframe
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

    def __len__(
        self
    ):

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

        image = (
            Image.open(
                row["image_path"]
            )
            .convert("RGB")
        )

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        ).permute(
            2,
            0,
            1
        )

        mean = torch.tensor(
            MEAN,
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            STD,
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        image = (
            image
            - mean
        ) / std

        label = int(
            row["label"]
        )

        return (
            image,
            label,
            row["filename"]
        )


# ============================================================
# 11. MODEL
# ============================================================

def create_model():

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    in_features = (
        model.classifier[
            -1
        ].in_features
    )

    model.classifier[
        -1
    ] = nn.Linear(
        in_features,
        3
    )

    return model


# ============================================================
# 12. METRICS
# ============================================================

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
# 13. VALIDATION
# ============================================================

def evaluate_validation(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for (
            images,
            labels,
            _
        ) in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            bs = labels.size(
                0
            )

            total_loss += (
                loss.item()
                * bs
            )

            total_samples += bs

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics["loss"] = (
        total_loss
        /
        total_samples
    )

    return metrics


# ============================================================
# 14. TRAIN ONE TARGETED MODEL
# ============================================================

def train_targeted_model(
    strategy,
    seed
):

    print("\n")
    print("=" * 100)
    print(
        f"TRAINING A3 | "
        f"{strategy} | "
        f"seed={seed}"
    )
    print("=" * 100)


    seed_everything(
        seed
    )


    # --------------------------------------------------------
    # Training dataframe
    # --------------------------------------------------------

    train_df = (
        build_targeted_training_df(
            strategy
        )
    )

    print(
        f"Training images: "
        f"{len(train_df)}"
    )

    assert len(train_df) == 1190


    # --------------------------------------------------------
    # Validation
    # ORIGINAL validation ONLY
    # --------------------------------------------------------

    val_df = (
        val_original.copy()
    )


    train_dataset = (
        BrinjalDataset(
            train_df
        )
    )

    val_dataset = (
        BrinjalDataset(
            val_df
        )
    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model = model.to(
        DEVICE
    )


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
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


    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    best_state = None

    epochs_without_improvement = 0

    history = []


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        total_samples = 0

        train_true = []
        train_pred = []


        for (
            images,
            labels,
            _
        ) in train_loader:

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


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


            bs = labels.size(
                0
            )

            running_loss += (
                loss.item()
                * bs
            )

            total_samples += bs


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            train_true.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            train_pred.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )


        train_loss = (
            running_loss
            /
            total_samples
        )


        train_metrics = (
            calculate_metrics(
                train_true,
                train_pred
            )
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = (
            evaluate_validation(
                model,
                val_loader,
                criterion
            )
        )


        scheduler.step(
            val_metrics["loss"]
        )


        lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        history.append({

            "strategy":
                strategy,

            "seed":
                seed,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

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

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],

            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],

            "learning_rate":
                lr
        })


        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train F1={train_metrics['macro_f1']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val F1={val_metrics['macro_f1']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"LR={lr:.2e}"
        )


        # ----------------------------------------------------
        # Best checkpoint = ORIGINAL validation loss
        # ----------------------------------------------------

        if (
            val_metrics["loss"]
            <
            best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"Early stopping at epoch "
                f"{epoch}"
            )

            break


    # ========================================================
    # RESTORE BEST CHECKPOINT
    # ========================================================

    assert best_state is not None

    model.load_state_dict(
        best_state
    )


    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"mobilenetv3_small_"
        f"{strategy.lower()}_"
        f"seed{seed}_best.pth"
    )


    torch.save(
        {

            "model_name":
                MODEL_NAME,

            "architecture":
                "MobileNetV3-Small",

            "strategy":
                strategy,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "best_val_loss":
                best_val_loss,

            "state_dict":
                model.state_dict()

        },
        checkpoint_path
    )


    history_df = pd.DataFrame(
        history
    )


    history_path = os.path.join(
        HISTORY_DIR,
        f"mobilenetv3_small_"
        f"{strategy.lower()}_"
        f"seed{seed}_history.csv"
    )


    history_df.to_csv(
        history_path,
        index=False
    )


    print(
        f"\n✓ Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"✓ Best val loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"✓ Saved checkpoint:\n"
        f"{checkpoint_path}"
    )


    del model
    del train_loader
    del val_loader

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return (
        history_df,
        checkpoint_path
    )


# ============================================================
# 15. TRAIN 6 TARGETED MODELS
# ============================================================

all_histories = []

target_checkpoint_map = {}


for strategy in TARGETED_STRATEGIES:

    target_checkpoint_map[
        strategy
    ] = {}

    for seed in SEEDS:

        history_df, checkpoint_path = (
            train_targeted_model(
                strategy=strategy,
                seed=seed
            )
        )

        all_histories.append(
            history_df
        )

        target_checkpoint_map[
            strategy
        ][seed] = checkpoint_path


# ============================================================
# 16. SAVE COMBINED TRAINING HISTORY
# ============================================================

combined_history = pd.concat(
    all_histories,
    ignore_index=True
)

combined_history_path = os.path.join(
    A3_ROOT,
    "A3_targeted_training_history.csv"
)

combined_history.to_csv(
    combined_history_path,
    index=False
)


# ============================================================
# 17. TRAINING SUMMARY
# ============================================================

training_summary_rows = []


for strategy in TARGETED_STRATEGIES:

    for seed in SEEDS:

        subset = combined_history[
            (
                combined_history[
                    "strategy"
                ]
                == strategy
            )
            &
            (
                combined_history[
                    "seed"
                ]
                == seed
            )
        ]


        best_row = subset.loc[
            subset[
                "val_loss"
            ].idxmin()
        ]


        training_summary_rows.append({

            "strategy":
                strategy,

            "seed":
                seed,

            "best_epoch":
                int(
                    best_row[
                        "epoch"
                    ]
                ),

            "best_val_loss":
                best_row[
                    "val_loss"
                ],

            "best_val_accuracy":
                best_row[
                    "val_accuracy"
                ],

            "best_val_macro_f1":
                best_row[
                    "val_macro_f1"
                ],

            "best_val_balanced_accuracy":
                best_row[
                    "val_balanced_accuracy"
                ]
        })


training_summary = pd.DataFrame(
    training_summary_rows
)


training_summary_path = os.path.join(
    A3_ROOT,
    "A3_targeted_training_summary.csv"
)

training_summary.to_csv(
    training_summary_path,
    index=False
)


print("\n")
print("=" * 100)
print("A3 TRAINING SUMMARY")
print("=" * 100)

print(
    training_summary.to_string(
        index=False
    )
)


# ============================================================
# 18. UNSEEN TRANSFORMATION DEFINITIONS
# ============================================================

UNSEEN_CONDITIONS = {

    "Rotate": [
        "Rotate_-45",
        "Rotate_+45"
    ],

    "Brightness": [
        "Brightness_0.60",
        "Brightness_1.60"
    ]
}


# ============================================================
# 19. APPLY UNSEEN TRANSFORMATION
# ============================================================

def prepare_unseen_image(
    image,
    condition
):

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )


    if condition == "Rotate_-45":

        image = TF.rotate(
            image,
            angle=-45,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0
        )


    elif condition == "Rotate_+45":

        image = TF.rotate(
            image,
            angle=45,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0
        )


    elif condition == "Brightness_0.60":

        image = TF.adjust_brightness(
            image,
            brightness_factor=0.60
        )


    elif condition == "Brightness_1.60":

        image = TF.adjust_brightness(
            image,
            brightness_factor=1.60
        )


    else:

        raise ValueError(
            f"Unknown unseen condition: "
            f"{condition}"
        )


    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


    image = torch.from_numpy(
        image
    ).permute(
        2,
        0,
        1
    )


    mean = torch.tensor(
        MEAN,
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )


    std = torch.tensor(
        STD,
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )


    image = (
        image
        - mean
    ) / std


    return image


# ============================================================
# 20. A3 UNSEEN DATASET
# ============================================================

class UnseenSeverityDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        condition
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.condition = condition


    def __len__(
        self
    ):

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


        image = (
            Image.open(
                row[
                    "image_path"
                ]
            )
            .convert("RGB")
        )


        image = prepare_unseen_image(
            image,
            self.condition
        )


        label = int(
            row[
                "label"
            ]
        )


        return (
            image,
            label,
            row[
                "filename"
            ]
        )


# ============================================================
# 21. TEST INFERENCE
# ============================================================

def evaluate_model_on_condition(
    model,
    condition
):

    dataset = (
        UnseenSeverityDataset(
            test_original,
            condition
        )
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    y_true = []

    y_pred = []

    filenames = []


    with torch.no_grad():

        for (
            images,
            labels,
            names
        ) in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                images
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions
                .cpu()
                .numpy()
            )

            filenames.extend(
                list(names)
            )


    metrics = calculate_metrics(
        y_true,
        y_pred
    )


    prediction_df = pd.DataFrame({

        "filename":
            filenames,

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "correct":
            (
                np.array(y_true)
                ==
                np.array(y_pred)
            ).astype(int)

    })


    return (
        metrics,
        prediction_df
    )


# ============================================================
# 22. LOAD SAVED MODEL
# ============================================================

def load_model_from_checkpoint(
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )


    model = create_model()


    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )


    model = model.to(
        DEVICE
    )

    model.eval()


    return model


# ============================================================
# 23. EVALUATE BASELINE + TARGETED
# ============================================================

all_test_metrics = []

all_test_predictions = []


for strategy in TARGETED_STRATEGIES:

    print("\n")
    print("=" * 100)

    print(
        f"A3 INFERENCE — {strategy}"
    )

    print("=" * 100)


    conditions = (
        UNSEEN_CONDITIONS[
            strategy
        ]
    )


    for seed in SEEDS:

        print(
            f"\nSeed {seed}"
        )


        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        baseline_path = (
            BASELINE_CHECKPOINTS[
                seed
            ]
        )


        assert os.path.exists(
            baseline_path
        ), (
            f"Baseline checkpoint not found:\n"
            f"{baseline_path}"
        )


        print(
            "Loading BASELINE:"
        )

        print(
            baseline_path
        )


        baseline_model = (
            load_model_from_checkpoint(
                baseline_path
            )
        )


        # ----------------------------------------------------
        # TARGETED
        # ----------------------------------------------------

        targeted_path = (
            target_checkpoint_map[
                strategy
            ][
                seed
            ]
        )


        print(
            "Loading TARGETED:"
        )

        print(
            targeted_path
        )


        targeted_model = (
            load_model_from_checkpoint(
                targeted_path
            )
        )


        # ----------------------------------------------------
        # CONDITIONS
        # ----------------------------------------------------

        for condition in conditions:

            # ................................................
            # Baseline
            # ................................................

            (
                baseline_metrics,
                baseline_predictions
            ) = evaluate_model_on_condition(
                baseline_model,
                condition
            )


            all_test_metrics.append({

                "model_type":
                    "Baseline",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    baseline_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    baseline_metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    baseline_metrics[
                        "balanced_accuracy"
                    ]

            })


            baseline_predictions[
                "model_type"
            ] = "Baseline"

            baseline_predictions[
                "strategy"
            ] = strategy

            baseline_predictions[
                "seed"
            ] = seed

            baseline_predictions[
                "condition"
            ] = condition


            all_test_predictions.append(
                baseline_predictions
            )


            # ................................................
            # Targeted
            # ................................................

            (
                targeted_metrics,
                targeted_predictions
            ) = evaluate_model_on_condition(
                targeted_model,
                condition
            )


            all_test_metrics.append({

                "model_type":
                    "Targeted",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    targeted_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    targeted_metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    targeted_metrics[
                        "balanced_accuracy"
                    ]

            })


            targeted_predictions[
                "model_type"
            ] = "Targeted"

            targeted_predictions[
                "strategy"
            ] = strategy

            targeted_predictions[
                "seed"
            ] = seed

            targeted_predictions[
                "condition"
            ] = condition


            all_test_predictions.append(
                targeted_predictions
            )


            print(
                f"{condition:20s} | "
                f"Baseline F1="
                f"{baseline_metrics['macro_f1']:.4f} | "
                f"Targeted F1="
                f"{targeted_metrics['macro_f1']:.4f}"
            )


        del baseline_model
        del targeted_model

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 24. SAVE RAW TEST RESULTS
# ============================================================

test_metrics_df = pd.DataFrame(
    all_test_metrics
)

test_predictions_df = pd.concat(
    all_test_predictions,
    ignore_index=True
)


TEST_METRICS_PATH = os.path.join(
    A3_ROOT,
    "A3_unseen_test_metrics.csv"
)

TEST_PREDICTIONS_PATH = os.path.join(
    A3_ROOT,
    "A3_unseen_test_predictions.csv"
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
# 25. MEAN ± SD SUMMARY
# ============================================================

summary_df = (
    test_metrics_df
    .groupby(
        [
            "model_type",
            "strategy",
            "condition"
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


SUMMARY_PATH = os.path.join(
    A3_ROOT,
    "A3_unseen_test_summary.csv"
)


summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


print("\n")
print("=" * 100)
print("A3 TEST SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 26. TARGETED - BASELINE DELTAS
# ============================================================

delta_rows = []


for strategy in TARGETED_STRATEGIES:

    for condition in (
        UNSEEN_CONDITIONS[
            strategy
        ]
    ):

        base = test_metrics_df[
            (
                test_metrics_df[
                    "model_type"
                ]
                == "Baseline"
            )
            &
            (
                test_metrics_df[
                    "strategy"
                ]
                == strategy
            )
            &
            (
                test_metrics_df[
                    "condition"
                ]
                == condition
            )
        ].set_index(
            "seed"
        )


        target = test_metrics_df[
            (
                test_metrics_df[
                    "model_type"
                ]
                == "Targeted"
            )
            &
            (
                test_metrics_df[
                    "strategy"
                ]
                == strategy
            )
            &
            (
                test_metrics_df[
                    "condition"
                ]
                == condition
            )
        ].set_index(
            "seed"
        )


        delta_rows.append({

            "strategy":
                strategy,

            "condition":
                condition,

            "baseline_macro_f1_mean":
                base[
                    "macro_f1"
                ].mean(),

            "baseline_macro_f1_sd":
                base[
                    "macro_f1"
                ].std(),

            "targeted_macro_f1_mean":
                target[
                    "macro_f1"
                ].mean(),

            "targeted_macro_f1_sd":
                target[
                    "macro_f1"
                ].std(),

            "delta_macro_f1":
                (
                    target[
                        "macro_f1"
                    ]
                    -
                    base[
                        "macro_f1"
                    ]
                ).mean(),

            "baseline_accuracy_mean":
                base[
                    "accuracy"
                ].mean(),

            "targeted_accuracy_mean":
                target[
                    "accuracy"
                ].mean(),

            "delta_accuracy":
                (
                    target[
                        "accuracy"
                    ]
                    -
                    base[
                        "accuracy"
                    ]
                ).mean(),

            "baseline_balanced_accuracy_mean":
                base[
                    "balanced_accuracy"
                ].mean(),

            "targeted_balanced_accuracy_mean":
                target[
                    "balanced_accuracy"
                ].mean(),

            "delta_balanced_accuracy":
                (
                    target[
                        "balanced_accuracy"
                    ]
                    -
                    base[
                        "balanced_accuracy"
                    ]
                ).mean()
        })


delta_df = pd.DataFrame(
    delta_rows
)


DELTA_PATH = os.path.join(
    A3_ROOT,
    "A3_unseen_targeted_vs_baseline_delta.csv"
)


delta_df.to_csv(
    DELTA_PATH,
    index=False
)


print("\n")
print("=" * 100)
print("TARGETED vs BASELINE")
print("=" * 100)

print(
    delta_df.to_string(
        index=False
    )
)


# ============================================================
# 27. EXACT McNEMAR
# ============================================================

def exact_mcnemar(
    baseline_correct,
    targeted_correct
):

    baseline_correct = np.asarray(
        baseline_correct,
        dtype=bool
    )

    targeted_correct = np.asarray(
        targeted_correct,
        dtype=bool
    )


    b01 = np.sum(
        baseline_correct
        &
        (~targeted_correct)
    )


    b10 = np.sum(
        (~baseline_correct)
        &
        targeted_correct
    )


    discordant = (
        b01
        +
        b10
    )


    if discordant == 0:

        p_value = 1.0

    else:

        p_value = binomtest(
            k=min(
                b01,
                b10
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue


    return (
        int(b01),
        int(b10),
        int(discordant),
        float(p_value)
    )


# ============================================================
# 28. BENJAMINI-HOCHBERG
# ============================================================

def benjamini_hochberg(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    q_ranked = np.empty(
        n,
        dtype=float
    )

    previous = 1.0


    for i in range(
        n - 1,
        -1,
        -1
    ):

        rank = i + 1

        q = (
            ranked[i]
            *
            n
            /
            rank
        )

        q = min(
            q,
            previous
        )

        q_ranked[i] = q

        previous = q


    q_values = np.empty(
        n,
        dtype=float
    )

    q_values[
        order
    ] = q_ranked


    return q_values


# ============================================================
# 29. McNEMAR: TARGETED vs BASELINE
# ============================================================

mcnemar_rows = []


for strategy in TARGETED_STRATEGIES:

    for condition in (
        UNSEEN_CONDITIONS[
            strategy
        ]
    ):

        for seed in SEEDS:

            baseline = test_predictions_df[
                (
                    test_predictions_df[
                        "model_type"
                    ]
                    == "Baseline"
                )
                &
                (
                    test_predictions_df[
                        "strategy"
                    ]
                    == strategy
                )
                &
                (
                    test_predictions_df[
                        "seed"
                    ]
                    == seed
                )
                &
                (
                    test_predictions_df[
                        "condition"
                    ]
                    == condition
                )
            ]


            targeted = test_predictions_df[
                (
                    test_predictions_df[
                        "model_type"
                    ]
                    == "Targeted"
                )
                &
                (
                    test_predictions_df[
                        "strategy"
                    ]
                    == strategy
                )
                &
                (
                    test_predictions_df[
                        "seed"
                    ]
                    == seed
                )
                &
                (
                    test_predictions_df[
                        "condition"
                    ]
                    == condition
                )
            ]


            merged = (
                baseline[
                    [
                        "filename",
                        "correct"
                    ]
                ]
                .merge(
                    targeted[
                        [
                            "filename",
                            "correct"
                        ]
                    ],
                    on="filename",
                    suffixes=(
                        "_baseline",
                        "_targeted"
                    )
                )
            )


            assert len(
                merged
            ) == 128


            (
                b01,
                b10,
                discordant,
                p
            ) = exact_mcnemar(

                merged[
                    "correct_baseline"
                ].values,

                merged[
                    "correct_targeted"
                ].values
            )


            baseline_acc = (
                merged[
                    "correct_baseline"
                ].mean()
            )


            targeted_acc = (
                merged[
                    "correct_targeted"
                ].mean()
            )


            mcnemar_rows.append({

                "strategy":
                    strategy,

                "condition":
                    condition,

                "seed":
                    seed,

                "baseline_correct_targeted_wrong":
                    b01,

                "baseline_wrong_targeted_correct":
                    b10,

                "discordant_pairs":
                    discordant,

                "p_value":
                    p,

                "baseline_accuracy":
                    baseline_acc,

                "targeted_accuracy":
                    targeted_acc,

                "accuracy_delta":
                    (
                        targeted_acc
                        -
                        baseline_acc
                    )
            })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)


# 12 comparisons total
mcnemar_df[
    "fdr_q"
] = benjamini_hochberg(
    mcnemar_df[
        "p_value"
    ].values
)


mcnemar_df[
    "significant_fdr_0.05"
] = (
    mcnemar_df[
        "fdr_q"
    ]
    < 0.05
)


MCNEMAR_PATH = os.path.join(
    A3_ROOT,
    "A3_targeted_vs_baseline_mcnemar.csv"
)


mcnemar_df.to_csv(
    MCNEMAR_PATH,
    index=False
)


# ============================================================
# 30. FINAL SANITY CHECKS
# ============================================================

print("\n")
print("=" * 100)
print("FINAL A3 SANITY CHECK")
print("=" * 100)


# 6 training runs
assert len(
    target_checkpoint_map
) == 2


assert sum(
    len(v)
    for v in target_checkpoint_map.values()
) == 6


# 12 comparison tests
assert len(
    mcnemar_df
) == 12


# 128 images each
prediction_group_counts = (
    test_predictions_df
    .groupby(
        [
            "model_type",
            "strategy",
            "seed",
            "condition"
        ]
    )
    .size()
)


assert (
    prediction_group_counts
    == 128
).all()


print(
    "✓ 6 targeted training runs completed."
)

print(
    "✓ 3 Rotate seeds + 3 Brightness seeds."
)

print(
    "✓ Original validation loss used for checkpoint selection."
)

print(
    "✓ Baseline checkpoints are the existing MNV3 checkpoints."
)

print(
    "✓ Unseen Rotate: -45° and +45°."
)

print(
    "✓ Unseen Brightness: 0.60 and 1.60."
)

print(
    "✓ 12 paired McNemar tests completed."
)

print(
    "✓ BH-FDR correction completed."
)

print(
    "✓ A3 completed successfully."
)


# ============================================================
# 31. OUTPUT FILES
# ============================================================

print("\n")
print("=" * 100)
print("A3 OUTPUT FILES")
print("=" * 100)

print(
    "\nTraining history:"
)

print(
    combined_history_path
)

print(
    "\nTraining summary:"
)

print(
    training_summary_path
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
    "\nMean ± SD summary:"
)

print(
    SUMMARY_PATH
)

print(
    "\nTargeted vs baseline delta:"
)

print(
    DELTA_PATH
)

print(
    "\nMcNemar + BH-FDR:"
)

print(
    MCNEMAR_PATH
)

print("\n")
print("TARGET CHECKPOINTS:")

for strategy in TARGETED_STRATEGIES:

    for seed in SEEDS:

        print(
            f"{strategy:10s} | "
            f"seed={seed} | "
            f"{target_checkpoint_map[strategy][seed]}"
        )

# %% [Cell 79]
# ============================================================
# A3 — INFERENCE ONLY
# NO RETRAINING
#
# Baseline MNV3:
#   seed 42, 1337, 2026
#
# Targeted MNV3:
#   Rotate      seed 42, 1337, 2026
#   Brightness  seed 42, 1337, 2026
#
# Unseen conditions:
#   Rotate: -45°, +45°
#   Brightness: 0.60, 1.60
#
# Outputs:
#   - test metrics
#   - mean ± SD
#   - targeted vs baseline deltas
#   - exact McNemar
#   - BH-FDR
# ============================================================

import sympy
import sympy.printing

import os
import gc
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

from scipy.stats import binomtest


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

A3_ROOT = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "A3_unseen_severity"
)

A3_CHECKPOINT_DIR = os.path.join(
    A3_ROOT,
    "checkpoints"
)

os.makedirs(
    A3_ROOT,
    exist_ok=True
)

MODEL_NAME = "MobileNetV3-Small"

SEEDS = [42, 1337, 2026]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
]

CLASS_TO_IDX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

BASELINE_CHECKPOINTS = {

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
}

TARGETED_CHECKPOINTS = {

    "Rotate": {

        42:
            os.path.join(
                A3_CHECKPOINT_DIR,
                "mobilenetv3_small_rotate_seed42_best.pth"
            ),

        1337:
            os.path.join(
                A3_CHECKPOINT_DIR,
                "mobilenetv3_small_rotate_seed1337_best.pth"
            ),

        2026:
            os.path.join(
                A3_CHECKPOINT_DIR,
                "mobilenetv3_small_rotate_seed2026_best.pth"
            )
    },

    "Brightness": {

        42:
            os.path.join(
                A3_CHECKPOINT_DIR,
                "mobilenetv3_small_brightness_seed42_best.pth"
            ),

        1337:
            os.path.join(
                A3_CHECKPOINT_DIR,
                "mobilenetv3_small_brightness_seed1337_best.pth"
            ),

        2026:
            os.path.join(
                A3_CHECKPOINT_DIR,
                "mobilenetv3_small_brightness_seed2026_best.pth"
            )
    }
}

UNSEEN_CONDITIONS = {

    "Rotate": [
        "Rotate_-45",
        "Rotate_+45"
    ],

    "Brightness": [
        "Brightness_0.60",
        "Brightness_1.60"
    ]
}


print("=" * 100)
print("A3 — INFERENCE ONLY")
print("=" * 100)

print("Device:", DEVICE)
print("Seeds :", SEEDS)


# ============================================================
# 2. VERIFY CHECKPOINTS
# ============================================================

print("\nChecking checkpoints...")

for seed in SEEDS:

    assert os.path.exists(
        BASELINE_CHECKPOINTS[seed]
    ), (
        f"Missing baseline checkpoint:\n"
        f"{BASELINE_CHECKPOINTS[seed]}"
    )


for strategy in TARGETED_CHECKPOINTS:

    for seed in SEEDS:

        path = TARGETED_CHECKPOINTS[
            strategy
        ][seed]

        assert os.path.exists(
            path
        ), (
            f"Missing targeted checkpoint:\n"
            f"{path}"
        )

        print(
            f"✓ {strategy:10s} seed {seed}"
        )


print("✓ All required checkpoints found.")

# ============================================================
# 3. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

metadata["label"] = (
    metadata["class_label"]
    .map(CLASS_TO_IDX)
)

assert metadata[
    "label"
].notna().all()

metadata["label"] = (
    metadata["label"]
    .astype(int)
)


# ============================================================
# 4. IMAGE INDEX
# ============================================================

image_index = {}

for base_dir in [
    os.path.join(ROOT, "Raw"),
    os.path.join(ROOT, "Augmented")
]:

    if not os.path.exists(base_dir):
        continue

    for dirpath, _, filenames in os.walk(
        base_dir
    ):

        for filename in filenames:

            if not filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                )
            ):
                continue

            full_path = os.path.join(
                dirpath,
                filename
            )

            image_index[filename] = full_path
            image_index[filename.lower()] = full_path


def resolve_image_path(filename):

    filename = str(filename)

    if filename in image_index:
        return image_index[filename]

    lower = filename.lower()

    if lower in image_index:
        return image_index[lower]

    basename = os.path.basename(
        filename
    )

    if basename in image_index:
        return image_index[basename]

    basename_lower = basename.lower()

    if basename_lower in image_index:
        return image_index[basename_lower]

    return None


metadata["image_path"] = (
    metadata["filename"]
    .apply(resolve_image_path)
)

assert metadata[
    "image_path"
].notna().all()

print(
    "\n✓ All image paths resolved."
)


# ============================================================
# 5. ORIGINAL TEST LEAVES
# ============================================================

test_original = metadata[
    (
        metadata["data_split"]
        == "test"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        == "Original"
    )
].copy()

test_original = (
    test_original
    .reset_index(drop=True)
)

assert len(test_original) == 128

print(
    "Original test leaves:",
    len(test_original)
)


# ============================================================
# 6. MODEL
# ============================================================

def create_model():

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    in_features = (
        model.classifier[
            -1
        ].in_features
    )

    model.classifier[
        -1
    ] = nn.Linear(
        in_features,
        3
    )

    return model


# ============================================================
# 7. ROBUST CHECKPOINT LOADER
# ============================================================

def load_model_from_checkpoint(
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model = create_model()

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# 8. UNSEEN IMAGE TRANSFORMATION
# ============================================================

def prepare_unseen_image(
    image,
    condition
):

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    if condition == "Rotate_-45":

         image = TF.rotate(
             image,
             angle=-45,
             interpolation=TF.InterpolationMode.BILINEAR,
             fill=0
         )

    elif condition == "Rotate_+45":

         image = TF.rotate(
             image,
             angle=45,
             interpolation=TF.InterpolationMode.BILINEAR,
             fill=0
         )

    elif condition == "Brightness_0.60":

         image = TF.adjust_brightness(
             image,
             brightness_factor=0.60
         )

    elif condition == "Brightness_1.60":

         image = TF.adjust_brightness(
             image,
             brightness_factor=1.60
         )

    else:

         raise ValueError(
             f"Unknown condition: {condition}"
         )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image = torch.from_numpy(
        image
    ).permute(
        2,
        0,
        1
    )

    mean = torch.tensor(
        MEAN,
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        STD,
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    image = (
        image - mean
    ) / std

    return image


# ============================================================
# 9. DATASET
# ============================================================

class UnseenSeverityDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        condition
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.condition = condition

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[
            idx
        ]

        image = (
            Image.open(
                row["image_path"]
            )
            .convert("RGB")
        )

        image = prepare_unseen_image(
            image,
            self.condition
        )

        label = int(
            row["label"]
        )

        return (
            image,
            label,
            row["filename"]
        )


# ============================================================
# 10. METRICS
# ============================================================

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
# 11. INFERENCE
# ============================================================

def evaluate_model(
    model,
    condition
):

    dataset = (
        UnseenSeverityDataset(
            test_original,
            condition
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    y_true = []
    y_pred = []
    filenames = []

    with torch.no_grad():

        for (
            images,
            labels,
            names
        ) in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

            filenames.extend(
                list(names)
            )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    prediction_df = pd.DataFrame({

        "filename":
            filenames,

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "correct":
            (
                np.array(y_true)
                ==
                np.array(y_pred)
            ).astype(int)
    })

    return (
        metrics,
        prediction_df
    )


# ============================================================
# 12. RUN BASELINE + TARGETED
# ============================================================

all_metrics = []
all_predictions = []

for strategy in [
    "Rotate",
    "Brightness"
]:

    print("\n")
    print("=" * 100)
    print(
        f"A3 INFERENCE — {strategy}"
    )
    print("=" * 100)

    conditions = (
        UNSEEN_CONDITIONS[
            strategy
        ]
    )

    for seed in SEEDS:

        print(
            f"\nSeed {seed}"
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_path = (
            BASELINE_CHECKPOINTS[
                seed
            ]
        )

        print(
            "Loading BASELINE:"
        )

        print(
            baseline_path
        )

        baseline_model = (
            load_model_from_checkpoint(
                baseline_path
            )
        )

        # ----------------------------------------------------
        # Targeted
        # ----------------------------------------------------

        targeted_path = (
            TARGETED_CHECKPOINTS[
                strategy
            ][seed]
        )

        print(
            "Loading TARGETED:"
        )

        print(
            targeted_path
        )

        targeted_model = (
            load_model_from_checkpoint(
                targeted_path
            )
        )

        # ----------------------------------------------------
        # Conditions
        # ----------------------------------------------------

        for condition in conditions:

            baseline_metrics, baseline_pred = (
                evaluate_model(
                    baseline_model,
                    condition
                )
            )

            targeted_metrics, targeted_pred = (
                evaluate_model(
                    targeted_model,
                    condition
                )
            )

            print(
                f"{condition:20s} | "
                f"Baseline F1="
                f"{baseline_metrics["macro_f1"]: .4f} | "
                f"Targeted F1="
                f"{targeted_metrics["macro_f1"]: .4f} Custom"
            ) #Custom

            # Baseline metrics
            all_metrics.append({

                "model_type":
                    "Baseline",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    baseline_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    baseline_metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    baseline_metrics[
                        "balanced_accuracy"
                    ]
            })

            baseline_pred[
                "model_type"
            ] = "Baseline"

            baseline_pred[
                "strategy"
            ] = strategy

            baseline_pred[
                "seed"
            ] = seed

            baseline_pred[
                "condition"
            ] = condition

            all_predictions.append(
                baseline_pred
            )

            # Targeted metrics
            all_metrics.append({

                "model_type":
                    "Targeted",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    targeted_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    targeted_metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    targeted_metrics[
                        "balanced_accuracy"
                    ]
            })

            targeted_pred[
                "model_type"
            ] = "Targeted"

            targeted_pred[
                "strategy"
            ] = strategy

            targeted_pred[
                "seed"
            ] = seed

            targeted_pred[
                "condition"
            ] = condition

            all_predictions.append(
                targeted_pred
            )

        del baseline_model
        del targeted_model

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 13. SAVE RAW RESULTS
# ============================================================

test_metrics_df = pd.DataFrame(
    all_metrics
)

test_predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

TEST_METRICS_PATH = os.path.join(
    A3_ROOT,
    "A3_unseen_test_metrics.csv"
)

TEST_PREDICTIONS_PATH = os.path.join(
    A3_ROOT,
    "A3_unseen_test_predictions.csv"
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
# 14. MEAN ± SD
# ============================================================

summary_df = (
    test_metrics_df
    .groupby(
        [
            "model_type",
            "strategy",
            "condition"
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

SUMMARY_PATH = os.path.join(
    A3_ROOT,
    "A3_unseen_test_summary.csv"
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


print("\n")
print("=" * 100)
print("A3 TEST SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 15. TARGETED - BASELINE DELTAS
# ============================================================

delta_rows = []

for strategy in [
    "Rotate",
    "Brightness"
]:

    for condition in (
        UNSEEN_CONDITIONS[
            strategy
        ]
    ):

        base = test_metrics_df[
            (
                test_metrics_df[
                    "model_type"
                ]
                == "Baseline"
            )
            &
            (
                test_metrics_df[
                    "strategy"
                ]
                == strategy
            )
            &
            (
                test_metrics_df[
                    "condition"
                ]
                == condition
            )
        ].set_index(
            "seed"
        )

        target = test_metrics_df[
            (
                test_metrics_df[
                    "model_type"
                ]
                == "Targeted"
            )
            &
            (
                test_metrics_df[
                    "strategy"
                ]
                == strategy
            )
            &
            (
                test_metrics_df[
                    "condition"
                ]
                == condition
            )
        ].set_index(
            "seed"
        )

        delta_rows.append({

            "strategy":
                strategy,

            "condition":
                condition,

            "baseline_macro_f1_mean":
                base[
                    "macro_f1"
                ].mean(),

            "baseline_macro_f1_sd":
                base[
                    "macro_f1"
                ].std(),

            "targeted_macro_f1_mean":
                target[
                    "macro_f1"
                ].mean(),

            "targeted_macro_f1_sd":
                target[
                    "macro_f1"
                ].std(),

            "delta_macro_f1":
                (
                    target[
                        "macro_f1"
                    ]
                    -
                    base[
                        "macro_f1"
                    ]
                ).mean(),

            "baseline_accuracy_mean":
                base[
                    "accuracy"
                ].mean(),

            "targeted_accuracy_mean":
                target[
                    "accuracy"
                ].mean(),

            "delta_accuracy":
                (
                    target[
                        "accuracy"
                    ]
                    -
                    base[
                        "accuracy"
                    ]
                ).mean(),

            "baseline_balanced_accuracy_mean":
                base[
                    "balanced_accuracy"
                ].mean(),

            "targeted_balanced_accuracy_mean":
                target[
                    "balanced_accuracy"
                ].mean(),

            "delta_balanced_accuracy":
                (
                    target[
                        "balanced_accuracy"
                    ]
                    -
                    base[
                        "balanced_accuracy"
                    ]
                ).mean()
        })


delta_df = pd.DataFrame(
    delta_rows
)

DELTA_PATH = os.path.join(
    A3_ROOT,
    "A3_targeted_vs_baseline_delta.csv"
)

delta_df.to_csv(
    DELTA_PATH,
    index=False
)

print("\n")
print("=" * 100)
print("TARGETED vs BASELINE")
print("=" * 100)

print(
    delta_df.to_string(
        index=False
    )
)


# ============================================================
# 16. EXACT McNEMAR
# ============================================================

def exact_mcnemar(
    baseline_correct,
    targeted_correct
):

    baseline_correct = np.asarray(
        baseline_correct,
        dtype=bool
    )

    targeted_correct = np.asarray(
        targeted_correct,
        dtype=bool
    )

    b01 = np.sum(
        baseline_correct
        &
        (~targeted_correct)
    )

    b10 = np.sum(
        (~baseline_correct)
        &
        targeted_correct
    )

    discordant = (
        b01 + b10
    )

    if discordant == 0:

        p_value = 1.0

    else:

        p_value = binomtest(
            k=min(
                b01,
                b10
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return (
        b01,
        b10,
        discordant,
        p_value
    )


# ============================================================
# 17. BH-FDR
# ============================================================

def benjamini_hochberg(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    q_ranked = np.empty(
        n,
        dtype=float
    )

    previous = 1.0

    for i in range(
        n - 1,
        -1,
        -1
    ):

        rank = i + 1

        q = (
            ranked[i]
            * n
            / rank
        )

        q = min(
            q,
            previous
        )

        q_ranked[i] = q

        previous = q

    q_values = np.empty(
        n,
        dtype=float
    )

    q_values[
        order
    ] = q_ranked

    return q_values


# ============================================================
# 18. McNEMAR TESTS
# ============================================================

mcnemar_rows = []

for strategy in [
    "Rotate",
    "Brightness"
]:

    for condition in (
        UNSEEN_CONDITIONS[
            strategy
        ]
    ):

        for seed in SEEDS:

            baseline = test_predictions_df[
                (
                    test_predictions_df[
                        "model_type"
                    ]
                    == "Baseline"
                )
                &
                (
                    test_predictions_df[
                        "strategy"
                    ]
                    == strategy
                )
                &
                (
                    test_predictions_df[
                        "seed"
                    ]
                    == seed
                )
                &
                (
                    test_predictions_df[
                        "condition"
                    ]
                    == condition
                )
            ]

            targeted = test_predictions_df[
                (
                    test_predictions_df[
                        "model_type"
                    ]
                    == "Targeted"
                )
                &
                (
                    test_predictions_df[
                        "strategy"
                    ]
                    == strategy
                )
                &
                (
                    test_predictions_df[
                        "seed"
                    ]
                    == seed
                )
                &
                (
                    test_predictions_df[
                        "condition"
                    ]
                    == condition
                )
            ]

            merged = (
                baseline[
                    [
                        "filename",
                        "correct"
                    ]
                ]
                .merge(
                    targeted[
                        [
                            "filename",
                            "correct"
                        ]
                    ],
                    on="filename",
                    suffixes=(
                        "_baseline",
                        "_targeted"
                    )
                )
            )

            assert len(
                merged
            ) == 128

            (
                b01,
                b10,
                discordant,
                p_value
            ) = exact_mcnemar(

                merged[
                    "correct_baseline"
                ].values,

                merged[
                    "correct_targeted"
                ].values
            )

            baseline_acc = (
                merged[
                    "correct_baseline"
                ].mean()
            )

            targeted_acc = (
                merged[
                    "correct_targeted"
                ].mean()
            )

            mcnemar_rows.append({

                "strategy":
                    strategy,

                "condition":
                    condition,

                "seed":
                    seed,

                "baseline_correct_targeted_wrong":
                    b01,

                "baseline_wrong_targeted_correct":
                    b10,

                "discordant_pairs":
                    discordant,

                "p_value":
                    p_value,

                "baseline_accuracy":
                    baseline_acc,

                "targeted_accuracy":
                    targeted_acc,

                "accuracy_delta":
                    (
                        targeted_acc
                        -
                        baseline_acc
                    )
            })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)

# 12 total tests
mcnemar_df["fdr_q"] = (
    benjamini_hochberg(
        mcnemar_df["p_value"].values
    )
)

mcnemar_df[
    "significant_fdr_0.05"
] = (
    mcnemar_df[
        "fdr_q"
    ]
    < 0.05
)


MCNEMAR_PATH = os.path.join(
    A3_ROOT,
    "A3_targeted_vs_baseline_mcnemar.csv"
)

mcnemar_df.to_csv(
    MCNEMAR_PATH,
    index=False
)


# ============================================================
# 19. FINAL CHECK
# ============================================================

group_counts = (
    test_predictions_df
    .groupby(
        [
            "model_type",
            "strategy",
            "seed",
            "condition"
        ]
    )
    .size()
)

assert (
    group_counts == 128
).all()

assert len(
    mcnemar_df
) == 12


print("\n")
print("=" * 100)
print("A3 INFERENCE COMPLETE")
print("=" * 100)

print(
    "✓ No training performed."
)

print(
    "✓ Baseline + targeted checkpoints loaded."
)

print(
    "✓ 128 test leaves per condition."
)

print(
    "✓ 12 McNemar comparisons completed."
)

print(
    "\nSaved files:"
)

print(
    TEST_METRICS_PATH
)

print(
    TEST_PREDICTIONS_PATH
)

print(
    SUMMARY_PATH
)

print(
    DELTA_PATH
)

print(
    MCNEMAR_PATH
)


# %% [Cell 80]
from google.colab import drive
drive.mount('/content/drive')

# %% [Cell 81]
# ============================================================
# FIX SYMPY -> PYTORCH IMPORT
# ============================================================

import sys
import sympy

print("SymPy version:", sympy.__version__)
print("SymPy path   :", sympy.__file__)

# Force-load the printing submodule
import sympy.printing
import sympy.printing.str
import sympy.printing.precedence

print(
    "Has sympy.printing:",
    hasattr(sympy, "printing")
)

# Now import torch/torchvision
import torch

print(
    "PyTorch version:",
    torch.__version__
)

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

print(
    "✓ torchvision imported successfully"
)

# %% [Cell 82]
# ============================================================
# FIX COLAB SYMPY ENVIRONMENT
# ============================================================

import sys
import subprocess

print("Python:", sys.version)

# Upgrade/reinstall a compatible SymPy version
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--upgrade",
    "sympy>=1.13.3,<1.15"
])

print("\n✓ SymPy installation repaired.")
print("IMPORTANT: Restart the Colab runtime now.")

# %% [Cell 83]
import sympy
print("SymPy version:", sympy.__version__)
print("SymPy path:", sympy.__file__)
print("Has printing:", hasattr(sympy, "printing"))

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

print("✓ torchvision.models imported successfully")

# %% [Cell 84]
# ============================================================
# A3 — CONTINUE FROM INFERENCE
# NO TRAINING
# NO RETRAINING
# ============================================================

import os
import gc
import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

from scipy.stats import binomtest


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

A3_ROOT = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "A3_unseen_severity"
)

A3_CHECKPOINT_DIR = os.path.join(
    A3_ROOT,
    "checkpoints"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

DEVICE = torch.device("cpu")

SEEDS = [42, 1337, 2026]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(3, 1, 1)


# ============================================================
# 2. CHECKPOINTS — EXACT PATHS
# ============================================================

BASELINE_CHECKPOINTS = {
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
}

TARGETED_CHECKPOINTS = {

    "Rotate": {
        42: os.path.join(
            A3_CHECKPOINT_DIR,
            "mobilenetv3_small_rotate_seed42_best.pth"
        ),
        1337: os.path.join(
            A3_CHECKPOINT_DIR,
            "mobilenetv3_small_rotate_seed1337_best.pth"
        ),
        2026: os.path.join(
            A3_CHECKPOINT_DIR,
            "mobilenetv3_small_rotate_seed2026_best.pth"
        )
    },

    "Brightness": {
        42: os.path.join(
            A3_CHECKPOINT_DIR,
            "mobilenetv3_small_brightness_seed42_best.pth"
        ),
        1337: os.path.join(
            A3_CHECKPOINT_DIR,
            "mobilenetv3_small_brightness_seed1337_best.pth"
        ),
        2026: os.path.join(
            A3_CHECKPOINT_DIR,
            "mobilenetv3_small_brightness_seed2026_best.pth"
        )
    }
}


# ============================================================
# 3. VERIFY
# ============================================================

print("=" * 100)
print("A3 — INFERENCE CONTINUATION")
print("=" * 100)

for seed in SEEDS:

    assert os.path.exists(
        BASELINE_CHECKPOINTS[seed]
    ), BASELINE_CHECKPOINTS[seed]


for strategy in TARGETED_CHECKPOINTS:

    for seed in SEEDS:

        path = TARGETED_CHECKPOINTS[
            strategy
        ][seed]

        assert os.path.exists(
            path
        ), path

        print(
            f"✓ {strategy} seed {seed}"
        )

print("\n✓ All 9 required checkpoints found.")


# ============================================================
# 4. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

CLASS_TO_IDX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

metadata["label"] = (
    metadata["class_label"]
    .map(CLASS_TO_IDX)
)

assert metadata["label"].notna().all()

metadata["label"] = (
    metadata["label"].astype(int)
)


# ============================================================
# 5. IMAGE INDEX
# ============================================================

print("\nBuilding image index...")

image_index = {}

for base_dir in [
    os.path.join(ROOT, "Raw"),
    os.path.join(ROOT, "Augmented")
]:

    if not os.path.exists(base_dir):
        continue

    for dirpath, _, filenames in os.walk(
        base_dir
    ):

        for filename in filenames:

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            full_path = os.path.join(
                dirpath,
                filename
            )

            image_index[filename] = full_path
            image_index[filename.lower()] = full_path


def resolve_path(filename):

    filename = str(filename)

    if filename in image_index:
        return image_index[filename]

    if filename.lower() in image_index:
        return image_index[filename.lower()]

    basename = os.path.basename(filename)

    if basename in image_index:
        return image_index[basename]

    if basename.lower() in image_index:
        return image_index[basename.lower()]

    return None


metadata["image_path"] = (
    metadata["filename"]
    .apply(resolve_path)
)

assert metadata["image_path"].notna().all()

print(
    "✓ All image paths resolved."
)


# ============================================================
# 6. ORIGINAL TEST LEAVES
# ============================================================

test_df = metadata[
    (
        metadata["data_split"] == "test"
    )
    &
    (
        metadata["preprocessing_technique"]
        == "Original"
    )
].copy()

test_df = test_df.reset_index(
    drop=True
)

assert len(test_df) == 128

print(
    f"Original test leaves: {len(test_df)}"
)


# ============================================================
# 7. MODEL
# ============================================================

def create_model():

    model = mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        3
    )

    return model


# ============================================================
# 8. CHECKPOINT LOADER
# ============================================================

def load_checkpoint(path):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model = create_model()

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# 9. UNSEEN TRANSFORMATIONS
# ============================================================

def transform_image(
    image,
    condition
):

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    if condition == "Rotate_-45":

        image = image.rotate(
            -45,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

    elif condition == "Rotate_+45":

        image = image.rotate(
            45,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

    elif condition == "Brightness_0.60":

        image = ImageEnhance.Brightness(
            image
        ).enhance(0.60)

    elif condition == "Brightness_1.60":

        image = ImageEnhance.Brightness(
            image
        ).enhance(1.60)

    else:

        raise ValueError(
            f"Unknown condition: {condition}"
        )

    arr = (
        np.asarray(
            image,
            dtype=np.float32
        ) / 255.0
    )

    tensor = torch.from_numpy(
        arr
    ).permute(2, 0, 1)

    tensor = (
        tensor - MEAN
    ) / STD

    return tensor


# ============================================================
# 10. DATASET
# ============================================================

class A3Dataset(Dataset):

    def __init__(
        self,
        dataframe,
        condition
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.condition = condition

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = (
            Image.open(
                row["image_path"]
            )
            .convert("RGB")
        )

        image = transform_image(
            image,
            self.condition
        )

        label = int(
            row["label"]
        )

        return (
            image,
            label,
            row["filename"]
        )


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate(
    model,
    condition
):

    dataset = A3Dataset(
        test_df,
        condition
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    y_true = []
    y_pred = []
    filenames = []

    with torch.no_grad():

        for images, labels, names in loader:

            outputs = model(
                images
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                preds.numpy()
            )

            filenames.extend(
                list(names)
            )

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "macro_f1": f1_score(
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

    pred_df = pd.DataFrame({
        "filename": filenames,
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": (
            np.array(y_true)
            ==
            np.array(y_pred)
        ).astype(int)
    })

    return metrics, pred_df


# ============================================================
# 12. RUN INFERENCE
# ============================================================

all_metrics = []
all_predictions = []

for strategy in [
    "Rotate",
    "Brightness"
]:

    if strategy == "Rotate":

        conditions = [
            "Rotate_-45",
            "Rotate_+45"
        ]

    else:

        conditions = [
            "Brightness_0.60",
            "Brightness_1.60"
        ]

    print("\n")
    print("=" * 100)
    print(
        f"A3 — {strategy}"
    )
    print("=" * 100)

    for seed in SEEDS:

        print(
            f"\nSeed {seed}"
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline = load_checkpoint(
            BASELINE_CHECKPOINTS[seed]
        )

        # ----------------------------------------------------
        # Targeted
        # ----------------------------------------------------

        targeted = load_checkpoint(
            TARGETED_CHECKPOINTS[
                strategy
            ][seed]
        )

        for condition in conditions:

            base_metrics, base_pred = evaluate(
                baseline,
                condition
            )

            target_metrics, target_pred = evaluate(
                targeted,
                condition
            )

            print(
                f"{condition:20s} | "
                f"Baseline F1={base_metrics['macro_f1']:.4f} | "
                f"Targeted F1={target_metrics['macro_f1']:.4f}"
            )

            all_metrics.append({

                "model_type":
                    "Baseline",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    base_metrics["accuracy"],

                "macro_f1":
                    base_metrics["macro_f1"],

                "balanced_accuracy":
                    base_metrics[
                        "balanced_accuracy"
                    ]
            })

            base_pred["model_type"] = "Baseline"
            base_pred["strategy"] = strategy
            base_pred["seed"] = seed
            base_pred["condition"] = condition

            all_predictions.append(
                base_pred
            )


            all_metrics.append({

                "model_type":
                    "Targeted",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    target_metrics["accuracy"],

                "macro_f1":
                    target_metrics["macro_f1"],

                "balanced_accuracy":
                    target_metrics[
                        "balanced_accuracy"
                    ]
            })

            target_pred["model_type"] = "Targeted"
            target_pred["strategy"] = strategy
            target_pred["seed"] = seed
            target_pred["condition"] = condition

            all_predictions.append(
                target_pred
            )

        del baseline
        del targeted

        gc.collect()


# ============================================================
# 13. SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

metrics_path = os.path.join(
    A3_ROOT,
    "A3_unseen_test_metrics.csv"
)

predictions_path = os.path.join(
    A3_ROOT,
    "A3_unseen_test_predictions.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 14. MEAN ± SD
# ============================================================

summary_df = (
    metrics_df
    .groupby(
        [
            "model_type",
            "strategy",
            "condition"
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


summary_path = os.path.join(
    A3_ROOT,
    "A3_unseen_test_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print("\n")
print("=" * 100)
print("A3 TEST SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 15. DELTAS: TARGETED - BASELINE
# ============================================================

delta_rows = []

for strategy in [
    "Rotate",
    "Brightness"
]:

    if strategy == "Rotate":

        conditions = [
            "Rotate_-45",
            "Rotate_+45"
        ]

    else:

        conditions = [
            "Brightness_0.60",
            "Brightness_1.60"
        ]

    for condition in conditions:

        base = metrics_df[
            (
                metrics_df[
                    "model_type"
                ] == "Baseline"
            )
            &
            (
                metrics_df[
                    "strategy"
                ] == strategy
            )
            &
            (
                metrics_df[
                    "condition"
                ] == condition
            )
        ].set_index("seed")

        target = metrics_df[
            (
                metrics_df[
                    "model_type"
                ] == "Targeted"
            )
            &
            (
                metrics_df[
                    "strategy"
                ] == strategy
            )
            &
            (
                metrics_df[
                    "condition"
                ] == condition
            )
        ].set_index("seed")

        delta_rows.append({

            "strategy":
                strategy,

            "condition":
                condition,

            "baseline_macro_f1_mean":
                base["macro_f1"].mean(),

            "baseline_macro_f1_sd":
                base["macro_f1"].std(),

            "targeted_macro_f1_mean":
                target["macro_f1"].mean(),

            "targeted_macro_f1_sd":
                target["macro_f1"].std(),

            "delta_macro_f1":
                (
                    target["macro_f1"]
                    -
                    base["macro_f1"]
                ).mean(),

            "baseline_accuracy_mean":
                base["accuracy"].mean(),

            "targeted_accuracy_mean":
                target["accuracy"].mean(),

            "delta_accuracy":
                (
                    target["accuracy"]
                    -
                    base["accuracy"]
                ).mean(),

            "baseline_balanced_accuracy_mean":
                base[
                    "balanced_accuracy"
                ].mean(),

            "targeted_balanced_accuracy_mean":
                target[
                    "balanced_accuracy"
                ].mean(),

            "delta_balanced_accuracy":
                (
                    target[
                        "balanced_accuracy"
                    ]
                    -
                    base[
                        "balanced_accuracy"
                    ]
                ).mean()
        })


delta_df = pd.DataFrame(
    delta_rows
)

delta_path = os.path.join(
    A3_ROOT,
    "A3_unseen_targeted_vs_baseline_delta.csv"
)

delta_df.to_csv(
    delta_path,
    index=False
)


print("\n")
print("=" * 100)
print("A3 TARGETED vs BASELINE")
print("=" * 100)

print(
    delta_df.to_string(
        index=False
    )
)


# ============================================================
# 16. McNEMAR
# ============================================================

def exact_mcnemar(
    baseline_correct,
    targeted_correct
):

    baseline_correct = np.asarray(
        baseline_correct,
        dtype=bool
    )

    targeted_correct = np.asarray(
        targeted_correct,
        dtype=bool
    )

    b01 = np.sum(
        baseline_correct
        &
        (~targeted_correct)
    )

    b10 = np.sum(
        (~baseline_correct)
        &
        targeted_correct
    )

    discordant = (
        b01 + b10
    )

    if discordant == 0:

        p = 1.0

    else:

        p = binomtest(
            k=min(
                b01,
                b10
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return (
        int(b01),
        int(b10),
        int(discordant),
        float(p)
    )


def benjamini_hochberg(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    q_ranked = np.empty(
        n
    )

    previous = 1.0

    for i in range(
        n - 1,
        -1,
        -1
    ):

        rank = i + 1

        q = (
            ranked[i]
            * n
            / rank
        )

        q = min(
            q,
            previous
        )

        q_ranked[i] = q

        previous = q

    q_values = np.empty(
        n
    )

    q_values[
        order
    ] = q_ranked

    return q_values


mcnemar_rows = []

for strategy in [
    "Rotate",
    "Brightness"
]:

    if strategy == "Rotate":

        conditions = [
            "Rotate_-45",
            "Rotate_+45"
        ]

    else:

        conditions = [
            "Brightness_0.60",
            "Brightness_1.60"
        ]

    for condition in conditions:

        for seed in SEEDS:

            base = predictions_df[
                (
                    predictions_df[
                        "model_type"
                    ] == "Baseline"
                )
                &
                (
                    predictions_df[
                        "strategy"
                    ] == strategy
                )
                &
                (
                    predictions_df[
                        "seed"
                    ] == seed
                )
                &
                (
                    predictions_df[
                        "condition"
                    ] == condition
                )
            ]

            target = predictions_df[
                (
                    predictions_df[
                        "model_type"
                    ] == "Targeted"
                )
                &
                (
                    predictions_df[
                        "strategy"
                    ] == strategy
                )
                &
                (
                    predictions_df[
                        "seed"
                    ] == seed
                )
                &
                (
                    predictions_df[
                        "condition"
                    ] == condition
                )
            ]

            merged = (
                base[
                    [
                        "filename",
                        "correct"
                    ]
                ]
                .merge(
                    target[
                        [
                            "filename",
                            "correct"
                        ]
                    ],
                    on="filename",
                    suffixes=(
                        "_baseline",
                        "_targeted"
                    )
                )
            )

            assert len(
                merged
            ) == 128

            (
                b01,
                b10,
                discordant,
                p
            ) = exact_mcnemar(

                merged[
                    "correct_baseline"
                ].values,

                merged[
                    "correct_targeted"
                ].values
            )

            base_acc = (
                merged[
                    "correct_baseline"
                ].mean()
            )

            target_acc = (
                merged[
                    "correct_targeted"
                ].mean()
            )

            mcnemar_rows.append({

                "strategy":
                    strategy,

                "condition":
                    condition,

                "seed":
                    seed,

                "baseline_correct_targeted_wrong":
                    b01,

                "baseline_wrong_targeted_correct":
                    b10,

                "discordant_pairs":
                    discordant,

                "p_value":
                    p,

                "baseline_accuracy":
                    base_acc,

                "targeted_accuracy":
                    target_acc,

                "accuracy_delta":
                    target_acc - base_acc
            })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)

mcnemar_df["fdr_q"] = (
    benjamini_hochberg(
        mcnemar_df[
            "p_value"
        ].values
    )
)

mcnemar_df[
    "significant_fdr_0.05"
] = (
    mcnemar_df[
        "fdr_q"
    ] < 0.05
)


mcnemar_path = os.path.join(
    A3_ROOT,
    "A3_targeted_vs_baseline_mcnemar.csv"
)

mcnemar_df.to_csv(
    mcnemar_path,
    index=False
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("A3 McNEMAR RESULTS")
print("=" * 100)

print(
    mcnemar_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("A3 FINISHED SUCCESSFULLY")
print("=" * 100)

print(
    "Metrics:",
    metrics_path
)

print(
    "Summary:",
    summary_path
)

print(
    "Delta:",
    delta_path
)

print(
    "McNemar:",
    mcnemar_path
)

print(
    "\n✓ No training was performed."
)

print(
    "✓ 6 targeted checkpoints evaluated."
)

print(
    "✓ 12 paired McNemar tests completed."
)

# %% [Cell 85]
# ============================================================
# A3 / TABLE 9 PIPELINE RECONCILIATION
# NO TRAINING
# BASELINE MNV3-SMALL ONLY
#
# Compare:
#   Pipeline A = transform -> resize -> normalize
#   Pipeline B = resize -> transform -> normalize
#
# Target Table 9 baseline values:
#   Rotate -45°   = 57.98
#   Rotate +45°   = 60.29
#   Brightness 0.60x = 85.58
#   Brightness 1.60x = 68.91
# ============================================================

import os
import gc
import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision.models import mobilenet_v3_small

from sklearn.metrics import f1_score


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

SEEDS = [42, 1337, 2026]

IMAGE_SIZE = 224
BATCH_SIZE = 32

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(3, 1, 1)


BASELINE_CHECKPOINTS = {

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
}


TARGET_TABLE9 = {

    "Rotate_-45": 57.98,
    "Rotate_+45": 60.29,
    "Brightness_0.60": 85.58,
    "Brightness_1.60": 68.91
}


# ============================================================
# 2. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

CLASS_TO_IDX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

metadata["label"] = (
    metadata["class_label"]
    .map(CLASS_TO_IDX)
)

assert metadata["label"].notna().all()

metadata["label"] = (
    metadata["label"].astype(int)
)


# ============================================================
# 3. IMAGE INDEX
# ============================================================

image_index = {}

for base_dir in [
    os.path.join(ROOT, "Raw"),
    os.path.join(ROOT, "Augmented")
]:

    if not os.path.exists(base_dir):
        continue

    for dirpath, _, filenames in os.walk(
        base_dir
    ):

        for filename in filenames:

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            path = os.path.join(
                dirpath,
                filename
            )

            image_index[filename] = path
            image_index[filename.lower()] = path


def resolve_path(filename):

    filename = str(filename)

    if filename in image_index:
        return image_index[filename]

    if filename.lower() in image_index:
        return image_index[filename.lower()]

    basename = os.path.basename(
        filename
    )

    if basename in image_index:
        return image_index[basename]

    if basename.lower() in image_index:
        return image_index[basename.lower()]

    return None


metadata["image_path"] = (
    metadata["filename"]
    .apply(resolve_path)
)

assert metadata["image_path"].notna().all()


# ============================================================
# 4. ORIGINAL TEST LEAVES
# ============================================================

test_df = metadata[
    (
        metadata["data_split"] == "test"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        == "Original"
    )
].copy()

test_df = test_df.reset_index(
    drop=True
)

assert len(test_df) == 128


# ============================================================
# 5. MODEL
# ============================================================

def create_model():

    model = mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model.classifier[-1]
        .in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        3
    )

    return model


def load_model(path):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model = create_model()

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and
        "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# 6. TRANSFORMATIONS
# ============================================================

def apply_transform(
    image,
    condition
):

    if condition == "Rotate_-45":

        image = image.rotate(
            -45,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

    elif condition == "Rotate_+45":

        image = image.rotate(
            45,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

    elif condition == "Brightness_0.60":

        image = ImageEnhance.Brightness(
            image
        ).enhance(
            0.60
        )

    elif condition == "Brightness_1.60":

        image = ImageEnhance.Brightness(
            image
        ).enhance(
            1.60
        )

    return image


# ============================================================
# 7. DATASET
# ============================================================

class ReconDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        condition,
        mode
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.condition = condition
        self.mode = mode

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        image = (
            Image.open(
                row["image_path"]
            )
            .convert("RGB")
        )

        # ----------------------------------------------------
        # PIPELINE A
        # transform -> resize
        # ----------------------------------------------------

        if self.mode == "transform_then_resize":

            image = apply_transform(
                image,
                self.condition
            )

            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.BILINEAR
            )

        # ----------------------------------------------------
        # PIPELINE B
        # resize -> transform
        # ----------------------------------------------------

        elif self.mode == "resize_then_transform":

            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                Image.Resampling.BILINEAR
            )

            image = apply_transform(
                image,
                self.condition
            )

        else:

            raise ValueError(
                self.mode
            )

        arr = (
            np.asarray(
                image,
                dtype=np.float32
            )
            /
            255.0
        )

        tensor = torch.from_numpy(
            arr
        ).permute(
            2, 0, 1
        )

        tensor = (
            tensor - MEAN
        ) / STD

        return (
            tensor,
            int(row["label"])
        )


# ============================================================
# 8. EVALUATE
# ============================================================

def evaluate(
    model,
    condition,
    mode
):

    dataset = ReconDataset(
        test_df,
        condition,
        mode
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(
                images
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                preds.numpy()
            )

    return (
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
        * 100.0
    )


# ============================================================
# 9. RUN BOTH PIPELINES
# ============================================================

conditions = [
    "Rotate_-45",
    "Rotate_+45",
    "Brightness_0.60",
    "Brightness_1.60"
]

rows = []


for mode in [
    "transform_then_resize",
    "resize_then_transform"
]:

    print("\n")
    print("=" * 100)
    print(
        f"PIPELINE: {mode}"
    )
    print("=" * 100)

    for condition in conditions:

        seed_values = []

        for seed in SEEDS:

            model = load_model(
                BASELINE_CHECKPOINTS[
                    seed
                ]
            )

            f1 = evaluate(
                model,
                condition,
                mode
            )

            seed_values.append(
                f1
            )

            del model
            gc.collect()


        mean_f1 = np.mean(
            seed_values
        )

        sd_f1 = np.std(
            seed_values,
            ddof=1
        )

        target = TARGET_TABLE9[
            condition
        ]

        diff = (
            mean_f1
            -
            target
        )

        print(
            f"{condition:20s} | "
            f"Mean={mean_f1:.2f} | "
            f"SD={sd_f1:.2f} | "
            f"Table9={target:.2f} | "
            f"Difference={diff:+.2f}"
        )

        rows.append({

            "pipeline":
                mode,

            "condition":
                condition,

            "mean_macro_f1":
                mean_f1,

            "sd_macro_f1":
                sd_f1,

            "table9_macro_f1":
                target,

            "difference_vs_table9":
                diff
        })


# ============================================================
# 10. FINAL COMPARISON
# ============================================================

results = pd.DataFrame(
    rows
)

print("\n")
print("=" * 100)
print("FINAL PIPELINE COMPARISON")
print("=" * 100)

print(
    results.to_string(
        index=False
    )
)

print("\n")
print(
    "Average absolute difference from Table 9:"
)

for pipeline in results[
    "pipeline"
].unique():

    subset = results[
        results["pipeline"]
        == pipeline
    ]

    mae = np.mean(
        np.abs(
            subset[
                "difference_vs_table9"
            ]
        )
    )

    print(
        f"{pipeline}: {mae:.3f} Macro-F1 points"
    )

# %% [Cell 86]
# ============================================================
# A3 FINAL — TRANSFORM THEN RESIZE
# NO TRAINING
#
# Pipeline:
#   Original 512x512
#      -> unseen transformation
#      -> resize to 224x224
#      -> ImageNet normalization
#
# This is the pipeline matching Table 9.
# ============================================================

import os
import gc
import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision.models import mobilenet_v3_small

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

from scipy.stats import binomtest


# ============================================================
# CONFIG
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

A3_ROOT = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "A3_unseen_severity"
)

CHECKPOINT_DIR = os.path.join(
    A3_ROOT,
    "checkpoints"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

SEEDS = [42, 1337, 2026]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(3, 1, 1)


# ============================================================
# BASELINE CHECKPOINTS
# ============================================================

BASELINE_CHECKPOINTS = {

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
}


# ============================================================
# TARGETED A3 CHECKPOINTS
# ============================================================

TARGETED_CHECKPOINTS = {

    "Rotate": {

        42:
            os.path.join(
                CHECKPOINT_DIR,
                "mobilenetv3_small_rotate_seed42_best.pth"
            ),

        1337:
            os.path.join(
                CHECKPOINT_DIR,
                "mobilenetv3_small_rotate_seed1337_best.pth"
            ),

        2026:
            os.path.join(
                CHECKPOINT_DIR,
                "mobilenetv3_small_rotate_seed2026_best.pth"
            )
    },

    "Brightness": {

        42:
            os.path.join(
                CHECKPOINT_DIR,
                "mobilenetv3_small_brightness_seed42_best.pth"
            ),

        1337:
            os.path.join(
                CHECKPOINT_DIR,
                "mobilenetv3_small_brightness_seed1337_best.pth"
            ),

        2026:
            os.path.join(
                CHECKPOINT_DIR,
                "mobilenetv3_small_brightness_seed2026_best.pth"
            )
    }
}


CONDITIONS = {

    "Rotate": [
        "Rotate_-45",
        "Rotate_+45"
    ],

    "Brightness": [
        "Brightness_0.60",
        "Brightness_1.60"
    ]
}


# ============================================================
# 1. VERIFY CHECKPOINTS
# ============================================================

print("=" * 100)
print("A3 FINAL — TRANSFORM THEN RESIZE")
print("=" * 100)

for seed in SEEDS:

    assert os.path.exists(
        BASELINE_CHECKPOINTS[seed]
    )

for strategy in TARGETED_CHECKPOINTS:

    for seed in SEEDS:

        path = TARGETED_CHECKPOINTS[
            strategy
        ][seed]

        assert os.path.exists(
            path
        )

        print(
            f"✓ {strategy:10s} seed={seed}"
        )

print(
    "\n✓ All checkpoints found."
)


# ============================================================
# 2. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

CLASS_TO_IDX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

metadata["label"] = (
    metadata["class_label"]
    .map(CLASS_TO_IDX)
)

assert metadata["label"].notna().all()

metadata["label"] = (
    metadata["label"]
    .astype(int)
)


# ============================================================
# 3. IMAGE INDEX
# ============================================================

image_index = {}

for base_dir in [
    os.path.join(ROOT, "Raw"),
    os.path.join(ROOT, "Augmented")
]:

    if not os.path.exists(
        base_dir
    ):
        continue

    for dirpath, _, filenames in os.walk(
        base_dir
    ):

        for filename in filenames:

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            path = os.path.join(
                dirpath,
                filename
            )

            image_index[
                filename
            ] = path

            image_index[
                filename.lower()
            ] = path


def resolve_path(
    filename
):

    filename = str(
        filename
    )

    if filename in image_index:

        return image_index[
            filename
        ]

    if filename.lower() in image_index:

        return image_index[
            filename.lower()
        ]

    basename = os.path.basename(
        filename
    )

    if basename in image_index:

        return image_index[
            basename
        ]

    if basename.lower() in image_index:

        return image_index[
            basename.lower()
        ]

    return None


metadata["image_path"] = (
    metadata["filename"]
    .apply(resolve_path)
)

assert metadata[
    "image_path"
].notna().all()


# ============================================================
# 4. ORIGINAL TEST LEAVES
# ============================================================

test_df = metadata[
    (
        metadata["data_split"]
        == "test"
    )
    &
    (
        metadata[
            "preprocessing_technique"
        ]
        == "Original"
    )
].copy()

test_df = (
    test_df
    .reset_index(drop=True)
)

assert len(test_df) == 128

print(
    f"Original test leaves: "
    f"{len(test_df)}"
)


# ============================================================
# 5. MODEL
# ============================================================

def create_model():

    model = mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model.classifier[
            -1
        ].in_features
    )

    model.classifier[
        -1
    ] = nn.Linear(
        in_features,
        3
    )

    return model


# ============================================================
# 6. CHECKPOINT LOADER
# ============================================================

def load_model(
    path
):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model = create_model()

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict
        )
        and
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# 7. UNSEEN TRANSFORMATION
#
# IMPORTANT:
# Transformation is applied BEFORE resize.
# ============================================================

def apply_unseen_transform(
    image,
    condition
):

    if condition == "Rotate_-45":

        image = image.rotate(
            -45,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

    elif condition == "Rotate_+45":

        image = image.rotate(
            45,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0)
        )

    elif condition == "Brightness_0.60":

        image = ImageEnhance.Brightness(
            image
        ).enhance(
            0.60
        )

    elif condition == "Brightness_1.60":

        image = ImageEnhance.Brightness(
            image
        ).enhance(
            1.60
        )

    else:

        raise ValueError(
            f"Unknown condition: {condition}"
        )

    # --------------------------------------------------------
    # CRITICAL:
    # resize AFTER transformation
    # --------------------------------------------------------

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    return image


# ============================================================
# 8. DATASET
# ============================================================

class A3Dataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        condition
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.condition = condition


    def __len__(
        self
    ):

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

        image = (
            Image.open(
                row["image_path"]
            )
            .convert("RGB")
        )

        image = (
            apply_unseen_transform(
                image,
                self.condition
            )
        )

        arr = (
            np.asarray(
                image,
                dtype=np.float32
            )
            /
            255.0
        )

        image = torch.from_numpy(
            arr
        ).permute(
            2, 0, 1
        )

        image = (
            image - MEAN
        ) / STD

        label = int(
            row["label"]
        )

        return (
            image,
            label,
            row["filename"]
        )


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate(
    model,
    condition
):

    dataset = A3Dataset(
        test_df,
        condition
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    y_true = []
    y_pred = []
    filenames = []

    with torch.no_grad():

        for (
            images,
            labels,
            names
        ) in loader:

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.numpy()
            )

            filenames.extend(
                list(names)
            )

    metrics = {

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

    pred_df = pd.DataFrame({

        "filename":
            filenames,

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "correct":
            (
                np.array(y_true)
                ==
                np.array(y_pred)
            ).astype(int)
    })

    return (
        metrics,
        pred_df
    )


# ============================================================
# 10. RUN
# ============================================================

all_metrics = []
all_predictions = []


for strategy in [
    "Rotate",
    "Brightness"
]:

    print("\n")
    print("=" * 100)
    print(
        f"A3 FINAL — {strategy}"
    )
    print("=" * 100)

    for seed in SEEDS:

        print(
            f"\nSeed {seed}"
        )

        baseline_model = load_model(
            BASELINE_CHECKPOINTS[
                seed
            ]
        )

        targeted_model = load_model(
            TARGETED_CHECKPOINTS[
                strategy
            ][seed]
        )

        for condition in (
            CONDITIONS[
                strategy
            ]
        ):

            baseline_metrics, baseline_pred = (
                evaluate(
                    baseline_model,
                    condition
                )
            )

            targeted_metrics, targeted_pred = (
                evaluate(
                    targeted_model,
                    condition
                )
            )

            print(
                f"{condition:20s} | "
                f"Baseline F1="
                f"{baseline_metrics['macro_f1']*100:.2f} | "
                f"Targeted F1="
                f"{targeted_metrics['macro_f1']*100:.2f}"
            )

            all_metrics.append({

                "model_type":
                    "Baseline",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    baseline_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    baseline_metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    baseline_metrics[
                        "balanced_accuracy"
                    ]
            })


            baseline_pred[
                "model_type"
            ] = "Baseline"

            baseline_pred[
                "strategy"
            ] = strategy

            baseline_pred[
                "seed"
            ] = seed

            baseline_pred[
                "condition"
            ] = condition

            all_predictions.append(
                baseline_pred
            )


            all_metrics.append({

                "model_type":
                    "Targeted",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "condition":
                    condition,

                "accuracy":
                    targeted_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    targeted_metrics[
                        "macro_f1"
                    ],

                "balanced_accuracy":
                    targeted_metrics[
                        "balanced_accuracy"
                    ]
            })


            targeted_pred[
                "model_type"
            ] = "Targeted"

            targeted_pred[
                "strategy"
            ] = strategy

            targeted_pred[
                "seed"
            ] = seed

            targeted_pred[
                "condition"
            ] = condition

            all_predictions.append(
                targeted_pred
            )


        del baseline_model
        del targeted_model

        gc.collect()


# ============================================================
# 11. METRICS
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# ============================================================
# 12. MEAN ± SD
# ============================================================

summary_df = (
    metrics_df
    .groupby(
        [
            "model_type",
            "strategy",
            "condition"
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


# ============================================================
# 13. TARGETED - BASELINE
# ============================================================

delta_rows = []

for strategy in [
    "Rotate",
    "Brightness"
]:

    for condition in (
        CONDITIONS[
            strategy
        ]
    ):

        base = metrics_df[
            (
                metrics_df[
                    "model_type"
                ]
                == "Baseline"
            )
            &
            (
                metrics_df[
                    "strategy"
                ]
                == strategy
            )
            &
            (
                metrics_df[
                    "condition"
                ]
                == condition
            )
        ].set_index(
            "seed"
        )

        target = metrics_df[
            (
                metrics_df[
                    "model_type"
                ]
                == "Targeted"
            )
            &
            (
                metrics_df[
                    "strategy"
                ]
                == strategy
            )
            &
            (
                metrics_df[
                    "condition"
                ]
                == condition
            )
        ].set_index(
            "seed"
        )

        delta_rows.append({

            "strategy":
                strategy,

            "condition":
                condition,

            "baseline_macro_f1_mean":
                base[
                    "macro_f1"
                ].mean(),

            "baseline_macro_f1_sd":
                base[
                    "macro_f1"
                ].std(),

            "targeted_macro_f1_mean":
                target[
                    "macro_f1"
                ].mean(),

            "targeted_macro_f1_sd":
                target[
                    "macro_f1"
                ].std(),

            "delta_macro_f1":
                (
                    target[
                        "macro_f1"
                    ]
                    -
                    base[
                        "macro_f1"
                    ]
                ).mean()
        })


delta_df = pd.DataFrame(
    delta_rows
)


# ============================================================
# 14. MCNEMAR
# ============================================================

def exact_mcnemar(
    baseline_correct,
    targeted_correct
):

    baseline_correct = np.asarray(
        baseline_correct,
        dtype=bool
    )

    targeted_correct = np.asarray(
        targeted_correct,
        dtype=bool
    )

    b01 = np.sum(
        baseline_correct
        &
        (~targeted_correct)
    )

    b10 = np.sum(
        (~baseline_correct)
        &
        targeted_correct
    )

    discordant = (
        b01 + b10
    )

    if discordant == 0:

        p = 1.0

    else:

        p = binomtest(
            min(
                b01,
                b10
            ),
            discordant,
            0.5,
            alternative="two-sided"
        ).pvalue

    return (
        int(b01),
        int(b10),
        int(discordant),
        float(p)
    )


def bh_fdr(
    p_values
):

    p_values = np.asarray(
        p_values
    )

    n = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    q_ranked = np.empty(
        n
    )

    previous = 1.0

    for i in range(
        n - 1,
        -1,
        -1
    ):

        rank = i + 1

        q = (
            ranked[i]
            * n
            / rank
        )

        q = min(
            q,
            previous
        )

        q_ranked[i] = q

        previous = q

    q_values = np.empty(
        n
    )

    q_values[
        order
    ] = q_ranked

    return q_values


mcnemar_rows = []


for strategy in [
    "Rotate",
    "Brightness"
]:

    for condition in (
        CONDITIONS[
            strategy
        ]
    ):

        for seed in SEEDS:

            base = predictions_df[
                (
                    predictions_df[
                        "model_type"
                    ]
                    == "Baseline"
                )
                &
                (
                    predictions_df[
                        "strategy"
                    ]
                    == strategy
                )
                &
                (
                    predictions_df[
                        "seed"
                    ]
                    == seed
                )
                &
                (
                    predictions_df[
                        "condition"
                    ]
                    == condition
                )
            ]

            target = predictions_df[
                (
                    predictions_df[
                        "model_type"
                    ]
                    == "Targeted"
                )
                &
                (
                    predictions_df[
                        "strategy"
                    ]
                    == strategy
                )
                &
                (
                    predictions_df[
                        "seed"
                    ]
                    == seed
                )
                &
                (
                    predictions_df[
                        "condition"
                    ]
                    == condition
                )
            ]

            merged = (
                base[
                    [
                        "filename",
                        "correct"
                    ]
                ]
                .merge(
                    target[
                        [
                            "filename",
                            "correct"
                        ]
                    ],
                    on="filename",
                    suffixes=(
                        "_baseline",
                        "_targeted"
                    )
                )
            )

            assert len(
                merged
            ) == 128

            (
                b01,
                b10,
                discordant,
                p
            ) = exact_mcnemar(

                merged[
                    "correct_baseline"
                ].values,

                merged[
                    "correct_targeted"
                ].values
            )

            mcnemar_rows.append({

                "strategy":
                    strategy,

                "condition":
                    condition,

                "seed":
                    seed,

                "baseline_correct_targeted_wrong":
                    b01,

                "baseline_wrong_targeted_correct":
                    b10,

                "discordant_pairs":
                    discordant,

                "p_value":
                    p
            })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)

mcnemar_df[
    "fdr_q"
] = bh_fdr(
    mcnemar_df[
        "p_value"
    ].values
)

mcnemar_df[
    "significant_fdr_0.05"
] = (
    mcnemar_df[
        "fdr_q"
    ]
    < 0.05
)


# ============================================================
# 15. SAVE
# ============================================================

metrics_path = os.path.join(
    A3_ROOT,
    "A3_FINAL_transform_then_resize_metrics.csv"
)

predictions_path = os.path.join(
    A3_ROOT,
    "A3_FINAL_transform_then_resize_predictions.csv"
)

summary_path = os.path.join(
    A3_ROOT,
    "A3_FINAL_transform_then_resize_summary.csv"
)

delta_path = os.path.join(
    A3_ROOT,
    "A3_FINAL_transform_then_resize_delta.csv"
)

mcnemar_path = os.path.join(
    A3_ROOT,
    "A3_FINAL_transform_then_resize_mcnemar.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

delta_df.to_csv(
    delta_path,
    index=False
)

mcnemar_df.to_csv(
    mcnemar_path,
    index=False
)


# ============================================================
# 16. OUTPUT
# ============================================================

print("\n")
print("=" * 100)
print("A3 FINAL SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 100)
print("A3 TARGETED - BASELINE")
print("=" * 100)

print(
    delta_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 100)
print("A3 MCNEMAR")
print("=" * 100)

print(
    mcnemar_df.to_string(
        index=False
    )
)

print("\n✓ A3 final inference completed.")
print("✓ Transform → Resize → Normalize.")
print("✓ No retraining.")

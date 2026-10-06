
# %% [Cell 48]
# ============================================================
# PHASE 6.1 — AUGMENTATION-TO-ROBUSTNESS PILOT
# MobileNetV3-Small | Seed 42
#
# Strategies:
#   1) Baseline
#   2) + Grayscale
#   3) + Rotate
#   4) + Brightness
#
# Checkpoint selection ONLY on Original Validation Loss
# ============================================================

import os
import gc
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
MAX_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
BATCH_SIZE = 32

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

TRAIN_ROOT = Path(
    "/content/brinjal_original"
)

TRAIN_AUG_ROOT = Path(
    "/content/brinjal_train_augmentation"
)

VAL_ROOT = Path(
    "/content/brinjal_val_robustness"
)

# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(SEED)

# ============================================================
# 3. IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# 4. GENERIC DATASET
# ============================================================

class LocalBrinjalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        root_dir,
        transform=None
    ):

        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = self.root_dir / row["filename"]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_IDX[
            row["class_label"]
        ]

        return image, label


# ============================================================
# 5. TRAIN DATASETS FOR EACH STRATEGY
# ============================================================

train_original_df = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Original")
].copy()

train_gray_df = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Grayscale")
].copy()

train_rotate_df = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Rotate")
].copy()

train_brightness_df = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Brightness")
].copy()


# Baseline
baseline_train_df = train_original_df.copy()

# Original + Grayscale
gray_train_df = pd.concat(
    [
        train_original_df,
        train_gray_df
    ],
    ignore_index=True
)

# Original + Rotate
rotate_train_df = pd.concat(
    [
        train_original_df,
        train_rotate_df
    ],
    ignore_index=True
)

# Original + Brightness
brightness_train_df = pd.concat(
    [
        train_original_df,
        train_brightness_df
    ],
    ignore_index=True
)


TRAIN_STRATEGIES = {
    "Baseline": (
        baseline_train_df,
        TRAIN_ROOT
    ),

    "Grayscale": (
        gray_train_df,
        TRAIN_ROOT
    ),

    "Rotate": (
        rotate_train_df,
        TRAIN_ROOT
    ),

    "Brightness": (
        brightness_train_df,
        TRAIN_ROOT
    )
}

# ============================================================
# 6. FIX PATHS FOR AUGMENTED TRAINING FILES
# ============================================================
#
# Original rows use /content/brinjal_original
# Augmented rows use /content/brinjal_train_augmentation
#
# We split each strategy internally.

def create_training_datasets(strategy_name):

    if strategy_name == "Baseline":

        dataset = LocalBrinjalDataset(
            baseline_train_df,
            TRAIN_ROOT,
            transform=IMAGE_TRANSFORM
        )

        return dataset

    if strategy_name == "Grayscale":

        original_dataset = LocalBrinjalDataset(
            train_original_df,
            TRAIN_ROOT,
            transform=IMAGE_TRANSFORM
        )

        augmented_dataset = LocalBrinjalDataset(
            train_gray_df,
            TRAIN_AUG_ROOT,
            transform=IMAGE_TRANSFORM
        )

        return torch.utils.data.ConcatDataset(
            [
                original_dataset,
                augmented_dataset
            ]
        )

    if strategy_name == "Rotate":

        original_dataset = LocalBrinjalDataset(
            train_original_df,
            TRAIN_ROOT,
            transform=IMAGE_TRANSFORM
        )

        augmented_dataset = LocalBrinjalDataset(
            train_rotate_df,
            TRAIN_AUG_ROOT,
            transform=IMAGE_TRANSFORM
        )

        return torch.utils.data.ConcatDataset(
            [
                original_dataset,
                augmented_dataset
            ]
        )

    if strategy_name == "Brightness":

        original_dataset = LocalBrinjalDataset(
            train_original_df,
            TRAIN_ROOT,
            transform=IMAGE_TRANSFORM
        )

        augmented_dataset = LocalBrinjalDataset(
            train_brightness_df,
            TRAIN_AUG_ROOT,
            transform=IMAGE_TRANSFORM
        )

        return torch.utils.data.ConcatDataset(
            [
                original_dataset,
                augmented_dataset
            ]
        )

    raise ValueError(strategy_name)


# ============================================================
# 7. VALIDATION DATASETS
# ============================================================

VAL_TRANSFORMS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]

val_datasets = {}
val_loaders = {}

for transformation in VAL_TRANSFORMS:

    subset = df[
        (df["data_split"] == "val") &
        (
            df["preprocessing_technique"]
            == transformation
        )
    ].copy()

    dataset = LocalBrinjalDataset(
        subset,
        VAL_ROOT,
        transform=IMAGE_TRANSFORM
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    val_datasets[transformation] = dataset
    val_loaders[transformation] = loader


# ============================================================
# 8. MODEL FACTORY
# ============================================================

def build_mobilenetv3():

    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT
    )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    return model.to(device)


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    y_true = []
    y_pred = []

    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

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

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            total_samples += images.size(0)

            preds = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {

        "loss":
            total_loss / total_samples,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro"
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            )
    }


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_strategy(
    strategy_name,
    train_dataset
):

    print("\n" + "=" * 85)
    print(
        f"TRAINING STRATEGY: "
        f"{strategy_name}"
    )
    print("=" * 85)

    seed_everything(SEED)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    model = build_mobilenetv3()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    best_original_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_original_val_acc = 0.0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        epoch_start = time.time()

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        loss_sum = 0.0
        correct = 0
        total = 0

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

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += (
                loss.item() *
                images.size(0)
            )

            preds = outputs.argmax(
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        # ====================================================
        # ORIGINAL VALIDATION
        # ====================================================

        original_metrics = evaluate_model(
            model,
            val_loaders["Original"]
        )

        scheduler.step(
            original_metrics["loss"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({

            "strategy":
                strategy_name,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_acc":
                train_acc,

            "original_val_loss":
                original_metrics["loss"],

            "original_val_acc":
                original_metrics["accuracy"],

            "original_val_f1":
                original_metrics["macro_f1"],

            "lr":
                current_lr
        })

        # ====================================================
        # CHECKPOINT BY ORIGINAL VAL LOSS ONLY
        # ====================================================

        if (
            original_metrics["loss"]
            <
            best_original_val_loss
        ):

            best_original_val_loss = (
                original_metrics["loss"]
            )

            best_original_val_acc = (
                original_metrics["accuracy"]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            checkpoint_path = (
                OUTPUT_DIR /
                f"augmentation_pilot_"
                f"{strategy_name.lower()}_seed42.pth"
            )

            torch.save(
                {
                    "model_state_dict":
                        best_state,

                    "strategy":
                        strategy_name,

                    "seed":
                        SEED,

                    "best_epoch":
                        best_epoch,

                    "best_original_val_loss":
                        best_original_val_loss,

                    "best_original_val_acc":
                        best_original_val_acc,

                    "class_to_idx":
                        CLASS_TO_IDX
                },
                checkpoint_path
            )

            status = "✅ BEST"

        else:

            patience_counter += 1
            status = ""

        elapsed = (
            time.time() -
            epoch_start
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"{elapsed:.1f}s | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Orig Val F1 "
            f"{original_metrics['macro_f1']:.4f} | "
            f"Orig Val Acc "
            f"{original_metrics['accuracy']:.4f} | "
            f"LR {current_lr:.6f} "
            f"{status}"
        )

        if patience_counter >= PATIENCE:

            print(
                f"\n⏹ Early stopping at epoch "
                f"{epoch}"
            )

            break

    # Restore best state
    model.load_state_dict(
        best_state
    )

    # ========================================================
    # Final validation across all 4 conditions
    # ========================================================

    final_results = []

    original_f1 = None

    for transformation in VAL_TRANSFORMS:

        metrics = evaluate_model(
            model,
            val_loaders[transformation]
        )

        if transformation == "Original":

            original_f1 = (
                metrics["macro_f1"]
            )

        delta_f1 = (
            metrics["macro_f1"]
            -
            original_f1
            if original_f1 is not None
            else 0.0
        )

        final_results.append({

            "strategy":
                strategy_name,

            "seed":
                SEED,

            "transformation":
                transformation,

            "accuracy":
                metrics["accuracy"],

            "macro_f1":
                metrics["macro_f1"],

            "balanced_accuracy":
                metrics["balanced_accuracy"],

            "delta_f1_vs_original":
                delta_f1,

            "best_epoch":
                best_epoch
        })

    print("\n" + "-" * 85)
    print(
        f"FINAL VALIDATION RESULTS — "
        f"{strategy_name}"
    )
    print("-" * 85)

    display(
        pd.DataFrame(final_results)
        .style.format({
            "accuracy": "{:.4f}",
            "macro_f1": "{:.4f}",
            "balanced_accuracy": "{:.4f}",
            "delta_f1_vs_original": "{:+.4f}"
        })
    )

    return (
        model,
        history,
        final_results
    )


# ============================================================
# 11. RUN FOUR PILOT STRATEGIES
# ============================================================

all_final_results = []
all_histories = []

trained_models = {}

for strategy_name in [
    "Baseline",
    "Grayscale",
    "Rotate",
    "Brightness"
]:

    train_dataset = create_training_datasets(
        strategy_name
    )

    print(
        f"\nTraining samples for "
        f"{strategy_name}: "
        f"{len(train_dataset)}"
    )

    model, history, final_results = (
        train_strategy(
            strategy_name,
            train_dataset
        )
    )

    trained_models[strategy_name] = model

    all_histories.extend(
        history
    )

    all_final_results.extend(
        final_results
    )

# ============================================================
# 12. Save results
# ============================================================

final_results_df = pd.DataFrame(
    all_final_results
)

history_df = pd.DataFrame(
    all_histories
)

final_results_path = (
    OUTPUT_DIR /
    "augmentation_pilot_validation_results.csv"
)

history_path = (
    OUTPUT_DIR /
    "augmentation_pilot_training_history.csv"
)

final_results_df.to_csv(
    final_results_path,
    index=False
)

history_df.to_csv(
    history_path,
    index=False
)

print("\n" + "=" * 85)
print("AUGMENTATION PILOT COMPLETE")
print("=" * 85)

display(
    final_results_df.style.format({
        "accuracy": "{:.4f}",
        "macro_f1": "{:.4f}",
        "balanced_accuracy": "{:.4f}",
        "delta_f1_vs_original": "{:+.4f}"
    })
)

print("\nSaved:")
print(final_results_path)
print(history_path)

# %% [Cell 49]
# ============================================================
# BRINJAL - AUGMENTATION TO ROBUSTNESS ABLATION
# MobileNetV3-Small | 4 Strategies × 3 Seeds = 12 Runs
# ============================================================

import os
import gc
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------

SEEDS = [42, 1337, 2026]

STRATEGIES = {
    "Baseline": None,
    "Grayscale": "Grayscale",
    "Rotate": "Rotate",
    "Brightness": "Brightness",
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 1e-4

# Existing cache directories
ORIGINAL_DIR = Path("/content/brinjal_original")
TRAIN_AUG_DIR = Path("/content/brinjal_train_augmentation")
VAL_ROBUST_DIR = Path("/content/brinjal_val_robustness")

# Dataset metadata in Google Drive
META_PATH = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/metadata.csv"
)

# Output directory in Drive
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "augmentation_ablation_final"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Output:", OUTPUT_DIR)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# ------------------------------------------------------------
# 2. REPRODUCIBILITY
# ------------------------------------------------------------

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------
# 3. LOAD METADATA
# ------------------------------------------------------------

metadata = pd.read_csv(META_PATH)

print("\nMetadata shape:", metadata.shape)
print(metadata.columns.tolist())

CLASS_NAMES = sorted(metadata["class_label"].unique())
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

print("\nClasses:")
for k, v in CLASS_TO_ID.items():
    print(v, "->", k)


# ------------------------------------------------------------
# 4. BUILD FILENAME -> METADATA LOOKUP
# ------------------------------------------------------------

meta_by_filename = {}

for _, row in metadata.iterrows():
    filename = str(row["filename"])
    basename = os.path.basename(filename)

    meta_by_filename[filename] = row
    meta_by_filename[basename] = row


def lookup_metadata(path):
    """
    Resolve cached image path back to the metadata row.
    """
    path = Path(path)

    candidates = [
        path.name,
        str(path),
        path.as_posix(),
    ]

    for candidate in candidates:
        if candidate in meta_by_filename:
            return meta_by_filename[candidate]

    return None


# ------------------------------------------------------------
# 5. COLLECT ORIGINAL TRAINING IMAGES
# ------------------------------------------------------------

def collect_original_train():

    rows = []

    for path in ORIGINAL_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        row = lookup_metadata(path)

        if row is None:
            continue

        if row["data_split"] != "train":
            continue

        # Only original images
        if str(row["preprocessing_technique"]).lower() != "original":
            continue

        rows.append({
            "path": str(path),
            "label": CLASS_TO_ID[row["class_label"]],
            "class_name": row["class_label"],
            "original_image_id": row["original_image_id"],
            "transformation": "Original"
        })

    return pd.DataFrame(rows)


train_original = collect_original_train()

print("\nOriginal training:")
print(train_original.shape)
print(train_original["class_name"].value_counts())


# ------------------------------------------------------------
# 6. COLLECT AUGMENTED TRAINING IMAGES
# ------------------------------------------------------------

def collect_train_transformation(transformation):

    rows = []

    for path in TRAIN_AUG_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        row = lookup_metadata(path)

        if row is None:
            continue

        if row["data_split"] != "train":
            continue

        tech = str(row["preprocessing_technique"])

        if tech != transformation:
            continue

        rows.append({
            "path": str(path),
            "label": CLASS_TO_ID[row["class_label"]],
            "class_name": row["class_label"],
            "original_image_id": row["original_image_id"],
            "transformation": transformation
        })

    return pd.DataFrame(rows)


train_aug = {}

for transformation in ["Grayscale", "Rotate", "Brightness"]:

    df = collect_train_transformation(transformation)

    train_aug[transformation] = df

    print(f"\n{transformation}:")
    print(df.shape)
    print(df["class_name"].value_counts())


# ------------------------------------------------------------
# 7. VALIDATION ROBUSTNESS SETS
# ------------------------------------------------------------

def collect_validation(transformation):

    rows = []

    for path in VAL_ROBUST_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        row = lookup_metadata(path)

        if row is None:
            continue

        if row["data_split"] != "val":
            continue

        tech = str(row["preprocessing_technique"])

        if tech != transformation:
            continue

        rows.append({
            "path": str(path),
            "label": CLASS_TO_ID[row["class_label"]],
            "class_name": row["class_label"],
            "original_image_id": row["original_image_id"],
            "transformation": transformation
        })

    return pd.DataFrame(rows)


VAL_TRANSFORMS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]

val_sets = {}

for transformation in VAL_TRANSFORMS:

    df = collect_validation(transformation)

    val_sets[transformation] = df

    print(f"\nValidation - {transformation}:")
    print(df.shape)
    print(df["class_name"].value_counts())


# ------------------------------------------------------------
# 8. SANITY CHECKS
# ------------------------------------------------------------

assert len(train_original) == 595, \
    f"Expected 595 original train images, got {len(train_original)}"

for t in ["Grayscale", "Rotate", "Brightness"]:
    assert len(train_aug[t]) == 595, \
        f"Expected 595 {t} images, got {len(train_aug[t])}"

assert len(val_sets["Original"]) == 127
assert len(val_sets["Grayscale"]) == 127
assert len(val_sets["Rotate"]) == 127
assert len(val_sets["Brightness"]) == 127

print("\n✅ Cache sanity checks passed.")


# ------------------------------------------------------------
# 9. TF.DATA PIPELINE
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE


def decode_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method=tf.image.ResizeMethod.BILINEAR
    )

    # Keep [0,255] because MobileNetV3 includes preprocessing.
    image = tf.cast(image, tf.float32)

    return image, label


def make_dataset(df, shuffle=False, seed=42):

    paths = df["path"].values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=seed,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        decode_image,
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTOTUNE)

    return ds


# ------------------------------------------------------------
# 10. MODEL
# ------------------------------------------------------------

def build_model():

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    # Same fine-tuning setup for every strategy
    base.trainable = True

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = base(inputs, training=True)

    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------------------------------
# 11. METRICS
# ------------------------------------------------------------

def evaluate_dataset(model, df):

    ds = make_dataset(df, shuffle=False)

    y_true = []
    y_pred = []

    for x_batch, y_batch in ds:

        preds = model.predict(
            x_batch,
            verbose=0
        )

        y_true.extend(
            y_batch.numpy().tolist()
        )

        y_pred.extend(
            np.argmax(preds, axis=1).tolist()
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
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

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred
        ).tolist()
    }


# ------------------------------------------------------------
# 12. TRAIN ONE RUN
# ------------------------------------------------------------

def train_one(
    strategy_name,
    augmentation,
    seed
):

    print("\n" + "=" * 80)
    print(
        f"RUN | Strategy={strategy_name} | "
        f"Seed={seed}"
    )
    print("=" * 80)

    seed_everything(seed)

    # ---------------------------------------------
    # Build training dataframe
    # ---------------------------------------------

    if augmentation is None:

        train_df = train_original.copy()

    else:

        train_df = pd.concat(
            [
                train_original,
                train_aug[augmentation]
            ],
            ignore_index=True
        )

    train_df = train_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    print(
        f"Training images: {len(train_df)}"
    )

    # ---------------------------------------------
    # datasets
    # ---------------------------------------------

    train_ds = make_dataset(
        train_df,
        shuffle=True,
        seed=seed
    )

    val_original_ds = make_dataset(
        val_sets["Original"],
        shuffle=False
    )

    # ---------------------------------------------
    # Model
    # ---------------------------------------------

    model = build_model()

    run_name = (
        f"{strategy_name.lower()}_seed{seed}"
    )

    checkpoint_path = (
        OUTPUT_DIR /
        f"{run_name}.weights.h5"
    )

    history_path = (
        OUTPUT_DIR /
        f"{run_name}_history.csv"
    )

    # ---------------------------------------------
    # callbacks
    # ---------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
            restore_best_weights=True,
            verbose=1
        )
    ]

    # ---------------------------------------------
    # training
    # ---------------------------------------------

    history = model.fit(
        train_ds,
        validation_data=val_original_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save history
    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        history_path,
        index=False
    )

    # ---------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------

    model.load_weights(
        str(checkpoint_path)
    )

    # ---------------------------------------------
    # Validation evaluations
    # ---------------------------------------------

    result_rows = []

    for transformation in VAL_TRANSFORMS:

        print(
            f"\nEvaluating: {transformation}"
        )

        metrics = evaluate_dataset(
            model,
            val_sets[transformation]
        )

        row = {
            "strategy": strategy_name,
            "seed": seed,
            "train_augmentation": augmentation or "Original",
            "validation_transformation": transformation,
            "n_validation": len(
                val_sets[transformation]
            ),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy":
                metrics["balanced_accuracy"],
            "macro_precision":
                metrics["macro_precision"],
            "macro_recall":
                metrics["macro_recall"],
            "checkpoint":
                str(checkpoint_path)
        }

        result_rows.append(row)

        print(
            f"Accuracy = {metrics['accuracy']:.4f} | "
            f"Macro-F1 = {metrics['macro_f1']:.4f} | "
            f"Balanced Acc = "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        # Save confusion matrix separately
        cm_path = (
            OUTPUT_DIR /
            f"{run_name}_{transformation}_cm.json"
        )

        with open(cm_path, "w") as f:
            json.dump(
                {
                    "strategy": strategy_name,
                    "seed": seed,
                    "validation_transformation":
                        transformation,
                    "confusion_matrix":
                        metrics["confusion_matrix"]
                },
                f,
                indent=2
            )

    # ---------------------------------------------
    # Save per-run result
    # ---------------------------------------------

    run_results = pd.DataFrame(result_rows)

    run_results.to_csv(
        OUTPUT_DIR /
        f"{run_name}_results.csv",
        index=False
    )

    # Cleanup
    del model
    del train_ds
    del val_original_ds

    tf.keras.backend.clear_session()

    gc.collect()

    return run_results


# ------------------------------------------------------------
# 13. RUN ALL 12 EXPERIMENTS
# ------------------------------------------------------------

all_results = []

for strategy_name, augmentation in STRATEGIES.items():

    for seed in SEEDS:

        result = train_one(
            strategy_name=strategy_name,
            augmentation=augmentation,
            seed=seed
        )

        all_results.append(result)

        # Incremental save
        master_df = pd.concat(
            all_results,
            ignore_index=True
        )

        master_df.to_csv(
            OUTPUT_DIR /
            "augmentation_ablation_all_results.csv",
            index=False
        )

        print("\n✅ Saved master results.")


# ------------------------------------------------------------
# 14. FINAL AGGREGATION
# ------------------------------------------------------------

all_results = pd.concat(
    all_results,
    ignore_index=True
)

all_results.to_csv(
    OUTPUT_DIR /
    "augmentation_ablation_all_results.csv",
    index=False
)


# Mean / SD across seeds
aggregate = (
    all_results
    .groupby(
        [
            "strategy",
            "validation_transformation"
        ]
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
    .reset_index()
)

aggregate.to_csv(
    OUTPUT_DIR /
    "augmentation_ablation_aggregate.csv",
    index=False
)


# ------------------------------------------------------------
# 15. TARGET-ROBUSTNESS TABLE
# ------------------------------------------------------------

target_map = {
    "Baseline": None,
    "Grayscale": "Grayscale",
    "Rotate": "Rotate",
    "Brightness": "Brightness"
}

target_rows = []

for strategy, target_transform in target_map.items():

    if target_transform is None:
        continue

    current = all_results[
        (all_results["strategy"] == strategy) &
        (
            all_results[
                "validation_transformation"
            ] == target_transform
        )
    ]

    baseline = all_results[
        (all_results["strategy"] == "Baseline") &
        (
            all_results[
                "validation_transformation"
            ] == target_transform
        )
    ]

    merged = current.merge(
        baseline[
            [
                "seed",
                "macro_f1",
                "accuracy",
                "balanced_accuracy"
            ]
        ],
        on="seed",
        suffixes=(
            "_augmented",
            "_baseline"
        )
    )

    target_rows.append({
        "strategy": strategy,
        "target_transformation":
            target_transform,

        "augmented_f1_mean":
            merged["macro_f1_augmented"].mean(),

        "augmented_f1_sd":
            merged["macro_f1_augmented"].std(),

        "baseline_f1_mean":
            merged["macro_f1_baseline"].mean(),

        "baseline_f1_sd":
            merged["macro_f1_baseline"].std(),

        "delta_f1_mean":
            (
                merged["macro_f1_augmented"]
                - merged["macro_f1_baseline"]
            ).mean(),

        "delta_f1_sd":
            (
                merged["macro_f1_augmented"]
                - merged["macro_f1_baseline"]
            ).std(),

        "augmented_accuracy_mean":
            merged["accuracy_augmented"].mean(),

        "baseline_accuracy_mean":
            merged["accuracy_baseline"].mean(),

        "augmented_balanced_acc_mean":
            merged[
                "balanced_accuracy_augmented"
            ].mean(),

        "baseline_balanced_acc_mean":
            merged[
                "balanced_accuracy_baseline"
            ].mean()
    })

target_df = pd.DataFrame(target_rows)

target_df.to_csv(
    OUTPUT_DIR /
    "target_robustness_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 16. ORIGINAL VALIDATION IMPACT
# ------------------------------------------------------------

original_results = all_results[
    all_results[
        "validation_transformation"
    ] == "Original"
].copy()

original_summary = (
    original_results
    .groupby("strategy")
    .agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
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

original_summary.to_csv(
    OUTPUT_DIR /
    "original_validation_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 17. DISPLAY FINAL RESULTS
# ------------------------------------------------------------

print("\n\n" + "#" * 100)
print("FINAL AGGREGATE")
print("#" * 100)

print(
    aggregate.to_string(index=False)
)

print("\n\n" + "#" * 100)
print("TARGET ROBUSTNESS")
print("#" * 100)

print(
    target_df.to_string(index=False)
)

print("\n\n" + "#" * 100)
print("ORIGINAL VALIDATION IMPACT")
print("#" * 100)

print(
    original_summary.to_string(index=False)
)

print("\n\n✅ ALL 12 RUNS COMPLETED")
print(
    "Results saved to:",
    OUTPUT_DIR
)

# %% [Cell 50]
# ============================================================
# DIAGNOSTIC: RE-EVALUATE EXISTING CHECKPOINTS
# WITH SAFE INFERENCE MODE
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "augmentation_ablation_final"
)

RESULT_FILE = (
    OUTPUT_DIR /
    "augmentation_ablation_all_results.csv"
)

old_results = pd.read_csv(RESULT_FILE)

print("Existing result rows:", len(old_results))
print(old_results[["strategy", "seed", "checkpoint"]].drop_duplicates())


# ------------------------------------------------------------
# SAFE MODEL
# ------------------------------------------------------------

def build_model_safe():

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    base.trainable = True

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    # IMPORTANT:
    # BatchNorm runs in inference mode
    x = base(inputs, training=False)

    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    return model


# ------------------------------------------------------------
# SAFE EVALUATION
# ------------------------------------------------------------

def evaluate_safe(model, df):

    ds = make_dataset(
        df,
        shuffle=False
    )

    y_true = []
    y_pred = []

    for x_batch, y_batch in ds:

        preds = model.predict(
            x_batch,
            verbose=0
        )

        y_true.extend(
            y_batch.numpy().tolist()
        )

        y_pred.extend(
            np.argmax(preds, axis=1).tolist()
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
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

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


# ------------------------------------------------------------
# UNIQUE RUNS
# ------------------------------------------------------------

runs = (
    old_results[
        [
            "strategy",
            "seed",
            "checkpoint"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

diagnostic_rows = []


for _, run in runs.iterrows():

    strategy = run["strategy"]
    seed = int(run["seed"])
    checkpoint = run["checkpoint"]

    print("\n" + "=" * 80)
    print(
        f"{strategy} | Seed {seed}"
    )
    print("=" * 80)

    model = build_model_safe()

    model.load_weights(
        checkpoint
    )

    for transformation in VAL_TRANSFORMS:

        metrics = evaluate_safe(
            model,
            val_sets[transformation]
        )

        diagnostic_rows.append({
            "strategy": strategy,
            "seed": seed,
            "validation_transformation":
                transformation,
            **metrics
        })

        print(
            f"{transformation:12s} | "
            f"F1={metrics['macro_f1']:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"BalAcc={metrics['balanced_accuracy']:.4f}"
        )

    del model
    tf.keras.backend.clear_session()
    gc.collect()


diagnostic_df = pd.DataFrame(
    diagnostic_rows
)

diagnostic_df.to_csv(
    OUTPUT_DIR /
    "diagnostic_safe_inference.csv",
    index=False
)


# ------------------------------------------------------------
# AGGREGATE
# ------------------------------------------------------------

diagnostic_aggregate = (
    diagnostic_df
    .groupby(
        [
            "strategy",
            "validation_transformation"
        ]
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
    .reset_index()
)

print("\n\n")
print("#" * 100)
print("SAFE INFERENCE DIAGNOSTIC")
print("#" * 100)

print(
    diagnostic_aggregate.to_string(
        index=False
    )
)

diagnostic_aggregate.to_csv(
    OUTPUT_DIR /
    "diagnostic_safe_inference_aggregate.csv",
    index=False
)

print(
    "\nSaved:",
    OUTPUT_DIR /
    "diagnostic_safe_inference_aggregate.csv"
)

# %% [Cell 51]
# ============================================================
# SINGLE SANITY RUN
# MobileNetV3-Small | Baseline | Seed 42
# FIXED INFERENCE/TRAINING BEHAVIOR
# ============================================================

import gc
import numpy as np
import tensorflow as tf
import pandas as pd

seed = 42

seed_everything(seed)

# ------------------------------------------------------------
# Correct model
# ------------------------------------------------------------

base = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

base.trainable = True

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

# IMPORTANT:
# Do NOT force training=True.
# BatchNorm should use inference statistics.
x = base(inputs)

outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

train_ds = make_dataset(
    train_original,
    shuffle=True,
    seed=42
)

val_ds = make_dataset(
    val_sets["Original"],
    shuffle=False
)

# ------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------

sanity_checkpoint = (
    OUTPUT_DIR /
    "SANITY_baseline_seed42.weights.h5"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        str(sanity_checkpoint),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )
]

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------------------
# Best checkpoint
# ------------------------------------------------------------

model.load_weights(
    str(sanity_checkpoint)
)

# ------------------------------------------------------------
# Evaluate Original Validation
# ------------------------------------------------------------

metrics = evaluate_dataset(
    model,
    val_sets["Original"]
)

print("\n" + "=" * 70)
print("SANITY RUN RESULT")
print("=" * 70)

print(
    f"Accuracy:          {metrics['accuracy']:.4f}"
)

print(
    f"Macro-F1:          {metrics['macro_f1']:.4f}"
)

print(
    f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
)

print("=" * 70)

# Show best validation loss
hist = pd.DataFrame(history.history)

best_epoch = hist["val_loss"].idxmin()

print(
    f"\nBest epoch: {best_epoch + 1}"
)

print(
    f"Best val_loss: {hist.loc[best_epoch, 'val_loss']:.6f}"
)

gc.collect()

# %% [Cell 52]
# ============================================================
# BRINJAL AUGMENTATION ABLATION
# PYTORCH SANITY CHECK
# MobileNetV3-Small | Baseline | Seed 42 | 1 Epoch
# ============================================================

import os
import gc
import random
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
    balanced_accuracy_score
)

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

LR = 1e-4
WEIGHT_DECAY = 1e-4

META_PATH = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/metadata.csv"
)

ORIGINAL_DIR = Path(
    "/content/brinjal_original"
)

print("PyTorch:", torch.__version__)
print("Torchvision:", __import__("torchvision").__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ------------------------------------------------------------
# 2. SEED
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 3. METADATA
# ------------------------------------------------------------

metadata = pd.read_csv(META_PATH)

CLASS_NAMES = sorted(
    metadata["class_label"].unique()
)

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

print("\nClasses:")
print(CLASS_TO_ID)


# ------------------------------------------------------------
# 4. FILE LOOKUP
# ------------------------------------------------------------

meta_lookup = {}

for _, row in metadata.iterrows():

    filename = str(row["filename"])
    basename = os.path.basename(filename)

    meta_lookup[filename] = row
    meta_lookup[basename] = row


def get_metadata(path):

    path = Path(path)

    candidates = [
        path.name,
        path.as_posix(),
        str(path)
    ]

    for c in candidates:
        if c in meta_lookup:
            return meta_lookup[c]

    return None


# ------------------------------------------------------------
# 5. COLLECT ORIGINAL TRAIN
# ------------------------------------------------------------

train_rows = []

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

    if str(row["preprocessing_technique"]).lower() != "original":
        continue

    train_rows.append({
        "path": str(path),
        "label": CLASS_TO_ID[row["class_label"]],
        "class_name": row["class_label"]
    })


train_df = pd.DataFrame(train_rows)

print("\nTraining samples:", len(train_df))

print(
    train_df["class_name"].value_counts()
)

assert len(train_df) == 595, (
    f"Expected 595 training images, "
    f"got {len(train_df)}"
)


# ------------------------------------------------------------
# 6. VALIDATION ORIGINAL
# ------------------------------------------------------------

# Find Original validation images from the existing
# validation cache.

VAL_DIR = Path(
    "/content/brinjal_val_robustness"
)

val_rows = []

for path in VAL_DIR.rglob("*"):

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

    if row["data_split"] != "val":
        continue

    if str(row["preprocessing_technique"]) != "Original":
        continue

    val_rows.append({
        "path": str(path),
        "label": CLASS_TO_ID[row["class_label"]],
        "class_name": row["class_label"]
    })


val_df = pd.DataFrame(val_rows)

print(
    "\nValidation samples:",
    len(val_df)
)

assert len(val_df) == 127, (
    f"Expected 127 validation images, "
    f"got {len(val_df)}"
)


# ------------------------------------------------------------
# 7. TRANSFORM
# ------------------------------------------------------------

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

transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ------------------------------------------------------------
# 8. DATASET
# ------------------------------------------------------------

class BrinjalDataset(Dataset):

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
            row["path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])

        return image, label


train_dataset = BrinjalDataset(
    train_df,
    transform=transform
)

val_dataset = BrinjalDataset(
    val_df,
    transform=transform
)


# ------------------------------------------------------------
# 9. DATALOADERS
# ------------------------------------------------------------

generator = torch.Generator()

generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    generator=generator,
    persistent_workers=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False
)


# ------------------------------------------------------------
# 10. MODEL
# ------------------------------------------------------------

weights = (
    models.MobileNet_V3_Small_Weights.DEFAULT
)

model = models.mobilenet_v3_small(
    weights=weights
)

# Replace classifier
in_features = (
    model.classifier[3].in_features
)

model.classifier[3] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

# IMPORTANT:
# Keep the complete backbone trainable,
# matching the previous fine-tuning setup.
for param in model.parameters():
    param.requires_grad = True


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)


# ------------------------------------------------------------
# 11. LOSS + OPTIMIZER
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ------------------------------------------------------------
# 12. TRAIN — ONE EPOCH ONLY
# ------------------------------------------------------------

model.train()

running_loss = 0.0
correct = 0
total = 0

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

    outputs = model(images)

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()

    optimizer.step()

    running_loss += (
        loss.item()
        * images.size(0)
    )

    predictions = outputs.argmax(
        dim=1
    )

    correct += (
        predictions == labels
    ).sum().item()

    total += labels.size(0)


train_loss = running_loss / total
train_acc = correct / total


# ------------------------------------------------------------
# 13. VALIDATION
# ------------------------------------------------------------

model.eval()

y_true = []
y_pred = []

val_loss_total = 0.0
val_total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        val_loss_total += (
            loss.item()
            * images.size(0)
        )

        val_total += labels.size(0)

        predictions = outputs.argmax(
            dim=1
        )

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )


val_loss = (
    val_loss_total /
    val_total
)

val_acc = accuracy_score(
    y_true,
    y_pred
)

val_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

val_balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred
)


# ------------------------------------------------------------
# 14. RESULT
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("PYTORCH SANITY CHECK — EPOCH 1")
print("=" * 70)

print(
    f"Train Loss:        {train_loss:.4f}"
)

print(
    f"Train Accuracy:    {train_acc:.4f}"
)

print(
    f"Val Loss:          {val_loss:.4f}"
)

print(
    f"Val Accuracy:      {val_acc:.4f}"
)

print(
    f"Val Macro-F1:      {val_f1:.4f}"
)

print(
    f"Val Balanced Acc:  {val_balanced_acc:.4f}"
)

print("=" * 70)

print("\nReference from the original pilot:")
print(
    "Train Loss ≈ 0.8858"
)

print(
    "Train Acc  ≈ 0.6185"
)

print(
    "Val F1     ≈ 0.2686"
)

print(
    "Val Acc    ≈ 0.4961"
)

# %% [Cell 53]
# ============================================================
# BRINJAL AUGMENTATION -> ROBUSTNESS ABLATION
# FINAL PYTORCH EXPERIMENT
#
# MobileNetV3-Small
# 4 Strategies × 3 Seeds = 12 Runs
# ============================================================

import os
import gc
import json
import random
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
# 1. CONFIG
# ============================================================

SEEDS = [42, 1337, 2026]

STRATEGIES = {
    "Baseline": None,
    "Grayscale": "Grayscale",
    "Rotate": "Rotate",
    "Brightness": "Brightness"
}

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

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

META_PATH = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "metadata.csv"
)

ORIGINAL_DIR = Path(
    "/content/brinjal_original"
)

TRAIN_AUG_DIR = Path(
    "/content/brinjal_train_augmentation"
)

VAL_ROBUST_DIR = Path(
    "/content/brinjal_val_robustness"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "augmentation_ablation_final_pytorch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("PyTorch:", torch.__version__)
print(
    "Torchvision:",
    __import__("torchvision").__version__
)
print(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

# ============================================================
# 2. REPRODUCIBILITY
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


# ============================================================
# 3. METADATA
# ============================================================

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

print("\nClasses:")
print(CLASS_TO_ID)

# ============================================================
# 4. METADATA LOOKUP
# ============================================================

meta_lookup = {}

for _, row in metadata.iterrows():

    filename = str(row["filename"])
    basename = os.path.basename(filename)

    meta_lookup[filename] = row
    meta_lookup[basename] = row


def get_metadata(path):

    path = Path(path)

    candidates = [
        path.name,
        path.as_posix(),
        str(path)
    ]

    for candidate in candidates:

        if candidate in meta_lookup:
            return meta_lookup[candidate]

    return None


# ============================================================
# 5. COLLECT ORIGINAL TRAIN
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
            "path": str(path),
            "label": CLASS_TO_ID[
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


train_original = collect_original_train()

print(
    "\nOriginal training:",
    len(train_original)
)

assert len(train_original) == 595


# ============================================================
# 6. COLLECT TRAINING TRANSFORMATIONS
# ============================================================

def collect_train_transformation(
    transformation
):

    rows = []

    for path in TRAIN_AUG_DIR.rglob("*"):

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
            != transformation
        ):
            continue

        rows.append({
            "path": str(path),
            "label": CLASS_TO_ID[
                row["class_label"]
            ],
            "class_name":
                row["class_label"],
            "original_image_id":
                row["original_image_id"],
            "transformation":
                transformation
        })

    return pd.DataFrame(rows)


train_aug = {}

for transformation in [
    "Grayscale",
    "Rotate",
    "Brightness"
]:

    df = collect_train_transformation(
        transformation
    )

    train_aug[transformation] = df

    print(
        f"{transformation}: {len(df)}"
    )

    assert len(df) == 595


# ============================================================
# 7. COLLECT VALIDATION SETS
# ============================================================

def collect_validation(
    transformation
):

    rows = []

    for path in VAL_ROBUST_DIR.rglob("*"):

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

        if row["data_split"] != "val":
            continue

        if (
            str(row["preprocessing_technique"])
            != transformation
        ):
            continue

        rows.append({
            "path": str(path),
            "label": CLASS_TO_ID[
                row["class_label"]
            ],
            "class_name":
                row["class_label"],
            "original_image_id":
                row["original_image_id"],
            "transformation":
                transformation
        })

    return pd.DataFrame(rows)


VAL_TRANSFORMS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]

val_sets = {}

for transformation in VAL_TRANSFORMS:

    df = collect_validation(
        transformation
    )

    val_sets[transformation] = df

    print(
        f"Validation {transformation}: "
        f"{len(df)}"
    )

    assert len(df) == 127


# ============================================================
# 8. IMAGE TRANSFORM
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
# 9. DATASET
# ============================================================

class BrinjalDataset(Dataset):

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
            row["path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])

        return image, label


# ============================================================
# 10. DATA LOADERS
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

    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=generator,
        persistent_workers=False
    )

    return loader


# ============================================================
# 11. MODEL
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

    # Full fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    return model


# ============================================================
# 12. EVALUATION
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

            outputs = model(images)

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

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {

        "loss":
            total_loss / total_samples,

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

    return metrics


# ============================================================
# 13. TRAIN ONE EXPERIMENT
# ============================================================

def train_one(
    strategy,
    augmentation,
    seed
):

    print("\n")
    print("=" * 90)
    print(
        f"STRATEGY = {strategy} | "
        f"SEED = {seed}"
    )
    print("=" * 90)

    seed_everything(seed)

    # --------------------------------------------------------
    # Training dataframe
    # --------------------------------------------------------

    if augmentation is None:

        train_df = (
            train_original.copy()
        )

    else:

        train_df = pd.concat(
            [
                train_original,
                train_aug[
                    augmentation
                ]
            ],
            ignore_index=True
        )

    train_df = train_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(
        drop=True
    )

    print(
        "Training samples:",
        len(train_df)
    )

    # --------------------------------------------------------
    # loaders
    # --------------------------------------------------------

    train_loader = make_loader(
        train_df,
        shuffle=True,
        seed=seed
    )

    val_loader = make_loader(
        val_sets["Original"],
        shuffle=False,
        seed=seed
    )

    # --------------------------------------------------------
    # device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = build_model()

    model = model.to(device)

    # --------------------------------------------------------
    # loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR
    )

    # --------------------------------------------------------
    # paths
    # --------------------------------------------------------

    run_name = (
        f"{strategy.lower()}_seed{seed}"
    )

    checkpoint_path = (
        OUTPUT_DIR /
        f"{run_name}.pt"
    )

    history_path = (
        OUTPUT_DIR /
        f"{run_name}_history.csv"
    )

    # --------------------------------------------------------
    # training state
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # epochs
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

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

            train_loss_total += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += (
                labels.size(0)
            )

        train_loss = (
            train_loss_total /
            train_total
        )

        train_acc = (
            train_correct /
            train_total
        )

        # ----------------------------------------------------
        # validation
        # ----------------------------------------------------

        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        val_loss = (
            val_metrics["loss"]
        )

        val_f1 = (
            val_metrics["macro_f1"]
        )

        val_acc = (
            val_metrics["accuracy"]
        )

        # ----------------------------------------------------
        # scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # checkpoint by val LOSS
        # ----------------------------------------------------

        improved = (
            val_loss < best_val_loss
        )

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "strategy":
                        strategy,

                    "augmentation":
                        augmentation,

                    "seed":
                        seed,

                    "epoch":
                        epoch,

                    "best_val_loss":
                        best_val_loss
                },
                checkpoint_path
            )

            marker = "✅ BEST"

        else:

            epochs_without_improvement += 1

            marker = ""

        # ----------------------------------------------------
        # history
        # ----------------------------------------------------

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
                val_metrics[
                    "balanced_accuracy"
                ],

            "learning_rate":
                current_lr

        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Orig Val Loss {val_loss:.4f} | "
            f"Orig Val F1 {val_f1:.4f} | "
            f"Orig Val Acc {val_acc:.4f} | "
            f"LR {current_lr:.6f} "
            f"{marker}"
        )

        # ----------------------------------------------------
        # early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            print(
                "\n⏹ Early stopping"
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Evaluate all validation conditions
    # --------------------------------------------------------

    result_rows = []

    for transformation in VAL_TRANSFORMS:

        print(
            f"\nEvaluating "
            f"{transformation}..."
        )

        loader = make_loader(
            val_sets[
                transformation
            ],
            shuffle=False,
            seed=seed
        )

        metrics = evaluate_model(
            model,
            loader,
            criterion,
            device
        )

        print(
            f"Accuracy = "
            f"{metrics['accuracy']:.4f} | "
            f"Macro-F1 = "
            f"{metrics['macro_f1']:.4f} | "
            f"Balanced Acc = "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        result_rows.append({

            "strategy":
                strategy,

            "seed":
                seed,

            "train_augmentation":
                augmentation or "Original",

            "validation_transformation":
                transformation,

            "accuracy":
                metrics["accuracy"],

            "macro_f1":
                metrics["macro_f1"],

            "balanced_accuracy":
                metrics[
                    "balanced_accuracy"
                ],

            "macro_precision":
                metrics[
                    "macro_precision"
                ],

            "macro_recall":
                metrics[
                    "macro_recall"
                ],

            "best_epoch":
                best_epoch,

            "best_val_loss":
                best_val_loss,

            "checkpoint":
                str(checkpoint_path)

        })

        cm_path = (
            OUTPUT_DIR /
            f"{run_name}_"
            f"{transformation}_"
            f"cm.json"
        )

        with open(
            cm_path,
            "w"
        ) as f:

            json.dump(
                {
                    "strategy":
                        strategy,

                    "seed":
                        seed,

                    "validation_transformation":
                        transformation,

                    "confusion_matrix":
                        metrics[
                            "confusion_matrix"
                        ]
                },
                f,
                indent=2
            )

    run_results = pd.DataFrame(
        result_rows
    )

    run_results.to_csv(
        OUTPUT_DIR /
        f"{run_name}_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # cleanup
    # --------------------------------------------------------

    del model
    del train_loader
    del val_loader

    torch.cuda.empty_cache()
    gc.collect()

    return run_results


# ============================================================
# 14. RUN ALL 12
# ============================================================

all_results = []

for strategy, augmentation in STRATEGIES.items():

    for seed in SEEDS:

        result = train_one(
            strategy=
                strategy,

            augmentation=
                augmentation,

            seed=
                seed
        )

        all_results.append(
            result
        )

        # Save after EVERY run
        master_results = pd.concat(
            all_results,
            ignore_index=True
        )

        master_results.to_csv(
            OUTPUT_DIR /
            "augmentation_ablation_all_results.csv",
            index=False
        )

        print(
            "\n💾 Incremental results saved."
        )


# ============================================================
# 15. AGGREGATE
# ============================================================

all_results = pd.concat(
    all_results,
    ignore_index=True
)

all_results.to_csv(
    OUTPUT_DIR /
    "augmentation_ablation_all_results.csv",
    index=False
)


aggregate = (
    all_results
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

aggregate.to_csv(
    OUTPUT_DIR /
    "augmentation_ablation_aggregate.csv",
    index=False
)


# ============================================================
# 16. TARGET ROBUSTNESS
# ============================================================

target_map = {
    "Grayscale":
        "Grayscale",

    "Rotate":
        "Rotate",

    "Brightness":
        "Brightness"
}

target_rows = []

for strategy, target in target_map.items():

    augmented = all_results[
        (
            all_results["strategy"]
            == strategy
        )
        &
        (
            all_results[
                "validation_transformation"
            ]
            == target
        )
    ]

    baseline = all_results[
        (
            all_results["strategy"]
            == "Baseline"
        )
        &
        (
            all_results[
                "validation_transformation"
            ]
            == target
        )
    ]

    merged = augmented.merge(
        baseline[
            [
                "seed",
                "macro_f1",
                "accuracy",
                "balanced_accuracy"
            ]
        ],
        on="seed",
        suffixes=(
            "_augmented",
            "_baseline"
        )
    )

    delta_f1 = (
        merged["macro_f1_augmented"]
        -
        merged["macro_f1_baseline"]
    )

    target_rows.append({

        "strategy":
            strategy,

        "target_transformation":
            target,

        "augmented_f1_mean":
            merged[
                "macro_f1_augmented"
            ].mean(),

        "augmented_f1_sd":
            merged[
                "macro_f1_augmented"
            ].std(),

        "baseline_f1_mean":
            merged[
                "macro_f1_baseline"
            ].mean(),

        "baseline_f1_sd":
            merged[
                "macro_f1_baseline"
            ].std(),

        "delta_f1_mean":
            delta_f1.mean(),

        "delta_f1_sd":
            delta_f1.std(),

        "augmented_accuracy_mean":
            merged[
                "accuracy_augmented"
            ].mean(),

        "baseline_accuracy_mean":
            merged[
                "accuracy_baseline"
            ].mean(),

        "augmented_balanced_acc_mean":
            merged[
                "balanced_accuracy_augmented"
            ].mean(),

        "baseline_balanced_acc_mean":
            merged[
                "balanced_accuracy_baseline"
            ].mean()

    })


target_summary = pd.DataFrame(
    target_rows
)

target_summary.to_csv(
    OUTPUT_DIR /
    "target_robustness_summary.csv",
    index=False
)


# ============================================================
# 17. ORIGINAL VALIDATION IMPACT
# ============================================================

original_summary = (
    all_results[
        all_results[
            "validation_transformation"
        ] == "Original"
    ]
    .groupby("strategy")
    .agg(

        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),

        macro_f1_sd=(
            "macro_f1",
            "std"
        ),

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_sd=(
            "accuracy",
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

original_summary.to_csv(
    OUTPUT_DIR /
    "original_validation_summary.csv",
    index=False
)


# ============================================================
# 18. FINAL DISPLAY
# ============================================================

print("\n")
print("#" * 100)
print("FINAL AGGREGATE")
print("#" * 100)

print(
    aggregate.to_string(
        index=False
    )
)

print("\n")
print("#" * 100)
print("TARGET ROBUSTNESS")
print("#" * 100)

print(
    target_summary.to_string(
        index=False
    )
)

print("\n")
print("#" * 100)
print("ORIGINAL VALIDATION IMPACT")
print("#" * 100)

print(
    original_summary.to_string(
        index=False
    )
)

print("\n")
print(
    "✅ ALL 12 PYTORCH RUNS COMPLETED"
)

print(
    "Results:",
    OUTPUT_DIR
)

# %% [Cell 54]
# ============================================================
# FINAL LOCKED TEST EVALUATION
#
# 12 final PyTorch checkpoints
# ×
# 4 test conditions
#
# NO TRAINING
# NO MODEL SELECTION
# ============================================================

import os
import gc
import random
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

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. CONFIG
# ============================================================

SEEDS = [42, 1337, 2026]

STRATEGIES = {
    "Baseline": None,
    "Grayscale": "Grayscale",
    "Rotate": "Rotate",
    "Brightness": "Brightness"
}

TEST_TRANSFORMS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

META_PATH = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "metadata.csv"
)

TEST_DIR = Path(
    "/content/brinjal_test_all"
)

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "augmentation_ablation_final_pytorch"
)

OUTPUT_DIR = CHECKPOINT_DIR / "final_locked_test"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. ENVIRONMENT
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("PyTorch:", torch.__version__)
print(
    "Torchvision:",
    __import__("torchvision").__version__
)
print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. METADATA
# ============================================================

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

print("\nClasses:")
print(CLASS_TO_ID)


# ============================================================
# 4. METADATA LOOKUP
# ============================================================

meta_lookup = {}

for _, row in metadata.iterrows():

    filename = str(row["filename"])
    basename = os.path.basename(filename)

    meta_lookup[filename] = row
    meta_lookup[basename] = row


def get_metadata(path):

    path = Path(path)

    candidates = [
        path.name,
        path.as_posix(),
        str(path)
    ]

    for c in candidates:

        if c in meta_lookup:
            return meta_lookup[c]

    return None


# ============================================================
# 5. COLLECT TEST IMAGES
# ============================================================

def collect_test_transformation(
    transformation
):

    rows = []

    for path in TEST_DIR.rglob("*"):

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

        if str(row["data_split"]) != "test":
            continue

        if (
            str(row["preprocessing_technique"])
            != transformation
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
                str(row["original_image_id"]),

            "filename":
                str(row["filename"]),

            "transformation":
                transformation
        })

    return pd.DataFrame(rows)


test_sets = {}

for transformation in TEST_TRANSFORMS:

    df = collect_test_transformation(
        transformation
    )

    test_sets[transformation] = df

    print(
        f"Test {transformation}: "
        f"{len(df)} images"
    )

    assert len(df) == 128, (
        f"{transformation}: "
        f"expected 128, got {len(df)}"
    )


# ============================================================
# 6. SANITY: SAME 128 LEAVES
# ============================================================

original_ids = set(
    test_sets["Original"][
        "original_image_id"
    ]
)

assert len(original_ids) == 128

for transformation in TEST_TRANSFORMS:

    ids = set(
        test_sets[transformation][
            "original_image_id"
        ]
    )

    assert (
        ids == original_ids
    ), (
        f"Leaf IDs mismatch for "
        f"{transformation}"
    )

print(
    "\n✅ All four test conditions "
    "contain the same 128 leaves."
)


# ============================================================
# 7. IMAGE TRANSFORM
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
# 8. DATASET
# ============================================================

class BrinjalDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["path"]
        ).convert("RGB")

        image = image_transform(
            image
        )

        return (
            image,
            int(row["label"]),
            row["original_image_id"],
            row["filename"]
        )


def make_loader(df):

    dataset = BrinjalDataset(df)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False
    )


# ============================================================
# 9. MODEL
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

    return model


# ============================================================
# 10. LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    strategy,
    seed
):

    filename = (
        f"{strategy.lower()}_seed{seed}.pt"
    )

    path = (
        CHECKPOINT_DIR /
        filename
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    model = build_model()

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model, path, checkpoint


# ============================================================
# 11. EVALUATE + SAVE PREDICTIONS
# ============================================================

criterion = nn.CrossEntropyLoss()

metric_rows = []
prediction_rows = []

for strategy in STRATEGIES:

    for seed in SEEDS:

        print("\n")
        print("=" * 90)
        print(
            f"MODEL: {strategy} | "
            f"SEED: {seed}"
        )
        print("=" * 90)

        model, checkpoint_path, checkpoint = (
            load_checkpoint(
                strategy,
                seed
            )
        )

        print(
            "Best epoch:",
            checkpoint.get(
                "epoch",
                "NA"
            )
        )

        for transformation in TEST_TRANSFORMS:

            df = test_sets[
                transformation
            ]

            loader = make_loader(df)

            y_true = []
            y_pred = []
            y_prob = []
            leaf_ids = []
            filenames = []

            with torch.no_grad():

                for (
                    images,
                    labels,
                    batch_leaf_ids,
                    batch_filenames
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

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    predictions = (
                        probabilities.argmax(
                            dim=1
                        )
                    )

                    y_true.extend(
                        labels.cpu().numpy()
                    )

                    y_pred.extend(
                        predictions.cpu().numpy()
                    )

                    y_prob.extend(
                        probabilities.cpu().numpy()
                    )

                    leaf_ids.extend(
                        list(batch_leaf_ids)
                    )

                    filenames.extend(
                        list(batch_filenames)
                    )

            y_true_np = np.asarray(
                y_true
            )

            y_pred_np = np.asarray(
                y_pred
            )

            y_prob_np = np.asarray(
                y_prob
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_true_np,
                y_pred_np
            )

            macro_f1 = f1_score(
                y_true_np,
                y_pred_np,
                average="macro",
                zero_division=0
            )

            balanced_acc = (
                balanced_accuracy_score(
                    y_true_np,
                    y_pred_np
                )
            )

            macro_precision = (
                precision_score(
                    y_true_np,
                    y_pred_np,
                    average="macro",
                    zero_division=0
                )
            )

            macro_recall = (
                recall_score(
                    y_true_np,
                    y_pred_np,
                    average="macro",
                    zero_division=0
                )
            )

            cm = confusion_matrix(
                y_true_np,
                y_pred_np
            )

            metric_rows.append({

                "strategy":
                    strategy,

                "seed":
                    seed,

                "validation_or_test":
                    "TEST",

                "transformation":
                    transformation,

                "n":
                    len(df),

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,

                "balanced_accuracy":
                    balanced_acc,

                "macro_precision":
                    macro_precision,

                "macro_recall":
                    macro_recall,

                "checkpoint":
                    str(checkpoint_path),

                "best_epoch":
                    checkpoint.get(
                        "epoch",
                        np.nan
                    )

            })

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            for i in range(
                len(df)
            ):

                prediction_rows.append({

                    "strategy":
                        strategy,

                    "seed":
                        seed,

                    "transformation":
                        transformation,

                    "original_image_id":
                        leaf_ids[i],

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
                        int(
                            y_true_np[i]
                            ==
                            y_pred_np[i]
                        ),

                    "prob_healthy":
                        float(
                            y_prob_np[i, 0]
                        ),

                    "prob_little_leaf":
                        float(
                            y_prob_np[i, 1]
                        ),

                    "prob_phomopsis":
                        float(
                            y_prob_np[i, 2]
                        )

                })

            print(
                f"{transformation:12s} | "
                f"Acc={accuracy:.4f} | "
                f"F1={macro_f1:.4f} | "
                f"BalAcc={balanced_acc:.4f}"
            )

        del model

        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# 12. SAVE RAW RESULTS
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)

predictions_df = pd.DataFrame(
    prediction_rows
)

metrics_df.to_csv(
    OUTPUT_DIR /
    "final_test_metrics.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR /
    "final_test_predictions.csv",
    index=False
)


# ============================================================
# 13. TEST AGGREGATE
# ============================================================

test_aggregate = (
    metrics_df
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

test_aggregate.to_csv(
    OUTPUT_DIR /
    "final_test_aggregate.csv",
    index=False
)


# ============================================================
# 14. TARGET ROBUSTNESS ON LOCKED TEST
# ============================================================

target_map = {
    "Grayscale":
        "Grayscale",

    "Rotate":
        "Rotate",

    "Brightness":
        "Brightness"
}

target_test_rows = []

for strategy, target in target_map.items():

    augmented = metrics_df[
        (
            metrics_df["strategy"]
            == strategy
        )
        &
        (
            metrics_df["transformation"]
            == target
        )
    ]

    baseline = metrics_df[
        (
            metrics_df["strategy"]
            == "Baseline"
        )
        &
        (
            metrics_df["transformation"]
            == target
        )
    ]

    merged = augmented.merge(
        baseline[
            [
                "seed",
                "accuracy",
                "macro_f1",
                "balanced_accuracy"
            ]
        ],
        on="seed",
        suffixes=(
            "_augmented",
            "_baseline"
        )
    )

    f1_delta = (
        merged[
            "macro_f1_augmented"
        ]
        -
        merged[
            "macro_f1_baseline"
        ]
    )

    acc_delta = (
        merged[
            "accuracy_augmented"
        ]
        -
        merged[
            "accuracy_baseline"
        ]
    )

    balacc_delta = (
        merged[
            "balanced_accuracy_augmented"
        ]
        -
        merged[
            "balanced_accuracy_baseline"
        ]
    )

    target_test_rows.append({

        "strategy":
            strategy,

        "target_transformation":
            target,

        "augmented_f1_mean":
            merged[
                "macro_f1_augmented"
            ].mean(),

        "augmented_f1_sd":
            merged[
                "macro_f1_augmented"
            ].std(),

        "baseline_f1_mean":
            merged[
                "macro_f1_baseline"
            ].mean(),

        "baseline_f1_sd":
            merged[
                "macro_f1_baseline"
            ].std(),

        "delta_f1_mean":
            f1_delta.mean(),

        "delta_f1_sd":
            f1_delta.std(),

        "accuracy_delta_mean":
            acc_delta.mean(),

        "balanced_accuracy_delta_mean":
            balacc_delta.mean()

    })


target_test_df = pd.DataFrame(
    target_test_rows
)

target_test_df.to_csv(
    OUTPUT_DIR /
    "final_test_target_robustness.csv",
    index=False
)


# ============================================================
# 15. PAIRED McNEMAR TEST
# ============================================================

mcnemar_rows = []

for strategy, target in target_map.items():

    for seed in SEEDS:

        aug = predictions_df[
            (
                predictions_df["strategy"]
                == strategy
            )
            &
            (
                predictions_df["seed"]
                == seed
            )
            &
            (
                predictions_df["transformation"]
                == target
            )
        ].copy()

        base = predictions_df[
            (
                predictions_df["strategy"]
                == "Baseline"
            )
            &
            (
                predictions_df["seed"]
                == seed
            )
            &
            (
                predictions_df["transformation"]
                == target
            )
        ].copy()

        merged = aug.merge(
            base[
                [
                    "original_image_id",
                    "correct"
                ]
            ],
            on="original_image_id",
            suffixes=(
                "_augmented",
                "_baseline"
            )
        )

        # Both evaluated on exact same 128 leaves
        assert len(merged) == 128

        aug_correct = (
            merged[
                "correct_augmented"
            ].astype(bool)
        )

        base_correct = (
            merged[
                "correct_baseline"
            ].astype(bool)
        )

        # Discordant pairs
        baseline_correct_aug_wrong = int(
            (
                base_correct
                &
                ~aug_correct
            ).sum()
        )

        baseline_wrong_aug_correct = int(
            (
                ~base_correct
                &
                aug_correct
            ).sum()
        )

        discordant = (
            baseline_correct_aug_wrong
            +
            baseline_wrong_aug_correct
        )

        if discordant == 0:

            p_value = 1.0

        else:

            # Exact two-sided McNemar
            p_value = (
                binomtest(
                    baseline_wrong_aug_correct,
                    n=discordant,
                    p=0.5,
                    alternative="two-sided"
                ).pvalue
            )

        mcnemar_rows.append({

            "strategy":
                strategy,

            "seed":
                seed,

            "target_transformation":
                target,

            "baseline_correct_augmented_wrong":
                baseline_correct_aug_wrong,

            "baseline_wrong_augmented_correct":
                baseline_wrong_aug_correct,

            "discordant_pairs":
                discordant,

            "p_value":
                p_value

        })


mcnemar_df = pd.DataFrame(
    mcnemar_rows
)

# FDR across the 9 primary paired comparisons
reject, q_values, _, _ = (
    multipletests(
        mcnemar_df["p_value"].values,
        alpha=0.05,
        method="fdr_bh"
    )
)

mcnemar_df[
    "fdr_q"
] = q_values

mcnemar_df[
    "significant_after_fdr"
] = reject

mcnemar_df.to_csv(
    OUTPUT_DIR /
    "final_test_mcnemar.csv",
    index=False
)


# ============================================================
# 16. CLEAN TEST PERFORMANCE
# ============================================================

clean_test = metrics_df[
    metrics_df["transformation"]
    == "Original"
].copy()

clean_test_summary = (
    clean_test
    .groupby("strategy")
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

clean_test_summary.to_csv(
    OUTPUT_DIR /
    "final_test_clean_summary.csv",
    index=False
)


# ============================================================
# 17. FINAL DISPLAY
# ============================================================

print("\n\n")
print("#" * 110)
print("FINAL LOCKED TEST AGGREGATE")
print("#" * 110)

print(
    test_aggregate.to_string(
        index=False
    )
)

print("\n\n")
print("#" * 110)
print("TARGET ROBUSTNESS — LOCKED TEST")
print("#" * 110)

print(
    target_test_df.to_string(
        index=False
    )
)

print("\n\n")
print("#" * 110)
print("McNEMAR — LOCKED TEST")
print("#" * 110)

print(
    mcnemar_df.to_string(
        index=False
    )
)

print("\n\n")
print("#" * 110)
print("CLEAN TEST PERFORMANCE")
print("#" * 110)

print(
    clean_test_summary.to_string(
        index=False
    )
)

print("\n\n✅ FINAL LOCKED TEST COMPLETE")

print(
    "\nSaved to:",
    OUTPUT_DIR
)

# %% [Cell 55]
# ============================================================
# BRINJAL — ERROR ANALYSIS + GRAD-CAM
# Final Locked Test
# ============================================================

import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

TEST_DIR = Path(
    "/content/brinjal_test_all"
)

RESULT_DIR = (
    BASE_DIR /
    "augmentation_ablation_final_pytorch" /
    "final_locked_test"
)

ANALYSIS_DIR = (
    RESULT_DIR /
    "error_analysis"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PRED_FILE = (
    RESULT_DIR /
    "final_test_predictions.csv"
)

META_FILE = (
    BASE_DIR /
    "metadata.csv"
)

CHECKPOINT_DIR = (
    BASE_DIR /
    "augmentation_ablation_final_pytorch"
)

# ============================================================
# 2. LOAD DATA
# ============================================================

predictions = pd.read_csv(
    PRED_FILE
)

metadata = pd.read_csv(
    META_FILE
)

print(
    "Prediction rows:",
    len(predictions)
)

print(
    "Columns:",
    predictions.columns.tolist()
)

CLASS_NAMES = [
    "Healthy_Leaves",
    "Little_Leaf",
    "Phomopsis_Blight"
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

ID_TO_CLASS = {
    i: name
    for name, i in CLASS_TO_ID.items()
}

predictions["true_class"] = (
    predictions["y_true"]
    .map(ID_TO_CLASS)
)

predictions["predicted_class"] = (
    predictions["y_pred"]
    .map(ID_TO_CLASS)
)

# ============================================================
# 3. BASIC ERROR ANALYSIS
# ============================================================

print("\n")
print("=" * 100)
print("ERROR ANALYSIS")
print("=" * 100)

# ------------------------------------------------------------
# 3.1 Per-class classification reports
# ------------------------------------------------------------

per_class_rows = []

for strategy in sorted(
    predictions["strategy"].unique()
):

    for seed in sorted(
        predictions["seed"].unique()
    ):

        for transformation in sorted(
            predictions["transformation"].unique()
        ):

            subset = predictions[
                (
                    predictions["strategy"]
                    == strategy
                )
                &
                (
                    predictions["seed"]
                    == seed
                )
                &
                (
                    predictions["transformation"]
                    == transformation
                )
            ]

            report = classification_report(
                subset["y_true"],
                subset["y_pred"],
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0
            )

            for class_name in CLASS_NAMES:

                per_class_rows.append({

                    "strategy":
                        strategy,

                    "seed":
                        seed,

                    "transformation":
                        transformation,

                    "class":
                        class_name,

                    "precision":
                        report[
                            class_name
                        ]["precision"],

                    "recall":
                        report[
                            class_name
                        ]["recall"],

                    "f1":
                        report[
                            class_name
                        ]["f1-score"],

                    "support":
                        report[
                            class_name
                        ]["support"]

                })


per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_df.to_csv(
    ANALYSIS_DIR /
    "per_class_metrics.csv",
    index=False
)

print(
    "\n✅ Per-class metrics saved."
)


# ============================================================
# 4. CONFUSION MATRICES
# ============================================================

cm_rows = []

for strategy in sorted(
    predictions["strategy"].unique()
):

    for transformation in sorted(
        predictions["transformation"].unique()
    ):

        subset = predictions[
            (
                predictions["strategy"]
                == strategy
            )
            &
            (
                predictions[
                    "transformation"
                ]
                == transformation
            )
        ]

        cm = confusion_matrix(
            subset["y_true"],
            subset["y_pred"],
            labels=[0, 1, 2]
        )

        cm_df = pd.DataFrame(
            cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        )

        cm_df.to_csv(
            ANALYSIS_DIR /
            f"CM_{strategy}_{transformation}.csv"
        )

        cm_rows.append({

            "strategy":
                strategy,

            "transformation":
                transformation,

            "matrix":
                cm.tolist()

        })


# ============================================================
# 5. TARGETED ERROR TRANSITIONS
# ============================================================

TARGETS = {
    "Grayscale": "Grayscale",
    "Rotate": "Rotate",
    "Brightness": "Brightness"
}

transition_rows = []
example_rows = []

for strategy, target in TARGETS.items():

    for seed in sorted(
        predictions["seed"].unique()
    ):

        augmented = predictions[
            (
                predictions["strategy"]
                == strategy
            )
            &
            (
                predictions["seed"]
                == seed
            )
            &
            (
                predictions["transformation"]
                == target
            )
        ].copy()

        baseline = predictions[
            (
                predictions["strategy"]
                == "Baseline"
            )
            &
            (
                predictions["seed"]
                == seed
            )
            &
            (
                predictions["transformation"]
                == target
            )
        ].copy()

        merged = augmented.merge(
            baseline[
                [
                    "original_image_id",
                    "y_true",
                    "y_pred",
                    "correct"
                ]
            ],
            on=[
                "original_image_id",
                "y_true"
            ],
            suffixes=(
                "_augmented",
                "_baseline"
            )
        )

        assert len(merged) == 128

        base_correct = (
            merged[
                "correct_baseline"
            ].astype(bool)
        )

        aug_correct = (
            merged[
                "correct_augmented"
            ].astype(bool)
        )

        baseline_correct_aug_wrong = (
            base_correct &
            ~aug_correct
        )

        baseline_wrong_aug_correct = (
            ~base_correct &
            aug_correct
        )

        both_correct = (
            base_correct &
            aug_correct
        )

        both_wrong = (
            ~base_correct &
            ~aug_correct
        )

        transition_rows.append({

            "strategy":
                strategy,

            "seed":
                seed,

            "target":
                target,

            "baseline_correct_augmented_wrong":
                int(
                    baseline_correct_aug_wrong.sum()
                ),

            "baseline_wrong_augmented_correct":
                int(
                    baseline_wrong_aug_correct.sum()
                ),

            "both_correct":
                int(
                    both_correct.sum()
                ),

            "both_wrong":
                int(
                    both_wrong.sum()
                )

        })

        # ----------------------------------------------------
        # Save corrected examples
        # ----------------------------------------------------

        corrected = merged[
            baseline_wrong_aug_correct
        ].copy()

        corrected = corrected.head(
            10
        )

        for _, row in corrected.iterrows():

            example_rows.append({

                "strategy":
                    strategy,

                "seed":
                    seed,

                "target":
                    target,

                "original_image_id":
                    row[
                        "original_image_id"
                    ],

                "filename":
                    row[
                        "filename"
                    ],

                "true_class":
                    ID_TO_CLASS[
                        int(row["y_true"])
                    ],

                "baseline_prediction":
                    ID_TO_CLASS[
                        int(
                            row[
                                "y_pred_baseline"
                            ]
                        )
                    ],

                "augmented_prediction":
                    ID_TO_CLASS[
                        int(
                            row[
                                "y_pred_augmented"
                            ]
                        )
                    ]

            })


transition_df = pd.DataFrame(
    transition_rows
)

transition_df.to_csv(
    ANALYSIS_DIR /
    "target_error_transitions.csv",
    index=False
)

examples_df = pd.DataFrame(
    example_rows
)

examples_df.to_csv(
    ANALYSIS_DIR /
    "corrected_examples.csv",
    index=False
)

print(
    "\nTARGET ERROR TRANSITIONS"
)

print(
    transition_df.to_string(
        index=False
    )
)


# ============================================================
# 6. HARD LEAVES ON CLEAN TEST
# ============================================================

clean = predictions[
    predictions["transformation"]
    == "Original"
].copy()

# Number of models/seeds that misclassified
hard = (
    clean
    .groupby("original_image_id")
    .agg(
        true_class=(
            "true_class",
            "first"
        ),
        errors=(
            "correct",
            lambda x: int(
                (~x.astype(bool)).sum()
            )
        ),
        total_models=(
            "correct",
            "count"
        )
    )
    .reset_index()
)

hard = hard.sort_values(
    "errors",
    ascending=False
)

hard.to_csv(
    ANALYSIS_DIR /
    "hard_clean_test_leaves.csv",
    index=False
)

print(
    "\nTop hard clean-test leaves:"
)

print(
    hard.head(15).to_string(
        index=False
    )
)


# ============================================================
# 7. BUILD TEST IMAGE PATH LOOKUP
# ============================================================

image_lookup = {}

for path in TEST_DIR.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:
        continue

    image_lookup[
        path.name
    ] = str(path)


print(
    "\nIndexed test images:",
    len(image_lookup)
)


# ============================================================
# 8. GRAD-CAM UTILITIES
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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

    return model


def load_checkpoint(
    strategy,
    seed
):

    path = (
        CHECKPOINT_DIR /
        f"{strategy.lower()}_seed{seed}.pt"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        device
    )

    model.eval()

    return model


def find_last_conv(model):

    conv_layers = []

    for module in model.modules():

        if isinstance(
            module,
            nn.Conv2d
        ):

            conv_layers.append(
                module
            )

    if not conv_layers:

        raise RuntimeError(
            "No Conv2d layer found."
        )

    return conv_layers[-1]


# ImageNet preprocessing
gradcam_transform = transforms.Compose([

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


def gradcam(
    model,
    image_path,
    target_class
):

    image = Image.open(
        image_path
    ).convert("RGB")

    original = np.asarray(
        image.resize(
            (224, 224)
        )
    ).astype(
        np.float32
    ) / 255.0

    x = gradcam_transform(
        image
    ).unsqueeze(0).to(
        device
    )

    target_layer = find_last_conv(
        model
    )

    activations = None
    gradients = None

    def forward_hook(
        module,
        input,
        output
    ):
        nonlocal activations
        activations = output

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        nonlocal gradients
        gradients = grad_output[0]

    h1 = target_layer.register_forward_hook(
        forward_hook
    )

    h2 = target_layer.register_full_backward_hook(
        backward_hook
    )

    model.zero_grad(
        set_to_none=True
    )

    output = model(x)

    score = output[
        0,
        target_class
    ]

    score.backward()

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights *
        activations
    ).sum(
        dim=1
    )

    cam = torch.relu(
        cam
    )

    cam = cam.squeeze().detach().cpu().numpy()

    if cam.max() > 0:

        cam = (
            cam -
            cam.min()
        ) / (
            cam.max() -
            cam.min() +
            1e-8
        )

    cam_img = Image.fromarray(
        np.uint8(
            cam * 255
        )
    ).resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    cam = (
        np.asarray(
            cam_img
        ).astype(
            np.float32
        ) / 255.0
    )

    h1.remove()
    h2.remove()

    return (
        original,
        cam,
        output.detach().cpu().numpy()[0]
    )


# ============================================================
# 9. SELECT REPRESENTATIVE CORRECTED CASES
# ============================================================

selected_cases = []

# We focus on Seed 42 for figures
# because it is easy to reproduce and was also
# used throughout the earlier analysis.

for strategy, target in [
    ("Grayscale", "Grayscale"),
    ("Rotate", "Rotate")
]:

    subset = examples_df[
        (
            examples_df["strategy"]
            == strategy
        )
        &
        (
            examples_df["seed"]
            == 42
        )
        &
        (
            examples_df["target"]
            == target
        )
    ]

    for _, row in subset.head(2).iterrows():

        selected_cases.append(
            row.to_dict()
        )


selected_cases_df = pd.DataFrame(
    selected_cases
)

selected_cases_df.to_csv(
    ANALYSIS_DIR /
    "selected_gradcam_cases.csv",
    index=False
)

print(
    "\nSelected Grad-CAM cases:"
)

print(
    selected_cases_df[
        [
            "strategy",
            "target",
            "original_image_id",
            "true_class",
            "baseline_prediction",
            "augmented_prediction"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. GENERATE GRAD-CAM FIGURES
# ============================================================

for _, case in selected_cases_df.iterrows():

    strategy = case[
        "strategy"
    ]

    target = case[
        "target"
    ]

    leaf_id = case[
        "original_image_id"
    ]

    filename = case[
        "filename"
    ]

    basename = os.path.basename(
        filename
    )

    image_path = image_lookup.get(
        basename
    )

    if image_path is None:

        print(
            "Image not found:",
            basename
        )

        continue

    # --------------------------------------------------------
    # Baseline model
    # --------------------------------------------------------

    baseline_model = load_checkpoint(
        "Baseline",
        42
    )

    augmented_model = load_checkpoint(
        strategy,
        42
    )

    # --------------------------------------------------------
    # Determine classes
    # --------------------------------------------------------

    true_class = int(
        case.get(
            "true_class_id",
            CLASS_TO_ID[
                case["true_class"]
            ]
        )
    )

    baseline_pred = (
        CLASS_TO_ID[
            case["baseline_prediction"]
        ]
    )

    augmented_pred = (
        CLASS_TO_ID[
            case["augmented_prediction"]
        ]
    )

    # --------------------------------------------------------
    # CAMs target TRUE CLASS
    # --------------------------------------------------------

    original, cam_base, logits_base = gradcam(
        baseline_model,
        image_path,
        true_class
    )

    _, cam_aug, logits_aug = gradcam(
        augmented_model,
        image_path,
        true_class
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4)
    )

    axes[0].imshow(
        original
    )

    axes[0].set_title(
        "Original image"
    )

    axes[1].imshow(
        original
    )

    axes[1].imshow(
        cam_base,
        alpha=0.45,
        cmap="jet"
    )

    axes[1].set_title(
        "Baseline\nTrue-class Grad-CAM"
    )

    axes[2].imshow(
        original
    )

    axes[2].imshow(
        cam_aug,
        alpha=0.45,
        cmap="jet"
    )

    axes[2].set_title(
        f"{strategy}-trained\n"
        "True-class Grad-CAM"
    )

    axes[3].bar(
        range(3),
        torch.softmax(
            torch.tensor(
                logits_base
            ),
            dim=0
        ).numpy()
    )

    axes[3].set_xticks(
        range(3)
    )

    axes[3].set_xticklabels(
        [
            "Healthy",
            "Little",
            "Phomopsis"
        ],
        rotation=35
    )

    axes[3].set_title(
        "Baseline probabilities"
    )

    axes[4].bar(
        range(3),
        torch.softmax(
            torch.tensor(
                logits_aug
            ),
            dim=0
        ).numpy()
    )

    axes[4].set_xticks(
        range(3)
    )

    axes[4].set_xticklabels(
        [
            "Healthy",
            "Little",
            "Phomopsis"
        ],
        rotation=35
    )

    axes[4].set_title(
        f"{strategy}-trained\nprobabilities"
    )

    for ax in axes:
        ax.axis("off")

    # restore axes 3/4
    axes[3].axis("on")
    axes[4].axis("on")

    fig.suptitle(
        (
            f"{strategy} robustness correction | "
            f"Seed 42 | "
            f"True={case['true_class']} | "
            f"Baseline={case['baseline_prediction']} | "
            f"Augmented={case['augmented_prediction']}"
        ),
        fontsize=13
    )

    fig.tight_layout()

    safe_target = target.lower()

    output_path = (
        ANALYSIS_DIR /
        f"GradCAM_{safe_target}_"
        f"{str(leaf_id).replace('/', '_')}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    del baseline_model
    del augmented_model

    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n")
print("#" * 100)
print("ERROR ANALYSIS COMPLETE")
print("#" * 100)

print(
    "\nFiles saved to:"
)

print(
    ANALYSIS_DIR
)

print(
    "\nMain outputs:"
)

print(
    "1. per_class_metrics.csv"
)

print(
    "2. target_error_transitions.csv"
)

print(
    "3. corrected_examples.csv"
)

print(
    "4. hard_clean_test_leaves.csv"
)

print(
    "5. selected_gradcam_cases.csv"
)

print(
    "6. GradCAM_*.png"
)

print(
    "\n✅ Error Analysis + Grad-CAM completed."
)

# %% [Cell 56]
# ============================================================
# BRINJAL — FINAL STATISTICAL SYNTHESIS
# Paper-ready tables + figures
# NO TRAINING
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import t

# ============================================================
# 1. PATHS
# ============================================================

BASE = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

ABLATION = (
    BASE /
    "augmentation_ablation_final_pytorch"
)

TEST_DIR = (
    ABLATION /
    "final_locked_test"
)

PAPER_DIR = (
    BASE /
    "paper_final"
)

PAPER_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR = PAPER_DIR / "tables"
TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD FINAL TEST RESULTS
# ============================================================

test_metrics = pd.read_csv(
    TEST_DIR /
    "final_test_metrics.csv"
)

test_aggregate = pd.read_csv(
    TEST_DIR /
    "final_test_aggregate.csv"
)

target_test = pd.read_csv(
    TEST_DIR /
    "final_test_target_robustness.csv"
)

mcnemar = pd.read_csv(
    TEST_DIR /
    "final_test_mcnemar.csv"
)

clean_test = pd.read_csv(
    TEST_DIR /
    "final_test_clean_summary.csv"
)

predictions = pd.read_csv(
    TEST_DIR /
    "final_test_predictions.csv"
)


# ============================================================
# 3. LOAD FULL ROBUSTNESS RESULTS
# ============================================================

FULL_ROBUSTNESS = (
    BASE /
    "full_robustness_aggregate.csv"
)

full_robustness = pd.read_csv(
    FULL_ROBUSTNESS
)

print(
    "Loaded full robustness:",
    full_robustness.shape
)


# ============================================================
# 4. TARGET EFFECT — 95% CI ACROSS SEEDS
# ============================================================

target_ci_rows = []

for strategy, target in {
    "Grayscale": "Grayscale",
    "Rotate": "Rotate",
    "Brightness": "Brightness"
}.items():

    aug = test_metrics[
        (
            test_metrics["strategy"]
            == strategy
        )
        &
        (
            test_metrics["transformation"]
            == target
        )
    ][
        ["seed", "macro_f1"]
    ].rename(
        columns={
            "macro_f1":
                "augmented_f1"
        }
    )

    base = test_metrics[
        (
            test_metrics["strategy"]
            == "Baseline"
        )
        &
        (
            test_metrics["transformation"]
            == target
        )
    ][
        ["seed", "macro_f1"]
    ].rename(
        columns={
            "macro_f1":
                "baseline_f1"
        }
    )

    merged = aug.merge(
        base,
        on="seed"
    )

    merged["delta_f1"] = (
        merged["augmented_f1"]
        -
        merged["baseline_f1"]
    )

    values = (
        merged["delta_f1"]
        .to_numpy()
    )

    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    se = sd / np.sqrt(n)

    if n > 1:
        critical = t.ppf(
            0.975,
            df=n - 1
        )
    else:
        critical = np.nan

    ci_low = (
        mean -
        critical * se
    )

    ci_high = (
        mean +
        critical * se
    )

    target_ci_rows.append({

        "strategy":
            strategy,

        "target_transformation":
            target,

        "delta_f1_mean":
            mean,

        "delta_f1_sd":
            sd,

        "ci95_low":
            ci_low,

        "ci95_high":
            ci_high,

        "n_seeds":
            n

    })


target_ci = pd.DataFrame(
    target_ci_rows
)

target_ci.to_csv(
    TABLE_DIR /
    "table_target_effects_with_ci.csv",
    index=False
)


# ============================================================
# 5. TABLE — LOCKED TEST TARGET ROBUSTNESS
# ============================================================

table_target = target_test.copy()

table_target = table_target[
    [
        "strategy",
        "target_transformation",
        "baseline_f1_mean",
        "baseline_f1_sd",
        "augmented_f1_mean",
        "augmented_f1_sd",
        "delta_f1_mean",
        "delta_f1_sd",
        "accuracy_delta_mean",
        "balanced_accuracy_delta_mean"
    ]
]

table_target.to_csv(
    TABLE_DIR /
    "table_locked_test_target_robustness.csv",
    index=False
)


# ============================================================
# 6. TABLE — MCNEMAR
# ============================================================

table_mcnemar = mcnemar[
    [
        "strategy",
        "seed",
        "target_transformation",
        "baseline_correct_augmented_wrong",
        "baseline_wrong_augmented_correct",
        "discordant_pairs",
        "p_value",
        "fdr_q",
        "significant_after_fdr"
    ]
].copy()

table_mcnemar.to_csv(
    TABLE_DIR /
    "table_mcnemar.csv",
    index=False
)


# ============================================================
# 7. TABLE — CLEAN TEST
# ============================================================

table_clean = clean_test.copy()

table_clean.to_csv(
    TABLE_DIR /
    "table_clean_test_performance.csv",
    index=False
)


# ============================================================
# 8. CROSS-ROBUSTNESS MATRIX
# ============================================================

cross = test_aggregate.pivot(
    index="strategy",
    columns="transformation",
    values="macro_f1_mean"
)

cross.to_csv(
    TABLE_DIR /
    "table_cross_robustness_macro_f1.csv"
)


# ============================================================
# 9. FIGURE 1 — TARGETED ROBUSTNESS
# ============================================================

strategies = [
    "Baseline",
    "Grayscale",
    "Rotate",
    "Brightness"
]

targets = [
    "Grayscale",
    "Rotate",
    "Brightness"
]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(len(targets))
width = 0.18

for i, strategy in enumerate(
    strategies
):

    values = []

    for target in targets:

        row = test_aggregate[
            (
                test_aggregate["strategy"]
                == strategy
            )
            &
            (
                test_aggregate["transformation"]
                == target
            )
        ]

        values.append(
            float(
                row["macro_f1_mean"].iloc[0]
            )
        )

    ax.bar(
        x + (
            i -
            (len(strategies)-1)/2
        ) * width,
        values,
        width,
        label=strategy
    )

ax.set_xticks(x)
ax.set_xticklabels(
    targets
)

ax.set_ylabel(
    "Macro-F1"
)

ax.set_xlabel(
    "Test transformation"
)

ax.set_title(
    "Transformation Robustness on Locked Test Set"
)

ax.legend()

ax.set_ylim(
    0,
    1
)

fig.tight_layout()

fig.savefig(
    FIG_DIR /
    "figure_1_locked_test_robustness.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# 10. FIGURE 2 — TARGET AUGMENTATION EFFECT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

labels = [
    "Grayscale",
    "Rotate",
    "Brightness"
]

deltas = (
    target_test[
        "delta_f1_mean"
    ]
    .to_numpy()
)

errors = (
    target_test[
        "delta_f1_sd"
    ]
    .to_numpy()
)

x = np.arange(
    len(labels)
)

ax.bar(
    x,
    deltas,
    yerr=errors,
    capsize=5
)

ax.axhline(
    0
)

ax.set_xticks(x)

ax.set_xticklabels(
    labels
)

ax.set_ylabel(
    "Δ Macro-F1 vs Baseline"
)

ax.set_xlabel(
    "Targeted training augmentation"
)

ax.set_title(
    "Effect of Targeted Augmentation on Transformation Robustness"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR /
    "figure_2_target_augmentation_effect.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# 11. FIGURE 3 — CLEAN PERFORMANCE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

clean = clean_test.copy()

x = np.arange(
    len(clean)
)

ax.bar(
    x,
    clean["macro_f1_mean"],
    yerr=clean["macro_f1_sd"],
    capsize=5
)

ax.set_xticks(x)

ax.set_xticklabels(
    clean["strategy"]
)

ax.set_ylabel(
    "Macro-F1"
)

ax.set_xlabel(
    "Training strategy"
)

ax.set_title(
    "Clean Test Performance"
)

ax.set_ylim(
    0.8,
    1.0
)

fig.tight_layout()

fig.savefig(
    FIG_DIR /
    "figure_3_clean_test_performance.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# 12. FIGURE 4 — CROSS-ROBUSTNESS HEATMAP
# ============================================================

matrix = cross.loc[
    [
        "Baseline",
        "Grayscale",
        "Rotate",
        "Brightness"
    ],
    [
        "Original",
        "Grayscale",
        "Rotate",
        "Brightness"
    ]
].to_numpy()

fig, ax = plt.subplots(
    figsize=(8, 6)
)

im = ax.imshow(
    matrix,
    aspect="auto"
)

ax.set_xticks(
    np.arange(4)
)

ax.set_xticklabels(
    [
        "Original",
        "Grayscale",
        "Rotate",
        "Brightness"
    ],
    rotation=30
)

ax.set_yticks(
    np.arange(4)
)

ax.set_yticklabels(
    [
        "Baseline",
        "Grayscale",
        "Rotate",
        "Brightness"
    ]
)

for i in range(
    matrix.shape[0]
):

    for j in range(
        matrix.shape[1]
    ):

        ax.text(
            j,
            i,
            f"{matrix[i, j]:.3f}",
            ha="center",
            va="center"
        )

ax.set_xlabel(
    "Test condition"
)

ax.set_ylabel(
    "Training strategy"
)

ax.set_title(
    "Cross-Transformation Macro-F1"
)

fig.colorbar(
    im,
    ax=ax,
    label="Macro-F1"
)

fig.tight_layout()

fig.savefig(
    FIG_DIR /
    "figure_4_cross_robustness_heatmap.png",
    dpi=300
)

plt.show()
plt.close(fig)


# ============================================================
# 13. PAPER SUMMARY TEXT
# ============================================================

grayscale = target_test[
    target_test["strategy"]
    == "Grayscale"
].iloc[0]

rotate = target_test[
    target_test["strategy"]
    == "Rotate"
].iloc[0]

brightness = target_test[
    target_test["strategy"]
    == "Brightness"
].iloc[0]

summary = f"""
FINAL EXPERIMENTAL SYNTHESIS

LOCKED TEST SET
--------------
All models were evaluated on the same 128 test leaves.

Grayscale-targeted augmentation:
Baseline Macro-F1 =
{grayscale['baseline_f1_mean']:.4f} ±
{grayscale['baseline_f1_sd']:.4f}

Augmented Macro-F1 =
{grayscale['augmented_f1_mean']:.4f} ±
{grayscale['augmented_f1_sd']:.4f}

Mean Δ Macro-F1 =
{grayscale['delta_f1_mean']:.4f} ±
{grayscale['delta_f1_sd']:.4f}

Rotate-targeted augmentation:
Baseline Macro-F1 =
{rotate['baseline_f1_mean']:.4f} ±
{rotate['baseline_f1_sd']:.4f}

Augmented Macro-F1 =
{rotate['augmented_f1_mean']:.4f} ±
{rotate['augmented_f1_sd']:.4f}

Mean Δ Macro-F1 =
{rotate['delta_f1_mean']:.4f} ±
{rotate['delta_f1_sd']:.4f}

Brightness-targeted augmentation:
Baseline Macro-F1 =
{brightness['baseline_f1_mean']:.4f} ±
{brightness['baseline_f1_sd']:.4f}

Augmented Macro-F1 =
{brightness['augmented_f1_mean']:.4f} ±
{brightness['augmented_f1_sd']:.4f}

Mean Δ Macro-F1 =
{brightness['delta_f1_mean']:.4f} ±
{brightness['delta_f1_sd']:.4f}

INTERPRETATION
--------------
The largest robustness gains were observed for
Grayscale and Rotate targeted augmentation.

Brightness-targeted augmentation produced a
smaller change.

McNemar testing with Benjamini-Hochberg FDR
correction was performed across the 9 paired
strategy-seed comparisons.

Note:
Only 3 random seeds were used, so seed-level
uncertainty should be reported explicitly.
"""

with open(
    PAPER_DIR /
    "final_experimental_synthesis.txt",
    "w"
) as f:

    f.write(
        summary
    )


# ============================================================
# 14. FINAL PRINT
# ============================================================

print("\n")
print("=" * 100)
print("FINAL PAPER SYNTHESIS")
print("=" * 100)

print(
    target_ci.to_string(
        index=False
    )
)

print("\n")

print(
    summary
)

print("\n")
print(
    "✅ Paper-ready tables saved to:",
    TABLE_DIR
)

print(
    "✅ Paper-ready figures saved to:",
    FIG_DIR
)

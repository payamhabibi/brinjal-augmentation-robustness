
# %% [Cell 57]
# ============================================================
# ULTRA-FAST FULL-DATA BASELINE
#
# 1) Read + resize 9,520 train images ONCE
# 2) Keep them in RAM as uint8 tensors
# 3) Training reads directly from RAM
# 4) No repeated PIL / Drive I/O per epoch
#
# Full_Data_9520 | Seed 42
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import torch

from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

print("=" * 100)
print("ULTRA-FAST FULL-DATA BASELINE | SEED 42")
print("=" * 100)

# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32

CACHE_DIR = "/content/brinjal_ram_cache"

TRAIN_CACHE = os.path.join(
    CACHE_DIR,
    "train_9520_uint8.npy"
)

TRAIN_LABEL_CACHE = os.path.join(
    CACHE_DIR,
    "train_9520_labels.npy"
)

VAL_CACHE = os.path.join(
    CACHE_DIR,
    "val_2032_uint8.npy"
)

VAL_LABEL_CACHE = os.path.join(
    CACHE_DIR,
    "val_2032_labels.npy"
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

ROOT = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)

STANDARD_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_standard_pytorch"
)

os.makedirs(
    STANDARD_DIR,
    exist_ok=True
)

# ============================================================
# 2. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    os.path.join(
        ROOT,
        "metadata.csv"
    )
)

filename_col = "filename"
class_col = "class_label"
split_col = "data_split"

train_meta = metadata[
    metadata[split_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("train")
].copy()

val_meta = metadata[
    metadata[split_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("val")
].copy()

assert len(train_meta) == 9520
assert len(val_meta) == 2032

# ============================================================
# 3. USE EXISTING LOCAL CACHE
# ============================================================

LOCAL_TRAIN_DIR = (
    "/content/brinjal_train_9520"
)

LOCAL_VAL_DIR = (
    "/content/brinjal_val_original"
)

assert os.path.isdir(
    LOCAL_TRAIN_DIR
), (
    "Local train cache not found. "
    "Run the previous cache-copy cell first."
)

assert os.path.isdir(
    LOCAL_VAL_DIR
), (
    "Local validation cache not found. "
    "Run the previous cache-copy cell first."
)

# Build local paths
train_paths = [
    os.path.join(
        LOCAL_TRAIN_DIR,
        os.path.basename(
            str(x)
        )
    )
    for x in train_meta[filename_col]
]

val_paths = [
    os.path.join(
        LOCAL_VAL_DIR,
        os.path.basename(
            str(x)
        )
    )
    for x in val_meta[filename_col]
]

assert all(
    os.path.exists(p)
    for p in train_paths
)

assert all(
    os.path.exists(p)
    for p in val_paths
)

# ============================================================
# 4. CLASS MAPPING
# ============================================================

CLASS_TO_IDX = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

train_labels = (
    train_meta[class_col]
    .map(CLASS_TO_IDX)
    .astype(np.int64)
    .to_numpy()
)

val_labels = (
    val_meta[class_col]
    .map(CLASS_TO_IDX)
    .astype(np.int64)
    .to_numpy()
)

# ============================================================
# 5. FAST IMAGE LOADER
# ============================================================

def load_resize_image(
    path
):
    """
    Read RGB image and resize once.
    Returns H x W x 3 uint8 numpy array.
    """

    with Image.open(path) as img:

        img = img.convert("RGB")

        img = img.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        return np.asarray(
            img,
            dtype=np.uint8
        )

# ============================================================
# 6. BUILD / LOAD RAM CACHE
# ============================================================

def build_ram_cache(
    paths,
    image_cache_path,
    label_array
):

    # --------------------------------------------------------
    # Existing image cache
    # --------------------------------------------------------

    if os.path.exists(
        image_cache_path
    ):

        print(
            f"\n✅ Cache already exists:\n"
            f"{image_cache_path}"
        )

        images = np.load(
            image_cache_path,
            mmap_mode=None
        )

        assert images.shape == (
            len(paths),
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )

        return images

    # --------------------------------------------------------
    # Build cache
    # --------------------------------------------------------

    print(
        f"\nBuilding RAM cache for "
        f"{len(paths)} images..."
    )

    start = time.time()

    images = np.empty(
        (
            len(paths),
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        dtype=np.uint8
    )

    def load_one(item):

        idx, path = item

        return (
            idx,
            load_resize_image(path)
        )

    jobs = list(
        enumerate(paths)
    )

    completed = 0

    # Moderate worker count to avoid
    # overwhelming Colab RAM/CPU.
    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        for idx, arr in executor.map(
            load_one,
            jobs
        ):

            images[idx] = arr

            completed += 1

            if (
                completed % 500 == 0
                or
                completed == len(paths)
            ):

                elapsed = (
                    time.time()
                    - start
                )

                print(
                    f"{completed}/"
                    f"{len(paths)} "
                    f"| {elapsed:.1f}s"
                )

    np.save(
        image_cache_path,
        images
    )

    print(
        f"✅ Cache saved in "
        f"{time.time() - start:.1f}s"
    )

    return images


# ============================================================
# 7. TRAIN RAM CACHE
# ============================================================

train_images = build_ram_cache(
    train_paths,
    TRAIN_CACHE,
    train_labels
)

assert train_images.shape == (
    9520,
    224,
    224,
    3
)

# ============================================================
# 8. VALIDATION RAM CACHE
# ============================================================

val_images = build_ram_cache(
    val_paths,
    VAL_CACHE,
    val_labels
)

assert val_images.shape == (
    2032,
    224,
    224,
    3
)

# ============================================================
# 9. LOAD LABEL CACHE
# ============================================================

if not os.path.exists(
    TRAIN_LABEL_CACHE
):

    np.save(
        TRAIN_LABEL_CACHE,
        train_labels
    )

if not os.path.exists(
    VAL_LABEL_CACHE
):

    np.save(
        VAL_LABEL_CACHE,
        val_labels
    )

# ============================================================
# 10. CONVERT TO TORCH TENSORS
# ============================================================

# HWC -> CHW
train_tensor = torch.from_numpy(
    train_images
).permute(
    0,
    3,
    1,
    2
).contiguous()

val_tensor = torch.from_numpy(
    val_images
).permute(
    0,
    3,
    1,
    2
).contiguous()

train_label_tensor = torch.from_numpy(
    train_labels
)

val_label_tensor = torch.from_numpy(
    val_labels
)

print("\nRAM tensors:")
print(
    "Train:",
    train_tensor.shape,
    train_tensor.dtype
)

print(
    "Val:",
    val_tensor.shape,
    val_tensor.dtype
)

# ============================================================
# 11. TENSOR DATASETS
# ============================================================

train_dataset = TensorDataset(
    train_tensor,
    train_label_tensor
)

val_dataset = TensorDataset(
    val_tensor,
    val_label_tensor
)

# ============================================================
# 12. FAST LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# 13. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

assert device.type == "cuda"

print(
    "\nDevice:",
    device
)

# ============================================================
# 14. GPU NORMALIZATION
# ============================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(
    1, 3, 1, 1
)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(
    1, 3, 1, 1
)

def prepare_images(
    images
):

    images = images.to(
        device,
        non_blocking=True
    )

    images = images.float().div_(
        255.0
    )

    images = (
        images - MEAN
    ) / STD

    return images

# ============================================================
# 15. FIXED INPUT OPTIMIZATION
# ============================================================

torch.backends.cudnn.benchmark = True

# ============================================================
# 16. SEED
# ============================================================

seed_everything(
    SEED
)

# ============================================================
# 17. MODEL
# ============================================================

model = build_model()

model = model.to(
    device
)

criterion = (
    torch.nn.CrossEntropyLoss()
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
)

# ============================================================
# 18. VALIDATION
# ============================================================

def evaluate_validation():

    model.eval()

    total_loss = 0.0
    total = 0

    predictions = []
    labels_all = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = prepare_images(
                images
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
                *
                labels.size(0)
            )

            total += labels.size(0)

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                labels.cpu().numpy()
            )

    return {
        "loss":
            total_loss / total,

        "accuracy":
            accuracy_score(
                labels_all,
                predictions
            ),

        "macro_f1":
            f1_score(
                labels_all,
                predictions,
                average="macro",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels_all,
                predictions
            )
    }


# ============================================================
# 19. TRAIN
# ============================================================

best_val_loss = float(
    "inf"
)

best_epoch = 0
patience_counter = 0

history = []

checkpoint_path = os.path.join(
    STANDARD_DIR,
    "full_data_9520_seed42_fast.pt"
)

history_path = os.path.join(
    STANDARD_DIR,
    "full_data_9520_seed42_fast_history.csv"
)

validation_path = os.path.join(
    STANDARD_DIR,
    "full_data_9520_seed42_fast_validation.csv"
)

print("\n")
print("=" * 100)
print("TRAINING | FULL_DATA_9520 | SEED 42")
print("=" * 100)

for epoch in range(
    1,
    21
):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = prepare_images(
            images
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

        running_loss += (
            loss.item()
            *
            labels.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / total
    )

    train_acc = (
        correct / total
    )

    val_metrics = (
        evaluate_validation()
    )

    val_loss = (
        val_metrics["loss"]
    )

    scheduler.step(
        val_loss
    )

    lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    elapsed = (
        time.time()
        - epoch_start
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "strategy":
                    "Full_Data_9520",

                "seed":
                    SEED,

                "best_epoch":
                    best_epoch,

                "best_val_loss":
                    best_val_loss
            },
            checkpoint_path
        )

        best_flag = " ✅ BEST"

    else:

        patience_counter += 1

        best_flag = ""

    history.append({
        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "val_loss":
            val_loss,

        "val_macro_f1":
            val_metrics["macro_f1"],

        "val_accuracy":
            val_metrics["accuracy"],

        "val_balanced_accuracy":
            val_metrics[
                "balanced_accuracy"
            ],

        "lr":
            lr,

        "epoch_time_sec":
            elapsed
    })

    print(
        f"Epoch {epoch:02d}/20 | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Orig Val Loss {val_loss:.4f} | "
        f"Orig Val F1 "
        f"{val_metrics['macro_f1']:.4f} | "
        f"Orig Val Acc "
        f"{val_metrics['accuracy']:.4f} | "
        f"{elapsed:.1f}s"
        f"{best_flag}"
    )

    if patience_counter >= 4:

        print(
            "\n⏹ Early stopping"
        )

        break

# ============================================================
# 20. SAVE RESULTS
# ============================================================

pd.DataFrame(
    history
).to_csv(
    history_path,
    index=False
)

best_checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

final_val = (
    evaluate_validation()
)

result = pd.DataFrame([
    {
        "strategy":
            "Full_Data_9520",

        "seed":
            SEED,

        "training_samples":
            9520,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "original_val_accuracy":
            final_val["accuracy"],

        "original_val_macro_f1":
            final_val["macro_f1"],

        "original_val_balanced_accuracy":
            final_val[
                "balanced_accuracy"
            ],

        "checkpoint":
            checkpoint_path
    }
])

result.to_csv(
    validation_path,
    index=False
)

# ============================================================
# 21. FINAL
# ============================================================

print("\n")
print("=" * 100)
print("✅ FULL_DATA_9520 | SEED 42 COMPLETE")
print("=" * 100)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best val loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Val Macro-F1: "
    f"{final_val['macro_f1']:.4f}"
)

print(
    f"Val Accuracy: "
    f"{final_val['accuracy']:.4f}"
)

print(
    f"Checkpoint:\n"
    f"{checkpoint_path}"
)

print(
    "\n⚠️ LOCKED TEST WAS NOT USED."
)

print("=" * 100)

# %% [Cell 58]
# ============================================================
# FULL-DATA 9520 | SEEDS 1337 + 2026
# Uses EXISTING RAM cache
# No Drive I/O
# ============================================================

SEEDS_TO_RUN = [1337, 2026]

for SEED in SEEDS_TO_RUN:

    print("\n")
    print("=" * 100)
    print(f"FULL_DATA_9520 | SEED {SEED}")
    print("=" * 100)

    seed_everything(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = build_model().to(device)

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
        STANDARD_DIR,
        f"full_data_9520_seed{SEED}_fast.pt"
    )

    history_path = os.path.join(
        STANDARD_DIR,
        f"full_data_9520_seed{SEED}_fast_history.csv"
    )

    validation_path = os.path.join(
        STANDARD_DIR,
        f"full_data_9520_seed{SEED}_fast_validation.csv"
    )

    for epoch in range(1, 21):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = prepare_images(images)

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
                loss.item() * labels.size(0)
            )

            preds = outputs.argmax(dim=1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_metrics = evaluate_validation()

        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - epoch_start

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "strategy":
                        "Full_Data_9520",
                    "seed":
                        SEED,
                    "best_epoch":
                        best_epoch,
                    "best_val_loss":
                        best_val_loss
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
            "val_balanced_accuracy":
                val_metrics["balanced_accuracy"],
            "lr": lr,
            "epoch_time_sec": elapsed
        })

        print(
            f"Epoch {epoch:02d}/20 | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Orig Val Loss {val_loss:.4f} | "
            f"Orig Val F1 {val_f1:.4f} | "
            f"Orig Val Acc {val_acc:.4f} | "
            f"LR {lr:.6f} | "
            f"{elapsed:.1f}s"
            f"{best_flag}"
        )

        if patience_counter >= 4:

            print("\n⏹ Early stopping")
            break

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_val = evaluate_validation()

    result = pd.DataFrame([
        {
            "strategy":
                "Full_Data_9520",
            "seed":
                SEED,
            "training_samples":
                9520,
            "best_epoch":
                best_epoch,
            "best_val_loss":
                best_val_loss,
            "original_val_accuracy":
                final_val["accuracy"],
            "original_val_macro_f1":
                final_val["macro_f1"],
            "original_val_balanced_accuracy":
                final_val["balanced_accuracy"],
            "checkpoint":
                checkpoint_path
        }
    ])

    result.to_csv(
        validation_path,
        index=False
    )

    print("\n✅ COMPLETE")
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Val Macro-F1: "
        f"{final_val['macro_f1']:.4f}"
    )
    print(
        f"Val Accuracy: "
        f"{final_val['accuracy']:.4f}"
    )
    print(
        f"Checkpoint: {checkpoint_path}"
    )

print("\n" + "=" * 100)
print("✅ FULL-DATA SEEDS 1337 + 2026 COMPLETE")
print("=" * 100)

# %% [Cell 59]
# ============================================================
# MASTER STEP:
#
# A) Train RandAugment × 3 seeds
# B) Train AugMix × 3 seeds
# C) Locked-test evaluation for:
#       Full_Data_9520
#       RandAugment
#       AugMix
#    × 3 seeds × 16 transformations
#
# IMPORTANT:
# - Uses existing RAM training cache
# - Uses ORIGINAL 595 training leaves for RandAugment/AugMix
# - Full_Data_9520 uses all 9,520 training images
# - Validation = original validation only
# - Test = inference only
# ============================================================

import os
import time
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

ROOT = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)

STANDARD_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_standard_pytorch"
)

os.makedirs(
    STANDARD_DIR,
    exist_ok=True
)

RAW_DIR = os.path.join(
    ROOT,
    "Raw"
)

AUG_DIR = os.path.join(
    ROOT,
    "Augmented"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

TEST_RAM_CACHE = (
    "/content/brinjal_test_2048_ram"
)

os.makedirs(
    TEST_RAM_CACHE,
    exist_ok=True
)

# ============================================================
# 2. VERIFY EXISTING TRAINING RAM CACHE
# ============================================================

assert "train_tensor" in globals(), (
    "train_tensor not found. "
    "Run the optimized Full_Data_9520 cache cell first."
)

assert "val_tensor" in globals(), (
    "val_tensor not found."
)

assert "original_train_tensor" in globals(), (
    "original_train_tensor not found."
)

assert "original_train_labels" in globals(), (
    "original_train_labels not found."
)

assert train_tensor.shape == (
    9520, 3, 224, 224
)

assert original_train_tensor.shape == (
    595, 3, 224, 224
)

# ============================================================
# 3. METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

filename_col = "filename"
class_col = "class_label"
transformation_col = "preprocessing_technique"
split_col = "data_split"

# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

assert device.type == "cuda"

print("=" * 100)
print("MASTER STANDARD AUGMENTATION EXPERIMENT")
print("=" * 100)

print(
    "Device:",
    device
)

# ============================================================
# 5. NORMALIZATION
# ============================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(
    1, 3, 1, 1
)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(
    1, 3, 1, 1
)


def normalize_batch(images):

    images = images.to(
        device,
        non_blocking=True
    )

    if images.dtype == torch.uint8:

        images = images.float().div_(
            255.0
        )

    return (
        images - MEAN
    ) / STD


# ============================================================
# 6. VALIDATION LOADER
# ============================================================

val_labels = val_label_tensor

class RAMValidationDataset(Dataset):

    def __init__(
        self,
        images,
        labels
    ):

        self.images = images
        self.labels = labels

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        idx
    ):

        return (
            self.images[idx],
            self.labels[idx]
        )


val_dataset = RAMValidationDataset(
    val_tensor,
    val_labels
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# 7. AUGMENTATIONS
# ============================================================

rand_augment = transforms.RandAugment(
    num_ops=2,
    magnitude=9
)

augmix = transforms.AugMix(
    severity=3,
    mixture_width=3,
    chain_depth=-1,
    alpha=1.0
)

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# ============================================================
# 8. STANDARD TRAINING DATASET
# ============================================================

class StandardTrainDataset(
    Dataset
):

    def __init__(
        self,
        images,
        labels,
        strategy
    ):

        self.images = images
        self.labels = labels
        self.strategy = strategy

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        idx
    ):

        image = self.images[
            idx
        ]

        label = int(
            self.labels[
                idx
            ]
        )

        # -----------------------------------------------
        # Convert cached uint8 tensor to PIL
        # -----------------------------------------------

        image = to_pil(
            image
        )

        # -----------------------------------------------
        # Online augmentation
        # -----------------------------------------------

        if self.strategy == "RandAugment":

            image = rand_augment(
                image
            )

        elif self.strategy == "AugMix":

            image = augmix(
                image
            )

        else:

            raise ValueError(
                f"Unknown strategy: "
                f"{self.strategy}"
            )

        # PIL -> [0,1] Tensor
        image = to_tensor(
            image
        )

        return (
            image,
            label
        )


# ============================================================
# 9. VALIDATION FUNCTION
# ============================================================

criterion_global = (
    torch.nn.CrossEntropyLoss()
)


def evaluate_validation(
    model
):

    model.eval()

    total_loss = 0.0
    total = 0

    predictions = []
    labels_all = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = normalize_batch(
                images
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion_global(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                *
                labels.size(0)
            )

            total += (
                labels.size(0)
            )

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                labels.cpu().numpy()
            )

    return {
        "loss":
            total_loss / total,

        "accuracy":
            accuracy_score(
                labels_all,
                predictions
            ),

        "macro_f1":
            f1_score(
                labels_all,
                predictions,
                average="macro",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels_all,
                predictions
            )
    }


# ============================================================
# 10. TRAIN RAND AUGMENT + AUGMIX
# ============================================================

SEEDS = [
    42,
    1337,
    2026
]

TRAINING_STRATEGIES = [
    "RandAugment",
    "AugMix"
]

training_results = []

for strategy in TRAINING_STRATEGIES:

    for SEED in SEEDS:

        checkpoint_path = os.path.join(
            STANDARD_DIR,
            f"{strategy.lower()}_seed{SEED}.pt"
        )

        history_path = os.path.join(
            STANDARD_DIR,
            f"{strategy.lower()}_seed{SEED}_history.csv"
        )

        validation_path = os.path.join(
            STANDARD_DIR,
            f"{strategy.lower()}_seed{SEED}_validation.csv"
        )

        # ----------------------------------------------------
        # Skip completed runs
        # ----------------------------------------------------

        if os.path.exists(
            checkpoint_path
        ):

            print(
                f"\n⏭️ {strategy} Seed {SEED} "
                f"checkpoint already exists."
            )

            if os.path.exists(
                validation_path
            ):

                old_result = pd.read_csv(
                    validation_path
                )

                if len(old_result) > 0:

                    training_results.append(
                        old_result.iloc[0].to_dict()
                    )

            continue

        print("\n")
        print("=" * 100)
        print(
            f"TRAINING | {strategy} | SEED {SEED}"
        )
        print("=" * 100)

        seed_everything(
            SEED
        )

        # ----------------------------------------------------
        # ORIGINAL 595 TRAINING LEAVES ONLY
        # ----------------------------------------------------

        train_dataset = StandardTrainDataset(
            original_train_tensor,
            original_train_labels,
            strategy
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        print(
            "Training samples:",
            len(train_dataset)
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = build_model().to(
            device
        )

        criterion = (
            torch.nn.CrossEntropyLoss()
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        )

        best_val_loss = float(
            "inf"
        )

        best_epoch = 0
        patience_counter = 0

        history = []

        # ----------------------------------------------------
        # Epochs
        # ----------------------------------------------------

        for epoch in range(
            1,
            21
        ):

            epoch_start = time.time()

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = normalize_batch(
                    images
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

                running_loss += (
                    loss.item()
                    *
                    labels.size(0)
                )

                preds = outputs.argmax(
                    dim=1
                )

                correct += (
                    preds == labels
                ).sum().item()

                total += labels.size(0)

            train_loss = (
                running_loss / total
            )

            train_acc = (
                correct / total
            )

            val_metrics = (
                evaluate_validation(
                    model
                )
            )

            val_loss = (
                val_metrics["loss"]
            )

            scheduler.step(
                val_loss
            )

            lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            elapsed = (
                time.time()
                - epoch_start
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "strategy":
                            strategy,

                        "seed":
                            SEED,

                        "best_epoch":
                            best_epoch,

                        "best_val_loss":
                            best_val_loss
                    },
                    checkpoint_path
                )

                best_flag = (
                    " ✅ BEST"
                )

            else:

                patience_counter += 1
                best_flag = ""

            history.append(
                {
                    "epoch":
                        epoch,

                    "train_loss":
                        train_loss,

                    "train_accuracy":
                        train_acc,

                    "val_loss":
                        val_loss,

                    "val_macro_f1":
                        val_metrics[
                            "macro_f1"
                        ],

                    "val_accuracy":
                        val_metrics[
                            "accuracy"
                        ],

                    "val_balanced_accuracy":
                        val_metrics[
                            "balanced_accuracy"
                        ],

                    "lr":
                        lr,

                    "epoch_time_sec":
                        elapsed
                }
            )

            print(
                f"Epoch {epoch:02d}/20 | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Orig Val Loss {val_loss:.4f} | "
                f"Orig Val F1 "
                f"{val_metrics['macro_f1']:.4f} | "
                f"Orig Val Acc "
                f"{val_metrics['accuracy']:.4f} | "
                f"LR {lr:.6f} | "
                f"{elapsed:.1f}s"
                f"{best_flag}"
            )

            if patience_counter >= 4:

                print(
                    "\n⏹ Early stopping"
                )

                break

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False
        )

        # ----------------------------------------------------
        # Reload best
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        final_val = evaluate_validation(
            model
        )

        result = {
            "strategy":
                strategy,

            "seed":
                SEED,

            "training_samples":
                595,

            "best_epoch":
                best_epoch,

            "best_val_loss":
                best_val_loss,

            "original_val_accuracy":
                final_val[
                    "accuracy"
                ],

            "original_val_macro_f1":
                final_val[
                    "macro_f1"
                ],

            "original_val_balanced_accuracy":
                final_val[
                    "balanced_accuracy"
                ],

            "checkpoint":
                checkpoint_path
        }

        training_results.append(
            result
        )

        pd.DataFrame(
            [result]
        ).to_csv(
            validation_path,
            index=False
        )

        print(
            "\n✅ COMPLETE"
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Val Macro-F1:",
            f"{final_val['macro_f1']:.4f}"
        )

        print(
            "Val Accuracy:",
            f"{final_val['accuracy']:.4f}"
        )

        print(
            "\n⚠️ LOCKED TEST NOT USED."
        )


# ============================================================
# 11. SAVE TRAINING SUMMARY
# ============================================================

training_summary = pd.DataFrame(
    training_results
)

training_summary = (
    training_summary
    .sort_values(
        [
            "strategy",
            "seed"
        ]
    )
    .reset_index(
        drop=True
    )
)

training_summary_path = os.path.join(
    STANDARD_DIR,
    "randaugment_augmix_all_seeds_summary.csv"
)

training_summary.to_csv(
    training_summary_path,
    index=False
)

print("\n")
print("=" * 100)
print("✅ RAND AUGMENT + AUGMIX TRAINING COMPLETE")
print("=" * 100)

display(
    training_summary
)

# ============================================================
# 12. BUILD TEST FILE LOOKUP
# ============================================================

print("\n")
print("=" * 100)
print("BUILDING LOCKED TEST RAM CACHE")
print("=" * 100)

test_meta = metadata[
    metadata[split_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("test")
].copy()

assert len(test_meta) == 2048

# ------------------------------------------------------------
# Drive file lookup
# ------------------------------------------------------------

all_drive_files = []

for base_dir in [
    RAW_DIR,
    AUG_DIR
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

                all_drive_files.append(
                    os.path.join(
                        root,
                        fname
                    )
                )

assert len(all_drive_files) == 13600

drive_lookup = {
    os.path.basename(path):
        path
    for path in all_drive_files
}

# ------------------------------------------------------------
# Test arrays
# ------------------------------------------------------------

test_paths = []

test_labels_np = []

test_names = []

test_transforms = []

test_original_ids = []

for _, row in test_meta.iterrows():

    fname = os.path.basename(
        str(row[filename_col])
    )

    path = drive_lookup.get(
        fname
    )

    assert path is not None, (
        f"Missing test file: {fname}"
    )

    test_paths.append(
        path
    )

    test_labels_np.append(
        {
            "Healthy_Leaves": 0,
            "Little_Leaf": 1,
            "Phomopsis_Blight": 2
        }[
            row[class_col]
        ]
    )

    test_names.append(
        fname
    )

    test_transforms.append(
        row[transformation_col]
    )

    test_original_ids.append(
        row["original_image_id"]
    )

test_labels_np = np.asarray(
    test_labels_np,
    dtype=np.int64
)

# ============================================================
# 13. TEST RAM CACHE
# ============================================================

TEST_CACHE_FILE = os.path.join(
    TEST_RAM_CACHE,
    "test_2048_uint8.npy"
)

if os.path.exists(
    TEST_CACHE_FILE
):

    print(
        "✅ Existing test RAM cache found."
    )

    test_images = np.load(
        TEST_CACHE_FILE
    )

else:

    print(
        "Building test image cache..."
    )

    test_images = np.empty(
        (
            2048,
            224,
            224,
            3
        ),
        dtype=np.uint8
    )

    start = time.time()

    from concurrent.futures import ThreadPoolExecutor

    def load_test_one(item):

        idx, path = item

        with Image.open(path) as img:

            img = img.convert(
                "RGB"
            )

            img = img.resize(
                (
                    224,
                    224
                ),
                Image.Resampling.BILINEAR
            )

            arr = np.asarray(
                img,
                dtype=np.uint8
            )

        return idx, arr

    jobs = list(
        enumerate(test_paths)
    )

    completed = 0

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        for idx, arr in executor.map(
            load_test_one,
            jobs
        ):

            test_images[idx] = arr

            completed += 1

            if (
                completed % 500 == 0
                or
                completed == 2048
            ):

                print(
                    f"{completed}/2048"
                )

    np.save(
        TEST_CACHE_FILE,
        test_images
    )

    print(
        f"✅ Test cache built in "
        f"{time.time() - start:.1f}s"
    )

assert test_images.shape == (
    2048,
    224,
    224,
    3
)

# ============================================================
# 14. TEST TENSOR
# ============================================================

test_tensor = torch.from_numpy(
    test_images
).permute(
    0,
    3,
    1,
    2
).contiguous()

test_labels_tensor = torch.from_numpy(
    test_labels_np
)

test_names = np.asarray(
    test_names
)

test_transforms = np.asarray(
    test_transforms
)

test_original_ids = np.asarray(
    test_original_ids
)

# ============================================================
# 15. TEST DATASET
# ============================================================

class TensorTestDataset(
    Dataset
):

    def __init__(
        self,
        images,
        labels,
        names,
        transformations,
        original_ids
    ):

        self.images = images
        self.labels = labels
        self.names = names
        self.transformations = transformations
        self.original_ids = original_ids

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        idx
    ):

        return (
            self.images[idx],
            self.labels[idx],
            idx
        )


test_dataset = TensorTestDataset(
    test_tensor,
    test_labels_tensor,
    test_names,
    test_transforms,
    test_original_ids
)

# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================

def evaluate_test_condition(
    model,
    indices
):

    subset = torch.utils.data.Subset(
        test_dataset,
        indices.tolist()
    )

    loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_indices = []

    with torch.no_grad():

        for images, labels, idxs in loader:

            images = normalize_batch(
                images
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            preds = outputs.argmax(
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_indices.extend(
                idxs.numpy()
            )

    all_preds = np.asarray(
        all_preds
    )

    all_labels = np.asarray(
        all_labels
    )

    return {
        "predictions":
            all_preds,

        "labels":
            all_labels,

        "accuracy":
            accuracy_score(
                all_labels,
                all_preds
            ),

        "macro_f1":
            f1_score(
                all_labels,
                all_preds,
                average="macro",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                all_labels,
                all_preds
            ),

        "indices":
            np.asarray(
                all_indices
            )
    }


# ============================================================
# 17. STANDARD TEST STRATEGIES
# ============================================================

STANDARD_TEST_STRATEGIES = [
    "Full_Data_9520",
    "RandAugment",
    "AugMix"
]

test_results = []
test_prediction_records = []

for strategy in STANDARD_TEST_STRATEGIES:

    for SEED in SEEDS:

        if strategy == "Full_Data_9520":

            checkpoint_path = os.path.join(
                STANDARD_DIR,
                f"full_data_9520_seed{SEED}_fast.pt"
            )

        elif strategy == "RandAugment":

            checkpoint_path = os.path.join(
                STANDARD_DIR,
                f"randaugment_seed{SEED}.pt"
            )

        else:

            checkpoint_path = os.path.join(
                STANDARD_DIR,
                f"augmix_seed{SEED}.pt"
            )

        assert os.path.exists(
            checkpoint_path
        ), (
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

        print("\n")
        print("=" * 100)
        print(
            f"LOCKED TEST | "
            f"{strategy} | SEED {SEED}"
        )
        print("=" * 100)

        model = build_model().to(
            device
        )

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

        for transformation in TRANSFORMATIONS:

            indices = np.flatnonzero(
                test_transforms
                ==
                transformation
            )

            assert len(
                indices
            ) == 128

            evaluation = (
                evaluate_test_condition(
                    model,
                    indices
                )
            )

            test_results.append(
                {
                    "strategy":
                        strategy,

                    "seed":
                        SEED,

                    "transformation":
                        transformation,

                    "n":
                        128,

                    "accuracy":
                        evaluation[
                            "accuracy"
                        ],

                    "macro_f1":
                        evaluation[
                            "macro_f1"
                        ],

                    "balanced_accuracy":
                        evaluation[
                            "balanced_accuracy"
                        ]
                }
            )

            preds = (
                evaluation[
                    "predictions"
                ]
            )

            labels = (
                evaluation[
                    "labels"
                ]
            )

            original_indices = (
                evaluation[
                    "indices"
                ]
            )

            for j in range(
                len(preds)
            ):

                local_idx = int(
                    original_indices[j]
                )

                true_label = int(
                    labels[j]
                )

                pred_label = int(
                    preds[j]
                )

                test_prediction_records.append(
                    {
                        "strategy":
                            strategy,

                        "seed":
                            SEED,

                        "transformation":
                            transformation,

                        "original_image_id":
                            test_original_ids[
                                local_idx
                            ],

                        "filename":
                            test_names[
                                local_idx
                            ],

                        "y_true":
                            true_label,

                        "y_pred":
                            pred_label,

                        "correct":
                            int(
                                true_label
                                ==
                                pred_label
                            )
                    }
                )

            print(
                f"{transformation:15s} | "
                f"Acc={evaluation['accuracy']:.4f} | "
                f"Macro-F1="
                f"{evaluation['macro_f1']:.4f} | "
                f"BalAcc="
                f"{evaluation['balanced_accuracy']:.4f}"
            )

# ============================================================
# 18. SAVE STANDARD TEST RESULTS
# ============================================================

standard_test_df = pd.DataFrame(
    test_results
)

standard_predictions_df = (
    pd.DataFrame(
        test_prediction_records
    )
)

standard_metrics_path = os.path.join(
    STANDARD_DIR,
    "standard_locked_test_metrics.csv"
)

standard_predictions_path = os.path.join(
    STANDARD_DIR,
    "standard_locked_test_predictions.csv"
)

standard_test_df.to_csv(
    standard_metrics_path,
    index=False
)

standard_predictions_df.to_csv(
    standard_predictions_path,
    index=False
)

# ============================================================
# 19. AGGREGATE
# ============================================================

standard_aggregate = (
    standard_test_df
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

standard_aggregate_path = os.path.join(
    STANDARD_DIR,
    "standard_locked_test_aggregate.csv"
)

standard_aggregate.to_csv(
    standard_aggregate_path,
    index=False
)

# ============================================================
# 20. TARGET CONDITIONS
# ============================================================

target_conditions = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness",
    "Gaussian_Blur"
]

target_standard_summary = (
    standard_aggregate[
        standard_aggregate[
            "transformation"
        ].isin(
            target_conditions
        )
    ]
    .copy()
    .sort_values(
        [
            "strategy",
            "transformation"
        ]
    )
)

target_standard_path = os.path.join(
    STANDARD_DIR,
    "standard_locked_test_target_summary.csv"
)

target_standard_summary.to_csv(
    target_standard_path,
    index=False
)

# ============================================================
# 21. MACRO-F1 WIDE
# ============================================================

standard_f1_wide = (
    standard_aggregate
    .pivot(
        index="strategy",
        columns="transformation",
        values="macro_f1_mean"
    )
)

standard_f1_wide_path = os.path.join(
    STANDARD_DIR,
    "standard_locked_test_macro_f1_wide.csv"
)

standard_f1_wide.to_csv(
    standard_f1_wide_path
)

# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n\n")
print("=" * 100)
print("✅ MASTER STANDARD EXPERIMENT COMPLETE")
print("=" * 100)

print("\nTraining results:")
display(
    training_summary
)

print("\nLocked-test target summary:")
display(
    target_standard_summary
)

print("\nLocked-test Macro-F1 across all transformations:")
display(
    standard_f1_wide
)

print("\nSaved:")
print(
    training_summary_path
)

print(
    standard_metrics_path
)

print(
    standard_predictions_path
)

print(
    standard_aggregate_path
)

print(
    target_standard_path
)

print(
    standard_f1_wide_path
)

print("\n⚠️ TEST DATA WAS USED FOR INFERENCE ONLY.")
print("⚠️ NO TEST DATA WAS USED FOR CHECKPOINT SELECTION.")
print("=" * 100)

# %% [Cell 60]
# ============================================================
# DEBUG RUN | RandAugment + AugMix | Seed 42
# فقط برای sanity check — هنوز TEST اجرا نمی‌شود
# ============================================================

import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEBUG_SEED = 42

# ------------------------------------------------------------
# 1. Mild augmentation settings
# ------------------------------------------------------------

rand_augment_debug = transforms.RandAugment(
    num_ops=1,
    magnitude=3
)

augmix_debug = transforms.AugMix(
    severity=1,
    mixture_width=2,
    chain_depth=1,
    alpha=1.0
)

to_pil_debug = transforms.ToPILImage()
to_tensor_debug = transforms.ToTensor()


# ------------------------------------------------------------
# 2. Dataset
# ------------------------------------------------------------

class DebugTrainDataset(Dataset):

    def __init__(self, images, labels, strategy):
        self.images = images
        self.labels = labels
        self.strategy = strategy

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = int(self.labels[idx])

        # uint8 CHW -> PIL RGB
        image = to_pil_debug(image)

        if self.strategy == "RandAugment":
            image = rand_augment_debug(image)

        elif self.strategy == "AugMix":
            image = augmix_debug(image)

        else:
            raise ValueError(self.strategy)

        # PIL -> [0,1]
        image = to_tensor_debug(image)

        return image, label


# ------------------------------------------------------------
# 3. Validation loader already exists
# ------------------------------------------------------------

# normalize_batch() and evaluate_validation()
# already exist in your runtime.


# ------------------------------------------------------------
# 4. One-seed test
# ------------------------------------------------------------

DEBUG_RESULTS = []

for strategy in ["RandAugment", "AugMix"]:

    print("\n" + "=" * 90)
    print(f"DEBUG TRAINING | {strategy} | SEED {DEBUG_SEED}")
    print("=" * 90)

    seed_everything(DEBUG_SEED)

    train_dataset = DebugTrainDataset(
        original_train_tensor,
        original_train_labels,
        strategy
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = build_model().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # فقط 3 epoch برای diagnostic
    for epoch in range(1, 4):

        start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = normalize_batch(images)

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item() * labels.size(0)
            )

            preds = outputs.argmax(dim=1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_metrics = evaluate_validation(model)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train Acc={train_acc:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val F1={val_metrics['macro_f1']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Time={elapsed:.1f}s"
        )

        DEBUG_RESULTS.append({
            "strategy": strategy,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"]
        })

debug_df = pd.DataFrame(DEBUG_RESULTS)

print("\n" + "=" * 90)
print("DEBUG RESULT")
print("=" * 90)

display(debug_df)

# %% [Cell 61]
# ============================================================
# SANITY CONTROL | ORIGINAL-ONLY | 595 IMAGES | SEED 42
# ============================================================

import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DEBUG_SEED = 42


class OriginalOnlyDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        return image, label


seed_everything(DEBUG_SEED)

train_dataset = OriginalOnlyDataset(
    original_train_tensor,
    original_train_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

model = build_model().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

print("=" * 90)
print("ORIGINAL-ONLY SANITY CONTROL | SEED 42")
print("=" * 90)

for epoch in range(1, 4):

    start = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = normalize_batch(images)

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    val_metrics = evaluate_validation(model)

    print(
        f"Epoch {epoch} | "
        f"Train Loss={train_loss:.4f} | "
        f"Train Acc={train_acc:.4f} | "
        f"Val Loss={val_metrics['loss']:.4f} | "
        f"Val F1={val_metrics['macro_f1']:.4f} | "
        f"Val Acc={val_metrics['accuracy']:.4f} | "
        f"Val BalAcc={val_metrics['balanced_accuracy']:.4f} | "
        f"Time={time.time()-start:.1f}s"
    )

# %% [Cell 62]
# ============================================================
# REBUILD ORIGINAL 595 TRAIN CACHE
# Source of truth = metadata.csv + actual Drive files
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
RAW_DIR = os.path.join(ROOT, "Raw")
AUG_DIR = os.path.join(ROOT, "Augmented")
METADATA_PATH = os.path.join(ROOT, "metadata.csv")

metadata_check = pd.read_csv(METADATA_PATH)

# ------------------------------------------------------------
# Expected rows:
# 595 ORIGINAL training leaves
# ------------------------------------------------------------

original_train_meta = metadata_check[
    metadata_check["data_split"].astype(str).str.strip().str.lower().eq("train")
    &
    metadata_check["preprocessing_technique"].astype(str).str.strip().eq("Original")
].copy()

print("Original train rows:", len(original_train_meta))
assert len(original_train_meta) == 595

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

CLASS_MAP = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

original_train_labels_rebuilt = np.array(
    [
        CLASS_MAP[x]
        for x in original_train_meta["class_label"]
    ],
    dtype=np.int64
)

print("\nRebuilt class counts:")
print(pd.Series(original_train_labels_rebuilt).value_counts().sort_index())

assert dict(
    pd.Series(original_train_labels_rebuilt).value_counts().sort_index()
) == {
    0: 280,
    1: 140,
    2: 175
}

# ------------------------------------------------------------
# Build exact filename -> actual file lookup
# ------------------------------------------------------------

drive_files = []

for base_dir in [RAW_DIR, AUG_DIR]:

    for root, _, files in os.walk(base_dir):

        for fname in files:

            if fname.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                drive_files.append(
                    os.path.join(root, fname)
                )

print("\nTotal Drive image files:", len(drive_files))
assert len(drive_files) == 13600

drive_lookup = {
    os.path.basename(path): path
    for path in drive_files
}

# ------------------------------------------------------------
# Verify every original training file exists
# ------------------------------------------------------------

paths = []

for _, row in original_train_meta.iterrows():

    fname = os.path.basename(
        str(row["filename"])
    )

    path = drive_lookup.get(fname)

    assert path is not None, f"Missing file: {fname}"

    paths.append(path)

print("All 595 original training files found ✅")

# ------------------------------------------------------------
# Load images in EXACT metadata order
# ------------------------------------------------------------

rebuilt_images = np.empty(
    (595, 224, 224, 3),
    dtype=np.uint8
)

def load_one(item):

    idx, path = item

    with Image.open(path) as img:

        img = img.convert("RGB")

        img = img.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        arr = np.asarray(
            img,
            dtype=np.uint8
        )

    return idx, arr


with ThreadPoolExecutor(max_workers=8) as executor:

    for idx, arr in executor.map(
        load_one,
        enumerate(paths)
    ):

        rebuilt_images[idx] = arr


# ------------------------------------------------------------
# Convert to CHW tensor
# ------------------------------------------------------------

original_train_tensor_rebuilt = torch.from_numpy(
    rebuilt_images
).permute(
    0, 3, 1, 2
).contiguous()

original_train_labels_rebuilt = torch.from_numpy(
    original_train_labels_rebuilt
)

print("\nRebuilt tensor:")
print("shape:", original_train_tensor_rebuilt.shape)
print("dtype:", original_train_tensor_rebuilt.dtype)
print(
    "range:",
    int(original_train_tensor_rebuilt.min()),
    "to",
    int(original_train_tensor_rebuilt.max())
)

# ------------------------------------------------------------
# IMPORTANT: replace suspicious cache
# ------------------------------------------------------------

original_train_tensor = original_train_tensor_rebuilt
original_train_labels = original_train_labels_rebuilt

print("\n✅ ORIGINAL TRAIN CACHE REBUILT")
print("Images :", original_train_tensor.shape)
print("Labels :", original_train_labels.shape)

# %% [Cell 63]
# ============================================================
# ORIGINAL-ONLY SANITY CONTROL — REBUILT 595 TRAIN IMAGES
# SEED 42
# NO AUGMENTATION
# ============================================================

import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DEBUG_SEED = 42


# ============================================================
# 1. VERIFY REBUILT CACHE
# ============================================================

assert "original_train_tensor" in globals(), \
    "original_train_tensor not found."

assert "original_train_labels" in globals(), \
    "original_train_labels not found."

assert original_train_tensor.shape == (595, 3, 224, 224), \
    f"Unexpected tensor shape: {original_train_tensor.shape}"

assert len(original_train_labels) == 595, \
    f"Unexpected label count: {len(original_train_labels)}"

assert original_train_tensor.dtype == torch.uint8, \
    f"Unexpected dtype: {original_train_tensor.dtype}"

assert int(original_train_tensor.min()) >= 0
assert int(original_train_tensor.max()) <= 255

print("=" * 90)
print("REBUILT CACHE VERIFIED")
print("=" * 90)
print("Images :", original_train_tensor.shape)
print("Labels :", original_train_labels.shape)
print("Dtype  :", original_train_tensor.dtype)
print(
    "Range  :",
    int(original_train_tensor.min()),
    "to",
    int(original_train_tensor.max())
)

print("\nClass counts:")
print(
    pd.Series(
        original_train_labels.cpu().numpy()
    ).value_counts().sort_index()
)


# ============================================================
# 2. ORIGINAL-ONLY DATASET
# ============================================================

class OriginalOnlyDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = int(self.labels[idx])

        return image, label


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

seed_everything(DEBUG_SEED)


# ============================================================
# 4. DATA LOADER
# ============================================================

train_dataset = OriginalOnlyDataset(
    original_train_tensor,
    original_train_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print("\nTraining samples:", len(train_dataset))


# ============================================================
# 5. MODEL
# ============================================================

model = build_model().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# ============================================================
# 6. TRAIN — ONLY 3 EPOCHS FOR SANITY CHECK
# ============================================================

print("\n" + "=" * 90)
print("ORIGINAL-ONLY SANITY CONTROL | REBUILT CACHE | SEED 42")
print("=" * 90)

results = []

for epoch in range(1, 4):

    start = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # ----------------------------------------------------
        # uint8 [0,255] -> float [0,1] -> ImageNet normalize
        # ----------------------------------------------------

        images = normalize_batch(images)

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
            loss.item() * labels.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # --------------------------------------------------------
    # Original validation only
    # --------------------------------------------------------

    val_metrics = evaluate_validation(model)

    elapsed = time.time() - start

    results.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_metrics["loss"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"]
    })

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss={train_loss:.4f} | "
        f"Train Acc={train_acc:.4f} | "
        f"Val Loss={val_metrics['loss']:.4f} | "
        f"Val F1={val_metrics['macro_f1']:.4f} | "
        f"Val Acc={val_metrics['accuracy']:.4f} | "
        f"Val BalAcc={val_metrics['balanced_accuracy']:.4f} | "
        f"Time={elapsed:.1f}s"
    )


# ============================================================
# 7. FINAL TABLE
# ============================================================

sanity_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("SANITY CHECK RESULT")
print("=" * 90)

display(sanity_df)

# %% [Cell 64]
# ============================================================
# MASTER STANDARD AUGMENTATION EXPERIMENT — FIXED VERSION
#
# FIXES:
# 1) Rebuilds ORIGINAL 595 training images directly from Drive
#    using metadata.csv as the source of truth.
# 2) Does NOT reuse old broken RandAugment/AugMix checkpoints.
# 3) Uses original 595 training leaves for RandAugment/AugMix.
# 4) Keeps Full_Data_9520 checkpoints from the previous experiment.
# 5) Validation = ORIGINAL validation set only.
# 6) Test = inference only; never used for checkpoint selection.
# 7) Evaluates:
#       Full_Data_9520
#       RandAugment
#       AugMix
#    × 3 seeds × 16 transformations.
#
# IMPORTANT:
# - Run this as ONE CELL after the existing model/helper definitions
#   (build_model, seed_everything, TRANSFORMATIONS, etc.) are available.
# ============================================================


import os
import time
import numpy as np
import pandas as pd
import torch

from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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

RAW_DIR = os.path.join(
    ROOT,
    "Raw"
)

AUG_DIR = os.path.join(
    ROOT,
    "Augmented"
)

METADATA_PATH = os.path.join(
    ROOT,
    "metadata.csv"
)

# ------------------------------------------------------------
# OLD directory:
# Contains previous Full_Data_9520 checkpoints.
# ------------------------------------------------------------

OLD_STANDARD_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_standard_pytorch"
)

# ------------------------------------------------------------
# NEW directory:
# ALL corrected RandAugment / AugMix outputs go here.
# ------------------------------------------------------------

FIXED_STANDARD_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_standard_pytorch_FIXED"
)

os.makedirs(
    FIXED_STANDARD_DIR,
    exist_ok=True
)

TEST_RAM_CACHE = (
    "/content/brinjal_test_2048_ram"
)

os.makedirs(
    TEST_RAM_CACHE,
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

assert device.type == "cuda", (
    "CUDA GPU is required for this experiment."
)

print("=" * 100)
print("MASTER STANDARD AUGMENTATION EXPERIMENT — FIXED")
print("=" * 100)
print("Device:", device)


# ============================================================
# 3. VERIFY REQUIRED EXISTING OBJECTS
# ============================================================

required_objects = [
    "build_model",
    "seed_everything",
    "TRANSFORMATIONS",
    "val_tensor",
    "val_label_tensor",
    "train_tensor"
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

assert not missing_objects, (
    "Missing required objects: "
    + ", ".join(missing_objects)
)

assert train_tensor.shape == (
    9520, 3, 224, 224
), (
    f"Unexpected train_tensor shape: "
    f"{train_tensor.shape}"
)

assert val_tensor.shape == (
    2032, 3, 224, 224
), (
    f"Unexpected val_tensor shape: "
    f"{val_tensor.shape}"
)


# ============================================================
# 4. NORMALIZATION
# ============================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(
    1, 3, 1, 1
)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(
    1, 3, 1, 1
)


def normalize_batch(images):

    images = images.to(
        device,
        non_blocking=True
    )

    if images.dtype == torch.uint8:

        images = images.float().div_(
            255.0
        )

    return (
        images - MEAN
    ) / STD


# ============================================================
# 5. VALIDATION DATASET
# ============================================================

class RAMValidationDataset(Dataset):

    def __init__(
        self,
        images,
        labels
    ):

        self.images = images
        self.labels = labels

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(self, idx):

        return (
            self.images[idx],
            self.labels[idx]
        )


val_labels = val_label_tensor

val_dataset = RAMValidationDataset(
    val_tensor,
    val_labels
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# 6. VALIDATION FUNCTION
# ============================================================

criterion_global = (
    torch.nn.CrossEntropyLoss()
)


def evaluate_validation(model):

    model.eval()

    total_loss = 0.0
    total = 0

    predictions = []
    labels_all = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = normalize_batch(
                images
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion_global(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total += labels.size(0)

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels_all.extend(
                labels.cpu().numpy()
            )

    return {

        "loss":
            total_loss / total,

        "accuracy":
            accuracy_score(
                labels_all,
                predictions
            ),

        "macro_f1":
            f1_score(
                labels_all,
                predictions,
                average="macro",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels_all,
                predictions
            )
    }


# ============================================================
# 7. REBUILD ORIGINAL 595 TRAINING CACHE
#
# IMPORTANT:
# We do NOT trust the previous original_train_tensor.
# The actual files on Drive + metadata.csv are the source
# of truth.
# ============================================================

print("\n")
print("=" * 100)
print("REBUILDING ORIGINAL 595 TRAINING CACHE")
print("=" * 100)

metadata = pd.read_csv(
    METADATA_PATH
)

filename_col = "filename"
class_col = "class_label"
transformation_col = "preprocessing_technique"
split_col = "data_split"


# ------------------------------------------------------------
# Select EXACTLY:
# data_split = train
# preprocessing_technique = Original
# ------------------------------------------------------------

original_train_meta = metadata[
    metadata[split_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("train")
    &
    metadata[transformation_col]
    .astype(str)
    .str.strip()
    .eq("Original")
].copy()

print(
    "Original training rows:",
    len(original_train_meta)
)

assert len(original_train_meta) == 595


# ============================================================
# 8. CLASS MAPPING
# ============================================================

CLASS_MAP = {
    "Healthy_Leaves": 0,
    "Little_Leaf": 1,
    "Phomopsis_Blight": 2
}

labels_np = np.asarray(
    [
        CLASS_MAP[label]
        for label in original_train_meta[class_col]
    ],
    dtype=np.int64
)

print("\nRebuilt class counts:")

class_counts = (
    pd.Series(labels_np)
    .value_counts()
    .sort_index()
)

print(class_counts)

assert dict(
    class_counts
) == {
    0: 280,
    1: 140,
    2: 175
}


# ============================================================
# 9. BUILD DRIVE FILE LOOKUP
# ============================================================

print("\nScanning Drive image files...")

all_drive_files = []

for base_dir in [
    RAW_DIR,
    AUG_DIR
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

                all_drive_files.append(
                    os.path.join(
                        root,
                        fname
                    )
                )

print(
    "Total Drive image files:",
    len(all_drive_files)
)

assert len(all_drive_files) == 13600

drive_lookup = {

    os.path.basename(path):
        path

    for path in all_drive_files
}


# ============================================================
# 10. RESOLVE EXACT 595 FILE PATHS
# ============================================================

original_train_paths = []

for _, row in original_train_meta.iterrows():

    fname = os.path.basename(
        str(row[filename_col])
    )

    path = drive_lookup.get(
        fname
    )

    assert path is not None, (
        f"Missing original training file: {fname}"
    )

    original_train_paths.append(
        path
    )

assert len(
    original_train_paths
) == 595

print(
    "All 595 original training files found ✅"
)


# ============================================================
# 11. LOAD 595 ORIGINAL IMAGES
# ============================================================

original_train_images_np = np.empty(
    (
        595,
        224,
        224,
        3
    ),
    dtype=np.uint8
)


def load_original_train_one(item):

    idx, path = item

    with Image.open(path) as img:

        img = img.convert(
            "RGB"
        )

        img = img.resize(
            (
                224,
                224
            ),
            Image.Resampling.BILINEAR
        )

        arr = np.asarray(
            img,
            dtype=np.uint8
        )

    return idx, arr


start_rebuild = time.time()

with ThreadPoolExecutor(
    max_workers=8
) as executor:

    for idx, arr in executor.map(
        load_original_train_one,
        enumerate(
            original_train_paths
        )
    ):

        original_train_images_np[idx] = arr


print(
    "Original cache built in "
    f"{time.time() - start_rebuild:.1f}s"
)


# ============================================================
# 12. CONVERT TO CHW TENSOR
# ============================================================

original_train_tensor = torch.from_numpy(
    original_train_images_np
).permute(
    0,
    3,
    1,
    2
).contiguous()

original_train_labels = torch.from_numpy(
    labels_np
)

print("\nRebuilt tensor:")
print(
    "Images:",
    original_train_tensor.shape
)

print(
    "Labels:",
    original_train_labels.shape
)

print(
    "Dtype:",
    original_train_tensor.dtype
)

print(
    "Range:",
    int(original_train_tensor.min()),
    "to",
    int(original_train_tensor.max())
)

assert original_train_tensor.shape == (
    595, 3, 224, 224
)

assert original_train_labels.shape == (
    595,
)

assert original_train_tensor.dtype == torch.uint8

assert int(
    original_train_tensor.min()
) >= 0

assert int(
    original_train_tensor.max()
) <= 255

print(
    "\n✅ ORIGINAL TRAIN CACHE REBUILT SUCCESSFULLY"
)


# ============================================================
# 13. AUGMENTATIONS
#
# IMPORTANT:
# These are the ORIGINAL configurations used in the
# previous standard experiment.
#
# We are not changing their strength.
# We are only fixing the source cache.
# ============================================================

rand_augment = transforms.RandAugment(
    num_ops=2,
    magnitude=9
)

augmix = transforms.AugMix(
    severity=3,
    mixture_width=3,
    chain_depth=-1,
    alpha=1.0
)

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


# ============================================================
# 14. STANDARD TRAINING DATASET
# ============================================================

class StandardTrainDataset(Dataset):

    def __init__(
        self,
        images,
        labels,
        strategy
    ):

        self.images = images
        self.labels = labels
        self.strategy = strategy

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(self, idx):

        image = self.images[idx]

        label = int(
            self.labels[idx]
        )

        # ----------------------------------------------------
        # uint8 tensor -> PIL
        # ----------------------------------------------------

        image = to_pil(
            image
        )

        # ----------------------------------------------------
        # ONLINE AUGMENTATION
        # ----------------------------------------------------

        if self.strategy == "RandAugment":

            image = rand_augment(
                image
            )

        elif self.strategy == "AugMix":

            image = augmix(
                image
            )

        else:

            raise ValueError(
                f"Unknown strategy: "
                f"{self.strategy}"
            )

        # ----------------------------------------------------
        # PIL -> [0,1] tensor
        # ----------------------------------------------------

        image = to_tensor(
            image
        )

        return (
            image,
            label
        )


# ============================================================
# 15. TRAIN RAND AUGMENT + AUGMIX
#
# NEW FIXED CHECKPOINT DIRECTORY
# No bad checkpoint from the previous run can be reused.
# ============================================================

SEEDS = [
    42,
    1337,
    2026
]

TRAINING_STRATEGIES = [
    "RandAugment",
    "AugMix"
]

training_results = []


for strategy in TRAINING_STRATEGIES:

    for SEED in SEEDS:

        print("\n")
        print("=" * 100)
        print(
            f"TRAINING | {strategy} | SEED {SEED}"
        )
        print("=" * 100)

        # ----------------------------------------------------
        # NEW checkpoint names
        # ----------------------------------------------------

        checkpoint_path = os.path.join(
            FIXED_STANDARD_DIR,
            f"{strategy.lower()}_FIXED_seed{SEED}.pt"
        )

        history_path = os.path.join(
            FIXED_STANDARD_DIR,
            f"{strategy.lower()}_FIXED_seed{SEED}_history.csv"
        )

        validation_path = os.path.join(
            FIXED_STANDARD_DIR,
            f"{strategy.lower()}_FIXED_seed{SEED}_validation.csv"
        )

        # ----------------------------------------------------
        # Intentionally overwrite/retrain.
        # The old broken checkpoint is NEVER reused.
        # ----------------------------------------------------

        seed_everything(
            SEED
        )

        # ----------------------------------------------------
        # ORIGINAL 595 TRAINING LEAVES ONLY
        # ----------------------------------------------------

        train_dataset = StandardTrainDataset(
            original_train_tensor,
            original_train_labels,
            strategy
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        print(
            "Training samples:",
            len(train_dataset)
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = build_model().to(
            device
        )

        criterion = (
            torch.nn.CrossEntropyLoss()
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        )

        best_val_loss = float(
            "inf"
        )

        best_epoch = 0
        patience_counter = 0

        history = []

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        for epoch in range(
            1,
            21
        ):

            epoch_start = time.time()

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:

                images = normalize_batch(
                    images
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

                running_loss += (
                    loss.item()
                    * labels.size(0)
                )

                preds = outputs.argmax(
                    dim=1
                )

                correct += (
                    preds == labels
                ).sum().item()

                total += labels.size(0)

            train_loss = (
                running_loss / total
            )

            train_acc = (
                correct / total
            )

            # ------------------------------------------------
            # ORIGINAL VALIDATION
            # ------------------------------------------------

            val_metrics = evaluate_validation(
                model
            )

            val_loss = (
                val_metrics["loss"]
            )

            scheduler.step(
                val_loss
            )

            lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            elapsed = (
                time.time()
                - epoch_start
            )

            # ------------------------------------------------
            # CHECKPOINT BY ORIGINAL VALIDATION LOSS
            # ------------------------------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_epoch = epoch

                patience_counter = 0

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "strategy":
                            strategy,

                        "seed":
                            SEED,

                        "best_epoch":
                            best_epoch,

                        "best_val_loss":
                            best_val_loss
                    },
                    checkpoint_path
                )

                best_flag = " ✅ BEST"

            else:

                patience_counter += 1

                best_flag = ""

            history.append(
                {
                    "epoch":
                        epoch,

                    "train_loss":
                        train_loss,

                    "train_accuracy":
                        train_acc,

                    "val_loss":
                        val_loss,

                    "val_macro_f1":
                        val_metrics["macro_f1"],

                    "val_accuracy":
                        val_metrics["accuracy"],

                    "val_balanced_accuracy":
                        val_metrics[
                            "balanced_accuracy"
                        ],

                    "lr":
                        lr,

                    "epoch_time_sec":
                        elapsed
                }
            )

            print(
                f"Epoch {epoch:02d}/20 | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Orig Val Loss {val_loss:.4f} | "
                f"Orig Val F1 "
                f"{val_metrics['macro_f1']:.4f} | "
                f"Orig Val Acc "
                f"{val_metrics['accuracy']:.4f} | "
                f"LR {lr:.6f} | "
                f"{elapsed:.1f}s"
                f"{best_flag}"
            )

            if patience_counter >= 4:

                print(
                    "\n⏹ Early stopping"
                )

                break

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False
        )

        # ----------------------------------------------------
        # RELOAD BEST CHECKPOINT
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        final_val = evaluate_validation(
            model
        )

        result = {

            "strategy":
                strategy,

            "seed":
                SEED,

            "training_samples":
                595,

            "best_epoch":
                best_epoch,

            "best_val_loss":
                best_val_loss,

            "original_val_accuracy":
                final_val["accuracy"],

            "original_val_macro_f1":
                final_val["macro_f1"],

            "original_val_balanced_accuracy":
                final_val[
                    "balanced_accuracy"
                ],

            "checkpoint":
                checkpoint_path
        }

        training_results.append(
            result
        )

        pd.DataFrame(
            [result]
        ).to_csv(
            validation_path,
            index=False
        )

        print(
            "\n✅ COMPLETE"
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Val Macro-F1:",
            f"{final_val['macro_f1']:.4f}"
        )

        print(
            "Val Accuracy:",
            f"{final_val['accuracy']:.4f}"
        )

        print(
            "Val Balanced Accuracy:",
            f"{final_val['balanced_accuracy']:.4f}"
        )

        print(
            "⚠️ LOCKED TEST NOT USED."
        )


# ============================================================
# 16. SAVE TRAINING SUMMARY
# ============================================================

training_summary = pd.DataFrame(
    training_results
)

training_summary = (
    training_summary
    .sort_values(
        [
            "strategy",
            "seed"
        ]
    )
    .reset_index(
        drop=True
    )
)

training_summary_path = os.path.join(
    FIXED_STANDARD_DIR,
    "randaugment_augmix_FIXED_all_seeds_summary.csv"
)

training_summary.to_csv(
    training_summary_path,
    index=False
)

print("\n")
print("=" * 100)
print("✅ RAND AUGMENT + AUGMIX FIXED TRAINING COMPLETE")
print("=" * 100)

display(
    training_summary
)


# ============================================================
# 17. BUILD LOCKED TEST METADATA
# ============================================================

print("\n")
print("=" * 100)
print("BUILDING LOCKED TEST RAM CACHE")
print("=" * 100)

test_meta = metadata[
    metadata[split_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("test")
].copy()

assert len(
    test_meta
) == 2048


# ============================================================
# 18. TEST FILE LOOKUP
# ============================================================

test_paths = []
test_labels_np = []
test_names = []
test_transforms = []
test_original_ids = []

for _, row in test_meta.iterrows():

    fname = os.path.basename(
        str(row[filename_col])
    )

    path = drive_lookup.get(
        fname
    )

    assert path is not None, (
        f"Missing test file: {fname}"
    )

    test_paths.append(
        path
    )

    test_labels_np.append(
        CLASS_MAP[
            row[class_col]
        ]
    )

    test_names.append(
        fname
    )

    test_transforms.append(
        row[transformation_col]
    )

    test_original_ids.append(
        row["original_image_id"]
    )


test_labels_np = np.asarray(
    test_labels_np,
    dtype=np.int64
)


# ============================================================
# 19. BUILD TEST IMAGE CACHE
# ============================================================

TEST_CACHE_FILE = os.path.join(
    TEST_RAM_CACHE,
    "test_2048_uint8.npy"
)

if os.path.exists(
    TEST_CACHE_FILE
):

    print(
        "✅ Existing test RAM cache found."
    )

    test_images = np.load(
        TEST_CACHE_FILE
    )

else:

    print(
        "Building test image cache..."
    )

    test_images = np.empty(
        (
            2048,
            224,
            224,
            3
        ),
        dtype=np.uint8
    )

    start_test_cache = time.time()

    def load_test_one(item):

        idx, path = item

        with Image.open(path) as img:

            img = img.convert(
                "RGB"
            )

            img = img.resize(
                (
                    224,
                    224
                ),
                Image.Resampling.BILINEAR
            )

            arr = np.asarray(
                img,
                dtype=np.uint8
            )

        return idx, arr


    jobs = list(
        enumerate(
            test_paths
        )
    )

    completed = 0

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        for idx, arr in executor.map(
            load_test_one,
            jobs
        ):

            test_images[idx] = arr

            completed += 1

            if (
                completed % 500 == 0
                or
                completed == 2048
            ):

                print(
                    f"{completed}/2048"
                )

    np.save(
        TEST_CACHE_FILE,
        test_images
    )

    print(
        "✅ Test cache built in "
        f"{time.time() - start_test_cache:.1f}s"
    )


assert test_images.shape == (
    2048,
    224,
    224,
    3
)


# ============================================================
# 20. TEST TENSORS
# ============================================================

test_tensor = torch.from_numpy(
    test_images
).permute(
    0,
    3,
    1,
    2
).contiguous()

test_labels_tensor = torch.from_numpy(
    test_labels_np
)

test_names = np.asarray(
    test_names
)

test_transforms = np.asarray(
    test_transforms
)

test_original_ids = np.asarray(
    test_original_ids
)


# ============================================================
# 21. TEST DATASET
# ============================================================

class TensorTestDataset(Dataset):

    def __init__(
        self,
        images,
        labels,
        names,
        transformations,
        original_ids
    ):

        self.images = images
        self.labels = labels
        self.names = names
        self.transformations = transformations
        self.original_ids = original_ids

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(self, idx):

        return (
            self.images[idx],
            self.labels[idx],
            idx
        )


test_dataset = TensorTestDataset(
    test_tensor,
    test_labels_tensor,
    test_names,
    test_transforms,
    test_original_ids
)


# ============================================================
# 22. TEST EVALUATION
# ============================================================

def evaluate_test_condition(
    model,
    indices
):

    subset = torch.utils.data.Subset(
        test_dataset,
        indices.tolist()
    )

    loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_indices = []

    with torch.no_grad():

        for images, labels, idxs in loader:

            images = normalize_batch(
                images
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            preds = outputs.argmax(
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_indices.extend(
                idxs.numpy()
            )

    all_preds = np.asarray(
        all_preds
    )

    all_labels = np.asarray(
        all_labels
    )

    return {

        "predictions":
            all_preds,

        "labels":
            all_labels,

        "accuracy":
            accuracy_score(
                all_labels,
                all_preds
            ),

        "macro_f1":
            f1_score(
                all_labels,
                all_preds,
                average="macro",
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                all_labels,
                all_preds
            ),

        "indices":
            np.asarray(
                all_indices
            )
    }


# ============================================================
# 23. TEST STRATEGIES
#
# Full_Data_9520:
# Use existing VALID checkpoint from old directory.
#
# RandAugment/AugMix:
# Use ONLY new FIXED checkpoints.
# ============================================================

STANDARD_TEST_STRATEGIES = [
    "Full_Data_9520",
    "RandAugment",
    "AugMix"
]

test_results = []
test_prediction_records = []


for strategy in STANDARD_TEST_STRATEGIES:

    for SEED in SEEDS:

        # ----------------------------------------------------
        # CHECKPOINT PATH
        # ----------------------------------------------------

        if strategy == "Full_Data_9520":

            checkpoint_path = os.path.join(
                OLD_STANDARD_DIR,
                f"full_data_9520_seed{SEED}_fast.pt"
            )

        elif strategy == "RandAugment":

            checkpoint_path = os.path.join(
                FIXED_STANDARD_DIR,
                f"randaugment_FIXED_seed{SEED}.pt"
            )

        else:

            checkpoint_path = os.path.join(
                FIXED_STANDARD_DIR,
                f"augmix_FIXED_seed{SEED}.pt"
            )

        assert os.path.exists(
            checkpoint_path
        ), (
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

        print("\n")
        print("=" * 100)
        print(
            f"LOCKED TEST | "
            f"{strategy} | SEED {SEED}"
        )
        print("=" * 100)

        model = build_model().to(
            device
        )

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

        for transformation in TRANSFORMATIONS:

            indices = np.flatnonzero(
                test_transforms
                ==
                transformation
            )

            assert len(
                indices
            ) == 128

            evaluation = evaluate_test_condition(
                model,
                indices
            )

            test_results.append(
                {

                    "strategy":
                        strategy,

                    "seed":
                        SEED,

                    "transformation":
                        transformation,

                    "n":
                        128,

                    "accuracy":
                        evaluation[
                            "accuracy"
                        ],

                    "macro_f1":
                        evaluation[
                            "macro_f1"
                        ],

                    "balanced_accuracy":
                        evaluation[
                            "balanced_accuracy"
                        ]
                }
            )

            preds = evaluation[
                "predictions"
            ]

            labels = evaluation[
                "labels"
            ]

            original_indices = (
                evaluation["indices"]
            )

            for j in range(
                len(preds)
            ):

                local_idx = int(
                    original_indices[j]
                )

                true_label = int(
                    labels[j]
                )

                pred_label = int(
                    preds[j]
                )

                test_prediction_records.append(
                    {

                        "strategy":
                            strategy,

                        "seed":
                            SEED,

                        "transformation":
                            transformation,

                        "original_image_id":
                            test_original_ids[
                                local_idx
                            ],

                        "filename":
                            test_names[
                                local_idx
                            ],

                        "y_true":
                            true_label,

                        "y_pred":
                            pred_label,

                        "correct":
                            int(
                                true_label
                                ==
                                pred_label
                            )
                    }
                )

            print(
                f"{transformation:15s} | "
                f"Acc="
                f"{evaluation['accuracy']:.4f} | "
                f"Macro-F1="
                f"{evaluation['macro_f1']:.4f} | "
                f"BalAcc="
                f"{evaluation['balanced_accuracy']:.4f}"
            )


# ============================================================
# 24. SAVE TEST RESULTS
# ============================================================

standard_test_df = pd.DataFrame(
    test_results
)

standard_predictions_df = pd.DataFrame(
    test_prediction_records
)

standard_metrics_path = os.path.join(
    FIXED_STANDARD_DIR,
    "standard_FIXED_locked_test_metrics.csv"
)

standard_predictions_path = os.path.join(
    FIXED_STANDARD_DIR,
    "standard_FIXED_locked_test_predictions.csv"
)

standard_test_df.to_csv(
    standard_metrics_path,
    index=False
)

standard_predictions_df.to_csv(
    standard_predictions_path,
    index=False
)


# ============================================================
# 25. AGGREGATE ACROSS 3 SEEDS
# ============================================================

standard_aggregate = (
    standard_test_df
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

standard_aggregate_path = os.path.join(
    FIXED_STANDARD_DIR,
    "standard_FIXED_locked_test_aggregate.csv"
)

standard_aggregate.to_csv(
    standard_aggregate_path,
    index=False
)


# ============================================================
# 26. TARGET CONDITIONS
# ============================================================

target_conditions = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness",
    "Gaussian_Blur"
]

target_standard_summary = (
    standard_aggregate[
        standard_aggregate[
            "transformation"
        ].isin(
            target_conditions
        )
    ]
    .copy()
    .sort_values(
        [
            "strategy",
            "transformation"
        ]
    )
)

target_standard_path = os.path.join(
    FIXED_STANDARD_DIR,
    "standard_FIXED_locked_test_target_summary.csv"
)

target_standard_summary.to_csv(
    target_standard_path,
    index=False
)


# ============================================================
# 27. MACRO-F1 WIDE
# ============================================================

standard_f1_wide = (
    standard_aggregate
    .pivot(
        index="strategy",
        columns="transformation",
        values="macro_f1_mean"
    )
)

standard_f1_wide_path = os.path.join(
    FIXED_STANDARD_DIR,
    "standard_FIXED_locked_test_macro_f1_wide.csv"
)

standard_f1_wide.to_csv(
    standard_f1_wide_path
)


# ============================================================
# 28. FINAL OUTPUT
# ============================================================

print("\n\n")
print("=" * 100)
print("✅ FIXED MASTER STANDARD EXPERIMENT COMPLETE")
print("=" * 100)

print("\nTraining results:")
display(
    training_summary
)

print("\nLocked-test target summary:")
display(
    target_standard_summary
)

print("\nLocked-test Macro-F1 across all transformations:")
display(
    standard_f1_wide
)

print("\nSaved files:")

print(
    training_summary_path
)

print(
    standard_metrics_path
)

print(
    standard_predictions_path
)

print(
    standard_aggregate_path
)

print(
    target_standard_path
)

print(
    standard_f1_wide_path
)

print("\n⚠️ TEST DATA WAS USED FOR INFERENCE ONLY.")
print("⚠️ NO TEST DATA WAS USED FOR CHECKPOINT SELECTION.")
print("⚠️ OLD BROKEN RAND AUGMENT / AUGMIX CHECKPOINTS WERE NOT REUSED.")
print("=" * 100)

# %% [Cell 65]
# ============================================================
# STANDARD AUGMENTATION STATISTICAL COMPARISON
#
# Baseline vs RandAugment vs AugMix
# Exact McNemar + Benjamini-Hochberg FDR
#
# Conditions:
#   Original
#   Grayscale
#   Rotate
#   Brightness
#
# 3 seeds × 2 comparisons × 4 conditions = 24 tests
#
# Test set is used only for paired inference.
# No model selection is performed here.
# ============================================================

import os
import glob
import numpy as np
import pandas as pd

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. PATHS
# ============================================================

ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

BASELINE_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_final_pytorch",
    "final_locked_test"
)

STANDARD_FIXED_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_standard_pytorch_FIXED"
)

OUTPUT_DIR = os.path.join(
    STANDARD_FIXED_DIR,
    "statistical_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. FIND BASELINE PREDICTION FILE
# ============================================================

baseline_candidates = glob.glob(
    os.path.join(
        BASELINE_DIR,
        "*prediction*.csv"
    )
)

if len(baseline_candidates) == 0:

    raise FileNotFoundError(
        "No baseline prediction CSV found in:\n"
        f"{BASELINE_DIR}"
    )

print("=" * 100)
print("BASELINE PREDICTION FILE")
print("=" * 100)

for path in baseline_candidates:
    print(path)

# Prefer the most likely final predictions file
preferred = [
    path
    for path in baseline_candidates
    if "final_locked_test_predictions" in os.path.basename(path).lower()
]

if len(preferred) > 0:
    baseline_path = preferred[0]
else:
    baseline_path = baseline_candidates[0]

print("\nSelected:")
print(baseline_path)


# ============================================================
# 3. STANDARD FIXED PREDICTIONS
# ============================================================

standard_path = os.path.join(
    STANDARD_FIXED_DIR,
    "standard_FIXED_locked_test_predictions.csv"
)

assert os.path.exists(
    standard_path
), (
    "Standard FIXED prediction file not found:\n"
    f"{standard_path}"
)

print("\nStandard prediction file:")
print(standard_path)


# ============================================================
# 4. LOAD
# ============================================================

baseline_df = pd.read_csv(
    baseline_path
)

standard_df = pd.read_csv(
    standard_path
)

print("\n")
print("=" * 100)
print("LOADED PREDICTIONS")
print("=" * 100)

print(
    "Baseline shape:",
    baseline_df.shape
)

print(
    "Standard shape:",
    standard_df.shape
)

print("\nBaseline columns:")
print(
    list(baseline_df.columns)
)

print("\nStandard columns:")
print(
    list(standard_df.columns)
)


# ============================================================
# 5. STANDARDIZE COLUMN NAMES
# ============================================================

# -------------------------------
# Baseline
# -------------------------------

baseline_rename = {}

if "pred_label" in baseline_df.columns:
    baseline_rename["pred_label"] = "y_pred"

if "true_label" in baseline_df.columns:
    baseline_rename["true_label"] = "y_true"

baseline_df = baseline_df.rename(
    columns=baseline_rename
)


# -------------------------------
# Standard
# -------------------------------

standard_rename = {}

if "pred_label" in standard_df.columns:
    standard_rename["pred_label"] = "y_pred"

if "true_label" in standard_df.columns:
    standard_rename["true_label"] = "y_true"

standard_df = standard_df.rename(
    columns=standard_rename
)


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

required_baseline = [
    "strategy",
    "seed",
    "transformation",
    "original_image_id",
    "y_true",
    "y_pred"
]

required_standard = [
    "strategy",
    "seed",
    "transformation",
    "original_image_id",
    "y_true",
    "y_pred"
]

for col in required_baseline:

    assert col in baseline_df.columns, (
        f"Baseline missing column: {col}"
    )

for col in required_standard:

    assert col in standard_df.columns, (
        f"Standard missing column: {col}"
    )


print("\n✅ Required columns verified.")


# ============================================================
# 7. FILTER STANDARD STRATEGIES
# ============================================================

standard_df = standard_df[
    standard_df["strategy"].isin(
        [
            "RandAugment",
            "AugMix"
        ]
    )
].copy()


# ============================================================
# 8. BASELINE FILTER
# ============================================================

baseline_df = baseline_df[
    baseline_df["strategy"]
    .astype(str)
    .str.strip()
    .eq("Baseline")
].copy()


# ============================================================
# 9. TARGET CONDITIONS
# ============================================================

TARGET_CONDITIONS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]

SEEDS = [
    42,
    1337,
    2026
]

STRATEGIES = [
    "RandAugment",
    "AugMix"
]


# ============================================================
# 10. VERIFY GROUP SIZES
# ============================================================

print("\n")
print("=" * 100)
print("GROUP COUNTS")
print("=" * 100)

for seed in SEEDS:

    for transformation in TARGET_CONDITIONS:

        base_group = baseline_df[
            (baseline_df["seed"] == seed)
            &
            (
                baseline_df["transformation"]
                == transformation
            )
        ]

        print(
            f"Baseline | seed={seed:4d} | "
            f"{transformation:12s} | n={len(base_group)}"
        )

        assert len(base_group) == 128


for strategy in STRATEGIES:

    for seed in SEEDS:

        for transformation in TARGET_CONDITIONS:

            group = standard_df[
                (standard_df["strategy"] == strategy)
                &
                (standard_df["seed"] == seed)
                &
                (
                    standard_df["transformation"]
                    == transformation
                )
            ]

            print(
                f"{strategy:12s} | seed={seed:4d} | "
                f"{transformation:12s} | n={len(group)}"
            )

            assert len(group) == 128


# ============================================================
# 11. EXACT McNEMAR FUNCTION
# ============================================================

def exact_mcnemar(
    baseline_predictions,
    augmented_predictions
):

    baseline_correct = (
        baseline_predictions["y_true"].to_numpy()
        ==
        baseline_predictions["y_pred"].to_numpy()
    )

    augmented_correct = (
        augmented_predictions["y_true"].to_numpy()
        ==
        augmented_predictions["y_pred"].to_numpy()
    )

    # Baseline correct, augmented wrong
    a = int(
        np.sum(
            baseline_correct
            &
            (~augmented_correct)
        )
    )

    # Baseline wrong, augmented correct
    b = int(
        np.sum(
            (~baseline_correct)
            &
            augmented_correct
        )
    )

    discordant = a + b

    if discordant == 0:

        p_value = 1.0

    else:

        p_value = binomtest(
            k=min(a, b),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return {
        "baseline_correct_augmented_wrong": a,
        "baseline_wrong_augmented_correct": b,
        "discordant_pairs": discordant,
        "p_value": p_value
    }


# ============================================================
# 12. RUN ALL PAIRED TESTS
# ============================================================

results = []

print("\n")
print("=" * 100)
print("RUNNING EXACT McNEMAR TESTS")
print("=" * 100)

for strategy in STRATEGIES:

    for seed in SEEDS:

        for transformation in TARGET_CONDITIONS:

            base_group = baseline_df[
                (baseline_df["seed"] == seed)
                &
                (
                    baseline_df["transformation"]
                    == transformation
                )
            ].copy()

            aug_group = standard_df[
                (standard_df["strategy"] == strategy)
                &
                (standard_df["seed"] == seed)
                &
                (
                    standard_df["transformation"]
                    == transformation
                )
            ].copy()

            # ------------------------------------------------
            # Align on original image ID
            # ------------------------------------------------

            merged = base_group.merge(
                aug_group,
                on="original_image_id",
                suffixes=(
                    "_baseline",
                    "_augmented"
                ),
                how="inner"
            )

            assert len(merged) == 128, (
                f"Pairing failure: "
                f"{strategy} / {seed} / "
                f"{transformation} -> "
                f"{len(merged)} pairs"
            )

            # ------------------------------------------------
            # Calculate paired correctness
            # ------------------------------------------------

            baseline_correct = (
                merged["y_true_baseline"].to_numpy()
                ==
                merged["y_pred_baseline"].to_numpy()
            )

            augmented_correct = (
                merged["y_true_augmented"].to_numpy()
                ==
                merged["y_pred_augmented"].to_numpy()
            )

            a = int(
                np.sum(
                    baseline_correct
                    &
                    (~augmented_correct)
                )
            )

            b = int(
                np.sum(
                    (~baseline_correct)
                    &
                    augmented_correct
                )
            )

            discordant = a + b

            if discordant == 0:

                p_value = 1.0

            else:

                p_value = binomtest(
                    k=min(a, b),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided"
                ).pvalue

            results.append({

                "comparison":
                    f"Baseline_vs_{strategy}",

                "strategy":
                    strategy,

                "seed":
                    seed,

                "transformation":
                    transformation,

                "baseline_correct_augmented_wrong":
                    a,

                "baseline_wrong_augmented_correct":
                    b,

                "discordant_pairs":
                    discordant,

                "p_value":
                    p_value
            })

            print(
                f"{strategy:12s} | "
                f"seed={seed:4d} | "
                f"{transformation:12s} | "
                f"{a:3d} vs {b:3d} | "
                f"p={p_value:.4e}"
            )


# ============================================================
# 13. BENJAMINI-HOCHBERG FDR
# ============================================================

stats_df = pd.DataFrame(
    results
)

reject, q_values, _, _ = multipletests(
    stats_df["p_value"].to_numpy(),
    alpha=0.05,
    method="fdr_bh"
)

stats_df["fdr_q"] = q_values
stats_df["significant_after_fdr"] = reject


# ============================================================
# 14. PREFERRED MODEL
# ============================================================

stats_df["preferred_model"] = np.where(
    stats_df[
        "baseline_correct_augmented_wrong"
    ]
    <
    stats_df[
        "baseline_wrong_augmented_correct"
    ],
    stats_df["strategy"],
    np.where(
        stats_df[
            "baseline_correct_augmented_wrong"
        ]
        >
        stats_df[
            "baseline_wrong_augmented_correct"
        ],
        "Baseline",
        "Tie"
    )
)


# ============================================================
# 15. SUMMARY
# ============================================================

summary_rows = []

for strategy in STRATEGIES:

    subset = stats_df[
        stats_df["strategy"]
        == strategy
    ]

    for transformation in TARGET_CONDITIONS:

        s = subset[
            subset["transformation"]
            == transformation
        ]

        summary_rows.append({

            "strategy":
                strategy,

            "transformation":
                transformation,

            "n_seeds":
                len(s),

            "significant_seeds":
                int(
                    s[
                        "significant_after_fdr"
                    ].sum()
                ),

            "mean_fdr_q":
                s["fdr_q"].mean(),

            "min_fdr_q":
                s["fdr_q"].min(),

            "max_fdr_q":
                s["fdr_q"].max(),

            "mean_discordant_pairs":
                s["discordant_pairs"].mean()
        })

summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 16. SAVE
# ============================================================

stats_path = os.path.join(
    OUTPUT_DIR,
    "standard_augmentation_mcnemar_all_tests.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "standard_augmentation_mcnemar_summary.csv"
)

stats_df.to_csv(
    stats_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("EXACT McNEMAR RESULTS")
print("=" * 100)

display(
    stats_df[
        [
            "comparison",
            "seed",
            "transformation",
            "baseline_correct_augmented_wrong",
            "baseline_wrong_augmented_correct",
            "discordant_pairs",
            "p_value",
            "fdr_q",
            "significant_after_fdr",
            "preferred_model"
        ]
    ]
    .sort_values(
        [
            "comparison",
            "transformation",
            "seed"
        ]
    )
)

print("\n")
print("=" * 100)
print("SUMMARY ACROSS SEEDS")
print("=" * 100)

display(
    summary_df
    .sort_values(
        [
            "strategy",
            "transformation"
        ]
    )
)

print("\n")
print("=" * 100)
print("SAVED")
print("=" * 100)

print(stats_path)
print(summary_path)

print("\n✅ EXACT McNEMAR + BH-FDR COMPLETE")
print("✅ TEST USED ONLY FOR PAIRED INFERENCE")
print("✅ NO CHECKPOINT SELECTION INVOLVED")

# %% [Cell 66]
# ============================================================
# DIRECT RAND AUGMENT vs AUGMIX COMPARISON
#
# Exact McNemar + Benjamini-Hochberg FDR
#
# 3 seeds × 4 test conditions = 12 paired tests
#
# Conditions:
#   Original
#   Grayscale
#   Rotate
#   Brightness
#
# TEST SET = INFERENCE / PAIRED STATISTICAL ANALYSIS ONLY
# NO TRAINING
# NO CHECKPOINT SELECTION
# ============================================================

import os
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
    "augmentation_ablation_standard_pytorch_FIXED",
    "standard_FIXED_locked_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "augmentation_ablation_standard_pytorch_FIXED",
    "statistical_comparison"
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

pred_df = pd.read_csv(
    PREDICTION_PATH
)

print("=" * 100)
print("RAND AUGMENT vs AUGMIX — DIRECT PAIRED COMPARISON")
print("=" * 100)

print(
    "Prediction file:",
    PREDICTION_PATH
)

print(
    "Shape:",
    pred_df.shape
)

print(
    "Strategies:",
    sorted(
        pred_df["strategy"].unique()
    )
)


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "strategy",
    "seed",
    "transformation",
    "original_image_id",
    "y_true",
    "y_pred"
]

for col in required_columns:

    assert col in pred_df.columns, (
        f"Missing column: {col}"
    )


# ============================================================
# 4. FILTER REQUIRED STRATEGIES
# ============================================================

pred_df = pred_df[
    pred_df["strategy"].isin(
        [
            "RandAugment",
            "AugMix"
        ]
    )
].copy()


SEEDS = [
    42,
    1337,
    2026
]

TRANSFORMATIONS = [
    "Original",
    "Grayscale",
    "Rotate",
    "Brightness"
]


# ============================================================
# 5. VERIFY EACH GROUP HAS 128 LEAVES
# ============================================================

print("\n")
print("=" * 100)
print("GROUP VERIFICATION")
print("=" * 100)

for strategy in [
    "RandAugment",
    "AugMix"
]:

    for seed in SEEDS:

        for transformation in TRANSFORMATIONS:

            group = pred_df[
                (pred_df["strategy"] == strategy)
                &
                (pred_df["seed"] == seed)
                &
                (
                    pred_df["transformation"]
                    == transformation
                )
            ]

            print(
                f"{strategy:12s} | "
                f"seed={seed:4d} | "
                f"{transformation:12s} | "
                f"n={len(group)}"
            )

            assert len(group) == 128, (
                f"Expected 128 rows, got "
                f"{len(group)} for "
                f"{strategy}, {seed}, "
                f"{transformation}"
            )


# ============================================================
# 6. EXACT McNEMAR
# ============================================================

results = []


for seed in SEEDS:

    for transformation in TRANSFORMATIONS:

        # ----------------------------------------------------
        # RandAugment
        # ----------------------------------------------------

        rand_df = pred_df[
            (pred_df["strategy"] == "RandAugment")
            &
            (pred_df["seed"] == seed)
            &
            (
                pred_df["transformation"]
                == transformation
            )
        ].copy()

        # ----------------------------------------------------
        # AugMix
        # ----------------------------------------------------

        augmix_df = pred_df[
            (pred_df["strategy"] == "AugMix")
            &
            (pred_df["seed"] == seed)
            &
            (
                pred_df["transformation"]
                == transformation
            )
        ].copy()

        # ----------------------------------------------------
        # Pair the SAME original leaves
        # ----------------------------------------------------

        merged = rand_df.merge(
            augmix_df,
            on="original_image_id",
            suffixes=(
                "_randaugment",
                "_augmix"
            ),
            how="inner"
        )

        assert len(merged) == 128, (
            f"Pairing failure for "
            f"seed={seed}, "
            f"transformation={transformation}: "
            f"{len(merged)} pairs"
        )

        # ----------------------------------------------------
        # Correctness of each method
        # ----------------------------------------------------

        rand_correct = (
            merged["y_true_randaugment"].to_numpy()
            ==
            merged["y_pred_randaugment"].to_numpy()
        )

        augmix_correct = (
            merged["y_true_augmix"].to_numpy()
            ==
            merged["y_pred_augmix"].to_numpy()
        )

        # ----------------------------------------------------
        # McNemar discordant cells
        #
        # a = RandAugment correct / AugMix wrong
        # b = RandAugment wrong / AugMix correct
        # ----------------------------------------------------

        a = int(
            np.sum(
                rand_correct
                &
                (~augmix_correct)
            )
        )

        b = int(
            np.sum(
                (~rand_correct)
                &
                augmix_correct
            )
        )

        discordant = a + b

        # ----------------------------------------------------
        # Exact two-sided binomial test
        # ----------------------------------------------------

        if discordant == 0:

            p_value = 1.0

        else:

            p_value = binomtest(
                k=min(a, b),
                n=discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue

        # ----------------------------------------------------
        # Preferred model from discordant pairs
        # ----------------------------------------------------

        if a > b:

            preferred = "RandAugment"

        elif b > a:

            preferred = "AugMix"

        else:

            preferred = "Tie"

        results.append({

            "comparison":
                "RandAugment_vs_AugMix",

            "seed":
                seed,

            "transformation":
                transformation,

            "randaugment_correct_augmix_wrong":
                a,

            "randaugment_wrong_augmix_correct":
                b,

            "discordant_pairs":
                discordant,

            "p_value":
                p_value,

            "preferred_model":
                preferred
        })


# ============================================================
# 7. DATAFRAME
# ============================================================

stats_df = pd.DataFrame(
    results
)


# ============================================================
# 8. BENJAMINI-HOCHBERG FDR
#
# 12 total tests
# ============================================================

reject, q_values, _, _ = multipletests(
    stats_df["p_value"].to_numpy(),
    alpha=0.05,
    method="fdr_bh"
)

stats_df["fdr_q"] = q_values

stats_df["significant_after_fdr"] = reject


# ============================================================
# 9. SORT
# ============================================================

stats_df = (
    stats_df
    .sort_values(
        [
            "transformation",
            "seed"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 10. SUMMARY ACROSS SEEDS
# ============================================================

summary_rows = []

for transformation in TRANSFORMATIONS:

    subset = stats_df[
        stats_df["transformation"]
        == transformation
    ].copy()

    summary_rows.append({

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

        "randaugment_wins_mean":
            subset[
                "randaugment_correct_augmix_wrong"
            ].mean(),

        "augmix_wins_mean":
            subset[
                "randaugment_wrong_augmix_correct"
            ].mean(),

        "mean_discordant_pairs":
            subset[
                "discordant_pairs"
            ].mean(),

        "overall_preferred":
            (
                "RandAugment"
                if
                subset[
                    "randaugment_correct_augmix_wrong"
                ].sum()
                >
                subset[
                    "randaugment_wrong_augmix_correct"
                ].sum()
                else
                "AugMix"
                if
                subset[
                    "randaugment_correct_augmix_wrong"
                ].sum()
                <
                subset[
                    "randaugment_wrong_augmix_correct"
                ].sum()
                else
                "Tie"
            )
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

all_tests_path = os.path.join(
    OUTPUT_DIR,
    "randaugment_vs_augmix_mcnemar_all_tests.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "randaugment_vs_augmix_mcnemar_summary.csv"
)

stats_df.to_csv(
    all_tests_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 12. DISPLAY FULL RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("EXACT McNEMAR — RAND AUGMENT vs AUGMIX")
print("=" * 100)

display(
    stats_df[
        [
            "comparison",
            "seed",
            "transformation",
            "randaugment_correct_augmix_wrong",
            "randaugment_wrong_augmix_correct",
            "discordant_pairs",
            "p_value",
            "fdr_q",
            "significant_after_fdr",
            "preferred_model"
        ]
    ]
)


print("\n")
print("=" * 100)
print("SUMMARY ACROSS SEEDS")
print("=" * 100)

display(
    summary_df
)


print("\n")
print("=" * 100)
print("SAVED FILES")
print("=" * 100)

print(
    all_tests_path
)

print(
    summary_path
)

print("\n✅ DIRECT RAND AUGMENT vs AUGMIX TEST COMPLETE")
print("✅ EXACT McNEMAR")
print("✅ BENJAMINI-HOCHBERG FDR")
print("✅ SAME 128 LEAVES PAIRED")
print("✅ NO TRAINING")
print("✅ NO CHECKPOINT SELECTION")

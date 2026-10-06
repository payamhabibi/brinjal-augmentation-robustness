
# %% [Cell 0]

!nvidia-smi

# %% [Cell 1]
from google.colab import drive
drive.mount('/content/drive')

# %% [Cell 2]
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )
else:
    print("❌ GPU در دسترس نیست")

# %% [Cell 3]
from google.colab import drive
drive.mount('/content/drive')

# %% [Cell 4]
import os

base = "/content/drive/MyDrive"

for root, dirs, files in os.walk(base):
    if "metadata.csv" in files:
        print("FOUND:", os.path.join(root, "metadata.csv"))

# %% [Cell 5]
import os

dataset_root = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

print("Exists:", os.path.exists(dataset_root))

if os.path.exists(dataset_root):
    print("\nFolders/files:")
    for item in os.listdir(dataset_root):
        print(item)

# %% [Cell 6]
import os
import pandas as pd
from pathlib import Path

# ============================================================
# 1. Paths
# ============================================================
DATASET_ROOT = Path("/content/drive/MyDrive/Brinjal_Final_Preprocessed")
METADATA_PATH = DATASET_ROOT / "metadata.csv"

RAW_DIR = DATASET_ROOT / "Raw"
AUG_DIR = DATASET_ROOT / "Augmented"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# 2. Load metadata
# ============================================================
df = pd.read_csv(METADATA_PATH)

print("=" * 70)
print("DATASET AUDIT")
print("=" * 70)

print(f"\nMetadata shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
display(df.head())

# ============================================================
# 3. Missing values
# ============================================================
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()
display(missing)

# ============================================================
# 4. Class distribution
# ============================================================
print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

display(df["class_label"].value_counts())

# ============================================================
# 5. Split distribution
# ============================================================
print("\n" + "=" * 70)
print("SPLIT DISTRIBUTION")
print("=" * 70)

display(df["data_split"].value_counts())

# ============================================================
# 6. Preprocessing / transformation distribution
# ============================================================
print("\n" + "=" * 70)
print("TRANSFORMATION DISTRIBUTION")
print("=" * 70)

display(df["preprocessing_technique"].value_counts())

# ============================================================
# 7. Count image files
# ============================================================
raw_files = [
    p for p in RAW_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]

aug_files = [
    p for p in AUG_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]

all_image_files = raw_files + aug_files

print("\n" + "=" * 70)
print("IMAGE FILE COUNTS")
print("=" * 70)

print(f"Raw images       : {len(raw_files):,}")
print(f"Augmented images : {len(aug_files):,}")
print(f"Total images     : {len(all_image_files):,}")

# ============================================================
# 8. Folder-level counts
# ============================================================
print("\nRaw folders:")
for p in sorted(RAW_DIR.iterdir()):
    if p.is_dir():
        count = sum(
            1 for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        print(f"  {p.name}: {count:,}")

print("\nAugmented folders:")
for p in sorted(AUG_DIR.iterdir()):
    if p.is_dir():
        count = sum(
            1 for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        print(f"  {p.name}: {count:,}")

# ============================================================
# 9. Metadata ↔ image file consistency
# ============================================================
def normalize_name(x):
    return os.path.basename(str(x)).strip()

df["filename_clean"] = df["filename"].map(normalize_name)

available_filenames = {p.name for p in all_image_files}

missing_files = df.loc[
    ~df["filename_clean"].isin(available_filenames),
    ["filename", "class_label", "data_split", "preprocessing_technique"]
]

print("\n" + "=" * 70)
print("METADATA ↔ IMAGE CONSISTENCY")
print("=" * 70)

print(f"Metadata rows                 : {len(df):,}")
print(f"Unique filenames in metadata  : {df['filename_clean'].nunique():,}")
print(f"Unique image files found      : {len(available_filenames):,}")
print(f"Missing image files           : {len(missing_files):,}")

if len(missing_files) == 0:
    print("✅ Every metadata filename has a corresponding image file.")
else:
    print("❌ Some metadata files are missing.")
    display(missing_files.head(20))

# ============================================================
# 10. Exactly 16 variants per original leaf?
# ============================================================
variants_per_leaf = df.groupby("original_image_id").size()

bad_leaves = variants_per_leaf[variants_per_leaf != 16]

print("\n" + "=" * 70)
print("VARIANTS PER ORIGINAL LEAF")
print("=" * 70)

print(f"Unique original leaves: {variants_per_leaf.size:,}")

print("\nNumber of variants per leaf:")
display(variants_per_leaf.value_counts().sort_index())

print(f"Leaves not having exactly 16 variants: {len(bad_leaves):,}")

if len(bad_leaves) == 0:
    print("✅ Every original leaf has exactly 16 versions.")
else:
    print("❌ Some leaves do not have exactly 16 versions.")
    display(bad_leaves.head(20))

# ============================================================
# 11. Leakage check: one leaf across multiple splits?
# ============================================================
leaf_split_counts = df.groupby("original_image_id")["data_split"].nunique()

bad_split_leaves = leaf_split_counts[leaf_split_counts > 1]

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(f"Leaves appearing in multiple splits: {len(bad_split_leaves):,}")

if len(bad_split_leaves) == 0:
    print("✅ No original leaf appears across train/validation/test.")
else:
    print("❌ Potential data leakage detected!")
    display(bad_split_leaves.head(20))

# ============================================================
# 12. Class × split
# ============================================================
print("\n" + "=" * 70)
print("CLASS × SPLIT")
print("=" * 70)

class_split = pd.crosstab(
    df["class_label"],
    df["data_split"]
)

display(class_split)

# ============================================================
# 13. Final summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"✅ Metadata rows      : {len(df):,}")
print(f"✅ Unique leaves      : {df['original_image_id'].nunique():,}")
print(f"✅ Total image files  : {len(all_image_files):,}")
print(f"✅ Missing files      : {len(missing_files):,}")
print(f"✅ Bad leaf variants  : {len(bad_leaves):,}")
print(f"✅ Split leakage      : {len(bad_split_leaves):,}")

# %% [Cell 7]
import os
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
DATASET_ROOT = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

# فقط 300 تصویر تصادفی برای integrity check
SAMPLE_SIZE = 300

# ============================================================
# 1. FAST IMAGE INTEGRITY CHECK
# ============================================================
sample_df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
)

bad_images = []
sizes = {}
modes = {}

for _, row in sample_df.iterrows():
    path = os.path.join(DATASET_ROOT, row["filename"])

    try:
        with Image.open(path) as img:
            # فقط یک بار باز می‌کنیم
            img.load()

            size = img.size
            mode = img.mode

            sizes[size] = sizes.get(size, 0) + 1
            modes[mode] = modes.get(mode, 0) + 1

    except Exception as e:
        bad_images.append((path, str(e)))

print("=" * 65)
print("FAST IMAGE INTEGRITY CHECK")
print("=" * 65)

print(f"Images checked: {len(sample_df)}")
print(f"Bad/unreadable images: {len(bad_images)}")
print(f"Image sizes: {sizes}")
print(f"Image modes: {modes}")

if len(bad_images) == 0:
    print("✅ All sampled images are readable.")
else:
    print("❌ Some sampled images could not be read.")
    for item in bad_images[:10]:
        print(item)

# ============================================================
# 2. SELECT 3 RANDOM LEAVES
# ============================================================
random.seed(42)

leaf_ids = random.sample(
    list(df["original_image_id"].unique()),
    3
)

# ============================================================
# 3. VISUALIZE 16 TRANSFORMATIONS FOR EACH LEAF
# ============================================================
for leaf_id in leaf_ids:

    sample = (
        df[df["original_image_id"] == leaf_id]
        .copy()
        .sort_values("preprocessing_technique")
    )

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):

        path = os.path.join(DATASET_ROOT, row["filename"])

        with Image.open(path) as img:
            img = img.convert("RGB")
            ax.imshow(img)

        ax.set_title(
            row["preprocessing_technique"],
            fontsize=9
        )
        ax.axis("off")

    plt.suptitle(
        f"Leaf: {leaf_id}",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()

# %% [Cell 8]
from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("/content/drive/MyDrive/Brinjal_Final_Preprocessed")
METADATA_PATH = DATASET_ROOT / "metadata.csv"

df = pd.read_csv(METADATA_PATH)

# فقط Original برای baseline training
train_original = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Original")
].copy()

val_original = df[
    (df["data_split"] == "val") &
    (df["preprocessing_technique"] == "Original")
].copy()

test_all = df[
    df["data_split"] == "test"
].copy()

print("Train original :", len(train_original))
print("Val original   :", len(val_original))
print("Test all       :", len(test_all))

print("\nExpected:")
print("Train = 595")
print("Val   = 127")
print("Test  = 2048")

# %% [Cell 9]
from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("/content/drive/MyDrive/Brinjal_Final_Preprocessed")
METADATA_PATH = DATASET_ROOT / "metadata.csv"

df = pd.read_csv(METADATA_PATH)

train_original = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Original")
].copy()

val_original = df[
    (df["data_split"] == "val") &
    (df["preprocessing_technique"] == "Original")
].copy()

test_all = df[
    df["data_split"] == "test"
].copy()

print("Train original :", len(train_original))
print("Val original   :", len(val_original))
print("Test all       :", len(test_all))

print("\nExpected:")
print("Train = 595")
print("Val   = 127")
print("Test  = 2048")

# %% [Cell 10]
# ============================================================
# BRINJAL - BASELINE DATALOADER + PILOT SETUP
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torchvision

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
DATASET_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

METADATA_PATH = DATASET_ROOT / "metadata.csv"

# ------------------------------------------------------------
# 3. Device
# ------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print("PyTorch      :", torch.__version__)
print("Torchvision  :", torchvision.__version__)
print("Device       :", device)

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
    print(
        "VRAM         :",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

# ------------------------------------------------------------
# 4. Load metadata
# ------------------------------------------------------------
df = pd.read_csv(METADATA_PATH)

# ------------------------------------------------------------
# 5. Class mapping
# ------------------------------------------------------------
CLASS_NAMES = sorted(df["class_label"].unique())
CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}
IDX_TO_CLASS = {
    idx: name
    for name, idx in CLASS_TO_IDX.items()
}

print("\nClasses:")
print(CLASS_TO_IDX)

# ------------------------------------------------------------
# 6. Select ONLY original images
# ------------------------------------------------------------
train_df = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Original")
].copy()

val_df = df[
    (df["data_split"] == "val") &
    (df["preprocessing_technique"] == "Original")
].copy()

test_df = df[
    (df["data_split"] == "test") &
    (df["preprocessing_technique"] == "Original")
].copy()

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

# ------------------------------------------------------------
# 7. Image transformations
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# ------------------------------------------------------------
# 8. Custom Dataset
# ------------------------------------------------------------
class BrinjalDataset(Dataset):

    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.root_dir / row["filename"]

        image = Image.open(image_path).convert("RGB")

        label = CLASS_TO_IDX[row["class_label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------------------------------------------------
# 9. Create datasets
# ------------------------------------------------------------
train_dataset = BrinjalDataset(
    train_df,
    DATASET_ROOT,
    transform=base_transform
)

val_dataset = BrinjalDataset(
    val_df,
    DATASET_ROOT,
    transform=base_transform
)

test_dataset = BrinjalDataset(
    test_df,
    DATASET_ROOT,
    transform=base_transform
)

# ------------------------------------------------------------
# 10. DataLoaders
# ------------------------------------------------------------
BATCH_SIZE = 32

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\nDataLoaders created successfully.")

# ------------------------------------------------------------
# 11. Test one batch
# ------------------------------------------------------------
images, labels = next(iter(train_loader))

print("\nFirst batch:")
print("Images shape :", images.shape)
print("Labels shape :", labels.shape)
print("Labels       :", labels[:10].tolist())

# ------------------------------------------------------------
# 12. Move batch to GPU
# ------------------------------------------------------------
images = images.to(device)
labels = labels.to(device)

print("\nGPU test:")
print("Image device :", images.device)
print("Label device :", labels.device)

# ------------------------------------------------------------
# 13. Create MobileNetV3-Small
# ------------------------------------------------------------
weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(weights=weights)

# Replace classifier
in_features = model.classifier[-1].in_features

model.classifier[-1] = torch.nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

model = model.to(device)

print("\nModel:")
print(model.classifier)

print("\n✅ PILOT SETUP COMPLETE")

# %% [Cell 11]
# ============================================================
# BRINJAL - MOBILEV3 SMALL PILOT TRAINING
# 2 epochs فقط برای تست کامل pipeline
# ============================================================

import time
import copy
import torch
import torch.nn as nn
from pathlib import Path

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
NUM_EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = (
    DATASET_ROOT / "pilot_mobilenetv3_small.pth"
)

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

scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

best_val_loss = float("inf")
best_state = None

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
print("=" * 70)
print("MOBILENETV3-SMALL PILOT TRAINING")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    # ========================================================
    # TRAIN
    # ========================================================
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ========================================================
    # VALIDATION
    # ========================================================
    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    scheduler.step(val_loss)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    epoch_time = time.time() - start_time

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "class_to_idx": CLASS_TO_IDX,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc
            },
            CHECKPOINT_PATH
        )

        checkpoint_status = "✅ BEST CHECKPOINT SAVED"

    else:
        checkpoint_status = ""

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] "
        f"| Time: {epoch_time:.1f}s"
    )

    print(
        f"Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc:.4f}"
    )

    print(
        f"Val Loss  : {val_loss:.4f} "
        f"| Val Acc  : {val_acc:.4f}"
    )

    print(
        f"LR        : "
        f"{optimizer.param_groups[0]['lr']:.6f}"
    )

    if checkpoint_status:
        print(checkpoint_status)

# ------------------------------------------------------------
# Restore best model
# ------------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

print("\n" + "=" * 70)
print("PILOT TRAINING COMPLETE")
print("=" * 70)

print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

print("\nHistory:")
for i in range(NUM_EPOCHS):
    print(
        f"Epoch {i+1}: "
        f"Train Acc={history['train_acc'][i]:.4f}, "
        f"Val Acc={history['val_acc'][i]:.4f}"
    )

# %% [Cell 12]
# ============================================================
# CACHE ONLY ORIGINAL IMAGES LOCALLY
# 850 images - NOT the full 1GB dataset
# ============================================================

import shutil
import time
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

LOCAL_ROOT = Path("/content/brinjal_original")

# Clean old cache if it exists
if LOCAL_ROOT.exists():
    shutil.rmtree(LOCAL_ROOT)

LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

# Only Original images
original_df = df[
    df["preprocessing_technique"] == "Original"
].copy()

print("Original images to cache:", len(original_df))

start = time.time()

for _, row in original_df.iterrows():

    src = DRIVE_ROOT / row["filename"]

    # Preserve the same relative structure
    relative_path = Path(row["filename"])
    dst = LOCAL_ROOT / relative_path

    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)

elapsed = time.time() - start

print("\n✅ Original images cached locally.")
print(f"Images copied: {len(original_df)}")
print(f"Time: {elapsed:.1f} seconds")

# Verify
local_count = sum(
    1 for p in LOCAL_ROOT.rglob("*")
    if p.is_file()
)

print(f"Local files found: {local_count}")

# %% [Cell 13]
# ============================================================
# LOCAL CACHE DATA LOADER + SPEED TEST
# ============================================================

import time
from torch.utils.data import DataLoader

LOCAL_ROOT = Path("/content/brinjal_original")

# Recreate datasets using local storage
train_dataset_local = BrinjalDataset(
    train_df,
    LOCAL_ROOT,
    transform=base_transform
)

val_dataset_local = BrinjalDataset(
    val_df,
    LOCAL_ROOT,
    transform=base_transform
)

test_dataset_local = BrinjalDataset(
    test_df,
    LOCAL_ROOT,
    transform=base_transform
)

train_loader_local = DataLoader(
    train_dataset_local,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

val_loader_local = DataLoader(
    val_dataset_local,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

test_loader_local = DataLoader(
    test_dataset_local,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# ------------------------------------------------------------
# Speed test
# ------------------------------------------------------------
print("=" * 65)
print("LOCAL DATA LOADING SPEED TEST")
print("=" * 65)

start = time.time()

num_batches = 0

for images, labels in train_loader_local:
    num_batches += 1

elapsed = time.time() - start

print(f"Batches      : {num_batches}")
print(f"Time         : {elapsed:.2f} sec")
print(f"Time/batch   : {elapsed / num_batches:.3f} sec")

print("\n✅ Local DataLoader test complete.")

# %% [Cell 14]
# ============================================================
# BRINJAL BASELINE - MobileNetV3-Small
# Real training | Seed 42
# ============================================================

import time
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------
MAX_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "baseline_mobilenetv3_small_seed42.pth"
)

# ------------------------------------------------------------
# 3. Fresh model
# ------------------------------------------------------------
weights = MobileNet_V3_Small_Weights.DEFAULT

baseline_model = mobilenet_v3_small(
    weights=weights
)

in_features = baseline_model.classifier[-1].in_features

baseline_model.classifier[-1] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

baseline_model = baseline_model.to(device)

# ------------------------------------------------------------
# 4. Loss / optimizer / scheduler
# ------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    baseline_model.parameters(),
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

# ------------------------------------------------------------
# 5. Tracking
# ------------------------------------------------------------
history = []

best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0

# ------------------------------------------------------------
# 6. Training loop
# ------------------------------------------------------------
print("=" * 70)
print("MOBILENETV3-SMALL BASELINE")
print("=" * 70)

for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start = time.time()

    # ========================================================
    # TRAIN
    # ========================================================
    baseline_model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader_local:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            outputs = baseline_model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    # ========================================================
    # VALIDATION
    # ========================================================
    baseline_model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader_local:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                outputs = baseline_model(images)
                loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            val_correct += (
                preds == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    # ========================================================
    # Save history
    # ========================================================
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": current_lr
    })

    # ========================================================
    # Check improvement
    # ========================================================
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        best_state = copy.deepcopy(
            baseline_model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_state,
                "class_to_idx": CLASS_TO_IDX,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seed": SEED
            },
            CHECKPOINT_PATH
        )

        checkpoint_msg = "✅ BEST"

    else:

        epochs_without_improvement += 1
        checkpoint_msg = ""

    # ========================================================
    # Print
    # ========================================================
    elapsed = time.time() - epoch_start

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"{elapsed:.1f}s | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.4f} | "
        f"LR {current_lr:.6f} "
        f"{checkpoint_msg}"
    )

    # ========================================================
    # Early stopping
    # ========================================================
    if epochs_without_improvement >= PATIENCE:

        print(
            f"\n⏹ Early stopping at epoch {epoch}"
        )

        break

# ------------------------------------------------------------
# 7. Restore best model
# ------------------------------------------------------------
if best_state is not None:
    baseline_model.load_state_dict(best_state)

# ------------------------------------------------------------
# 8. Save training history
# ------------------------------------------------------------
history_df = pd.DataFrame(history)

HISTORY_PATH = (
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "baseline_mobilenetv3_small_seed42_history.csv"
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

# ------------------------------------------------------------
# 9. Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print(f"Best epoch      : {best_epoch}")
print(f"Best val loss   : {best_val_loss:.4f}")

best_row = history_df.loc[
    history_df["val_loss"].idxmin()
]

print(
    f"Best val acc    : "
    f"{best_row['val_acc']:.4f}"
)

print(f"\nCheckpoint saved:")
print(CHECKPOINT_PATH)

print("\nHistory saved:")
print(HISTORY_PATH)

# %% [Cell 15]
# ============================================================
# BRINJAL - BASELINE TEST EVALUATION
# MobileNetV3-Small | Seed 42
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. Load best checkpoint
# ------------------------------------------------------------
CHECKPOINT_PATH = (
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed/"
    "baseline_mobilenetv3_small_seed42.pth"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

baseline_model.load_state_dict(
    checkpoint["model_state_dict"]
)

baseline_model = baseline_model.to(device)
baseline_model.eval()

# ------------------------------------------------------------
# 2. Predictions
# ------------------------------------------------------------
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for images, labels in test_loader_local:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            outputs = baseline_model(images)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ------------------------------------------------------------
# 3. Metrics
# ------------------------------------------------------------
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

print("=" * 70)
print("MOBILENETV3-SMALL TEST RESULTS")
print("=" * 70)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro-F1          : {macro_f1:.4f}")
print(f"Balanced Accuracy : {balanced_acc:.4f}")

# ------------------------------------------------------------
# 4. Classification report
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_preds,
    target_names=[
        IDX_TO_CLASS[i]
        for i in range(len(CLASS_NAMES))
    ],
    digits=4
)

print(report)

# ------------------------------------------------------------
# 5. Confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(
    all_labels,
    all_preds
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MobileNetV3-Small — Original Test Set")
plt.tight_layout()

cm_path = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed/"
    "baseline_mobilenetv3_small_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 6. Save predictions
# ------------------------------------------------------------
predictions_df = test_df.copy()

predictions_df["true_label"] = [
    IDX_TO_CLASS[x]
    for x in all_labels
]

predictions_df["predicted_label"] = [
    IDX_TO_CLASS[x]
    for x in all_preds
]

predictions_df["correct"] = (
    all_labels == all_preds
)

for i, class_name in IDX_TO_CLASS.items():
    predictions_df[f"prob_{class_name}"] = all_probs[:, i]

pred_path = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed/"
    "baseline_mobilenetv3_small_test_predictions.csv"
)

predictions_df.to_csv(
    pred_path,
    index=False
)

print("\nSaved:")
print(cm_path)
print(pred_path)

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)

# %% [Cell 16]
# ============================================================
# MOBILENETV3-SMALL - MULTI-SEED BASELINE
# Seeds: 1337, 2026
# Same protocol as Seed 42
# ============================================================

import os
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
SEEDS = [1337, 2026]

MAX_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
BATCH_SIZE = 32

OUTPUT_DIR = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

# ------------------------------------------------------------
# Reproducibility helper
# ------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_one_seed(seed):

    print("\n" + "=" * 80)
    print(f"MOBILENETV3-SMALL — SEED {seed}")
    print("=" * 80)

    seed_everything(seed)

    # --------------------------------------------------------
    # DataLoader with seed-specific shuffle
    # --------------------------------------------------------
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss / optimizer / scheduler
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------
    history = []

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    for epoch in range(1, MAX_EPOCHS + 1):

        start_time = time.time()

        # ====================================================
        # TRAIN
        # ====================================================
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            train_correct += (
                preds == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ====================================================
        # VALIDATION
        # ====================================================
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ====================================================
        # History
        # ====================================================
        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })

        # ====================================================
        # Best checkpoint
        # ====================================================
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            checkpoint_path = os.path.join(
                OUTPUT_DIR,
                f"baseline_mobilenetv3_small_seed{seed}.pth"
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_to_idx": CLASS_TO_IDX,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "seed": seed
                },
                checkpoint_path
            )

            status = "✅ BEST"

        else:

            epochs_without_improvement += 1
            status = ""

        # ====================================================
        # Print
        # ====================================================
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"{elapsed:.1f}s | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"LR {current_lr:.6f} {status}"
        )

        # ====================================================
        # Early stopping
        # ====================================================
        if epochs_without_improvement >= PATIENCE:

            print(
                f"\n⏹ Early stopping at epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best
    # --------------------------------------------------------
    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------
    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_mobilenetv3_small_seed{seed}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n" + "-" * 80)
    print(f"Seed {seed} complete")
    print(f"Best epoch    : {best_epoch}")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")
    print("-" * 80)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint": checkpoint_path
    }


# ------------------------------------------------------------
# Run both seeds
# ------------------------------------------------------------
results = []

for seed in SEEDS:
    result = train_one_seed(seed)
    results.append(result)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MULTI-SEED TRAINING SUMMARY")
print("=" * 80)

display(results_df)

summary_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv3_multiseed_training_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print(f"\nSaved summary: {summary_path}")

# %% [Cell 17]
# ============================================================
# MULTI-SEED TEST EVALUATION
# MobileNetV3-Small | Seeds 42, 1337, 2026
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
SEEDS = [42, 1337, 2026]

OUTPUT_DIR = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------
def evaluate_seed(seed):

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_mobilenetv3_small_seed{seed}.pth"
    )

    # Create fresh architecture
    model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for images, labels in test_loader_local:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                outputs = model(images)

            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Metrics
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

    macro_precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return {
        "seed": seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    }


# ------------------------------------------------------------
# Evaluate all seeds
# ------------------------------------------------------------
results = []

for seed in SEEDS:

    print(f"\nEvaluating seed {seed}...")

    result = evaluate_seed(seed)

    results.append(result)

    print(
        f"Accuracy: {result['accuracy']:.4f} | "
        f"Macro-F1: {result['macro_f1']:.4f} | "
        f"Balanced Acc: {result['balanced_accuracy']:.4f}"
    )


# ------------------------------------------------------------
# Results table
# ------------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("MULTI-SEED TEST RESULTS")
print("=" * 75)

display(
    results_df.style.format({
        "accuracy": "{:.4f}",
        "macro_f1": "{:.4f}",
        "balanced_accuracy": "{:.4f}",
        "macro_precision": "{:.4f}",
        "macro_recall": "{:.4f}"
    })
)

# ------------------------------------------------------------
# Mean ± SD
# ------------------------------------------------------------
metric_cols = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall"
]

summary = []

for metric in metric_cols:

    summary.append({
        "metric": metric,
        "mean": results_df[metric].mean(),
        "std": results_df[metric].std(ddof=1),
        "min": results_df[metric].min(),
        "max": results_df[metric].max()
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 75)
print("MEAN ± SD")
print("=" * 75)

display(
    summary_df.style.format({
        "mean": "{:.4f}",
        "std": "{:.4f}",
        "min": "{:.4f}",
        "max": "{:.4f}"
    })
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
results_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv3_multiseed_test_results.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv3_multiseed_test_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSaved:")
print(results_path)
print(summary_path)


# %% [Cell 18]
# ============================================================
# EFFICIENTNET-B0 - MULTI-SEED BASELINE
# Same protocol as MobileNetV3-Small
# ============================================================

import os
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
SEEDS = [42, 1337, 2026]

MAX_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
BATCH_SIZE = 32

OUTPUT_DIR = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_efficientnet_seed(seed):

    print("\n" + "=" * 80)
    print(f"EFFICIENTNET-B0 — SEED {seed}")
    print("=" * 80)

    seed_everything(seed)

    # --------------------------------------------------------
    # Seed-specific DataLoader
    # --------------------------------------------------------
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )

    # --------------------------------------------------------
    # Fresh EfficientNet-B0
    # --------------------------------------------------------
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss / optimizer / scheduler
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------
    history = []

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, MAX_EPOCHS + 1):

        start_time = time.time()

        # ====================================================
        # TRAIN
        # ====================================================
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            train_correct += (
                preds == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ====================================================
        # VALIDATION
        # ====================================================
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ====================================================
        # Save history
        # ====================================================
        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })

        # ====================================================
        # Best checkpoint
        # ====================================================
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            checkpoint_path = os.path.join(
                OUTPUT_DIR,
                f"baseline_efficientnet_b0_seed{seed}.pth"
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_to_idx": CLASS_TO_IDX,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "seed": seed
                },
                checkpoint_path
            )

            status = "✅ BEST"

        else:

            epochs_without_improvement += 1
            status = ""

        # ====================================================
        # Print
        # ====================================================
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"{elapsed:.1f}s | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"LR {current_lr:.6f} {status}"
        )

        # ====================================================
        # Early stopping
        # ====================================================
        if epochs_without_improvement >= PATIENCE:

            print(
                f"\n⏹ Early stopping at epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------
    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------
    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_efficientnet_b0_seed{seed}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n" + "-" * 80)
    print(f"Seed {seed} complete")
    print(f"Best epoch    : {best_epoch}")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")
    print("-" * 80)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint": checkpoint_path
    }


# ------------------------------------------------------------
# Run all seeds
# ------------------------------------------------------------
results = []

for seed in SEEDS:
    result = train_efficientnet_seed(seed)
    results.append(result)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("EFFICIENTNET-B0 MULTI-SEED TRAINING SUMMARY")
print("=" * 80)

display(results_df)

summary_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_multiseed_training_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print(f"\nSaved summary: {summary_path}")

# %% [Cell 19]
# ============================================================
# EFFICIENTNET-B0 - MULTI-SEED TEST EVALUATION
# Seeds: 42, 1337, 2026
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

SEEDS = [42, 1337, 2026]

OUTPUT_DIR = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)


def evaluate_efficientnet_seed(seed):

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_efficientnet_b0_seed{seed}.pth"
    )

    # Fresh architecture
    model = efficientnet_b0(weights=None)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for images, labels in test_loader_local:

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

            preds = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return {
        "seed": seed,
        "accuracy": accuracy_score(
            all_labels,
            all_preds
        ),
        "macro_f1": f1_score(
            all_labels,
            all_preds,
            average="macro"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            all_labels,
            all_preds
        ),
        "macro_precision": precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )
    }


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

results = []

for seed in SEEDS:

    print(f"\nEvaluating EfficientNet-B0 seed {seed}...")

    result = evaluate_efficientnet_seed(seed)

    results.append(result)

    print(
        f"Accuracy: {result['accuracy']:.4f} | "
        f"Macro-F1: {result['macro_f1']:.4f} | "
        f"Balanced Acc: {result['balanced_accuracy']:.4f}"
    )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("EFFICIENTNET-B0 MULTI-SEED TEST RESULTS")
print("=" * 75)

display(
    results_df.style.format({
        "accuracy": "{:.4f}",
        "macro_f1": "{:.4f}",
        "balanced_accuracy": "{:.4f}",
        "macro_precision": "{:.4f}",
        "macro_recall": "{:.4f}"
    })
)


# ------------------------------------------------------------
# Mean ± SD
# ------------------------------------------------------------

metric_cols = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall"
]

summary = []

for metric in metric_cols:

    summary.append({
        "metric": metric,
        "mean": results_df[metric].mean(),
        "std": results_df[metric].std(ddof=1),
        "min": results_df[metric].min(),
        "max": results_df[metric].max()
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 75)
print("EFFICIENTNET-B0 — MEAN ± SD")
print("=" * 75)

display(
    summary_df.style.format({
        "mean": "{:.4f}",
        "std": "{:.4f}",
        "min": "{:.4f}",
        "max": "{:.4f}"
    })
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_multiseed_test_results.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_multiseed_test_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSaved:")
print(results_path)
print(summary_path)

# %% [Cell 20]
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

print("=" * 70)
print("RESUME CHECK")
print("=" * 70)

print("\nDataset:")
print("metadata.csv:", (DRIVE_ROOT / "metadata.csv").exists())
print("Raw:", (DRIVE_ROOT / "Raw").exists())
print("Augmented:", (DRIVE_ROOT / "Augmented").exists())

print("\nMobileNetV3 checkpoints:")

for seed in [42, 1337, 2026]:
    path = DRIVE_ROOT / f"baseline_mobilenetv3_small_seed{seed}.pth"
    print(f"Seed {seed}: {'✅ FOUND' if path.exists() else '❌ MISSING'}")

print("\nEfficientNet-B0 checkpoints:")

for seed in [42, 1337, 2026]:
    path = DRIVE_ROOT / f"baseline_efficientnet_b0_seed{seed}.pth"
    print(f"Seed {seed}: {'✅ FOUND' if path.exists() else '❌ MISSING'}")

print("\nSaved result files:")
for p in sorted(DRIVE_ROOT.glob("*.csv")):
    print("✅", p.name)

# %% [Cell 21]
# ============================================================
# RESUME PROJECT AFTER COLAB RUNTIME RESET
# Rebuild environment + cache 850 Original images
# ============================================================

import os
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torchvision

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
DATASET_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

METADATA_PATH = DATASET_ROOT / "metadata.csv"

# ------------------------------------------------------------
# 2. Load metadata
# ------------------------------------------------------------
df = pd.read_csv(METADATA_PATH)

# ------------------------------------------------------------
# 3. Classes
# ------------------------------------------------------------
CLASS_NAMES = sorted(df["class_label"].unique())

CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    idx: name
    for name, idx in CLASS_TO_IDX.items()
}

# ------------------------------------------------------------
# 4. Device
# ------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ENVIRONMENT RESTORED")
print("=" * 70)

print("PyTorch     :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("Device      :", device)

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print(
        "VRAM        :",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

print("\nClasses:")
print(CLASS_TO_IDX)

# ------------------------------------------------------------
# 5. Original-only splits
# ------------------------------------------------------------
train_df = df[
    (df["data_split"] == "train") &
    (df["preprocessing_technique"] == "Original")
].copy()

val_df = df[
    (df["data_split"] == "val") &
    (df["preprocessing_technique"] == "Original")
].copy()

test_df = df[
    (df["data_split"] == "test") &
    (df["preprocessing_technique"] == "Original")
].copy()

print("\nDataset:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

# ------------------------------------------------------------
# 6. Transform
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# ------------------------------------------------------------
# 7. Dataset class
# ------------------------------------------------------------
class BrinjalDataset(Dataset):

    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.root_dir / row["filename"]

        image = Image.open(
            image_path
        ).convert("RGB")

        label = CLASS_TO_IDX[
            row["class_label"]
        ]

        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------------------------------------------------
# 8. Cache original images locally
# ------------------------------------------------------------
LOCAL_ROOT = Path(
    "/content/brinjal_original"
)

if LOCAL_ROOT.exists():
    shutil.rmtree(LOCAL_ROOT)

LOCAL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

original_df = df[
    df["preprocessing_technique"] == "Original"
].copy()

print("\nCaching Original images...")

for _, row in original_df.iterrows():

    src = DATASET_ROOT / row["filename"]

    relative_path = Path(row["filename"])

    dst = LOCAL_ROOT / relative_path

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(src, dst)

local_count = sum(
    1
    for p in LOCAL_ROOT.rglob("*")
    if p.is_file()
)

print("Original images cached:", local_count)

# ------------------------------------------------------------
# 9. Local datasets
# ------------------------------------------------------------
train_dataset_local = BrinjalDataset(
    train_df,
    LOCAL_ROOT,
    transform=base_transform
)

val_dataset_local = BrinjalDataset(
    val_df,
    LOCAL_ROOT,
    transform=base_transform
)

test_dataset_local = BrinjalDataset(
    test_df,
    LOCAL_ROOT,
    transform=base_transform
)

# ------------------------------------------------------------
# 10. Local loaders
# ------------------------------------------------------------
train_loader_local = DataLoader(
    train_dataset_local,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

val_loader_local = DataLoader(
    val_dataset_local,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

test_loader_local = DataLoader(
    test_dataset_local,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# ------------------------------------------------------------
# 11. Sanity check
# ------------------------------------------------------------
images, labels = next(iter(train_loader_local))

images = images.to(
    device,
    non_blocking=True
)

labels = labels.to(
    device,
    non_blocking=True
)

print("\n" + "=" * 70)
print("RESUME CHECK COMPLETE")
print("=" * 70)

print("Batch shape :", images.shape)
print("Labels shape:", labels.shape)
print("Image device:", images.device)
print("Label device:", labels.device)

print("\n✅ Ready for ResNet18")

# %% [Cell 22]
# ============================================================
# RESNET18 - MULTI-SEED BASELINE
# Same protocol as MobileNetV3-Small and EfficientNet-B0
# ============================================================

import os
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
SEEDS = [42, 1337, 2026]

MAX_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
BATCH_SIZE = 32

OUTPUT_DIR = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_resnet_seed(seed):

    print("\n" + "=" * 80)
    print(f"RESNET18 — SEED {seed}")
    print("=" * 80)

    seed_everything(seed)

    # --------------------------------------------------------
    # Seed-specific DataLoaders
    # --------------------------------------------------------
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss / optimizer / scheduler
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------
    history = []

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, MAX_EPOCHS + 1):

        start_time = time.time()

        # ====================================================
        # TRAIN
        # ====================================================
        model.train()

        train_loss_sum = 0.0
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

            train_loss_sum += (
                loss.item() * images.size(0)
            )

            preds = outputs.argmax(
                dim=1
            )

            train_correct += (
                preds == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = (
            train_loss_sum /
            train_total
        )

        train_acc = (
            train_correct /
            train_total
        )

        # ====================================================
        # VALIDATION
        # ====================================================
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
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

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                val_loss_sum += (
                    loss.item() * images.size(0)
                )

                preds = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = (
            val_loss_sum /
            val_total
        )

        val_acc = (
            val_correct /
            val_total
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------
        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            checkpoint_path = os.path.join(
                OUTPUT_DIR,
                f"baseline_resnet18_seed{seed}.pth"
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_to_idx": CLASS_TO_IDX,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "seed": seed
                },
                checkpoint_path
            )

            status = "✅ BEST"

        else:

            epochs_without_improvement += 1
            status = ""

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------
        elapsed = (
            time.time() -
            start_time
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"{elapsed:.1f}s | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"LR {current_lr:.6f} "
            f"{status}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------
        if epochs_without_improvement >= PATIENCE:

            print(
                f"\n⏹ Early stopping at epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best
    # --------------------------------------------------------
    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------
    history_df = pd.DataFrame(
        history
    )

    history_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_resnet18_seed{seed}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n" + "-" * 80)
    print(f"Seed {seed} complete")
    print(f"Best epoch    : {best_epoch}")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")
    print("-" * 80)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint": checkpoint_path
    }


# ------------------------------------------------------------
# Run all seeds
# ------------------------------------------------------------
results = []

for seed in SEEDS:

    result = train_resnet_seed(seed)

    results.append(result)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 80)
print("RESNET18 MULTI-SEED TRAINING SUMMARY")
print("=" * 80)

display(results_df)

summary_path = os.path.join(
    OUTPUT_DIR,
    "resnet18_multiseed_training_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print(
    f"\nSaved summary: {summary_path}"
)

# %% [Cell 23]
# ============================================================
# RESNET18 - MULTI-SEED TEST EVALUATION
# Seeds: 42, 1337, 2026
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import resnet18

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

SEEDS = [42, 1337, 2026]

OUTPUT_DIR = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)


def evaluate_resnet_seed(seed):

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_resnet18_seed{seed}.pth"
    )

    # Fresh architecture
    model = resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for images, labels in test_loader_local:

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

            preds = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return {
        "seed": seed,
        "accuracy": accuracy_score(
            all_labels,
            all_preds
        ),
        "macro_f1": f1_score(
            all_labels,
            all_preds,
            average="macro"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            all_labels,
            all_preds
        ),
        "macro_precision": precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )
    }


# ------------------------------------------------------------
# Evaluate all seeds
# ------------------------------------------------------------

results = []

for seed in SEEDS:

    print(f"\nEvaluating ResNet18 seed {seed}...")

    result = evaluate_resnet_seed(seed)

    results.append(result)

    print(
        f"Accuracy: {result['accuracy']:.4f} | "
        f"Macro-F1: {result['macro_f1']:.4f} | "
        f"Balanced Acc: {result['balanced_accuracy']:.4f}"
    )


# ------------------------------------------------------------
# Results table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("RESNET18 MULTI-SEED TEST RESULTS")
print("=" * 75)

display(
    results_df.style.format({
        "accuracy": "{:.4f}",
        "macro_f1": "{:.4f}",
        "balanced_accuracy": "{:.4f}",
        "macro_precision": "{:.4f}",
        "macro_recall": "{:.4f}"
    })
)


# ------------------------------------------------------------
# Mean ± SD
# ------------------------------------------------------------

metric_cols = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall"
]

summary = []

for metric in metric_cols:

    summary.append({
        "metric": metric,
        "mean": results_df[metric].mean(),
        "std": results_df[metric].std(ddof=1),
        "min": results_df[metric].min(),
        "max": results_df[metric].max()
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 75)
print("RESNET18 — MEAN ± SD")
print("=" * 75)

display(
    summary_df.style.format({
        "mean": "{:.4f}",
        "std": "{:.4f}",
        "min": "{:.4f}",
        "max": "{:.4f}"
    })
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results_path = os.path.join(
    OUTPUT_DIR,
    "resnet18_multiseed_test_results.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "resnet18_multiseed_test_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSaved:")
print(results_path)
print(summary_path)

# %% [Cell 24]
# ============================================================
# MOBILENETV2 - MULTI-SEED BASELINE
# Same protocol as MobileNetV3-Small / EfficientNet-B0 / ResNet18
# ============================================================

import os
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
SEEDS = [42, 1337, 2026]

MAX_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
BATCH_SIZE = 32

OUTPUT_DIR = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_mobilenetv2_seed(seed):

    print("\n" + "=" * 80)
    print(f"MOBILENETV2 — SEED {seed}")
    print("=" * 80)

    seed_everything(seed)

    # --------------------------------------------------------
    # Seed-specific DataLoaders
    # --------------------------------------------------------
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    weights = MobileNet_V2_Weights.DEFAULT

    model = mobilenet_v2(
        weights=weights
    )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss / optimizer / scheduler
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------
    history = []

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, MAX_EPOCHS + 1):

        start_time = time.time()

        # ====================================================
        # TRAIN
        # ====================================================
        model.train()

        train_loss_sum = 0.0
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

            train_loss_sum += (
                loss.item() * images.size(0)
            )

            preds = outputs.argmax(
                dim=1
            )

            train_correct += (
                preds == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = (
            train_loss_sum /
            train_total
        )

        train_acc = (
            train_correct /
            train_total
        )

        # ====================================================
        # VALIDATION
        # ====================================================
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
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

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                val_loss_sum += (
                    loss.item() * images.size(0)
                )

                preds = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = (
            val_loss_sum /
            val_total
        )

        val_acc = (
            val_correct /
            val_total
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------
        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            checkpoint_path = os.path.join(
                OUTPUT_DIR,
                f"baseline_mobilenetv2_seed{seed}.pth"
            )

            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_to_idx": CLASS_TO_IDX,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "seed": seed
                },
                checkpoint_path
            )

            status = "✅ BEST"

        else:

            epochs_without_improvement += 1
            status = ""

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------
        elapsed = (
            time.time() -
            start_time
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"{elapsed:.1f}s | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"LR {current_lr:.6f} "
            f"{status}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------
        if epochs_without_improvement >= PATIENCE:

            print(
                f"\n⏹ Early stopping at epoch {epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best
    # --------------------------------------------------------
    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------
    history_df = pd.DataFrame(
        history
    )

    history_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_mobilenetv2_seed{seed}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n" + "-" * 80)
    print(f"Seed {seed} complete")
    print(f"Best epoch    : {best_epoch}")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")
    print("-" * 80)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "checkpoint": checkpoint_path
    }


# ------------------------------------------------------------
# Run all seeds
# ------------------------------------------------------------
results = []

for seed in SEEDS:

    result = train_mobilenetv2_seed(seed)

    results.append(result)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 80)
print("MOBILENETV2 MULTI-SEED TRAINING SUMMARY")
print("=" * 80)

display(results_df)

summary_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv2_multiseed_training_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print(
    f"\nSaved summary: {summary_path}"
)

# %% [Cell 25]
# ============================================================
# MOBILENETV2 - MULTI-SEED TEST EVALUATION
# Seeds: 42, 1337, 2026
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

SEEDS = [42, 1337, 2026]

OUTPUT_DIR = (
    "/content/drive/MyDrive/"
    "Brinjal_Final_Preprocessed"
)


def evaluate_mobilenetv2_seed(seed):

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"baseline_mobilenetv2_seed{seed}.pth"
    )

    # Fresh architecture
    model = mobilenet_v2(weights=None)

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for images, labels in test_loader_local:

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

            preds = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return {
        "seed": seed,
        "accuracy": accuracy_score(
            all_labels,
            all_preds
        ),
        "macro_f1": f1_score(
            all_labels,
            all_preds,
            average="macro"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            all_labels,
            all_preds
        ),
        "macro_precision": precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )
    }


# ------------------------------------------------------------
# Evaluate all seeds
# ------------------------------------------------------------

results = []

for seed in SEEDS:

    print(f"\nEvaluating MobileNetV2 seed {seed}...")

    result = evaluate_mobilenetv2_seed(seed)

    results.append(result)

    print(
        f"Accuracy: {result['accuracy']:.4f} | "
        f"Macro-F1: {result['macro_f1']:.4f} | "
        f"Balanced Acc: {result['balanced_accuracy']:.4f}"
    )


# ------------------------------------------------------------
# Results table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("MOBILENETV2 MULTI-SEED TEST RESULTS")
print("=" * 75)

display(
    results_df.style.format({
        "accuracy": "{:.4f}",
        "macro_f1": "{:.4f}",
        "balanced_accuracy": "{:.4f}",
        "macro_precision": "{:.4f}",
        "macro_recall": "{:.4f}"
    })
)


# ------------------------------------------------------------
# Mean ± SD
# ------------------------------------------------------------

metric_cols = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall"
]

summary = []

for metric in metric_cols:

    summary.append({
        "metric": metric,
        "mean": results_df[metric].mean(),
        "std": results_df[metric].std(ddof=1),
        "min": results_df[metric].min(),
        "max": results_df[metric].max()
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 75)
print("MOBILENETV2 — MEAN ± SD")
print("=" * 75)

display(
    summary_df.style.format({
        "mean": "{:.4f}",
        "std": "{:.4f}",
        "min": "{:.4f}",
        "max": "{:.4f}"
    })
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv2_multiseed_test_results.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "mobilenetv2_multiseed_test_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSaved:")
print(results_path)
print(summary_path)

# %% [Cell 26]
# ============================================================
# EFFICIENCY AUDIT - ALL FOUR BASELINE MODELS
# ============================================================

!pip -q install thop

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path

from torchvision.models import (
    mobilenet_v2,
    mobilenet_v3_small,
    efficientnet_b0,
    resnet18
)

from thop import profile


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

NUM_WARMUP = 20
NUM_TIMING = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------

def create_model(model_name):

    if model_name == "MobileNetV2":

        model = mobilenet_v2(
            weights=None
        )

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            3
        )

    elif model_name == "MobileNetV3-Small":

        model = mobilenet_v3_small(
            weights=None
        )

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            3
        )

    elif model_name == "EfficientNet-B0":

        model = efficientnet_b0(
            weights=None
        )

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            3
        )

    elif model_name == "ResNet18":

        model = resnet18(
            weights=None
        )

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            3
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ------------------------------------------------------------
# Checkpoint paths
# ------------------------------------------------------------

checkpoint_map = {

    "MobileNetV2":
        OUTPUT_DIR /
        "baseline_mobilenetv2_seed42.pth",

    "MobileNetV3-Small":
        OUTPUT_DIR /
        "baseline_mobilenetv3_small_seed42.pth",

    "EfficientNet-B0":
        OUTPUT_DIR /
        "baseline_efficientnet_b0_seed42.pth",

    "ResNet18":
        OUTPUT_DIR /
        "baseline_resnet18_seed42.pth"
}


# ------------------------------------------------------------
# Latency benchmark
# ------------------------------------------------------------

def benchmark_latency(
    model,
    batch_size
):

    model.eval()

    dummy = torch.randn(
        batch_size,
        3,
        224,
        224,
        device=DEVICE
    )

    # Warm-up
    with torch.no_grad():

        for _ in range(NUM_WARMUP):
            _ = model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(NUM_TIMING):
            _ = model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_images = (
        batch_size *
        NUM_TIMING
    )

    latency_per_batch_ms = (
        elapsed /
        NUM_TIMING *
        1000
    )

    latency_per_image_ms = (
        elapsed /
        total_images *
        1000
    )

    throughput = (
        total_images /
        elapsed
    )

    return {
        "batch_latency_ms": latency_per_batch_ms,
        "image_latency_ms": latency_per_image_ms,
        "throughput_img_s": throughput
    }


# ------------------------------------------------------------
# Audit
# ------------------------------------------------------------

MODEL_NAMES = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18"
]

results = []


for model_name in MODEL_NAMES:

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model = create_model(
        model_name
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # --------------------------------------------------------
    # FLOPs / MACs
    # --------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device=DEVICE
    )

    macs, _ = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )

    # THOP reports MACs.
    # FLOPs ≈ 2 × MACs.
    flops = 2 * macs

    # --------------------------------------------------------
    # Checkpoint size
    # --------------------------------------------------------

    checkpoint_path = checkpoint_map[
        model_name
    ]

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size /
        (1024 ** 2)
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency_bs1 = benchmark_latency(
        model,
        batch_size=1
    )

    latency_bs32 = benchmark_latency(
        model,
        batch_size=32
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "model":
            model_name,

        "parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "parameters_M":
            total_params / 1e6,

        "MACs_G":
            macs / 1e9,

        "FLOPs_G_approx":
            flops / 1e9,

        "checkpoint_MB":
            checkpoint_size_mb,

        "latency_batch1_ms":
            latency_bs1[
                "batch_latency_ms"
            ],

        "latency_image_batch1_ms":
            latency_bs1[
                "image_latency_ms"
            ],

        "latency_batch32_ms":
            latency_bs32[
                "batch_latency_ms"
            ],

        "latency_image_batch32_ms":
            latency_bs32[
                "image_latency_ms"
            ],

        "throughput_batch32_img_s":
            latency_bs32[
                "throughput_img_s"
            ]
    }

    results.append(result)

    print(
        f"Parameters       : "
        f"{result['parameters_M']:.2f} M"
    )

    print(
        f"MACs             : "
        f"{result['MACs_G']:.3f} G"
    )

    print(
        f"Approx FLOPs     : "
        f"{result['FLOPs_G_approx']:.3f} G"
    )

    print(
        f"Checkpoint       : "
        f"{result['checkpoint_MB']:.2f} MB"
    )

    print(
        f"BS=1 latency     : "
        f"{result['latency_image_batch1_ms']:.3f} ms/image"
    )

    print(
        f"BS=32 latency    : "
        f"{result['latency_image_batch32_ms']:.3f} ms/image"
    )

    print(
        f"BS=32 throughput : "
        f"{result['throughput_batch32_img_s']:.2f} img/s"
    )

    # Free model
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------
# Final table
# ------------------------------------------------------------

efficiency_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("EFFICIENCY AUDIT")
print("=" * 70)

display(
    efficiency_df.style.format({

        "parameters_M":
            "{:.2f}",

        "MACs_G":
            "{:.3f}",

        "FLOPs_G_approx":
            "{:.3f}",

        "checkpoint_MB":
            "{:.2f}",

        "latency_batch1_ms":
            "{:.3f}",

        "latency_image_batch1_ms":
            "{:.3f}",

        "latency_batch32_ms":
            "{:.3f}",

        "latency_image_batch32_ms":
            "{:.3f}",

        "throughput_batch32_img_s":
            "{:.2f}"
    })
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

efficiency_path = (
    OUTPUT_DIR /
    "baseline_efficiency_audit.csv"
)

efficiency_df.to_csv(
    efficiency_path,
    index=False
)

print(
    f"\nSaved: {efficiency_path}"
)

# %% [Cell 27]
# ============================================================
# PAIRED CLEAN-TEST ANALYSIS
# Compare predictions of all 4 baseline models
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v2,
    mobilenet_v3_small,
    efficientnet_b0,
    resnet18
)

from sklearn.metrics import confusion_matrix


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------

def build_model(model_name):

    if model_name == "MobileNetV2":

        model = mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "MobileNetV3-Small":

        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "EfficientNet-B0":

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "ResNet18":

        model = resnet18(weights=None)
        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    else:
        raise ValueError(model_name)

    return model


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

OUTPUT_DIR = "/content/drive/MyDrive/Brinjal_Final_Preprocessed"

checkpoint_map = {

    "MobileNetV2":
        f"{OUTPUT_DIR}/baseline_mobilenetv2_seed42.pth",

    "MobileNetV3-Small":
        f"{OUTPUT_DIR}/baseline_mobilenetv3_small_seed42.pth",

    "EfficientNet-B0":
        f"{OUTPUT_DIR}/baseline_efficientnet_b0_seed42.pth",

    "ResNet18":
        f"{OUTPUT_DIR}/baseline_resnet18_seed42.pth"
}


MODEL_NAMES = list(checkpoint_map.keys())


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

all_labels = []

for _, labels in test_loader_local:

    all_labels.extend(
        labels.numpy()
    )

all_labels = np.array(all_labels)


# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------

prediction_dict = {}

for model_name in MODEL_NAMES:

    print(f"\nPredicting with {model_name}...")

    model = build_model(model_name)

    checkpoint = torch.load(
        checkpoint_map[model_name],
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():

        for images, labels in test_loader_local:

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

    prediction_dict[model_name] = np.array(
        predictions
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------
# Prediction DataFrame
# ------------------------------------------------------------

pred_df = test_df.copy().reset_index(drop=True)

pred_df["true_class"] = [
    IDX_TO_CLASS[x]
    for x in all_labels
]

for model_name in MODEL_NAMES:

    pred_df[f"pred_{model_name}"] = [
        IDX_TO_CLASS[x]
        for x in prediction_dict[model_name]
    ]

    pred_df[f"correct_{model_name}"] = (
        prediction_dict[model_name] ==
        all_labels
    )


# ------------------------------------------------------------
# Individual correct counts
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("INDIVIDUAL CORRECT PREDICTIONS")
print("=" * 75)

for model_name in MODEL_NAMES:

    correct = (
        pred_df[f"correct_{model_name}"]
        .sum()
    )

    print(
        f"{model_name:<20} : "
        f"{correct}/128 = "
        f"{correct / 128:.4f}"
    )


# ------------------------------------------------------------
# Pairwise agreement
# ------------------------------------------------------------

agreement_matrix = pd.DataFrame(
    index=MODEL_NAMES,
    columns=MODEL_NAMES,
    dtype=float
)

for model_a in MODEL_NAMES:

    for model_b in MODEL_NAMES:

        agreement = np.mean(
            prediction_dict[model_a] ==
            prediction_dict[model_b]
        )

        agreement_matrix.loc[
            model_a,
            model_b
        ] = agreement


print("\n" + "=" * 75)
print("PAIRWISE PREDICTION AGREEMENT")
print("=" * 75)

display(
    agreement_matrix.style.format(
        "{:.4f}"
    )
)


# ------------------------------------------------------------
# All-model agreement
# ------------------------------------------------------------

prediction_matrix = np.column_stack([
    prediction_dict[m]
    for m in MODEL_NAMES
])

all_same = np.all(
    prediction_matrix ==
    prediction_matrix[:, [0]],
    axis=1
)

print("\nAll 4 models made the same prediction:")
print(
    f"{all_same.sum()}/128 "
    f"({all_same.mean():.2%})"
)


# ------------------------------------------------------------
# All-model correct
# ------------------------------------------------------------

correct_matrix = np.column_stack([
    prediction_dict[m] == all_labels
    for m in MODEL_NAMES
])

all_correct = np.all(
    correct_matrix,
    axis=1
)

all_wrong = np.all(
    ~correct_matrix,
    axis=1
)

print("\nAll 4 models correct:")
print(
    f"{all_correct.sum()}/128 "
    f"({all_correct.mean():.2%})"
)

print("\nAll 4 models wrong:")
print(
    f"{all_wrong.sum()}/128 "
    f"({all_wrong.mean():.2%})"
)


# ------------------------------------------------------------
# Number of correct models per leaf
# ------------------------------------------------------------

num_models_correct = correct_matrix.sum(
    axis=1
)

distribution = pd.Series(
    num_models_correct
).value_counts().sort_index()

print("\n" + "=" * 75)
print("NUMBER OF MODELS CORRECT PER TEST LEAF")
print("=" * 75)

for k, count in distribution.items():

    print(
        f"{k} correct models : "
        f"{count} leaves"
    )


# ------------------------------------------------------------
# Cases where models disagree
# ------------------------------------------------------------

disagreement_mask = (
    prediction_matrix !=
    prediction_matrix[:, [0]]
).any(axis=1)

disagreements = pred_df.loc[
    disagreement_mask
].copy()

print("\n" + "=" * 75)
print("MODEL DISAGREEMENTS")
print("=" * 75)

print(
    f"Leaves with disagreement: "
    f"{len(disagreements)}/128"
)

display(
    disagreements[
        [
            "filename",
            "true_class"
        ]
        +
        [
            f"pred_{m}"
            for m in MODEL_NAMES
        ]
    ].head(30)
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

prediction_path = (
    f"{OUTPUT_DIR}/"
    "four_model_clean_test_predictions.csv"
)

agreement_path = (
    f"{OUTPUT_DIR}/"
    "four_model_prediction_agreement.csv"
)

pred_df.to_csv(
    prediction_path,
    index=False
)

agreement_matrix.to_csv(
    agreement_path
)

print("\nSaved:")
print(prediction_path)
print(agreement_path)

# %% [Cell 28]
# ============================================================
# CACHE ALL TEST VARIANTS LOCALLY
# 128 test leaves × 16 variants = 2048 images
# ============================================================

import shutil
import time
from pathlib import Path

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

ROBUSTNESS_ROOT = Path(
    "/content/brinjal_test_all"
)

if ROBUSTNESS_ROOT.exists():
    shutil.rmtree(ROBUSTNESS_ROOT)

ROBUSTNESS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

test_all_df = df[
    df["data_split"] == "test"
].copy()

print("Test images to cache:", len(test_all_df))

start = time.time()

for _, row in test_all_df.iterrows():

    src = DRIVE_ROOT / row["filename"]

    relative_path = Path(row["filename"])

    dst = ROBUSTNESS_ROOT / relative_path

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
    for p in ROBUSTNESS_ROOT.rglob("*")
    if p.is_file()
)

print("\n✅ Test cache complete")
print(f"Copied: {len(test_all_df)}")
print(f"Local files: {local_count}")
print(f"Time: {elapsed:.1f} sec")

# %% [Cell 29]
# ============================================================
# PHASE 5 — PILOT TRANSFORMATION ROBUSTNESS
# 4 models × 16 transformations × Seed 42
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchvision.models import (
    mobilenet_v2,
    mobilenet_v3_small,
    efficientnet_b0,
    resnet18
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

# ============================================================
# 1. Paths
# ============================================================

DATASET_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

ROBUSTNESS_ROOT = Path(
    "/content/brinjal_test_all"
)

OUTPUT_DIR = DATASET_ROOT

# ============================================================
# 2. Load metadata
# ============================================================

test_all_df = df[
    df["data_split"] == "test"
].copy().reset_index(drop=True)

# ============================================================
# 3. Transformation order
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

print("Transformations:", len(TRANSFORMATIONS))
print("Test leaves:", test_all_df["original_image_id"].nunique())
print("Test images:", len(test_all_df))

# ============================================================
# 4. Image transform
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
# 5. Dataset
# ============================================================

class RobustnessDataset(Dataset):

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

        path = (
            self.root_dir /
            row["filename"]
        )

        image = Image.open(
            path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_IDX[
            row["class_label"]
        ]

        return (
            image,
            label,
            row["original_image_id"],
            row["filename"]
        )

# ============================================================
# 6. DataLoader factory
# ============================================================

def make_loader(
    transformation
):

    subset = test_all_df[
        test_all_df[
            "preprocessing_technique"
        ] == transformation
    ].copy()

    dataset = RobustnessDataset(
        subset,
        ROBUSTNESS_ROOT,
        transform=IMAGE_TRANSFORM
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return subset, loader

# ============================================================
# 7. Model factory
# ============================================================

def build_model(model_name):

    if model_name == "MobileNetV2":

        model = mobilenet_v2(
            weights=None
        )

        in_features = (
            model.classifier[-1].in_features
        )

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "MobileNetV3-Small":

        model = mobilenet_v3_small(
            weights=None
        )

        in_features = (
            model.classifier[-1].in_features
        )

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "EfficientNet-B0":

        model = efficientnet_b0(
            weights=None
        )

        in_features = (
            model.classifier[-1].in_features
        )

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "ResNet18":

        model = resnet18(
            weights=None
        )

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    else:
        raise ValueError(model_name)

    return model

# ============================================================
# 8. Checkpoints — Seed 42
# ============================================================

MODEL_NAMES = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18"
]

CHECKPOINTS = {
    "MobileNetV2":
        OUTPUT_DIR /
        "baseline_mobilenetv2_seed42.pth",

    "MobileNetV3-Small":
        OUTPUT_DIR /
        "baseline_mobilenetv3_small_seed42.pth",

    "EfficientNet-B0":
        OUTPUT_DIR /
        "baseline_efficientnet_b0_seed42.pth",

    "ResNet18":
        OUTPUT_DIR /
        "baseline_resnet18_seed42.pth"
}

# ============================================================
# 9. Prediction function
# ============================================================

def predict_model(
    model,
    loader
):

    model.eval()

    labels_all = []
    preds_all = []
    ids_all = []
    filenames_all = []

    with torch.no_grad():

        for (
            images,
            labels,
            leaf_ids,
            filenames
        ) in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

            preds = outputs.argmax(
                dim=1
            )

            labels_all.extend(
                labels.numpy()
            )

            preds_all.extend(
                preds.cpu().numpy()
            )

            ids_all.extend(
                list(leaf_ids)
            )

            filenames_all.extend(
                list(filenames)
            )

    return (
        np.array(labels_all),
        np.array(preds_all),
        np.array(ids_all),
        np.array(filenames_all)
    )

# ============================================================
# 10. Run robustness benchmark
# ============================================================

all_results = []
all_predictions = []

for model_name in MODEL_NAMES:

    print("\n" + "=" * 75)
    print(model_name)
    print("=" * 75)

    # Load checkpoint
    model = build_model(
        model_name
    )

    checkpoint = torch.load(
        CHECKPOINTS[model_name],
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    baseline_f1 = None

    for transformation in TRANSFORMATIONS:

        subset, loader = make_loader(
            transformation
        )

        labels, preds, leaf_ids, filenames = (
            predict_model(
                model,
                loader
            )
        )

        accuracy = accuracy_score(
            labels,
            preds
        )

        macro_f1 = f1_score(
            labels,
            preds,
            average="macro"
        )

        balanced_acc = balanced_accuracy_score(
            labels,
            preds
        )

        # Save Original F1
        if transformation == "Original":

            baseline_f1 = macro_f1

        performance_drop = (
            baseline_f1 - macro_f1
        )

        # ----------------------------------------------------
        # Prediction consistency
        # ----------------------------------------------------

        if transformation == "Original":

            original_predictions = preds.copy()

            consistency = 1.0

        else:

            consistency = np.mean(
                preds ==
                original_predictions
            )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        all_results.append({

            "model":
                model_name,

            "seed":
                42,

            "transformation":
                transformation,

            "n_samples":
                len(labels),

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "balanced_accuracy":
                balanced_acc,

            "performance_drop":
                performance_drop,

            "prediction_consistency":
                consistency
        })

        # ----------------------------------------------------
        # Individual predictions
        # ----------------------------------------------------

        for i in range(len(labels)):

            all_predictions.append({

                "model":
                    model_name,

                "seed":
                    42,

                "transformation":
                    transformation,

                "original_image_id":
                    leaf_ids[i],

                "filename":
                    filenames[i],

                "true_label":
                    IDX_TO_CLASS[
                        int(labels[i])
                    ],

                "predicted_label":
                    IDX_TO_CLASS[
                        int(preds[i])
                    ],

                "correct":
                    bool(
                        labels[i] ==
                        preds[i]
                    )
            })

        print(
            f"{transformation:<18} "
            f"Acc={accuracy:.4f} | "
            f"F1={macro_f1:.4f} | "
            f"Cons={consistency:.4f}"
        )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# 11. Save results
# ============================================================

robustness_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)

results_path = (
    OUTPUT_DIR /
    "robustness_pilot_seed42_results.csv"
)

predictions_path = (
    OUTPUT_DIR /
    "robustness_pilot_seed42_predictions.csv"
)

robustness_df.to_csv(
    results_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print("\n" + "=" * 75)
print("ROBUSTNESS PILOT COMPLETE")
print("=" * 75)

print("Results:")
display(robustness_df)

print("\nSaved:")
print(results_path)
print(predictions_path)

# %% [Cell 30]
# ============================================================
# ROBUSTNESS ALIGNMENT SANITY CHECK
# ============================================================

import pandas as pd

test_all_df = df[
    df["data_split"] == "test"
].copy()

transformations = [
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

original_ids = set(
    test_all_df.loc[
        test_all_df["preprocessing_technique"] == "Original",
        "original_image_id"
    ]
)

print("Original test leaves:", len(original_ids))

all_ok = True

for transformation in transformations:

    ids = list(
        test_all_df.loc[
            test_all_df["preprocessing_technique"] == transformation,
            "original_image_id"
        ]
    )

    ids_set = set(ids)

    same_set = (
        ids_set == original_ids
    )

    unique_count = (
        len(ids) == len(set(ids))
    )

    print(
        f"{transformation:<18} "
        f"n={len(ids):3d} | "
        f"same leaf set={same_set} | "
        f"unique IDs={unique_count}"
    )

    if not same_set or not unique_count:
        all_ok = False

print("\n" + "=" * 70)

if all_ok:
    print(
        "✅ ALIGNMENT CHECK PASSED: "
        "Every transformation contains exactly the same 128 test leaves."
    )
else:
    print(
        "❌ ALIGNMENT CHECK FAILED: "
        "Do NOT continue until alignment is fixed."
    )

# %% [Cell 31]
# ============================================================
# PHASE 5 — FULL TRANSFORMATION ROBUSTNESS
# 4 MODELS × 3 SEEDS × 16 TRANSFORMATIONS
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchvision.models import (
    mobilenet_v2,
    mobilenet_v3_small,
    efficientnet_b0,
    resnet18
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score
)

# ============================================================
# 1. Paths
# ============================================================

DATASET_ROOT = Path(
    "/content/drive/MyDrive/Brinjal_Final_Preprocessed"
)

ROBUSTNESS_ROOT = Path(
    "/content/brinjal_test_all"
)

OUTPUT_DIR = DATASET_ROOT

# ============================================================
# 2. Settings
# ============================================================

MODEL_NAMES = [
    "MobileNetV2",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet18"
]

SEEDS = [42, 1337, 2026]

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
# 3. Metadata
# ============================================================

test_all_df = df[
    df["data_split"] == "test"
].copy()

# ============================================================
# 4. Image preprocessing
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
# 5. Dataset
# ============================================================

class RobustnessDataset(Dataset):

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

        path = (
            self.root_dir /
            row["filename"]
        )

        image = Image.open(
            path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_IDX[
            row["class_label"]
        ]

        return (
            image,
            label,
            row["original_image_id"],
            row["filename"]
        )

# ============================================================
# 6. Model factory
# ============================================================

def build_model(model_name):

    if model_name == "MobileNetV2":

        model = mobilenet_v2(
            weights=None
        )

        in_features = (
            model.classifier[-1].in_features
        )

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "MobileNetV3-Small":

        model = mobilenet_v3_small(
            weights=None
        )

        in_features = (
            model.classifier[-1].in_features
        )

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "EfficientNet-B0":

        model = efficientnet_b0(
            weights=None
        )

        in_features = (
            model.classifier[-1].in_features
        )

        model.classifier[-1] = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    elif model_name == "ResNet18":

        model = resnet18(
            weights=None
        )

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            len(CLASS_NAMES)
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model

# ============================================================
# 7. Checkpoint map
# ============================================================

CHECKPOINTS = {

    "MobileNetV2":
        lambda seed:
        OUTPUT_DIR /
        f"baseline_mobilenetv2_seed{seed}.pth",

    "MobileNetV3-Small":
        lambda seed:
        OUTPUT_DIR /
        f"baseline_mobilenetv3_small_seed{seed}.pth",

    "EfficientNet-B0":
        lambda seed:
        OUTPUT_DIR /
        f"baseline_efficientnet_b0_seed{seed}.pth",

    "ResNet18":
        lambda seed:
        OUTPUT_DIR /
        f"baseline_resnet18_seed{seed}.pth"
}

# ============================================================
# 8. DataLoader creation
# ============================================================

def make_loader(transformation):

    subset = test_all_df[
        test_all_df[
            "preprocessing_technique"
        ] == transformation
    ].copy()

    dataset = RobustnessDataset(
        subset,
        ROBUSTNESS_ROOT,
        transform=IMAGE_TRANSFORM
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return subset, loader

# ============================================================
# 9. Prediction function
# ============================================================

def predict_model(
    model,
    loader
):

    model.eval()

    records = []

    with torch.no_grad():

        for (
            images,
            labels,
            leaf_ids,
            filenames
        ) in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

            preds = outputs.argmax(
                dim=1
            )

            labels_np = labels.numpy()
            preds_np = preds.cpu().numpy()

            for i in range(len(labels_np)):

                records.append({

                    "original_image_id":
                        str(leaf_ids[i]),

                    "filename":
                        str(filenames[i]),

                    "true_label":
                        int(labels_np[i]),

                    "pred_label":
                        int(preds_np[i])
                })

    return pd.DataFrame(records)

# ============================================================
# 10. Main robustness experiment
# ============================================================

metric_results = []
prediction_results = []

for model_name in MODEL_NAMES:

    for seed in SEEDS:

        print("\n" + "=" * 85)
        print(
            f"MODEL: {model_name} | "
            f"SEED: {seed}"
        )
        print("=" * 85)

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        model = build_model(
            model_name
        )

        checkpoint_path = CHECKPOINTS[
            model_name
        ](seed)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(device)

        original_predictions = None
        original_f1 = None

        # ----------------------------------------------------
        # All transformations
        # ----------------------------------------------------

        for transformation in TRANSFORMATIONS:

            subset, loader = make_loader(
                transformation
            )

            pred_df = predict_model(
                model,
                loader
            )

            y_true = pred_df[
                "true_label"
            ].to_numpy()

            y_pred = pred_df[
                "pred_label"
            ].to_numpy()

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
                average="macro"
            )

            balanced_acc = balanced_accuracy_score(
                y_true,
                y_pred
            )

            # ------------------------------------------------
            # Original baseline
            # ------------------------------------------------

            if transformation == "Original":

                original_predictions = (
                    pred_df[
                        [
                            "original_image_id",
                            "pred_label"
                        ]
                    ]
                    .rename(
                        columns={
                            "pred_label":
                            "original_pred"
                        }
                    )
                )

                original_f1 = macro_f1

                consistency = 1.0

            else:

                merged = pred_df[
                    [
                        "original_image_id",
                        "pred_label"
                    ]
                ].merge(
                    original_predictions,
                    on="original_image_id",
                    how="inner",
                    validate="one_to_one"
                )

                # Safety check
                assert len(merged) == 128, (
                    f"Alignment error: "
                    f"{model_name}, "
                    f"{seed}, "
                    f"{transformation}"
                )

                consistency = np.mean(
                    merged["pred_label"].to_numpy()
                    ==
                    merged["original_pred"].to_numpy()
                )

            # ------------------------------------------------
            # Performance delta
            # ------------------------------------------------

            delta_f1 = (
                macro_f1 -
                original_f1
            )

            # positive = improvement
            # negative = degradation

            performance_drop = (
                original_f1 -
                macro_f1
            )

            # ------------------------------------------------
            # Metrics record
            # ------------------------------------------------

            metric_results.append({

                "model":
                    model_name,

                "seed":
                    seed,

                "transformation":
                    transformation,

                "n_samples":
                    len(y_true),

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,

                "balanced_accuracy":
                    balanced_acc,

                "original_macro_f1":
                    original_f1,

                "delta_f1":
                    delta_f1,

                "performance_drop":
                    performance_drop,

                "prediction_consistency":
                    consistency
            })

            # ------------------------------------------------
            # Prediction records
            # ------------------------------------------------

            pred_df["model"] = model_name
            pred_df["seed"] = seed
            pred_df["transformation"] = transformation

            prediction_results.append(
                pred_df
            )

            print(
                f"{transformation:<18} "
                f"F1={macro_f1:.4f} | "
                f"ΔF1={delta_f1:+.4f} | "
                f"Cons={consistency:.4f}"
            )

        # ----------------------------------------------------
        # Free model
        # ----------------------------------------------------

        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================
# 11. Combine results
# ============================================================

robustness_results = pd.DataFrame(
    metric_results
)

robustness_predictions = pd.concat(
    prediction_results,
    ignore_index=True
)

# ============================================================
# 12. Save raw results
# ============================================================

results_path = (
    OUTPUT_DIR /
    "full_robustness_results.csv"
)

predictions_path = (
    OUTPUT_DIR /
    "full_robustness_predictions.csv"
)

robustness_results.to_csv(
    results_path,
    index=False
)

robustness_predictions.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# 13. Aggregate across seeds
# ============================================================

aggregate = (
    robustness_results
    .groupby(
        [
            "model",
            "transformation"
        ],
        as_index=False
    )
    .agg({

        "accuracy":
            ["mean", "std"],

        "macro_f1":
            ["mean", "std"],

        "balanced_accuracy":
            ["mean", "std"],

        "delta_f1":
            ["mean", "std"],

        "performance_drop":
            ["mean", "std"],

        "prediction_consistency":
            ["mean", "std"]
    })
)

aggregate.columns = [
    "_".join(col).strip("_")
    if isinstance(col, tuple)
    else col
    for col in aggregate.columns
]

aggregate_path = (
    OUTPUT_DIR /
    "full_robustness_aggregate.csv"
)

aggregate.to_csv(
    aggregate_path,
    index=False
)

# ============================================================
# 14. Print key summary
# ============================================================

print("\n" + "=" * 90)
print("FULL ROBUSTNESS EXPERIMENT COMPLETE")
print("=" * 90)

print(
    f"Rows of metrics: "
    f"{len(robustness_results)}"
)

print(
    f"Rows of predictions: "
    f"{len(robustness_predictions)}"
)

print("\nSaved:")
print(results_path)
print(predictions_path)
print(aggregate_path)

print("\nAggregate results:")
display(aggregate)

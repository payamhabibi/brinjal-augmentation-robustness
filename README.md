# Brinjal Augmentation Robustness

Reproducible code and result artifacts for the brinjal leaf image-classification study on augmentation, robustness, targeted augmentation, and statistical comparison.

## Scope

This repository documents the experiments implemented in the accompanying Jupyter notebook:

- Four CNN baselines: MobileNetV2, MobileNetV3-Small, EfficientNet-B0, and ResNet18
- Three random seeds: 42, 1337, 2026
- Locked-test robustness evaluation across 16 image transformations
- Targeted augmentation ablations
- RandAugment and AugMix comparisons
- ResNet18 augmentation/control experiments
- Unseen-severity A3 evaluation
- Exact paired McNemar tests with Benjamini-Hochberg FDR correction
- Error analysis / Grad-CAM and efficiency auditing

## Dataset

The experiment metadata contains 13,600 image records derived from 850 original brinjal leaves, with 16 versions per leaf.

Classes:

| Class | Original leaves |
|---|---:|
| Healthy_Leaves | 400 |
| Little_Leaf | 200 |
| Phomopsis_Blight | 250 |
| **Total** | **850** |

The original-leaf split is:

| Split | Original leaves |
|---|---:|
| Train | 595 |
| Validation | 127 |
| Test | 128 |

The metadata audit reported 13,600 unique filenames, zero missing image files, exactly 16 variants per original leaf, and zero leaf-level split leakage.

The full dataset is **not committed to GitHub** because of size and data-distribution constraints. Use the dataset location/configuration described in the notebook and the paper.

## Environment

The recorded final experiment environment was:

- Python 3
- PyTorch 2.11.0+cu130
- Torchvision 0.26.0+cu130
- NVIDIA Tesla T4
- 14.56 GB VRAM
- Input size: 224 × 224
- Batch size: 32 in the main training protocols
- ImageNet normalization
- AdamW optimizer
- Initial learning rate: 1e-4
- Weight decay: 1e-4
- Maximum epochs: 20
- Early stopping / validation-based checkpoint selection

## Baseline results

The locked original test set contains 128 original leaves.

### MobileNetV3-Small

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 42 | 0.9219 | 0.8995 | 0.8936 |
| 1337 | 0.9219 | 0.8979 | 0.8912 |
| 2026 | 0.9297 | 0.9091 | 0.9023 |

### EfficientNet-B0

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 42 | 0.9297 | 0.9091 | 0.9023 |
| 1337 | 0.9219 | 0.8979 | 0.8912 |
| 2026 | 0.9219 | 0.9006 | 0.8936 |

### ResNet18

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 42 | 0.9141 | 0.8899 | 0.8848 |
| 1337 | 0.9297 | 0.9094 | 0.9047 |
| 2026 | 0.9297 | 0.9105 | 0.9047 |

### MobileNetV2

| Seed | Accuracy | Macro-F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 42 | 0.9219 | 0.8997 | 0.8959 |
| 1337 | 0.9219 | 0.8995 | 0.8936 |
| 2026 | 0.9297 | 0.9104 | 0.9070 |

## Robustness evaluation

Robustness was evaluated on 16 transformation conditions:

1. Original
2. Grayscale
3. CLAHE
4. Gamma
5. HSV
6. Brightness
7. Contrast
8. Sharpen
9. Gaussian_Blur
10. Median_Blur
11. Bilateral
12. TopHat
13. BlackHat
14. Rotate
15. Flip
16. Unsharp_Mask

For the pilot robustness evaluation, all four models were evaluated on the same 128 original test leaves with transformation-specific variants.

The study also includes locked-test paired statistical comparisons using exact McNemar tests and Benjamini-Hochberg FDR correction. These analyses use predictions only and do not perform model selection on the test set.

## Targeted augmentation and controls

The final ablation work includes:

- Baseline training on the 595 original training leaves
- Single-transformation targeted training using Original + transformation (1,190 images)
- Duplicate-Original control (1,190 images)
- Gaussian-Blur control (1,190 images)
- Validation restricted to the original 127 validation leaves for checkpoint selection
- Locked test used only for final inference and paired statistical analysis

## RandAugment / AugMix

The repository documents a final PyTorch protocol comparing:

- Full_Data_9520
- RandAugment
- AugMix

with three seeds and locked testing on:

- Original
- Grayscale
- Rotate
- Brightness

A direct RandAugment-vs-AugMix analysis uses exact paired McNemar tests with Benjamini-Hochberg FDR correction.

## A3: unseen transformation severity

A3 trains targeted MobileNetV3-Small models for:

- Rotate × seeds 42, 1337, 2026
- Brightness × seeds 42, 1337, 2026

and evaluates baseline and targeted models on unseen conditions:

- Rotate: -45°, +45°
- Brightness: 0.60, 1.60

The final pipeline is:

**original image → unseen transformation → resize to 224×224 → ImageNet normalization**

Selected final Macro-F1 results:

| Strategy | Condition | Baseline Macro-F1 | Targeted Macro-F1 | Δ Macro-F1 |
|---|---|---:|---:|---:|
| Rotate | -45° | 0.5647 | 0.8949 | +0.3302 |
| Rotate | +45° | 0.6160 | 0.8909 | +0.2749 |
| Brightness | 0.60 | 0.8558 | 0.8956 | +0.0398 |
| Brightness | 1.60 | 0.6891 | 0.8895 | +0.2004 |

Paired significance was assessed using exact McNemar tests with Benjamini-Hochberg FDR correction.

## Repository structure

```
.
├── README.md
├── notebooks/
│   └── Brinjal_Final_Preprocessed.ipynb
├── results/
│   ├── baseline/
│   ├── robustness/
│   ├── augmentation_ablation/
│   ├── statistical_tests/
│   └── a3_unseen_severity/
├── figures/
├── src/
├── configs/
└── LICENSE
```

Large datasets, raw image folders, and heavyweight binary checkpoints are intentionally excluded from the public repository unless explicitly released.

## Reproducibility

1. Prepare the dataset according to the metadata structure used in the notebook.
2. Set the dataset root in the configuration/path cells.
3. Install the Python dependencies matching the recorded environment.
4. Run baseline training with seeds 42, 1337, and 2026.
5. Run locked-test robustness evaluation.
6. Run the augmentation ablations and controls.
7. Run the statistical analysis cells after all predictions are generated.
8. Run the A3 unseen-severity experiment last.

The notebook contains the exact experiment code and the saved output paths used during the study.

## Citation

This repository is intended to accompany the corresponding research article. Please cite the paper when using the dataset protocol, code, or reported experimental results.

Repository: https://github.com/payamhabibi/brinjal-augmentation-robustness

## Reproducibility note

The repository reports the results produced by the final notebook execution. No test-set predictions are used for checkpoint selection in the locked-test protocols.

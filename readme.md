# COMP 4330SEF — Facial Keypoint Detection (P06 Group 5)

A PyTorch project for predicting facial landmarks from grayscale face images. It compares a NaimishNet-style baseline against an experimental **BAM + CoordConv** variant and includes mechanistic interpretability analyses (feature-map activation statistics in Rust and weight-space SVD rank reduction).

---

## Scope

This repository evaluates the baseline and advanced models using test Normalized Mean Error (NME) and exploratory mechanistic interpretability analyses. Broader transfer-learning roadmaps and Raspberry Pi deployment demonstrations are out of scope for this evaluation.

## Repository Structure

```text
├── Main.ipynb                     # Model training, checkpointing, and prediction plots
├── MechInerp_Analysis.ipynb       # Feature-map extraction and weight-space SVD analysis
├── models/
│   └── naimishnet.py              # Baseline NaimishNet & BAM + CoordConv architectures
├── dataset/
│   └── data.py                    # Udacity dataset loader, resizing, normalization, and masking
├── src/
│   └── evalution.py               # Test-set NME evaluation script
├── rust_data_pipeline/            # Rust CLI tool for layer activation SD and t-statistics
├── docs/assets/                   # Historical feature-map visualizations
├── svd_rank_results.csv           # Recorded NME across retained SVD ranks (1–136)
└── requirements.txt               # Python dependencies
```

## Models

Both models accept a `1 × 96 × 96` grayscale image and output 180 values representing 90 normalized `(x, y)` landmark slots (see [`models/naimishnet.py`](models/naimishnet.py)):

* **Baseline (NaimishNet-style):** Four convolutional blocks followed by fully connected regression layers.
* **Advanced (BAM + CoordConv):** Appends normalized `(x, y)` coordinate channels to the input and applies Bottleneck Attention Module (BAM) channel and spatial gating to the backbone features.

## Dataset and Setup

The project uses the Udacity facial keypoints dataset. `create_loaders` in [`dataset/data.py`](dataset/data.py) downloads missing dataset files, resizes images to `96 × 96`, normalizes coordinates by source image width and height, pads landmarks to 90 slots with a binary validity mask, caches tensors in memory, and splits the training set using a fixed random seed.

Expected data directory layout relative to the project root:

```text
data/
├── training/
├── training_frames_keypoints.csv
├── test/
└── test_frames_keypoints.csv
```

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Jupyter separately if needed, then run [`Main.ipynb`](Main.ipynb) from the repository root.

## Training

Both models optimize masked mean-squared error (MSE) with early stopping on validation loss (`max_epochs=50`, `patience=5`), restoring the best validation checkpoint before saving:

1. **Baseline:** Trained from scratch with Adam (`lr=0.001`) and saved to `output/naimishnet_aligned.pth`.
2. **BAM + CoordConv:** Initialized with compatible convolutional weights from the trained baseline, trained with Adam (`lr=0.0005`), and saved to `output/naimishnet_bam_coord_aligned.pth`.

The training notebook automatically selects CUDA when available and plots side-by-side landmark predictions on a sample training image.

## Test Results

In the recorded run, [`src/evalution.py`](src/evalution.py) evaluated **770 test images** (**52,360 valid landmarks**, zero excluded images). NME is computed as the per-image mean Euclidean landmark distance in normalized square-image coordinates, divided by the unit-square diagonal ($\sqrt{2}$) and averaged across all test images (lower is better):

| Model | Test NME |
| :--- | ---: |
| Baseline | 2.65% |
| **BAM + CoordConv** | **2.28%** |

> **Methodological Note:** These figures reflect a single training run. Because the models differ in learning rate and baseline-backbone weight initialization, this comparison does not isolate architectural effects alone. Diagonal-normalized NME is also not directly comparable to interocular-normalized NME reported in other literature.

## Feature-Map Analysis (Python + Rust)

[`MechInerp_Analysis.ipynb`](MechInerp_Analysis.ipynb) extracts and visualizes convolutional feature maps for the same test image across both trained models. The first four captured convolution outputs come from the shared backbone prior to the advanced model's BAM gate. Because channel indices are not aligned across models and each grid cell is normalized independently, channel ordering and relative brightness are not directly comparable across models.

The notebook exports Layer 1 and Layer 4 activations to `.npy` files for summary statistics in Rust:

```bash
cd rust_data_pipeline
cargo run -- baseline_layer1.npy baseline_layer4.npy
cargo run -- advanced_layer1.npy advanced_layer4.npy
```

For the recorded test image, the Rust pipeline produced:

| Model | Layer 1 SD | Layer 4 SD | Welch's T-Statistic (Magnitude) |
| :--- | ---: | ---: | ---: |
| Baseline | 0.151789 | 0.016106 | 57.925 |
| BAM + CoordConv | 0.100795 | 0.020940 | 27.210 |

> **Interpretation Caveat:** These statistics summarize spatial/channel activations on a single test image rather than landmark regression accuracy. Because flattened feature-map activations are correlated rather than independent samples, the t-statistics do not establish formal statistical significance and do not by themselves prove feature collapse or greater representational expressiveness.

*Note: Generated `.npy` activation dumps and `.pth` model checkpoints are git-ignored and must be regenerated locally. Earlier feature-map plots for the [baseline](docs/assets/feature-maps-baseline.png) and [advanced model](docs/assets/feature-maps-bam-coordconv.png) are preserved as historical references.*

## Weight-Space SVD Analysis

[`MechInerp_Analysis.ipynb`](MechInerp_Analysis.ipynb) evaluates how low-rank approximation of the final regression layer impacts test NME. Singular Value Decomposition (SVD) is applied to the first 136 output rows (corresponding to the 68 valid test landmarks), evaluating retained ranks from `1` to `136` without retraining or modifying saved checkpoints. Full results are recorded in [`svd_rank_results.csv`](svd_rank_results.csv):

| Model | Full Rank (136) NME | Rank 68 NME | Change |
| :--- | ---: | ---: | ---: |
| Baseline | 2.65% | 3.71% | +1.06% |
| **BAM + CoordConv** | **2.28%** | **2.43%** | **+0.15%** |

In this run, truncating the final regression layer to rank 68 increased test error for both models, though **BAM + CoordConv** degraded substantially less (`+0.15%` vs. `+1.06%`). This serves as an exploratory weight-space intervention rather than a deployment compression benchmark; any rank selection for production compression should be tuned on validation data rather than the test set.

The notebook also overlays the primary SVD output direction onto a sample test face; vector arrows are scaled purely for visual clarity and do not represent actual prediction errors or landmark displacements.
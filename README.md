# COMP 4330SEF - Facial Keypoint Detection (P06 Group 5)

A PyTorch project for predicting facial landmarks from grayscale face images. It compares a NaimishNet-style baseline with an experimental BAM + CoordConv variant and includes feature-map analysis.

## Scope

The project compares the baseline and advanced models using test NME and exploratory feature-map analysis. The broader transfer-learning roadmap and Raspberry Pi demonstration are not evaluated.

## Models

Both models accept a `1 x 96 x 96` grayscale image and output 180 values representing 90 normalized `(x, y)` landmark slots. The baseline uses four convolutional blocks and fully connected regression layers. The advanced variant adds coordinate channels and applies channel and spatial attention to the backbone features. See [`models/naimishnet.py`](models/naimishnet.py).

## Dataset and setup

The project uses the Udacity facial keypoints dataset. The loader resizes images to `96 x 96`, normalizes coordinates by the source image width and height, and pads landmarks to 90 slots. A mask marks the valid landmarks.

The expected paths, relative to the project root, are:

```text
data/
├── training/
├── training_frames_keypoints.csv
├── test/
└── test_frames_keypoints.csv
```

`create_loaders` in [`dataset/data.py`](dataset/data.py) downloads missing dataset items, caches the training tensors in memory, and splits the training data into training and validation sets with a fixed seed. The separate test dataset is used for evaluation.

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Jupyter separately if needed, then run [`Main.ipynb`](Main.ipynb) from the project root.

## Training

Both models use masked mean-squared error and early stopping based on validation loss, with a maximum of 50 epochs and patience of 5. The best validation weights are restored before saving.

1. The baseline trains with Adam at `lr=0.001`.
2. The advanced model initializes compatible convolution weights from the trained baseline and trains with Adam at `lr=0.0005`.

The notebook selects CUDA when available and saves the models to `output/naimishnet_aligned.pth` and `output/naimishnet_bam_coord_aligned.pth`. It plots both models' predictions for the same training image.

## Test results

In the recorded run, [`src/evalution.py`](src/evalution.py) evaluated 770 test images with 52,360 valid landmarks and no excluded images. NME is the per-image mean landmark distance in normalized square-image coordinates, divided by the unit-square diagonal and averaged across images. Lower is better.

| Model | Test NME |
| --- | ---: |
| Baseline | 2.65% |
| BAM + CoordConv | 2.28% |

These are results from one run. Different learning rates and baseline-backbone transfer mean the comparison does not isolate the effect of architecture alone. Diagonal-normalized NME is not directly comparable with metrics using another normalization, such as interocular distance.

## Feature-map analysis

[`MechInerp_Analysis.ipynb`](MechInerp_Analysis.ipynb) extracts and visualizes convolutional feature maps for the same test image in both trained models. Its first four captured convolution outputs are from the backbone, before the advanced model's BAM gate. Channel positions do not necessarily represent matching features between models. The grid normalizes channels individually, so brightness is not directly comparable between channels or models.

The notebook exports layers 1 and 4 to NumPy files. For the recorded example, Rust produced:

| Model | Layer 1 SD | Layer 4 SD | Layer 4 minus layer 1 T-statistic |
| --- | ---: | ---: | ---: |
| Baseline | 0.151789 | 0.016106 | 57.925 |
| BAM + CoordConv | 0.100795 | 0.020940 | 27.210 |

These summaries describe activations from one image, not landmark accuracy. Flattened activations are dependent, so the T-statistics alone do not establish statistical significance. The figures do not prove feature collapse or greater expressiveness.

Earlier visualizations are available for the [baseline](docs/assets/feature-maps-baseline.png) and [advanced model](docs/assets/feature-maps-bam-coordconv.png). They are historical examples, not evidence from the current matched-input analysis.

After running the extraction notebook, run the Rust analysis from its directory:

```bash
cd rust_data_pipeline
cargo run -- baseline_layer1.npy baseline_layer4.npy
cargo run -- advanced_layer1.npy advanced_layer4.npy
```

The generated NumPy files are ignored by Git and must be regenerated locally. Reproducing the recorded results also requires the corresponding trained checkpoint files.

## SVD analysis

[`MechInerp_Analysis.ipynb`](MechInerp_Analysis.ipynb) explores how
approximating the final regression-layer weights affects test NME.

The analysis applies singular value decomposition (SVD) to the first
136 output rows, corresponding to the 68 valid test landmarks. It
evaluates retained ranks from 1 to 136 without retraining or changing
the saved checkpoints. Results are saved in
[`svd_rank_results.csv`](svd_rank_results.csv).

| Model | Full rank (136) NME | Rank 68 NME |
| --- | ---: | ---: |
| Baseline | 2.65% | 3.71% |
| BAM + CoordConv | 2.28% | 2.43% |

In this run, reducing the rank increased test error for both models.
This is an exploratory weight intervention, not a demonstration of
deployed model compression. Rank selection for deployment should use
validation data rather than the test set.

The notebook also plots the first SVD output direction over an image.
The arrows are scaled for visualization and do not show actual
prediction errors or landmark movements.
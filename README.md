# Sea Surface Temperature Reconstruction under Cloud Occlusion

Deep learning model (U-Net, TensorFlow/Keras) that reconstructs **Sea Surface Temperature (SST)** in areas hidden by clouds in satellite images of the **North Adriatic Sea**.

Satellites measure SST through the infrared radiation emitted by the sea surface, but clouds absorb it and leave large gaps in the observations. The goal of this project is to fill those gaps, improving on a statistical (climatological) baseline.

All the code is in a single notebook: [`clouds-occlusion-adriatics.ipynb`](clouds-occlusion-adriatics.ipynb).

---

## Results

The model is evaluated on the **test set** (2021-07-04 → 2023-12-31) by calling the test generator 50 times (50 batches × 32 images = 1,600 samples). The metric is the **RMSE in °C**, computed only on the pixels that were visible in the real image but artificially hidden by the generator (the only pixels with a known ground truth).

| Method | RMSE (°C) | RMSE std (°C) |
|---|---|---|
| Statistical baseline (tuned daily climatology) | 0.629 | 0.188 |
| **U-Net (this work)** | **0.468** | **0.141** |

The U-Net reduces the reconstruction error by about **25%** compared to the baseline, and the error is also more stable across samples (lower standard deviation).

### Training behaviour

- Training ran for 90 epochs; at the end the masked RMSE loss was **0.110** on training and **0.109** on validation (normalized units), with a validation MAE of **0.069**.
- Training and validation curves follow the same trajectory throughout, with **no sign of overfitting**, and the loss was still slowly decreasing at the last epoch.

### Qualitative observations

The predicted maps are generally accurate, but the model tends to **smooth the sharp boundaries between water masses at different temperatures** (thermal fronts), which in the ground truth appear more distinct. This is consistent with the loss used: a pixel-wise RMSE rewards predictions close to the average and does not explicitly penalise the loss of fine details.

---

## Data

The data are **nightly MODIS observations from the Aqua satellite** over a 128 × 128 pixel region of the North Adriatic. Missing values (land or clouds) are encoded as `NaN`.

| File | Shape | Period |
|---|---|---|
| `x_train_night.npy` | (5832, 128, 128) | 2002-07-04 → 2018-07-04 |
| `x_val_night.npy` | (1090, 128, 128) | 2018-07-04 → 2021-07-04 |
| `x_test_night.npy` | (895, 128, 128) | 2021-07-04 → 2023-12-31 |
| `dates_*_night.npy` | one date per image | — |
| `land_sea_mask.npy` | (128, 128) | 1 = sea, 0 = land (sea ≈ 40.8% of the area) |
| `stat_baseline.npy` | (366, 128, 128) | statistical baseline, one map per day of the year |

Training set temperatures range from 0.09 °C to 30.65 °C (mean 19.48 °C, std 5.09 °C).

The files are downloaded directly from Google Drive inside the notebook (via `gdown`) and are **not included in this repository**.

---

## Methodology

### 1. Normalization

Data are standardized (z-score) using the mean and standard deviation of the **training set only**; the same parameters are applied to validation, test and the statistical baseline (with land pixels set to 0). Gaussian normalization gave the best results among the options tested.

### 2. Artificial ground truth (generator)

Under real clouds there is no ground truth. To create one, the generator:

1. picks an image with at least **40% of visible sea**;
2. superimposes the clouds of **another random day**, so that the remaining visible sea is between 5% and 40% (and at least 10 percentage points less than the original);
3. the pixels that were visible but are now hidden become the region on which the model is trained and evaluated.

The statistical baseline is also **tuned** to the current day: the climatological map of that day of the year is shifted so that its mean matches the mean temperature of the visible area.

Each sample has:

- **Input (4 channels):** artificially clouded image · artificial mask · land-sea mask · tuned baseline
- **Target (3 channels):** real image · real mask · difference mask (pixels to reconstruct)

### 3. Model

A **U-Net** with about **7.86M parameters** (7,851,107 trainable):

- **Encoder:** 4 blocks with 32 → 64 → 128 → 256 filters (two 3×3 convolutions + max pooling each); the two deeper blocks use **dilated convolutions** (rate 2) and Batch Normalization.
- **Bottleneck:** 512 filters with dilation rate 3, Batch Normalization and Dropout 0.4, to capture wide spatial context.
- **Decoder:** 4 symmetric blocks with upsampling and skip connections; the two deeper ones use dilation, Batch Normalization and Dropout 0.2.
- **Output:** 1×1 convolution with linear activation.
- ReLU activations, **He normal** initialization and **L2 regularization** (1e-4) on the convolutional layers.

Other activation functions (Swish, Leaky ReLU, SiLU) were tested, but ReLU performed best. Dropout, L2 and Batch Normalization in the bottleneck and deeper layers were essential to avoid overfitting as the network grew.

### 4. Loss and metrics

- **Loss: masked RMSE**, computed only on the pixels of the difference mask. Switching from a standard loss to this masked version significantly improved performance, because the model focuses only on the areas that actually need to be reconstructed.
- **Metric:** masked MAE on the same pixels.

### 5. Training

| Parameter | Value |
|---|---|
| Optimizer | Adam, initial learning rate 1e-3 |
| Batch size | 32 |
| Epochs | 90 (150 steps per epoch) |
| Learning rate schedule | `ReduceLROnPlateau` (factor 0.3, patience 5) — final LR ≈ 8.1e-6 |
| Early stopping | patience 10 on `val_loss` (not triggered) |
| Other | Exponential Moving Average callback (decay 0.9) |
| Seed | 38 (TensorFlow and NumPy) |

Training takes about 76 seconds per epoch on a GPU (≈ 2 hours in total).

---

## How to use the notebook

### Requirements

- Python 3.10+
- TensorFlow 2.16+ / Keras 3
- NumPy, pandas, Matplotlib
- `gdown` (to download the data)
- A **GPU** is strongly recommended for training

The notebook was developed on **Google Colab / Kaggle** with an NVIDIA T4 GPU.

```bash
pip install tensorflow numpy pandas matplotlib gdown
```

### Steps

1. **Open the notebook** in Google Colab (recommended) or in a local Jupyter environment with a GPU.
2. **Download the data:** the `!gdown ...` cells download the datasets, the dates, the land-sea mask and the statistical baseline.
3. **Choose where to store the data:** the notebook copies the files to `/content/drive/MyDrive/DeepProject/` on Google Drive.
   - If you are not using Colab, skip the Drive cell and set `destination_folder` to the folder containing the `.npy` files (e.g. `destination_folder = './'`).
4. **Run the cells in order:** data inspection → normalization → generator → model → loss and metrics → training → evaluation → visualization.
5. **Training** saves the following files in the working directory:
   - `model_weights.weights.h5` — model weights
   - `model.keras` — full model
   - `training_history_m.pkl` and `training_history_m.csv` — loss and metric history
6. **Evaluation:** the last cells compute the RMSE on the test set for both the baseline and the model, and plot a prediction next to its ground truth.

> **Note:** the test generator must not be modified, otherwise results are not comparable with the baseline. Since the generator samples images and clouds randomly, the RMSE can vary slightly between runs.

### Reloading a trained model

The masked RMSE loss is a custom function, so it must be passed when loading the model:

```python
model = keras.models.load_model(
    'model.keras',
    custom_objects={'masked_rmse_loss': masked_rmse_loss}
)
```

---

## ⚠️ Files not included

The **trained weights** (`model_weights.weights.h5`, `model.keras`) and the **training history** (`training_history_m.pkl`, `training_history_m.csv`) are **not shared** in this repository.

To reproduce the results you need to run the training in the notebook. The cell that plots the loss curves reads `training_history_m.pkl`, so it only works after training has been run.

---

## Repository structure

```
.
├── clouds-occlusion-adriatics.ipynb   # notebook: data, model, training, evaluation
└── README.md
```

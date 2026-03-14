# CT Kernel Conversion

This project is a deep learning-based approach to convert between different CT (Computed Tomography) reconstruction kernels. The model learns to transform images from a smooth kernel to a sharp kernel, and vice-versa, by estimating the Modulation Transfer Function (MTF) of the kernels in the Fourier domain.

## Architecture

The model pipeline consists of the following steps:

1.  **Input:** A CT image slice is taken as input, and its Power Spectral Density (PSD) is computed.
2.  **Network:** A U-Net based model called `KernelEstimator` processes the PSD.
3.  **Output:** The model outputs B-spline knots and control points, which represent the MTF curve of the kernel.
4.  **Conversion:** The predicted MTF curves are used to create Optical Transfer Function (OTF) filters. These filters are then applied to the Fourier transform of the input image to perform the kernel conversion.

### Key Components

-   **`Code/SplineEstimator.py`**: Contains the neural network architecture (`KernelEstimator`).
-   **`Code/utils.py`**: Includes core signal processing functions for PSD calculation, FFT, spline generation, and image conversion.
-   **`Code/PSDDataset.py`** and **`Code/Dataset.py`**: Data loaders for training the model.
-   **`train.py`**: The main training script.

## How to Run

### 1. Setup Environment

Install the required dependencies using `uv`:

```bash
uv sync
```

### 2. Prepare Data

This project expects the data to be in a specific format. You will need to modify the hardcoded paths in the `main()` function of `train.py` to point to your data:

-   `IMAGE_ROOT`: Path to the directory containing the CT image volumes in NIfTI format (`.nii.gz`).
-   `MTF_FOLDER`: Path to the directory containing the ground truth MTF data in MATLAB `.mat` format.
-   `PSD_FOLDER`: Path to the directory containing the precomputed PSD data in NumPy `.npy` format.

### 3. Run Training

To start the training, run the following command:

```bash
uv run train.py
```

The training script will save checkpoints and visualization images in the `training_output_{alpha}` directory.

## Project Structure

```
.
├── Code/
│   ├── SplineEstimator.py
│   ├── utils.py
│   ├── PSDDataset.py
│   ├── Dataset.py
│   └── ...
├── train.py
├── pyproject.toml
└── README.md
```

## License

MIT

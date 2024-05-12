# Project Structure Overview

The project repository is organized for clarity and ease of use, ensuring that contributors can navigate and utilize the structure efficiently. Below is a detailed breakdown of the repository's organization.

## Directory and File Structure

```
project/
│
├── data/
│   ├── external/                   # Data from third-party sources
│   ├── processed/                  # Cleaned and preprocessed data ready for analysis
│   │   ├── EEG_Spectrograms/       # Contains EEG spectrogram data
│   │   │   ├── eeg_specs.npy       # NumPy array of EEG spectrograms
│   │   │   └── specs.npy           # NumPy array of spectrograms
│   │   └── train.pqt               # Processed training data in Parquet format
│   └── raw/                        # Original, unprocessed datasets
│
├── models/
│   ├── MLP_Model/                  # Multi-Layer Perceptron model files
│   └── TF EfficientNet ImageNet Weights/ # Pre-trained EfficientNet weights
│
├── notebooks/
│   ├── EDA.ipynb                   # Notebook for exploratory data analysis
│   ├── EEG_TO_SPEC.ipynb           # Notebook for EEG to spectrogram conversion
│   └── StarterNotebook.ipynb       # Introductory notebook with main code that preprocess, trains and evaluates our model.
│
├── src/
│   ├── buildmodel.py               # Script to build models
│   ├── dataloader.py               # Script for loading and handling data
│   ├── kaggle_kl_div.py            # Script for KL divergence related functions
│   ├── kaggle_metric_utilities.py  # Utility functions for Kaggle metrics
│   ├── preprocessing.py            # Data preprocessing scripts
│   ├── scheduler.py                # Learning rate scheduler
│   ├── train_model.py              # Script for model training
│   └── visualization.py            # Visualization utilities
│
├── requirements.txt                # Dependencies for the project
│
└── README.md                       # Documentation and overview of the project
```

## Data Acquisition for Processed Directory

To populate the `processed` directory with the necessary datasets, download the following Kaggle datasets:

- EEG Spectrograms: [Download from Kaggle](https://www.kaggle.com/datasets/cdeotte/brain-spectrograms)
- Brain EEG Spectrograms: [Download from Kaggle](https://www.kaggle.com/datasets/cdeotte/brain-eeg-spectrograms)

The contents of these downloads should be placed within the `data/processed` directory.

## Source Code Inclusion from Kaggle

The `src` directory includes scripts from the following Kaggle datasets:

- Kaggle KL Divergence Code: [Include in `src`](https://www.kaggle.com/datasets/cdeotte/kaggle-kl-div)
  - Download and ensure `kaggle_kl_div.py` is placed within the `src` directory.

## Model Weights Inclusion from Kaggle

The model weights for EfficientNet can be obtained from Kaggle and must be included in the `models` directory:

- TF EfficientNet ImageNet Weights: [Include in `models`](https://www.kaggle.com/datasets/cdeotte/tf-efficientnet-imagenet-weights)
  - After downloading, place the weights within the `models/TF EfficientNet ImageNet Weights` directory.

## Getting Started

To begin working with the project, follow these steps:

1. Clone the repository to your local machine.

2. Create a Miniconda environment using Python 3.11:

   ```bash
   conda create --name hbaEnv python=3.11
   ```

3. Activate the newly created environment:

   ```bash
   conda activate hbaEnv
   ```

4. **Install 'pip-tools':

   ```bash
   pip install pip-tools
   ```

5. Use requiremnts.in file for installing dependecies

    ```bash
    pip-compile requrements.in
    ```

6. Install requirements using pip

      ```bash
      pip install -r requirements.txt
      ```

7. Pull the dataset using DVC:

   Assuming you have a `.dvc` file set up in your repository for your datasets, run:

   ```bash
   dvc pull
   ```

   This will download the datasets from the DVC remote storage to your local machine. Ensure you have access to the remote storage, and your DVC remote is configured correctly.

8. Download the datasets manually from the provided Kaggle links and place them in their respective directories if needed.

9. Start with the `StarterNotebook.ipynb` to get a walkthrough of the project setup and initial analysis.

Ensure you have the necessary permissions and access to the Kaggle datasets and any required API keys for DVC remotes, especially for Google Drive integration.

> This guide assumes familiarity with basic terminal operations, Conda, and Python environments. The instructions include environment setup, package installation, and the initial steps to begin project analysis with a Jupyter notebook.
>
> For more detailed instructions on each step, refer to the `StarterNotebook.ipynb`.

# src/data_loader.py
from pathlib import Path

import albumentations as albu
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


def load_train_spectrograms(path, read_spec_files):
    path = Path(path)
    files = list(path.glob("*.parquet"))
    print(f"There are {len(files)} spectrogram parquets")
    spectrograms = {}
    if read_spec_files:
        for i, file in enumerate(files):
            if i % 100 == 0:
                print(f"{i}, ", end="")
            tmp = pd.read_parquet(file)
            name = int(file.stem)
            spectrograms[name] = tmp.iloc[:, 1:].values
    else:
        spectrograms = np.load(
            path.parent / "processed/specs.npy", allow_pickle=True
        ).item()
    return spectrograms


def load_eeg_spectrograms(train, read_eeg_spec_files):
    base_path = Path("../data/processed/EEG_Spectrograms")
    all_eegs = {}
    if read_eeg_spec_files:
        for i, e in enumerate(train.eeg_id.values):
            if i % 100 == 0:
                print(f"{i}, ", end="")
            file_path = base_path / f"{e}.npy"
            all_eegs[e] = np.load(file_path)
    else:
        all_eegs = np.load(base_path.parent / "eeg_specs.npy", allow_pickle=True).item()
    return all_eegs


class DataGenerator(Sequence):
    def __init__(
        self,
        data,
        batch_size=32,
        shuffle=False,
        augment=False,
        mode="train",
        specs=None,
        eeg_specs=None,
        TARGETS=None,
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.TARGETS = TARGETS
        self.on_epoch_end()

    def __len__(self):
        """Calculates the number of batches per epoch."""
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self._data_generation(indexes)
        if self.augment:
            X = self._augment_batch(X)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        # Initialize X and y
        X = np.zeros((len(indexes), 128, 256, 8), dtype="float32")
        y = np.zeros((len(indexes), len(self.TARGETS)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            r = 0 if self.mode == "test" else int((row["min"] + row["max"]) // 4)

            for k in range(4):
                img = self.specs[row.spec_id][r : r + 300, k * 100 : (k + 1) * 100].T
                img = self._process_image(img)
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

            # EEG spectrograms
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img

            if self.mode != "test":
                y[j,] = row[self.TARGETS]

        return X, y

    def _process_image(self, img):
        """Process and standardize a single image."""
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        return np.nan_to_num(img, nan=0.0)

    def _random_transform(self, img):
        """Applies random transformations for data augmentation."""
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                # Additional augmentations can be added here
            ]
        )
        return composition(image=img)["image"]

    def _augment_batch(self, img_batch):
        """Applies augmentation to a batch of images."""
        return np.array([self._random_transform(img) for img in img_batch])

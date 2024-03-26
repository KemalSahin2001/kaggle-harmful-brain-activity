# src/data_loader.py
import pandas as pd
import numpy as np
import os
import albumentations as albu
import tensorflow as tf
        

def load_train_spectrograms(path, read_spec_files):
    """
    Loads train spectrograms from the specified path.
    
    Parameters:
    - path: Directory path where spectrogram files are located.
    - read_spec_files: Boolean indicating whether to read the spectrogram files.
    
    Returns:
    - A dictionary of spectrograms if read_spec_files is True, else a preloaded numpy array.
    """
    files = os.listdir(path)
    print(f'There are {len(files)} spectrogram parquets')

    if read_spec_files:    
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ', ', end='')
            tmp = pd.read_parquet(os.path.join(path, f))
            name = int(f.split('.')[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
    else:
        spectrograms = np.load('..\data\processed\specs.npy',allow_pickle=True).item()
    return spectrograms

def load_eeg_spectrograms(train, READ_EEG_SPEC_FILES):
    """
    Loads EEG spectrograms based on the training data provided.
    
    Parameters:
    - train: DataFrame containing training data, specifically 'eeg_id' values.
    - READ_EEG_SPEC_FILES: Boolean indicating whether to read EEG spectrogram files.
    
    Returns:
    - A dictionary of EEG spectrograms if read_eeg_spec_files is True, else a preloaded numpy array.
    """
    if READ_EEG_SPEC_FILES:
        all_eegs = {}
        for i,e in enumerate(train.eeg_id.values):
            if i%100==0: print(i,', ',end='')
            x = np.load(f'..\data\processed\EEG_Spectrograms\{e}.npy')
            all_eegs[e] = x
    else:
        all_eegs = np.load('..\data\processed\eeg_specs.npy',allow_pickle=True).item()

    return all_eegs

class DataGenerator(tf.keras.utils.Sequence):
    """
    DataGenerator inherits from tf.keras.utils.Sequence and is meant to generate batches of data for training or evaluation.
    
    Attributes:
        data (DataFrame): The input data frame containing samples to generate data for.
        batch_size (int): The size of the batch to generate.
        shuffle (bool): Whether to shuffle the data at the end of each epoch.
        augment (bool): Whether to apply data augmentation.
        mode (str): Mode in which the generator operates ('train' or 'test').
        specs (dict): A dictionary containing spectrogram data.
        eeg_specs (dict): A dictionary containing EEG spectrogram data.
        TARGETS (list): A list of target column names.
    """
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train',
                 specs=None, eeg_specs=None, TARGETS=None):
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
        X = np.zeros((len(indexes), 128, 256, 8), dtype='float32')
        y = np.zeros((len(indexes), len(self.TARGETS)), dtype='float32')

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            r = 0 if self.mode == 'test' else int((row['min'] + row['max']) // 4)

            for k in range(4):
                img = self.specs[row.spec_id][r:r + 300, k * 100:(k + 1) * 100].T
                img = self._process_image(img)
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

            # EEG spectrograms
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img

            if self.mode != 'test':
                y[j, ] = row[self.TARGETS]

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
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            # Additional augmentations can be added here
        ])
        return composition(image=img)['image']

    def _augment_batch(self, img_batch):
        """Applies augmentation to a batch of images."""
        return np.array([self._random_transform(img) for img in img_batch])

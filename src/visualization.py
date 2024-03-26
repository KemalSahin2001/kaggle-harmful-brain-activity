import matplotlib.pyplot as plt

def plot_batch(x_batch, y_batch, row_count, col_count, batch_index, eeg_ids, targets):
    """
    Plots a grid of spectrogram images from a batch.

    Parameters:
    - x_batch: Batch of input images.
    - y_batch: Batch of target labels.
    - row_count: Number of rows in the plot grid.
    - col_count: Number of columns in the plot grid.
    - batch_index: Index of the current batch.
    - eeg_ids: Array of EEG ids corresponding to the batch.
    - targets: Array of target names.
    """
    plt.figure(figsize=(20, 8))
    for row in range(row_count):
        for col in range(col_count):
            idx = row * col_count + col
            plt.subplot(row_count, col_count, idx + 1)
            img = x_batch[idx, :, :, 0][::-1, ]
            mn, mx = img.min(), img.max()
            img_normalized = (img - mn) / (mx - mn + 1e-5)  # To avoid division by zero
            plt.imshow(img_normalized)
            target_values = ', '.join([f'{value:0.2f}' for value in y_batch[idx]])
            eeg_id = eeg_ids[batch_index * row_count * col_count + idx]
            plt.title(f'EEG = {eeg_id}\nTarget = [{target_values}]', size=12)
            plt.yticks([])
            plt.xlabel('Time (sec)', size=16)
            plt.ylabel('Frequencies (Hz)', size=14)
    plt.tight_layout()
    plt.show()
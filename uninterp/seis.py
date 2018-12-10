import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import label, regionprops


def _contrast(data, norm=True):
    """
    Calculates image contrast as interquartile difference by substracting lower quartile (Q1)
    from upper quartile (Q3). By default (norm=True) it normalizes the data between 0 and 255.

    Used in Alcalde et al. (2017).
    """
    if norm:
        data = data.flatten()
        data = data / np.abs(data).max() * 255 * 0.5  # squish between -1,+1, stretch to 255/2
        data -= np.min(data)  # shift to positive (everything between 0,255)
    iq = np.percentile(data, 75) - np.percentile(data, 25)  # interquartile difference
    return iq


def image_contrast(data, window_shape=30, step=5):
    """

    Args:
        data:
        window_shape:
        step:

    Returns:

    """
    wd = view_as_windows(data, window_shape, step=step)
    img_contrast = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            img_contrast[xw, yw] = _contrast(wd[xw, yw])

    return img_contrast


def binary_threshold(data, threshold=65):
    """

    Args:
        data (np.ndarray): Slice or Cube of seismic
        threshold (int): Threshold at which to split the dataset into binary.

    Returns:
        np.ndarray
    """
    data_binary = np.zeros_like(data)
    data_norm = data / data.max()
    data_binary[data >= np.percentile(data, threshold)] = 1
    return data_binary


def label_reflectors(data_binary, normalized=False):
    """

    Args:
        data_binary:
        normalized:

    Returns:

    """
    labels = label(data_binary)

    length = np.zeros_like(data_binary)
    for rp in regionprops(labels):
        length[labels == rp.label] = rp.major_axis_length
    if normalized:
        length /= length.max()
    return length


def image_reflector_length(data_labeled, window_shape=30, step=1):
    """

    Args:
        data_labeled:
        window_shape:
        step:

    Returns:

    """
    wd = view_as_windows(data_labeled, window_shape, step=step)
    length_mean = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            length_mean[xw, yw] = np.mean(wd[xw, yw])

    return length_mean


def signal_to_noise_ratio(data, window_shape=30, step=5):
    """Simple singal-to-noise calculation (mean / stdev) using a convolving window of given size.

    Args:
        data (np.ndarray):
        window_shape (int):
        step (int):

    Returns:

    """
    wd = view_as_windows(np.abs(data), window_shape, step=step)
    snr = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            snr[xw, yw] = np.mean(wd[xw, yw]) / np.std(wd[xw, yw])
    return snr
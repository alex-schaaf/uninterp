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
    wd = view_as_windows(data, window_shape, step=step)
    img_contrast = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            img_contrast[xw, yw] = _contrast(wd[xw, yw])

    return img_contrast


def _binary_threshold(data, threshold=50):
    data_binary = np.zeros_like(data)
    data_norm = data / data.max()
    data_binary[data >= np.percentile(data, threshold)] = 1
    return data_binary


def image_reflector_length(data_binary, window_shape=30, step=1, normalized=True):
    labels = label(data_binary)

    length = np.zeros_like(data_binary)
    for rp in regionprops(labels):
        length[labels == rp.label] = rp.major_axis_length
    if normalized:
        length /= length.max()

    wd = view_as_windows(length, window_shape, step=step)
    length_mean = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            length_mean[xw, yw] = np.mean(wd[xw, yw])

    return length_mean

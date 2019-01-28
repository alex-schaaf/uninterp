import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import label, regionprops


def _contrast(data:np.ndarray, norm:bool=True):
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


def image_contrast(data:np.ndarray, window_shp:int=30, step:int=5):
    """Calculates image contrast averaged across given window shape size at
    each given step.

    Args:
        data:
        window_shp:
        step:

    Returns:

    """
    wd = view_as_windows(data, window_shp, step=step)
    img_contrast = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            img_contrast[xw, yw] = _contrast(wd[xw, yw])

    return img_contrast


def binary_threshold(data:np.ndarray, threshold:int=65):
    """

    Args:
        data (np.ndarray): Slice or Cube of seismic
        threshold (int): Threshold at which to split the dataset into binary.

    Returns:
        np.ndarray
    """
    data_binary = np.zeros_like(data)
    data_binary[data >= np.percentile(data, threshold)] = 1
    return data_binary


def label_reflector_length(data_binary:np.ndarray, norm:bool=False):
    """Label unique areas in image (thresholded seismic cube) and assign major
     axis length as value.

    Args:
        data_binary: Thresholded seismic cube or image.
        norm: (Default: False)

    Returns:
        (np.ndarray): Same as input, filled with max length of blobs.
    """
    labels = label(data_binary)

    length = np.zeros_like(data_binary)
    for rp in regionprops(labels):
        length[labels == rp.label] = rp.major_axis_length
    if norm:
        length /= length.max()
    return length


def mean_reflector_length(data_labeled:np.ndarray,
                          window_shp:int=30,
                          step:int=1):
    """

    Args:
        data_labeled:
        window_shp:
        step:

    Returns:

    """
    wd = view_as_windows(data_labeled, window_shp, step=step)
    length_mean = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            length_mean[xw, yw] = np.mean(wd[xw, yw])

    return length_mean


def signal_to_noise_ratio(data:np.ndarray, window_shp:int=30, step:int=5):
    """Simple singal-to-noise calculation (mean / stdev) using a convolving
    window of given size.

    Args:
        data (np.ndarray):
        window_shp (int):
        step (int):

    Returns:

    """
    wd = view_as_windows(np.abs(data), window_shp, step=step)
    snr = np.zeros(wd.shape[:2])
    for xw in range(wd.shape[0]):
        for yw in range(wd.shape[1]):
            snr[xw, yw] = np.mean(wd[xw, yw]) / np.std(wd[xw, yw])
    return snr


def get_xyz_coords(cube:np.ndarray, extent:tuple):
    """Calculate index coordinates for given seismic cube (np.ndarray [y,x,z])
     and extent (x,X,y,Y,z,Z). Returns tuple (size 3) of arrays containing
     coordinates."""
    x_coords = np.linspace(extent[0], extent[1], cube.shape[1])
    y_coords = np.linspace(extent[2], extent[3], cube.shape[0])
    z_coords = np.linspace(extent[4], extent[5], cube.shape[2])
    return x_coords, y_coords, z_coords
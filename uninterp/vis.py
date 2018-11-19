import matplotlib.pyplot as plt


def plot_seis_section(cube, i, direction="inline", extent=None, aspect=None):
    """Plots seismic section at position i of given cube. Designed especially for use with ipywidgets.

    Args:
        cube (numpy.ndarray): Seismic cube as a 3-D numpy ndarray [inline, crossline, timeslice]
        i (int): Slice position
        direction (str): Which slice direction to plot: ["inline", "crossline", "timeslice"]
        extent (list): The extent of the data in format [x,X,y,Y,z,Z]
        aspect (int): Aspect ratio of the plot (used for vertical exaggeration)

    Returns:
        matplotlib.axes
    """
    if extent is None:
        extent = (None, None, None, None, None, None)

    fig, ax = plt.subplots(figsize=(15,10))
    if direction == "inline":
        ax.imshow(cube[i,:,:].T, cmap="seismic", interpolation="bicubic", extent=extent[:2] + extent[4:], aspect=aspect)  # extent=EXTENT[:2] + EXTENT[4:],
    elif direction == "xline":
        ax.imshow(cube[:,i,:].T, cmap="seismic", interpolation="bicubic", extent=extent[2:4] + extent[4:], aspect=aspect)
    elif direction == "timeslice":
        ax.imshow(cube[:,:,i].T, cmap="seismic", interpolation="bicubic", extent=extent[:4], aspect=aspect)
    return ax
import matplotlib.pyplot as plt


def plot_seis_section(cube, i, direction="inline", extent=None, aspect=None,
                      plot_kwargs={}, imshow_kwargs={}):
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
        extent = (0, cube.shape[0], 0, cube.shape[1], 0, cube.shape[2])

    pkwargs = {"figsize": (15, 10)}
    pkwargs.update(plot_kwargs)
    imkwargs = {"interpolation": "bicubic", "cmap": "seismic"}
    imkwargs.update(imshow_kwargs)

    fig, ax = plt.subplots(**pkwargs)
    if direction == "inline":
        ax.imshow(cube[i,:,:].T,
                  extent=extent[:2] + extent[4:], aspect=aspect, **imkwargs)
    elif direction == "xline":
        ax.imshow(cube[:,i,:].T,
                  extent=extent[2:4] + extent[4:], aspect=aspect, **imkwargs)
    elif direction == "timeslice":
        ax.imshow(cube[:,:,i].T,
                  extent=extent[:4], aspect=aspect, **imkwargs)
    return ax
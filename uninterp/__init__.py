import numpy as np

def histogram_2d(df, min_, max_, direction="Z", bins=40, range_=None):
    """

    Arguments:
        df (obj, pandas.DataFrame): Pandas DataFrame with datapoints.
        min_ (int): Lower extent of slice.
        max_ (int): Upper extent of slice.
        direction (str, optional): Default "Z", other: "X", "Y".
        bins (int, optional): Number of bins per dimension.
    Returns:
        (H, xedges, yedges)
    """
    f = (df[direction] > min_) & (df[direction] < max_)
    if direction == "Z":
        if range_ is None:
            range_ = [[df.X.min(), df.X.max()], [df.Y.min(), df.Y.max()]]
        return np.histogram2d(df[f].X, df[f].Y, bins=bins, range=range_)
    elif direction == "Y":
        if range_ is None:
            range_ = [[df.X.min(), df.X.max()], [df.Z.min(), df.Z.max()]]
        return np.histogram2d(df[f].X, df[f].Z, bins=bins, range=range_)
    elif direction == "X":
        if range_ is None:
            range_ = [[df.Y.min(), df.Y.max()], [df.Z.min(), df.Z.max()]]
        return np.histogram2d(df[f].Y, df[f].Z, bins=bins, range=range_)
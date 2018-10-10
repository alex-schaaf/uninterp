import numpy as np
from scipy.spatial import Delaunay
import sys
sys.path.append("../../gempy")
from gempy.utils.petrel_integration import plane_fit
import mplstereonet
import pandas as pd


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


def concave_trisurf(df, formation, interp):
    tri = []

    # dataframe filters
    f1 = df["formation"] == formation
    f2 = df["interp"] == interp

    # stick id's
    sid = df[f1 & f2]["stick id"].unique()

    for s1, s2 in zip(sid[:-1], sid[1:]):  # in case they're not exactly consecutive
        f = f1 & f2 & (df["stick id"] == s1)
        p1 = np.array([df[f].X, df[f].Y, df[f].Z]).T
        f = f1 & f2 & (df["stick id"] == s2)
        p2 = np.array([df[f].X, df[f].Y, df[f].Z]).T
        p = np.append(p1, p2, axis=0)

        tri.append(Delaunay(p))

    return tri


def normals_from_delaunay(tris):
    Cs, Ns, Sm = [], [], []
    for tri in tris:
        for sim in tri.simplices:
            C, N = plane_fit(tri.points[sim].T)
            Cs.append(C)
            Ns.append(N)
            Sm.append(sim)

    Cs = np.array(Cs)
    Ns = np.array(Ns)
    Sm = np.array(Sm)

    return Cs, Ns, Sm


def orient_for_interp(Cs, Ns, formation, interp, filter_=True):
    ndf = pd.DataFrame(columns="X Y Z G_x G_y G_z".split(" "))
    ndf.X, ndf.Y, ndf.Z = Cs[:, 0], Cs[:, 1], Cs[:, 2]
    ndf.G_x, ndf.G_y, ndf.G_z = Ns[:, 0], Ns[:, 1], Ns[:, 2]
    ndf["formation"] = formation
    ndf["interp"] = interp

    plunge, bearing = mplstereonet.vector2plunge_bearing(Ns[:, 0], Ns[:, 1], Ns[:, 2])
    ndf["dip"], ndf["azimuth"] = plunge, bearing
    ndf["polarity"] = 1

    if filter_:
        vertbool = ndf["dip"].astype(int) != 0
        return ndf[vertbool]
    else:
        return ndf
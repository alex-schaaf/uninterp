import numpy as np
from scipy.spatial import Delaunay
import sys
sys.path.append("../../gempy")
from gempy.utils.petrel_integration import plane_fit
import mplstereonet
import pandas as pd


def histogram_2d(df: pd.DataFrame, min_:float, max_:float, direction:str="Z", bins:int=40, range_=None):
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


def concave_trisurf(df: pd.DataFrame, formation: str, interp: int):
    """Construct 'concave' fault surface from given fault sticks by use of iterative Delaunay triangulation.

    Args:
        df: pandas DataFrame containing fault stick information with columns [X Y Z formation interp "stick_id"]
        formation: Formation to construct surface for.
        interp: Interpretation to construct surface for.

    Returns:
        List of Delaunay objects.
    """
    tri = []

    # dataframe filters
    f1 = df.formation == formation
    f2 = df.interp == interp

    # stick id's
    sid = df[f1 & f2]["stick id"].unique()

    for s1, s2 in zip(sid[:-1], sid[1:]):  # in case they're not exactly consecutive
        f = f1 & f2 & (df["stick id"] == s1)
        p1 = np.array([df[f].X, df[f].Y, df[f].Z]).T
        f = f1 & f2 & (df["stick id"] == s2)
        if np.count_nonzero(f) <= 1:
            continue
        p2 = np.array([df[f].X, df[f].Y, df[f].Z]).T
        p = np.append(p1, p2, axis=0)

        tri.append(Delaunay(p))

    return tri


def normals_from_delaunay(tris: list):
    """Compute normals for given list of Delaunay objects containing triangles.

    Args:
        tris (list): List of Delaunay objects.

    Returns:
        (tuple) centroids (np.ndrarray), normals (np.ndarray), simplices (np.ndarray)
    """
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


def orient_for_interp(Cs: np.ndarray, Ns:np.ndarray, formation:str, interp:int, filter_:bool=True):
    """Converts given centroids and normals into pandas.DataFrame compatible with GemPy orientation data structure.

    Args:
        Cs (np.ndarray): Centroids of planes (N,3)
        Ns (np.ndarray): Normals of planes (N,3)
        formation (str): Formation name
        interp (int): Interpretation id
        filter_ (bool): If orientation data should be filtered or not (default: True).
            Currently only filters completely horizontal planes.

    Returns:
        (pd.DataFrame) Orientations dataframe compatible with GemPy's data structure.
    """
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
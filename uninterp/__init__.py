import numpy as np
from scipy.spatial import Delaunay
import sys
sys.path.append("../../gempy")
import mplstereonet
import pandas as pd

def histogram_2d(df:pd.DataFrame, min_:float, max_:float, direction:str="Z", bins:int=40, range_=None):
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


def bin_df(df:pd.DataFrame, nbins:iter, extent:iter=None):
    """Inplace binning of given dataframe along x,y,z axis using given number of divisions in each direction.

    Args:
        df (pd.DataFrame): DataFrame to be binned, must contain columns X,Y,Z
        nbins (iter): List or tuple of int containing the number of bins in each direction (x,y,z)
        extent (iter): List or tuple of float containing the extent (x,X,y,Y,z,Z)

    Returns:
        None
    """
    if not extent:
        extent = [df.X.min(), df.X.max(), df.Y.min(), df.Y.max(), df.Z.min(), df.Z.max()]
    df["xbin"] = pd.cut(df["X"], np.linspace(extent[0], extent[1], nbins[0]))
    df["ybin"] = pd.cut(df["Y"], np.linspace(extent[2], extent[3], nbins[1]))
    df["zbin"] = pd.cut(df["Z"], np.linspace(extent[4], extent[5], nbins[2]))
    return None


def concave_faultstick_trisurf(df:pd.DataFrame, formation: str, interp: int):
    """Construct 'concave' fault surface from given fault sticks by use of iterative Delaunay triangulation.

    Args:
        df: pandas DataFrame containing fault stick information with columns [X Y Z formation interp "stick_id"]
        formation: Formation to construct surface for.
        interp: Interpretation to construct surface for.

    Returns:
        List of Delaunay objects.
    """
    triangulators = []

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
        points = np.append(p1, p2, axis=0)

        triangulators.append(Delaunay(points))

    return triangulators


def normals_from_delaunay(delaunays:list):
    """Compute normals for given list of Delaunay objects containing triangles.

    Args:
        delaunays (list): List of Delaunay objects.

    Returns:
        (tuple) centroids (np.ndrarray), normals (np.ndarray), simplices (np.ndarray)
    """
    centroids, normals, simplices = [], [], []
    for tri in delaunays:
        for sim in tri.simplices:
            C, N = plane_fit(tri.points[sim].T)
            centroids.append(C)
            normals.append(N)
            simplices.append(sim)

    centroids = np.array(centroids)
    normals = np.array(normals)
    simplices = np.array(simplices)

    return centroids, normals, simplices


def orient_for_interp(centroids:np.ndarray, normals:np.ndarray, formation:str, interp:int, filter_:bool=True):
    """Converts given centroids and normals into pandas.DataFrame compatible with GemPy orientation data structure.

    Args:
        centroids (np.ndarray): Centroids of planes (N,3)
        normals (np.ndarray): Normals of planes (N,3)
        formation (str): Formation name
        interp (int): Interpretation id
        filter_ (bool): If orientation data should be filtered or not (default: True).
            Currently only filters completely horizontal planes.

    Returns:
        (pd.DataFrame) Orientations dataframe compatible with GemPy's data structure.
    """
    ndf = pd.DataFrame(columns="X Y Z G_x G_y G_z".split(" "))
    ndf.X, ndf.Y, ndf.Z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    ndf.G_x, ndf.G_y, ndf.G_z = normals[:, 0], normals[:, 1], normals[:, 2]
    ndf["formation"] = formation
    ndf["interp"] = interp

    plunge, bearing = mplstereonet.vector2plunge_bearing(normals[:, 0], normals[:, 1], normals[:, 2])
    ndf["dip"], ndf["azimuth"] = plunge, bearing
    ndf["polarity"] = 1

    if filter_:
        vertbool = ndf["dip"].astype(int) != 0
        return ndf[vertbool]
    else:
        return ndf


def get_fault_orientations(df:pd.DataFrame, fault:str, nbins:iter=(5,5,4)):
    """Fits planes to binned fault interpretations (per interpretation to estimate orientation, returns GemPy-compatible
    pd.DataFrame with orientation data.

    Args:
        df (pd.DataFrame): DataFrame containing the fault stick interpretation points.
        fault (str): Formation string of the fault to be extracted.
        nbins (iter): List or tuple containing the number of bins in each spatial direction (x,y,z), default (5,5,4)

    Returns:
        pd.DataFrame with GemPy-compatible orientations structure
    """
    df = df[df.formation == fault]
    bin_df(df, nbins)
    groups = df.groupby(["xbin", "ybin", "zbin", "interp"]).groups

    df_columns = "X Y Z G_x G_y G_z dip azimuth polarity formation interp xbin ybin zbin".split()
    orientations = pd.DataFrame(columns=df_columns)

    for key, i in groups.items():
        if len(i) < 3:
            continue  # can't fit plane
        points = df.iloc[i][["X", "Y", "Z"]].values
        centroid, normal = plane_fit(points.T)
        dip, azimuth = mplstereonet.vector2plunge_bearing(normal[0], normal[1], normal[2])

        orientation = [centroid[0], centroid[1], centroid[2],
                       normal[0], normal[1], normal[2],
                       float(dip), float(azimuth), 1,
                       fault,
                       key[3],
                       key[0],
                       key[1],
                       key[2]]

        orientations.loc[len(orientations)] = orientation

    return orientations


def plane_fit(points):
    """
    Fit plane to points in PointSet
    Fit an d-dimensional plane to the points in a point set.
    adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

    Args:
        point_list (array_like): array of points XYZ

    Returns:
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
    """

    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    centroid = points.mean(axis=1)
    x = points - centroid[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    normal = svd(M)[0][:, -1]
    # return ctr, svd(M)[0][:, -1]
    if normal[2] < 0:
        normal = -normal

    return centroid, normal
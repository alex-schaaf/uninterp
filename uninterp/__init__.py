import numpy as np
from scipy.spatial import Delaunay
import sys
sys.path.append("../../gempy")
import mplstereonet
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist, euclidean
import bokeh.plotting as bp
import warnings


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


def get_interf_orientations(df:pd.DataFrame, fault:str, nbins:iter=(5, 5, 4), extent=None):
    """Fits planes to binned fault interpretations (per interpretation to estimate orientation, returns GemPy-compatible
    pd.DataFrame with orientation data.

    Args:
        df (pd.DataFrame): DataFrame containing the fault stick interpretation points.
        fault (str): Formation string of the fault to be extracted.
        nbins (iter): List or tuple containing the number of bins in each spatial direction (x,y,z), default (5,5,4)
        extent (tuple): (x,X,y,Y,z,Z), if not given will determin extent from data

    Returns:
        pd.DataFrame with GemPy-compatible orientations structure
    """
    df = df[df.formation == fault]
    bin_df(df, nbins, extent=extent)
    groups = df.groupby(["xbin", "ybin", "zbin", "interp"]).groups

    df_columns = "X Y Z G_x G_y G_z dip azimuth polarity formation interp xbin ybin zbin".split()
    orientations = pd.DataFrame(columns=df_columns)

    for key, i in groups.items():
        if len(i) < 3:
            continue  # can't fit plane
        points = df.loc[i][["X", "Y", "Z"]].values
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


def mean_std_from_interp(df:pd.DataFrame, fmt:str, axis:str):
    """Extracts mean, standard deviation and count of from fault or horizon interpretations from given binned dataframe
    along specified direction for given formation.

    Args:
        df: pd.DataFrame containing interpretation data, prebinned!
        fmt: Name of formation to be analysed
        axis: Along which axis (x,y,z) to compute the mean, std and count

    Returns:
        pd.DataFrame
    """
    if axis == "x":
        grp = df[df.formation == fmt].groupby(["ybin", "zbin"])
        mean, std, count = grp.X.mean(), grp.X.std(), grp.X.count()
    elif axis == "z":
        grp = df[df.formation == fmt].groupby(["xbin", "ybin"])
        mean, std, count = grp.Z.mean(), grp.Z.std(), grp.Z.count()
    elif axis == "y":
        grp = df[df.formation == fmt].groupby(["xbin", "zbin"])
        mean, std, count = grp.Y.mean(), grp.Y.std(), grp.Y.count()
    else:
        raise ValueError("Direction must be either x, y or z.")

    rdf = pd.DataFrame(mean)
    rdf["std"], rdf["count"] = std, count

    return rdf

def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """modified from source: https://stackoverflow.com/a/51127674"""


    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        except Warning as e:
            print("warning found:", e, "for input data: ", x1, y1, x2, y2, x3, y3, x4, y4)
            return None
    return np.array([px[0], py[0]]).flatten()


def get_fault_throw(fd, hor1, hor2, n_dist=3, plot=True, grad_filter:int=None):
    """Analyses fault throw at given fault between two given sets of horizons.

    Args:
        fd (pd.DataFrame): fault data (fault interp and stick pre-filtered)
        hor1 (pd.DataFrame): horizon 1 data
        hor2 (pd.DataFrame): horizon 2 data

    Returns:
        pd.DataFrame containing fault throw, heave and dip separation and auxiliary information.
    """
    # TODO: fix negative Z workaround
    # ------------------------------------------
    # FAULT
    # LinReg
    f_linreg = LinearRegression()
    f_linreg.fit(fd.X[:, np.newaxis], -fd.Z[:, np.newaxis])

    # Fault stick centroid
    fc = fd[["X", "Y", "Z"]].mean()

    # PLOT FAULT
    if plot:
        # predict for plot
        nx = np.linspace(fd.X.min() - 500, fd.X.max() + 500, 100)
        f_linreg_p = f_linreg.predict(nx[:, np.newaxis])

        p = bp.figure()
        p.circle(fd.X, -fd.Z, color="black", legend="Fault")
        p.line(fd.X, -fd.Z, color="black")
        p.line(nx, f_linreg_p[:, 0], color="lightgrey")
        p.circle(fc[0], -fc[2], color="red", legend="Fault stick centroid")

    # ------------------------------------------
    # DISTANCES
    # find nearest point (euclidean)
    h1pi = np.argsort(cdist([tuple(fc)], hor1[["X", "Y", "Z"]].values))[0, :n_dist]
    h2pi = np.argsort(cdist([tuple(fc)], hor2[["X", "Y", "Z"]].values))[0, :n_dist]

    h1d = hor1[hor1["Y"].isin(hor1.iloc[h1pi].Y.values)]
    h2d = hor2[hor2["Y"].isin(hor2.iloc[h2pi].Y.values)]

    if len(h1d) == 0 or len(h2d) == 0:
        return None

    # ------------------------------------------
    # BLOCK 1

    # linear regression
    # gradf = h1_zgrad < -0  # what if all errors bigger?

    if grad_filter is not None and len(h1d) >= 4:
        gradf = np.gradient(h1d.Z) < np.percentile(np.gradient(h1d.Z), grad_filter)
    else:
        gradf = slice(None)
    h1_linreg = LinearRegression()
    h1_linreg.fit(h1d.X[gradf][:, np.newaxis], -h1d.Z[gradf][:, np.newaxis])

    intercept1 = findIntersection(fd.X.min(), f_linreg.predict(np.array([fd.X.min()])[:, np.newaxis]),
                                  fd.X.max(), f_linreg.predict(np.array([fd.X.max()])[:, np.newaxis]),
                                  h1d.X.min(), h1_linreg.predict(np.array([h1d.X.min()])[:, np.newaxis]),
                                  h1d.X.max(), h1_linreg.predict(np.array([h1d.X.max()])[:, np.newaxis]))
    if intercept1 is None:
        return None

    if plot:
        ## predict and plot
        nx = np.linspace(h1d.X.min() - 500, h1d.X.max() + 500, 100)
        h1_linreg_p = h1_linreg.predict(nx[:, np.newaxis])
        p.circle(h1d.X, -h1d.Z, color="lightblue", legend="Block 1")
        p.line(nx, h1_linreg_p[:, 0], color="lightblue")

    # ------------------------------------------
    # BLOCK 2
    # linear regression
    if grad_filter is not None and len(h2d) >= 4:
            gradf = np.gradient(h2d.Z) < np.percentile(np.gradient(h2d.Z), grad_filter)
    else:
        gradf = slice(None)
    h2_linreg = LinearRegression()
    h2_linreg.fit(h2d.X[gradf][:, np.newaxis], -h2d.Z[gradf][:, np.newaxis])

    intercept2 = findIntersection(fd.X.min(), f_linreg.predict(np.array([fd.X.min()])[:, np.newaxis]),
                                  fd.X.max(), f_linreg.predict(np.array([fd.X.max()])[:, np.newaxis]),
                                  h2d.X.min(), h2_linreg.predict(np.array([h2d.X.min()])[:, np.newaxis]),
                                  h2d.X.max(), h2_linreg.predict(np.array([h2d.X.max()])[:, np.newaxis]))
    if intercept2 is None:
        return None

    if plot:
        nx = np.linspace(h2d.X.min() - 500, h2d.X.max() + 500, 100)
        h2_linreg_p = h2_linreg.predict(nx[:, np.newaxis])
        p.circle(h2d.X, -h2d.Z, color="orange", legend="Block 2")
        p.line(nx, h2_linreg_p[:, 0], color="orange")
        p.circle(intercept1[0], intercept1[1], color="pink", legend="Intersect 1", size=10)
        p.circle(intercept2[0], intercept2[1], color="lime", legend="Intersect 2", size=10)

        bp.show(p)
    # slip
    # TODO: positive for normal fault slip, negative for reverse fault slip
    try:
        dip_separation = euclidean(intercept1, intercept2)
    except ValueError:
        return None
    heave = abs(intercept1[0] - intercept2[0])
    throw = abs(intercept1[1] - intercept2[1])

    # get data ready for return
    data = {"X": fc[0], "Y": fc[1], "Z": fc[2], "throw": throw, "heave": heave, "dipsep": dip_separation,
            "i1x": intercept1[0], "i1z": intercept1[1], "i2x": intercept2[0], "i2z": intercept2[1],
            "interp": fd.interp.unique(), "stick": fd.stick.unique(), "formation": fd.formation.unique(), "block": fd.block.unique()}
    return pd.DataFrame(data, index=[np.nan])

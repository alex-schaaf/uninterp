import numpy as np
import pandas as pd


def read_fault_sticks_kingdom(fp, formation=None):
    """
    Reads in Kingdom fault stick files (kingdom) exported from Petrel (tested with Petrel 2017) and returns
    pandas DataFrame.

    Args:
        fp (str): Filepath.
        formation (str, optional): Default: None.

    Returns:
        (pandas.DataFrame) Fault stick information stored in dataframe with ["X", "Y", "Z", "formation", "stick id"] columns.

    """
    storage = []
    with open(fp, "r") as file:  # due to the variable delimiter length its easier to just manually read this in
        lines = file.readlines()
        for line in lines:
            line = line.split(" ")
            X = float(line[6])
            Y = float(line[7])
            Z = float(line[9])
            if formation is None:
                name = line[10]
            else:
                name = formation
            stick = int(line[-2])
            storage.append([X, Y, Z, name, stick])

    df = pd.DataFrame(storage)
    df.columns = ["X", "Y", "Z", "formation", "stick id"]
    return df


def read_fault_sticks_charisma(fp:str, formation=None):
    """
    Reads in charisma fault stick files exported from Petrel (tested with Petrel 2017) and returns
    pandas DataFrame.

    Args:
        fp (str): Filepath.
        formation (str, optional): Default: None.

    Returns:
        (pandas.DataFrame) Fault stick information stored in dataframe with ["X", "Y", "Z", "formation", "stick id"] columns.

    """
    storage = []
    with open(fp, "r") as file:  # due to the variable delimiter length its easier to just manually read this in
        lines = file.readlines()
        for line in lines:
            line = line.split(" ")
            line = [l for l in line if len(l) >= 1]
            X = float(line[3])
            Y = float(line[4])
            Z = float(line[5])
            if formation is None:
                name = line[6]
            else:
                name = formation
            stick = int(line[-1])
            storage.append([X, Y, Z, name, stick])

    df = pd.DataFrame(storage)
    df.columns = ["X", "Y", "Z", "formation", "stick id"]
    return df


def read_cps3_grid(fp:str, dropshit:bool=True):
    """Read CPS-3 gridded regular surface files exported by Petrel 2017 and returns a Pandas DataFrame.

    Args:
        fp (str): Filepath.
        dropshit (bool): To drop or not to drop the useless grid points. Default: True

    Returns:
        (pandas.DataFrame) Fault stick information stored in dataframe with ["X", "Y", "Z"] columns.
    """

    with open(fp, "r") as file:
        lines = list(map(str.rstrip, file))

    extent = np.array(lines[2].split(" ")[1:]).astype(float)
    fsnrow = np.array(lines[3].split(" ")[1:]).astype(int)
    fsxinc = np.array(lines[4].split(" ")[1:]).astype(float)

    grid = []
    for line in lines[6:]:
        grid.append(line.split(" "))

    rows = []
    de = np.arange(0, len(grid) + 1, len(grid) / fsnrow[1]).astype(int)
    for aa, bb in zip(de[:-1], de[1:]):
        row = grid[aa:bb]
        flat_row = np.array([item for sublist in row for item in sublist]).astype(float)
        rows.append((flat_row))

    rows = np.array(rows)

    Xs = np.linspace(extent[0], extent[1], fsnrow[1])
    Ys = np.linspace(extent[2], extent[3], fsnrow[0])

    XY = []
    for x in Xs:
        for y in Ys:
            XY.append([x, y])
    XY = np.array(XY)

    g = np.concatenate((XY, rows.flatten()[:, np.newaxis]), axis=1).T
    if dropshit:
        g = g[:, g[2, :] != np.max(g)]

    df = pd.DataFrame(columns=['X', 'Y', 'Z'])
    df["X"] = g[0,:]
    df["Y"] = g[1, :]
    df["Z"] = g[2, :]

    return df


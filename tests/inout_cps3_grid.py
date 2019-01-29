"""
Visual check to assess proper cps3 grid loading.
"""

import matplotlib.pyplot as plt

import sys

sys.path.append("../")
import uninterp.inout

df = uninterp.inout.read_cps3_grid("data/A")

fig, ax = plt.subplots(ncols=3)
style = {"s": 0.1}
ax[0].scatter(df.X, df.Y, **style)
ax[1].scatter(df.X, df.Z, **style)
ax[2].scatter(df.Y, df.Z, **style)
plt.show()
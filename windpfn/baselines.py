"""Reference forecasts. Anything fitted is fitted on Mar–Jul 2024 and scored on Aug–Dec 2024."""
import numpy as np
from sklearn.linear_model import LinearRegression
from .dataset import WS

SPLIT = "2024-08-01"
# generic normalised turbine power curve (m/s -> fraction of capacity); fleet cut-out ramps down
PC_V = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 32]
PC_P = [0, 0, .03, .08, .16, .27, .41, .57, .73, .86, .95, 1, 1, 0]


def add(d):
    """powercurve: capacity x non-negative weighted sum of per-point power curves. blend: its mean with WINDFOR."""
    P = np.interp(d[WS].to_numpy(), PC_V, PC_P)
    tr = d.index < SPLIT
    pc = LinearRegression(positive=True).fit(P[tr], (d.y / d.cap)[tr]).predict(P) * d.cap
    return d.assign(curtailed=d.y - d.y_metered, powercurve=pc, blend=(d.windfor + pc) / 2)

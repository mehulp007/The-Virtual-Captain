"""Convert Cricinfo wagon-wheel readings into field coordinates.

The source gives `wagon_angle` in degrees with 0 straight behind the batter, increasing
through the leg side (fine leg ~27, mid-wicket ~115, mid-off ~203, third man ~335), and
`wagon_distance` as a fraction of the boundary radius measured from the pitch centre.
Left-handers are already mirrored, so the off side is always 180-360 degrees.
"""

import numpy as np
import polars as pl

from virtual_captain.field.geometry import BOUNDARY_RADIUS


def wagon_to_xy(angle_deg, distance, radius: float = BOUNDARY_RADIUS):
    """Wagon angle/distance -> (x, y) in metres in the mirrored right-hander frame."""
    theta = np.deg2rad(np.asarray(angle_deg, dtype=float))
    r = np.asarray(distance, dtype=float) * radius
    return -np.sin(theta) * r, -np.cos(theta) * r


def xy_to_wagon(x, y, radius: float = BOUNDARY_RADIUS):
    """Inverse of `wagon_to_xy`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    angle = np.rad2deg(np.arctan2(-x, -y)) % 360
    return angle, np.hypot(x, y) / radius


def wagon_xy_exprs(radius: float = BOUNDARY_RADIUS) -> list[pl.Expr]:
    """Polars expressions adding x_m / y_m columns from wagon_angle / wagon_distance."""
    theta = pl.col("wagon_angle").radians()
    r = pl.col("wagon_distance") * radius
    return [(-theta.sin() * r).alias("x_m"), (-theta.cos() * r).alias("y_m")]

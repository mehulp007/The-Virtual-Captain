"""The "wagon-wheel analyst" baseline: put fielders where the batter hits the ball most.

It keeps the conventional field's appetite for risk (the same number of boundary riders) and
then fills the deep and the ring from the batter's own shot chart, busiest direction first,
never two fielders in the same or neighbouring sectors. No simulation is involved, which is
the point: it is what an analyst with a wagon wheel and no model would do.
"""

import numpy as np
import polars as pl

from virtual_captain.field import rules
from virtual_captain.field.positions import DEEP, POSITIONS, xy
from virtual_captain.field.templates import TEMPLATES

SECTORS = 24
RING_RADIUS = 26.0  # inside the 30-yard ring in every direction
DEEP_FROM = 0.7  # shots that ended beyond this fraction of the boundary count as "deep"


def _busiest(weights: np.ndarray, n: int) -> list[int]:
    w = weights.astype(float).copy()
    chosen = []
    for _ in range(n):
        s = int(np.argmax(w))
        chosen.append(s)
        w[[s, (s - 1) % SECTORS, (s + 1) % SECTORS]] = -1.0
    return chosen


def analyst_field(shots: pl.DataFrame, fmt: str, phase: str, bowling: str) -> np.ndarray:
    """shots: the batter's balls put into play in this context (theta, dist columns)."""
    names = TEMPLATES[(fmt, phase, "spin" if bowling == "spin" else "pace")]
    n_deep = sum(POSITIONS[n][1] > 60 for n in names)

    theta = shots["theta"].to_numpy()
    dist = shots["dist"].to_numpy()
    sector = (theta / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    deep = np.bincount(sector[dist >= DEEP_FROM], minlength=SECTORS)
    ring = np.bincount(sector[dist < DEEP_FROM], minlength=SECTORS)

    width = 360 / SECTORS
    field = [xy(((s + 0.5) * width, DEEP)) for s in _busiest(deep, n_deep)]
    field += [xy(((s + 0.5) * width, RING_RADIUS)) for s in _busiest(ring, 9 - n_deep)]
    return rules.repair(np.stack(field), fmt, phase)

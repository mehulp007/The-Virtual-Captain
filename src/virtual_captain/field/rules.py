"""Fielding restrictions and a repair step that turns any field into a legal one.

Rules (for the nine fielders other than bowler and keeper):
  * no more than two behind square on the leg side (every format)
  * no more than N outside the 30-yard ring, where N depends on format and phase
    (T20 2 / 5, ODI 2 / 4 / 5, no limit in Tests)
Plus practical constraints: inside the boundary, off the pitch, not on top of the striker.
"""

from dataclasses import dataclass

import numpy as np

from virtual_captain.config import load_format
from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF, in_ring

POPPING_CREASE_Y = -(PITCH_HALF - 1.22)  # striker's popping crease
PITCH_HALF_WIDTH = 1.6
MIN_FROM_STRIKER = 3.0
MAX_RADIUS = BOUNDARY_RADIUS - 1.0


def max_outside_ring(fmt: str, phase: str) -> int:
    limits = load_format(fmt)["fielding"]["max_outside_ring"]
    return 9 if limits is None else int(limits[phase])


def max_leg_behind_square(fmt: str) -> int:
    return int(load_format(fmt)["fielding"]["max_leg_behind_square"])


def leg_behind_square(field: np.ndarray) -> np.ndarray:
    return (field[:, 0] < 0) & (field[:, 1] < POPPING_CREASE_Y)


def outside_ring(field: np.ndarray) -> np.ndarray:
    return ~in_ring(field[:, 0], field[:, 1])


@dataclass
class Check:
    legal: bool
    outside_ring: int
    leg_behind_square: int
    problems: list[str]


def check(field: np.ndarray, fmt: str, phase: str) -> Check:
    field = np.asarray(field, dtype=float)
    problems = []
    n_out = int(outside_ring(field).sum())
    n_leg = int(leg_behind_square(field).sum())
    if field.shape != (9, 2):
        problems.append(f"expected 9 fielders, got {field.shape[0]}")
    if n_out > max_outside_ring(fmt, phase):
        problems.append(f"{n_out} outside the ring, limit {max_outside_ring(fmt, phase)}")
    leg_limit = max_leg_behind_square(fmt)
    if n_leg > leg_limit:
        problems.append(f"{n_leg} behind square on the leg side, limit {leg_limit}")
    if (np.hypot(field[:, 0], field[:, 1]) > MAX_RADIUS + 1e-6).any():
        problems.append("fielder outside the boundary")
    on_pitch = (np.abs(field[:, 0]) < PITCH_HALF_WIDTH) & (np.abs(field[:, 1]) < PITCH_HALF)
    if on_pitch.any():
        problems.append("fielder on the pitch")
    return Check(not problems, n_out, n_leg, problems)


def _inside_ring_along_ray(p: np.ndarray, margin: float = 1.0) -> np.ndarray:
    """Pull a point straight towards the pitch centre until it is `margin` m inside the ring."""
    r = np.hypot(*p)
    unit = p / r
    for step in np.arange(r, 0, -0.25):
        q = unit * step
        if in_ring(*(unit * (step + margin))) and in_ring(*q):
            return q
    return unit * 1.0


def repair(field: np.ndarray, fmt: str, phase: str) -> np.ndarray:
    """Smallest set of moves (fielder by fielder) that makes a field legal."""
    f = np.array(field, dtype=float).copy()

    # stay inside the boundary
    r = np.hypot(f[:, 0], f[:, 1])
    too_far = r > MAX_RADIUS
    f[too_far] *= (MAX_RADIUS / r[too_far])[:, None]

    # off the pitch: step sideways to the nearest edge of the strip
    on_pitch = (np.abs(f[:, 0]) < PITCH_HALF_WIDTH) & (np.abs(f[:, 1]) < PITCH_HALF)
    f[on_pitch, 0] = np.where(f[on_pitch, 0] >= 0, PITCH_HALF_WIDTH, -PITCH_HALF_WIDTH)

    # keep clear of the striker
    striker = np.array([0.0, -PITCH_HALF])
    d = np.hypot(*(f - striker).T)
    close = d < MIN_FROM_STRIKER
    if close.any():
        away = f[close] - striker
        away[np.hypot(*away.T) < 1e-9] = [1.0, 0.0]
        away /= np.hypot(*away.T)[:, None]
        f[close] = striker + away * MIN_FROM_STRIKER

    # leg side behind square: bring the cheapest ones level with the popping crease
    excess = int(leg_behind_square(f).sum()) - max_leg_behind_square(fmt)
    if excess > 0:
        idx = np.flatnonzero(leg_behind_square(f))
        cost = POPPING_CREASE_Y - f[idx, 1]
        for i in idx[np.argsort(cost)][:excess]:
            f[i, 1] = POPPING_CREASE_Y + 0.5

    # too many on the boundary: bring in the ones closest to the ring
    excess = int(outside_ring(f).sum()) - max_outside_ring(fmt, phase)
    if excess > 0:
        idx = np.flatnonzero(outside_ring(f))
        cost = np.array([np.hypot(*(f[i] - _inside_ring_along_ray(f[i]))) for i in idx])
        for i in idx[np.argsort(cost)][:excess]:
            f[i] = _inside_ring_along_ray(f[i])
    return f

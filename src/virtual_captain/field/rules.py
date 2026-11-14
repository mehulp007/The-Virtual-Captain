"""Fielding restrictions and a repair step that turns any field into a legal one.

Rules (for the nine fielders other than bowler and keeper):
  * no more than two behind square on the leg side (every format)
  * no more than N outside the 30-yard ring, where N depends on format and phase
    (T20 2 / 5, ODI 2 / 4 / 5, no limit in Tests)
Plus practical constraints: inside the boundary, off the pitch, not on top of the striker.
"""

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from numba import njit

from virtual_captain.config import load_format
from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF, RING_RADIUS, in_ring

POPPING_CREASE_Y = -(PITCH_HALF - 1.22)  # striker's popping crease
PITCH_HALF_WIDTH = 1.6
MIN_FROM_STRIKER = 3.0
MAX_RADIUS = BOUNDARY_RADIUS - 1.0


@cache
def max_outside_ring(fmt: str, phase: str) -> int:
    limits = load_format(fmt)["fielding"]["max_outside_ring"]
    return 9 if limits is None else int(limits[phase])


@cache
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


RAY_STEP = 0.25  # m, how far a fielder is walked in at a time when brought inside the ring
RING_MARGIN = 1.0  # m, how far inside the ring they end up


@njit
def _in_ring(x, y):
    return (
        math.hypot(x, y + PITCH_HALF) <= RING_RADIUS
        or math.hypot(x, y - PITCH_HALF) <= RING_RADIUS
        or (abs(x) <= RING_RADIUS and abs(y) <= PITCH_HALF)
    )


@njit
def _ring_edge(ux, uy):
    """Distance from the pitch centre to the edge of the ring along unit direction (ux, uy).

    The ring is every point within RING_RADIUS of the segment joining the two sets of stumps.
    """
    if ux != 0.0:
        t = RING_RADIUS / abs(ux)
        if abs(t * uy) <= PITCH_HALF:
            return t
    h = PITCH_HALF * abs(uy)
    return h + math.sqrt(h * h - PITCH_HALF * PITCH_HALF + RING_RADIUS * RING_RADIUS)


@njit
def _pull_inside(px, py):
    """Walk a point straight towards the pitch centre, RAY_STEP at a time from where it stands,
    until it is RING_MARGIN inside the ring. Returns the new (x, y)."""
    r = math.hypot(px, py)
    ux, uy = px / r, py / r
    # jump to just before the first step that can work, then check step by step
    k = max(0, int(math.floor((r - _ring_edge(ux, uy) + RING_MARGIN) / RAY_STEP)) - 2)
    step = r - k * RAY_STEP
    while step > 0:
        reach = step + RING_MARGIN
        if _in_ring(ux * reach, uy * reach) and _in_ring(ux * step, uy * step):
            return ux * step, uy * step
        k += 1
        step = r - k * RAY_STEP
    return ux, uy


@njit
def _repair_one(f, max_out, max_leg, out):
    n = f.shape[0]
    for i in range(n):
        out[i, 0] = f[i, 0]
        out[i, 1] = f[i, 1]

    for i in range(n):
        # stay inside the boundary
        r = math.hypot(out[i, 0], out[i, 1])
        if r > MAX_RADIUS:
            out[i, 0] *= MAX_RADIUS / r
            out[i, 1] *= MAX_RADIUS / r
        # off the pitch: step sideways to the nearest edge of the strip
        if abs(out[i, 0]) < PITCH_HALF_WIDTH and abs(out[i, 1]) < PITCH_HALF:
            out[i, 0] = PITCH_HALF_WIDTH if out[i, 0] >= 0 else -PITCH_HALF_WIDTH
        # keep clear of the striker
        ax, ay = out[i, 0], out[i, 1] + PITCH_HALF
        d = math.hypot(ax, ay)
        if d < MIN_FROM_STRIKER:
            if d < 1e-9:
                ax, ay, d = 1.0, 0.0, 1.0
            out[i, 0] = ax / d * MIN_FROM_STRIKER
            out[i, 1] = -PITCH_HALF + ay / d * MIN_FROM_STRIKER

    # leg side behind square: bring the cheapest ones level with the popping crease
    idx = np.empty(n, dtype=np.int64)
    cost = np.empty(n)
    m = 0
    for i in range(n):
        if out[i, 0] < 0 and out[i, 1] < POPPING_CREASE_Y:
            idx[m] = i
            cost[m] = POPPING_CREASE_Y - out[i, 1]
            m += 1
    if m > max_leg:
        order = np.argsort(cost[:m], kind="mergesort")
        for j in range(m - max_leg):
            i = idx[order[j]]
            out[i, 1] = POPPING_CREASE_Y + 0.5
            if out[i, 0] > -PITCH_HALF_WIDTH:  # was right behind the stumps: off the pitch too
                out[i, 0] = -PITCH_HALF_WIDTH

    # too many on the boundary: bring in the ones closest to the ring
    m = 0
    qx = np.empty(n)
    qy = np.empty(n)
    for i in range(n):
        if not _in_ring(out[i, 0], out[i, 1]):
            x, y = _pull_inside(out[i, 0], out[i, 1])
            idx[m] = i
            qx[m] = x
            qy[m] = y
            cost[m] = math.hypot(out[i, 0] - x, out[i, 1] - y)
            m += 1
    if m > max_out:
        order = np.argsort(cost[:m], kind="mergesort")
        for j in range(m - max_out):
            k = order[j]
            out[idx[k], 0] = qx[k]
            out[idx[k], 1] = qy[k]


@njit
def _repair_many(fields, max_out, max_leg):
    out = np.empty_like(fields)
    for b in range(fields.shape[0]):
        _repair_one(fields[b], max_out[b], max_leg[b], out[b])
    return out


def _inside_ring_along_ray(p: np.ndarray) -> np.ndarray:
    return np.array(_pull_inside(float(p[0]), float(p[1])))


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


def repair_many(fields: np.ndarray, max_out: np.ndarray, max_leg: np.ndarray) -> np.ndarray:
    """Repair a batch of fields (B, 9, 2) at once, each with its own limits.

    The same moves as `repair`, compiled, with two differences. Where two fielders are equally
    cheap to move, this moves the lower-numbered one (`repair` leaves that to NumPy's sort). And
    a leg-side fielder standing almost directly behind the stumps, when brought up level with
    the crease, is also stepped off the pitch.
    """
    return _repair_many(
        np.ascontiguousarray(fields, dtype=np.float64),
        np.ascontiguousarray(max_out, dtype=np.int64),
        np.ascontiguousarray(max_leg, dtype=np.int64),
    )

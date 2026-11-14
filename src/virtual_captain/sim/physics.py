"""Ball-versus-fielder physics, compiled with numba.

One call resolves a batch of deliveries. Each delivery already has a contact outcome and,
if it was put into play, a shot (direction, recorded distance, aerial flag) from the shot
model. The kernel works out which fielder (if any) gets to it and what that's worth.

Field arrays are (11, 2): nine fielders, then keeper (slot 9) and bowler (slot 10).
"""

import math

import numpy as np
from numba import njit, prange

from virtual_captain.sim.params import PARAM_INDEX

# outcome kinds
NO_CONTACT, FIELD_FREE_OUT, CAUGHT, DROPPED, FIELDED, FOUR, SIX, INTO_GAP, AERIAL_SAFE = range(9)
KIND_NAMES = (
    "no_contact", "bowled_lbw_stumped", "caught", "dropped", "fielded", "four", "six",
    "into_gap", "aerial_safe",
)  # fmt: skip
IN_PLAY, SIX_HIT = 2, 3  # contact outcome codes from the shot model
KEEPER = 9


# Parameter positions in the array passed to the kernel. numba freezes these as constants
# when it compiles, which is why the kernels are not cached to disk: a cached build would
# keep stale positions if SimParams changed.
_AIR_EXTRA = PARAM_INDEX["air_extra"]
_CATCH_MAX = PARAM_INDEX["catch_max"]
_CATCH_TAU = PARAM_INDEX["catch_tau"]
_DECEL = PARAM_INDEX["decel"]
_GROUND_CARRY = PARAM_INDEX["ground_carry"]
_EDGE_REACH = PARAM_INDEX["edge_reach"]
_EDGE_TIME = PARAM_INDEX["edge_time"]
_HANG_EXP = PARAM_INDEX["hang_exp"]
_HANG_FACTOR = PARAM_INDEX["hang_factor"]
_HANG_MIN = PARAM_INDEX["hang_min"]
_KEEPER_REACH = PARAM_INDEX["keeper_reach"]
_MISFIELD = PARAM_INDEX["misfield"]
_PICKUP = PARAM_INDEX["pickup"]
_REACH_AIR = PARAM_INDEX["reach_air"]
_REACH_GROUND = PARAM_INDEX["reach_ground"]
_REACT = PARAM_INDEX["react"]
_ROLL = PARAM_INDEX["roll"]
_RUN_SIGMA = PARAM_INDEX["run_sigma"]
_RUN_SPEED = PARAM_INDEX["run_speed"]
_THROW_SPEED = PARAM_INDEX["throw_speed"]
_TURN = PARAM_INDEX["turn"]

G = 9.81
EDGE_RANGE = 30.0  # metres from the striker
STEP = 1.0  # metres between checks along a ground ball's path


@njit
def _runs_from(total_time, run_time, start, u_run, p):
    """Completed runs given the time until the ball is back at the stumps.

    The k-th run needs start + k * run_time + (k - 1) * turn seconds: every run after the
    first means turning, with the throw already on its way. Each run is taken with a
    probability that rises smoothly with the time to spare, so a close single is sometimes
    risked and made. start is negative when batters are backing up and set off early.
    """
    spare = total_time - start
    runs = 0
    for k in range(1, 4):
        z = (spare - k * run_time - (k - 1) * p[_TURN]) / p[_RUN_SIGMA]
        if u_run < 1.0 / (1.0 + math.exp(-z)):
            runs = k
        else:
            break
    return runs


@njit
def _return_time(px, py, half, p):
    """Pick up and throw to the nearer set of stumps."""
    d = min(math.hypot(px, py + half), math.hypot(px, py - half))
    return p[_PICKUP] + d / p[_THROW_SPEED]


@njit
def _fielder_time(fx, fy, px, py, reach, p):
    d = max(math.hypot(fx - px, fy - py) - reach, 0.0)
    return p[_REACT] + d / p[_RUN_SPEED]


@njit
def _chase(qx, qy, ready_at, field, half, run_time, start, u_run, p):
    """Ball has stopped (or is dribbling) at q: nearest fielder retrieves it."""
    best = 1e9
    for j in range(field.shape[0]):
        t = _fielder_time(field[j, 0], field[j, 1], qx, qy, 0.0, p)
        best = min(best, t)
    t = max(best, ready_at) + _return_time(qx, qy, half, p)
    return _runs_from(t, run_time, start, u_run, p)


@njit
def resolve(
    theta, dist, aerial, u_air, u_catch, u_mis, u_run, field, radius, half, run_time, start, p
):
    """Runs and outcome kind for one ball put into play."""
    sx, sy = 0.0, -half
    ex = -math.sin(theta) * dist * radius
    ey = -math.cos(theta) * dist * radius
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 0.5:
        return 0, FIELDED
    ux, uy = dx / length, dy / length
    # distance along the ray from the striker to the rope (striker is inside the circle)
    b = sx * ux + sy * uy
    c = sx * sx + sy * sy - radius * radius
    to_rope = -b + math.sqrt(b * b - c)

    in_air = aerial or (length > 15.0 and u_air < p[_AIR_EXTRA])
    if in_air:
        if length >= to_rope:
            return 6, SIX
        edge = ey < sy and length < EDGE_RANGE
        best_margin = -1e9
        if edge:
            # no time to move: catchable only by a fielder already standing there, who then
            # holds it at the usual rate for a catch that comes straight to them
            hang = p[_EDGE_TIME]
            for j in range(field.shape[0]):
                reach = max(p[_EDGE_REACH], p[_KEEPER_REACH] if j == KEEPER else 0.0)
                if math.hypot(field[j, 0] - ex, field[j, 1] - ey) <= reach:
                    best_margin = 10.0
        else:
            hang = p[_HANG_FACTOR] * (length / 30.0) ** p[_HANG_EXP] + p[_HANG_MIN]
            for j in range(field.shape[0]):
                reach = p[_KEEPER_REACH] if j == KEEPER else p[_REACH_AIR]
                margin = hang - _fielder_time(field[j, 0], field[j, 1], ex, ey, reach, p)
                best_margin = max(best_margin, margin)
        if best_margin >= 0.0:
            p_catch = p[_CATCH_MAX] * (1.0 - math.exp(-(best_margin + 0.15) / p[_CATCH_TAU]))
            if u_catch < p_catch:
                return 0, CAUGHT
            # put down: the ball is loose next to the fielder
            t = hang + 0.8 + _return_time(ex, ey, half, p)
            return _runs_from(t, run_time, start, u_run, p), DROPPED
        carry = length * (1.0 + p[_ROLL])
        if carry >= to_rope:
            return 4, FOUR
        qx, qy = sx + ux * carry, sy + uy * carry
        return _chase(qx, qy, hang + 1.0, field, half, run_time, start, u_run, p), AERIAL_SAFE

    # along the ground: constant deceleration, free travel scaled up from where it was stopped
    decel = p[_DECEL]
    travel = length * p[_GROUND_CARRY]
    v0 = math.sqrt(2.0 * decel * travel)
    end = min(travel, to_rope)
    s = 0.5
    while s <= end:
        px, py = sx + ux * s, sy + uy * s
        t_ball = (v0 - math.sqrt(max(v0 * v0 - 2.0 * decel * s, 0.0))) / decel
        for j in range(field.shape[0]):
            reach = p[_KEEPER_REACH] if j == KEEPER else p[_REACH_GROUND]
            if _fielder_time(field[j, 0], field[j, 1], px, py, reach, p) <= t_ball:
                t_back = t_ball + _return_time(px, py, half, p)
                runs = _runs_from(t_back, run_time, start, u_run, p)
                if u_mis < p[_MISFIELD]:
                    runs = min(runs + 1, 4)
                return runs, FIELDED
        s += STEP
    if travel >= to_rope:
        return 4, FOUR
    qx, qy = sx + ux * travel, sy + uy * travel
    return _chase(qx, qy, v0 / decel, field, half, run_time, start, u_run, p), INTO_GAP


@njit(parallel=True)
def simulate_batch(
    outcome, theta, dist, aerial, field_idx, run_time, start, fields, uniforms, radius, half, p
):
    n = outcome.shape[0]
    runs = np.zeros(n, dtype=np.int8)
    kind = np.zeros(n, dtype=np.int8)
    for i in prange(n):
        if outcome[i] == SIX_HIT:
            runs[i] = 6
            kind[i] = SIX
            continue
        if outcome[i] != IN_PLAY:
            kind[i] = NO_CONTACT if outcome[i] == 0 else FIELD_FREE_OUT
            continue
        r, k = resolve(
            theta[i], dist[i], aerial[i], uniforms[i, 0], uniforms[i, 1], uniforms[i, 2],
            uniforms[i, 3],
            fields[field_idx[i]], radius, half, run_time[i], start[i], p,
        )  # fmt: skip
        runs[i] = r
        kind[i] = k
    return runs, kind

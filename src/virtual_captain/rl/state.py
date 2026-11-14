"""What the captain sees and how its moves turn into fields.

Kept apart from the environment so that anything holding a trained policy's weights (the demo
app, say) can build the same observations and fields without the training stack.
"""

import numpy as np

from virtual_captain.data.coords import wagon_to_xy, xy_to_wagon
from virtual_captain.field.geometry import BOUNDARY_RADIUS
from virtual_captain.rl.data import N_PROFILE

FORMATS = ("T20", "ODI", "TEST")
INNINGS_BALLS = {"T20": 120, "ODI": 300}
NEW_BALL_EVERY = 480
R_MIN, R_MAX = 5.0, 66.0  # fielders' distance from the pitch centre, as in optim.cma_search
STEP = np.tile([0.25, 0.3], 9)  # largest move per unit of action: 90 degrees round, 18 m in/out
N_OBS = 3 + 2 + 8 + 4 + 2 * N_PROFILE
WICKET_SCALE = 40.0
OBS_LIMIT = 10.0  # standardised profiles are clipped to +/- this


def encode_fields(fields: np.ndarray) -> np.ndarray:
    """(B, 9, 2) fields -> (B, 18): wagon-wheel angle / 360 and scaled distance per fielder."""
    angle, dist = xy_to_wagon(fields[..., 0], fields[..., 1])
    r = np.clip(dist * BOUNDARY_RADIUS, R_MIN, R_MAX)
    z = np.stack([angle / 360.0, (r - R_MIN) / (R_MAX - R_MIN)], axis=-1)
    return z.reshape(len(fields), 18)


def decode_fields(z: np.ndarray) -> np.ndarray:
    """(B, 18) -> (B, 9, 2). Angles wrap round the ground; distances are clipped."""
    z = z.reshape(len(z), 9, 2)
    angle = np.mod(z[..., 0], 1.0) * 360.0
    r = R_MIN + np.clip(z[..., 1], 0.0, 1.0) * (R_MAX - R_MIN)
    x, y = wagon_to_xy(angle, r / BOUNDARY_RADIUS)
    return np.stack([x, y], axis=-1)


def moved_fields(start: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """(B, 9, 2) fields from starting fields and (B, 18) actions in [-1, 1], before any repair."""
    a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
    return decode_fields(encode_fields(start) + a * STEP)


def progress(fmt: np.ndarray, balls_remaining: np.ndarray, ball_age: np.ndarray) -> np.ndarray:
    """How far through the innings (limited overs) or the ball's life (Tests), 0 to 1."""
    innings = np.array([INNINGS_BALLS.get(f, 1) for f in FORMATS])[fmt]
    return np.where(fmt == 2, ball_age / NEW_BALL_EVERY, 1.0 - balls_remaining / innings)


def observation(
    fmt: np.ndarray,
    spin: np.ndarray,
    situation: np.ndarray,
    wickets: np.ndarray,
    progress: np.ndarray,
    wicket_value: np.ndarray,
    ball_of_over: np.ndarray,
    striker: np.ndarray,
    other: np.ndarray,
) -> np.ndarray:
    """(n, N_OBS): format, pace or spin and (format, phase) one-hot; wickets down, progress,
    the wicket's value and how far through the over; the two batters' standardised profiles."""
    n = len(fmt)
    rows = np.arange(n)
    obs = np.zeros((n, N_OBS), dtype=np.float32)
    obs[rows, fmt] = 1.0
    obs[rows, 3 + spin] = 1.0
    obs[rows, 5 + situation] = 1.0
    obs[:, 13] = wickets / 10.0
    obs[:, 14] = progress
    obs[:, 15] = wicket_value / WICKET_SCALE
    obs[:, 16] = ball_of_over
    obs[:, 17 : 17 + N_PROFILE] = striker
    obs[:, 17 + N_PROFILE :] = other
    return np.clip(obs, -OBS_LIMIT, OBS_LIMIT)

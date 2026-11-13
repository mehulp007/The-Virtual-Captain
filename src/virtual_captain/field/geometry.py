"""Field geometry.

Coordinates are in metres with the origin at the centre of the pitch:
  +y points down the ground towards the bowler's end,
  +x points to the off side of a right-handed batter.
The striker's stumps are at (0, -PITCH_HALF). Left-handers are modelled in the same
mirrored frame, so +x is always their off side; `to_ground_frame` flips them back.
"""

import numpy as np

PITCH_LENGTH = 20.12  # stump to stump
PITCH_HALF = PITCH_LENGTH / 2
RING_RADIUS = 27.432  # 30 yards, measured from each set of middle stumps

# Wagon-wheel distances are a fraction of the boundary radius. 68 m is the radius that
# best reproduces Cricinfo's own in-ring flag given the ring shape below (see README).
BOUNDARY_RADIUS = 68.0

STRIKER = np.array([0.0, -PITCH_HALF])
NON_STRIKER = np.array([0.0, PITCH_HALF])


def in_ring(x, y):
    """True inside the fielding ring: two 30-yard semicircles joined by straight lines."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    near_striker = np.hypot(x, y + PITCH_HALF) <= RING_RADIUS
    near_bowler = np.hypot(x, y - PITCH_HALF) <= RING_RADIUS
    between = (np.abs(x) <= RING_RADIUS) & (np.abs(y) <= PITCH_HALF)
    return near_striker | near_bowler | between


def in_field(x, y, radius: float = BOUNDARY_RADIUS):
    return np.hypot(x, y) <= radius


def ring_outline(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Closed outline of the ring, for plotting."""
    t = np.linspace(0, np.pi, n)
    top_x, top_y = RING_RADIUS * np.cos(t), PITCH_HALF + RING_RADIUS * np.sin(t)
    bot_x, bot_y = -RING_RADIUS * np.cos(t), -PITCH_HALF - RING_RADIUS * np.sin(t)
    x = np.concatenate([top_x, bot_x, top_x[:1]])
    y = np.concatenate([top_y, bot_y, top_y[:1]])
    return x, y


def to_ground_frame(x, is_left_handed):
    """Flip x for left-handers so points land on the side of the ground they really went."""
    x = np.asarray(x, dtype=float)
    return np.where(np.asarray(is_left_handed, dtype=bool), -x, x)

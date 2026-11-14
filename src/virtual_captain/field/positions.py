"""Named fielding positions for a right-handed batter.

Each position is (angle, radius): the angle uses the wagon-wheel convention (0 straight behind
the striker, increasing through the leg side, so 90 is square leg and 270 is point) and the
radius is metres from the centre of the pitch. Left-handers use the same names in the
mirrored frame.
"""

import numpy as np

from virtual_captain.data.coords import wagon_to_xy
from virtual_captain.field.geometry import BOUNDARY_RADIUS

DEEP = 64.0  # boundary riders stand a few metres in from a 68 m rope

POSITIONS: dict[str, tuple[float, float]] = {
    # close catchers. The slip cordon and leg slip sit where Cricinfo's wagon wheel plots the
    # catches taken there (it draws edges closer to the bat than they really carry), so the
    # simulated edges land where these fielders stand.
    "first_slip": (344.1, 18.2),
    "second_slip": (334.8, 18.8),
    "third_slip": (326.7, 19.1),
    "gully": (312.9, 19.1),
    "leg_slip": (15.9, 18.2),
    "short_leg": (30, 8),
    "silly_point": (330, 8),
    "silly_mid_off": (210, 7),
    # inner ring
    "short_third_man": (335, 32),
    "backward_point": (305, 28),
    "point": (282, 26),
    "cover_point": (262, 25),
    "cover": (240, 26),
    "extra_cover": (215, 30),
    "mid_off": (195, 34),
    "mid_on": (165, 34),
    "mid_wicket": (115, 28),
    "square_leg": (80, 26),
    "backward_square_leg": (55, 28),
    "short_fine_leg": (25, 30),
    # boundary
    "third_man": (335, DEEP),
    "deep_backward_point": (300, DEEP),
    "deep_point": (282, DEEP),
    "deep_cover": (245, DEEP),
    "deep_extra_cover": (222, DEEP),
    "long_off": (198, DEEP),
    "long_on": (162, DEEP),
    "cow_corner": (138, DEEP),
    "deep_midwicket": (115, DEEP),
    "deep_square_leg": (86, DEEP),
    "deep_backward_square": (55, DEEP),
    "fine_leg": (25, DEEP),
}

# Not part of the nine: the keeper stands back to pace and up to the stumps for spin,
# and the bowler finishes their follow-through a couple of metres down the pitch.
KEEPER_BACK = (353.3, 17.1)  # where keepers' catches off pace are plotted
KEEPER_UP = (0, 11.5)
BOWLER = (180, 13.0)


def xy(name_or_polar) -> np.ndarray:
    """(x, y) in metres for a position name or an (angle, radius) pair."""
    angle, radius = POSITIONS[name_or_polar] if isinstance(name_or_polar, str) else name_or_polar
    x, y = wagon_to_xy(angle, radius / BOUNDARY_RADIUS)
    return np.array([float(x), float(y)])


def field_xy(names: list[str]) -> np.ndarray:
    """(9, 2) array of fielder coordinates from position names."""
    return np.stack([xy(n) for n in names])


def with_keeper_and_bowler(field: np.ndarray, keeper_up: bool) -> np.ndarray:
    """Nine fielders plus keeper and bowler: the (11, 2) array the simulator uses."""
    keeper = xy(KEEPER_UP if keeper_up else KEEPER_BACK)
    return np.vstack([field, keeper, xy(BOWLER)])


def nearest_name(point: np.ndarray) -> str:
    """Closest named position to a point, for labelling optimised fields."""
    names = list(POSITIONS)
    coords = field_xy(names)
    return names[int(np.argmin(np.hypot(*(coords - point).T)))]

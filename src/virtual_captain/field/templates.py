"""Conventional fields a captain would set, by format, phase and type of bowling.

These stand in for the (unrecorded) fields in the historical data when calibrating the
simulator, and are the baseline that optimised fields are compared against.
"""

import numpy as np

from virtual_captain.field.positions import field_xy

TEMPLATES: dict[tuple[str, str, str], list[str]] = {
    # T20
    ("T20", "powerplay", "pace"): [
        "first_slip", "backward_point", "cover", "mid_off", "mid_on", "mid_wicket",
        "square_leg", "third_man", "fine_leg",
    ],
    ("T20", "powerplay", "spin"): [
        "backward_point", "cover", "extra_cover", "mid_off", "mid_on", "mid_wicket",
        "short_fine_leg", "deep_square_leg", "long_on",
    ],
    ("T20", "middle", "pace"): [
        "point", "cover", "mid_off", "mid_wicket", "third_man", "deep_cover", "long_on",
        "deep_square_leg", "fine_leg",
    ],
    ("T20", "middle", "spin"): [
        "backward_point", "extra_cover", "mid_off", "short_fine_leg", "deep_cover",
        "long_off", "long_on", "deep_midwicket", "deep_square_leg",
    ],
    ("T20", "death", "pace"): [
        "point", "extra_cover", "mid_wicket", "short_fine_leg", "third_man", "deep_cover",
        "long_off", "long_on", "deep_square_leg",
    ],
    ("T20", "death", "spin"): [
        "point", "cover", "mid_off", "short_fine_leg", "deep_cover", "long_off", "long_on",
        "deep_midwicket", "deep_square_leg",
    ],
    # ODI
    ("ODI", "powerplay", "pace"): [
        "first_slip", "second_slip", "point", "cover", "mid_off", "mid_on", "mid_wicket",
        "third_man", "fine_leg",
    ],
    ("ODI", "powerplay", "spin"): [
        "point", "cover", "extra_cover", "mid_off", "mid_on", "mid_wicket", "short_fine_leg",
        "deep_square_leg", "long_on",
    ],
    ("ODI", "middle", "pace"): [
        "first_slip", "point", "cover", "mid_off", "mid_wicket", "third_man", "deep_cover",
        "long_on", "fine_leg",
    ],
    ("ODI", "middle", "spin"): [
        "point", "cover", "mid_off", "mid_on", "short_fine_leg", "deep_cover", "long_off",
        "long_on", "deep_midwicket",
    ],
    ("ODI", "death", "pace"): [
        "point", "extra_cover", "mid_wicket", "short_fine_leg", "third_man", "deep_cover",
        "long_off", "long_on", "deep_square_leg",
    ],
    ("ODI", "death", "spin"): [
        "point", "cover", "mid_off", "short_fine_leg", "deep_cover", "long_off", "long_on",
        "deep_midwicket", "deep_square_leg",
    ],
    # Test
    ("TEST", "new_ball", "pace"): [
        "first_slip", "second_slip", "third_slip", "gully", "point", "cover", "mid_off",
        "mid_on", "fine_leg",
    ],
    ("TEST", "new_ball", "spin"): [
        "first_slip", "short_leg", "silly_point", "point", "cover", "mid_off", "mid_on",
        "mid_wicket", "deep_square_leg",
    ],
    ("TEST", "old_ball", "pace"): [
        "first_slip", "gully", "point", "cover", "mid_off", "mid_on", "deep_square_leg",
        "fine_leg", "third_man",
    ],
    ("TEST", "old_ball", "spin"): [
        "first_slip", "short_leg", "point", "cover", "mid_off", "mid_wicket", "deep_point",
        "long_on", "deep_square_leg",
    ],
}  # fmt: skip

KEYS = list(TEMPLATES)


def template(fmt: str, phase: str, pace_or_spin: str | None) -> np.ndarray:
    kind = "spin" if pace_or_spin == "spin" else "pace"
    return field_xy(TEMPLATES[(fmt, phase, kind)])


def template_index(fmt: str, phase: str, pace_or_spin: str | None) -> int:
    kind = "spin" if pace_or_spin == "spin" else "pace"
    return KEYS.index((fmt, phase, kind))

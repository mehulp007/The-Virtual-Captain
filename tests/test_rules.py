import numpy as np
import pytest

from virtual_captain.field import rules
from virtual_captain.field.geometry import in_ring
from virtual_captain.field.positions import POSITIONS, field_xy, xy
from virtual_captain.field.templates import TEMPLATES

RING_POSITIONS = [
    "short_third_man", "backward_point", "point", "cover_point", "cover", "extra_cover",
    "mid_off", "mid_on", "mid_wicket", "square_leg", "backward_square_leg", "short_fine_leg",
]  # fmt: skip


@pytest.mark.parametrize("name", list(POSITIONS))
def test_positions_sit_where_their_names_say(name):
    x, y = xy(name)
    deep = POSITIONS[name][1] > 60
    assert bool(in_ring(x, y)) is not deep
    angle = POSITIONS[name][0]
    if 10 < angle < 170:
        assert x < 0, "leg-side positions should be on the leg side"
    if 190 < angle < 350:
        assert x > 0, "off-side positions should be on the off side"


@pytest.mark.parametrize("name", RING_POSITIONS)
def test_ring_positions_are_inside_the_ring(name):
    assert in_ring(*xy(name))


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_every_template_is_legal(key):
    fmt, phase, _ = key
    names = TEMPLATES[key]
    assert len(names) == 9 and len(set(names)) == 9
    check = rules.check(field_xy(names), fmt, phase)
    assert check.legal, check.problems


@pytest.mark.parametrize(
    ("fmt", "phase", "limit"),
    [("T20", "powerplay", 2), ("T20", "death", 5), ("ODI", "middle", 4), ("TEST", "new_ball", 9)],
)
def test_ring_limits(fmt, phase, limit):
    assert rules.max_outside_ring(fmt, phase) == limit


def _all_on_the_boundary():
    names = [
        "third_man", "deep_point", "deep_cover", "long_off", "long_on", "deep_midwicket",
        "deep_square_leg", "deep_backward_square", "fine_leg",
    ]  # fmt: skip
    return field_xy(names)


def test_repair_brings_in_just_enough_fielders():
    field = _all_on_the_boundary()
    assert not rules.check(field, "T20", "powerplay").legal
    fixed = rules.repair(field, "T20", "powerplay")
    check = rules.check(fixed, "T20", "powerplay")
    assert check.legal, check.problems
    assert check.outside_ring == 2
    moved = np.hypot(*(fixed - field).T) > 1e-9
    assert moved.sum() == 7


def test_repair_leaves_a_legal_field_alone():
    field = field_xy(TEMPLATES[("ODI", "middle", "pace")])
    np.testing.assert_allclose(rules.repair(field, "ODI", "middle"), field)


def test_repair_fixes_too_many_behind_square_on_the_leg_side():
    names = [
        "leg_slip", "short_fine_leg", "fine_leg", "backward_square_leg", "point", "cover",
        "mid_off", "mid_on", "mid_wicket",
    ]  # fmt: skip
    field = field_xy(names)
    assert rules.check(field, "TEST", "old_ball").leg_behind_square == 4
    fixed = rules.repair(field, "TEST", "old_ball")
    assert rules.check(fixed, "TEST", "old_ball").legal


def test_repair_handles_random_fields():
    rng = np.random.default_rng(0)
    for fmt, phase in [("T20", "powerplay"), ("ODI", "middle"), ("TEST", "new_ball")]:
        for _ in range(50):
            angle = rng.uniform(0, 2 * np.pi, 9)
            r = rng.uniform(0, 80, 9)
            field = np.column_stack([np.sin(angle) * r, np.cos(angle) * r])
            check = rules.check(rules.repair(field, fmt, phase), fmt, phase)
            assert check.legal, check.problems

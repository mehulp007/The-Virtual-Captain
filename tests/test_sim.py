import math

import numpy as np
import pytest

from virtual_captain.field.positions import field_xy, with_keeper_and_bowler
from virtual_captain.field.templates import TEMPLATES
from virtual_captain.sim import physics
from virtual_captain.sim.engine import Shots, simulate, summary
from virtual_captain.sim.params import SimParams

PARAMS = SimParams()
FAR_AWAY = np.full((11, 2), 500.0)


def _shots(theta_deg, dist, aerial=False, outcome=2):
    theta_deg, dist = np.atleast_1d(theta_deg), np.atleast_1d(dist)
    n = max(len(theta_deg), len(dist))
    return Shots(
        outcome=np.full(n, outcome, dtype=np.int8),
        theta=np.broadcast_to(np.deg2rad(theta_deg), (n,)).astype(np.float64).copy(),
        dist=np.broadcast_to(dist, (n,)).astype(np.float64).copy(),
        aerial=np.full(n, aerial, dtype=np.bool_),
    )


def _random_shots(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return Shots(
        outcome=np.full(n, 2, dtype=np.int8),
        theta=rng.uniform(0, 2 * math.pi, n),
        dist=rng.uniform(0.1, 1.1, n),
        aerial=rng.random(n) < 0.1,
    )


def _field(names, keeper_up=False):
    return with_keeper_and_bowler(field_xy(names), keeper_up)


RING = _field(TEMPLATES[("T20", "powerplay", "pace")])
SPREAD = _field(TEMPLATES[("T20", "middle", "pace")])


def test_same_seed_same_result():
    shots = _random_shots()
    a = simulate(shots, SPREAD, params=PARAMS, seed=3)
    b = simulate(shots, SPREAD, params=PARAMS, seed=3)
    np.testing.assert_array_equal(a.runs, b.runs)
    np.testing.assert_array_equal(a.kind, b.kind)


def test_empty_field_concedes_far_more():
    shots = _random_shots()
    empty = summary(simulate(shots, FAR_AWAY, params=PARAMS))
    real = summary(simulate(shots, SPREAD, params=PARAMS))
    assert empty["caught"] == 0
    assert empty["runs_per_ball"] > 2 * real["runs_per_ball"]
    assert empty["dot"] < real["dot"]


def test_boundary_riders_cut_fours_but_give_twos():
    shots = _random_shots()
    ring = summary(simulate(shots, RING, params=PARAMS))
    spread = summary(simulate(shots, SPREAD, params=PARAMS))
    assert spread["four"] < ring["four"]
    assert spread["two_three"] > ring["two_three"]


def test_ball_hit_straight_at_mid_off_is_stopped():
    field = _field(["mid_off"] * 9)
    result = simulate(_shots(195, 0.8), field, params=PARAMS)
    assert result.kind[0] == physics.FIELDED
    assert result.runs[0] <= 1


def test_same_shot_through_an_empty_off_side_goes_for_four():
    field = _field(["mid_wicket"] * 9)
    result = simulate(_shots(195, 0.98), field, params=PARAMS)
    assert result.kind[0] == physics.FOUR


def test_catch_is_taken_when_a_fielder_is_underneath():
    # long-on is at (162 deg, 64 m); a skier landing right there
    field = _field(["long_on"] * 9)
    params = SimParams(catch_max=1.0)
    uniforms = np.zeros((1, 4))
    result = simulate(_shots(162, 64 / 68, aerial=True), field, params=params, uniforms=uniforms)
    assert result.kind[0] == physics.CAUGHT
    assert result.wicket[0]


def test_clearing_the_rope_is_six():
    result = simulate(_shots(160, 1.08, aerial=True), SPREAD, params=PARAMS)
    assert result.kind[0] == physics.SIX
    assert result.runs[0] == 6


@pytest.mark.parametrize(
    ("outcome", "kind"), [(0, physics.NO_CONTACT), (1, physics.FIELD_FREE_OUT)]
)
def test_balls_not_in_play_skip_the_field(outcome, kind):
    result = simulate(_shots(180, 0.5, outcome=outcome), SPREAD, params=PARAMS)
    assert result.kind[0] == kind
    assert result.runs[0] == 0
    assert bool(result.wicket[0]) is (outcome == 1)


def test_six_from_the_shot_model_is_six():
    result = simulate(_shots(180, 0.5, outcome=3), SPREAD, params=PARAMS)
    assert result.kind[0] == physics.SIX
    assert result.runs[0] == 6


def test_runs_are_valid_cricket_scores():
    result = simulate(_random_shots(), SPREAD, params=PARAMS)
    assert set(np.unique(result.runs)) <= {0, 1, 2, 3, 4, 6}

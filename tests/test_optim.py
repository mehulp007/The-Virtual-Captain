import math

import numpy as np
import polars as pl
import pytest

from virtual_captain.field import rules
from virtual_captain.field.positions import POSITIONS, field_xy, xy
from virtual_captain.field.templates import TEMPLATES
from virtual_captain.optim.cma_search import decode, encode, optimise
from virtual_captain.optim.heuristics import analyst_field
from virtual_captain.optim.objective import Scenario, cost, paired_difference
from virtual_captain.optim.sensitivity import STRIKER, _directions, best_moves
from virtual_captain.sim.engine import Shots
from virtual_captain.sim.params import SimParams


def _scenario(theta_deg, n=4000, dist=0.8, fmt="T20", phase="middle", seed=0):
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(rng.normal(theta_deg, 6, n)) % (2 * math.pi)
    shots = Shots(
        outcome=np.full(n, 2, dtype=np.int8),
        theta=theta,
        dist=np.clip(rng.normal(dist, 0.1, n), 0.05, 1.2),
        aerial=np.zeros(n, dtype=np.bool_),
    )
    return Scenario(
        fmt=fmt,
        phase=phase,
        bowling="pace",
        wicket_value=8.0,
        shots=shots,
        uniforms=rng.random((n, 4)),
        params=SimParams(),
    )


def test_encode_decode_round_trip():
    field = field_xy(TEMPLATES[("ODI", "middle", "pace")])
    np.testing.assert_allclose(decode(encode(field)), field, atol=1e-6)


def test_analyst_keeps_the_number_of_boundary_riders():
    key = ("T20", "death", "pace")
    rng = np.random.default_rng(1)
    shots = pl.DataFrame({"theta": rng.uniform(0, 2 * math.pi, 3000), "dist": rng.random(3000)})
    field = analyst_field(shots, *key)
    assert rules.check(field, "T20", "death").legal
    n_deep = sum(POSITIONS[n][1] > 60 for n in TEMPLATES[key])
    assert int(rules.outside_ring(field).sum()) == n_deep


def test_identical_fields_cost_the_same():
    scen = _scenario(115)
    field = field_xy(TEMPLATES[("T20", "middle", "pace")])
    mean, lo, hi = paired_difference(field, field, scen)
    assert mean == 0 and lo == 0 and hi == 0


def test_optimiser_covers_the_one_place_the_batter_hits():
    # every shot goes along the ground to deep mid-wicket; the conventional field has a rider
    # at long on and deep square leg but nobody in between
    scen = _scenario(120)
    start = field_xy(TEMPLATES[("T20", "middle", "pace")])
    result = optimise(scen, start, generations=25, population=12)
    assert result.cost < result.start_cost
    assert rules.check(result.field, "T20", "middle").legal
    # some fielder now stands on the line the ball travels along (anywhere on it will do)
    direction = xy((120, 60)) - STRIKER
    direction /= np.linalg.norm(direction)
    rel = result.field - STRIKER
    along = rel @ direction
    across = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
    assert np.any((along > 10) & (across < 6))


def test_cost_counts_wickets_at_their_value():
    scen = _scenario(162, dist=64 / 68)
    scen.shots.aerial[:] = True
    catcher = field_xy(["long_on"] * 9)
    nobody = np.full((9, 2), 300.0)
    c_catch = cost(catcher, scen, legalise=False)[0]
    c_none = cost(nobody, scen, legalise=False)[0]
    assert c_catch < c_none - 1.0  # wickets at 8 runs each make the catcher far better


@pytest.mark.parametrize(
    ("position", "sideways"),
    [("cover", {"straighter", "squarer"}), ("third_man", {"squarer", "finer"})],
)
def test_move_names(position, sideways):
    p = xy(position)
    moves = _directions(p, 5.0)
    assert set(moves) == {"in", "out"} | sideways
    assert np.hypot(*(p + 5 * moves["in"] - STRIKER)) < np.hypot(*(p - STRIKER))


def test_best_moves_reports_intervals():
    scen = _scenario(115, n=2000)
    field = field_xy(TEMPLATES[("T20", "middle", "pace")])
    moves = best_moves(field, scen)
    assert moves.height > 0
    assert (moves["runs_ci"] >= 0).all()
    assert moves["cost_change"].is_sorted()


def test_resource_curve_fit_recovers_known_values():
    from virtual_captain.optim.value import _fit_resources, _resource_curve

    f_true = np.array([1.0, 0.93, 0.85, 0.75, 0.64, 0.52, 0.4, 0.28, 0.17, 0.08])
    o = np.repeat(np.arange(1, 51), 10).astype(float)
    w = np.tile(np.arange(10), 50)
    r = _resource_curve(o, w, 280.0, 9.0, f_true)
    fitted = _fit_resources(pl.DataFrame({"overs_left": o, "wickets_before": w, "r": r, "n": 100}))
    np.testing.assert_allclose(_resource_curve(o, w, *fitted), r, rtol=1e-3, atol=0.05)
    assert np.all(np.diff(fitted[2]) < 0), "fewer wickets in hand, fewer runs to come"

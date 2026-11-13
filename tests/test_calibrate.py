import numpy as np
import pytest

from virtual_captain.data.coords import xy_to_wagon
from virtual_captain.field import rules
from virtual_captain.field.templates import TEMPLATES
from virtual_captain.sim.calibrate import PROFILE_SECTORS, adapted_field


def _profile(*sectors):
    w = np.zeros(PROFILE_SECTORS)
    for s in sectors:
        w[s] = 100.0
    return w


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_adapted_fields_stay_legal(key):
    rng = np.random.default_rng(0)
    for _ in range(10):
        field = adapted_field(key, rng.random(PROFILE_SECTORS))
        assert field.shape == (11, 2)
        check = rules.check(field[:9], key[0], key[1])
        assert check.legal, check.problems


def test_riders_go_where_the_batter_scores():
    # T20 middle overs, pace: five boundary riders
    key = ("T20", "middle", "pace")
    busiest = [7, 10, 13, 16, 19]  # 105, 150, 195, 240, 285 degrees
    field = adapted_field(key, _profile(*busiest))
    angle, dist = xy_to_wagon(field[:9, 0], field[:9, 1])
    deep_angles = np.sort(angle[dist > 0.9])
    expected = np.sort([(s + 0.5) * 15 for s in busiest])
    np.testing.assert_allclose(deep_angles, expected, atol=1e-6)


def test_no_profile_means_plain_template():
    key = ("ODI", "middle", "spin")
    a = adapted_field(key, None)
    b = adapted_field(key, None)
    np.testing.assert_allclose(a, b)

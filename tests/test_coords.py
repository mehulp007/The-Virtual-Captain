import numpy as np
import polars as pl
import pytest

from virtual_captain.data.coords import wagon_to_xy, wagon_xy_exprs, xy_to_wagon
from virtual_captain.field.geometry import (
    BOUNDARY_RADIUS,
    PITCH_HALF,
    RING_RADIUS,
    in_ring,
    to_ground_frame,
)


@pytest.mark.parametrize(
    ("angle", "sign_x", "sign_y"),
    [
        (0, 0, -1),  # straight behind the batter
        (90, -1, 0),  # square leg
        (180, 0, 1),  # straight down the ground
        (270, 1, 0),  # point
        (115, -1, 1),  # mid-wicket: leg side, in front of square
        (245, 1, 1),  # cover: off side, in front of square
        (335, 1, -1),  # third man: off side, behind square
    ],
)
def test_wagon_angle_orientation(angle, sign_x, sign_y):
    x, y = wagon_to_xy(angle, 1.0)
    assert np.sign(np.round(x, 6)) == sign_x
    assert np.sign(np.round(y, 6)) == sign_y


def test_distance_is_fraction_of_boundary():
    x, y = wagon_to_xy(200.0, 0.5)
    assert np.hypot(x, y) == pytest.approx(0.5 * BOUNDARY_RADIUS)


def test_round_trip():
    angles = np.array([1.0, 45.0, 120.0, 200.0, 359.0])
    dists = np.array([0.1, 0.5, 0.9, 1.0, 1.2])
    a, d = xy_to_wagon(*wagon_to_xy(angles, dists))
    np.testing.assert_allclose(a, angles, atol=1e-9)
    np.testing.assert_allclose(d, dists, atol=1e-9)


def test_polars_exprs_match_numpy():
    df = pl.DataFrame({"wagon_angle": [30.0, 250.0], "wagon_distance": [0.4, 1.0]})
    out = df.with_columns(*wagon_xy_exprs())
    x, y = wagon_to_xy(df["wagon_angle"].to_numpy(), df["wagon_distance"].to_numpy())
    np.testing.assert_allclose(out["x_m"].to_numpy(), x)
    np.testing.assert_allclose(out["y_m"].to_numpy(), y)


def test_left_hander_cover_drive_goes_to_leg_side_of_ground():
    x, _ = wagon_to_xy(245, 0.8)  # cover in the mirrored frame
    assert x > 0
    assert to_ground_frame(x, True) < 0
    assert to_ground_frame(x, False) == pytest.approx(x)


def test_mirroring_twice_is_identity():
    x = np.array([-12.0, 0.0, 30.5])
    np.testing.assert_allclose(to_ground_frame(to_ground_frame(x, True), True), x)


@pytest.mark.parametrize(
    ("x", "y", "inside"),
    [
        (0, 0, True),
        (RING_RADIUS - 0.5, 0, True),  # square, just inside
        (RING_RADIUS + 0.5, 0, False),  # square, just outside
        (0, PITCH_HALF + RING_RADIUS - 0.5, True),  # straight, the ring bulges out
        (0, -(PITCH_HALF + RING_RADIUS) - 0.5, False),  # fine, just beyond the ring
        (25, 25, False),  # 35 m away at 45 degrees
        (20, 20, True),
    ],
)
def test_in_ring(x, y, inside):
    assert bool(in_ring(x, y)) is inside

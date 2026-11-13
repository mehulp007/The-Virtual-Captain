import math

import polars as pl

from virtual_captain.models import encoding as enc


def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = {
        "has_wagon": [False] * n,
        "striker_out": [False] * n,
        "wicket_type": [None] * n,
        "runs_off_bat": [0] * n,
        "is_six": [False] * n,
        "ball_events": [None] * n,
        "wagon_angle": [None] * n,
        "wagon_distance": [None] * n,
    }
    base.update(cols)
    return pl.DataFrame(base, schema_overrides={"wagon_angle": pl.Float64})


def test_contact_outcomes():
    df = _rows(
        has_wagon=[False, False, True, False, False, False],
        striker_out=[False, True, False, True, False, True],
        wicket_type=[None, "lbw", None, "caught", None, "run out"],
        runs_off_bat=[0, 0, 2, 0, 1, 0],
        wagon_angle=[None, None, 120.0, None, None, None],
        wagon_distance=[None, None, 0.6, None, None, None],
    )
    out = enc.add_labels(df.lazy()).collect()
    # caught / run out without a wagon point and untracked scoring shots are dropped
    assert out["outcome"].to_list() == [enc.NO_CONTACT, enc.FIELD_FREE_OUT, enc.IN_PLAY]


def test_aerial_label_and_polar_shot():
    df = _rows(
        has_wagon=[True, True, True, True],
        striker_out=[False, True, False, False],
        wicket_type=[None, "caught", None, None],
        is_six=[True, False, False, False],
        ball_events=[None, None, "dropped catch", None],
        wagon_angle=[180.0, 10.0, 90.0, 270.0],
        wagon_distance=[1.05, 0.2, 2.0, 0.0001],
    )
    out = enc.add_labels(df.lazy()).collect()
    assert out["aerial"].to_list() == [True, True, True, False]
    assert out["theta"][0] == math.pi
    assert out["dist"].to_list() == [1.05, 0.2, enc.D_MAX, enc.D_MIN]


def test_holdout_split_by_year():
    df = pl.DataFrame({"year": [2019, 2024, 2025, 2026], "match_id": [1, 2, 3, 4]})
    splits = enc.split_column(df.lazy()).collect()["split"].to_list()
    assert splits[2:] == ["test", "test"]
    assert set(splits[:2]) <= {"train", "val"}


def test_encode_shapes_and_unknown_players():
    df = pl.DataFrame(
        {
            "format": ["T20", "TEST"],
            "phase": ["death", "new_ball"],
            "bowler_kind": ["fast", None],
            "bowler_arm": ["right", None],
            "spin_turn": [None, None],
            "ball_line": ["outside offstump", None],
            "ball_length": ["yorker", "something new"],
            "left_handed": [True, None],
            "competition_type": ["Fran(IPL)", "Intl"],
            "year": [2023, 2018],
            "wickets_before": [3, 0],
            "balls_before": [100, 30],
            "balls_remaining": [20, None],
            "ball_age": [100, 30],
            "target": [180, None],
            "runs_required": [40, None],
            "runs_before": [140, 12],
            "striker_balls_faced": [30, 0],
            "innings": [2, 3],
            "arm_vs_bat": ["opposite", None],
            "venue_runs_per_ball": [1.3, None],
            "venue_boundary_rate": [0.15, None],
            "venue_spin_wicket_share": [0.3, None],
            "striker_id": ["known", "stranger"],
            "bowler_id": ["b1", None],
        }
    )
    vocab = enc.Vocab(
        batters=["known"], bowlers=["b1"], venue_mean=[1.0, 0.1, 0.35], venue_std=[0.3, 0.04, 0.1]
    )
    e = enc.encode(df, vocab)
    assert e["x"].shape == (2, enc.n_features())
    assert e["batter"].tolist() == [1, 0]
    assert e["bowler"].tolist() == [1, 0]
    assert not (e["x"] != e["x"]).any()  # no NaNs
    # each one-hot block has exactly one 1 per row
    start = len(enc.NUMERIC) + len(enc.VENUE_COLS) + 1
    for values in enc.CATEGORIES.values():
        block = e["x"][:, start : start + len(values)]
        assert (block.sum(axis=1) == 1).all()
        start += len(values)

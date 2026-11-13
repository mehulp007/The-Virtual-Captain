from datetime import date

import polars as pl
import pytest

from virtual_captain.data import features
from virtual_captain.models.context import (
    add_bowler_context,
    parse_bowling_style,
    venue_key,
)


def _innings(fmt, innings, runs, legal=None, wickets=None, bpo=6, match_id=1):
    n = len(runs)
    legal = legal or [True] * n
    wickets = wickets or [None] * n
    return {
        "match_id": [match_id] * n,
        "innings": [innings] * n,
        "delivery_number": list(range(1, n + 1)),
        "format": [fmt] * n,
        "runs_off_bat": runs,
        "extras": [0 if lg else 1 for lg in legal],
        "non_boundary": [False] * n,
        "wicket_type": wickets,
        "other_wicket_type": [None] * n,
        "player_dismissed": ["A" if w else None for w in wickets],
        "striker": ["A"] * n,
        "is_legal": legal,
        "balls_per_over": [bpo] * n,
        "start_date": [date(2024, 5, 1)] * n,
    }


def _frame(*parts):
    cols = {}
    for p in parts:
        for k, v in p.items():
            cols.setdefault(k, []).extend(v)
    return pl.DataFrame(cols).lazy()


def _build(lf):
    return (
        lf.pipe(features.drop_super_overs)
        .pipe(features.add_ball_flags)
        .pipe(features.add_match_state)
        .pipe(features.add_phase)
        .pipe(features.add_era)
        .collect()
    )


def test_state_is_measured_before_the_ball():
    df = _build(
        _frame(
            _innings(
                "T20",
                1,
                [4, 0, 6, 1],
                legal=[True, False, True, True],
                wickets=[None, None, "bowled", None],
            )
        )
    )
    assert df["runs_before"].to_list() == [0, 4, 5, 11]  # wide adds an extra run
    assert df["balls_before"].to_list() == [0, 1, 1, 2]  # wide does not count
    assert df["wickets_before"].to_list() == [0, 0, 0, 1]
    assert df["is_six"].to_list() == [False, False, True, False]


def test_retired_hurt_is_not_a_wicket():
    df = _build(_frame(_innings("ODI", 1, [0, 0], wickets=["retired hurt", None])))
    assert df["wickets_before"].to_list() == [0, 0]
    assert df["striker_out"].to_list() == [False, False]


def test_chase_target_and_super_overs():
    first = _innings("T20", 1, [6, 6, 1])
    second = _innings("T20", 2, [1, 2])
    super_over = _innings("T20", 3, [4])
    df = _build(_frame(first, second, super_over))
    assert set(df["innings"]) == {1, 2}
    chase = df.filter(pl.col("innings") == 2)
    assert chase["target"].to_list() == [14, 14]
    assert chase["runs_required"].to_list() == [14, 13]
    assert df.filter(pl.col("innings") == 1)["target"].null_count() == 3


@pytest.mark.parametrize(
    ("fmt", "bpo", "balls", "expected"),
    [
        ("T20", 6, 35, "powerplay"),
        ("T20", 6, 36, "middle"),
        ("T20", 6, 90, "death"),
        ("T20", 5, 25, "middle"),  # The Hundred: 25-ball powerplay
        ("T20", 5, 75, "death"),
        ("ODI", 6, 59, "powerplay"),
        ("ODI", 6, 240, "death"),
        ("TEST", 6, 119, "new_ball"),
        ("TEST", 6, 120, "old_ball"),
        ("TEST", 6, 480, "new_ball"),  # second new ball after 80 overs
    ],
)
def test_phase(fmt, bpo, balls, expected):
    lf = pl.DataFrame({"format": [fmt], "balls_before": [balls], "balls_per_over": [bpo]}).lazy()
    assert features.add_phase(lf).collect()["phase"].item() == expected


@pytest.mark.parametrize(
    ("style", "arm", "kind"),
    [
        ("right-arm fast", "right", "fast"),
        ("left-arm fast-medium, slow left-arm orthodox", "left", "fast"),
        ("right-arm medium-fast", "right", "fast"),
        ("right-arm slow-medium", "right", "medium"),
        ("right-arm offbreak", "right", "finger_spin"),
        ("slow left-arm orthodox", "left", "finger_spin"),
        ("legbreak googly", "right", "wrist_spin"),
        ("left-arm wrist-spin", "left", "wrist_spin"),
        ("right-arm bowler", "right", "unknown"),
        (None, None, "unknown"),
    ],
)
def test_parse_bowling_style(style, arm, kind):
    out = parse_bowling_style(style)
    assert out["bowler_arm"] == arm
    assert out["bowler_kind"] == kind


@pytest.mark.parametrize(
    ("style", "bat", "turn", "arm_vs_bat"),
    [
        ("right-arm offbreak", "right-hand bat", "in", "same"),
        ("right-arm offbreak", "left-hand bat", "away", "opposite"),
        ("legbreak", "right-hand bat", "away", "same"),
        ("slow left-arm orthodox", "right-hand bat", "away", "opposite"),
        ("left-arm wrist-spin", "right-hand bat", "in", "opposite"),
        ("left-arm fast", "left-hand bat", None, "same"),
    ],
)
def test_spin_turn(style, bat, turn, arm_vs_bat):
    lf = pl.DataFrame({"bowling_style": [style], "batting_style": [bat]}).lazy()
    out = add_bowler_context(lf, pl.Series([style])).collect()
    assert out["spin_turn"].item() == turn
    assert out["arm_vs_bat"].item() == arm_vs_bat


def test_venue_key_merges_city_suffix():
    s = pl.Series(["Wankhede Stadium, Mumbai", "Wankhede Stadium", " Lord's , London"])
    keys = pl.select(venue_key(pl.lit(s))).to_series().to_list()
    assert keys == ["wankhede stadium", "wankhede stadium", "lord's"]

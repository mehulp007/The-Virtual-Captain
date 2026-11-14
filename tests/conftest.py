"""Shared fixtures: a small synthetic version of the captain's data."""

import numpy as np
import polars as pl
import pytest

from virtual_captain.field.templates import KEYS, template
from virtual_captain.rl.data import Bank, CaptainData, profile
from virtual_captain.sim.engine import Shots


def _shots(rng, n, angle=None):
    outcome = rng.choice([0, 1, 2, 3], size=n, p=[0.3, 0.03, 0.62, 0.05]).astype(np.int8)
    theta = rng.uniform(0, 2 * np.pi, n) if angle is None else rng.normal(angle, 0.2, n)
    return Shots(
        outcome=outcome,
        theta=np.mod(theta, 2 * np.pi),
        dist=rng.uniform(0.05, 1.0, n),
        aerial=rng.random(n) < 0.15,
    )


@pytest.fixture(scope="session")
def data():
    """16 all-batters contexts plus two named batters (one unseen) in every situation."""
    rng = np.random.default_rng(0)
    rows, learn, judge = [], [], []
    specs = [("all batters", k) for k in KEYS] + [(b, k) for b in ("A", "B") for k in KEYS]
    for c, (who, key) in enumerate(specs):
        angle = {"all batters": None, "A": 1.5, "B": 4.5}[who]
        learn.append(_shots(rng, 600, angle))
        judge.append(_shots(rng, 600, angle))
        field = template(*key).ravel().tolist()
        rows.append(
            {
                "context": c, "batter": who, "batter_id": None if who == "all batters" else who,
                "format": key[0], "phase": key[1], "bowling": key[2], "key": KEYS.index(key),
                "balls": 1000, "pooled": who == "all batters", "unseen": who == "B",
                "wickets_median": 3.0, "balls_remaining_median": 60.0, "ball_age_median": 200.0,
                "profile": profile(learn[-1]).tolist(), "searched": field, "analyst": field,
            }
        )  # fmt: skip
    keys = pl.DataFrame(
        {
            "key": list(range(16)),
            "format": [k[0] for k in KEYS],
            "phase": [k[1] for k in KEYS],
            "bowling": [k[2] for k in KEYS],
            "wicket_value": [10.0] * 16,
            "conventional": [template(*k).ravel().tolist() for k in KEYS],
            "optimised": [template(*k).ravel().tolist() for k in KEYS],
        }
    )
    starts = pl.DataFrame(
        {
            "format": ["T20", "ODI", "TEST"],
            "key": [
                KEYS.index(("T20", "middle", "spin")),
                KEYS.index(("ODI", "death", "pace")),
                KEYS.index(("TEST", "old_ball", "pace")),
            ],  # fmt: skip
            "balls_remaining": [60, 30, 0],
            "wickets_before": [3, 6, 2],
            "ball_age": [60, 270, 200],
        }
    )
    lim = [
        (f, o, w) for f, n in (("T20", 20), ("ODI", 50)) for o in range(1, n + 1) for w in range(10)
    ]
    states = pl.concat(
        [
            pl.DataFrame(
                {
                    "format": [f for f, _, _ in lim],
                    "overs_left": [o for _, o, _ in lim],
                    "wickets_before": [w for _, _, w in lim],
                    "wicket_value": [o * 0.5 for _, o, _ in lim],
                }
            ),  # fmt: skip
            pl.DataFrame(
                {
                    "format": ["TEST"] * 10,
                    "overs_left": pl.Series([None] * 10, dtype=pl.Int64),
                    "wickets_before": list(range(10)),
                    "wicket_value": [30.0] * 10,
                }
            ),  # fmt: skip
        ]
    )
    return CaptainData(
        contexts=pl.DataFrame(rows),
        keys=keys,
        starts=starts,
        wicket_states=states,
        learn=Bank.from_shots(learn),
        judge=Bank.from_shots(judge),
    )

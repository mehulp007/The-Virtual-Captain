import numpy as np
import polars as pl
import pytest
from gymnasium.utils.env_checker import check_env

from virtual_captain.field import rules
from virtual_captain.field.templates import KEYS, template
from virtual_captain.rl.data import Bank, CaptainData, profile
from virtual_captain.rl.env import (
    N_OBS,
    CaptainEnv,
    CaptainVecEnv,
    Overs,
    decode_fields,
    encode_fields,
)
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


@pytest.fixture(scope="module")
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


def test_gymnasium_checker_passes(data):
    check_env(CaptainEnv(data, shots_per_ball=32), skip_render_check=True)


def test_field_encoding_round_trip():
    fields = np.stack([template(*k) for k in KEYS])
    np.testing.assert_allclose(decode_fields(encode_fields(fields)), fields, atol=1e-6)


def test_no_move_is_the_starting_field(data):
    overs = Overs(data, n=8, shots_per_ball=16)
    fields = overs.fields_from_actions(np.zeros((8, 18)))
    np.testing.assert_allclose(fields, data.pooled_fields[overs.key], atol=1e-6)


def test_every_action_gives_a_legal_field(data):
    overs = Overs(data, n=200, shots_per_ball=16)
    actions = np.random.default_rng(1).uniform(-1, 1, (200, 18))
    for f, k in zip(overs.fields_from_actions(actions), overs.key, strict=True):
        fmt, phase, _ = KEYS[k]
        assert rules.check(f, fmt, phase).legal


def test_overs_last_six_balls_and_spells_twenty_four(data):
    for fmt, balls in (("T20", 6), ("ODI", 6), ("TEST", 24)):
        env = CaptainEnv(data, formats=(fmt,), shots_per_ball=16)
        env.reset(seed=0)
        played, done = 0, False
        while not done:
            _, _, done, _, _ = env.step(np.zeros(18))
            played += 1
        assert played == balls or env.overs.wickets[0] >= 10


def test_same_seed_same_over(data):
    def play(seed):
        env = CaptainEnv(data, shots_per_ball=32)
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for _ in range(6):
            obs, reward, *_ = env.step(np.full(18, 0.3))
            out += [obs, [reward]]
        return np.concatenate(out)

    np.testing.assert_array_equal(play(3), play(3))
    assert not np.array_equal(play(3), play(4))


def test_strike_changes_on_odd_runs_and_at_the_end_of_the_over(data):
    overs = Overs(data, n=400, formats=("T20",), shots_per_ball=8, seed=2)
    for _ in range(5):  # play to the last ball of the over
        striker, other = overs.striker.copy(), overs.other.copy()
        _, _, info = overs.step(overs.fields_from_actions(np.zeros((400, 18))))
        kept = ~info["wicket"]
        odd = kept & (info["runs"] % 2 == 1)
        even = kept & (info["runs"] % 2 == 0)
        np.testing.assert_array_equal(overs.striker[odd], other[odd])
        np.testing.assert_array_equal(overs.striker[even], striker[even])
    striker, other = overs.striker.copy(), overs.other.copy()
    _, _, info = overs.step(overs.fields_from_actions(np.zeros((400, 18))))
    even = ~info["wicket"] & (info["runs"] % 2 == 0)
    np.testing.assert_array_equal(overs.striker[even], other[even])  # ends changed


def test_unseen_batters_stay_out_of_training(data):
    unseen = set(data.contexts.filter(pl.col("unseen"))["context"])
    overs = Overs(data, n=500, generic_share=0.0, shots_per_ball=8)
    for _ in range(6):
        overs.step(overs.fields_from_actions(np.zeros((500, 18))))
        assert not unseen & set(overs.striker) and not unseen & set(overs.other)
    judged = Overs(data, n=500, batters="unseen", generic_share=0.0, shots_per_ball=8)
    assert set(judged.striker) <= unseen


def test_vec_env_restarts_finished_overs(data):
    env = CaptainVecEnv(data, n_envs=16, formats=("T20",), shots_per_ball=16)
    obs = env.reset()
    assert obs.shape == (16, N_OBS)
    for _ in range(6):
        env.step_async(np.zeros((16, 18), dtype=np.float32))
        obs, reward, done, infos = env.step_wait()
    assert done.all()
    assert all("terminal_observation" in i and i["episode"]["l"] == 6 for i in infos)
    assert (env.overs.ball == 0).all()


def test_wicket_value_follows_the_state(data):
    overs = Overs(data, n=2, formats=("T20",), shots_per_ball=8)
    overs.balls_remaining[:] = [114, 6]
    np.testing.assert_allclose(overs.wicket_value(), [19 * 0.5, 1 * 0.5])


def test_relative_reward_is_zero_for_the_starting_field(data):
    overs = Overs(data, n=50, relative=True, shots_per_ball=64)
    reward, _, _ = overs.step(overs.fields_from_actions(np.zeros((50, 18))))
    np.testing.assert_allclose(reward, 0.0, atol=1e-12)


def test_relative_reward_is_the_cost_saved_on_the_same_deliveries(data):
    actions = np.random.default_rng(5).uniform(-1, 1, (50, 18))
    rel = Overs(data, n=50, relative=True, shots_per_ball=64, seed=9)
    absolute = Overs(data, n=50, shots_per_ball=64, seed=9)
    start = Overs(data, n=50, shots_per_ball=64, seed=9)
    r_rel, _, _ = rel.step(rel.fields_from_actions(actions))
    r_abs, _, _ = absolute.step(absolute.fields_from_actions(actions))
    r_start, _, _ = start.step(start.fields_from_actions(np.zeros((50, 18))))
    np.testing.assert_allclose(r_rel, r_abs - r_start, atol=1e-9)

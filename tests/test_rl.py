import numpy as np
import polars as pl
from gymnasium.utils.env_checker import check_env

from virtual_captain.field import rules
from virtual_captain.field.templates import KEYS, template
from virtual_captain.rl import evaluate as ev
from virtual_captain.rl.env import (
    N_OBS,
    CaptainEnv,
    CaptainVecEnv,
    Overs,
    decode_fields,
    encode_fields,
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


def test_judging_a_field_against_itself_is_exactly_even(data):
    ids = np.flatnonzero(~data.contexts["pooled"].to_numpy())
    fields = data.searched_fields[ids]
    a = ev.judge_fields(data, ids, fields, n=300)
    b = ev.judge_fields(data, ids, fields.copy(), n=300)
    diff, half = ev.paired(a, b)
    assert np.all(diff == 0) and np.all(half == 0)


def test_every_setter_plays_the_same_overs(data):
    played = {
        name: ev.play(s, data, "T20", 40, seed=3, shots_per_ball=16)
        for name, s in (("pooled", ev.optimised_for_all), ("again", ev.optimised_for_all))
    }
    gap = ev.compare(played, "pooled")
    assert gap["again"] == (0.0, 0.0)
    assert np.all(played["pooled"].balls == 6)

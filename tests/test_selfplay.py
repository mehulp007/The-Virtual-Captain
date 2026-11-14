from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from pettingzoo.test import api_test

from virtual_captain.rl import evaluate as ev
from virtual_captain.rl.batter import (
    N_BATTER_OBS,
    attack_score,
    field_coverage,
    fit_forcing_risk,
    intent_weights,
)
from virtual_captain.rl.env import CaptainVecEnv, Overs
from virtual_captain.rl.multi_env import (
    BatterVecEnv,
    CaptainBatterGame,
    Population,
    SelfPlayCaptainEnv,
    batter_observe,
    natural,
)


@pytest.fixture(scope="module")
def risky(data):
    """The synthetic data with a forcing risk."""
    risk = pl.DataFrame(
        {"format": ["T20", "ODI", "TEST"], "per_attack": [0.1] * 3, "per_shift": [0.03] * 3,
         "max_shift": [0.2] * 3}
    )  # fmt: skip
    return replace(data, forcing_risk=risk)


def _deliveries(rng, n=4, k=500):
    outcome = rng.choice([0, 1, 2, 3], size=(n, k), p=[0.3, 0.03, 0.62, 0.05])
    theta, dist = rng.uniform(0, 2 * np.pi, (n, k)), rng.uniform(0.05, 1.0, (n, k))
    return outcome, theta, dist, rng.random((n, k)) < 0.05


def test_no_intent_is_the_natural_game():
    outcome, theta, dist, aerial = _deliveries(np.random.default_rng(0))
    w, kl = intent_weights(np.zeros((4, 3)), outcome, theta, dist, aerial)
    np.testing.assert_allclose(w, 1 / 500)
    np.testing.assert_allclose(kl, 0, atol=1e-12)


def test_aggression_and_tilt_move_the_shots_the_right_way():
    outcome, theta, dist, aerial = _deliveries(np.random.default_rng(1))
    score = attack_score(outcome, dist, aerial)
    w, kl = intent_weights(
        np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0]]), outcome, theta, dist, aerial
    )
    np.testing.assert_allclose(w.sum(1), 1)
    assert (w[0] * score[0]).sum() > score[0].mean() > (w[1] * score[1]).sum()
    towards = (np.cos(theta[2]) > 0.5) & (outcome[2] == 2)
    assert w[2][towards].sum() > towards.mean()
    six = outcome[2] == 3
    assert w[2][six].sum() < six.mean()  # sixes have no direction to tilt towards
    assert kl[0] > 0 and kl[2] > 0 and kl[3] == 0


def test_intent_never_moves_the_game_further_than_the_cap():
    outcome, theta, dist, aerial = _deliveries(np.random.default_rng(2))
    intent = np.array([[1, 1, 0], [-1, 0, 1], [0.05, 0, 0], [0, 0, 0]])
    free, kl_free = intent_weights(intent, outcome, theta, dist, aerial)
    w, kl = intent_weights(intent, outcome, theta, dist, aerial, max_shift=0.05)
    assert kl_free[0] > 0.05 and kl_free[1] > 0.05 and kl_free[2] < 0.05
    np.testing.assert_allclose(kl[:2], 0.05, rtol=1e-3)
    np.testing.assert_allclose(w[2:], free[2:])  # small intents are left alone
    score = attack_score(outcome, dist, aerial)
    assert (w[0] * score[0]).sum() > score[0].mean()  # still the same way, just less of it
    np.testing.assert_allclose(w.sum(1), 1)


def test_field_coverage_counts_every_fielder(data):
    cover = field_coverage(data.pooled_fields)
    assert cover.shape == (16, 24)
    np.testing.assert_allclose(cover.sum(1), 9)


def test_forcing_fit_recovers_known_slopes():
    rng = np.random.default_rng(0)
    kl = rng.uniform(0, 0.2, 3000)
    d_attack = rng.normal(0, 0.05, 3000)
    shifts = pl.DataFrame(
        {
            "format": np.repeat(["T20", "ODI", "TEST"], 1000),
            "kl": kl,
            "d_attack": d_attack,
            "excess_out": 0.1 * d_attack + 0.02 * kl + rng.normal(0, 0.001, 3000),
            "balls": rng.integers(300, 2000, 3000),
        }
    )
    fit = fit_forcing_risk(shifts)
    np.testing.assert_allclose(fit["per_attack"], 0.1, atol=0.005)
    np.testing.assert_allclose(fit["per_shift"], 0.02, atol=0.003)
    assert (fit["per_attack_ci"] > 0).all() and (fit["max_shift"] <= 0.2).all()


def test_natural_batter_changes_nothing(risky):
    o = Overs(risky, n=32, seed=4)
    fields = o.data.pooled_fields[o.key]
    _, _, info = o.step(fields, intent=np.zeros((32, 3)))
    np.testing.assert_allclose(info["cost"], info["natural_cost"])
    np.testing.assert_allclose(info["forcing"], 0)


def test_forcing_the_game_brings_more_wickets(risky):
    riskless = risky.forcing_risk.with_columns(per_attack=pl.lit(0.0), per_shift=pl.lit(0.0))
    free = Overs(replace(risky, forcing_risk=riskless), n=64, seed=5)
    forced = Overs(risky, n=64, seed=5)
    fields = free.data.pooled_fields[free.key]
    tilt = np.tile([0.5, 1.0, 0.0], (64, 1))
    _, _, a = free.step(fields, intent=tilt)
    _, _, b = forced.step(fields, intent=tilt)
    assert np.all(a["forcing"] == 0) and np.all(b["forcing"] > 0)
    expected = b["forcing"] + (1 - b["forcing"]) * a["expected_wickets"]
    np.testing.assert_allclose(b["expected_wickets"], expected)
    assert np.all(b["cost"] < a["cost"])  # dearer for the batter, cheaper for the captain


def test_batter_env_pays_what_the_intent_gained(risky):
    env = BatterVecEnv(risky, Population([ev.optimised_for_all]), n_envs=16, seed=0)
    obs = env.reset()
    assert obs.shape == (16, N_BATTER_OBS)
    _, reward, _, _ = env.step(np.zeros((16, 3), dtype=np.float32))
    np.testing.assert_allclose(reward, 0, atol=1e-6)
    for _ in range(30):
        obs, reward, done, infos = env.step(np.random.default_rng(0).uniform(-1, 1, (16, 3)))
    assert np.all(np.isfinite(reward)) and np.all(np.abs(obs) <= 10)


def test_population_plays_each_over_against_one_member(risky):
    pop = Population([ev.optimised_for_all, ev.conventional], weights=[1, 1])
    env = BatterVecEnv(risky, pop, n_envs=64, seed=1)
    env.reset()
    o = env.overs
    expected = np.where(
        (pop.slot == 0)[:, None, None],
        o.data.pooled_fields[o.key],
        o.data.conventional_fields[o.key],
    )
    np.testing.assert_allclose(env.fields, expected)
    assert 0 < pop.slot.mean() < 1


def test_self_play_with_natural_batters_is_the_plain_game(risky):
    plain = CaptainVecEnv(risky, n_envs=16, relative=True, seed=3)
    duel = SelfPlayCaptainEnv(risky, Population([natural]), n_envs=16, seed=3)
    plain.reset()
    duel.reset()
    actions = np.random.default_rng(2).uniform(-1, 1, (16, 18))
    _, r1, _, _ = plain.step(actions)
    _, r2, _, _ = duel.step(actions)
    np.testing.assert_allclose(r1, r2, atol=1e-5)


def test_batter_sees_the_field_it_faces(risky):
    o = Overs(risky, n=16, seed=6)
    a = batter_observe(o, o.data.pooled_fields[o.key])
    turned = o.data.pooled_fields[o.key][..., ::-1] * [1, -1]  # every fielder 90 degrees round
    b = batter_observe(o, turned)
    assert a.shape == (16, N_BATTER_OBS)
    np.testing.assert_allclose(a[:, :-24], b[:, :-24])
    assert not np.allclose(a[:, -24:], b[:, -24:])


def test_pettingzoo_api(risky):
    api_test(CaptainBatterGame(risky), num_cycles=30)


def test_game_is_zero_sum(risky):
    game = CaptainBatterGame(risky)
    game.reset(seed=0)
    totals = {"captain": 0.0, "batter": 0.0}
    for agent in game.agent_iter(max_iter=200):
        _, reward, term, trunc, _ = game.last()
        totals[agent] += reward
        action = None if term or trunc else game.action_space(agent).sample()
        game.step(action)
    assert totals["captain"] == pytest.approx(-totals["batter"])
    assert totals["batter"] != 0

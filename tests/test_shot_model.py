import math

import numpy as np
import polars as pl
import pytest
import torch

from virtual_captain.models import encoding as enc
from virtual_captain.models.baselines import BIN_AREA, BatterHistogram
from virtual_captain.models.shot_model import ShotModel, ShotNet, sample_shots, shot_log_prob


def _mixture(n_rows=1, k=3, seed=0):
    torch.manual_seed(seed)
    net = ShotNet(n_features=enc.n_features(), n_batters=4, n_bowlers=3, n_components=k)
    x = torch.randn(n_rows, enc.n_features())
    _, mix = net(x, torch.zeros(n_rows, dtype=torch.long), torch.zeros(n_rows, dtype=torch.long))
    return {k: v.detach() for k, v in mix.items()}


def test_density_integrates_to_one():
    mix = _mixture()
    nt, nd = 720, 3000
    theta = torch.linspace(0, 2 * math.pi, nt + 1)[:-1]
    dist = torch.linspace(1e-4, 30, nd)  # log-normal tail is negligible well before 30
    tt, dd = torch.meshgrid(theta, dist, indexing="ij")
    total = 0.0
    for aerial in (0.0, 1.0):
        rep = {k: v.expand(tt.numel(), *v.shape[1:]) for k, v in mix.items()}
        lp = shot_log_prob(rep, tt.reshape(-1), torch.full((tt.numel(),), aerial), dd.reshape(-1))
        total += lp.exp().sum().item() * (2 * math.pi / nt) * (dist[1] - dist[0]).item()
    assert total == pytest.approx(1.0, abs=0.01)


def test_samples_follow_the_mixture():
    mix = _mixture(k=2, seed=3)
    n = 20000
    rep = {k: v.expand(n, *v.shape[1:]) for k, v in mix.items()}
    theta, aerial, dist = sample_shots(rep, torch.Generator().manual_seed(0))
    assert theta.min() >= 0 and theta.max() < 2 * math.pi
    assert dist.min() >= enc.D_MIN and dist.max() <= enc.D_MAX

    w = mix["log_w"].exp()[0]
    p_air = torch.sigmoid(mix["aerial_logit"][0])
    expected_air = (w * p_air).sum().item()
    assert aerial.float().mean().item() == pytest.approx(expected_air, abs=0.02)


def _toy_frame():
    rng = np.random.default_rng(0)
    n = 4000
    outcome = rng.choice([0, 1, 2], size=n, p=[0.3, 0.05, 0.65])
    return pl.DataFrame(
        {
            "striker_id": rng.choice(["a", "b", "c"], size=n),
            "outcome": outcome,
            "theta": rng.uniform(0, 2 * math.pi, n),
            "dist": rng.uniform(enc.D_MIN, enc.D_MAX, n),
            "aerial": rng.random(n) < 0.1,
        }
    )


def test_histogram_baseline_is_a_density():
    df = _toy_frame()
    base = BatterHistogram().fit(df)
    for probs in [base.global_shot, *base.shot.values()]:
        assert probs.sum() == pytest.approx(1.0)
    # uniform data -> density close to 1 / (2*pi * D_MAX) per aerial class share
    shots = df.filter(pl.col("outcome") == enc.IN_PLAY)
    assert np.isfinite(base.shot_log_prob(shots)).all()
    assert base.global_shot.max() / BIN_AREA < 1.0


def test_histogram_outcome_falls_back_for_unknown_batter():
    df = _toy_frame()
    base = BatterHistogram().fit(df)
    unseen = df.head(5).with_columns(pl.lit("zz").alias("striker_id"))
    np.testing.assert_allclose(base.outcome_log_prob(unseen), base.global_outcome_log_prob(unseen))


def test_same_seed_same_deliveries():
    torch.manual_seed(0)
    vocab = enc.Vocab(
        batters=[], bowlers=[], venue_mean=[1.0, 0.1, 0.3], venue_std=[0.3, 0.05, 0.1]
    )
    net = ShotNet(enc.n_features(), vocab.n_batters, vocab.n_bowlers, n_components=3)
    model = ShotModel(net, vocab, {"net": {"n_components": 3}}, device="cpu")
    rows = pl.DataFrame(
        {
            "format": ["T20"] * 4, "phase": ["middle"] * 4, "bowler_kind": ["fast"] * 4,
            "bowler_arm": ["right"] * 4, "spin_turn": [None] * 4, "ball_line": [None] * 4,
            "ball_length": [None] * 4, "left_handed": [False] * 4, "competition_type": ["Intl"] * 4,
            "year": [2020] * 4, "wickets_before": [2] * 4, "balls_before": [50] * 4,
            "balls_remaining": [70] * 4, "ball_age": [50] * 4, "target": [None] * 4,
            "runs_required": [None] * 4, "runs_before": [60] * 4, "striker_balls_faced": [10] * 4,
            "innings": [1] * 4, "arm_vs_bat": ["same"] * 4, "venue_runs_per_ball": [1.2] * 4,
            "venue_boundary_rate": [0.12] * 4, "venue_spin_wicket_share": [0.3] * 4,
            "striker_id": ["x"] * 4, "bowler_id": ["y"] * 4,
        },
        schema_overrides={"spin_turn": pl.String, "ball_line": pl.String, "ball_length": pl.String,
                          "target": pl.Int64, "runs_required": pl.Int64},
    )  # fmt: skip
    a = model.sample(rows, n=200, seed=5)
    torch.rand(10)  # disturb the global generator in between
    b = model.sample(rows, n=200, seed=5)
    assert a.equals(b)
    assert not a.equals(model.sample(rows, n=200, seed=6))

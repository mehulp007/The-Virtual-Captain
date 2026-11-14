"""The demo app: its numpy captain and scoring agree with the training stack, and the app runs."""

from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from virtual_captain.app.bundle import Bundle, fetch, from_captain_data, policy_weights
from virtual_captain.app.engine import (
    CAPTAIN,
    INNINGS_OVERS,
    PHASE_OVERS,
    SEARCHED,
    Engine,
    Situation,
)
from virtual_captain.rl.env import CaptainVecEnv, Overs

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "virtual_captain" / "app" / "streamlit_app.py"
ENTRY = ROOT / "app" / "streamlit_app.py"  # what Streamlit Community Cloud runs


@pytest.fixture(scope="module")
def model(data):
    return PPO("MlpPolicy", CaptainVecEnv(data, n_envs=2), n_steps=8, batch_size=8, seed=0)


@pytest.fixture(scope="module")
def bundle(data, model, tmp_path_factory):
    n = data.contexts.height
    rng = np.random.default_rng(1)
    wagon = {c: (rng.uniform(0, 6.2, 50), rng.uniform(0.1, 1, 50), rng.integers(-1, 5, 50))
             for c in range(n)}  # fmt: skip
    b = from_captain_data(data, policy_weights(model), wagon=wagon, shots=256)
    path = tmp_path_factory.mktemp("app") / "bundle.npz"
    b.save(path)
    return path


@pytest.fixture(scope="module")
def engine(bundle):
    return Engine(Bundle.load(bundle))


def _overs_at(data, s: Situation, fmt: str) -> Overs:
    overs = Overs(data, n=1, shots_per_ball=1)
    if fmt == "TEST":
        balls_remaining, ball_age = 0, 6 * s.over
    else:
        balls_remaining, ball_age = 6 * (INNINGS_OVERS[fmt] - s.over + 1), 0
    c = np.array([s.context])
    overs.place(c, c, s.wickets, balls_remaining, ball_age)
    return overs


@pytest.mark.parametrize("c", [16, 23, 30])  # a T20, an ODI and a Test context of batter A
def test_numpy_captain_matches_the_trained_policy(data, model, engine, c):
    fmt, phase = str(engine.b.format[c]), str(engine.b.phase[c])
    s = Situation(c, 4, PHASE_OVERS[(fmt, phase)][0])
    overs = _overs_at(data, s, fmt)
    np.testing.assert_allclose(engine.observe(s), overs.observe(), atol=1e-6)
    assert engine.wicket_value(s) == pytest.approx(float(overs.wicket_value()[0]))
    action, _ = model.predict(overs.observe(), deterministic=True)
    np.testing.assert_allclose(engine.captain_field(s), overs.fields_from_actions(action)[0],
                               atol=1e-4)  # fmt: skip


def test_bundle_round_trip(data, bundle):
    b = Bundle.load(bundle)
    assert len(b.batter) == data.contexts.height
    assert b.shot_outcome.shape == (data.contexts.height, 256)
    assert b.wagon_offsets[-1] == 50 * data.contexts.height
    np.testing.assert_allclose(b.optimised, data.pooled_fields)


def test_every_field_is_scored_on_the_same_deliveries(engine):
    s = engine.typical(20)
    fields = engine.fields(s)
    assert set(fields) >= {CAPTAIN, SEARCHED}
    twice = dict(fields) | {"again": fields[CAPTAIN].copy()}
    scored = engine.score(s, twice)
    assert scored["again"].minus(scored[CAPTAIN]) == (0.0, 0.0)
    m = scored[CAPTAIN].summary()
    assert 0 < m["runs_per_ball"] < 3 and 0 <= m["wickets_per_ball"] < 0.5


def test_the_all_batters_context_has_no_dedicated_search(engine):
    assert SEARCHED not in engine.fields(engine.typical(0))


def test_value_map_is_zero_where_the_fielder_already_stands(engine):
    s = engine.typical(20)
    field = engine.fields(s)[CAPTAIN].copy()
    field[0] = [-3.0, 37.0]  # put fielder 0 on a grid point (it runs from -68 m in 5s)
    points, change, legal = engine.value_map(s, field, 0, spacing=5.0, n=128)
    here = np.flatnonzero(np.hypot(*(points - field[0]).T) < 1e-9)
    assert len(here) == 1 and change[here[0]] == 0.0
    assert legal.any() and not legal.all()  # the pitch itself is off limits


def test_a_missing_bundle_is_downloaded(bundle, tmp_path):
    path = tmp_path / "app" / "bundle.npz"
    assert fetch(path, bundle.as_uri()) == path
    assert path.read_bytes() == bundle.read_bytes()
    assert fetch(path, "file:///nowhere/at/all") == path  # already there: no download


@pytest.mark.parametrize("script", [APP, ENTRY])
def test_the_app_runs(bundle, monkeypatch, script):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("VC_BUNDLE", str(bundle))
    at = AppTest.from_file(str(script), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "The Virtual Captain"
    at.sidebar.radio[0].set_value("TEST").run()
    assert not at.exception

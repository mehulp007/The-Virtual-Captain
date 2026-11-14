"""Everything the demo app needs, in one file: no raw data, no torch.

  contexts      every batter context and the 16 all-batters ones: names, situation, typical
                match state, standardised profile and the field a dedicated search found
  keys          the conventional field and the field optimised for all batters, by situation
  wicket value  by format, overs left and wickets down (limited overs) or wickets down (Tests)
  deliveries    SHOTS simulated deliveries per context, from its judging bank
  wagon wheel   up to WAGON real shots per context that stayed in the ground, from the seasons
                the models were trained on
  captain       the policy's weights, run with numpy

Built by scripts/build_app_bundle.py into data/app/bundle.npz, and attached to the GitHub
release the deployed app downloads it from.
"""

import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from virtual_captain.paths import DATA_DIR

BUNDLE = DATA_DIR / "app" / "bundle.npz"
BUNDLE_URL = "https://github.com/mehulp007/The-Virtual-Captain/releases/latest/download/bundle.npz"
SHOTS = 8192
WAGON = 1500


@dataclass
class Bundle:
    # contexts (C,)
    batter: np.ndarray  # str
    format: np.ndarray  # str
    phase: np.ndarray  # str
    bowling: np.ndarray  # str
    key: np.ndarray  # int
    pooled: np.ndarray  # bool, the "all batters" context of its situation
    unseen: np.ndarray  # bool, kept out of the captain's training
    left_handed: np.ndarray  # bool
    balls: np.ndarray  # int, tracked balls behind the context
    wickets_median: np.ndarray
    balls_remaining_median: np.ndarray
    ball_age_median: np.ndarray
    profiles: np.ndarray  # (C, N_PROFILE) standardised as the captain sees them
    searched: np.ndarray  # (C, 9, 2)
    # situations (16,)
    key_format: np.ndarray  # str
    key_phase: np.ndarray  # str
    key_bowling: np.ndarray  # str
    key_wicket_value: np.ndarray  # a wicket's value for the phase as a whole
    conventional: np.ndarray  # (16, 9, 2)
    optimised: np.ndarray  # (16, 9, 2), for all batters
    # wicket value by state
    lam_limited: np.ndarray  # (2, 51, 10): T20 / ODI, overs left, wickets down
    lam_test: np.ndarray  # (10,)
    # simulated deliveries (C, SHOTS)
    shot_outcome: np.ndarray  # int8
    shot_theta: np.ndarray  # float16, radians
    shot_dist: np.ndarray  # float16, fraction of the boundary radius
    shot_aerial: np.ndarray  # bool
    # real shots kept in the ground, context c's are wagon_offsets[c]:wagon_offsets[c + 1]
    wagon_theta: np.ndarray  # float16
    wagon_dist: np.ndarray  # float16
    wagon_runs: np.ndarray  # int8
    wagon_offsets: np.ndarray  # int64
    real_six: np.ndarray  # (C,) share of real balls hit for six
    # captain: tanh MLP, weights stored (in, out)
    w0: np.ndarray
    b0: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def save(self, path: Path = BUNDLE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: Path = BUNDLE) -> "Bundle":
        with np.load(path, allow_pickle=False) as a:
            return cls(**{f.name: a[f.name] for f in fields(cls)})


def fetch(path: Path = BUNDLE, url: str = BUNDLE_URL) -> Path:
    """The bundle at `path`, downloading it first if it isn't there yet."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(url, part)
        part.replace(path)
    return path


def policy_weights(model) -> dict[str, np.ndarray]:
    """The deterministic actor of a stable-baselines3 PPO MlpPolicy with two tanh layers."""
    pol = model.policy
    net = pol.mlp_extractor.policy_net
    layers = [net[0], net[2], pol.action_net]
    out = {}
    for i, layer in enumerate(layers):
        out[f"w{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        out[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    return out


def from_captain_data(
    data,
    weights: dict[str, np.ndarray],
    wagon: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    real_six: np.ndarray | None = None,
    left_handed: np.ndarray | None = None,
    shots: int = SHOTS,
    seed: int = 0,
) -> Bundle:
    """Pack a CaptainData (rl.data) and a captain's weights. `wagon` maps a context to its real
    shots (theta, dist, runs); contexts without any get none."""
    from virtual_captain.rl.env import Overs  # the training stack is only needed to build

    overs = Overs(data, n=1, shots_per_ball=1)
    ctx = data.contexts
    n_ctx = ctx.height
    rng = np.random.default_rng(seed)
    bank = data.judge
    picks = np.stack(
        [bank.offsets[c] + rng.integers(0, bank.lengths[c], shots) for c in range(n_ctx)]
    )
    wagon = wagon or {}
    parts = [wagon.get(c, (np.zeros(0), np.zeros(0), np.zeros(0))) for c in range(n_ctx)]
    lengths = [len(p[0]) for p in parts]
    keys = data.keys
    return Bundle(
        batter=ctx["batter"].to_numpy().astype(str),
        format=ctx["format"].to_numpy().astype(str),
        phase=ctx["phase"].to_numpy().astype(str),
        bowling=ctx["bowling"].to_numpy().astype(str),
        key=ctx["key"].to_numpy().astype(np.int64),
        pooled=ctx["pooled"].to_numpy(),
        unseen=ctx["unseen"].to_numpy(),
        left_handed=np.zeros(n_ctx, bool) if left_handed is None else left_handed,
        balls=ctx["balls"].to_numpy().astype(np.int64),
        wickets_median=ctx["wickets_median"].to_numpy(),
        balls_remaining_median=ctx["balls_remaining_median"].to_numpy(),
        ball_age_median=ctx["ball_age_median"].to_numpy(),
        profiles=overs.profiles,
        searched=data.searched_fields,
        key_format=keys["format"].to_numpy().astype(str),
        key_phase=keys["phase"].to_numpy().astype(str),
        key_bowling=keys["bowling"].to_numpy().astype(str),
        key_wicket_value=keys["wicket_value"].to_numpy(),
        conventional=data.conventional_fields,
        optimised=data.pooled_fields,
        lam_limited=overs.lam_limited,
        lam_test=overs.lam_test,
        shot_outcome=bank.outcome[picks],
        shot_theta=bank.theta[picks].astype(np.float16),
        shot_dist=bank.dist[picks].astype(np.float16),
        shot_aerial=bank.aerial[picks],
        wagon_theta=np.concatenate([p[0] for p in parts]).astype(np.float16),
        wagon_dist=np.concatenate([p[1] for p in parts]).astype(np.float16),
        wagon_runs=np.concatenate([p[2] for p in parts]).astype(np.int8),
        wagon_offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        real_six=np.zeros(n_ctx) if real_six is None else real_six,
        **weights,
    )

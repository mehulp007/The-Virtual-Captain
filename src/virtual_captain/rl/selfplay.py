"""Fictitious self-play between the captain and the batter, and how exploitable a captain is.

Round r of self-play:

  1. a batter is trained against every captain so far, picked at random over by over
  2. the captain carries on training against the natural batter and every batter so far

Opponents are frozen copies, so each round faces a fixed population; this is fictitious
play, which keeps the captain from chasing only the newest batter and forgetting the old ones.

A captain's exploitability is what a batter who knows it can gain: a fresh batter is trained
against that captain alone for a fixed number of balls, then plays it on deliveries neither
trained on. The gain is its cost per ball minus what the same balls would have cost had it
played its natural game, ball by ball against the same fields.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from virtual_captain.rl import evaluate as ev
from virtual_captain.rl.data import CaptainData
from virtual_captain.rl.env import FORMATS
from virtual_captain.rl.multi_env import (
    BatterVecEnv,
    Population,
    SelfPlayCaptainEnv,
    batter_policy,
)

FORMAT_WEIGHTS = (0.45, 0.45, 0.10)  # overs of each format; a Test spell is four overs long
EVAL_OVERS = {"T20": 4000, "ODI": 4000, "TEST": 1000}


@dataclass
class Settings:
    envs: int = 256
    n_steps: int = 64
    batch_size: int = 4096
    epochs: int = 10
    batter_lr: float = 3e-4
    captain_lr: float = 1e-4
    batter_log_std: float = -1.0
    seed: int = 0


def frozen(model: PPO, path: Path) -> PPO:
    """Save a model and load a copy of it that later training won't touch."""
    model.save(path)
    return PPO.load(path, device="cpu")


def new_batter(env, s: Settings, seed: int) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        n_steps=s.n_steps,
        batch_size=s.batch_size,
        n_epochs=s.epochs,
        learning_rate=s.batter_lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "log_std_init": s.batter_log_std,
        },  # fmt: skip
        device="cpu",
        seed=seed,
        verbose=0,
    )


def load_captain(path: Path, s: Settings) -> PPO:
    """A trained captain, ready to carry on training at the self-play learning rate."""
    lr = s.captain_lr
    return PPO.load(
        path, device="cpu", custom_objects={"learning_rate": lr, "lr_schedule": lambda _: lr}
    )


def train_batter(
    data: CaptainData,
    captains: list,
    steps: int,
    s: Settings,
    model: PPO | None = None,
    seed: int = 0,
    callback=None,
) -> PPO:
    """Train a batter (or carry on training one) against a population of field-setters."""
    env = BatterVecEnv(
        data, Population(captains, seed=seed), n_envs=s.envs, formats=FORMATS,
        format_weights=FORMAT_WEIGHTS, seed=seed,
    )  # fmt: skip
    if model is None:
        model = new_batter(env, s, seed)
    else:
        model.set_env(env)
    model.learn(steps, reset_num_timesteps=True, callback=callback)
    return model


def train_captain(
    data: CaptainData,
    batters: list,
    steps: int,
    model: PPO,
    s: Settings,
    seed: int = 0,
    callback=None,
) -> PPO:
    """Carry on training a captain against a population of batters."""
    env = SelfPlayCaptainEnv(
        data, Population(batters, seed=seed), n_envs=s.envs, formats=FORMATS,
        format_weights=FORMAT_WEIGHTS, seed=seed,
    )  # fmt: skip
    model.set_env(env)
    model.learn(steps, reset_num_timesteps=True, callback=callback)
    return model


def exploit(
    data: CaptainData, setter, batter, batters: str = "train", seed: int = 21
) -> list[dict]:
    """A batter against a field-setter, by format, on the judging deliveries."""
    rows = []
    for fmt in FORMATS:
        p = ev.play(setter, data, fmt, EVAL_OVERS[fmt], batters=batters, seed=seed, batter=batter)
        gain = (p.cost - p.natural) / np.maximum(p.balls, 1)
        rows.append(
            {
                "format": fmt,
                "batters": batters,
                "natural_cost": float(p.natural.sum() / p.balls.sum()),
                "cost": float(p.cost.sum() / p.balls.sum()),
                "gain": float(gain.mean()),
                "gain_ci": float(1.96 * gain.std(ddof=1) / np.sqrt(len(gain))),
            }
        )
    return rows


def exploitability(
    data: CaptainData, setter, steps: int, s: Settings, seed: int = 100, path: Path | None = None
) -> tuple[list[dict], PPO]:
    """Train a best-response batter against one field-setter and measure what it gains."""
    br = train_batter(data, [setter], steps, s, seed=seed)
    if path is not None:
        br.save(path)
    rows = []
    for batters in ("train", "unseen"):
        rows += exploit(data, setter, batter_policy(br), batters=batters)
    return rows, br

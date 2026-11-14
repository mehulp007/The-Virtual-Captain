"""What the captain learns from and is judged on. Built once by scripts/build_captain_data.py.

A *context* is a batter in a (format, phase, type of bowling) situation, or "all batters" in
one. Each context carries:

  two banks of simulated deliveries   one to learn from, a separate one to judge on
  a profile                           where and how the batter hits the ball, read off the
                                      learning bank; this is how the captain tells batters apart
  a searched field                    what a dedicated CMA-ES search finds for this context

Plus, for each of the 16 (format, phase, bowling) keys: the conventional field and the field
optimised for all batters together, and the real match states overs are started from; and by
format, the risk a batter takes by forcing their game (see rl.batter).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from virtual_captain.field.templates import KEYS
from virtual_captain.paths import DATA_DIR
from virtual_captain.sim.engine import Shots

CAPTAIN_DIR = DATA_DIR / "captain"

PROFILE_SECTORS = 12
DEEP_FROM = 30.0 / 68.0  # a shot that ends past about 30 m, roughly the ring
PROFILE_NAMES = (
    [f"share_{i * 30}" for i in range(PROFILE_SECTORS)]
    + [f"deep_{i * 30}" for i in range(PROFILE_SECTORS)]
    + ["no_contact", "bowled_lbw", "six", "aerial", "mean_dist"]
)
N_PROFILE = len(PROFILE_NAMES)


def profile(shots: Shots) -> np.ndarray:
    """Where and how a batter scores, from their simulated deliveries.

    Shares of balls put into play by 30-degree sector (wagon-wheel angle), the share that go
    past the ring in each sector, the rates of no contact, bowled or lbw and sixes, the share of
    balls in play hit in the air and their mean distance.
    """
    n = len(shots)
    in_play = shots.outcome == 2
    k = max(int(in_play.sum()), 1)
    sector = (shots.theta[in_play] / (2 * np.pi) * PROFILE_SECTORS).astype(int) % PROFILE_SECTORS
    deep = shots.dist[in_play] >= DEEP_FROM
    return np.concatenate(
        [
            np.bincount(sector, minlength=PROFILE_SECTORS) / k,
            np.bincount(sector[deep], minlength=PROFILE_SECTORS) / k,
            [
                (shots.outcome == 0).mean() if n else 0.0,
                (shots.outcome == 1).mean() if n else 0.0,
                (shots.outcome == 3).mean() if n else 0.0,
                shots.aerial[in_play].mean() if in_play.any() else 0.0,
                shots.dist[in_play].mean() if in_play.any() else 0.0,
            ],
        ]
    ).astype(np.float32)


@dataclass
class Bank:
    """Simulated deliveries for every context, stored end to end."""

    outcome: np.ndarray  # int8
    theta: np.ndarray  # float64
    dist: np.ndarray  # float64
    aerial: np.ndarray  # bool
    offsets: np.ndarray  # int64, context c's deliveries are offsets[c]:offsets[c + 1]

    @classmethod
    def from_shots(cls, shots: list[Shots]) -> "Bank":
        lengths = [len(s) for s in shots]
        return cls(
            outcome=np.concatenate([s.outcome for s in shots]).astype(np.int8),
            theta=np.concatenate([s.theta for s in shots]),
            dist=np.concatenate([s.dist for s in shots]),
            aerial=np.concatenate([s.aerial for s in shots]),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        )

    def shots(self, context: int) -> Shots:
        a, b = self.offsets[context], self.offsets[context + 1]
        return Shots(self.outcome[a:b], self.theta[a:b], self.dist[a:b], self.aerial[a:b])

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}_{k}": getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_arrays(cls, arrays, prefix: str) -> "Bank":
        return cls(**{k: arrays[f"{prefix}_{k}"] for k in cls.__dataclass_fields__})


def _fields(column: pl.Series) -> np.ndarray:
    return np.stack(column.to_list()).reshape(-1, 9, 2).astype(np.float64)


@dataclass
class CaptainData:
    contexts: pl.DataFrame  # one row per context, see scripts/build_captain_data.py
    keys: pl.DataFrame  # one row per (format, phase, bowling), in field.templates.KEYS order
    starts: pl.DataFrame  # real match states at the start of an over
    wicket_states: pl.DataFrame  # optim.value.wicket_value_states
    learn: Bank
    judge: Bank
    forcing_risk: pl.DataFrame | None = None  # rl.batter.fit_forcing_risk, by format

    def save(self, folder: Path = CAPTAIN_DIR) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.contexts.write_parquet(folder / "contexts.parquet")
        self.keys.write_parquet(folder / "keys.parquet")
        self.starts.write_parquet(folder / "starts.parquet")
        self.wicket_states.write_parquet(folder / "wicket_states.parquet")
        np.savez(folder / "banks.npz", **self.learn.arrays("learn"), **self.judge.arrays("judge"))
        if self.forcing_risk is not None:
            self.forcing_risk.write_parquet(folder / "forcing_risk.parquet")

    @classmethod
    def load(cls, folder: Path = CAPTAIN_DIR) -> "CaptainData":
        with np.load(folder / "banks.npz") as arrays:
            learn, judge = Bank.from_arrays(arrays, "learn"), Bank.from_arrays(arrays, "judge")
        return cls(
            contexts=pl.read_parquet(folder / "contexts.parquet"),
            keys=pl.read_parquet(folder / "keys.parquet"),
            starts=pl.read_parquet(folder / "starts.parquet"),
            wicket_states=pl.read_parquet(folder / "wicket_states.parquet"),
            learn=learn,
            judge=judge,
            forcing_risk=(
                pl.read_parquet(folder / "forcing_risk.parquet")
                if (folder / "forcing_risk.parquet").exists()
                else None
            ),
        )

    # handy arrays -------------------------------------------------------------------------
    @property
    def profiles(self) -> np.ndarray:
        return np.stack(self.contexts["profile"].to_list()).astype(np.float32)

    @property
    def searched_fields(self) -> np.ndarray:
        """(C, 9, 2): each context's own CMA-ES field."""
        return _fields(self.contexts["searched"])

    @property
    def analyst_fields(self) -> np.ndarray:
        return _fields(self.contexts["analyst"])

    @property
    def conventional_fields(self) -> np.ndarray:
        """(16, 9, 2), by key."""
        return _fields(self.keys["conventional"])

    @property
    def pooled_fields(self) -> np.ndarray:
        """(16, 9, 2), by key: optimised for all batters in that situation."""
        return _fields(self.keys["optimised"])

    def key_of(self, fmt: str, phase: str, bowling: str) -> int:
        return KEYS.index((fmt, phase, bowling))

    def context_of(self, batter: str, fmt: str, phase: str, bowling: str) -> int:
        hit = self.contexts.filter(
            (pl.col("batter") == batter)
            & (pl.col("format") == fmt)
            & (pl.col("phase") == phase)
            & (pl.col("bowling") == bowling)
        )
        if hit.height != 1:
            raise KeyError(f"no context for {batter} in {fmt} {phase} {bowling}")
        return int(hit["context"][0])

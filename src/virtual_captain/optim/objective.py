"""A scenario to set a field for, and how a field is scored in it.

A scenario is a batter (or a whole group of batters) in a format, phase and type of bowling.
Its deliveries are the real contexts that batter faced there (line, length, bowler, match
state, venue); the shot model turns each into simulated shots. Every candidate field is scored
on the same simulated shots and the same random numbers, so differences between fields are
differences in the field, not in luck.

    cost per ball = runs conceded - lambda x wickets taken
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from virtual_captain.field import rules
from virtual_captain.field.positions import with_keeper_and_bowler
from virtual_captain.field.templates import template
from virtual_captain.sim.engine import Shots, simulate
from virtual_captain.sim.params import SimParams, situation_code


@dataclass
class Scenario:
    fmt: str
    phase: str
    bowling: str  # "pace" or "spin"
    wicket_value: float
    shots: Shots
    uniforms: np.ndarray
    label: str = ""
    batter_id: str | None = None
    params: SimParams = field(default_factory=SimParams.load)

    @property
    def keeper_up(self) -> bool:
        return self.bowling == "spin"

    @property
    def situation(self) -> int:
        return situation_code(self.fmt, self.phase)

    def template(self) -> np.ndarray:
        return template(self.fmt, self.phase, self.bowling)

    def legal(self, field9: np.ndarray) -> np.ndarray:
        return rules.repair(field9, self.fmt, self.phase)


def make_scenario(
    rows: pl.DataFrame,
    model,
    fmt: str,
    phase: str,
    bowling: str,
    wicket_value: float,
    n: int = 30000,
    seed: int = 0,
    label: str = "",
    batter_id: str | None = None,
) -> Scenario:
    """Simulated shots for the given real deliveries, topped up to about n by resampling."""
    reps = max(1, int(np.ceil(n / max(rows.height, 1))))
    sample = model.sample(rows, n=reps, seed=seed)
    return Scenario(
        fmt=fmt,
        phase=phase,
        bowling=bowling,
        wicket_value=wicket_value,
        shots=Shots.from_sample(sample),
        uniforms=np.random.default_rng(seed + 1).random((sample.height, 4)),
        label=label,
        batter_id=batter_id,
    )


def outcomes(fields9: np.ndarray, scen: Scenario, legalise: bool = True):
    """Per-delivery runs and wickets for one or more fields. fields9: (9, 2) or (F, 9, 2).

    Returns runs (F, n) and wickets (F, n) as float arrays.
    """
    fields9 = np.asarray(fields9, dtype=float)
    if fields9.ndim == 2:
        fields9 = fields9[None]
    if legalise:
        fields9 = np.stack([scen.legal(f) for f in fields9])
    full = np.stack([with_keeper_and_bowler(f, scen.keeper_up) for f in fields9])
    n = len(scen.shots)
    runs = np.empty((len(full), n))
    wkts = np.empty((len(full), n))
    for i, f in enumerate(full):
        situation = np.full(n, scen.situation)
        r = simulate(scen.shots, f, None, scen.params, uniforms=scen.uniforms, situation=situation)
        runs[i] = r.runs
        wkts[i] = r.wicket
    return runs, wkts


def cost(fields9: np.ndarray, scen: Scenario, legalise: bool = True) -> np.ndarray:
    """Expected cost per ball for each field (lower is better for the fielding side)."""
    runs, wkts = outcomes(fields9, scen, legalise)
    return runs.mean(axis=1) - scen.wicket_value * wkts.mean(axis=1)


def summary(field9: np.ndarray, scen: Scenario) -> dict[str, float]:
    runs, wkts = outcomes(field9, scen)
    return {
        "runs_per_ball": float(runs.mean()),
        "wickets_per_ball": float(wkts.mean()),
        "cost": float(runs.mean() - scen.wicket_value * wkts.mean()),
    }


def paired_difference(a: np.ndarray, b: np.ndarray, scen: Scenario) -> tuple[float, float, float]:
    """Cost of field b minus field a per ball, with a 95% interval.

    Both fields face the same simulated deliveries, so the per-delivery differences are paired
    and independent across deliveries; the interval is mean +/- 1.96 standard errors.
    """
    runs, wkts = outcomes(np.stack([a, b]), scen)
    per_ball = (runs[1] - runs[0]) - scen.wicket_value * (wkts[1] - wkts[0])
    mean = float(per_ball.mean())
    half = 1.96 * float(per_ball.std(ddof=1)) / np.sqrt(len(per_ball))
    return mean, mean - half, mean + half

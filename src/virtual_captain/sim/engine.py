"""Simulate deliveries against a field.

    shots = Shots.from_sample(shot_model.sample(rows))
    result = simulate(shots, fields, field_idx, params)
    summary(result)

`fields` is (F, 11, 2): nine fielders plus keeper and bowler (see field.positions), and
`field_idx` says which field each delivery is bowled to.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF
from virtual_captain.sim import physics
from virtual_captain.sim.params import SimParams, situation_code


@dataclass
class Shots:
    outcome: np.ndarray  # int8: 0 no contact, 1 bowled/lbw/stumped, 2 in play, 3 six
    theta: np.ndarray  # radians
    dist: np.ndarray  # fraction of boundary radius
    aerial: np.ndarray  # bool

    @classmethod
    def from_sample(cls, sample: pl.DataFrame) -> "Shots":
        return cls(
            outcome=sample["outcome"].to_numpy().astype(np.int8),
            theta=sample["theta"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64),
            dist=sample["dist"].fill_nan(0.5).fill_null(0.5).to_numpy().astype(np.float64),
            aerial=sample["aerial"].fill_null(False).to_numpy().astype(np.bool_),
        )

    def __len__(self) -> int:
        return len(self.outcome)


@dataclass
class Result:
    runs: np.ndarray
    kind: np.ndarray

    @property
    def wicket(self) -> np.ndarray:
        return (self.kind == physics.CAUGHT) | (self.kind == physics.FIELD_FREE_OUT)


def simulate(
    shots: Shots,
    fields: np.ndarray,
    field_idx: np.ndarray | None = None,
    params: SimParams | None = None,
    seed: int = 0,
    uniforms: np.ndarray | None = None,
    radius: float = BOUNDARY_RADIUS,
    situation: tuple[str, str] | np.ndarray = ("T20", "middle"),
) -> Result:
    """situation is a (format, phase) pair for every delivery, or a per-delivery array of
    codes from params.SITUATIONS; it sets how readily the batters run."""
    params = params or SimParams.load()
    fields = np.ascontiguousarray(fields, dtype=np.float64)
    if fields.ndim == 2:
        fields = fields[None]
    n = len(shots)
    if field_idx is None:
        field_idx = np.zeros(n, dtype=np.int32)
    if uniforms is None:
        uniforms = np.random.default_rng(seed).random((n, 4))
    if isinstance(situation, tuple):
        codes = np.full(n, situation_code(*situation))
    else:
        codes = np.asarray(situation)
    run_time = params.run_times()[codes]
    start = params.starts()[codes]
    runs, kind = physics.simulate_batch(
        shots.outcome,
        shots.theta,
        shots.dist,
        shots.aerial,
        np.ascontiguousarray(field_idx, dtype=np.int32),
        np.ascontiguousarray(run_time, dtype=np.float64),
        np.ascontiguousarray(start, dtype=np.float64),
        fields,
        uniforms,
        float(radius),
        float(PITCH_HALF),
        params.to_array(),
    )
    return Result(runs, kind)


def summary(result: Result) -> dict[str, float]:
    runs, kind = result.runs, result.kind
    return {
        "runs_per_ball": float(runs.mean()),
        "dot": float(((runs == 0) & ~result.wicket).mean()),
        "one": float((runs == 1).mean()),
        "two_three": float(((runs == 2) | (runs == 3)).mean()),
        "four": float((kind == physics.FOUR).mean()),
        "six": float((kind == physics.SIX).mean()),
        "caught": float((kind == physics.CAUGHT).mean()),
        "wicket": float(result.wicket.mean()),
    }

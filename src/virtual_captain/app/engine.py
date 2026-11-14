"""The demo app's engine: the captain's field for a situation, what any field is worth against
a batter, and where one fielder would be worth most. NumPy and the numba simulator only.

Every field in a situation is scored on the same deliveries with the same random numbers, so
the difference between two fields is measured far more tightly than either field's own cost.
"""

from dataclasses import dataclass

import numpy as np

from virtual_captain.app.bundle import Bundle
from virtual_captain.field import rules
from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF, in_ring
from virtual_captain.field.positions import BOWLER, KEEPER_BACK, KEEPER_UP, xy
from virtual_captain.rl.state import FORMATS, moved_fields, observation, progress
from virtual_captain.sim import physics
from virtual_captain.sim.params import SimParams, situation_code

REPLAYS = 6  # each delivery is played this many times, with fresh fielding luck
INNINGS_OVERS = {"T20": 20, "ODI": 50}
# overs of each phase: the over about to be bowled in limited overs, overs since the new ball
# in Tests (a new ball is due after 80)
PHASE_OVERS = {
    ("T20", "powerplay"): (1, 6),
    ("T20", "middle"): (7, 15),
    ("T20", "death"): (16, 20),
    ("ODI", "powerplay"): (1, 10),
    ("ODI", "middle"): (11, 40),
    ("ODI", "death"): (41, 50),
    ("TEST", "new_ball"): (0, 19),
    ("TEST", "old_ball"): (20, 79),
}
CAPTAIN = "RL captain"
CONVENTIONAL = "Conventional field"
ALL_BATTERS = "Best field for all batters"
SEARCHED = "Dedicated search for this batter"


@dataclass(frozen=True)
class Situation:
    context: int
    wickets: int
    over: int  # see PHASE_OVERS


@dataclass
class Scored:
    """A field's outcomes on every simulated delivery, averaged over the replays."""

    runs: np.ndarray  # (n,)
    wickets: np.ndarray  # (n,)
    fours: np.ndarray
    sixes: np.ndarray
    dots: np.ndarray
    wicket_value: float

    @property
    def cost(self) -> np.ndarray:
        return self.runs - self.wicket_value * self.wickets

    def summary(self) -> dict[str, float]:
        return {
            "runs_per_ball": float(self.runs.mean()),
            "wickets_per_ball": float(self.wickets.mean()),
            "cost_per_ball": float(self.cost.mean()),
            "fours": float(self.fours.mean()),
            "sixes": float(self.sixes.mean()),
            "dots": float(self.dots.mean()),
        }

    def minus(self, other: "Scored") -> tuple[float, float]:
        """Mean cost per ball of this field minus another's, and a 95% half-width."""
        d = self.cost - other.cost
        return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


class Engine:
    def __init__(self, bundle: Bundle, params: SimParams | None = None):
        self.b = bundle
        params = params or SimParams.load()
        self.params = params.to_array()
        self.run_time = params.run_times()
        self.start = params.starts()
        b = bundle
        self.key_fmt = np.array([FORMATS.index(f) for f in b.key_format])
        self.key_spin = (b.key_bowling == "spin").astype(int)
        self.key_situation = np.array(
            [situation_code(f, p) for f, p in zip(b.key_format, b.key_phase, strict=True)]
        )
        self.key_max_out = np.array(
            [rules.max_outside_ring(f, p) for f, p in zip(b.key_format, b.key_phase, strict=True)]
        )
        self.key_max_leg = np.array([rules.max_leg_behind_square(f) for f in b.key_format])
        self.n_shots = b.shot_outcome.shape[1]
        self.uniforms = np.random.default_rng(0).random((REPLAYS, self.n_shots, 4))
        self.extra = {
            False: np.stack([xy(KEEPER_BACK), xy(BOWLER)]),
            True: np.stack([xy(KEEPER_UP), xy(BOWLER)]),
        }

    # situations ---------------------------------------------------------------------------
    def contexts_for(self, fmt: str, phase: str, bowling: str) -> np.ndarray:
        b = self.b
        return np.flatnonzero((b.format == fmt) & (b.phase == phase) & (b.bowling == bowling))

    def typical(self, c: int) -> Situation:
        b = self.b
        fmt, phase = str(b.format[c]), str(b.phase[c])
        lo, hi = PHASE_OVERS[(fmt, phase)]
        if fmt == "TEST":
            over = round(b.ball_age_median[c] / 6)
        else:
            over = INNINGS_OVERS[fmt] - int(np.ceil(b.balls_remaining_median[c] / 6)) + 1
        return Situation(c, int(round(b.wickets_median[c])), int(np.clip(over, lo, hi)))

    def _state(self, s: Situation) -> tuple[int, int, int]:
        """(format index, balls remaining in the innings, ball age) at the start of the over."""
        fmt = str(self.b.format[s.context])
        if fmt == "TEST":
            return 2, 0, 6 * s.over
        return FORMATS.index(fmt), 6 * (INNINGS_OVERS[fmt] - s.over + 1), 0

    def wicket_value(self, s: Situation) -> float:
        f, balls_remaining, _ = self._state(s)
        w = min(s.wickets, 9)
        if f == 2:
            return float(self.b.lam_test[w])
        overs_left = int(np.clip((balls_remaining + 5) // 6, 1, 50))
        return float(self.b.lam_limited[f, overs_left, w])

    def keeper_up(self, c: int) -> bool:
        return bool(self.key_spin[self.b.key[c]])

    # the captain --------------------------------------------------------------------------
    def observe(self, s: Situation, wicket_value: float | None = None) -> np.ndarray:
        c, k = s.context, self.b.key[s.context]
        f, balls_remaining, ball_age = self._state(s)
        lam = self.wicket_value(s) if wicket_value is None else wicket_value
        one = np.ones(1)
        prof = self.b.profiles[c][None]
        return observation(
            np.array([f]), self.key_spin[k][None], self.key_situation[k][None],
            s.wickets * one, progress(np.array([f]), balls_remaining * one, ball_age * one),
            lam * one, 0 * one, prof, prof,
        )  # fmt: skip

    def act(self, obs: np.ndarray) -> np.ndarray:
        b = self.b
        h = np.tanh(obs @ b.w0 + b.b0)
        h = np.tanh(h @ b.w1 + b.b1)
        return np.clip(h @ b.w2 + b.b2, -1.0, 1.0)

    def captain_field(self, s: Situation, wicket_value: float | None = None) -> np.ndarray:
        k = self.b.key[s.context]
        moved = moved_fields(self.b.optimised[k][None], self.act(self.observe(s, wicket_value)))
        return rules.repair_many(moved, self.key_max_out[k][None], self.key_max_leg[k][None])[0]

    def fields(self, s: Situation) -> dict[str, np.ndarray]:
        """The captain's field and the ones to compare it with."""
        c = s.context
        k = self.b.key[c]
        out = {
            CAPTAIN: self.captain_field(s),
            CONVENTIONAL: self.b.conventional[k],
            ALL_BATTERS: self.b.optimised[k],
        }
        if not self.b.pooled[c]:
            out[SEARCHED] = self.b.searched[c]
        return out

    # rules --------------------------------------------------------------------------------
    def check(self, s: Situation, field: np.ndarray) -> rules.Check:
        c = s.context
        return rules.check(field, str(self.b.format[c]), str(self.b.phase[c]))

    def repair(self, s: Situation, field: np.ndarray) -> np.ndarray:
        c = s.context
        return rules.repair(field, str(self.b.format[c]), str(self.b.phase[c]))

    # scoring ------------------------------------------------------------------------------
    def _simulate(self, c: int, fields: np.ndarray, n: int, replays: int):
        """Every field (F, 9, 2) against the first n deliveries of context c, each played
        `replays` times. Returns runs and kind, (F, replays, n)."""
        b = self.b
        n_fields = len(fields)
        extra = self.extra[self.keeper_up(c)]
        full = np.concatenate([fields, np.broadcast_to(extra, (n_fields, 2, 2))], axis=1)
        total = n_fields * replays * n
        situation = self.key_situation[b.key[c]]

        def tile(a):
            return np.tile(np.asarray(a[c, :n]), n_fields * replays)

        runs, kind = physics.simulate_batch(
            tile(b.shot_outcome),
            tile(b.shot_theta).astype(np.float64),
            tile(b.shot_dist).astype(np.float64),
            tile(b.shot_aerial),
            np.repeat(np.arange(n_fields, dtype=np.int32), replays * n),
            np.full(total, self.run_time[situation]),
            np.full(total, self.start[situation]),
            np.ascontiguousarray(full, dtype=np.float64),
            np.tile(self.uniforms[:replays, :n].reshape(-1, 4), (n_fields, 1)),
            float(BOUNDARY_RADIUS),
            float(PITCH_HALF),
            self.params,
        )
        shape = (n_fields, replays, n)
        return runs.reshape(shape), kind.reshape(shape)

    def score(self, s: Situation, fields: dict[str, np.ndarray]) -> dict[str, Scored]:
        """Each field against about 50,000 simulated balls from this batter."""
        names = list(fields)
        runs, kind = self._simulate(
            s.context, np.stack([fields[k] for k in names]), self.n_shots, REPLAYS
        )
        wkt = (kind == physics.CAUGHT) | (kind == physics.FIELD_FREE_OUT)
        lam = self.wicket_value(s)
        return {
            name: Scored(
                runs=runs[i].mean(0),
                wickets=wkt[i].mean(0),
                fours=(kind[i] == physics.FOUR).mean(0),
                sixes=(kind[i] == physics.SIX).mean(0),
                dots=((runs[i] == 0) & ~wkt[i]).mean(0),
                wicket_value=lam,
            )
            for i, name in enumerate(names)
        }

    def value_map(self, s: Situation, field: np.ndarray, i: int, spacing: float = 5.0,
                  n: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
        """Change in cost per ball if fielder i stood at each point of a grid instead of where
        they are, with the rest of the field as it is. Returns the points (P, 2), the change
        (P,) and whether the field would still be legal (P,)."""
        c = s.context
        k = self.b.key[c]
        axis = np.arange(-BOUNDARY_RADIUS, BOUNDARY_RADIUS + 1e-9, spacing)
        gx, gy = np.meshgrid(axis, axis)
        inside = np.hypot(gx, gy) <= rules.MAX_RADIUS
        points = np.column_stack([gx[inside], gy[inside]])
        fields = np.repeat(field[None], len(points) + 1, axis=0)
        fields[1:, i] = points
        runs, kind = self._simulate(c, fields, min(n, self.n_shots), 1)
        wkt = (kind == physics.CAUGHT) | (kind == physics.FIELD_FREE_OUT)
        cost = (runs - self.wicket_value(s) * wkt).mean(axis=(1, 2))
        change = cost[1:] - cost[0]

        others = np.delete(field, i, axis=0)
        px, py = points[:, 0], points[:, 1]
        n_out = rules.outside_ring(others).sum() + ~in_ring(px, py)
        n_leg = rules.leg_behind_square(others).sum() + ((px < 0) & (py < rules.POPPING_CREASE_Y))
        on_pitch = (np.abs(px) < rules.PITCH_HALF_WIDTH) & (np.abs(py) < PITCH_HALF)
        near = np.hypot(px, py + PITCH_HALF) < rules.MIN_FROM_STRIKER
        legal = (n_out <= self.key_max_out[k]) & (n_leg <= self.key_max_leg[k]) & ~on_pitch & ~near
        return points, change, legal

    # wagon wheels -------------------------------------------------------------------------
    def simulated_wagon(self, s: Situation, field: np.ndarray, n: int = 1500):
        """Simulated shots that stay in the ground, with the runs they bring against `field`:
        (theta, dist, runs) for the first n deliveries put into play."""
        c = s.context
        runs, kind = self._simulate(c, field[None], self.n_shots, 1)
        in_play = self.b.shot_outcome[c] == 2
        theta = self.b.shot_theta[c][in_play].astype(float)[:n]
        dist = self.b.shot_dist[c][in_play].astype(float)[:n]
        out = (kind[0, 0][in_play] == physics.CAUGHT)[:n]
        return theta, dist, np.where(out, -1, runs[0, 0][in_play][:n])

    def real_wagon(self, c: int):
        b = self.b
        a, z = b.wagon_offsets[c], b.wagon_offsets[c + 1]
        return (
            b.wagon_theta[a:z].astype(float),
            b.wagon_dist[a:z].astype(float),
            b.wagon_runs[a:z].astype(int),
        )

    def simulated_six_share(self, c: int) -> float:
        return float((self.b.shot_outcome[c] == 3).mean())

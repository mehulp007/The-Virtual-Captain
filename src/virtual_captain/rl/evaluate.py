"""Judging the captain against the fields it is meant to beat.

Two tests:

  one batter, one situation   The field the captain sets for a batter (on strike at both ends,
                              in the typical state for that context, wicket valued as for the
                              phase) is scored like every field in the optimiser study: on the
                              batter's judging bank, which nothing was trained or searched on.
                              The bar is the dedicated CMA-ES search for that batter.

  overs as they unfold        Whole overs from real match states, two batters at the crease,
                              strike changing, wickets falling and the wicket's value moving
                              with them. Every way of setting the field plays exactly the same
                              overs: same starting states, same batters, same random numbers.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF
from virtual_captain.rl.data import CaptainData
from virtual_captain.rl.env import EPISODE_BALLS, Overs
from virtual_captain.sim import physics

# A field-setter looks at the overs in play and returns (n, 9, 2) legal fields.
FieldSetter = Callable[[Overs], np.ndarray]


def conventional(overs: Overs) -> np.ndarray:
    return overs.data.conventional_fields[overs.key]


def optimised_for_all(overs: Overs) -> np.ndarray:
    """One field per situation, optimised for all batters together."""
    return overs.data.pooled_fields[overs.key]


def searched_for_striker(overs: Overs) -> np.ndarray:
    """Switch to the field a dedicated search found for whoever is on strike."""
    return overs.data.searched_fields[overs.striker]


def captain(model, deterministic: bool = True) -> FieldSetter:
    def setter(overs: Overs) -> np.ndarray:
        actions, _ = model.predict(overs.observe(), deterministic=deterministic)
        return overs.fields_from_actions(actions)

    return setter


# One batter, one situation --------------------------------------------------------------
def typical_overs(data: CaptainData, contexts: np.ndarray, **kwargs) -> Overs:
    """Overs placed in each context's typical state, its batter at both ends."""
    contexts = np.asarray(contexts)
    overs = Overs(data, n=len(contexts), **kwargs)
    ctx = data.contexts
    overs.place(
        contexts,
        contexts,
        ctx["wickets_median"].to_numpy()[contexts].round().astype(int),
        ctx["balls_remaining_median"].to_numpy()[contexts].round().astype(int),
        ctx["ball_age_median"].to_numpy()[contexts].round().astype(int),
    )
    return overs


def captain_fields(model, data: CaptainData, contexts: np.ndarray) -> np.ndarray:
    """The captain's field for each context, with the wicket valued as for the phase."""
    overs = typical_overs(data, contexts, shots_per_ball=1)
    phase_value = overs.key_phase_value[overs.key]
    actions, _ = model.predict(overs.observe(wicket_value=phase_value), deterministic=True)
    return overs.fields_from_actions(actions)


@dataclass
class Judged:
    runs: np.ndarray  # (C, n) per simulated delivery
    wickets: np.ndarray  # (C, n)
    wicket_value: np.ndarray  # (C,)

    @property
    def cost(self) -> np.ndarray:
        """(C, n) runs - wicket value x wickets, per delivery."""
        return self.runs - self.wicket_value[:, None] * self.wickets


def judge_fields(
    data: CaptainData, contexts: np.ndarray, fields: np.ndarray, n: int = 20000, seed: int = 0
) -> Judged:
    """Score fields (C, 9, 2), one per context, on the first n deliveries of each context's
    judging bank. The same deliveries and random numbers are used whatever the field, so two
    sets of fields can be compared delivery by delivery."""
    contexts = np.asarray(contexts)
    overs = Overs(data, n=1, shots_per_ball=1)
    bank = data.judge
    n = int(min(n, bank.lengths[contexts].min()))
    idx = (bank.offsets[contexts][:, None] + np.arange(n)[None]).ravel()
    keys = overs.ctx_key[contexts]
    full = np.concatenate([fields, overs.key_extra[keys]], axis=1)
    situation = np.repeat(overs.key_situation[keys], n)
    uniforms = np.random.default_rng(seed).random((n, 4))
    runs, kind = physics.simulate_batch(
        bank.outcome[idx],
        bank.theta[idx],
        bank.dist[idx],
        bank.aerial[idx],
        np.repeat(np.arange(len(contexts), dtype=np.int32), n),
        overs.run_time[situation],
        overs.start_offset[situation],
        np.ascontiguousarray(full, dtype=np.float64),
        np.tile(uniforms, (len(contexts), 1)),
        float(BOUNDARY_RADIUS),
        float(PITCH_HALF),
        overs.params,
    )
    wickets = (kind == physics.CAUGHT) | (kind == physics.FIELD_FREE_OUT)
    return Judged(
        runs=runs.reshape(len(contexts), n).astype(np.float32),
        wickets=wickets.reshape(len(contexts), n).astype(np.float32),
        wicket_value=overs.key_phase_value[keys],
    )


def paired(a: Judged, b: Judged) -> tuple[np.ndarray, np.ndarray]:
    """Per context: mean cost of b minus a per ball, and the half-width of its 95% interval."""
    d = b.cost - a.cost
    return d.mean(1), 1.96 * d.std(1, ddof=1) / np.sqrt(d.shape[1])


# Overs as they unfold -------------------------------------------------------------------
@dataclass
class Played:
    cost: np.ndarray  # (E,) expected cost of the over, summed over its balls
    runs: np.ndarray  # (E,) expected runs
    wickets: np.ndarray  # (E,) expected wickets
    balls: np.ndarray  # (E,)
    natural: np.ndarray | None = None  # (E,) the cost had the batter played naturally, ball
    # by ball against the same fields (only when a batter with an intent is given)


def play(
    setter: FieldSetter,
    data: CaptainData,
    fmt: str,
    n_overs: int,
    batters: str = "train",
    generic_share: float | None = None,
    seed: int = 0,
    shots_per_ball: int = 256,
    batter=None,
) -> Played:
    """Play n_overs overs (spells in Tests) of one format with a field-setter. `batter` (see
    rl.multi_env) answers each field with an intent; without one batters play naturally."""
    kwargs = {} if generic_share is None else {"generic_share": generic_share}
    overs = Overs(
        data, n=n_overs, bank="judge", batters=batters, formats=(fmt,), seed=seed,
        shots_per_ball=shots_per_ball, **kwargs,
    )  # fmt: skip
    live = np.ones(n_overs, dtype=bool)
    cost, runs, wkts, balls, nat = (np.zeros(n_overs) for _ in range(5))
    for _ in range(EPISODE_BALLS[fmt]):
        fields = setter(overs)
        intent = None if batter is None else batter(overs, fields)
        reward, done, info = overs.step(fields, intent=intent)
        cost[live] -= reward[live]
        runs[live] += info["expected_runs"][live]
        wkts[live] += info["expected_wickets"][live]
        if batter is not None:
            nat[live] += info["natural_cost"][live]
        balls[live] += 1
        live &= ~done
    return Played(cost, runs, wkts, balls, None if batter is None else nat)


def compare(results: dict[str, Played], base: str) -> dict[str, tuple[float, float]]:
    """Mean per-ball cost difference of every setter against `base`, with a 95% half-width.
    Overs are paired: each was played by every setter from the same start."""
    ref = results[base]
    out = {}
    for name, r in results.items():
        d = (r.cost - ref.cost) / np.maximum(ref.balls, 1)
        out[name] = (float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))))
    return out

"""The captain's game: set the field for every ball of an over (a four-over spell in Tests).

State    who is on strike and who is at the other end (their profiles), format, phase, type of
         bowling, wickets down, how far through the innings, what a wicket is worth right now,
         and how far through the over.
Action   18 numbers in [-1, 1], two per fielder: how far to move them round the ground and in or
         out from the field optimised for all batters in this situation. All zeros is that
         field. The result is repaired into a legal field.
Reward   minus the expected cost of the ball, runs - wicket value x wickets, averaged over
         SHOTS_PER_BALL simulated deliveries from the striker's bank. The first of them is what
         actually happens: it decides the runs, the strike and any wicket for the next ball.

Averaging over many deliveries keeps the reward's noise low without changing what it rewards:
the expected cost of a field is the same either way. For training the reward can also be made
relative (`relative=True`): the field's cost is taken away from the cost of the starting field
on the very same deliveries, which cuts the noise by about three and a half times. That
reward no longer credits the captain for which batter ends up on strike; the true cost is
what every field is judged on.

`Overs` plays many overs side by side with one call to the simulator per ball. `CaptainEnv`
is the standard single-over Gymnasium environment built on it, and `CaptainVecEnv` the batched
environment PPO trains on.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from virtual_captain.data.coords import wagon_to_xy, xy_to_wagon
from virtual_captain.field import rules
from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF
from virtual_captain.field.positions import BOWLER, KEEPER_BACK, KEEPER_UP, xy
from virtual_captain.field.templates import KEYS
from virtual_captain.rl.batter import attack_shift, forcing_chance, intent_weights
from virtual_captain.rl.data import N_PROFILE, CaptainData
from virtual_captain.sim import physics
from virtual_captain.sim.params import SimParams, situation_code

FORMATS = ("T20", "ODI", "TEST")
EPISODE_BALLS = {"T20": 6, "ODI": 6, "TEST": 24}
INNINGS_BALLS = {"T20": 120, "ODI": 300}
NEW_BALL_EVERY = 480
SHOTS_PER_BALL = 256
GENERIC_SHARE = 0.25  # chance a new batter is "all batters", a batter with no profile of their own
R_MIN, R_MAX = 5.0, 66.0  # fielders' distance from the pitch centre, as in optim.cma_search
STEP = np.tile([0.25, 0.3], 9)  # largest move per unit of action: 90 degrees round, 18 m in/out
N_OBS = 3 + 2 + 8 + 4 + 2 * N_PROFILE
WICKET_SCALE = 40.0
OBS_LIMIT = 10.0  # standardised profiles are clipped to +/- this


def encode_fields(fields: np.ndarray) -> np.ndarray:
    """(B, 9, 2) fields -> (B, 18): wagon-wheel angle / 360 and scaled distance per fielder."""
    angle, dist = xy_to_wagon(fields[..., 0], fields[..., 1])
    r = np.clip(dist * BOUNDARY_RADIUS, R_MIN, R_MAX)
    z = np.stack([angle / 360.0, (r - R_MIN) / (R_MAX - R_MIN)], axis=-1)
    return z.reshape(len(fields), 18)


def decode_fields(z: np.ndarray) -> np.ndarray:
    """(B, 18) -> (B, 9, 2). Angles wrap round the ground; distances are clipped."""
    z = z.reshape(len(z), 9, 2)
    angle = np.mod(z[..., 0], 1.0) * 360.0
    r = R_MIN + np.clip(z[..., 1], 0.0, 1.0) * (R_MAX - R_MIN)
    x, y = wagon_to_xy(angle, r / BOUNDARY_RADIUS)
    return np.stack([x, y], axis=-1)


class Overs:
    """n overs (or spells) played side by side."""

    def __init__(
        self,
        data: CaptainData,
        n: int,
        bank: str = "learn",
        batters: str = "train",
        formats: tuple[str, ...] = FORMATS,
        format_weights: tuple[float, ...] | None = None,
        generic_share: float = GENERIC_SHARE,
        shots_per_ball: int = SHOTS_PER_BALL,
        relative: bool = False,
        seed: int = 0,
    ):
        self.data = data
        self.relative = relative
        self.n = n
        self.k = shots_per_ball
        self.bank = data.learn if bank == "learn" else data.judge
        self.generic_share = generic_share
        self.rng = np.random.default_rng(seed)
        self.set_formats(formats, format_weights)

        ctx = data.contexts
        keys = data.keys
        self.key_format = np.array([FORMATS.index(f) for f in keys["format"]])
        self.key_bowling = np.array([b == "spin" for b in keys["bowling"]], dtype=int)
        self.key_situation = np.array([situation_code(f, p) for f, p, _ in KEYS])
        self.key_max_out = np.array([rules.max_outside_ring(f, p) for f, p, _ in KEYS])
        self.key_max_leg = np.array([rules.max_leg_behind_square(f) for f, _, _ in KEYS])
        self.key_phase_value = keys["wicket_value"].to_numpy()
        self.base_z = encode_fields(data.pooled_fields)

        keeper_back, keeper_up, bowler = xy(KEEPER_BACK), xy(KEEPER_UP), xy(BOWLER)
        self.key_extra = np.stack(
            [np.stack([keeper_up if s else keeper_back, bowler]) for s in self.key_bowling]
        )  # (16, 2, 2): keeper and bowler

        params = SimParams.load()
        self.params = params.to_array()
        self.run_time = params.run_times()
        self.start_offset = params.starts()

        # who can come in to bat, by situation
        named = ~ctx["pooled"].to_numpy()
        unseen = ctx["unseen"].to_numpy()
        allowed = {"train": named & ~unseen, "unseen": named & unseen, "all": named}[batters]
        key_of = ctx["key"].to_numpy()
        self.pooled_ctx = np.array(
            [int(ctx.filter(ctx["pooled"] & (ctx["key"] == k))["context"][0]) for k in range(16)]
        )
        self.pool = [np.flatnonzero(allowed & (key_of == k)) for k in range(16)]

        # batter profiles, standardised over the named batters
        prof = data.profiles
        mean, std = prof[named].mean(0), prof[named].std(0) + 1e-6
        self.profiles = ((prof - mean) / std).astype(np.float32)
        self.ctx_key = key_of

        # match states overs start from
        st = data.starts
        self.start_format = np.array([FORMATS.index(f) for f in st["format"]])
        self.start_key = st["key"].to_numpy()
        self.start_balls = st["balls_remaining"].fill_null(0).to_numpy()
        self.start_wickets = st["wickets_before"].to_numpy()
        self.start_age = st["ball_age"].to_numpy()
        self.starts_by_format = [np.flatnonzero(self.start_format == f) for f in range(3)]

        # wicket value by state
        ws = data.wicket_states
        self.lam_limited = np.zeros((2, 51, 10))
        for f, fmt in enumerate(("T20", "ODI")):
            part = ws.filter(ws["format"] == fmt)
            o, w = part["overs_left"].to_numpy(), part["wickets_before"].to_numpy()
            self.lam_limited[f, o, w] = part["wicket_value"].to_numpy()
        tests = ws.filter(ws["format"] == "TEST").sort("wickets_before")
        self.lam_test = tests["wicket_value"].to_numpy()

        # extra dismissal chance per nat of forcing, by format (rl.batter)
        self.per_attack, self.per_shift = np.zeros(3), np.zeros(3)
        self.max_shift = np.full(3, np.inf)
        if data.forcing_risk is not None:
            risk = {r["format"]: r for r in data.forcing_risk.iter_rows(named=True)}
            self.per_attack = np.array([risk[f]["per_attack"] for f in FORMATS])
            self.per_shift = np.array([risk[f]["per_shift"] for f in FORMATS])
            self.max_shift = np.array([risk[f]["max_shift"] for f in FORMATS])

        z = np.zeros(n, dtype=np.int64)
        self.key, self.striker, self.other = z.copy(), z.copy(), z.copy()
        self.wickets, self.balls_remaining, self.ball_age = z.copy(), z.copy(), z.copy()
        self.ball, self.length = z.copy(), z.copy()
        self.ep_return = np.zeros(n)
        self.reset_all()

    # setup ------------------------------------------------------------------------------
    def set_formats(self, formats, weights=None) -> None:
        w = np.array(weights if weights is not None else [1.0] * len(formats), dtype=float)
        self.format_p = np.zeros(3)
        for f, wf in zip(formats, w, strict=True):
            self.format_p[FORMATS.index(f)] = wf
        self.format_p /= self.format_p.sum()

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _new_batter(self, keys: np.ndarray, u: np.ndarray, avoid: np.ndarray) -> np.ndarray:
        out = np.empty(len(keys), dtype=np.int64)
        for i, (k, (u_kind, u_pick)) in enumerate(zip(keys, u, strict=True)):
            pool = self.pool[k]
            if u_kind < self.generic_share or len(pool) == 0:
                out[i] = self.pooled_ctx[k]
                continue
            if len(pool) > 1:
                pool = pool[pool != avoid[i]]
            out[i] = pool[int(u_pick * len(pool))]
        return out

    def reset(self, idx: np.ndarray) -> None:
        m = len(idx)
        fmt = self.rng.choice(3, size=m, p=self.format_p)
        pick = self.rng.random(m)
        u = self.rng.random((m, 2, 2))
        rows = np.array(
            [
                self.starts_by_format[f][int(p * len(self.starts_by_format[f]))]
                for f, p in zip(fmt, pick, strict=True)
            ]
        )
        key = self.start_key[rows]
        self.key[idx] = key
        self.wickets[idx] = self.start_wickets[rows]
        self.balls_remaining[idx] = self.start_balls[rows]
        self.ball_age[idx] = self.start_age[rows]
        self.ball[idx] = 0
        self.length[idx] = [EPISODE_BALLS[FORMATS[f]] for f in fmt]
        self.ep_return[idx] = 0.0
        self.striker[idx] = self._new_batter(key, u[:, 0], np.full(m, -1))
        self.other[idx] = self._new_batter(key, u[:, 1], self.striker[idx])

    def reset_all(self) -> None:
        self.reset(np.arange(self.n))

    def place(self, contexts: np.ndarray, others: np.ndarray, wickets, balls_remaining, ball_age):
        """Put every over in a given state (used to ask the captain about one situation)."""
        self.key[:] = self.ctx_key[contexts]
        self.striker[:], self.other[:] = contexts, others
        self.wickets[:] = wickets
        self.balls_remaining[:] = balls_remaining
        self.ball_age[:] = ball_age
        self.ball[:] = 0
        self.length[:] = [EPISODE_BALLS[FORMATS[f]] for f in self.key_format[self.key]]

    # state --------------------------------------------------------------------------------
    def wicket_value(self) -> np.ndarray:
        fmt = self.key_format[self.key]
        w = np.minimum(self.wickets, 9)
        overs_left = np.clip((self.balls_remaining + 5) // 6, 1, 50)
        limited = self.lam_limited[np.minimum(fmt, 1), overs_left, w]
        return np.where(fmt == 2, self.lam_test[w], limited)

    def observe(self, wicket_value: np.ndarray | None = None) -> np.ndarray:
        fmt = self.key_format[self.key]
        lam = self.wicket_value() if wicket_value is None else wicket_value
        innings = np.array([INNINGS_BALLS.get(f, 1) for f in FORMATS])[fmt]
        progress = np.where(
            fmt == 2, self.ball_age / NEW_BALL_EVERY, 1.0 - self.balls_remaining / innings
        )
        obs = np.zeros((self.n, N_OBS), dtype=np.float32)
        obs[np.arange(self.n), fmt] = 1.0
        obs[np.arange(self.n), 3 + self.key_bowling[self.key]] = 1.0
        obs[np.arange(self.n), 5 + self.key_situation[self.key]] = 1.0
        obs[:, 13] = self.wickets / 10.0
        obs[:, 14] = progress
        obs[:, 15] = lam / WICKET_SCALE
        obs[:, 16] = self.ball / self.length
        obs[:, 17 : 17 + N_PROFILE] = self.profiles[self.striker]
        obs[:, 17 + N_PROFILE :] = self.profiles[self.other]
        return np.clip(obs, -OBS_LIMIT, OBS_LIMIT)

    # acting -------------------------------------------------------------------------------
    def fields_from_actions(self, actions: np.ndarray) -> np.ndarray:
        """Legal (n, 9, 2) fields from (n, 18) actions in [-1, 1]."""
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        fields = decode_fields(self.base_z[self.key] + a * STEP)
        return rules.repair_many(fields, self.key_max_out[self.key], self.key_max_leg[self.key])

    def step(
        self,
        fields: np.ndarray,
        wicket_value: np.ndarray | None = None,
        intent: np.ndarray | None = None,
        start_intent: np.ndarray | None = None,
    ):
        """Bowl one ball in every over to the given (n, 9, 2) fields, which must be legal.

        `intent` (n, 3) is the striker's intent against these fields (rl.batter); without one
        the batter plays their natural game. `start_intent` is their intent against the
        starting field, for the relative reward.

        Returns reward (n,), done (n,) and a dict of per-over arrays: the expected runs and
        wickets of the ball, the runs and wicket that actually happened, and with an intent the
        ball's expected cost, what it would have cost had the batter played naturally, and the
        extra dismissal chance the batter took on by forcing their game.
        """
        n, k = self.n, self.k
        lam = self.wicket_value() if wicket_value is None else wicket_value
        full = np.concatenate([fields, self.key_extra[self.key]], axis=1)

        a = self.bank.offsets[self.striker]
        length = self.bank.lengths[self.striker]
        u_pick = self.rng.random((n, k))
        idx = (a[:, None] + (u_pick * length[:, None]).astype(np.int64)).ravel()
        uniforms = self.rng.random((n * k, 4))
        u_new = self.rng.random((n, 2))
        adaptive = intent is not None or start_intent is not None
        u_adapt = self.rng.random((n, 2)) if adaptive else None  # which delivery, forced error
        situation = np.repeat(self.key_situation[self.key], k)
        copies = 2 if self.relative else 1  # the starting field faces the same deliveries
        if self.relative:
            start = self.data.pooled_fields[self.key]
            full = np.concatenate([full, np.concatenate([start, self.key_extra[self.key]], axis=1)])
        runs, kind = physics.simulate_batch(
            np.tile(self.bank.outcome[idx], copies),
            np.tile(self.bank.theta[idx], copies),
            np.tile(self.bank.dist[idx], copies),
            np.tile(self.bank.aerial[idx], copies),
            np.repeat(np.arange(copies * n, dtype=np.int32), k),
            np.tile(self.run_time[situation], copies),
            np.tile(self.start_offset[situation], copies),
            np.ascontiguousarray(full),
            np.tile(uniforms, (copies, 1)),
            float(BOUNDARY_RADIUS),
            float(PITCH_HALF),
            self.params,
        )
        runs = runs.reshape(copies, n, k).astype(np.float64)
        wicket = ((kind == physics.CAUGHT) | (kind == physics.FIELD_FREE_OUT)).reshape(copies, n, k)
        info = {}
        if adaptive:
            shots = (
                self.bank.outcome[idx].reshape(n, k),
                self.bank.theta[idx].reshape(n, k),
                self.bank.dist[idx].reshape(n, k),
                self.bank.aerial[idx].reshape(n, k),
            )
            fmt = self.key_format[self.key]
            exp_runs, exp_wkts, forcing, weights = [], [], [], []
            for c, it in enumerate((intent, start_intent)[:copies]):
                it = np.zeros((n, 3)) if it is None else it
                w, kl = intent_weights(it, *shots, max_shift=self.max_shift[fmt])
                f = forcing_chance(
                    attack_shift(w, shots[0], shots[2], shots[3]), kl,
                    self.per_attack[fmt], self.per_shift[fmt],
                )  # fmt: skip
                exp_runs.append((1 - f) * (w * runs[c]).sum(1))
                exp_wkts.append(f + (1 - f) * (w * wicket[c]).sum(1))
                forcing.append(f)
                weights.append(w)
            exp_runs, exp_wkts = np.array(exp_runs), np.array(exp_wkts)
            cost = exp_runs - lam * exp_wkts
            info["natural_cost"] = runs[0].mean(1) - lam * wicket[0].mean(1)
            info["forcing"] = forcing[0]
            # the delivery that actually happens is drawn by the batter's weights
            j = (weights[0].cumsum(1) < u_adapt[:, :1]).sum(1).clip(max=k - 1)
            forced_out = u_adapt[:, 1] < forcing[0]
            r0 = np.where(forced_out, 0, runs[0, np.arange(n), j]).astype(np.int64)
            w0 = forced_out | wicket[0, np.arange(n), j]
        else:
            cost = runs.mean(2) - lam * wicket.mean(2)
            exp_runs, exp_wkts = runs.mean(2), wicket.mean(2)
            # the first delivery is what actually happens
            r0, w0 = runs[0, :, 0].astype(np.int64), wicket[0, :, 0]
        reward = cost[1] - cost[0] if self.relative else -cost[0]
        exp_runs, exp_wkts = exp_runs[0], exp_wkts[0]

        # what actually happened decides the next ball
        out = np.flatnonzero(w0)
        self.wickets[out] += 1
        if len(out):  # the new batter takes strike
            self.striker[out] = self._new_batter(self.key[out], u_new[out], self.other[out])
        swap = ~w0 & (r0 % 2 == 1)
        self.ball += 1
        self.balls_remaining = np.maximum(self.balls_remaining - 1, 0)
        self.ball_age += 1
        swap ^= self.ball % 6 == 0  # change of ends at the end of the over
        self.striker[swap], self.other[swap] = self.other[swap], self.striker[swap].copy()

        self.ep_return += reward
        done = (self.ball >= self.length) | (self.wickets >= 10)
        info |= {"runs": r0, "wicket": w0, "expected_runs": exp_runs, "expected_wickets": exp_wkts}
        if adaptive:
            info["cost"] = cost[0]
        return reward, done, info


def _spaces():
    obs = gym.spaces.Box(-OBS_LIMIT, OBS_LIMIT, shape=(N_OBS,), dtype=np.float32)
    act = gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=np.float32)
    return obs, act


class CaptainEnv(gym.Env):
    """One over (or Test spell) at a time."""

    metadata = {"render_modes": []}

    def __init__(self, data: CaptainData, **kwargs):
        self.observation_space, self.action_space = _spaces()
        self.overs = Overs(data, n=1, **kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.overs.seed(seed)
        self.overs.reset_all()
        return self.overs.observe()[0], {}

    def step(self, action):
        fields = self.overs.fields_from_actions(np.asarray(action)[None])
        reward, done, info = self.overs.step(fields)
        obs = self.overs.observe()[0]
        info = {k: v[0].item() for k, v in info.items()} | {"field": fields[0]}
        return obs, float(reward[0]), bool(done[0]), False, info


class CaptainVecEnv(VecEnv):
    """n overs at once for training; finished overs restart straight away."""

    def __init__(self, data: CaptainData, n_envs: int = 64, **kwargs):
        obs_space, act_space = _spaces()
        self.render_mode = None
        super().__init__(n_envs, obs_space, act_space)
        self.overs = Overs(data, n=n_envs, **kwargs)
        self._actions = None

    def set_formats(self, formats, weights=None) -> None:
        self.overs.set_formats(formats, weights)

    def reset(self):
        self.overs.reset_all()
        return self.overs.observe()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        o = self.overs
        fields = o.fields_from_actions(self._actions)
        reward, done, info = o.step(fields)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        finished = np.flatnonzero(done)
        if len(finished):
            last = o.observe()
            for i in finished:
                infos[i]["terminal_observation"] = last[i]
                infos[i]["episode"] = {"r": float(o.ep_return[i]), "l": int(o.ball[i])}
            o.reset(finished)
        return o.observe(), reward.astype(np.float32), done, infos

    def close(self):
        pass

    def seed(self, seed=None):
        self.overs.seed(0 if seed is None else seed)
        return [seed] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

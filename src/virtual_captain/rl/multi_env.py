"""Captain against batter.

Before every ball the captain sets the field, then the batter on strike sees it and picks an
intent (rl.batter). The game is zero sum: the captain pays the ball's cost, runs minus the
wicket's value times wickets, and the batter gains it.

  CaptainBatterGame   the two-player game as a standard PettingZoo AEC environment, one over
                      (or Test spell) at a time
  BatterVecEnv        many overs at once from the batter's side, against a population of
                      captains, for training batters with PPO
  SelfPlayCaptainEnv  many overs at once from the captain's side, against a population of
                      batters

For training, both sides are rewarded relative to a baseline on the very same deliveries, which
keeps the reward's noise low (see rl.env). The batter is paid what its intent gained over
playing its natural game against the same field. The captain is paid what its field saved
against the starting field, with the batter answering each field in its own way.
"""

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from virtual_captain.rl.batter import INTENT_DIM, N_BATTER_OBS, N_STATE, field_coverage
from virtual_captain.rl.data import N_PROFILE, CaptainData
from virtual_captain.rl.env import OBS_LIMIT, CaptainVecEnv, Overs, _spaces

FieldSetter = Callable[[Overs], np.ndarray]  # overs -> (n, 9, 2) legal fields
Batter = Callable[[Overs, np.ndarray], np.ndarray]  # overs, fields -> (n, 3) intents


def batter_observe(overs: Overs, fields: np.ndarray) -> np.ndarray:
    """What the striker sees: the match state, their own profile and the field."""
    seen = overs.observe()[:, : N_STATE + N_PROFILE]
    return np.concatenate([seen, field_coverage(fields)], axis=1)


def natural(overs: Overs, fields: np.ndarray) -> np.ndarray:
    """The batter's natural game, whatever the field."""
    return np.zeros((overs.n, INTENT_DIM))


def batter_policy(model, deterministic: bool = True) -> Batter:
    def act(overs: Overs, fields: np.ndarray) -> np.ndarray:
        intents, _ = model.predict(batter_observe(overs, fields), deterministic=deterministic)
        return intents

    return act


def _batter_spaces():
    obs = gym.spaces.Box(-OBS_LIMIT, OBS_LIMIT, shape=(N_BATTER_OBS,), dtype=np.float32)
    act = gym.spaces.Box(-1.0, 1.0, shape=(INTENT_DIM,), dtype=np.float32)
    return obs, act


class Population:
    """Opponents to train against. Each over is played against one member, drawn at random
    (with the given weights) when the over starts."""

    def __init__(
        self, members: Sequence[Callable], weights: Sequence[float] | None = None, seed: int = 0
    ):
        self.members = list(members)
        w = np.ones(len(self.members)) if weights is None else np.asarray(weights, dtype=float)
        self.p = w / w.sum()
        self.slot = np.zeros(0, dtype=np.int64)
        self.rng = np.random.default_rng(seed)  # its own, so the overs played don't change

    def assign(self, idx: np.ndarray, n: int) -> None:
        if len(self.slot) != n:
            self.slot = np.zeros(n, dtype=np.int64)
        self.slot[idx] = self.rng.choice(len(self.members), size=len(idx), p=self.p)

    def __call__(self, *args) -> np.ndarray:
        out = None
        for m, member in enumerate(self.members):
            rows = self.slot == m
            if not rows.any():
                continue
            result = np.asarray(member(*args))
            if out is None:
                out = np.zeros_like(result, dtype=np.float64)
            out[rows] = result[rows]
        return out


class BatterVecEnv(VecEnv):
    """n overs at once, from the striker's side. Finished overs restart straight away."""

    def __init__(self, data: CaptainData, captains: Population, n_envs: int = 64, **kwargs):
        obs_space, act_space = _batter_spaces()
        self.render_mode = None
        super().__init__(n_envs, obs_space, act_space)
        self.overs = Overs(data, n=n_envs, **kwargs)
        self.captains = captains
        self.ep_return = np.zeros(n_envs)
        self.ep_length = np.zeros(n_envs, dtype=np.int64)
        self._actions = None
        self.fields = None

    def set_formats(self, formats, weights=None) -> None:
        self.overs.set_formats(formats, weights)

    def _restart(self, idx: np.ndarray) -> None:
        self.overs.reset(idx)
        self.captains.assign(idx, self.num_envs)
        self.ep_return[idx] = 0.0
        self.ep_length[idx] = 0

    def _observe(self) -> np.ndarray:
        self.fields = self.captains(self.overs)
        return batter_observe(self.overs, self.fields)

    def reset(self):
        self._restart(np.arange(self.num_envs))
        return self._observe()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        o = self.overs
        _, done, info = o.step(self.fields, intent=self._actions)
        reward = info["cost"] - info["natural_cost"]
        self.ep_return += reward
        self.ep_length += 1
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        finished = np.flatnonzero(done)
        if len(finished):
            last = batter_observe(o, self.fields)
            for i in finished:
                infos[i]["terminal_observation"] = last[i]
                infos[i]["episode"] = {"r": float(self.ep_return[i]), "l": int(self.ep_length[i])}
            self._restart(finished)
        return self._observe(), reward.astype(np.float32), done, infos

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


class SelfPlayCaptainEnv(CaptainVecEnv):
    """The captain's training environment with batters that answer the field."""

    def __init__(self, data: CaptainData, batters: Population, n_envs: int = 64, **kwargs):
        super().__init__(data, n_envs=n_envs, relative=True, **kwargs)
        self.batters = batters
        self.batters.assign(np.arange(n_envs), n_envs)

    def reset(self):
        obs = super().reset()
        self.batters.assign(np.arange(self.num_envs), self.num_envs)
        return obs

    def step_wait(self):
        o = self.overs
        fields = o.fields_from_actions(self._actions)
        start = o.data.pooled_fields[o.key]
        reward, done, info = o.step(
            fields, intent=self.batters(o, fields), start_intent=self.batters(o, start)
        )
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        finished = np.flatnonzero(done)
        if len(finished):
            last = o.observe()
            for i in finished:
                infos[i]["terminal_observation"] = last[i]
                infos[i]["episode"] = {"r": float(o.ep_return[i]), "l": int(o.ball[i])}
            o.reset(finished)
            self.batters.assign(finished, self.num_envs)
        return o.observe(), reward.astype(np.float32), done, infos


class CaptainBatterGame(AECEnv):
    """One over (a spell in Tests) of captain against batter, as a PettingZoo AEC game.

    The captain moves first and sets the field; the batter sees it and picks an intent; the
    ball is bowled. Rewards come after the batter's move: minus the ball's expected cost to the
    captain, plus to the batter.
    """

    metadata = {"render_modes": [], "name": "captain_batter_v0", "is_parallelizable": False}

    def __init__(self, data: CaptainData, **kwargs):
        super().__init__()
        self.overs = Overs(data, n=1, **kwargs)
        self.possible_agents = ["captain", "batter"]
        captain_obs, captain_act = _spaces()
        batter_obs, batter_act = _batter_spaces()
        self._obs_spaces = {"captain": captain_obs, "batter": batter_obs}
        self._act_spaces = {"captain": captain_act, "batter": batter_act}
        self.field = None

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.overs.seed(seed)
        self.overs.reset_all()
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._selector = AgentSelector(self.agents)
        self.agent_selection = self._selector.reset()
        self.field = self.overs.data.pooled_fields[self.overs.key]

    def observe(self, agent):
        if agent == "captain":
            return self.overs.observe()[0]
        return batter_observe(self.overs, self.field)[0].astype(np.float32)

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0.0
        self.rewards = {a: 0.0 for a in self.agents}
        if agent == "captain":
            self.field = self.overs.fields_from_actions(np.asarray(action)[None])
        else:
            _, done, info = self.overs.step(self.field, intent=np.asarray(action)[None])
            cost = float(info["cost"][0])
            self.rewards = {"captain": -cost, "batter": cost}
            self.infos = {
                a: {"runs": int(info["runs"][0]), "wicket": bool(info["wicket"][0])}
                for a in self.agents
            }
            if done[0]:
                self.terminations = {a: True for a in self.agents}
        self.agent_selection = self._selector.next()
        self._accumulate_rewards()

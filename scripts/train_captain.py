"""Train the RL captain with PPO.

The policy sets the field ball by ball (see virtual_captain.rl.env). With all-zero actions it
sets the field optimised for all batters in the situation, so it starts from the CMA-ES
solution and learns which moves away from it pay off for this batter in this state. Its reward
for a ball is the cost it saved against that field on the same simulated deliveries. Training
runs on T20 overs first, then on all three formats with one policy that is told the format.

Every --eval-every steps the captain's fields for a fixed set of batters are scored against
their dedicated CMA-ES fields on deliveries it never trained on; that, the training curves and
the settings go to MLflow, and a checkpoint is kept for the training animation.

Usage:
  uv run python scripts/train_captain.py
  uv run python scripts/train_captain.py --steps 300000 --eval-every 100000   # a quick look
"""

import argparse
import os
import shutil
import sys
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from virtual_captain.paths import DATA_DIR, REPORTS_DIR, ROOT  # noqa: E402
from virtual_captain.rl.data import CaptainData  # noqa: E402
from virtual_captain.rl.env import FORMATS, CaptainVecEnv  # noqa: E402
from virtual_captain.rl.evaluate import captain_fields, judge_fields, paired  # noqa: E402

MODEL_PATH = DATA_DIR / "models" / "captain.zip"
CHECKPOINTS = ROOT / "checkpoints" / "captain"
FORMAT_WEIGHTS = (0.45, 0.45, 0.10)  # overs of each format; a Test spell is four overs long
PROBE_SIZE = 150
PROBE_BALLS = 4000


def probe_contexts(data: CaptainData, seed: int = 0) -> dict[str, np.ndarray]:
    """A fixed set of batter contexts to watch during training: some it trains on, some
    batters it never sees."""
    ctx = data.contexts
    rng = np.random.default_rng(seed)
    named = ~ctx["pooled"].to_numpy()
    unseen = ctx["unseen"].to_numpy()
    out = {}
    for name, mask in (("train", named & ~unseen), ("unseen", named & unseen)):
        ids = np.flatnonzero(mask)
        if len(ids):
            out[name] = np.sort(rng.choice(ids, size=min(PROBE_SIZE, len(ids)), replace=False))
    return out


class Captaincy(BaseCallback):
    def __init__(self, data, probes, t20_steps, eval_every):
        super().__init__()
        self.data, self.probes = data, probes
        self.t20_steps, self.eval_every = t20_steps, eval_every
        self.switched = False
        self.next_eval = 0
        self.history: list[dict] = []
        self.reference = {
            name: (
                judge_fields(data, ids, data.searched_fields[ids], n=PROBE_BALLS),
                judge_fields(
                    data,
                    ids,
                    data.pooled_fields[data.contexts["key"].to_numpy()[ids]],
                    n=PROBE_BALLS,
                ),
            )  # fmt: skip
            for name, ids in probes.items()
        }
        self.t0 = time.perf_counter()

    def _on_step(self) -> bool:
        if not self.switched and self.num_timesteps >= self.t20_steps:
            self.training_env.set_formats(FORMATS, FORMAT_WEIGHTS)
            self.switched = True
            print(f"{self.num_timesteps:,} steps: all formats from here")
        if self.num_timesteps >= self.next_eval:
            self.evaluate()
            self.next_eval += self.eval_every
        return True

    def _on_rollout_end(self) -> None:
        values = dict(self.model.logger.name_to_value)
        if self.model.ep_info_buffer:
            values["rollout/ep_rew_mean"] = float(
                np.mean([e["r"] for e in self.model.ep_info_buffer])
            )
        metrics = {k.replace("/", "_"): float(v) for k, v in values.items()}
        mlflow.log_metrics(metrics, step=self.num_timesteps)

    def evaluate(self) -> None:
        metrics = {}
        for name, ids in self.probes.items():
            searched, pooled = self.reference[name]
            mine = judge_fields(
                self.data, ids, captain_fields(self.model, self.data, ids), n=PROBE_BALLS
            )
            gap_search, _ = paired(searched, mine)
            gap_pooled, _ = paired(pooled, mine)
            metrics[f"probe_{name}_vs_search"] = float(gap_search.mean())
            metrics[f"probe_{name}_vs_all_batters_field"] = float(gap_pooled.mean())
            metrics[f"probe_{name}_beats_search"] = float((gap_search < 0).mean())
        mlflow.log_metrics(metrics, step=self.num_timesteps)
        self.history.append({"steps": self.num_timesteps, **metrics})
        path = CHECKPOINTS / f"step_{self.num_timesteps:09d}.zip"
        self.model.save(path)
        rate = self.num_timesteps / max(time.perf_counter() - self.t0, 1e-9)
        shown = "  ".join(
            f"{name}: {metrics[f'probe_{name}_vs_search']:+.4f} vs search, "
            f"{metrics[f'probe_{name}_vs_all_batters_field']:+.4f} vs all-batters field"
            for name in self.probes
        )
        print(f"{self.num_timesteps:>11,} steps ({rate:,.0f}/s)  {shown}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--t20-share", type=float, default=0.2, help="share of steps on T20 only")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--log-std", type=float, default=-2.0)
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--absolute", action="store_true", help="reward the raw cost, not the saving")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.set_num_threads(4)
    data = CaptainData.load()
    env = CaptainVecEnv(
        data, n_envs=args.envs, formats=("T20",), relative=not args.absolute, seed=args.seed
    )
    lr = args.lr

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=lambda remaining: lr * remaining,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "log_std_init": args.log_std,
        },  # fmt: skip
        device="cpu",
        seed=args.seed,
        verbose=0,
    )
    shutil.rmtree(CHECKPOINTS, ignore_errors=True)
    CHECKPOINTS.mkdir(parents=True)
    callback = Captaincy(
        data, probe_contexts(data), int(args.t20_share * args.steps), args.eval_every
    )

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("captain")
    with mlflow.start_run():
        mlflow.log_params(vars(args) | {"contexts": data.contexts.height})
        t0 = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        callback.evaluate()
        minutes = (time.perf_counter() - t0) / 60
        mlflow.log_metric("train_minutes", minutes)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)
    pl.DataFrame(callback.history).write_csv(
        REPORTS_DIR / "captain_training.csv", float_precision=5
    )
    print(f"trained {args.steps:,} steps in {minutes:.1f} min, saved {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

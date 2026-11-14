"""Self-play between the captain and an adaptive batter, then how exploitable each captain is.

Starts from the trained captain (scripts/train_captain.py), which has only ever faced batters
playing their natural game. Each round trains a batter against every captain so far, then
carries on training the captain against every batter so far (see virtual_captain.rl.selfplay).

Then a fresh best-response batter is trained against each captain and against the two fixed
fields from the CMA-ES searches, with the same budget for all, and what it gains is measured on
deliveries none of them trained on.

  data/models/selfplay/          the captain and batter after every round, and the
                                 best-response batters
  data/models/captain_selfplay.zip   the final captain
  reports/selfplay_rounds.csv    what each round's batter gained against the captains so far
  reports/selfplay_exploitability.csv   every field-setter against its best-response batter

Usage:
  uv run python scripts/train_selfplay.py
  uv run python scripts/train_selfplay.py --rounds 2 --batter-steps 200000 \\
      --captain-steps 200000 --exploit-steps 200000   # a quick look
  uv run python scripts/train_selfplay.py --exploit-only   # re-measure saved captains
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
from virtual_captain.rl import evaluate as ev  # noqa: E402
from virtual_captain.rl.data import CaptainData  # noqa: E402
from virtual_captain.rl.multi_env import batter_policy, natural  # noqa: E402
from virtual_captain.rl.selfplay import (  # noqa: E402
    Settings,
    exploit,
    exploitability,
    frozen,
    load_captain,
    train_batter,
    train_captain,
)

CAPTAIN = DATA_DIR / "models" / "captain.zip"
FINAL = DATA_DIR / "models" / "captain_selfplay.zip"
OUT = DATA_DIR / "models" / "selfplay"


class Curves(BaseCallback):
    """Log the training reward to MLflow, on one step axis across all of self-play."""

    def __init__(self, prefix: str, offset: int):
        super().__init__()
        self.prefix, self.offset = prefix, offset

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.ep_info_buffer:
            reward = float(np.mean([e["r"] for e in self.model.ep_info_buffer]))
            mlflow.log_metric(
                f"{self.prefix}_reward", reward, step=self.offset + self.num_timesteps
            )


def self_play(data: CaptainData, args, s: Settings) -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    captains = [ev.captain(frozen(PPO.load(CAPTAIN, device="cpu"), OUT / "captain_0.zip"))]
    batters = [natural]
    captain, batter = load_captain(CAPTAIN, s), None
    rounds, clock = [], 0
    for r in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        batter = train_batter(
            data, captains, args.batter_steps, s, model=batter, seed=r,
            callback=Curves("batter", clock),
        )  # fmt: skip
        clock += args.batter_steps
        b = frozen(batter, OUT / f"batter_{r}.zip")
        batters.append(batter_policy(b))
        # what this round's batter gains against the newest captain, before it adapts
        against = exploit(data, captains[-1], batters[-1])
        captain = train_captain(
            data, batters, args.captain_steps, captain, s, seed=r,
            callback=Curves("captain", clock),
        )  # fmt: skip
        clock += args.captain_steps
        captains.append(ev.captain(frozen(captain, OUT / f"captain_{r}.zip")))
        after = exploit(data, captains[-1], batters[-1])
        for x, y in zip(against, after, strict=True):
            rounds.append(
                {"round": r, "format": x["format"], "batters": x["batters"],
                 "gain_before": x["gain"], "gain_before_ci": x["gain_ci"],
                 "gain_after": y["gain"], "gain_after_ci": y["gain_ci"]}
            )  # fmt: skip
        mean = lambda rows: float(np.mean([x["gain"] for x in rows]))  # noqa: E731
        mlflow.log_metrics(
            {"round_batter_gain": mean(against), "round_batter_gain_after": mean(after)}, step=r
        )
        print(
            f"round {r}: the new batter gains {mean(against):+.4f} a ball against the last "
            f"captain, {mean(after):+.4f} once it has adapted "
            f"({(time.perf_counter() - t0) / 60:.1f} min)",
            flush=True,
        )
    shutil.copy(OUT / f"captain_{args.rounds}.zip", FINAL)
    pl.DataFrame(rounds).write_csv(REPORTS_DIR / "selfplay_rounds.csv", float_precision=5)


def measure(data: CaptainData, args, s: Settings) -> pl.DataFrame:
    setters = {
        "all-batters field": (ev.optimised_for_all, None),
        "searched field": (ev.searched_for_striker, None),
    }
    for path in sorted(OUT.glob("captain_*.zip"), key=lambda p: int(p.stem.split("_")[1])):
        r = int(path.stem.split("_")[1])
        setters[f"captain, round {r}"] = (ev.captain(PPO.load(path, device="cpu")), r)
    rows = []
    for i, (name, (setter, r)) in enumerate(setters.items()):
        t0 = time.perf_counter()
        slug = name.replace(", ", "_").replace(" ", "_").replace("-", "_")
        found, _ = exploitability(
            data, setter, args.exploit_steps, s, seed=100 + i, path=OUT / f"best_response_{slug}"
        )
        rows += [{"setter": name, "round": r} | x for x in found]
        gain = np.mean([x["gain"] for x in found if x["batters"] == "train"])
        mlflow.log_metric(f"exploitability_{slug}", float(gain))
        print(f"{name}: a best-response batter gains {gain:+.4f} a ball "
              f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)  # fmt: skip
    table = pl.DataFrame(rows)
    table.write_csv(REPORTS_DIR / "selfplay_exploitability.csv", float_precision=5)
    return table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--batter-steps", type=int, default=2_000_000)
    p.add_argument("--captain-steps", type=int, default=3_000_000)
    p.add_argument("--exploit-steps", type=int, default=1_500_000)
    p.add_argument("--exploit-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.set_num_threads(4)
    data = CaptainData.load()
    s = Settings(seed=args.seed)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("selfplay")
    t0 = time.perf_counter()
    with mlflow.start_run():
        mlflow.log_params(vars(args) | vars(s))
        if not args.exploit_only:
            self_play(data, args, s)
        measure(data, args, s)
        mlflow.log_metric("minutes", (time.perf_counter() - t0) / 60)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

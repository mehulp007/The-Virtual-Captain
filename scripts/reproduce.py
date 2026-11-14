"""Everything, from the raw data to the figures, in order. About nine hours on a laptop with an
RTX 4060, most of it training the captain and self-play.

Each step is one of the scripts in this folder, run with its defaults, so any step can also be
run (or tuned) on its own. A step that fails stops the run; `--from` picks up where it stopped.

Usage:
  uv run python scripts/reproduce.py
  uv run python scripts/reproduce.py --list
  uv run python scripts/reproduce.py --from train_captain
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("download_data", "fetch the Kaggle dataset and Cricsheet", "a few minutes"),
    ("build_dataset", "one deliveries table in Parquet", "a few minutes"),
    ("train_shot_model", "the shot model", "5 min on a GPU"),
    ("calibrate", "fit the fielding simulator", "8 min"),
    ("optimize_fields", "CMA-ES fields for 76 scenarios", "16 min"),
    ("build_captain_data", "delivery banks and a search per batter context", "1.5 h"),
    ("train_captain", "the RL captain", "2 h"),
    ("evaluate_captain", "judge it against the searches", "a few minutes"),
    ("train_selfplay", "self-play and exploitability", "4 h"),
    ("evaluate_selfplay", "fields before and after self-play", "a few minutes"),
    ("build_app_bundle", "the app's data bundle", "2 min"),
    ("case_studies", "famous innings replayed", "1 min"),
    ("make_figures", "every figure in the README", "a few minutes"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="start", choices=[s for s, _, _ in STEPS])
    p.add_argument("--list", action="store_true", help="show the steps and stop")
    args = p.parse_args(argv)
    if args.list:
        for name, what, how_long in STEPS:
            print(f"{name:<20} {what} ({how_long})")
        return 0

    names = [s for s, _, _ in STEPS]
    first = names.index(args.start) if args.start else 0
    t_all = time.perf_counter()
    for name, what, how_long in STEPS[first:]:
        print(f"\n=== {name}: {what} ({how_long})", flush=True)
        t0 = time.perf_counter()
        done = subprocess.run([sys.executable, str(HERE / f"{name}.py")])
        if done.returncode:
            print(f"{name} failed; carry on with: uv run python scripts/reproduce.py --from {name}")
            return done.returncode
        print(f"=== {name} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    print(f"\nall done in {(time.perf_counter() - t_all) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())

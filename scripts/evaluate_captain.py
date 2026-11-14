"""Judge the trained captain against conventional, analyst and CMA-ES fields.

  reports/captain_static.csv    every batter context: each field's runs, wickets and cost per
                                ball on that batter's judging deliveries, and the captain's
                                difference from the dedicated search with a 95% interval
  reports/captain_overs.csv     whole overs from real match states, by format, for batters the
                                captain trained on and batters it never saw
  reports/captain_holdout.csv   the studied batters' real 2025-26 deliveries
  reports/captain_fields.json   fields for the showcase batters and one over, ball by ball

Usage:
  uv run python scripts/evaluate_captain.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO

from virtual_captain.field.positions import nearest_name
from virtual_captain.models import encoding as enc
from virtual_captain.models.shot_model import ShotModel
from virtual_captain.optim.objective import make_scenario, summary
from virtual_captain.paths import DATA_DIR, PROCESSED_DIR, REPORTS_DIR
from virtual_captain.rl import evaluate as ev
from virtual_captain.rl.data import CaptainData
from virtual_captain.rl.env import EPISODE_BALLS, FORMATS, Overs

MODEL_PATH = DATA_DIR / "models" / "captain.zip"
MIN_HOLDOUT_BALLS = 150
SHOWCASE = [
    ("SA Yadav", "T20", "middle", "spin"),
    ("V Kohli", "ODI", "middle", "pace"),
    ("JE Root", "TEST", "new_ball", "pace"),
]
OVERS = {"T20": 20000, "ODI": 20000, "TEST": 5000}


def describe(field9: np.ndarray) -> list[dict]:
    return [
        {"position": nearest_name(p), "x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}
        for p in field9
    ]


def static(data: CaptainData, model) -> pl.DataFrame:
    ctx = data.contexts
    ids = np.flatnonzero(~ctx["pooled"].to_numpy())
    keys = ctx["key"].to_numpy()[ids]
    fields = {
        "conventional": data.conventional_fields[keys],
        "analyst": data.analyst_fields[ids],
        "all_batters": data.pooled_fields[keys],
        "searched": data.searched_fields[ids],
        "captain": ev.captain_fields(model, data, ids),
    }
    judged = {name: ev.judge_fields(data, ids, f) for name, f in fields.items()}
    table = ctx[ids].select("context", "batter", "format", "phase", "bowling", "balls", "unseen")
    cols = {}
    for name, j in judged.items():
        cols[f"{name}_runs_per_ball"] = j.runs.mean(1)
        cols[f"{name}_wickets_per_ball"] = j.wickets.mean(1)
        cols[f"{name}_cost"] = j.cost.mean(1)
    for other in ("searched", "all_batters", "analyst"):
        diff, half = ev.paired(judged[other], judged["captain"])
        cols[f"captain_vs_{other}"] = diff
        cols[f"captain_vs_{other}_ci"] = half
    return table.with_columns(pl.Series(k, v) for k, v in cols.items())


def overs(data: CaptainData, model) -> pl.DataFrame:
    setters = {
        "conventional": ev.conventional,
        "all_batters": ev.optimised_for_all,
        "searched_for_striker": ev.searched_for_striker,
        "captain": ev.captain(model),
    }
    rows = []
    for fmt in FORMATS:
        for batters in ("train", "unseen"):
            played = {
                name: ev.play(s, data, fmt, OVERS[fmt], batters=batters, seed=11)
                for name, s in setters.items()
            }
            vs_search = ev.compare(played, "searched_for_striker")
            vs_all = ev.compare(played, "all_batters")
            for name, p in played.items():
                balls = p.balls.sum()
                rows.append(
                    {
                        "format": fmt, "batters": batters, "field": name, "overs": len(p.cost),
                        "runs_per_ball": p.runs.sum() / balls,
                        "wickets_per_100": 100 * p.wickets.sum() / balls,
                        "cost_per_ball": p.cost.sum() / balls,
                        "vs_searched": vs_search[name][0], "vs_searched_ci": vs_search[name][1],
                        "vs_all_batters": vs_all[name][0], "vs_all_batters_ci": vs_all[name][1],
                    }
                )  # fmt: skip
            print(f"{fmt} {batters}: captain vs search {vs_search['captain'][0]:+.4f} "
                  f"+/- {vs_search['captain'][1]:.4f} per ball")  # fmt: skip
    return pl.DataFrame(rows)


def holdout(data: CaptainData, model) -> pl.DataFrame:
    """The studied batters' contexts, judged on their real deliveries from 2025-26."""
    studied = pl.read_csv(REPORTS_DIR / "optimizer_results.csv")["batter"].unique().to_list()
    ctx = data.contexts.filter(pl.col("batter").is_in(studied) & ~pl.col("pooled"))
    shot_model = ShotModel.load(DATA_DIR / "models" / "shot_model.pt")
    frame = (
        pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")
        .pipe(enc.tracked_innings)
        .pipe(enc.add_labels)
        .pipe(enc.split_column)
        .filter(pl.col("split") == "test")
        .with_columns(
            pl.when(pl.col("pace_or_spin") == "spin")
            .then(pl.lit("spin"))
            .otherwise(pl.lit("pace"))
            .alias("bowling")
        )  # fmt: skip
        .collect()
    )
    lam = dict(zip(data.keys["key"], data.keys["wicket_value"], strict=True))
    out = []
    for c, who, fmt, phase, bowling, key in ctx.select(
        "context", "batter", "format", "phase", "bowling", "key"
    ).iter_rows():
        rows = frame.filter(
            (pl.col("striker") == who) & (pl.col("format") == fmt)
            & (pl.col("phase") == phase) & (pl.col("bowling") == bowling)
        )  # fmt: skip
        if rows.height < MIN_HOLDOUT_BALLS:
            continue
        scen = make_scenario(rows, shot_model, fmt, phase, bowling, lam[key], n=60000, seed=c)
        row = {"batter": who, "format": fmt, "phase": phase, "bowling": bowling,
               "holdout_balls": rows.height}  # fmt: skip
        fields = {
            "all_batters": data.pooled_fields[key],
            "searched": data.searched_fields[c],
            "captain": ev.captain_fields(model, data, np.array([c]))[0],
        }
        for name, f in fields.items():
            row[f"{name}_cost"] = summary(f, scen)["cost"]
        out.append(row)
    return pl.DataFrame(out)


def example_over(data: CaptainData, model, seed_from: int = 0) -> dict:
    """One T20 over between two named batters in which the strike changes, ball by ball."""
    setter = ev.captain(model)
    for seed in range(seed_from, seed_from + 500):
        o = Overs(data, n=1, bank="judge", formats=("T20",), generic_share=0.0, seed=seed)
        if data.contexts["phase"][int(o.striker[0])] != "death":
            continue
        names = data.contexts["batter"].to_list()
        balls = []
        for b in range(EPISODE_BALLS["T20"]):
            striker = names[int(o.striker[0])]
            lam = float(o.wicket_value()[0])
            field = setter(o)
            _, done, info = o.step(field)
            balls.append({"ball": b + 1, "striker": striker, "wicket_value": round(lam, 2),
                          "runs": int(info["runs"][0]), "wicket": bool(info["wicket"][0]),
                          "field": describe(field[0])})  # fmt: skip
            if done[0]:
                break
        strikers = [x["striker"] for x in balls]
        changes = sum(a != b for a, b in zip(strikers[:-1], strikers[1:], strict=True))
        if len(set(strikers)) == 2 and changes >= 2:
            return {"seed": seed, "balls": balls}
    return {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    args = p.parse_args(argv)
    data = CaptainData.load()
    model = PPO.load(args.model, device="cpu")

    table = static(data, model)
    table.write_csv(REPORTS_DIR / "captain_static.csv", float_precision=5)
    for unseen in (False, True):
        part = table.filter(pl.col("unseen") == unseen)
        print(f"{'unseen' if unseen else 'trained-on'} batters ({part.height} contexts): "
              f"captain vs search {part['captain_vs_searched'].mean():+.4f}/ball, beats it in "
              f"{(part['captain_vs_searched'] < 0).mean():.0%}")  # fmt: skip

    overs(data, model).write_csv(REPORTS_DIR / "captain_overs.csv", float_precision=5)
    held = holdout(data, model)
    held.write_csv(REPORTS_DIR / "captain_holdout.csv", float_precision=5)
    print(f"held-out 2025-26: {held.height} contexts")

    showcase = {}
    for who, fmt, phase, bowling in SHOWCASE:
        c = data.context_of(who, fmt, phase, bowling)
        key = int(data.contexts["key"][c])
        showcase[f"{who} | {fmt} {phase} | {bowling}"] = {
            "conventional": describe(data.conventional_fields[key]),
            "all_batters": describe(data.pooled_fields[key]),
            "searched": describe(data.searched_fields[c]),
            "captain": describe(ev.captain_fields(model, data, np.array([c]))[0]),
        }
    out = {"showcase": showcase, "over": example_over(data, model)}
    (REPORTS_DIR / "captain_fields.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote results to {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

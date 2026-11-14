"""Compare the self-play captain with the static optimum and the captain it started from.

  reports/selfplay_matrix.csv   the fixed fields and the captain before and after self-play,
                                each against every one's best-response batter and the natural
                                batter. Gain is
                                the batter's cost per ball minus its natural game's, ball by
                                ball on the same fields
  reports/selfplay_fields.json  for three batters: the field optimised for all batters, the
                                captain before and after self-play, what each costs against the
                                natural batter and against its own best-response batter, and
                                where that batter aims

Run after scripts/train_selfplay.py.

Usage:
  uv run python scripts/evaluate_selfplay.py
"""

import argparse
import json
import sys

import numpy as np
import polars as pl
import torch
from stable_baselines3 import PPO

from virtual_captain.field.positions import nearest_name
from virtual_captain.paths import DATA_DIR, REPORTS_DIR
from virtual_captain.rl import evaluate as ev
from virtual_captain.rl.batter import FIELD_SECTORS, intent_weights
from virtual_captain.rl.data import CaptainData
from virtual_captain.rl.env import FORMATS
from virtual_captain.rl.multi_env import batter_policy

SHOWCASE = [
    ("SA Yadav", "T20", "middle", "spin"),
    ("V Kohli", "ODI", "middle", "pace"),
    ("JE Root", "TEST", "new_ball", "pace"),
]
MATRIX_OVERS = {"T20": 2000, "ODI": 2000, "TEST": 500}
SHOWCASE_SHOTS = 20000


def slug(name: str) -> str:
    return name.replace(", ", "_").replace(" ", "_").replace("-", "_")


def setters_and_batters(folder):
    captains = sorted(folder.glob("captain_*.zip"), key=lambda p: int(p.stem.split("_")[1]))
    setters = {"all-batters field": ev.optimised_for_all, "searched field": ev.searched_for_striker}
    for path in captains:
        setters[f"captain, round {path.stem.split('_')[1]}"] = ev.captain(
            PPO.load(path, device="cpu")
        )
    batters = {"natural": None}
    for name in setters:
        path = folder / f"best_response_{slug(name)}.zip"
        if path.exists():
            batters[f"best response to {name}"] = batter_policy(PPO.load(path, device="cpu"))
    return setters, batters


def matrix(data, setters, batters) -> pl.DataFrame:
    rows = []
    for s_name, setter in setters.items():
        for b_name, batter in batters.items():
            for fmt in FORMATS:
                if batter is None:
                    p = ev.play(setter, data, fmt, MATRIX_OVERS[fmt], seed=31)
                    row = {"natural_cost": float(p.cost.sum() / p.balls.sum()), "gain": 0.0}
                else:
                    p = ev.play(setter, data, fmt, MATRIX_OVERS[fmt], seed=31, batter=batter)
                    row = {
                        "natural_cost": float(p.natural.sum() / p.balls.sum()),
                        "gain": float(((p.cost - p.natural) / p.balls).mean()),
                    }
                rows.append({"setter": s_name, "batter": b_name, "format": fmt} | row)
        print(f"{s_name}: done", flush=True)
    return pl.DataFrame(rows)


def where_they_aim(overs, field, batter) -> dict:
    """The batter's shots against one field, natural and with its intent: the share of balls
    kept in the ground that go into each 30-degree sector, and the share of sixes."""
    o = overs
    intent = batter(o, field[None]) if batter is not None else np.zeros((1, 3))
    bank = o.bank
    rng = np.random.default_rng(0)
    c = int(o.striker[0])
    idx = bank.offsets[c] + rng.integers(0, bank.lengths[c], SHOWCASE_SHOTS)
    shots = (bank.outcome[idx][None], bank.theta[idx][None], bank.dist[idx][None],
             bank.aerial[idx][None])  # fmt: skip
    fmt = int(o.key_format[o.key[0]])
    w, kl = intent_weights(intent, *shots, max_shift=o.max_shift[fmt])
    in_play = shots[0][0] == 2
    sector = (shots[1][0] / (2 * np.pi) * FIELD_SECTORS).astype(int) % FIELD_SECTORS

    def share(weights):
        return (
            np.bincount(sector[in_play], weights=weights[in_play], minlength=FIELD_SECTORS)
            / weights[in_play].sum()
        )

    return {
        "intent": [round(float(x), 3) for x in intent[0]],
        "shift_nats": round(float(kl[0]), 4),
        "sector_natural": [round(float(x), 4) for x in share(np.ones(SHOWCASE_SHOTS))],
        "sector_batter": [round(float(x), 4) for x in share(w[0])],
        "six_natural": round(float((shots[0][0] == 3).mean()), 4),
        "six_batter": round(float(w[0][shots[0][0] == 3].sum()), 4),
    }


def describe(field9: np.ndarray) -> list[dict]:
    return [
        {"position": nearest_name(p), "x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}
        for p in field9
    ]


def showcase(data, setters, batters) -> dict:
    last = max(k for k in setters if k.startswith("captain"))
    chosen = {"all-batters field": "all_batters", "captain, round 0": "before",
              last: "after"}  # fmt: skip
    out = {}
    for who, fmt, phase, bowling in SHOWCASE:
        c = data.context_of(who, fmt, phase, bowling)
        entry = {}
        for name, label in chosen.items():
            br = batters.get(f"best response to {name}")
            overs = ev.typical_overs(data, np.array([c]), bank="judge", shots_per_ball=1)
            field = setters[name](overs)[0]
            costs = {}
            for b_label, b in (("natural", None), ("best_response", br)):
                o = ev.typical_overs(
                    data, np.array([c]), bank="judge", shots_per_ball=SHOWCASE_SHOTS, seed=5
                )
                intent = np.zeros((1, 3)) if b is None else b(o, field[None])
                _, _, info = o.step(field[None], intent=intent)
                costs[b_label] = round(float(info["cost"][0]), 4)
            o = ev.typical_overs(data, np.array([c]), bank="judge", shots_per_ball=1)
            entry[label] = {
                "setter": name,
                "field": describe(field),
                "cost_natural": costs["natural"],
                "cost_best_response": costs["best_response"],
                "aim": where_they_aim(o, field, br),
            }
        out[f"{who} | {fmt} {phase} | {bowling}"] = entry
        print(who, {k: (v["cost_natural"], v["cost_best_response"]) for k, v in entry.items()})
    return out


def main(argv=None) -> int:
    folder = DATA_DIR / "models" / "selfplay"
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    torch.set_num_threads(4)
    data = CaptainData.load()
    setters, batters = setters_and_batters(folder)
    table = showcase(data, setters, batters)
    (REPORTS_DIR / "selfplay_fields.json").write_text(json.dumps(table, indent=1) + "\n")
    last = max(k for k in setters if k.startswith("captain"))
    keep = ["all-batters field", "searched field", "captain, round 0", last]
    setters = {k: setters[k] for k in keep}
    batters = {k: v for k, v in batters.items() if k == "natural" or k.endswith(tuple(keep))}
    matrix(data, setters, batters).write_csv(REPORTS_DIR / "selfplay_matrix.csv", float_precision=5)
    print(f"wrote results to {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Optimise fields and compare them with conventional and analyst-set fields.

For every scenario (a batter, or all batters, in a format / phase / type of bowling):
  conventional  the textbook field for the situation
  analyst       fielders placed on the batter's busiest zones (same number of riders)
  optimised     CMA-ES search, started from the analyst's field
Fields are optimised on one batch of simulated deliveries and judged on a separate one, and
for batters who played in 2025-26, also on their real contexts from those seasons.

Usage:
  uv run python scripts/optimize_fields.py
  uv run python scripts/optimize_fields.py --quick       # a few scenarios, short searches
"""

import argparse
import json
import sys
import time

import numpy as np
import polars as pl

from virtual_captain.field import rules
from virtual_captain.field.geometry import PITCH_HALF
from virtual_captain.field.positions import nearest_name
from virtual_captain.models import encoding as enc
from virtual_captain.models.shot_model import ShotModel
from virtual_captain.optim.cma_search import optimise
from virtual_captain.optim.heuristics import analyst_field
from virtual_captain.optim.objective import make_scenario, paired_difference, summary
from virtual_captain.optim.sensitivity import best_moves, value_map
from virtual_captain.optim.value import as_dict, wicket_values
from virtual_captain.paths import DATA_DIR, PROCESSED_DIR, REPORTS_DIR

BATTERS = [
    "V Kohli", "JE Root", "SPD Smith", "KS Williamson", "Babar Azam",
    "RG Sharma", "DA Warner", "AB de Villiers", "JC Buttler", "Q de Kock",
    "KL Rahul", "Mohammad Rizwan", "BA Stokes", "Shubman Gill", "GJ Maxwell",
    "TM Head", "RR Pant", "SA Yadav", "H Klaasen", "YBK Jaiswal",
]  # fmt: skip
MIN_CONTEXT_BALLS = 400
MIN_HOLDOUT_BALLS = 150
CONTEXTS_PER_BATTER = 3
SHOWCASE = ("SA Yadav", "T20", "middle", "spin")

STRIKER = np.array([0.0, -PITCH_HALF])


def bowling_col() -> pl.Expr:
    return pl.when(pl.col("pace_or_spin") == "spin").then(pl.lit("spin")).otherwise(pl.lit("pace"))


def field_shape(field9: np.ndarray) -> dict[str, int]:
    """Counts used by the rediscovery checks."""
    rel = field9 - STRIKER
    from_striker = np.hypot(rel[:, 0], rel[:, 1])
    angle = np.degrees(np.arccos(rel[:, 1] / from_striker))  # 0 = straight down the ground
    return {
        "outside_ring": int(rules.outside_ring(field9).sum()),
        # slips sit where Cricinfo plots their catches, 9-12 m behind and to the off side
        "slip_cordon": int(((rel[:, 0] > 0) & (from_striker < 20) & (angle > 115)).sum()),
        # bat-pad catchers in front of or beside the bat (short leg, silly point, silly mid-off)
        "close_catchers": int(((from_striker < 12) & (angle <= 115)).sum()),
    }


def describe(field9: np.ndarray) -> list[dict]:
    return [
        {"position": nearest_name(p), "x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}
        for p in field9
    ]


def plan(frame: pl.DataFrame, quick: bool) -> list[dict]:
    train = frame.filter(pl.col("split") != "test").with_columns(bowling_col().alias("bowling"))
    specs = []
    groups = train.group_by("format", "phase", "bowling").len().sort("format", "phase", "bowling")
    for fmt, phase, bowling, _ in groups.iter_rows():
        specs.append({"batter": None, "fmt": fmt, "phase": phase, "bowling": bowling})
    batters = BATTERS[:3] if quick else BATTERS
    contexts = (
        train.filter(pl.col("striker").is_in(batters))
        .group_by("striker", "format", "phase", "bowling")
        .len()
        .filter(pl.col("len") >= MIN_CONTEXT_BALLS)
        .sort("len", descending=True)
        .group_by("striker", maintain_order=True)
        .head(CONTEXTS_PER_BATTER)
    )
    for who, fmt, phase, bowling, _ in contexts.iter_rows():
        specs.append({"batter": who, "fmt": fmt, "phase": phase, "bowling": bowling})
    if quick:
        specs = [s for s in specs if s["batter"] is not None][:3] + specs[:2]
    return specs


def rows_for(frame: pl.DataFrame, spec: dict, split_filter: pl.Expr) -> pl.DataFrame:
    cond = (
        split_filter
        & (pl.col("format") == spec["fmt"])
        & (pl.col("phase") == spec["phase"])
        & (bowling_col() == spec["bowling"])
    )
    if spec["batter"]:
        cond &= pl.col("striker") == spec["batter"]
    return frame.filter(cond)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations", type=int, default=70)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--opt-balls", type=int, default=30000)
    p.add_argument("--eval-balls", type=int, default=60000)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    if args.quick:
        args.generations, args.opt_balls, args.eval_balls = 15, 8000, 15000

    model = ShotModel.load(DATA_DIR / "models" / "shot_model.pt")
    lf = pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")
    values = wicket_values(lf)
    values.write_csv(REPORTS_DIR / "wicket_values.csv", float_precision=3)
    lam = as_dict(values)
    frame = lf.pipe(enc.tracked_innings).pipe(enc.add_labels).pipe(enc.split_column).collect()

    specs = plan(frame, args.quick)
    print(f"{len(specs)} scenarios")
    results, fields_out, generic_fields, scenarios = [], [], {}, {}
    t_all = time.perf_counter()
    for k, spec in enumerate(specs):
        t0 = time.perf_counter()
        label = (
            f"{spec['batter'] or 'all batters'} | {spec['fmt']} {spec['phase']} | {spec['bowling']}"
        )
        rows = rows_for(frame, spec, pl.col("split") != "test")
        wv = lam[(spec["fmt"], spec["phase"])]
        if spec["batter"] is None:  # a group: draw distinct deliveries for search and judging
            rows = rows.sample(fraction=1.0, shuffle=True, seed=k)
            half = rows.height // 2
            opt_rows, eval_rows = rows.head(min(half, args.opt_balls)), rows.slice(half)
            eval_rows = eval_rows.head(args.eval_balls)
        else:
            opt_rows = eval_rows = rows
        context = (spec["fmt"], spec["phase"], spec["bowling"])
        opt_s = make_scenario(opt_rows, model, *context, wv, n=args.opt_balls, seed=10 * k)
        eval_s = make_scenario(eval_rows, model, *context, wv, n=args.eval_balls, seed=10 * k + 5)

        conventional = opt_s.template()
        analyst = analyst_field(opt_rows.filter(pl.col("outcome") == enc.IN_PLAY), *context)
        search = optimise(opt_s, analyst, args.generations, args.population, seed=k)
        fields = {"conventional": conventional, "analyst": analyst, "optimised": search.field}

        row = {"scenario": label, "batter": spec["batter"] or "all batters", "format": spec["fmt"]}
        row |= {"phase": spec["phase"], "bowling": spec["bowling"], "wicket_value": wv}
        row["training_balls"] = rows.height
        for name, f in fields.items():
            s = summary(f, eval_s)
            row |= {f"{name}_{m}": v for m, v in s.items()}
        for a, b in [
            ("conventional", "optimised"),
            ("analyst", "optimised"),
            ("conventional", "analyst"),
        ]:
            mean, lo, hi = paired_difference(fields[a], fields[b], eval_s)
            row |= {f"{b}_vs_{a}": mean, f"{b}_vs_{a}_lo": lo, f"{b}_vs_{a}_hi": hi}

        held = rows_for(frame, spec, pl.col("split") == "test") if spec["batter"] else None
        if held is not None and held.height >= MIN_HOLDOUT_BALLS:
            hold_s = make_scenario(held, model, *context, wv, n=args.eval_balls, seed=10 * k + 7)
            row["holdout_balls"] = held.height
            for name, f in fields.items():
                row[f"holdout_{name}_cost"] = summary(f, hold_s)["cost"]
        results.append(row)
        fields_out.append({"scenario": label, **{n: describe(f) for n, f in fields.items()}})
        if spec["batter"] is None:
            generic_fields[context] = fields
        scenarios[(spec["batter"], *context)] = (eval_s, fields)
        print(
            f"[{k + 1}/{len(specs)}] {label}: cost {row['conventional_cost']:.3f} / "
            f"{row['analyst_cost']:.3f} / {row['optimised_cost']:.3f}  "
            f"({time.perf_counter() - t0:.0f}s)"
        )
    print(f"all scenarios in {(time.perf_counter() - t_all) / 60:.1f} min")

    table = pl.DataFrame(results)
    table.write_csv(REPORTS_DIR / "optimizer_results.csv", float_precision=4)
    (REPORTS_DIR / "optimized_fields.json").write_text(json.dumps(fields_out, indent=1) + "\n")

    # Rediscovery: does the search find what captains know, without being told?
    checks = []
    for context, fields in generic_fields.items():
        for name in ("conventional", "optimised"):
            checks.append({"format": context[0], "phase": context[1], "bowling": context[2],
                           "field": name, **field_shape(fields[name])})  # fmt: skip
    pl.DataFrame(checks).write_csv(REPORTS_DIR / "rediscovery.csv")

    # Single 5 m moves from the analyst's field, per batter scenario
    moves = []
    for (who, *context), (eval_s, fields) in scenarios.items():
        if who is None:
            continue
        best = best_moves(fields["analyst"], eval_s).head(1)
        moves.append(best.with_columns(pl.lit(who).alias("batter"),
                                       pl.lit(" ".join(context)).alias("context")))  # fmt: skip
    if moves:
        pl.concat(moves).sort("cost_change").write_csv(
            REPORTS_DIR / "best_moves.csv", float_precision=4
        )

    if SHOWCASE in scenarios:
        eval_s, fields = scenarios[SHOWCASE]
        f = fields["optimised"]
        deep = np.flatnonzero(rules.outside_ring(f))
        drop = int(deep[0]) if len(deep) else 0
        coords, grid = value_map(f, drop, eval_s)
        (REPORTS_DIR / "value_map.json").write_text(
            json.dumps(
                {
                    "scenario": " | ".join(SHOWCASE),
                    "dropped": nearest_name(f[drop]),
                    "field": describe(np.delete(f, drop, axis=0)),
                    "coords": [round(float(c), 2) for c in coords],
                    "saved": [
                        [None if np.isnan(v) else round(float(v), 4) for v in r] for r in grid
                    ],
                }
            )
            + "\n"
        )

    print(f"wrote results to {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

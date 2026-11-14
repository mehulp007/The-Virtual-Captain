"""Build what the RL captain learns from and is judged on (see virtual_captain.rl.data).

Contexts are every batter with at least --min-balls tracked balls in a (format, phase, bowling)
situation before 2025, plus "all batters" for each of the 16 situations. About one batter in
seven is kept out of training entirely, to test the captain on batters it has never seen.

For every context: a bank of simulated deliveries to learn from, a separate one to judge on,
the batter's profile, the analyst's field and a dedicated CMA-ES field. The search is the one in
scripts/optimize_fields.py, run on the context's learning bank and started from the field
optimised there for all batters in the situation, which is also where the RL captain starts.
The all-batters contexts reuse those fields.

Also fits the forcing risk for the adaptive batter (rl.batter.fit_forcing_risk) and writes
reports/pattern_shifts.csv and reports/forcing_risk.csv.

Usage:
  uv run python scripts/build_captain_data.py
  uv run python scripts/build_captain_data.py --limit 20 --generations 10   # a quick look
  uv run python scripts/build_captain_data.py --risk-only   # refit the forcing risk only
"""

import argparse
import json
import sys
import time

import numpy as np
import polars as pl

from virtual_captain.field.templates import KEYS, template
from virtual_captain.models import encoding as enc
from virtual_captain.models.shot_model import ShotModel
from virtual_captain.optim.cma_search import optimise
from virtual_captain.optim.heuristics import analyst_field
from virtual_captain.optim.objective import Scenario
from virtual_captain.optim.value import as_dict, wicket_value_states, wicket_values
from virtual_captain.paths import DATA_DIR, PROCESSED_DIR, REPORTS_DIR
from virtual_captain.rl.batter import fit_forcing_risk, pattern_shifts
from virtual_captain.rl.data import Bank, CaptainData, profile
from virtual_captain.sim.engine import Shots

UNSEEN_SHARE = 0.15
SPELL = 24  # balls in a Test spell
NEW_BALL_UNTIL = 120  # ball age at which the Test phase turns from new to old ball
NEW_BALL_EVERY = 480
STARTS_PER_FORMAT = 60000


def bowling_col() -> pl.Expr:
    return pl.when(pl.col("pace_or_spin") == "spin").then(pl.lit("spin")).otherwise(pl.lit("pace"))


def phase4_fields() -> tuple[dict, list[str]]:
    """Fields optimised for all batters in each situation, and the batters studied there."""
    fields = {}
    for entry in json.loads((REPORTS_DIR / "optimized_fields.json").read_text()):
        who, situation, bowling = entry["scenario"].split(" | ")
        if who == "all batters":
            fmt, phase = situation.split()
            fields[(fmt, phase, bowling)] = [[p["x"], p["y"]] for p in entry["optimised"]]
    studied = pl.read_csv(REPORTS_DIR / "optimizer_results.csv")["batter"].unique().to_list()
    return fields, [b for b in studied if b != "all batters"]


def start_states(lf: pl.LazyFrame) -> pl.DataFrame:
    """Real match states at the first legal ball of an over, before 2025 (men)."""
    scheduled = (
        pl.when(pl.col("format") == "T20").then(120).when(pl.col("format") == "ODI").then(300)
    )
    limited = (pl.col("format") != "TEST") & (
        pl.col("balls_before") + pl.col("balls_remaining") == scheduled
    )
    spell_fits = (
        pl.when(pl.col("ball_age") < NEW_BALL_UNTIL)
        .then(pl.col("ball_age") + SPELL <= NEW_BALL_UNTIL)
        .otherwise(pl.col("ball_age") + SPELL <= NEW_BALL_EVERY)
    )
    test = (pl.col("format") == "TEST") & spell_fits
    states = (
        lf.filter(
            (pl.col("gender") == "male")
            & (pl.col("year") < enc.HOLDOUT_FROM_YEAR)
            & pl.col("is_legal")
            & (pl.col("balls_before") % 6 == 0)
            & (pl.col("wickets_before") <= 8)
            & pl.col("pace_or_spin").is_not_null()
            & (limited | test)
        )
        .select(
            "format",
            "phase",
            bowling_col().alias("bowling"),
            "innings",
            "balls_remaining",
            "wickets_before",
            "ball_age",
        )  # fmt: skip
        .collect()
    )
    states = pl.concat(
        [
            g.sample(min(g.height, STARTS_PER_FORMAT), seed=0)
            for _, g in states.group_by("format", maintain_order=True)
        ]
    )
    key = pl.struct("format", "phase", "bowling").map_elements(
        lambda s: KEYS.index((s["format"], s["phase"], s["bowling"])), return_dtype=pl.Int64
    )
    return states.with_columns(key.alias("key")).sort("format", "key")


def search(c: int, key: tuple, shots: Shots, start: np.ndarray, lam: dict, args) -> np.ndarray:
    """CMA-ES for one batter context on its learning bank, started from the field optimised
    for all batters in the situation (where the captain starts too)."""
    fmt, phase, bowling = key
    scen = Scenario(
        fmt=fmt,
        phase=phase,
        bowling=bowling,
        wicket_value=lam[(fmt, phase)],
        shots=shots,
        uniforms=np.random.default_rng(1000 + 2 * c).random((len(shots), 4)),
    )
    return optimise(scen, start, args.generations, args.population, seed=c).field


def search_again(args) -> int:
    """Redo only the searches, on the banks already built."""
    data = CaptainData.load()
    lam = as_dict(wicket_values(pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")))
    pooled = data.pooled_fields
    ctx = data.contexts
    fields = data.searched_fields
    t0 = time.perf_counter()
    for c, fmt, phase, bowling, key, is_pooled in ctx.select(
        "context", "format", "phase", "bowling", "key", "pooled"
    ).iter_rows():
        if not is_pooled:
            start = pooled[key]
            fields[c] = search(c, (fmt, phase, bowling), data.learn.shots(c), start, lam, args)
        if c % 50 == 0:
            print(f"[{c + 1}/{ctx.height}] {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    data.contexts = ctx.with_columns(pl.Series("searched", [f.ravel().tolist() for f in fields]))
    data.save()
    print(f"searched {ctx.height} contexts again in {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


def forcing_risk(lf: pl.LazyFrame) -> pl.DataFrame:
    shifts = pattern_shifts(lf)
    shifts.write_csv(REPORTS_DIR / "pattern_shifts.csv", float_precision=5)
    risk = fit_forcing_risk(shifts)
    risk.write_csv(REPORTS_DIR / "forcing_risk.csv", float_precision=5)
    for r in risk.iter_rows(named=True):
        print(f"{r['format']}: {r['fitted']:.4f} +/- {r['ci']:.4f} extra dismissals per ball "
              f"per nat of pattern shift ({r['cases']} cases)")  # fmt: skip
    return risk


def risk_again() -> int:
    """Refit only the forcing risk."""
    data = CaptainData.load()
    data.forcing_risk = forcing_risk(pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet"))
    data.save()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-balls", type=int, default=800)
    p.add_argument("--learn-balls", type=int, default=12000)
    p.add_argument("--judge-balls", type=int, default=20000)
    p.add_argument("--pooled-rows", type=int, default=100000)
    p.add_argument("--generations", type=int, default=70)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only the first N batter contexts")
    p.add_argument("--search-only", action="store_true", help="redo the searches, keep the banks")
    p.add_argument("--risk-only", action="store_true", help="refit the forcing risk only")
    args = p.parse_args(argv)
    if args.search_only:
        return search_again(args)
    if args.risk_only:
        return risk_again()

    t_all = time.perf_counter()
    model = ShotModel.load(DATA_DIR / "models" / "shot_model.pt")
    lf = pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")
    lam = as_dict(wicket_values(lf))
    states = wicket_value_states(lf)
    starts = start_states(lf)
    print(f"{starts.height} over starts, wicket values for {states.height} states")

    frame = (
        lf.pipe(enc.tracked_innings)
        .pipe(enc.add_labels)
        .pipe(enc.split_column)
        .filter(pl.col("split") != "test")
        .with_columns(bowling_col().alias("bowling"))
        .collect()
    )
    pooled, studied = phase4_fields()

    named = (
        frame.group_by("striker", "striker_id", "format", "phase", "bowling")
        .len()
        .filter(pl.col("len") >= args.min_balls)
        .sort("format", "phase", "bowling", "len", descending=[False, False, False, True])
    )
    if args.limit:
        named = named.head(args.limit)
    batters = sorted(set(named["striker"]) - set(studied))
    rng = np.random.default_rng(7)
    unseen = set(rng.choice(batters, size=round(UNSEEN_SHARE * len(batters)), replace=False))
    print(f"{named.height} batter contexts, {named['striker'].n_unique()} batters "
          f"({len(unseen)} kept out of training)")  # fmt: skip

    specs = [{"batter": "all batters", "batter_id": None, "key": k} for k in KEYS]
    specs += [
        {"batter": who, "batter_id": wid, "key": (fmt, phase, bowling), "balls": n}
        for who, wid, fmt, phase, bowling, n in named.iter_rows()
    ]

    rows_out, learn, judge = [], [], []
    for c, spec in enumerate(specs):
        t0 = time.perf_counter()
        fmt, phase, bowling = spec["key"]
        cond = (
            (pl.col("format") == fmt) & (pl.col("phase") == phase) & (pl.col("bowling") == bowling)
        )
        pooled_context = spec["batter_id"] is None
        if not pooled_context:
            cond &= pl.col("striker_id") == spec["batter_id"]
        rows = frame.filter(cond)
        n_rows = rows.height
        if pooled_context:
            rows = rows.sample(min(rows.height, args.pooled_rows), seed=c)
        scale = 4 if pooled_context else 1
        banks = []
        sizes = [(args.learn_balls * scale, 1000 + 2 * c), (args.judge_balls * scale, 1001 + 2 * c)]
        for n, seed in sizes:
            reps = max(1, int(np.ceil(n / rows.height)))
            banks.append(Shots.from_sample(model.sample(rows, n=reps, seed=seed)))
        learn.append(banks[0])
        judge.append(banks[1])

        analyst = analyst_field(rows.filter(pl.col("outcome") == enc.IN_PLAY), fmt, phase, bowling)
        start = np.array(pooled[spec["key"]])
        searched = start if pooled_context else search(c, spec["key"], banks[0], start, lam, args)

        rows_out.append(
            {
                "context": c,
                "batter": spec["batter"],
                "batter_id": spec["batter_id"],
                "format": fmt,
                "phase": phase,
                "bowling": bowling,
                "key": KEYS.index(spec["key"]),
                "balls": n_rows,
                "pooled": pooled_context,
                "unseen": spec["batter"] in unseen,
                "wickets_median": float(rows["wickets_before"].median()),
                "balls_remaining_median": float(rows["balls_remaining"].median() or 0),
                "ball_age_median": float(rows["ball_age"].median() or 0),
                "profile": profile(banks[0]).tolist(),
                "searched": np.asarray(searched).ravel().tolist(),
                "analyst": np.asarray(analyst).ravel().tolist(),
            }
        )
        if c % 25 == 0 or c == len(specs) - 1:
            elapsed = (time.perf_counter() - t_all) / 60
            took = time.perf_counter() - t0
            print(f"[{c + 1}/{len(specs)}] {spec['batter']} | {fmt} {phase} {bowling}: "
                  f"{took:.1f}s, {elapsed:.1f} min so far", flush=True)  # fmt: skip

    keys = pl.DataFrame(
        {
            "key": list(range(len(KEYS))),
            "format": [k[0] for k in KEYS],
            "phase": [k[1] for k in KEYS],
            "bowling": [k[2] for k in KEYS],
            "wicket_value": [lam[(k[0], k[1])] for k in KEYS],
            "conventional": [template(*k).ravel().tolist() for k in KEYS],
            "optimised": [np.asarray(pooled[k]).ravel().tolist() for k in KEYS],
        }
    )
    data = CaptainData(
        contexts=pl.DataFrame(rows_out),
        keys=keys,
        starts=starts,
        wicket_states=states,
        learn=Bank.from_shots(learn),
        judge=Bank.from_shots(judge),
        forcing_risk=forcing_risk(lf),
    )
    data.save()
    print(f"saved {len(specs)} contexts in {(time.perf_counter() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

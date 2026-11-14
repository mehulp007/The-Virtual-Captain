"""Three famous innings replayed against the captain's fields, and how the captain's fields
differ in shape from the textbook ones.

  reports/case_studies.csv     for each innings and way of setting the field: the expected runs
                               and dismissals had the batter's real shots, ball by ball, met
                               that field (each field set for the batter and the match state of
                               that very ball)
  reports/case_studies.json    the shots and fields behind the case-study figure
  reports/field_shapes.csv     by format and phase: how many fielders each kind of field puts
                               close to the bat, in the deep, on the leg side and behind square

A real shot is replayed as recorded: its direction, how far it went and whether it was in the
air. Each is played 200 times with fresh fielding luck. Against a different field the batter
might have played differently; this measures only how the same shots would have fared.

Uses the app bundle (scripts/build_app_bundle.py) and the deliveries table.

Usage:
  uv run python scripts/case_studies.py
"""

import argparse
import json
import sys

import numpy as np
import polars as pl

from virtual_captain.app.bundle import Bundle
from virtual_captain.app.engine import (
    ALL_BATTERS,
    CAPTAIN,
    CONVENTIONAL,
    PHASE_OVERS,
    SEARCHED,
    Engine,
    Situation,
)
from virtual_captain.field.geometry import PITCH_HALF, in_ring
from virtual_captain.field.positions import with_keeper_and_bowler
from virtual_captain.field.rules import POPPING_CREASE_Y
from virtual_captain.models import encoding as enc
from virtual_captain.paths import PROCESSED_DIR, REPORTS_DIR
from virtual_captain.sim.engine import Shots, simulate
from virtual_captain.sim.params import SimParams, situation_code

INNINGS = [
    {
        "label": "Suryakumar Yadav, 117 off 55",
        "detail": "T20I, England v India, Trent Bridge, 10 July 2022",
        "match_id": 1276906,
        "innings": 2,
        "striker": "SA Yadav",
    },
    {
        "label": "Rohit Sharma, 264 off 173",
        "detail": "ODI, India v Sri Lanka, Eden Gardens, 13 November 2014",
        "match_id": 792295,
        "innings": 1,
        "striker": "RG Sharma",
    },
    {
        "label": "Shubman Gill, 269 off 387",
        "detail": "Test, England v India, Edgbaston, July 2025 (a season no model saw)",
        "match_id": 1448350,
        "innings": 1,
        "striker": "Shubman Gill",
    },
]
SETTERS = [CAPTAIN, CONVENTIONAL, ALL_BATTERS, SEARCHED]
REPLAYS = 200
CLOSE = 15.0  # metres from the striker's stumps: slips, gully, short leg, silly point


def innings_rows(spec: dict) -> pl.DataFrame:
    return (
        pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")
        .filter(
            (pl.col("match_id") == spec["match_id"])
            & (pl.col("innings") == spec["innings"])
            & (pl.col("striker") == spec["striker"])
            & pl.col("is_legal")
        )
        .pipe(enc.add_labels)
        .with_columns(
            pl.when(pl.col("pace_or_spin") == "spin")
            .then(pl.lit("spin"))
            .otherwise(pl.lit("pace"))
            .alias("bowling")
        )
        .sort("over", "ball_in_over")
        .collect()
    )


def context_for(e: Engine, who: str, fmt: str, phase: str, bowling: str) -> tuple[int, bool]:
    """The batter's own context if they have one there, else the all-batters one."""
    b = e.b
    same = (b.format == fmt) & (b.phase == phase) & (b.bowling == bowling)
    own = np.flatnonzero(same & (b.batter == who))
    if len(own):
        return int(own[0]), True
    return int(np.flatnonzero(same & b.pooled)[0]), False


def situation(e: Engine, c: int, row: dict) -> Situation:
    fmt, phase = str(e.b.format[c]), str(e.b.phase[c])
    lo, hi = PHASE_OVERS[(fmt, phase)]
    over = row["ball_age"] // 6 if fmt == "TEST" else row["over"] + 1
    return Situation(c, int(min(row["wickets_before"], 9)), int(np.clip(over, lo, hi)))


def replay(e: Engine, spec: dict, params: SimParams) -> tuple[list[dict], dict]:
    rows = innings_rows(spec)
    fmt = rows["format"][0]
    per_ball, own = [], 0
    for row in rows.iter_rows(named=True):
        c, has_own = context_for(e, spec["striker"], fmt, row["phase"], row["bowling"])
        own += has_own
        s = situation(e, c, row)
        fields = e.fields(s)
        fields.setdefault(SEARCHED, fields[ALL_BATTERS])  # no dedicated search without a profile
        per_ball.append((row, s, [fields[k] for k in SETTERS]))

    n, k = len(per_ball), len(SETTERS)
    shots = Shots.from_sample(rows.select("outcome", "theta", "dist", "aerial"))
    full = np.stack([with_keeper_and_bowler(f, e.keeper_up(s.context))
                     for _, s, fs in per_ball for f in fs])  # fmt: skip
    idx = (np.arange(n)[:, None] * k + np.arange(k)[None]).ravel()
    reps = np.repeat(idx, REPLAYS)
    ball = reps // k
    codes = np.array([situation_code(rows["format"][0], r["phase"]) for r, _, _ in per_ball])
    out = simulate(
        Shots(shots.outcome[ball], shots.theta[ball], shots.dist[ball], shots.aerial[ball]),
        full, reps.astype(np.int32), params, seed=0, situation=codes[ball],
    )  # fmt: skip
    runs = out.runs.reshape(n, k, REPLAYS).mean(2)
    wkts = out.wicket.reshape(n, k, REPLAYS).mean(2)
    lam = np.array([e.wicket_value(s) for _, s, _ in per_ball])
    totals = [
        {
            "innings": spec["label"],
            "setter": name,
            "balls": n,
            "own_profile_share": own / n,
            "actual_runs": int(rows["runs_off_bat"].sum()),
            "expected_runs": float(runs[:, j].sum()),
            "expected_dismissals": float(wkts[:, j].sum()),
            "expected_cost": float((runs[:, j] - lam * wkts[:, j]).sum()),
        }
        for j, name in enumerate(SETTERS)
    ]

    # for the figure: the batter's shots kept in the ground, and the fields for the ball the
    # innings spent most time in (its most common context, at the median state)
    ctx = np.array([s.context for _, s, _ in per_ball])
    main = np.bincount(ctx).argmax()
    at = np.flatnonzero(ctx == main)[len(np.flatnonzero(ctx == main)) // 2]
    _, s_main, fs_main = per_ball[at]
    kept = rows.filter(pl.col("outcome") == enc.IN_PLAY)
    caught = kept["striker_out"].fill_null(False).to_numpy()
    figure = {
        "label": spec["label"],
        "detail": spec["detail"],
        "situation": f"{e.b.format[main]} {e.b.phase[main]}, {e.b.bowling[main]}".replace("_", " "),
        "keeper_up": e.keeper_up(int(main)),
        "left_handed": bool(rows["left_handed"].mean() > 0.5),
        "captain": np.round(fs_main[0], 2).tolist(),
        "conventional": np.round(fs_main[1], 2).tolist(),
        "shots": [
            [round(t, 4), round(d, 4), -1 if out_ else int(r)]
            for t, d, r, out_ in zip(
                kept["theta"].to_list(),
                kept["dist"].to_list(),
                kept["runs_off_bat"].to_list(),
                caught,
                strict=True,
            )
        ],  # fmt: skip
        "totals": {t["setter"]: t for t in totals},
    }
    return totals, figure


def shapes(field: np.ndarray) -> dict[str, int]:
    x, y = field[:, 0], field[:, 1]
    return {
        "close": int((np.hypot(x, y + PITCH_HALF) < CLOSE).sum()),
        "deep": int((~in_ring(x, y)).sum()),
        "leg_side": int((x < 0).sum()),
        "behind_square": int((y < POPPING_CREASE_Y).sum()),
    }


def field_shapes(e: Engine) -> pl.DataFrame:
    rows = []
    for c in np.flatnonzero(~e.b.pooled):
        s = e.typical(int(c))
        for name, field in e.fields(s).items():
            rows.append({"format": str(e.b.format[c]), "phase": str(e.b.phase[c]),
                         "setter": name} | shapes(field))  # fmt: skip
    return (
        pl.DataFrame(rows)
        .group_by("format", "phase", "setter")
        .agg(
            pl.len().alias("contexts"),
            pl.col("close", "deep", "leg_side", "behind_square").mean(),
        )
        .sort("format", "phase", "setter")
    )


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    e = Engine(Bundle.load())
    params = SimParams.load()
    totals, figures = [], {}
    for spec in INNINGS:
        t, f = replay(e, spec, params)
        totals += t
        figures[spec["label"]] = f
        print(f"{spec['label']}: {t[0]['balls']} tracked balls, "
              f"{t[0]['own_profile_share']:.0%} with the batter's own profile")  # fmt: skip
        for r in t:
            print(f"  {r['setter']:<34} {r['expected_runs']:6.1f} runs "
                  f"{r['expected_dismissals']:5.2f} out  (actual {r['actual_runs']})")  # fmt: skip
    pl.DataFrame(totals).write_csv(REPORTS_DIR / "case_studies.csv", float_precision=4)
    (REPORTS_DIR / "case_studies.json").write_text(json.dumps(figures, indent=1) + "\n")
    sh = field_shapes(e)
    sh.write_csv(REPORTS_DIR / "field_shapes.csv", float_precision=3)
    with pl.Config(tbl_rows=60, tbl_width_chars=160):
        print(sh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fit the simulator's physical parameters to real scoring patterns.

The historical data says how often balls went for 0, 1, 2-3, 4 or 6 and how often they were
caught, but not where the fielders stood. So each delivery is replayed against a stand-in
for the field the captain would have set, using a shot sampled from the shot model, and the
parameters are tuned until the simulated rates match the real ones.

The stand-in starts from the conventional field for the format, phase and type of bowling,
then moves its boundary riders to where that batter scores most of their runs in the deep,
as any captain with a scorecard would. Batter profiles come from training seasons only.
"""

from dataclasses import dataclass

import numpy as np
import optuna
import polars as pl

from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF
from virtual_captain.field.positions import DEEP, POSITIONS, field_xy, with_keeper_and_bowler, xy
from virtual_captain.field.rules import repair
from virtual_captain.field.templates import KEYS, TEMPLATES
from virtual_captain.models import encoding as enc
from virtual_captain.sim import physics
from virtual_captain.sim.engine import Shots, simulate
from virtual_captain.sim.params import CALIBRATED, SITUATIONS, SimParams

RATES = ["dot", "one", "two_three", "four", "six", "caught_edge", "caught_close", "caught_far"]
CLOSE_CATCH = 30.0  # metres from the striker
N_SECTORS = 8
RATE_FLOOR = 0.01  # relative errors on very rare outcomes are measured against at least this


PROFILE_SECTORS = 24
MIN_DEEP_BALLS = 40


def deep_profiles(rows: pl.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """How often the ball goes to the deep in each direction (24 sectors), per batter and
    format, counting every ball that got there: boundaries, catches and fielded hits."""
    sector = (pl.col("theta") / (2 * np.pi) * PROFILE_SECTORS).floor().cast(pl.Int64)
    deep = rows.filter(
        pl.col("outcome").is_in([enc.IN_PLAY, enc.SIX])
        & (pl.col("dist") >= 0.75)
        & pl.col("striker_id").is_not_null()
    ).with_columns((sector % PROFILE_SECTORS).alias("sector"))
    enough = deep.group_by("striker_id", "format").len().filter(pl.col("len") >= MIN_DEEP_BALLS)
    counts = (
        deep.join(enough.select("striker_id", "format"), on=["striker_id", "format"])
        .group_by("striker_id", "format", "sector")
        .agg(pl.len().alias("balls"))
    )
    profiles: dict[tuple[str, str], np.ndarray] = {}
    for (batter, fmt), g in counts.group_by(["striker_id", "format"]):
        w = np.zeros(PROFILE_SECTORS)
        w[g["sector"].to_numpy()] = g["balls"].to_numpy()
        profiles[(batter, fmt)] = w
    return profiles


def adapted_field(key: tuple[str, str, str], weights: np.ndarray | None) -> np.ndarray:
    """Template field with its boundary riders moved to the batter's busiest deep sectors."""
    names = TEMPLATES[key]
    field = field_xy(names)
    deep = [i for i, n in enumerate(names) if POSITIONS[n][1] > 60]
    if weights is not None and deep:
        w = weights.astype(float).copy()
        width = 360 / PROFILE_SECTORS
        for i in deep:
            s = int(np.argmax(w))
            field[i] = xy(((s + 0.5) * width, DEEP))
            # don't stack two riders in the same or neighbouring sectors
            w[[s, (s - 1) % PROFILE_SECTORS, (s + 1) % PROFILE_SECTORS]] = -1.0
        field = repair(field, key[0], key[1])
    return with_keeper_and_bowler(field, keeper_up=key[2] == "spin")


def with_group(df: pl.DataFrame) -> pl.DataFrame:
    kind = pl.when(pl.col("pace_or_spin") == "spin").then(pl.lit("spin")).otherwise(pl.lit("pace"))
    key = pl.concat_str([pl.col("format"), pl.col("phase"), kind], separator="|")
    lookup = {"|".join(k): i for i, k in enumerate(KEYS)}
    return df.with_columns(kind.alias("bowling"), key.replace_strict(lookup).alias("group"))


@dataclass
class CalibrationSet:
    rows: pl.DataFrame
    shots: Shots
    fields: np.ndarray
    uniforms: np.ndarray
    field_idx: np.ndarray
    real: np.ndarray  # (n_groups, n_rates)
    real_sector: np.ndarray  # (n_formats, n_sectors) runs per ball in play by direction


def situation_codes(rows: pl.DataFrame) -> np.ndarray:
    lookup = {f"{f}|{p}": i for i, (f, p) in enumerate(SITUATIONS)}
    key = pl.concat_str([pl.col("format"), pl.col("phase")], separator="|")
    return rows.select(key.replace_strict(lookup, return_dtype=pl.Int64)).to_series().to_numpy()


def _sector(theta_rad: np.ndarray) -> np.ndarray:
    return (theta_rad / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS


FORMATS = ["T20", "ODI", "TEST"]


def real_rates(rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    caught = pl.col("striker_out") & pl.col("wicket_type").is_in(["caught", "caught and bowled"])
    from_striker = (pl.col("x_m").pow(2) + (pl.col("y_m") + PITCH_HALF).pow(2)).sqrt()
    close = from_striker.fill_null(0.0) < CLOSE_CATCH
    behind = pl.col("y_m").fill_null(0.0) < -PITCH_HALF  # keeper, slips, gully, leg slip
    g = (
        rows.group_by("group")
        .agg(
            ((pl.col("runs_off_bat") == 0) & ~pl.col("striker_out")).mean().alias("dot"),
            (pl.col("runs_off_bat") == 1).mean().alias("one"),
            pl.col("runs_off_bat").is_in([2, 3]).mean().alias("two_three"),
            pl.col("is_four").mean().alias("four"),
            pl.col("is_six").mean().alias("six"),
            (caught & close & behind).mean().alias("caught_edge"),
            (caught & close & ~behind).mean().alias("caught_close"),
            (caught & ~close).mean().alias("caught_far"),
        )
        .sort("group")
    )
    rates = np.zeros((len(KEYS), len(RATES)))
    rates[g["group"].to_numpy()] = g.select(RATES).to_numpy()

    in_play = rows.filter(pl.col("outcome") == enc.IN_PLAY)
    sector = np.zeros((len(FORMATS), N_SECTORS))
    for f, fmt in enumerate(FORMATS):
        d = in_play.filter(pl.col("format") == fmt)
        s = _sector(d["theta"].to_numpy())
        runs = d["runs_off_bat"].to_numpy()
        sector[f] = np.bincount(s, weights=runs, minlength=N_SECTORS) / np.maximum(
            np.bincount(s, minlength=N_SECTORS), 1
        )
    return rates, sector


def sim_rates(cal: CalibrationSet, params: SimParams) -> tuple[np.ndarray, np.ndarray]:
    group = cal.rows["group"].to_numpy()
    result = simulate(
        cal.shots,
        cal.fields,
        cal.field_idx,
        params,
        uniforms=cal.uniforms,
        situation=situation_codes(cal.rows),
    )
    runs, kind = result.runs, result.kind
    ex = -np.sin(cal.shots.theta) * cal.shots.dist * BOUNDARY_RADIUS
    ey = -np.cos(cal.shots.theta) * cal.shots.dist * BOUNDARY_RADIUS
    close = np.hypot(ex, ey + PITCH_HALF) < CLOSE_CATCH
    behind = ey < -PITCH_HALF
    columns = [
        (runs == 0) & ~result.wicket,
        runs == 1,
        (runs == 2) | (runs == 3),
        kind == physics.FOUR,
        kind == physics.SIX,
        (kind == physics.CAUGHT) & close & behind,
        (kind == physics.CAUGHT) & close & ~behind,
        (kind == physics.CAUGHT) & ~close,
    ]
    counts = np.maximum(np.bincount(group, minlength=len(KEYS)), 1)
    rates = np.stack(
        [np.bincount(group, weights=c, minlength=len(KEYS)) / counts for c in columns], axis=1
    )

    fmt = cal.rows["format"].replace_strict({f: i for i, f in enumerate(FORMATS)}).to_numpy()
    in_play = cal.shots.outcome == enc.IN_PLAY
    s = _sector(cal.shots.theta)
    key = fmt * N_SECTORS + s
    total = np.bincount(key[in_play], weights=runs[in_play], minlength=len(FORMATS) * N_SECTORS)
    n = np.maximum(np.bincount(key[in_play], minlength=len(FORMATS) * N_SECTORS), 1)
    return rates, (total / n).reshape(len(FORMATS), N_SECTORS)


def build_set(
    rows: pl.DataFrame, model, per_group: int, profiles: dict, seed: int = 0
) -> CalibrationSet:
    rows = with_group(rows)
    rows = (
        rows.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("group").alias("_r"))
        .filter(pl.col("_r") < per_group)
        .drop("_r")
    )
    sample = model.sample(rows, n=1, seed=seed)
    real, real_sector = real_rates(rows)

    # one field per (template group, batter); batters without a profile share the template
    cols = (rows[c].to_list() for c in ("group", "striker_id", "format"))
    keys = list(zip(*cols, strict=True))
    index: dict[tuple, int] = {}
    fields, field_idx = [], np.empty(len(keys), dtype=np.int32)
    for i, (g, batter, fmt) in enumerate(keys):
        prof = profiles.get((batter, fmt))
        k = (g, batter if prof is not None else None)
        if k not in index:
            index[k] = len(fields)
            fields.append(adapted_field(KEYS[g], prof))
        field_idx[i] = index[k]

    return CalibrationSet(
        rows=rows,
        shots=Shots.from_sample(sample),
        fields=np.stack(fields),
        field_idx=field_idx,
        uniforms=np.random.default_rng(seed).random((rows.height, 4)),
        real=real,
        real_sector=real_sector,
    )


def loss(cal: CalibrationSet, params: SimParams) -> float:
    sim, sim_sector = sim_rates(cal, params)
    rel = (sim - cal.real) / np.maximum(cal.real, RATE_FLOOR)
    sector_rel = (sim_sector - cal.real_sector) / np.maximum(cal.real_sector, 0.1)
    return float((rel**2).mean() + 0.5 * (sector_rel**2).mean())


def fit(
    cal: CalibrationSet, n_trials: int = 300, n_refine: int = 300, seed: int = 0
) -> tuple[SimParams, float]:
    """Global search with TPE, then local refinement with CMA-ES from the best point.

    Both work in coordinates normalised to [0, 1] per parameter, so CMA-ES's step size
    means the same thing for a deceleration in m/s^2 and a misfield probability.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = SimParams().__dict__
    names = list(CALIBRATED)
    lo = np.array([CALIBRATED[k][0] for k in names])
    hi = np.array([CALIBRATED[k][1] for k in names])

    def params_from(trial) -> SimParams:
        u = np.array([trial.suggest_float(k, 0.0, 1.0) for k in names])
        return SimParams(**{**base, **dict(zip(names, lo + u * (hi - lo), strict=True))})

    def objective(trial):
        return loss(cal, params_from(trial))

    tpe = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    tpe.enqueue_trial({k: (base[k] - lo[i]) / (hi[i] - lo[i]) for i, k in enumerate(names)})
    tpe.optimize(objective, n_trials=n_trials)

    best_u, best_value = tpe.best_params, tpe.best_value
    if n_refine > 0:
        cma = optuna.create_study(
            sampler=optuna.samplers.CmaEsSampler(x0=best_u, sigma0=0.1, seed=seed)
        )
        cma.optimize(objective, n_trials=n_refine)
        if cma.best_value < best_value:
            best_u, best_value = cma.best_params, cma.best_value

    u = np.array([best_u[k] for k in names])
    best = SimParams(**{**base, **dict(zip(names, lo + u * (hi - lo), strict=True))})
    return best, best_value


def report(cal: CalibrationSet, params: SimParams) -> pl.DataFrame:
    """Real vs simulated rates per template group."""
    sim, _ = sim_rates(cal, params)
    sizes = np.bincount(cal.rows["group"].to_numpy(), minlength=len(KEYS))
    out = []
    for g, (fmt, phase, bowling) in enumerate(KEYS):
        if sizes[g] == 0:
            continue
        row = {"format": fmt, "phase": phase, "bowling": bowling, "balls": int(sizes[g])}
        for i, r in enumerate(RATES):
            row[f"real_{r}"] = cal.real[g, i]
            row[f"sim_{r}"] = sim[g, i]
        real_rpb = cal.rows.filter(pl.col("group") == g)["runs_off_bat"].mean()
        row["real_runs_per_ball"] = real_rpb
        for side in ("real", "sim"):
            row[f"{side}_caught"] = sum(row[f"{side}_caught_{z}"] for z in ("edge", "close", "far"))
        out.append(row)
    df = pl.DataFrame(out)
    group = cal.rows["group"].to_numpy()
    runs = simulate(
        cal.shots,
        cal.fields,
        cal.field_idx,
        params,
        uniforms=cal.uniforms,
        situation=situation_codes(cal.rows),
    ).runs
    sim_rpb = np.bincount(group, weights=runs, minlength=len(KEYS)) / np.maximum(sizes, 1)
    return df.with_columns(pl.Series("sim_runs_per_ball", sim_rpb[sizes > 0]))

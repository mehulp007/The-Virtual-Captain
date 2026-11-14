"""The batter's side of the game: an intent that reshapes their shots, and what it risks.

A batter can't play shots they don't have. Their intent only changes how often they play the
shots they do: the deliveries simulated for them are reweighted (exponential tilting) by

  aggression   in [-1, 1]: towards the long shots and sixes, or towards the short ones
  tilt         a vector in the unit disk: towards one direction round the ground

Only the balls the batter hits are reweighted, so intent can't make them connect more often
or get bowled less. With no intent every delivery has the same weight, the batter's natural
game.

Changing your game costs something. The forcing risk is fitted to the data by
`fit_forcing_risk`: when a batter's scoring pattern moves away from their usual one (a
different phase of the innings, same batter, same format, same type of bowling), they get out
more often than the new mix of shots explains on its own. That excess grows with how far the
pattern moved, measured as KL divergence. Here the same relation turns the KL divergence of the
reweighted shots into an extra chance of a dismissal on the ball.
"""

import numpy as np
import polars as pl

from virtual_captain.data.coords import xy_to_wagon
from virtual_captain.field.geometry import in_ring
from virtual_captain.models import encoding as enc
from virtual_captain.rl.data import N_PROFILE

INTENT_DIM = 3
AGGRESSION = 3.0  # strength of aggression, per unit of the attack score below
TILT = 2.0  # strength of the tilt towards a direction
SIX_REACH = 1.2  # a six scores as a shot that would have carried 1.2 x the boundary radius
MAX_FORCING = 0.5  # cap on the extra dismissal chance of one ball
FIELD_SECTORS = 12
N_STATE = 17  # format, bowling, situation, wickets, progress, wicket value, ball of the over
N_BATTER_OBS = N_STATE + N_PROFILE + 2 * FIELD_SECTORS

PATTERN_CELLS = 3 + 2 * 12  # no contact, bowled/lbw, six, then 12 sectors x (ring, deep)
DEEP_FROM = 30.0 / 68.0


def attack_score(outcome: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """How far each delivery went: its distance as a share of the boundary radius, SIX_REACH for
    a six and nothing for a ball not hit."""
    return np.where(outcome == 3, SIX_REACH, np.where(outcome == 2, dist, 0.0))


def intent_weights(
    intent: np.ndarray, outcome: np.ndarray, theta: np.ndarray, dist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weights (n, k) over each ball's k deliveries under the batter's intent (n, 3), and the
    KL divergence (n,) of the reweighted deliveries from the natural ones."""
    intent = np.clip(np.asarray(intent, dtype=np.float64), -1.0, 1.0)
    aggression = intent[:, :1]
    tilt = intent[:, 1:3]
    tilt = tilt / np.maximum(1.0, np.hypot(tilt[:, :1], tilt[:, 1:2]))
    hit = (outcome == 2) | (outcome == 3)
    k = outcome.shape[1]
    # intent changes which shots are played, not how often the batter connects: the weight
    # moves between the balls hit, and the misses and bowled/lbw keep their natural share
    toward = tilt[:, :1] * np.cos(theta) + tilt[:, 1:2] * np.sin(theta)
    logw = AGGRESSION * aggression * attack_score(outcome, dist) + TILT * toward
    logw = np.where(hit, logw, -np.inf)
    logw -= np.max(np.where(hit, logw, -1e300), axis=1, keepdims=True)
    w = np.exp(logw)
    hits = hit.sum(axis=1, keepdims=True)
    w = np.where(hit, w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300) * hits / k, 1.0 / k)
    kl = np.sum(w * np.log(np.maximum(w * k, 1e-300)), axis=1)
    return w, np.maximum(kl, 0.0)


def forcing_chance(kl: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Extra chance of a dismissal on a ball from forcing the game by `kl` nats."""
    return np.minimum(kappa * kl, MAX_FORCING)


def field_coverage(fields: np.ndarray) -> np.ndarray:
    """(n, 9, 2) -> (n, 24): fielders inside and outside the ring in each 30-degree sector of
    the wagon wheel, the batter's view of the field."""
    angle, _ = xy_to_wagon(fields[..., 0], fields[..., 1])
    sector = (angle // (360 / FIELD_SECTORS)).astype(int) % FIELD_SECTORS
    deep = ~in_ring(fields[..., 0], fields[..., 1])
    slot = sector + FIELD_SECTORS * deep
    out = np.zeros((len(fields), 2 * FIELD_SECTORS), dtype=np.float32)
    np.add.at(out, (np.repeat(np.arange(len(fields)), 9), slot.ravel()), 1.0)
    return out


# Fitting the forcing risk ---------------------------------------------------------------
def _pattern_cell() -> pl.Expr:
    sector = ((pl.col("theta") / (2 * np.pi) * 12).floor().cast(pl.Int32)) % 12
    return (
        pl.when(pl.col("outcome") == 0).then(0)
        .when(pl.col("outcome") == 1).then(1)
        .when(pl.col("outcome") == 3).then(2)
        .otherwise(3 + sector * 2 + (pl.col("dist") >= DEEP_FROM).cast(pl.Int32))
        .alias("cell")
    )  # fmt: skip


def pattern_shifts(deliveries: pl.LazyFrame, min_balls: int = 300) -> pl.DataFrame:
    """For every batter, format, type of bowling and phase: how far the batter's pattern in that
    phase is from their pattern overall (KL divergence over 27 cells: no contact, bowled or lbw,
    six, and 12 sectors split at the ring), and how much more often they got out than the
    phase's mix of shots explains, given the batter's own dismissal rate in each cell."""
    df = (
        deliveries.pipe(enc.tracked_innings)
        .pipe(enc.add_labels)
        .pipe(enc.split_column)
        .filter(pl.col("split") == "train")
        .with_columns(
            pl.when(pl.col("pace_or_spin") == "spin")
            .then(pl.lit("spin"))
            .otherwise(pl.lit("pace"))
            .alias("bowling"),
            _pattern_cell(),
        )
        .select("striker", "format", "bowling", "phase", "cell", "striker_out")
        .collect()
    )
    g = ["striker", "format", "bowling"]
    overall = df.group_by(g + ["cell"]).agg(
        pl.len().alias("n_cell"), pl.col("striker_out").mean().alias("out_cell")
    )
    by_phase = (
        df.group_by(g + ["phase", "cell"])
        .agg(pl.len().alias("n"))
        .with_columns(
            pl.col("n").sum().over(g + ["phase"]).alias("balls"),
            pl.col("n").sum().over(g).alias("balls_all"),
        )
        .join(overall, on=g + ["cell"])
        .with_columns(
            (pl.col("n") / pl.col("balls")).alias("share"),
            (pl.col("n_cell") / pl.col("balls_all")).alias("share_all"),
        )
    )
    shifts = by_phase.group_by(g + ["phase"]).agg(
        pl.col("balls").first(),
        pl.col("balls_all").first(),
        (pl.col("share") * pl.col("out_cell")).sum().alias("expected_out"),
        (pl.col("share") * (pl.col("share") / pl.col("share_all")).log()).sum().alias("kl"),
    )
    actual = df.group_by(g + ["phase"]).agg(pl.col("striker_out").mean().alias("out"))
    return (
        shifts.join(actual, on=g + ["phase"])
        .filter((pl.col("balls") >= min_balls) & (pl.col("balls") < pl.col("balls_all")))
        .with_columns((pl.col("out") - pl.col("expected_out")).alias("excess_out"))
        .sort(g + ["phase"])
    )


def fit_forcing_risk(shifts: pl.DataFrame) -> pl.DataFrame:
    """Extra dismissals per ball for each nat of pattern shift, by format: a line through the
    origin fitted by balls-weighted least squares, with a robust 95% interval. `kappa` is the
    fit, floored at zero."""
    rows = []
    for fmt in ("T20", "ODI", "TEST"):
        part = shifts.filter(pl.col("format") == fmt)
        w = part["balls"].to_numpy().astype(float)
        x, y = part["kl"].to_numpy(), part["excess_out"].to_numpy()
        sxx = (w * x * x).sum()
        slope = (w * x * y).sum() / sxx
        se = np.sqrt(((w * x * (y - slope * x)) ** 2).sum()) / sxx
        rows.append(
            {"format": fmt, "fitted": slope, "ci": 1.96 * se, "kappa": max(slope, 0.0),
             "cases": len(x)}
        )  # fmt: skip
    return pl.DataFrame(rows)

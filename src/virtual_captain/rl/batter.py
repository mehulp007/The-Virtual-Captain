"""The batter's side of the game: an intent that reshapes their shots, and what it risks.

A batter can't play shots they don't have. Their intent only changes how often they play the
shots they do: the deliveries simulated for them are reweighted (exponential tilting) by

  aggression   in [-1, 1]: towards the big shots (sixes, and the catches that come with
               going for them), or towards the short ones
  tilt         a vector in the unit disk: towards one direction round the ground (for balls
               kept in the ground: a six has no recorded direction)

Only the balls the batter hits are reweighted, so intent can't make them connect more often
or get bowled less. With no intent every delivery has the same weight, the batter's natural
game.

Changing your game costs something. The forcing risk is fitted to the data by
`fit_forcing_risk`: when a batter's game moves away from their usual one (a different phase of
the innings, same batter, same format, same type of bowling), they get out more often than the
new mix of shots explains on its own. That excess grows with how much more they attack (their
mean attack score, below) and with how far their scoring pattern moved (KL divergence). Here
the same relation turns the change in the reweighted shots into an extra chance of a dismissal
on the ball. And no intent may move a batter's game further than batters' games are seen to
move in the data.
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
MAX_SHIFT_QUANTILE = 0.99
FIELD_SECTORS = 12
N_STATE = 17  # format, bowling, situation, wickets, progress, wicket value, ball of the over
N_BATTER_OBS = N_STATE + N_PROFILE + 2 * FIELD_SECTORS

PATTERN_CELLS = 3 + 2 * 12  # no contact, bowled/lbw, six, then 12 sectors x (ring, deep)
DEEP_FROM = 30.0 / 68.0


def attack_score(outcome: np.ndarray, dist: np.ndarray, aerial: np.ndarray) -> np.ndarray:
    """How hard each delivery was hit: its distance as a share of the boundary radius, and
    SIX_REACH for a six. A catch or a drop scores like a six: it is nearly always a big shot
    that went wrong, so a batter who goes for more sixes gets more chances too, in their own
    ratio of the two."""
    big = (outcome == 3) | ((outcome == 2) & aerial)
    return np.where(big, SIX_REACH, np.where(outcome == 2, dist, 0.0))


def _weights(logw: np.ndarray, hit: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    logw = np.where(hit, logw, -np.inf)
    logw = logw - np.max(np.where(hit, logw, -1e300), axis=1, keepdims=True)
    w = np.exp(logw)
    hits = hit.sum(axis=1, keepdims=True)
    w = np.where(hit, w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300) * hits / k, 1.0 / k)
    kl = np.sum(w * np.log(np.maximum(w * k, 1e-300)), axis=1)
    return w, np.maximum(kl, 0.0)


def intent_weights(
    intent: np.ndarray,
    outcome: np.ndarray,
    theta: np.ndarray,
    dist: np.ndarray,
    aerial: np.ndarray,
    max_shift: np.ndarray | float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Weights (n, k) over each ball's k deliveries under the batter's intent (n, 3), and the
    KL divergence (n,) of the reweighted deliveries from the natural ones.

    An intent that would move the batter's game by more than `max_shift` nats is scaled back,
    keeping its direction, until it moves it by exactly that much.
    """
    intent = np.clip(np.asarray(intent, dtype=np.float64), -1.0, 1.0)
    aggression = intent[:, :1]
    tilt = intent[:, 1:3]
    tilt = tilt / np.maximum(1.0, np.hypot(tilt[:, :1], tilt[:, 1:2]))
    # intent changes which shots are played, not how often the batter connects: the weight
    # moves between the balls hit, and the misses and bowled/lbw keep their natural share
    hit = (outcome == 2) | (outcome == 3)
    k = outcome.shape[1]
    # sixes have no wagon-wheel point, so only aggression moves them, not the tilt
    toward = (outcome == 2) * (tilt[:, :1] * np.cos(theta) + tilt[:, 1:2] * np.sin(theta))
    logw = AGGRESSION * aggression * attack_score(outcome, dist, aerial) + TILT * toward
    w, kl = _weights(logw, hit, k)
    cap = np.broadcast_to(np.asarray(max_shift, dtype=np.float64), kl.shape)
    over = kl > cap
    if over.any():
        # the KL divergence grows with the strength of the tilt, so bisect on the strength
        lo, hi = np.zeros(over.sum()), np.ones(over.sum())
        lw, h = logw[over], hit[over]
        for _ in range(25):
            mid = (lo + hi) / 2
            _, kl_mid = _weights(lw * mid[:, None], h, k)
            big = kl_mid > cap[over]
            hi = np.where(big, mid, hi)
            lo = np.where(big, lo, mid)
        w[over], kl[over] = _weights(lw * lo[:, None], h, k)
    return w, kl


def attack_shift(w: np.ndarray, outcome, dist, aerial) -> np.ndarray:
    """(n,) How much the reweighted deliveries' mean attack score rose over the natural one."""
    score = attack_score(outcome, dist, aerial)
    return (w * score).sum(axis=1) - score.mean(axis=1)


def forcing_chance(
    d_attack: np.ndarray, kl: np.ndarray, per_attack: np.ndarray, per_shift: np.ndarray
) -> np.ndarray:
    """Extra chance of a dismissal on a ball from raising the attack score by `d_attack` and
    moving the scoring pattern by `kl` nats. Playing safer than usual earns no credit beyond
    what the safer shots themselves bring."""
    return np.clip(per_attack * d_attack + per_shift * kl, 0.0, MAX_FORCING)


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
    six, and 12 sectors split at the ring), how much their mean attack score changed, and how
    much more often they got out than the phase's mix of shots explains, given the batter's own
    dismissal rate in each cell."""
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
            pl.when((pl.col("outcome") == 3) | ((pl.col("outcome") == 2) & pl.col("aerial")))
            .then(SIX_REACH)
            .when(pl.col("outcome") == 2)
            .then(pl.col("dist"))
            .otherwise(0.0)
            .alias("attack"),
        )
        .select("striker", "format", "bowling", "phase", "cell", "striker_out", "attack")
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
    actual = df.group_by(g + ["phase"]).agg(
        pl.col("striker_out").mean().alias("out"), pl.col("attack").mean()
    )
    usual = df.group_by(g).agg(pl.col("attack").mean().alias("attack_all"))
    return (
        shifts.join(actual, on=g + ["phase"])
        .join(usual, on=g)
        .filter((pl.col("balls") >= min_balls) & (pl.col("balls") < pl.col("balls_all")))
        .with_columns(
            (pl.col("out") - pl.col("expected_out")).alias("excess_out"),
            (pl.col("attack") - pl.col("attack_all")).alias("d_attack"),
        )
        .sort(g + ["phase"])
    )


def fit_forcing_risk(shifts: pl.DataFrame) -> pl.DataFrame:
    """By format, the excess dismissals per ball explained by the change in attack score and the
    pattern shift: least squares through the origin, weighted by balls, with robust 95%
    intervals. Both are floored at zero for use. `max_shift` is as far as batters' patterns are
    seen to move (the 99th percentile), which is as far as an intent may move a batter's."""
    rows = []
    for fmt in ("T20", "ODI", "TEST"):
        part = shifts.filter(pl.col("format") == fmt)
        w = part["balls"].to_numpy().astype(float)
        x = np.stack([part["d_attack"].to_numpy(), part["kl"].to_numpy()], axis=1)
        y = part["excess_out"].to_numpy()
        bread = np.linalg.inv((x * w[:, None]).T @ x)
        coef = bread @ ((x * w[:, None]).T @ y)
        score = x * (w * (y - x @ coef))[:, None]
        se = np.sqrt(np.diag(bread @ (score.T @ score) @ bread))
        rows.append(
            {
                "format": fmt,
                "per_attack": max(coef[0], 0.0),
                "per_attack_fit": coef[0],
                "per_attack_ci": 1.96 * se[0],
                "per_shift": max(coef[1], 0.0),
                "per_shift_fit": coef[1],
                "per_shift_ci": 1.96 * se[1],
                "max_shift": float(np.quantile(part["kl"].to_numpy(), MAX_SHIFT_QUANTILE)),
                "cases": len(y),
            }
        )
    return pl.DataFrame(rows)

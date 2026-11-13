"""Turn rows of the deliveries table into model inputs and labels.

Only innings where Cricinfo tracked (almost) every ball are used. Within those, a ball with
no wagon-wheel point that wasn't scored off was left, missed or blocked dead, so the
absence of a wagon point carries information instead of being missing data.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

HOLDOUT_FROM_YEAR = 2025
MIN_SHOT_COVERAGE = 0.9

# Contact outcomes
NO_CONTACT, FIELD_FREE_OUT, IN_PLAY = 0, 1, 2
OUTCOMES = ("no_contact", "bowled_lbw_stumped", "in_play")
FIELD_FREE_WICKETS = ["bowled", "lbw", "stumped", "hit wicket"]

D_MIN, D_MAX = 0.01, 1.5  # wagon distance, as a fraction of the boundary radius

CATEGORIES = {
    "format_phase": [
        "T20/powerplay", "T20/middle", "T20/death",
        "ODI/powerplay", "ODI/middle", "ODI/death",
        "TEST/new_ball", "TEST/old_ball",
    ],
    "bowler_kind": ["fast", "medium", "finger_spin", "wrist_spin", "other_spin", "unknown"],
    "bowler_arm": ["left", "right", "unknown"],
    "spin_turn": ["in", "away", "none"],
    "ball_line": [
        "wide outside offstump", "outside offstump", "on the stumps", "down leg",
        "wide down leg", "unknown",
    ],
    "ball_length": [
        "yorker", "full toss", "full", "good length", "short of a good length", "short",
        "unknown",
    ],
}  # fmt: skip

VENUE_COLS = ["venue_runs_per_ball", "venue_boundary_rate", "venue_spin_wicket_share"]

NUMERIC = [
    "left_handed", "franchise", "year_c", "wickets", "innings_progress", "chasing",
    "required_rate", "run_rate", "set_batter", "test_innings", "same_arm",
]  # fmt: skip


def n_features() -> int:
    return len(NUMERIC) + len(VENUE_COLS) + 1 + sum(len(v) for v in CATEGORIES.values())


def tracked_innings(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Men's legal deliveries from innings with near-complete shot tracking."""
    legal = lf.filter((pl.col("gender") == "male") & pl.col("is_legal"))
    cov = legal.group_by("match_id", "innings").agg(
        pl.col("shot_played").is_not_null().mean().alias("_cov")
    )
    return legal.join(
        cov.filter(pl.col("_cov") >= MIN_SHOT_COVERAGE), on=["match_id", "innings"]
    ).drop("_cov")


def add_labels(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Contact outcome, plus aerial flag and polar shot for balls put into play.

    Rows that can't be labelled cleanly (scored off but untracked, or caught/run out
    without a wagon point) are dropped.
    """
    field_free = pl.col("striker_out") & pl.col("wicket_type").is_in(FIELD_FREE_WICKETS)
    outcome = (
        pl.when(pl.col("has_wagon"))
        .then(IN_PLAY)
        .when(field_free)
        .then(FIELD_FREE_OUT)
        .when((pl.col("runs_off_bat") == 0) & ~pl.col("striker_out"))
        .then(NO_CONTACT)
        .otherwise(None)
    )
    caught = pl.col("wicket_type").is_in(["caught", "caught and bowled"]) & pl.col("striker_out")
    dropped = pl.col("ball_events").fill_null("").str.to_lowercase().str.contains("drop")
    return (
        lf.with_columns(outcome.cast(pl.Int8).alias("outcome"))
        .filter(pl.col("outcome").is_not_null())
        .with_columns(
            (pl.col("is_six") | caught | dropped).alias("aerial"),
            pl.col("wagon_angle").radians().alias("theta"),
            pl.col("wagon_distance").clip(D_MIN, D_MAX).alias("dist"),
        )
    )


def split_column(lf: pl.LazyFrame) -> pl.LazyFrame:
    """train / val / test. Test is every season from 2025; val is 1 match in 20 before that."""
    return lf.with_columns(
        pl.when(pl.col("year") >= HOLDOUT_FROM_YEAR)
        .then(pl.lit("test"))
        .when(pl.col("match_id").hash(seed=7) % 20 == 0)
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


@dataclass
class Vocab:
    batters: list[str] = field(default_factory=list)
    bowlers: list[str] = field(default_factory=list)
    venue_mean: list[float] = field(default_factory=list)
    venue_std: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, train: pl.DataFrame, min_batter_balls=300, min_bowler_balls=500) -> "Vocab":
        def frequent(col, n):
            counts = train.group_by(col).len().filter(pl.col("len") >= n)
            return sorted(counts[col].drop_nulls().to_list())

        venue = train.select(VENUE_COLS)
        return cls(
            batters=frequent("striker_id", min_batter_balls),
            bowlers=frequent("bowler_id", min_bowler_balls),
            venue_mean=[float(venue[c].mean()) for c in VENUE_COLS],
            venue_std=[float(venue[c].std()) for c in VENUE_COLS],
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def n_batters(self) -> int:
        return len(self.batters) + 1  # index 0 is "everyone else"

    @property
    def n_bowlers(self) -> int:
        return len(self.bowlers) + 1


def _numeric_exprs() -> list[pl.Expr]:
    fmt = pl.col("format")
    limited = fmt != "TEST"
    scheduled = (pl.col("balls_before") + pl.col("balls_remaining")).cast(pl.Float32)
    progress = (
        pl.when(limited)
        .then(pl.col("balls_before") / scheduled.clip(lower_bound=1))
        .otherwise(pl.col("ball_age") / 480)
    )
    chasing = pl.col("target").is_not_null()
    overs_left = pl.col("balls_remaining").clip(lower_bound=1) / 6
    rrr = pl.when(chasing).then((pl.col("runs_required") / overs_left).clip(0, 36) / 12)
    rr = pl.col("runs_before") / (pl.col("balls_before").clip(lower_bound=6) / 6) / 12
    return [
        pl.col("left_handed").fill_null(False).cast(pl.Float32).alias("left_handed"),
        (pl.col("competition_type") != "Intl").cast(pl.Float32).alias("franchise"),
        ((pl.col("year") - 2016) / 10).alias("year_c"),
        (pl.col("wickets_before") / 10).alias("wickets"),
        progress.clip(0, 1).alias("innings_progress"),
        chasing.cast(pl.Float32).alias("chasing"),
        rrr.fill_null(0).alias("required_rate"),
        rr.clip(0, 3).alias("run_rate"),
        (pl.col("striker_balls_faced").fill_null(0).log1p() / 5).alias("set_batter"),
        pl.when(limited).then(0).otherwise((pl.col("innings") - 1) / 3).alias("test_innings"),
        (pl.col("arm_vs_bat") == "same").fill_null(False).cast(pl.Float32).alias("same_arm"),
    ]


def encode(df: pl.DataFrame, vocab: Vocab) -> dict[str, np.ndarray]:
    """Feature matrix `x` plus batter / bowler indices for the embeddings."""
    cats = df.select(
        (pl.col("format") + "/" + pl.col("phase")).alias("format_phase"),
        pl.col("bowler_kind").fill_null("unknown"),
        pl.col("bowler_arm").fill_null("unknown"),
        pl.col("spin_turn").fill_null("none"),
        pl.col("ball_line").fill_null("unknown"),
        pl.col("ball_length").fill_null("unknown"),
    )
    blocks = [df.select(_numeric_exprs()).to_numpy().astype(np.float32)]

    venue = df.select(VENUE_COLS).to_numpy().astype(np.float32)
    missing = np.isnan(venue).any(axis=1, keepdims=True).astype(np.float32)
    venue = (venue - np.array(vocab.venue_mean)) / np.array(vocab.venue_std)
    blocks += [np.nan_to_num(venue, nan=0.0).astype(np.float32), missing]

    for col, values in CATEGORIES.items():
        lookup = {v: i for i, v in enumerate(values)}
        idx = cats[col].replace_strict(lookup, default=len(values) - 1, return_dtype=pl.Int32)
        blocks.append(np.eye(len(values), dtype=np.float32)[idx.to_numpy()])

    def index(col, names):
        lookup = {n: i + 1 for i, n in enumerate(names)}
        idx = df[col].replace_strict(lookup, default=0, return_dtype=pl.Int64)
        return idx.to_numpy()

    return {
        "x": np.concatenate(blocks, axis=1),
        "batter": index("striker_id", vocab.batters),
        "bowler": index("bowler_id", vocab.bowlers),
    }


def labels(df: pl.DataFrame) -> dict[str, np.ndarray]:
    return {
        "outcome": df["outcome"].to_numpy().astype(np.int64),
        "aerial": df["aerial"].fill_null(False).to_numpy().astype(np.float32),
        "theta": df["theta"].fill_null(0.0).to_numpy().astype(np.float32),
        "dist": df["dist"].fill_null(1.0).to_numpy().astype(np.float32),
    }

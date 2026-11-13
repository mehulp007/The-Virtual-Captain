"""Per-batter histogram baselines the shot model has to beat.

Both are smoothed towards the population with pseudo-counts, so a batter with little data
falls back to the average batter. Densities are on the same scale as the shot model's
(theta in radians, dist as a fraction of the boundary), so log-likelihoods compare directly.
"""

import math

import numpy as np
import polars as pl

from virtual_captain.models import encoding as enc

N_THETA, N_DIST = 24, 15
THETA_EDGES = np.linspace(0, 2 * math.pi, N_THETA + 1)
DIST_EDGES = np.linspace(0, enc.D_MAX, N_DIST + 1)
BIN_AREA = (THETA_EDGES[1] - THETA_EDGES[0]) * (DIST_EDGES[1] - DIST_EDGES[0])


def _shot_bins(theta, dist, aerial):
    t = np.clip(np.digitize(theta, THETA_EDGES) - 1, 0, N_THETA - 1)
    d = np.clip(np.digitize(dist, DIST_EDGES) - 1, 0, N_DIST - 1)
    return (aerial.astype(int) * N_THETA + t) * N_DIST + d


class BatterHistogram:
    def __init__(self, outcome_prior: float = 50.0, shot_prior: float = 200.0):
        self.outcome_prior = outcome_prior
        self.shot_prior = shot_prior

    def fit(self, train: pl.DataFrame) -> "BatterHistogram":
        y = train["outcome"].to_numpy()
        n = len(enc.OUTCOMES)
        self.global_outcome = np.bincount(y, minlength=n) / len(y)
        self.outcome = {}
        for (batter,), g in train.group_by(["striker_id"]):
            counts = np.bincount(g["outcome"].to_numpy(), minlength=n)
            self.outcome[batter] = (counts + self.outcome_prior * self.global_outcome) / (
                counts.sum() + self.outcome_prior
            )

        shots = train.filter(pl.col("outcome") == enc.IN_PLAY)
        n_bins = 2 * N_THETA * N_DIST
        bins = _shot_bins(*(shots[c].to_numpy() for c in ("theta", "dist", "aerial")))
        # tiny floor so no bin has zero mass
        self.global_shot = (np.bincount(bins, minlength=n_bins) + 0.5) / (len(bins) + 0.5 * n_bins)
        self.shot = {}
        ids = shots["striker_id"].to_numpy()
        for batter in np.unique(ids[ids != None]):  # noqa: E711
            counts = np.bincount(bins[ids == batter], minlength=n_bins)
            self.shot[batter] = (counts + self.shot_prior * self.global_shot) / (
                counts.sum() + self.shot_prior
            )
        return self

    def outcome_log_prob(self, df: pl.DataFrame) -> np.ndarray:
        y = df["outcome"].to_numpy()
        p = np.stack([self.outcome.get(b, self.global_outcome) for b in df["striker_id"]])
        return np.log(p[np.arange(len(y)), y])

    def shot_log_prob(self, df: pl.DataFrame) -> np.ndarray:
        bins = _shot_bins(*(df[c].to_numpy() for c in ("theta", "dist", "aerial")))
        batters = df["striker_id"].to_list()
        p = np.array(
            [self.shot.get(b, self.global_shot)[i] for b, i in zip(batters, bins, strict=True)]
        )
        return np.log(p / BIN_AREA)

    def global_outcome_log_prob(self, df: pl.DataFrame) -> np.ndarray:
        return np.log(self.global_outcome[df["outcome"].to_numpy()])

    def global_shot_log_prob(self, df: pl.DataFrame) -> np.ndarray:
        bins = _shot_bins(*(df[c].to_numpy() for c in ("theta", "dist", "aerial")))
        return np.log(self.global_shot[bins] / BIN_AREA)

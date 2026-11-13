import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF, to_ground_frame
from virtual_captain.viz import style
from virtual_captain.viz.field_plot import draw_field

_BUCKETS = [
    ("1-3", (pl.col("runs_off_bat") < 4) | pl.col("non_boundary")),
    ("4", pl.col("is_four")),
    ("6", pl.col("is_six")),
]


def scoring_shots(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("has_wagon") & (pl.col("runs_off_bat") > 0))


def plot_wagon(
    ax: plt.Axes,
    shots: pl.DataFrame,
    title: str | None = None,
    ground_frame: bool = False,
) -> plt.Axes:
    """Classic wagon wheel: a line from the striker to where each scoring shot ended up.

    Readable for an innings or a handful of matches; for a career use `plot_run_rose`.
    With ground_frame=True left-handers are flipped back to the side of the ground the
    ball really went to; otherwise everyone is drawn as a right-hander.
    """
    draw_field(ax)
    origin = np.array([0.0, -PITCH_HALF])
    cap = BOUNDARY_RADIUS * 1.15
    for name, cond in _BUCKETS:
        sub = shots.filter(cond)
        if sub.is_empty():
            continue
        x = sub["x_m"].to_numpy()
        y = sub["y_m"].to_numpy()
        if ground_frame:
            x = to_ground_frame(x, sub["left_handed"].fill_null(False).to_numpy())
        r = np.hypot(x, y)
        scale = np.minimum(1.0, cap / np.maximum(r, 1e-9))
        ends = np.column_stack([x * scale, y * scale])
        segs = np.stack([np.broadcast_to(origin, ends.shape), ends], axis=1)
        ax.add_collection(LineCollection(segs, colors=style.RUN_COLOURS[name], linewidths=1.6))
    if title:
        ax.set_title(title, fontsize=9, pad=2)
    return ax


def run_legend(fig: plt.Figure, **kwargs) -> None:
    handles = [
        Line2D([0], [0], color=style.RUN_COLOURS[k], lw=2, label=label)
        for k, label in [("1-3", "1-3 runs"), ("4", "four"), ("6", "six")]
    ]
    fig.legend(handles=handles, ncol=3, fontsize=9, **kwargs)


def sector_shares(angle_deg, weights=None, sector_deg: float = 15) -> np.ndarray:
    """Share of the total (count, or weights such as runs) falling in each angle sector."""
    n = int(round(360 / sector_deg))
    idx = (np.asarray(angle_deg) // sector_deg).astype(int) % n
    totals = np.bincount(idx, weights=weights, minlength=n)
    return totals / max(totals.sum(), 1)


def run_shares(shots: pl.DataFrame, sector_deg: float = 15) -> np.ndarray:
    """Share of runs off the bat scored in each wagon-angle sector."""
    return sector_shares(
        shots["wagon_angle"].to_numpy(), shots["runs_off_bat"].to_numpy(), sector_deg
    )


def plot_rose(
    ax: plt.Axes,
    shares: np.ndarray,
    max_share: float,
    colour: str = style.SERIES[0],
    title: str | None = None,
) -> plt.Axes:
    """Rose of per-sector shares; wedge area is proportional to the share. Pass the same
    max_share to every panel so they share one scale."""
    draw_field(ax)
    sector_deg = 360 / len(shares)
    for i, share in enumerate(shares):
        r = BOUNDARY_RADIUS * 0.95 * np.sqrt(share / max_share)
        t = np.deg2rad(np.linspace(i * sector_deg, (i + 1) * sector_deg, 10))
        xs = np.concatenate([[0], -np.sin(t) * r, [0]])
        ys = np.concatenate([[0], -np.cos(t) * r, [0]])
        ax.fill(xs, ys, color=colour, ec=style.SURFACE, lw=1.0)

    top = int(np.argmax(shares))
    mid = np.deg2rad((top + 0.5) * sector_deg)
    r = BOUNDARY_RADIUS * 0.95 * np.sqrt(shares[top] / max_share) + 7
    ax.text(
        -np.sin(mid) * r,
        -np.cos(mid) * r,
        f"{shares[top]:.0%}",
        fontsize=8,
        color=style.INK_SECONDARY,
        ha="center",
        va="center",
    )
    if title:
        ax.set_title(title, fontsize=9, pad=2)
    return ax


def plot_run_rose(
    ax: plt.Axes,
    shots: pl.DataFrame,
    max_share: float,
    sector_deg: float = 15,
    colour: str = style.SERIES[0],
    title: str | None = None,
) -> np.ndarray:
    """Scoring zones: share of a batter's runs in each direction."""
    shares = run_shares(shots, sector_deg)
    plot_rose(ax, shares, max_share, colour, title)
    return shares

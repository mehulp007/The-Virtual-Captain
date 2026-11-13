"""Regenerate the figures in reports/figures from data/processed/deliveries.parquet.

Usage:
  uv run python scripts/make_figures.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from virtual_captain.models import encoding as enc  # noqa: E402
from virtual_captain.paths import DATA_DIR, FIGURES_DIR, PROCESSED_DIR  # noqa: E402
from virtual_captain.viz import style  # noqa: E402
from virtual_captain.viz.wagon import (  # noqa: E402
    plot_rose,
    plot_run_rose,
    plot_wagon,
    run_legend,
    run_shares,
    scoring_shots,
    sector_shares,
)

SHOT_MODEL = DATA_DIR / "models" / "shot_model.pt"

BATTERS = [
    "V Kohli", "JE Root", "SPD Smith", "KS Williamson", "Babar Azam",
    "RG Sharma", "DA Warner", "AB de Villiers", "JC Buttler", "Q de Kock",
    "KL Rahul", "Mohammad Rizwan", "BA Stokes", "Shubman Gill", "GJ Maxwell",
    "TM Head", "RR Pant", "SA Yadav", "H Klaasen", "YBK Jaiswal",
]  # fmt: skip

FORMAT_NAMES = {"TEST": "Test", "ODI": "ODI", "T20": "T20"}


def coverage_by_year(df: pl.LazyFrame, out: Path) -> None:
    cov = (
        df.filter(pl.col("is_legal") & (pl.col("gender") == "male"))
        .group_by("format", "year")
        .agg(pl.col("has_wagon").mean().alias("share"), pl.len().alias("balls"))
        .filter(pl.col("balls") >= 2000)
        .sort("year")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for fmt in ("TEST", "ODI", "T20"):
        d = cov.filter(pl.col("format") == fmt)
        colour = style.FORMAT_COLOURS[fmt]
        ax.plot(d["year"], d["share"] * 100, lw=2, color=colour, label=FORMAT_NAMES[fmt])
        ax.text(
            d["year"][-1] + 0.4,
            d["share"][-1] * 100,
            FORMAT_NAMES[fmt],
            color=style.INK_SECONDARY,
            fontsize=9,
            va="center",
        )
    ax.set_ylim(0, 100)
    ax.set_xlim(cov["year"].min() - 0.5, cov["year"].max() + 2.5)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel("% of legal balls")
    ax.set_title("Men's deliveries with a wagon-wheel point, by year", loc="left")
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def scoring_zone_grid(df: pl.LazyFrame, out: Path) -> None:
    shots = scoring_shots(
        df.filter(pl.col("striker").is_in(BATTERS) & (pl.col("gender") == "male")).collect()
    )
    per_batter = {name: shots.filter(pl.col("striker") == name) for name in BATTERS}
    max_share = max(run_shares(s).max() for s in per_batter.values())

    fig, axes = plt.subplots(4, 5, figsize=(12.5, 10.6))
    for ax, (name, s) in zip(axes.flat, per_batter.items(), strict=True):
        hand = "LHB" if s["left_handed"][0] else "RHB"
        runs = s["runs_off_bat"].sum()
        plot_run_rose(ax, s, max_share, title=f"{name} ({hand})  ·  {runs:,} runs")
    fig.suptitle(
        "Where the runs come from: share of runs by direction, all formats\n"
        "Wedge area is proportional to the share of runs in each 15° sector. "
        "Left-handers mirrored, off side on the right.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out)
    plt.close(fig)


def batter_by_format(df: pl.LazyFrame, name: str, out: Path) -> None:
    shots = scoring_shots(df.filter(pl.col("striker") == name).collect())
    per_format = {fmt: shots.filter(pl.col("format") == fmt) for fmt in ("TEST", "ODI", "T20")}
    max_share = max(run_shares(s).max() for s in per_format.values())

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, (fmt, s) in zip(axes, per_format.items(), strict=True):
        boundary_pct = (s["is_four"] | s["is_six"]).mean() * 100
        title = f"{FORMAT_NAMES[fmt]}  ·  {s['runs_off_bat'].sum():,} runs  ·  "
        title += f"{boundary_pct:.0f}% of scoring shots are boundaries"
        plot_run_rose(ax, s, max_share, colour=style.FORMAT_COLOURS[fmt], title=title)
    fig.suptitle(f"{name}: scoring zones by format (share of runs per 15° sector)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def single_innings(df: pl.LazyFrame, match_id: int, name: str, title: str, out: Path) -> None:
    shots = scoring_shots(
        df.filter((pl.col("match_id") == match_id) & (pl.col("striker") == name)).collect()
    )
    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    plot_wagon(ax, shots)
    ax.set_title(title, fontsize=10)
    run_legend(fig, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def held_out_shots(df: pl.LazyFrame) -> pl.DataFrame:
    return (
        df.pipe(enc.tracked_innings)
        .pipe(enc.add_labels)
        .filter(pl.col("year") >= enc.HOLDOUT_FROM_YEAR)
        .collect()
    )


def shot_model_holdout(test: pl.DataFrame, model, out: Path, n_batters: int = 6) -> None:
    """Real 2025-26 shot directions against the model's simulation of the same deliveries."""
    known = set(model.vocab.batters)
    busiest = (
        test.filter(pl.col("striker_id").is_in(known) & (pl.col("outcome") == enc.IN_PLAY))
        .group_by("striker", "striker_id")
        .agg(pl.len(), pl.col("format").n_unique().alias("formats"))
        .sort(["formats", "len"], descending=True)
        .head(n_batters)
    )
    fig, axes = plt.subplots(2, n_batters, figsize=(2.6 * n_batters, 5.9))
    for col, (name, bid, _, _) in enumerate(busiest.iter_rows()):
        rows = test.filter(pl.col("striker_id") == bid)
        real = rows.filter(pl.col("outcome") == enc.IN_PLAY)["wagon_angle"].to_numpy()
        sim = model.sample(rows, n=10, seed=col)
        sim = np.rad2deg(sim["theta"].drop_nans().to_numpy())
        real_s, sim_s = sector_shares(real), sector_shares(sim)
        top = max(real_s.max(), sim_s.max())
        title = f"{name}\n{len(real):,} balls in play"
        plot_rose(axes[0, col], real_s, top, style.SERIES[0], title)
        plot_rose(axes[1, col], sim_s, top, style.SERIES[1])
    for row, label in enumerate(("real, 2025-26", "model")):
        axes[row, 0].text(
            -95,
            0,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            color=style.INK_SECONDARY,
        )
    fig.suptitle(
        "Held-out seasons: where balls put into play went, real vs simulated "
        "(share of balls per 15° sector)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def shot_model_distance(test: pl.DataFrame, model, out: Path) -> None:
    """Distribution of where balls in play ended up, real vs simulated, by format."""
    sim = model.sample(test, n=1, seed=11)
    both = test.select("format", "outcome", "dist").with_columns(
        sim["outcome"].alias("sim_outcome"), sim["dist"].alias("sim_dist")
    )
    edges = np.linspace(0, enc.D_MAX, 61)
    mids = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, fmt in zip(axes, ("TEST", "ODI", "T20"), strict=True):
        d = both.filter(pl.col("format") == fmt)
        real = d.filter(pl.col("outcome") == enc.IN_PLAY)["dist"].to_numpy()
        fake = d.filter(pl.col("sim_outcome") == enc.IN_PLAY)["sim_dist"].to_numpy()
        series = ((real, style.SERIES[0], "real"), (fake, style.SERIES[1], "model"))
        for vals, colour, label in series:
            h, _ = np.histogram(vals, bins=edges, density=True)
            ax.plot(mids, h, color=colour, lw=2, label=label)
        ax.axvline(1.0, color=style.AXIS, lw=1, ls=(0, (3, 3)))
        ax.text(1.02, ax.get_ylim()[1] * 0.92, "boundary", fontsize=8, color=style.INK_MUTED)
        ax.set_title(FORMAT_NAMES[fmt], loc="left")
        ax.set_xlabel("distance from pitch centre (fraction of boundary)")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper right", fontsize=9)
    title = "Held-out seasons: how far balls in play travelled"
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=PROCESSED_DIR / "deliveries.parquet")
    p.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = p.parse_args(argv)

    style.apply()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pl.scan_parquet(args.data)

    coverage_by_year(df, args.out / "coverage_by_year.png")
    scoring_zone_grid(df, args.out / "scoring_zones.png")
    batter_by_format(df, "V Kohli", args.out / "kohli_by_format.png")
    single_innings(
        df,
        1298150,
        "V Kohli",
        "V Kohli 82* v Pakistan, MCG, T20 World Cup 2022",
        args.out / "kohli_mcg_2022.png",
    )
    if SHOT_MODEL.exists():
        from virtual_captain.models.shot_model import ShotModel

        model = ShotModel.load(SHOT_MODEL)
        test = held_out_shots(df)
        shot_model_holdout(test, model, args.out / "shot_model_holdout.png")
        shot_model_distance(test, model, args.out / "shot_model_distance.png")
    else:
        print(f"no shot model at {SHOT_MODEL}, skipping model figures")
    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

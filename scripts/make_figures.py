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
import polars as pl  # noqa: E402

from virtual_captain.paths import FIGURES_DIR, PROCESSED_DIR  # noqa: E402
from virtual_captain.viz import style  # noqa: E402
from virtual_captain.viz.wagon import (  # noqa: E402
    plot_run_rose,
    plot_wagon,
    run_legend,
    run_shares,
    scoring_shots,
)

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
    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

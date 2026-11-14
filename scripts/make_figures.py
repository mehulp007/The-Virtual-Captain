"""Regenerate the figures in reports/figures from data/processed/deliveries.parquet.

Usage:
  uv run python scripts/make_figures.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from virtual_captain.models import encoding as enc  # noqa: E402
from virtual_captain.paths import DATA_DIR, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from virtual_captain.viz import style  # noqa: E402
from virtual_captain.viz.field_plot import draw_field  # noqa: E402
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


def sim_calibration_plot(table: pl.DataFrame, out: Path) -> None:
    """Real vs simulated rates on held-out seasons, one point per format/phase/bowling."""
    panels = [
        ("runs_per_ball", "runs per ball"),
        ("dot", "dot balls"),
        ("four", "fours"),
        ("caught", "caught"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (col, label) in zip(axes, panels, strict=True):
        real, sim = table[f"real_{col}"].to_numpy(), table[f"sim_{col}"].to_numpy()
        top = max(real.max(), sim.max()) * 1.12
        line = np.array([0, top])
        ax.plot(line, line, color=style.AXIS, lw=1)
        for f in (0.9, 1.1):
            ax.plot(line, line * f, color=style.GRID, lw=1, ls=(0, (3, 3)))
        for fmt in ("TEST", "ODI", "T20"):
            m = (table["format"] == fmt).to_numpy()
            ax.scatter(
                real[m],
                sim[m],
                s=36,
                color=style.FORMAT_COLOURS[fmt],
                label=FORMAT_NAMES[fmt],
                edgecolor=style.SURFACE,
                linewidth=1,
                zorder=3,
            )
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_title(label, loc="left")
        ax.set_xlabel("real")
    axes[0].set_ylabel("simulated")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Simulator vs reality on held-out 2025-26 seasons (dashed lines: ±10%)",
        fontsize=11,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _draw_fielders(ax, field: np.ndarray, colour: str) -> None:
    from virtual_captain.field.positions import nearest_name

    for x, y in field[:9]:
        ax.scatter(x, y, s=70, color=colour, edgecolor=style.SURFACE, linewidth=1.5, zorder=4)
        name = nearest_name(np.array([x, y])).replace("_", " ")
        ax.text(x, y - 5.5, name, fontsize=6.5, ha="center", va="top", color=style.INK_SECONDARY)
    for x, y in field[9:]:
        ax.scatter(x, y, s=40, color=style.INK_MUTED, edgecolor=style.SURFACE, zorder=4)


def field_example(df: pl.LazyFrame, out: Path, batter: str = "H Klaasen") -> None:
    """The conventional T20 death field next to the one adapted to a batter's deep hitting."""
    from virtual_captain.sim.calibrate import PROFILE_SECTORS, adapted_field, deep_profiles

    rows = (
        df.filter((pl.col("striker") == batter) & (pl.col("format") == "T20"))
        .pipe(enc.add_labels)
        .collect()
    )
    weights = next(iter(deep_profiles(rows).values()))
    key = ("T20", "death", "pace")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
    titles = ["Conventional T20 death field", f"Boundary riders moved to where {batter} hits"]
    for ax, w, title in zip(axes, [None, weights], titles, strict=True):
        shares = weights / weights.sum()
        plot_rose(ax, shares, shares.max() * 1.6, colour=style.GRID)
        _draw_fielders(ax, adapted_field(key, w), style.SERIES[0] if w is None else style.SERIES[1])
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"Grey wedges: where {batter}'s balls reach the deep in T20s "
        f"({PROFILE_SECTORS} sectors, right-hander's view)",
        fontsize=9,
        color=style.INK_SECONDARY,
        y=0.04,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)


SHOWCASE = [
    "SA Yadav | T20 middle | spin",
    "V Kohli | ODI middle | pace",
    "JE Root | TEST new_ball | pace",
]


def _keeper_and_bowler(field9: np.ndarray, bowling: str) -> np.ndarray:
    from virtual_captain.field.positions import with_keeper_and_bowler

    return with_keeper_and_bowler(field9, keeper_up=bowling == "spin")


def optimised_fields(results: pl.DataFrame, fields: list[dict], out: Path) -> None:
    """Conventional, analyst and optimised fields for a few batters, with what each costs."""
    by_name = {f["scenario"]: f for f in fields}
    chosen = [s for s in SHOWCASE if s in by_name] or [
        f["scenario"] for f in fields if not f["scenario"].startswith("all")
    ][:3]
    kinds = [("conventional", style.SERIES[0]), ("analyst", style.SERIES[3]),
             ("optimised", style.SERIES[1])]  # fmt: skip
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11.5, 3.9 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, scenario in enumerate(chosen):
        r = results.filter(pl.col("scenario") == scenario).row(0, named=True)
        for col, (kind, colour) in enumerate(kinds):
            ax = axes[row, col]
            draw_field(ax)
            f9 = np.array([[p["x"], p["y"]] for p in by_name[scenario][kind]])
            _draw_fielders(ax, _keeper_and_bowler(f9, r["bowling"]), colour)
            title = f"{kind}: {r[f'{kind}_runs_per_ball']:.2f} runs, "
            title += f"{r[f'{kind}_wickets_per_ball'] * 100:.2f}% wkts per ball"
            ax.set_title(title, fontsize=9)
        axes[row, 0].text(
            -95, 0, scenario.replace(" | ", "\n"), rotation=90, ha="center", va="center",
            fontsize=9, color=style.INK_SECONDARY,
        )  # fmt: skip
    fig.suptitle(
        "Fields for three batters, judged on simulated deliveries not used in the search",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def value_map_figure(data: dict, out: Path) -> None:
    """Where one more fielder saves the most, given the rest of the optimised field."""
    from matplotlib.colors import LinearSegmentedColormap

    coords = np.array(data["coords"])
    grid = np.array([[np.nan if v is None else v for v in r] for r in data["saved"]], dtype=float)
    ramp = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    lo, hi = coords[0], coords[-1]
    img = ax.imshow(
        np.clip(grid, 0, None), origin="lower", extent=(lo, hi, lo, hi), cmap=ramp, zorder=0
    )
    draw_field(ax)
    f8 = np.array([[p["x"], p["y"]] for p in data["field"]])
    for x, y in f8:
        ax.scatter(x, y, s=60, color=style.SERIES[1], edgecolor=style.SURFACE, lw=1.5, zorder=4)
    cbar = fig.colorbar(img, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cost saved per ball (runs, net of wickets)", color=style.INK_SECONDARY)
    ax.set_title(
        f"{data['scenario']}: where the ninth fielder is worth most\n"
        f"(eight fielders fixed; {data['dropped'].replace('_', ' ')} taken out)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def optimiser_gains(results: pl.DataFrame, out: Path) -> None:
    """Cost saved per ball by the optimised field over the analyst's, per scenario."""
    r = (
        results.filter(pl.col("batter") != "all batters")
        .with_columns(
            (-pl.col("optimised_vs_analyst")).alias("gain"),
            (-pl.col("optimised_vs_analyst_hi")).alias("gain_lo"),
            (-pl.col("optimised_vs_analyst_lo")).alias("gain_hi"),
        )
        .sort("gain")
    )
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * r.height + 1.4))
    y = np.arange(r.height)
    for fmt in ("TEST", "ODI", "T20"):
        m = (r["format"] == fmt).to_numpy()
        ax.hlines(y[m], r["gain_lo"].to_numpy()[m], r["gain_hi"].to_numpy()[m],
                  color=style.FORMAT_COLOURS[fmt], lw=2)  # fmt: skip
        ax.scatter(r["gain"].to_numpy()[m], y[m], color=style.FORMAT_COLOURS[fmt], s=30,
                   label=FORMAT_NAMES[fmt], zorder=3)  # fmt: skip
    ax.axvline(0, color=style.AXIS, lw=1)
    labels = [f"{b} · {f} {p} · {bw}" for b, f, p, bw in
              r.select("batter", "format", "phase", "bowling").iter_rows()]  # fmt: skip
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("cost saved per ball vs the analyst's field (runs, net of wickets)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Optimised vs analyst field, with 95% intervals", loc="left")
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

    calibration = REPORTS_DIR / "sim_calibration.csv"
    if calibration.exists():
        sim_calibration_plot(pl.read_csv(calibration), args.out / "sim_calibration.png")
    field_example(df, args.out / "field_example.png")

    results_csv = REPORTS_DIR / "optimizer_results.csv"
    if results_csv.exists():
        results = pl.read_csv(results_csv)
        fields = json.loads((REPORTS_DIR / "optimized_fields.json").read_text())
        optimised_fields(results, fields, args.out / "optimised_fields.png")
        optimiser_gains(results, args.out / "optimiser_gains.png")
    vmap = REPORTS_DIR / "value_map.json"
    if vmap.exists():
        value_map_figure(json.loads(vmap.read_text()), args.out / "value_map.png")
    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

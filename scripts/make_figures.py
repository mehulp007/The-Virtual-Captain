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
from virtual_captain.paths import (  # noqa: E402
    DATA_DIR,
    FIGURES_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    ROOT,
)
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


# The RL captain ---------------------------------------------------------------------------
CAPTAIN_SHOWCASE = "SA Yadav | T20 middle | spin"


def _field9(points: list[dict]) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for p in points])


def captain_learning(history: pl.DataFrame, out: Path, search_level: float | None = None) -> None:
    """How the captain's fields compare with its starting field as training goes on."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    steps = history["steps"].to_numpy() / 1e6
    lines = [
        ("probe_train_vs_all_batters_field", "150 batters it trains on", style.SERIES[0]),
        ("probe_unseen_vs_all_batters_field", "150 batters it never sees", style.SERIES[1]),
    ]
    for col, label, colour in lines:
        if col in history.columns:
            ax.plot(steps, history[col].to_numpy(), lw=2, color=colour, label=label)
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.text(
        steps.max(), 0, "field optimised for all batters (the start) ", va="bottom", ha="right",
        fontsize=8,
        color=style.INK_SECONDARY,
    )  # fmt: skip
    if search_level is not None:
        ax.axhline(search_level, color=style.INK_SECONDARY, lw=1, ls="--")
        ax.text(
            0, search_level, " a CMA-ES search run for each batter, on average", va="top",
            fontsize=8, color=style.INK_SECONDARY,
        )  # fmt: skip
    ax.set_xlabel("training steps (millions of balls)")
    ax.set_ylabel("cost per ball vs the start (runs)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(
        "The captain's fields as it trains (below zero: better than where it started)",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def captain_fields_figure(fields: dict, static: pl.DataFrame, out: Path) -> None:
    """For three batters: the all-batters field, a dedicated search and the captain's field."""
    kinds = [
        ("all_batters", "all-batters field", style.SERIES[0]),
        ("searched", "dedicated search", style.SERIES[3]),
        ("captain", "RL captain", style.SERIES[1]),
    ]
    chosen = list(fields)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11.5, 3.9 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, scenario in enumerate(chosen):
        who, situation, bowling = scenario.split(" | ")
        fmt, phase = situation.split()
        r = static.filter(
            (pl.col("batter") == who)
            & (pl.col("format") == fmt)
            & (pl.col("phase") == phase)
            & (pl.col("bowling") == bowling)
        ).row(0, named=True)
        for col, (kind, label, colour) in enumerate(kinds):
            ax = axes[row, col]
            draw_field(ax)
            f9 = _field9(fields[scenario][kind])
            _draw_fielders(ax, _keeper_and_bowler(f9, bowling), colour)
            runs = r[f"{kind}_runs_per_ball"]
            wkts = r[f"{kind}_wickets_per_ball"] * 100
            ax.set_title(f"{label}: {runs:.2f} runs, {wkts:.2f}% wkts per ball", fontsize=9)
        axes[row, 0].text(
            -95, 0, scenario.replace(" | ", "\n"), rotation=90, ha="center", va="center",
            fontsize=9, color=style.INK_SECONDARY,
        )  # fmt: skip
    fig.suptitle(
        "Fields for three batters, judged on simulated deliveries nothing trained on", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def captain_over_figure(over: dict, out: Path) -> None:
    """One T20 death over: the captain's field for every ball, as the strike changes."""
    balls = over["balls"]
    names = sorted({b["striker"] for b in balls})
    colours = {n: style.SERIES[i] for i, n in enumerate(names)}
    cols = 3
    rows = int(np.ceil(len(balls) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 4.0 * rows))
    for ax, b in zip(axes.ravel(), balls, strict=False):
        draw_field(ax, labels=False)
        _draw_fielders(ax, _keeper_and_bowler(_field9(b["field"]), "pace"), colours[b["striker"]])
        result = "wicket" if b["wicket"] else f"{b['runs']} run{'s' if b['runs'] != 1 else ''}"
        ax.set_title(
            f"ball {b['ball']}: {b['striker']} on strike\n"
            f"wicket worth {b['wicket_value']:.1f} runs, result: {result}",
            fontsize=9,
        )
    for ax in axes.ravel()[len(balls) :]:
        ax.axis("off")
    fig.suptitle(
        "One T20 death over, field reset for every ball (colour: batter on strike)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def captain_vs_search(static: pl.DataFrame, out: Path) -> None:
    """Every batter context: the captain's cost per ball minus the dedicated search's."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, fmt in zip(axes, ("T20", "ODI", "TEST"), strict=True):
        for i, unseen in enumerate((False, True)):
            d = static.filter((pl.col("format") == fmt) & (pl.col("unseen") == unseen))
            y = d["captain_vs_searched"].to_numpy()
            x = i + rng.uniform(-0.25, 0.25, len(y))
            ax.scatter(x, y, s=9, alpha=0.6, color=style.SERIES[i], lw=0)
            if len(y):
                ax.hlines(np.mean(y), i - 0.32, i + 0.32, color=style.INK, lw=2)
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.set_xticks([0, 1], ["trained on", "never seen"])
        ax.set_title(FORMAT_NAMES[fmt], loc="left")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("captain minus search, cost per ball (runs)")
    fig.suptitle(
        "The captain against a dedicated search for every batter context "
        "(below zero: the captain is better; bars are means)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def captain_training_gif(checkpoints: Path, out: Path, scenario: str = CAPTAIN_SHOWCASE) -> None:
    """The captain's field for one batter at every training checkpoint."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from stable_baselines3 import PPO

    from virtual_captain.rl.data import CaptainData
    from virtual_captain.rl.evaluate import captain_fields, judge_fields

    paths = sorted(checkpoints.glob("step_*.zip"))
    if not paths:
        print(f"no checkpoints in {checkpoints}, skipping the training animation")
        return
    data = CaptainData.load()
    who, situation, bowling = scenario.split(" | ")
    fmt, phase = situation.split()
    c = np.array([data.context_of(who, fmt, phase, bowling)])
    search = judge_fields(data, c, data.searched_fields[c])
    frames = []
    for path in paths:
        model = PPO.load(path, device="cpu")
        f = captain_fields(model, data, c)
        judged = judge_fields(data, c, f)
        gap = float((judged.cost - search.cost).mean())
        frames.append((int(path.stem.split("_")[1]), f[0], float(judged.runs.mean()), gap))
    frames += [frames[-1]] * 6  # hold the last frame

    fig, ax = plt.subplots(figsize=(5.6, 6.0))

    def draw(i):
        ax.clear()
        steps, field, runs, gap = frames[i]
        draw_field(ax)
        _draw_fielders(ax, _keeper_and_bowler(field, bowling), style.SERIES[1])
        ax.set_title(
            f"{who}, {fmt} {phase} overs, {bowling}\n"
            f"after {steps / 1e6:.1f}M balls: {runs:.3f} runs per ball, {gap:+.3f} vs search",
            fontsize=9,
        )

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(out, writer=PillowWriter(fps=3))
    plt.close(fig)


# Self-play -------------------------------------------------------------------------------
def selfplay_exploitability(table: pl.DataFrame, out: Path) -> None:
    """What a best-response batter gains against each captain round, by format, with the two
    fixed CMA-ES fields for reference."""
    t = table.filter(pl.col("batters") == "train")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, fmt in zip(axes, ("T20", "ODI", "TEST"), strict=True):
        d = t.filter(pl.col("format") == fmt)
        cap = d.filter(pl.col("round").is_not_null()).sort("round")
        x = cap["round"].to_numpy()
        ax.errorbar(
            x, cap["gain"].to_numpy(), yerr=cap["gain_ci"].to_numpy(), color=style.SERIES[1],
            lw=2, marker="o", ms=5, capsize=0, label="RL captain",
        )  # fmt: skip
        for name, ls in (("all-batters field", "--"), ("searched field", ":")):
            row = d.filter(pl.col("setter") == name)
            if row.height:
                ax.axhline(row["gain"][0], color=style.INK_SECONDARY, lw=1.2, ls=ls, label=name)
        ax.set_xticks(x, ["start" if r == 0 else str(r) for r in x])
        ax.set_xlabel("self-play round")
        ax.set_title(FORMAT_NAMES[fmt], loc="left")
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("best-response batter's gain\n(cost per ball, runs)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(
        "How much a batter who knows the field can gain against it (lower: harder to exploit)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _aim_wedges(ax, natural: list[float], batter: list[float], colour: str) -> None:
    """Bars outside the rope: how much more of the batter's ground shots go into each 30-degree
    sector against this field."""
    from virtual_captain.data.coords import wagon_to_xy

    gain = np.array(batter) - np.array(natural)
    scale = 160.0  # metres of bar per unit of share
    for i, g in enumerate(gain):
        if g <= 0.002:
            continue
        a = np.linspace(30 * i + 2, 30 * i + 28, 12)
        r0, r1 = 70.0, 70.0 + scale * g
        x0, y0 = wagon_to_xy(a, r0 / 68.0)
        x1, y1 = wagon_to_xy(a[::-1], r1 / 68.0)
        ax.fill(np.r_[x0, x1], np.r_[y0, y1], color=colour, alpha=0.8, lw=0)


def selfplay_fields_figure(fields: dict, out: Path) -> None:
    """For three batters: the field optimised for all batters, the captain before and after
    self-play, and where each field's best-response batter aims its extra shots."""
    kinds = [
        ("all_batters", "all-batters field", style.SERIES[0]),
        ("before", "captain before self-play", style.SERIES[3]),
        ("after", "captain after self-play", style.SERIES[1]),
    ]
    chosen = list(fields)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11.5, 4.1 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, scenario in enumerate(chosen):
        bowling = scenario.split(" | ")[2]
        for col, (kind, label, colour) in enumerate(kinds):
            ax = axes[row, col]
            e = fields[scenario][kind]
            draw_field(ax)
            _aim_wedges(ax, e["aim"]["sector_natural"], e["aim"]["sector_batter"], style.SERIES[7])
            _draw_fielders(ax, _keeper_and_bowler(_field9(e["field"]), bowling), colour)
            lim = 92
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_title(
                f"{label}\n{e['cost_natural']:.3f} a ball natural, "
                f"{e['cost_best_response']:.3f} against its best response",
                fontsize=9,
            )
        axes[row, 0].text(
            -112, 0, scenario.replace(" | ", "\n"), rotation=90, ha="center", va="center",
            fontsize=9, color=style.INK_SECONDARY,
        )  # fmt: skip
    fig.suptitle(
        "Fields before and after self-play. Red bars: where the batter who knows each field "
        "sends more of its shots",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def selfplay_matrix_figure(matrix: pl.DataFrame, out: Path) -> None:
    """Every field-setter against every batter's best response, by format: runs per ball the
    batter gains over its natural game. Each format has its own colour range."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("gain", [style.SURFACE, style.SERIES[0]])
    d = matrix.filter(pl.col("batter") != "natural")
    setters = list(dict.fromkeys(d["setter"].to_list()))
    batters = list(dict.fromkeys(d["batter"].to_list()))
    short = [b.replace("best response to ", "") for b in batters]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.6 * len(setters) + 2.4))
    for k, (ax, fmt) in enumerate(zip(axes, ("T20", "ODI", "TEST"), strict=True)):
        f = d.filter(pl.col("format") == fmt)
        grid = np.array(
            [[f.filter((pl.col("setter") == s) & (pl.col("batter") == b))["gain"][0]
              for b in batters] for s in setters]
        )  # fmt: skip
        lo, hi = grid.min(), grid.max()
        ax.imshow(grid, cmap=cmap, vmin=lo - 0.15 * (hi - lo), vmax=hi, aspect="auto")
        for i in range(len(setters)):
            for j in range(len(batters)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color=style.INK, fontweight="bold" if i == j else "normal")  # fmt: skip
        ax.set_xticks(range(len(batters)), short, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(setters)), setters if k == 0 else [], fontsize=8)
        ax.set_xlabel("batter trained to beat ...")
        ax.set_title(FORMAT_NAMES[fmt], loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("field set by ...")
    fig.suptitle(
        "Runs per ball a best-response batter gains over its natural game "
        "(bold: against the field it was trained on)",
        fontsize=10,
    )
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
    captain_csv = REPORTS_DIR / "captain_static.csv"
    if captain_csv.exists():
        static = pl.read_csv(captain_csv)
        captain = json.loads((REPORTS_DIR / "captain_fields.json").read_text())
        history = pl.read_csv(REPORTS_DIR / "captain_training.csv")
        search_level = float((static["searched_cost"] - static["all_batters_cost"]).mean())
        captain_learning(history, args.out / "captain_learning.png", search_level)
        captain_fields_figure(captain["showcase"], static, args.out / "captain_fields.png")
        if captain["over"]:
            captain_over_figure(captain["over"], args.out / "captain_over.png")
        captain_vs_search(static, args.out / "captain_vs_search.png")
        captain_training_gif(ROOT / "checkpoints" / "captain", args.out / "captain_training.gif")
    exploit_csv = REPORTS_DIR / "selfplay_exploitability.csv"
    if exploit_csv.exists():
        selfplay_exploitability(pl.read_csv(exploit_csv), args.out / "selfplay_exploitability.png")
    if (REPORTS_DIR / "selfplay_fields.json").exists():
        fields = json.loads((REPORTS_DIR / "selfplay_fields.json").read_text())
        selfplay_fields_figure(fields, args.out / "selfplay_fields.png")
        matrix = pl.read_csv(REPORTS_DIR / "selfplay_matrix.csv")
        selfplay_matrix_figure(matrix, args.out / "selfplay_matrix.png")
    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

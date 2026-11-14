"""Plotly figures for the app: a field (optionally editable, with a value map under it) and
wagon wheels. Same colours as the repo's figures.

Fields are in the mirrored right-hander frame the models use (+x is the batter's off side).
`mirror=-1` draws them for a left-hander, off side on the left.
"""

import numpy as np
import plotly.graph_objects as go

from virtual_captain.data.coords import wagon_to_xy
from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF, in_ring, ring_outline
from virtual_captain.field.positions import BOWLER, KEEPER_BACK, KEEPER_UP, nearest_name, xy
from virtual_captain.viz import style

LIMIT = BOUNDARY_RADIUS + 6
CLICK_SPACING = 3.0  # metres between the invisible points you click to move a fielder
BETTER, WORSE, MID = style.SERIES[0], style.SERIES[1], "#ecebe6"


def _label(p: np.ndarray) -> str:
    return nearest_name(p).replace("_", " ")


def _ground(mirror: int) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(
        type="circle", x0=-BOUNDARY_RADIUS, y0=-BOUNDARY_RADIUS, x1=BOUNDARY_RADIUS,
        y1=BOUNDARY_RADIUS, line={"color": style.AXIS, "width": 1.5}, layer="below",
    )  # fmt: skip
    fig.add_shape(
        type="rect", x0=-1.5, x1=1.5, y0=-PITCH_HALF, y1=PITCH_HALF,
        fillcolor=style.GRID, line={"width": 0}, layer="below",
    )  # fmt: skip
    rx, ry = ring_outline()
    fig.add_trace(
        go.Scatter(
            x=mirror * rx, y=ry, mode="lines", hoverinfo="skip", showlegend=False,
            line={"color": style.AXIS, "width": 1, "dash": "dash"},
        )
    )  # fmt: skip
    for text, x in (("off", 0.8), ("leg", -0.8)):
        fig.add_annotation(
            x=mirror * x * BOUNDARY_RADIUS, y=-0.97 * BOUNDARY_RADIUS, text=text,
            showarrow=False, font={"size": 11, "color": style.INK_MUTED},
        )  # fmt: skip
    fig.update_xaxes(range=[-LIMIT, LIMIT], visible=False, fixedrange=True)
    fig.update_yaxes(
        range=[-LIMIT, LIMIT], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True
    )
    fig.update_layout(
        height=420, margin={"l": 4, "r": 4, "t": 36, "b": 4}, paper_bgcolor=style.SURFACE,
        plot_bgcolor=style.SURFACE, font={"family": "Segoe UI, DejaVu Sans, sans-serif",
        "color": style.INK}, showlegend=False, dragmode=False, hovermode="closest",
        title={"font": {"size": 14}, "x": 0.02, "y": 0.98},
    )  # fmt: skip
    return fig


def _value_layer(fig: go.Figure, points, change, legal, mirror: int, spacing: float) -> None:
    """The value map: what moving the selected fielder to each point does to the cost per
    ball. Blue saves runs, orange costs them; points that would break the rules are left out."""
    axis = np.arange(-BOUNDARY_RADIUS, BOUNDARY_RADIUS + 1e-9, spacing)
    z = np.full((len(axis), len(axis)), np.nan)
    ix = np.rint((points[:, 0] + BOUNDARY_RADIUS) / spacing).astype(int)
    iy = np.rint((points[:, 1] + BOUNDARY_RADIUS) / spacing).astype(int)
    z[iy[legal], ix[legal]] = change[legal]
    if mirror < 0:
        z = z[:, ::-1]  # x runs the other way for a left-hander
    span = float(np.nanmax(np.abs(change[legal]))) if legal.any() else 1.0
    fig.add_trace(
        go.Contour(
            x=axis, y=axis, z=z, zmid=0, zmin=-span,
            zmax=span, colorscale=[[0, BETTER], [0.5, MID], [1, WORSE]], opacity=0.8,
            contours={"coloring": "heatmap", "showlines": False}, line={"width": 0},
            connectgaps=False, showscale=False,
            hovertemplate="%{z:+.3f} runs a ball<extra></extra>",
        )
    )  # fmt: skip
    # mask whatever spills past the rope
    r, big = BOUNDARY_RADIUS + 0.5, LIMIT + 5
    t = np.linspace(0, 2 * np.pi, 145)[:-1]  # shape paths take no arcs: a 144-gon will do
    circle = "M " + " L ".join(f"{r * np.cos(a):.2f},{r * np.sin(a):.2f}" for a in t) + " Z"
    fig.add_shape(
        type="path", path=f"M {-big},{-big} H {big} V {big} H {-big} Z {circle}",
        fillcolor=style.SURFACE, line={"width": 0}, layer="above", fillrule="evenodd",
    )  # fmt: skip
    fig.add_shape(
        type="circle", x0=-BOUNDARY_RADIUS, y0=-BOUNDARY_RADIUS, x1=BOUNDARY_RADIUS,
        y1=BOUNDARY_RADIUS, line={"color": style.AXIS, "width": 1.5}, layer="above",
    )  # fmt: skip


def field_figure(
    field: np.ndarray,
    keeper_up: bool,
    title: str = "",
    colour: str = style.SERIES[0],
    mirror: int = 1,
    selected: int | None = None,
    editable: bool = False,
    value_map: tuple | None = None,
    spacing: float = 5.0,
) -> go.Figure:
    """A field of nine. With `editable`, every fielder and an invisible grid of points can be
    clicked; the app tells them apart by where the click landed."""
    fig = _ground(mirror)
    if value_map is not None:
        _value_layer(fig, *value_map, mirror=mirror, spacing=spacing)
    if editable:
        axis = np.arange(-BOUNDARY_RADIUS + 1, BOUNDARY_RADIUS, CLICK_SPACING)
        gx, gy = np.meshgrid(axis, axis)
        keep = np.hypot(gx, gy) <= BOUNDARY_RADIUS - 1
        fig.add_trace(
            go.Scatter(
                x=mirror * gx[keep], y=gy[keep], mode="markers",
                marker={"size": 9, "color": style.SURFACE, "opacity": 0}, hoverinfo="none",
            )
        )  # fmt: skip
    extra = np.stack([xy(KEEPER_UP if keeper_up else KEEPER_BACK), xy(BOWLER)])
    fig.add_trace(
        go.Scatter(
            x=mirror * extra[:, 0], y=extra[:, 1], mode="markers", hoverinfo="skip",
            marker={"size": 8, "color": style.INK_MUTED},
        )
    )  # fmt: skip
    sizes = np.full(9, 15.0)
    lines = np.full(9, 1.5)
    if selected is not None:
        sizes[selected], lines[selected] = 21.0, 4.0
    deep = ~in_ring(field[:, 0], field[:, 1])
    fig.add_trace(
        go.Scatter(
            x=mirror * field[:, 0], y=field[:, 1], mode="markers",
            selected={"marker": {"opacity": 1}}, unselected={"marker": {"opacity": 1}},
            marker={"size": sizes, "color": colour, "line": {"width": lines, "color": style.INK
                    if selected is not None else style.SURFACE}},
            hovertemplate=[
                f"{_label(p)}<br>{np.hypot(*p):.0f} m from the pitch centre, "
                f"{'outside' if d else 'inside'} the ring<extra></extra>"
                for p, d in zip(field, deep, strict=True)
            ],
        )
    )  # fmt: skip
    for p in field:  # names as annotations, so they sit above the value map's edge
        fig.add_annotation(
            x=mirror * p[0], y=p[1], yshift=-14, text=_label(p), showarrow=False,
            font={"size": 10, "color": style.INK_SECONDARY},
        )  # fmt: skip
    fig.update_layout(title={"text": title})
    return fig


RUN_GROUPS = [
    ("out (caught)", lambda r: r < 0, style.INK, "x", 6),
    ("no run", lambda r: r == 0, style.AXIS, "circle", 4),
    ("1 to 3", lambda r: (r >= 1) & (r <= 3), style.RUN_COLOURS["1-3"], "circle", 5),
    ("four", lambda r: r >= 4, style.RUN_COLOURS["4"], "circle", 6),
]


def wagon_figure(theta, dist, runs, title: str = "", mirror: int = 1) -> go.Figure:
    """Where shots that stayed in the ground ended up, coloured by what they brought."""
    fig = _ground(mirror)
    x, y = wagon_to_xy(np.degrees(theta), np.minimum(dist, 1.0))
    for name, pick, colour, symbol, size in RUN_GROUPS:
        m = pick(runs)
        fig.add_trace(
            go.Scatter(
                x=mirror * x[m], y=y[m], mode="markers", name=f"{name} ({m.mean():.0%})",
                marker={"size": size, "color": colour, "symbol": symbol, "opacity": 0.7},
                hoverinfo="skip",
            )
        )  # fmt: skip
    fig.update_layout(
        title={"text": title}, showlegend=True, height=520,
        legend={"orientation": "h", "y": -0.02, "x": 0.5, "xanchor": "center",
                "font": {"size": 11}},
    )  # fmt: skip
    return fig

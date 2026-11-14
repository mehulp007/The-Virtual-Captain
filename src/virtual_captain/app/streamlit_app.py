"""The Virtual Captain, live.

Pick a batter, a type of bowling and a match situation; see the field the RL captain sets next
to the usual ones and what each costs; move the fielders yourself and watch the numbers move.

  uv run streamlit run src/virtual_captain/app/streamlit_app.py

Reads data/app/bundle.npz (scripts/build_app_bundle.py), or the file named by VC_BUNDLE.
"""

import os
from pathlib import Path

import numpy as np
import streamlit as st

from virtual_captain.app import plots
from virtual_captain.app.bundle import BUNDLE, Bundle
from virtual_captain.app.engine import (
    ALL_BATTERS,
    CAPTAIN,
    CONVENTIONAL,
    PHASE_OVERS,
    SEARCHED,
    Engine,
    Situation,
)
from virtual_captain.data.coords import wagon_to_xy, xy_to_wagon
from virtual_captain.field.geometry import BOUNDARY_RADIUS
from virtual_captain.field.positions import nearest_name
from virtual_captain.viz import style

REPO = "https://github.com/mehulp007/The-Virtual-Captain"
FORMAT_NAMES = {"T20": "T20", "ODI": "ODI", "TEST": "Test"}
PHASES = {
    "T20": ["powerplay", "middle", "death"],
    "ODI": ["powerplay", "middle", "death"],
    "TEST": ["new_ball", "old_ball"],
}
PHASE_NAMES = {
    "powerplay": "Powerplay",
    "middle": "Middle overs",
    "death": "Death overs",
    "new_ball": "New ball",
    "old_ball": "Old ball",
}
FAMOUS = ["SA Yadav", "V Kohli", "JE Root", "JC Buttler", "SPD Smith", "KS Williamson",
          "RG Sharma", "Babar Azam", "Q de Kock", "DA Warner"]  # fmt: skip
COLOURS = {
    CAPTAIN: style.SERIES[1],
    CONVENTIONAL: style.INK_SECONDARY,
    ALL_BATTERS: style.SERIES[0],
    SEARCHED: style.SERIES[2],
}
EDITED = "Your field"
CHART_CONFIG = {"displayModeBar": False, "scrollZoom": False}
PICK_UP = 4.0  # metres: a click this close to a fielder picks them up

st.set_page_config(page_title="The Virtual Captain", page_icon="🏏", layout="wide")


@st.cache_resource(show_spinner="Loading the simulator...")
def engine() -> Engine:
    e = Engine(Bundle.load(Path(os.environ.get("VC_BUNDLE", BUNDLE))))
    s = e.typical(0)
    e.score(s, e.fields(s))  # compile the simulator before anyone waits on it
    return e


@st.cache_data(max_entries=256, show_spinner=False)
def scored(s: Situation, extra: tuple | None = None):
    """Fields for the situation and their scores. `extra` adds one more field (as a tuple of
    coordinates) scored on the same deliveries."""
    e = engine()
    fields = e.fields(s)
    if extra is not None:
        fields[EDITED] = np.array(extra).reshape(9, 2)
    sc = e.score(s, fields)
    summary = {k: v.summary() for k, v in sc.items()}
    vs = {k: v.minus(sc[CAPTAIN]) for k, v in sc.items()}
    return fields, summary, vs


@st.cache_data(max_entries=64, show_spinner=False)
def value_map(s: Situation, field: tuple, i: int):
    return engine().value_map(s, np.array(field).reshape(9, 2), i)


def batter_label(e: Engine, c: int) -> str:
    if e.b.pooled[c]:
        return "All batters"
    name = str(e.b.batter[c])
    return f"{name} (not seen in training)" if e.b.unseen[c] else name


def every(rate: float) -> str:
    return f"1 in {1 / rate:.0f} balls" if rate > 0 else "none"


def table(summary: dict, vs: dict, names: list[str]) -> list[dict]:
    rows = []
    for k in names:
        m = summary[k]
        d, ci = vs[k]
        rows.append(
            {
                "Field": k,
                "Runs an over": round(6 * m["runs_per_ball"], 2),
                "Wicket": every(m["wickets_per_ball"]),
                "Fours": f"{m['fours']:.1%}",
                "Sixes": f"{m['sixes']:.1%}",
                "Dot balls": f"{m['dots']:.0%}",
                "Cost a ball": round(m["cost_per_ball"], 3),
                "vs RL captain": "" if k == CAPTAIN else f"{d:+.3f} ± {ci:.3f}",
            }
        )
    return rows


# Sidebar: the situation ----------------------------------------------------------------
e = engine()
with st.sidebar:
    st.header("The situation")
    fmt = st.radio("Format", list(FORMAT_NAMES), format_func=FORMAT_NAMES.get, horizontal=True)
    bowling = st.radio("Bowling", ["pace", "spin"], format_func=str.title, horizontal=True)
    phase = st.selectbox("Phase", PHASES[fmt], format_func=PHASE_NAMES.get)
    cands = e.contexts_for(fmt, phase, bowling)
    cands = sorted(cands, key=lambda c: (not e.b.pooled[c], str(e.b.batter[c]).lower()))
    names = [str(e.b.batter[c]) for c in cands]
    first = next((names.index(n) for n in FAMOUS if n in names), 0)
    c = st.selectbox(
        f"Batter ({len(cands) - 1} with enough tracked balls)",
        cands,
        index=first,
        format_func=lambda c: batter_label(e, c),
    )
    typ = e.typical(c)
    lo, hi = PHASE_OVERS[(fmt, phase)]
    wickets = st.slider("Wickets down", 0, 9, typ.wickets, key=f"wickets-{c}")
    over_label = "Overs since the new ball" if fmt == "TEST" else "Over"
    over = st.slider(over_label, lo, hi, typ.over, key=f"over-{c}")
    s = Situation(int(c), int(wickets), int(over))
    st.caption(
        f"A wicket is worth **{e.wicket_value(s):.0f} runs** here: what the batting side's "
        "total drops by, on average, when one falls in this state. Every field is judged on "
        "runs minus wickets times that."
    )
    st.caption(
        f"{e.b.balls[c]:,} tracked balls before 2025 behind this batter"
        + (", drawn for a left-hander" if e.b.left_handed[c] else "")
        + "."
    )

mirror = -1 if e.b.left_handed[c] else 1
keeper_up = e.keeper_up(c)
who = batter_label(e, c)
st.title("The Virtual Captain")
st.markdown(
    f"**{who}**, {FORMAT_NAMES[fmt]} {PHASE_NAMES[phase].lower()}, against {bowling}. "
    f"Fields set from 5.85 million deliveries and a calibrated fielding simulator. "
    f"[How it works]({REPO})"
)

fields, summary, vs = scored(s)
tab_rec, tab_edit, tab_wagon, tab_about = st.tabs(
    ["Recommended field", "Move the fielders", "Wagon wheel", "About"]
)

# Recommended field ---------------------------------------------------------------------
with tab_rec:
    others = [k for k in fields if k != CAPTAIN]
    other = st.selectbox("Compare the RL captain's field with", others)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plots.field_figure(fields[CAPTAIN], keeper_up, "RL captain", COLOURS[CAPTAIN], mirror),
            config=CHART_CONFIG, key="rec-captain",
        )  # fmt: skip
    with right:
        st.plotly_chart(
            plots.field_figure(fields[other], keeper_up, other, COLOURS[other], mirror),
            config=CHART_CONFIG, key="rec-other",
        )  # fmt: skip
    st.dataframe(table(summary, vs, list(fields)), hide_index=True, width="stretch")
    st.caption(
        "Each field faces the same 8,192 simulated deliveries from this batter's shot model, "
        "each played six times with fresh fielding luck. *Cost a ball* is runs minus wickets "
        "times their value; lower is better. *vs RL captain* is the difference in cost a ball, "
        "with a 95% interval: below zero, that field does better than the captain here."
    )

# Move the fielders -----------------------------------------------------------------------
with tab_edit:
    ss = st.session_state
    ss.setdefault("clicks", 0)
    top = st.columns([2, 1, 1])
    start = top[0].selectbox("Start from", list(fields), key="edit-start")
    signature = (s, start)
    if top[1].button("Reset", width="stretch") or ss.get("edit-signature") != signature:
        ss["edit-field"] = np.array(fields[start], dtype=float)
        ss["edit-signature"] = signature
        ss["selected"] = None
        ss.clicks += 1
    show_map = top[2].toggle("Value map", value=True, help="Where the selected fielder saves most")

    field = ss["edit-field"]
    sel = ss.get("selected")

    def move(i: int, p: np.ndarray) -> None:
        field = ss["edit-field"].copy()
        field[i] = p
        ss["edit-field"] = field
        ss.clicks += 1

    vm = value_map(s, tuple(field.ravel()), sel) if (sel is not None and show_map) else None
    _, summary_e, vs_e = scored(s, tuple(field.ravel()))
    check = e.check(s, field)

    col_field, col_numbers = st.columns([3, 2])
    with col_field:
        hint = (
            f"Click anywhere to move {plots._label(field[sel])}, or click another fielder."
            if sel is not None
            else "Click a fielder to pick them up."
        )
        event = st.plotly_chart(
            plots.field_figure(
                field, keeper_up, hint, style.SERIES[1], mirror, selected=sel, editable=True,
                value_map=vm,
            ),
            config=CHART_CONFIG, key=f"edit-{ss.clicks}", on_select="rerun",
            selection_mode="points",
        )  # fmt: skip
        points = event.selection.points if event and event.selection else []
        if points:
            p = np.array([mirror * float(points[0]["x"]), float(points[0]["y"])])
            near = np.hypot(*(field - p).T)
            if near.min() < PICK_UP:  # a fielder: pick them up, or put them down again
                hit = int(near.argmin())
                ss["selected"] = None if sel == hit else hit
            elif sel is not None:
                field = field.copy()
                field[sel] = p
                ss["edit-field"] = field
            ss.clicks += 1  # a fresh chart, with nothing left selected
            st.rerun()

    with col_numbers:
        mine, rec = summary_e[EDITED], summary_e[CAPTAIN]
        d, ci = vs_e[EDITED]
        a, b = st.columns(2)
        d_runs = 6 * (mine["runs_per_ball"] - rec["runs_per_ball"])
        d_wkts = 100 * (mine["wickets_per_ball"] - rec["wickets_per_ball"])
        a.metric(
            "Runs an over", f"{6 * mine['runs_per_ball']:.2f}",
            f"{d_runs:+.2f} vs captain" if abs(d_runs) >= 0.005 else None,
            delta_color="inverse",
        )  # fmt: skip
        b.metric(
            "Balls a wicket", f"{1 / mine['wickets_per_ball']:.0f}",
            f"{d_wkts:+.2f} wickets per 100" if abs(d_wkts) >= 0.005 else None,
        )  # fmt: skip
        a.metric(
            "Cost a ball", f"{mine['cost_per_ball']:.3f}",
            f"{d:+.3f} ± {ci:.3f}" if abs(d) >= 0.0005 else None, delta_color="inverse",
        )  # fmt: skip
        b.metric("Boundaries", f"{mine['fours'] + mine['sixes']:.1%}")
        if check.legal:
            st.success("Legal field", icon=":material/check:")
        else:
            st.warning("Breaks the rules: " + "; ".join(check.problems), icon=":material/rule:")
            if st.button("Make it legal"):
                ss["edit-field"] = e.repair(s, field)
                ss.clicks += 1
                st.rerun()

        if sel is not None:
            angle, dist = xy_to_wagon(field[sel, 0], field[sel, 1])
            st.markdown(f"**{plots._label(field[sel]).capitalize()}**")

            def from_sliders() -> None:
                a = ss[f"angle-{ss.clicks}"]
                r = ss[f"dist-{ss.clicks}"]
                x, y = wagon_to_xy(a, r / BOUNDARY_RADIUS)
                move(sel, np.array([float(x), float(y)]))

            st.slider(
                "Angle round the ground", 0, 359, int(round(float(angle))) % 360,
                key=f"angle-{ss.clicks}", on_change=from_sliders,
                help="0 behind the stumps, 90 square leg, 180 straight, 270 point",
            )  # fmt: skip
            st.slider(
                "Metres from the pitch centre", 5, 67,
                int(np.clip(round(float(dist) * BOUNDARY_RADIUS), 5, 67)),
                key=f"dist-{ss.clicks}", on_change=from_sliders,
            )  # fmt: skip
            if vm is not None:
                pts, change, legal = vm
                if legal.any():
                    best = int(np.argmin(np.where(legal, change, np.inf)))
                    st.caption(
                        f"Blue spots save runs, orange cost them. The best legal spot for this "
                        f"fielder is near **{nearest_name(pts[best]).replace('_', ' ')}**, "
                        f"{-change[best]:.3f} a ball better than where they stand."
                    )
        st.caption(
            "Scored on the same simulated deliveries as the recommended field. The value map "
            "uses a quarter of them, so treat its smallest differences as noise."
        )

# Wagon wheel ----------------------------------------------------------------------------
with tab_wagon:
    left, right = st.columns(2)
    theta, dist, runs = e.real_wagon(c)
    with left:
        st.plotly_chart(
            plots.wagon_figure(theta, dist, runs, f"Real: {len(runs):,} shots, before 2025",
                               mirror),
            config=CHART_CONFIG, key="wagon-real",
        )  # fmt: skip
    theta, dist, runs = e.simulated_wagon(s, fields[CAPTAIN], n=len(runs) or 1500)
    with right:
        st.plotly_chart(
            plots.wagon_figure(theta, dist, runs, "Simulated, against the RL captain's field",
                               mirror),
            config=CHART_CONFIG, key="wagon-sim",
        )  # fmt: skip
    st.caption(
        f"Shots that stayed in the ground. Sixes: {e.b.real_six[c]:.1%} of balls in the data, "
        f"{e.simulated_six_share(c):.1%} in the shot model. Real runs depend on the fields "
        "batters actually faced, which the data doesn't record; simulated ones are against the "
        "captain's field. A real shot's point is where it was fielded or crossed the rope."
    )

# About ------------------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
**What's behind it.** A shot model (a mixture density network) learned where each batter hits
the ball from 3.46 million tracked deliveries. A fielding simulator, calibrated so that a
standard field reproduces real runs, boundaries, dots and catches, decides what each shot is
worth against any nine fielders. The RL captain is one PPO policy that sets the field for any
batter in any format, then sharpened by self-play against batters who adapt to the field.

**Why the fields differ.** *Conventional* is a textbook field for the situation. *Best field for
all batters* was found by a CMA-ES search against everyone at once. *Dedicated search* reruns
that search for this batter alone. The captain starts from the all-batters field and moves it
for the batter and the match state.

**What to keep in mind.** The data records where the ball went, never where the fielders stood,
so everything about fielders comes from the simulator. The batter here plays their natural
game whatever the field, and sixes have no direction in the shot model. Full method, results
and limitations are in the [repository]({REPO}).
"""
    )

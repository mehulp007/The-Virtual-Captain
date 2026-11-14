"""How much does moving one fielder matter?

best_moves   every fielder nudged a few metres in, out, left or right (relative to the striker);
             the change in runs and cost per ball, with a 95% interval
value_map    for a field with one fielder taken out, how much that fielder would save if
             placed at each point of a grid: a map of where a fielder is worth most
"""

import numpy as np
import polars as pl

from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF
from virtual_captain.field.positions import nearest_name
from virtual_captain.field.rules import check
from virtual_captain.optim.objective import Scenario, outcomes

STRIKER = np.array([0.0, -PITCH_HALF])


def _angle_from_straight(p: np.ndarray) -> float:
    """Degrees between the striker-to-fielder line and straight down the ground (0-180)."""
    v = p - STRIKER
    return float(np.degrees(np.arccos(v[1] / np.linalg.norm(v))))


def _directions(p: np.ndarray, step: float) -> dict[str, np.ndarray]:
    """Unit moves for a fielder at p: in / out along the line from the striker, and the two
    sideways moves, named the way a captain would (straighter / squarer in front of square,
    squarer / finer behind it)."""
    out = (p - STRIKER) / np.linalg.norm(p - STRIKER)
    side = np.array([-out[1], out[0]])
    a = _angle_from_straight(p)
    if _angle_from_straight(p + step * side) > a:
        side = -side  # side now turns the fielder towards straight down the ground
    in_front = a < 90
    return {
        "in": -out,
        "out": out,
        ("straighter" if in_front else "squarer"): side,
        ("squarer" if in_front else "finer"): -side,
    }


def best_moves(field9: np.ndarray, scen: Scenario, step: float = 5.0) -> pl.DataFrame:
    """Every legal single-fielder move of `step` metres, ranked by runs saved per ball."""
    base_runs, base_wkts = outcomes(field9, scen, legalise=False)
    base_runs, base_wkts = base_runs[0], base_wkts[0]
    rows = []
    for i, p in enumerate(field9):
        name = nearest_name(p).replace("_", " ")
        for label, d in _directions(p, step).items():
            moved = field9.copy()
            moved[i] = p + step * d
            off_field = np.hypot(*moved[i]) > BOUNDARY_RADIUS - 1
            if off_field or not check(moved, scen.fmt, scen.phase).legal:
                continue
            runs, wkts = outcomes(moved, scen, legalise=False)
            d_runs = runs[0] - base_runs
            d_cost = d_runs - scen.wicket_value * (wkts[0] - base_wkts)
            se_runs = d_runs.std(ddof=1) / np.sqrt(len(d_runs))
            se_cost = d_cost.std(ddof=1) / np.sqrt(len(d_cost))
            rows.append(
                {
                    "fielder": name,
                    "move": f"{step:g} m {label}",
                    "runs_change": d_runs.mean(),
                    "runs_ci": 1.96 * se_runs,
                    "cost_change": d_cost.mean(),
                    "cost_ci": 1.96 * se_cost,
                }
            )
    return pl.DataFrame(rows).sort("cost_change")


def value_map(field9: np.ndarray, drop: int, scen: Scenario, n_grid: int = 41):
    """Cost saved per ball by putting fielder `drop` at each grid point (NaN off the field)."""
    others = np.delete(field9, drop, axis=0)
    coords = np.linspace(-BOUNDARY_RADIUS, BOUNDARY_RADIUS, n_grid)
    gx, gy = np.meshgrid(coords, coords)
    inside = np.hypot(gx, gy) < BOUNDARY_RADIUS - 1
    points = np.column_stack([gx[inside], gy[inside]])

    far = np.array([[500.0, 500.0]])
    without = np.vstack([others, far])
    r0, w0 = outcomes(without, scen, legalise=False)
    base = r0.mean() - scen.wicket_value * w0.mean()

    fields = np.stack([np.vstack([others, pt]) for pt in points])
    runs, wkts = outcomes(fields, scen, legalise=False)
    saved = base - (runs.mean(axis=1) - scen.wicket_value * wkts.mean(axis=1))
    grid = np.full(gx.shape, np.nan)
    grid[inside] = saved
    return coords, grid

"""Search for the best field with CMA-ES.

Each of the nine fielders is two numbers in [0, 1]: direction (wagon-wheel angle) and distance
from the pitch centre. Every candidate is repaired into a legal field before it is scored, so
the search can roam freely and the rules are never broken.
"""

from dataclasses import dataclass

import numpy as np
from cmaes import CMA

from virtual_captain.data.coords import wagon_to_xy, xy_to_wagon
from virtual_captain.field.geometry import BOUNDARY_RADIUS
from virtual_captain.optim.objective import Scenario, cost

R_MIN, R_MAX = 5.0, 66.0


def encode(field9: np.ndarray) -> np.ndarray:
    angle, dist = xy_to_wagon(field9[:, 0], field9[:, 1])
    r = np.clip(dist * BOUNDARY_RADIUS, R_MIN, R_MAX)
    return np.column_stack([angle / 360.0, (r - R_MIN) / (R_MAX - R_MIN)]).ravel()


def decode(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z).reshape(9, 2), 0.0, 1.0)
    angle = z[:, 0] * 360.0
    r = R_MIN + z[:, 1] * (R_MAX - R_MIN)
    x, y = wagon_to_xy(angle, r / BOUNDARY_RADIUS)
    return np.column_stack([x, y])


@dataclass
class SearchResult:
    field: np.ndarray  # legal (9, 2)
    cost: float
    start_cost: float
    history: list[float]  # best cost after each generation


def optimise(
    scen: Scenario,
    start: np.ndarray,
    generations: int = 80,
    population: int = 16,
    sigma: float = 0.12,
    seed: int = 0,
) -> SearchResult:
    start = scen.legal(start)
    start_cost = float(cost(start, scen)[0])
    best_field, best_cost = start, start_cost
    opt = CMA(
        mean=encode(start),
        sigma=sigma,
        bounds=np.tile([[0.0, 1.0]], (18, 1)),
        population_size=population,
        seed=seed,
    )
    history = []
    for _ in range(generations):
        zs = [opt.ask() for _ in range(opt.population_size)]
        fields = np.stack([scen.legal(decode(z)) for z in zs])
        costs = cost(fields, scen, legalise=False)
        opt.tell(list(zip(zs, costs, strict=True)))
        i = int(np.argmin(costs))
        if costs[i] < best_cost:
            best_cost, best_field = float(costs[i]), fields[i]
        history.append(best_cost)
    return SearchResult(best_field, best_cost, start_cost, history)

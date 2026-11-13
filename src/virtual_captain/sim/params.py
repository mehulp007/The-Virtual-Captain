"""Simulator parameters. Fixed ones are physical constants or well known; the rest are fitted
by `virtual_captain.sim.calibrate` and saved to configs/sim.yaml."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from virtual_captain.paths import CONFIG_DIR

SIM_CONFIG = CONFIG_DIR / "sim.yaml"


@dataclass
class SimParams:
    decel: float = 3.5  # m/s^2, how quickly a ground ball slows on the outfield
    ground_carry: float = 1.25  # recorded distance of a fielded ball x this = free travel
    react: float = 0.35  # s, fielder reaction time
    run_speed: float = 6.0  # m/s, fielder's average speed over a short chase
    reach_ground: float = 1.5  # m, diving reach for a ground ball
    reach_air: float = 1.5  # m, reach for a catch
    keeper_reach: float = 2.5  # m, keeper's reach with gloves
    throw_speed: float = 25.0  # m/s
    pickup: float = 0.6  # s, gather and release
    # s, effective time per run (includes how readily batters run); differs by format
    run_time_t20: float = 2.6
    run_time_odi: float = 2.9
    run_time_test: float = 3.3
    run_sigma: float = 0.5  # s, how sharply run-taking switches on as time to spare grows
    turn: float = 0.5  # s, extra time for each run after the first (turning, throw incoming)
    # s, offset before the first run is under way (negative = backing up, set off early).
    # Batters run much harder at the death than in a Test, so this is per format and phase.
    start_t20_powerplay: float = 0.0
    start_t20_middle: float = 0.0
    start_t20_death: float = -0.5
    start_odi_powerplay: float = 0.2
    start_odi_middle: float = 0.2
    start_odi_death: float = -0.3
    start_test_new_ball: float = 0.4
    start_test_old_ball: float = 0.4
    hang_factor: float = 1.0  # x the flight time of a 45-degree hit to the same distance
    hang_min: float = 0.6  # s, extra air time for top-edges and skiers
    catch_max: float = 0.88  # chance of holding a straightforward catch
    catch_tau: float = 0.4  # s, how quickly catches get easier with more time
    misfield: float = 0.04  # chance a stop is fumbled for an extra run
    air_extra: float = 0.05  # share of unlabelled shots past 15 m that were really in the air
    roll: float = 0.15  # an uncaught aerial ball rolls on this fraction of its carry

    def run_times(self) -> np.ndarray:
        """Run time for each situation code (see SITUATIONS)."""
        by_format = {"T20": self.run_time_t20, "ODI": self.run_time_odi, "TEST": self.run_time_test}
        return np.array([by_format[fmt] for fmt, _ in SITUATIONS])

    def starts(self) -> np.ndarray:
        """Start offset for each situation code (see SITUATIONS)."""
        return np.array([getattr(self, f"start_{f.lower()}_{p}") for f, p in SITUATIONS])

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    def save(self, path: Path = SIM_CONFIG, note: str | None = None) -> None:
        body = yaml.safe_dump({k: round(float(v), 4) for k, v in asdict(self).items()})
        header = f"# {note}\n" if note else ""
        path.write_text(header + body, encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SIM_CONFIG) -> "SimParams":
        if not path.exists():
            return cls()
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))


# format and phase combinations; a delivery's situation code indexes this list
SITUATIONS = [
    ("T20", "powerplay"),
    ("T20", "middle"),
    ("T20", "death"),
    ("ODI", "powerplay"),
    ("ODI", "middle"),
    ("ODI", "death"),
    ("TEST", "new_ball"),
    ("TEST", "old_ball"),
]


def situation_code(fmt: str, phase: str) -> int:
    return SITUATIONS.index((fmt, phase))


# order must match the dataclass fields, the numba kernel reads parameters by index
PARAM_INDEX = {f.name: i for i, f in enumerate(fields(SimParams))}

# Search space for calibration: (low, high). Everything else stays at its default.
CALIBRATED = {
    "decel": (1.0, 10.0),
    "ground_carry": (1.0, 2.5),
    "react": (0.2, 1.0),
    "run_speed": (3.0, 8.0),
    "reach_ground": (0.3, 3.0),
    "reach_air": (0.5, 3.0),
    # an "effective" time per run: batters take singles to ring fielders because a run-out
    # needs a direct hit, so this sits well below the pure running time
    "run_time_t20": (1.5, 4.5),
    "run_time_odi": (1.5, 4.5),
    "run_time_test": (1.5, 4.5),
    "run_sigma": (0.1, 2.0),
    "turn": (0.0, 4.0),
    "start_t20_powerplay": (-2.0, 1.5),
    "start_t20_middle": (-2.0, 1.5),
    "start_t20_death": (-2.0, 1.5),
    "start_odi_powerplay": (-2.0, 1.5),
    "start_odi_middle": (-2.0, 1.5),
    "start_odi_death": (-2.0, 1.5),
    "start_test_new_ball": (-2.0, 1.5),
    "start_test_old_ball": (-2.0, 1.5),
    "hang_factor": (0.6, 2.0),
    "catch_max": (0.6, 1.0),
    "misfield": (0.0, 0.15),
    "air_extra": (0.0, 0.3),
}

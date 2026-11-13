from functools import cache

import yaml

from virtual_captain.paths import CONFIG_DIR

FORMATS = ("T20", "ODI", "TEST")


@cache
def load_format(name: str) -> dict:
    """Load configs/formats/<name>.yaml. Accepts T20 / ODI / TEST in any case."""
    path = CONFIG_DIR / "formats" / f"{name.lower()}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

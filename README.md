# The Virtual Captain

[![CI](https://github.com/mehulp007/the-virtual-captain/actions/workflows/ci.yml/badge.svg)](https://github.com/mehulp007/the-virtual-captain/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Field placement in cricket is mostly set on instinct. This project looks at it as a data problem:
given where a batter tends to hit the ball, against a particular type of bowling, where should the
nine fielders stand?

It works from ball-by-ball records across Test, ODI and T20 cricket, including wagon-wheel
(shot direction and distance) data, and covers all three formats.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is picked up or installed automatically.

```bash
# with an NVIDIA GPU (CUDA 12.8 wheels)
uv sync --extra cu128

# CPU only
uv sync --extra cpu
```

Run the checks:

```bash
uv run pytest
uv run ruff check .
```

Optional, to lint on every commit:

```bash
uv run pre-commit install
```

## Data

Raw data is not stored in the repo. To download it into `data/raw/`:

```bash
uv run python scripts/download_data.py
```

The Kaggle source needs an API token ([how to set one up](https://www.kaggle.com/docs/api)).
Use `--source kaggle` or `--source cricsheet` to fetch one source only.

| Source | What it provides |
|---|---|
| [The Ultimate Ball-by-Ball Cricket Dataset](https://www.kaggle.com/datasets/ariadaikalam/the-ultimate-ball-by-ball-cricket-dataset) | Every ball of international and major T20 league cricket, joined with Cricinfo tracking: wagon-wheel position, line, length, shot type and shot control |
| [Cricsheet](https://cricsheet.org/) | Open ball-by-ball match data, used for the most recent matches |

## Project layout

```
configs/        format rules and run settings
scripts/        command-line entry points
src/virtual_captain/
tests/
```

## Credits and licence

Code is MIT licensed. Ball-by-ball data comes from [Cricsheet](https://cricsheet.org/) under the
Open Data Commons Attribution License. Tracking fields derived from ESPNcricinfo are used for
non-commercial purposes only, with credit to ESPNcricinfo.

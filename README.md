# The Virtual Captain

[![CI](https://github.com/mehulp007/The-Virtual-Captain/actions/workflows/ci.yml/badge.svg)](https://github.com/mehulp007/The-Virtual-Captain/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Field placement in cricket is mostly set on instinct. This project looks at it as a data problem:
given where a batter tends to hit the ball, against a particular type of bowling, where should the
nine fielders stand?

It works from 5.85 million deliveries of Test, ODI and T20 cricket (2001–2026), 3.46 million
of which carry a wagon-wheel point showing where the ball went.

![Scoring zones for 20 batters](reports/figures/scoring_zones.png)

## What the data shows so far

The same batter scores in very different places depending on the ball they get, which is the
whole case for setting a field per batter and per situation rather than per format:

- **Length decides the side of the wicket.** Only 27% of runs off short balls come on the off
  side, against 51% off full balls and yorkers.
- **So does the direction of spin.** Against spin turning away, 44–48% of runs go to the off side;
  against spin turning in, 35–41%.
- **Formats change the shape, not just the volume.** In Tests, 22% of Kohli's runs come through
  the cover region and 23% through mid-wicket. In T20s that becomes 17% and 26%.

![Kohli by format](reports/figures/kohli_by_format.png)

| | |
|---|---|
| ![Kohli at the MCG, 2022](reports/figures/kohli_mcg_2022.png) | ![Tracking coverage](reports/figures/coverage_by_year.png) |

More in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Shot model

A neural network that predicts what happens when a ball is bowled, before any fielder is
involved. It is trained on 3.3 million men's deliveries up to 2024 and tested on the 2025–26
seasons, which it never saw.

Given the batter, the bowler, line and length, bowler type, format, phase, match situation and
venue, it outputs:

- **Contact:** the chance the ball is left or missed, the chance it bowls / traps lbw / stumps the
  batter (dismissals no field can prevent), and the chance it is put into play.
- **The shot:** for a ball put into play, a joint distribution over **direction**, **distance** and
  whether it went **in the air**. This is a mixture density network with 16 components; each
  component has its own direction (von Mises), its own chance of being aerial, and separate
  distance distributions (log-normal) for ground and aerial balls. Edges flying behind the wicket,
  lofted hits and pushes into the ring come out as different components.

Batters and bowlers are represented by learned embeddings, so the model can say how a specific
batter plays a specific kind of ball while batters with little data fall back on what similar
batters do.

### How well it predicts unseen seasons

Average negative log-likelihood on 2025–26 (lower is better):

| | Shot model | Per-batter histogram | Everyone pooled |
|---|---:|---:|---:|
| Contact outcome, per ball (394k balls) | **0.421** | 0.468 | 0.483 |
| Shot, per ball in play (333k shots) | **1.222** | 1.756 | 1.793 |

The per-batter histogram is a strong, honest baseline: each batter's own shot chart, smoothed
towards the population with a strength picked on validation data. The model assigns real shots
about 1.7× the likelihood the histogram does.

![Real vs simulated shot directions for held-out seasons](reports/figures/shot_model_holdout.png)

Most of that gain comes from the delivery itself (line, length, bowler type and phase), not from
knowing who the batter is. Swapping a batter's embedding for the "average batter" one shows how
much identity adds on unseen seasons:

| Batter's balls in training | 300–800 | 800–2,000 | 2,000–6,000 | 6,000+ |
|---|---:|---:|---:|---:|
| Shot NLL improvement from knowing the batter | 0.002 | 0.022 | 0.035 | 0.041 |

It grows with data and never makes things worse, which is what the pooling should do.

Simulated rates line up with reality by format and phase (full table in
[`reports/shot_model_calibration.csv`](reports/shot_model_calibration.csv)):

| Held-out 2025–26 | Real in play | Sim | Real aerial | Sim | Real off side | Sim |
|---|---:|---:|---:|---:|---:|---:|
| Test, new ball | 71.7% | 71.2% | 2.8% | 3.0% | 60.0% | 60.3% |
| ODI, middle | 90.0% | 88.2% | 3.8% | 4.1% | 55.9% | 57.8% |
| T20, death | 84.2% | 82.8% | 20.1% | 19.3% | 48.0% | 46.4% |

![Real vs simulated distances](reports/figures/shot_model_distance.png)

### Known limitations

- **A wagon-wheel point is where the ball was stopped, not where it would have gone.** The
  distance curves above show the field as much as the batter: the bump at about 0.8 of the
  boundary is where boundary riders stop the ball, and almost nothing is recorded just inside the
  rope. The model smooths over that gap. Distances for fielded balls are therefore a lower bound
  on how far the shot would have travelled.
- **"In the air" is only observed when it matters.** A ball is labelled aerial if it went for
  six, was caught or was dropped. Lofted shots that landed safely in a gap look like ground balls.
- **Batting is drifting.** Limited-overs batters put 86.5–87% of balls into play in 2025–26
  against 85% in 2024, so the model under-predicts contact by up to 3 points on those seasons. On
  validation matches from the training years it is within half a point.
- Men's cricket only.

## The deliveries table

`scripts/build_dataset.py` turns the raw CSVs into one Parquet file (about 110 MB, 75 columns,
20 seconds to build). Each row is a delivery with:

- **Match state before the ball:** score, wickets, legal balls bowled, balls remaining, chase
  target and runs required.
- **Phase of play:** powerplay / middle / death for T20 and ODI (The Hundred uses its own 25-ball
  powerplay), new ball / old ball for Tests based on the age of the ball. Defined in
  [`configs/formats`](configs/formats).
- **Bowler type:** fast, medium, finger spin or wrist spin, plus arm, whether the stock ball turns
  into or away from the batter, and whether bowler and batter are on the same side.
- **Venue features** standing in for pitch conditions: runs per ball, boundary rate and the share
  of wickets taken by spin at that ground, shrunk towards the format average for grounds with
  little data. Computed from matches before 2025 so later seasons can be held out.
- **Tracking where available:** line, length, shot, control, and the wagon-wheel point converted
  to metres on the field.

### Field coordinates

Positions are in metres with the origin at the centre of the pitch, the bowler's end at +y, and
the off side of a right-hander at +x. Left-handers are mirrored into the same frame.

The wagon-wheel distance is a fraction of the boundary radius. To turn that into metres, the
boundary radius was fitted against Cricinfo's own "inside the 30-yard circle" flag using the real
shape of the fielding ring (two 30-yard semicircles around each set of stumps, joined by straight
lines). A 68 m boundary reproduces the flag on **98.8%** of tracked balls, which also confirms
that the wagon wheel is measured from the centre of the pitch rather than from the batter.

### Tracking coverage

Share of legal deliveries with each tracking field (men's cricket):

| Format | Years | Balls | Wagon point | Line & length | Shot & control |
|---|---|---:|---:|---:|---:|
| Test | 2001–13 | 764k | 65% | – | 97% |
| Test | 2014–19 | 504k | 77% | 95% | 100% |
| Test | 2020+ | 441k | 77% | 100% | 100% |
| ODI | 2001–13 | 620k | 72% | – | 94% |
| ODI | 2014–19 | 341k | 78% | 81% | 87% |
| ODI | 2020+ | 375k | 67% | 77% | 77% |
| T20 | 2001–13 | 190k | 67% | – | 78% |
| T20 | 2014–19 | 363k | 55% | 59% | 62% |
| T20 | 2020+ | 1,092k | 53% | 61% | 61% |

The full table, including women's cricket, is written to [`reports/coverage.csv`](reports/coverage.csv).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is picked up or installed automatically.

```bash
# with an NVIDIA GPU (CUDA 12.8 wheels)
uv sync --extra cu128

# CPU only
uv sync --extra cpu
```

Then download the data, build the table, train the shot model and draw the figures:

```bash
uv run python scripts/download_data.py
uv run python scripts/build_dataset.py
uv run python scripts/train_shot_model.py
uv run python scripts/make_figures.py
```

Training takes about 5 minutes on an RTX 4060 (30 epochs of ~9 s each, plus evaluation) and logs
to a local MLflow database in `mlruns/` (`uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`
to browse runs). The checkpoint is written to `data/models/shot_model.pt`.

The Kaggle download needs an API token ([how to set one up](https://www.kaggle.com/docs/api)).
`--source kaggle` or `--source cricsheet` fetches one source only.

Checks:

```bash
uv run pytest
uv run ruff check .
uv run pre-commit install   # optional, lint on every commit
```

## Project layout

```
configs/formats/        phase definitions per format
scripts/                download, build, training and figure scripts
src/virtual_captain/
  data/                 loaders, match state and phase features, wagon-wheel coordinates
  field/                field geometry (ring, boundary, frames)
  models/               bowler and venue context, feature encoding, shot model, baselines
  viz/                  field drawing, wagon wheels and scoring-zone roses
notebooks/              exploration
reports/                coverage report, model metrics and figures
tests/
```

## Data sources

| Source | What it provides |
|---|---|
| [The Ultimate Ball-by-Ball Cricket Dataset](https://www.kaggle.com/datasets/ariadaikalam/the-ultimate-ball-by-ball-cricket-dataset) | Every ball of international and major T20 league cricket, joined with Cricinfo tracking: wagon-wheel position, line, length, shot type and shot control |
| [Cricsheet](https://cricsheet.org/) | Open ball-by-ball match data |

## Credits and licence

Code is MIT licensed. Ball-by-ball data comes from [Cricsheet](https://cricsheet.org/) under the
Open Data Commons Attribution License. Tracking fields derived from ESPNcricinfo are used for
non-commercial purposes only, with credit to ESPNcricinfo.

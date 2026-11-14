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
  batter (dismissals no field can prevent), the chance it is hit for six (also decided by the
  batter, not the field), and the chance it is put into play.
- **The shot:** for any other ball put into play, a joint distribution over **direction**,
  **distance** and
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
| Contact outcome, per ball (395k balls) | **0.563** | 0.630 | 0.657 |
| Shot, per ball in play (317k shots) | **1.175** | 1.658 | 1.687 |

The per-batter histogram is a strong, honest baseline: each batter's own shot chart, smoothed
towards the population with a strength picked on validation data. The model assigns real shots
about 1.6× the likelihood the histogram does.

![Real vs simulated shot directions for held-out seasons](reports/figures/shot_model_holdout.png)

Most of that gain comes from the delivery itself (line, length, bowler type and phase), not from
knowing who the batter is. Swapping a batter's embedding for the "average batter" one shows how
much identity adds on unseen seasons:

| Batter's balls in training | 300–800 | 800–2,000 | 2,000–6,000 | 6,000+ |
|---|---:|---:|---:|---:|
| Shot NLL improvement from knowing the batter | −0.001 | 0.016 | 0.029 | 0.038 |

It grows with the amount of data, and for batters with little history it falls back to roughly
the average batter rather than overfitting, which is what the pooling should do.

Simulated rates line up with reality by format and phase (full table in
[`reports/shot_model_calibration.csv`](reports/shot_model_calibration.csv)):

| Held-out 2025–26 | Real in play | Sim | Real six | Sim | Real aerial | Sim | Real off side | Sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test, new ball | 71.3% | 70.7% | 0.4% | 0.6% | 2.2% | 2.4% | 60.2% | 60.1% |
| ODI, middle | 88.4% | 86.5% | 1.6% | 1.9% | 2.1% | 2.3% | 56.5% | 57.6% |
| T20, death | 74.8% | 73.4% | 9.4% | 9.2% | 10.1% | 9.6% | 50.8% | 50.2% |

![Real vs simulated distances](reports/figures/shot_model_distance.png)

### Known limitations

- **A wagon-wheel point is where the ball was stopped, not where it would have gone.** The
  distance curves above show the field as much as the batter: the bump at about 0.8 of the
  boundary is where boundary riders stop the ball, and almost nothing is recorded just inside the
  rope. The model smooths over that gap. Distances for fielded balls are therefore a lower bound
  on how far the shot would have travelled.
- **"In the air" is only observed when it matters.** A ball is labelled aerial if it was caught
  or dropped. Lofted shots that landed safely in a gap look like ground balls.
- **Batting is drifting.** Limited-overs batters put more balls into play in 2025–26 than in
  any training year, so the model under-predicts contact by up to 3 points on those seasons. On
  validation matches from the training years it is within half a point.
- Men's cricket only.

## Fielding simulator

The simulator replays deliveries against a field and works out what each one is worth. It runs
about 6 million deliveries a second on a 16-core laptop CPU (compiled with Numba).

**Fielding restrictions** ([`src/virtual_captain/field/rules.py`](src/virtual_captain/field/rules.py)),
for the nine fielders other than bowler and keeper:

- no more than two behind square on the leg side, in every format
- at most 2 outside the 30-yard ring in the T20 powerplay and 5 after it; in ODIs 2 / 4 / 5 across
  overs 1-10, 11-40 and 41-50; no limit in Tests
- inside the boundary, off the pitch and not on top of the striker

Any field can be repaired into a legal one: the fewest fielders are moved, by the smallest
distance, until every rule holds. 32 named positions and 16 conventional fields (by format,
phase and pace or spin) are defined in [`src/virtual_captain/field`](src/virtual_captain/field).

**Physics, ball by ball.** The shot model decides whether the ball is put into play and, if so,
its direction, distance and whether it is in the air. Then:

- **Along the ground** the ball slows at a constant rate. Each fielder needs a reaction time,
  then runs at a fixed speed and can dive a little way; the first fielder who can get to the
  ball's line before it does stops it.
- **In the air** the hang time grows with the square root of the distance hit, as for a
  projectile. A fielder who reaches the landing spot in time holds the catch, less reliably the
  less time they had; otherwise it lands, rolls on and has to be chased.
- **Edges** (anything in the air that finishes behind the bat within 30 m) are too quick to
  move for: only a fielder already standing near where the edge lands can take it. The keeper,
  slips, gully and leg slip stand where Cricinfo's wagon wheel plots the catches taken there,
  since it draws edges closer to the bat than they really carry.
- **Runs** depend on the time until the ball is back at the stumps. Each run is taken with a
  probability that rises with the time to spare, and every run after the first needs extra time
  for the turn. How readily batters run differs by format and phase.
- Sixes, bowled / lbw / stumped and balls left alone come straight from the shot model.

### Calibrating it

Nobody recorded where the fielders stood, so the simulator can't be checked against the real
fields. Instead each historical delivery is replayed against a stand-in for the field the captain
would have set: the conventional field for that format, phase and type of bowling, with its
boundary riders moved to where that batter's hits reach the deep (learned from training seasons
only).

![Conventional field vs one adapted to Klaasen](reports/figures/field_example.png)

The uncertain parameters are then fitted so simulated dots, singles, twos and threes, fours,
sixes, catches (behind the bat, close in front of it and in the deep) and runs by direction match
reality across all 16 format / phase / bowling groups. The fit uses 480k deliveries before 2025,
a TPE search (1,500 trials) and a CMA-ES refinement (800 trials), and takes about 8 minutes.
The fitted values, in [`configs/sim.yaml`](configs/sim.yaml), are effective rather than literal:

| | Fitted |
|---|---|
| Ground ball deceleration | 8.7 m/s² |
| Fielder speed / effective reaction time | 5.9 m/s / 0.79 s |
| Diving reach, ground / catch | 2.0 m / 1.1 m |
| Straightforward catches held | 95% |
| Time per run: T20 / ODI / Test | 3.3 s / 4.2 s / 3.7 s |
| Extra time for each run after the first | 1.5 s |

### How close it gets

On the held-out 2025–26 seasons, averaged over the 16 groups and weighted by balls:

| | Runs per ball | Dots | Singles | Twos & threes | Fours | Sixes | Caught |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean absolute error | **8.2%** | 11.3% | 11.9% | 18.9% | 11.4% | 21.4% | 14.8% |

Where catches are taken is less accurate than how many: the error is 47% for catches behind the
bat, 34% close in front of it and 28% in the deep.

![Simulator vs reality](reports/figures/sim_calibration.png)

Per-group figures are in [`reports/sim_calibration.csv`](reports/sim_calibration.csv).

### Known limitations

- **The real fields are unknown.** Calibration assumes captains set conventional fields with
  their boundary riders where the batter hits; any difference between that and what they really
  did is absorbed into the fitted parameters, which are effective rather than literal (the 0.79 s
  reaction time also covers anticipation and the first stride).
- **Where catches go is the weakest part.** Edges are plotted with several metres of scatter and
  lofted shots that fell safely aren't labelled as aerial in the data, so the split of catches
  between the cordon, close fielders and the deep is only roughly right.
- **Twos and threes are among the least accurate outcomes** (about ±19%), though they are rare.
- **Sixes in Tests drift.** Sixes are within 6% on training seasons but over-predicted for Tests
  in 2025–26, where they are rare (0.4-1% of balls).
- Run-outs, overthrows, extras and ground-specific boundary sizes are not modelled.

## Setting the field

For any scenario (a batter, or all batters, in a format, phase and type of bowling) three
fields are compared on the same simulated deliveries:

- **Conventional:** the textbook field for the situation.
- **Analyst:** what someone with the batter's wagon wheel and no model would do. Same number of
  boundary riders as the conventional field, placed on the batter's busiest zones.
- **Optimised:** a CMA-ES search over all nine positions (70 generations of 16 fields, started
  from the analyst's field). Every candidate is repaired into a legal field before it is
  scored, so the search can never break the fielding restrictions.

Fields are judged on **runs conceded minus wickets taken, each wicket valued at the runs it
costs the batting side** at that stage. Wicket values come from the data: the drop in expected
runs to the end of the innings when one more wicket is down (first innings of limited-overs
matches, innings one to three of Tests, seasons before 2025):

| Runs a wicket is worth | Powerplay | Middle | Death | | New ball | Old ball |
|---|---:|---:|---:|---|---:|---:|
| T20 | 12.9 | 8.6 | 3.1 | Test | 38.0 | 37.2 |
| ODI | 27.7 | 19.9 | 7.6 | | | |

Each field is searched on 30,000 simulated deliveries and judged on a separate 60,000, so an
optimiser that overfits its own sample gains nothing. All three fields face the same deliveries,
so their differences come with tight 95% intervals. Batters who played in 2025–26 are also judged
on their real contexts from those seasons, which the search never saw.

### Results

76 scenarios: the 16 format / phase / bowling groups with all batters pooled, plus 20 leading
batters in up to three contexts each where they have at least 400 tracked balls. Averages over
the 60 batter scenarios:

| | Conventional | Analyst | Optimised |
|---|---:|---:|---:|
| ODI runs per ball | 0.96 | 0.90 | **0.83** |
| T20 runs per ball | 1.36 | 1.31 | **1.23** |
| Test runs per ball | 0.58 | 0.52 | **0.49** |
| Test wickets per 100 balls | 1.47 | 1.42 | **1.84** |

- The optimised field beats the analyst's in **all 60 batter scenarios** (every 95% interval
  excludes zero) and in **all 40** checks on held-out 2025–26 deliveries.
- The analyst's field beats the conventional one in 48 of 60, and in 38 of 40 held-out checks:
  setting the field to the batter's wagon wheel already helps.
- The 95% intervals compare fields on the same deliveries. Drawing a fresh batch of simulated
  deliveries moves every field's cost up or down together by about 0.02 per ball, so read the
  averages above to about that precision. Every run is seeded and reproduces exactly.

![Fields for three batters](reports/figures/optimised_fields.png)

![Optimised vs analyst, every scenario](reports/figures/optimiser_gains.png)

Full results: [`reports/optimizer_results.csv`](reports/optimizer_results.csv) and every field
in [`reports/optimized_fields.json`](reports/optimized_fields.json).

### What 5 metres is worth

Starting from the analyst's field, every fielder is moved 5 m in, out and to either side, one
at a time:

> **Against David Warner in the T20 powerplay, moving backward point 5 m deeper saves
> 0.026 ± 0.003 runs per ball, two and a half runs every 100 balls.** Against Suryakumar Yadav
> facing spin in the middle overs, moving mid-wicket 5 m straighter saves 0.025 ± 0.003.

The best move for every batter scenario is in [`reports/best_moves.csv`](reports/best_moves.csv).
The map below shows, for Suryakumar against spin, how much one more fielder would save at each
point of the ground with the other eight left where the optimiser put them.

![Where a fielder is worth most](reports/figures/value_map.png)

### Does it rediscover what captains know?

Nothing about cricket tactics was given to the search. Checked on the 16 pooled groups:

| Check | Result |
|---|---|
| Uses every boundary rider the rules allow in limited overs | ✅ all 12 limited-overs groups |
| Slips with the new ball in Tests | ⚠️ partly: keeps one slip where convention sets three |
| Close catchers against spin in Tests | ❌ none |

The Test results are not trustworthy yet. With a wicket worth 38 runs, a captain attacks, but
the optimised Test fields are more defensive than convention (3 to 5 boundary riders). The cause
is in the simulator. Cricinfo's wagon wheel scatters edges by several metres, so the simulator
can't tell first, second and third slip apart: one slip covers the cordon and extra slips add
little. Close catches in front of the bat are similarly under-counted. Limited-overs fields,
where the value is mostly in runs, rest on firmer ground.

### Known limitations

- **The gains are measured inside the simulator.** The comparisons are fair to each field, but
  their size depends on the simulator's assumptions, so the ranking is more reliable than the
  exact number of runs saved.
- **Test fields undervalue catchers** (see above).
- **One field per scenario.** The search finds one field for all the deliveries in a scenario.
  Setting it ball by ball is the RL captain's job, below.
- The optimiser sometimes puts two fielders close together, which the simulator rewards but a
  captain might not.

## The RL captain

A search like the one above has to be rerun for every batter and every situation, and it never
sees the match move. The RL captain is one policy, trained with PPO, that sets the field before
every ball for whoever is on strike, in any format, and is expected to keep up with a dedicated
search for each batter.

**What it sees** before each ball (75 numbers): the format, pace or spin, the phase, wickets
down, how far through the innings it is, the ball of the over, what a wicket is worth right now,
and a 29-number profile of each batter at the crease: where their runs go by sector, how often
they hit to the deep, in the air or for six, and how often they are bowled or lbw.

**What it does:** it moves each of the nine fielders from the field optimised for all batters in
that situation (18 numbers: a change of angle and distance for each). The result is repaired into
a legal field, so the policy can't break the fielding restrictions. With no moves it sets the
CMA-ES field from the section above, which makes that field its starting point.

**What it is rewarded for:** the cost it saves against that starting field (runs minus wickets
times their value) on the same 256 deliveries simulated from the striker's shot model. Scoring
both fields on the same deliveries removes most of the luck from the reward. One of those
deliveries is played for real: it decides the runs, whether the strike changes and whether a
wicket falls, in which case a new batter comes in. An episode is an over in T20s and ODIs, and a
four-over spell in Tests.

**What a wicket is worth** moves with the match. A Duckworth–Lewis style resource curve is fitted
to the data for each limited-overs format, and a wicket's value is the resources it takes away at
that point. In a T20 with 19 overs left, the first wicket is worth about 11 runs and the fifth
about 22. In Tests the value comes directly from the data for each wicket.

It learns from 305 batters with at least 800 tracked balls in a situation before 2025 (1,024
batter contexts). Another 43 batters, about one in seven, are kept out of training entirely.
Every context has its own bank of simulated deliveries to learn from and a separate bank to be
judged on. Training ran for 20 million balls in 256 environments simulated side by side, T20
only for the first 4 million and then all three formats. It took about two hours on a laptop CPU
and was logged to MLflow.

![Learning curve](reports/figures/captain_learning.png)

### Results

The bar is the dedicated search: the CMA-ES search from the section above, rerun for every batter
context on its learning bank and started from the same all-batters field as the captain.
Everything below is judged on deliveries that neither the search nor the captain trained on.

**One batter, one situation.** The captain sets a field for each batter context in its typical
match state. Cost per ball, captain minus dedicated search, below zero meaning the captain is
better:

| | T20 | ODI | Test |
|---|---:|---:|---:|
| Batters it trained on | **−0.014** (beats it in 78%) | **−0.006** (63%) | +0.003 (53%) |
| Batters it never saw | **−0.006** (60%) | **−0.008** (69%) | **−0.007** (59%) |

Across all 1,024 contexts, one policy beats the dedicated searches in 61%. It is significantly
better in 218 and significantly worse in 101. The searches get more out of a single context in
Tests, where a wicket is worth up to 45 runs and each search can fit its own deliveries closely.
Against batters it has never seen it is ahead in all three formats: a search over 12,000
simulated deliveries partly fits that sample's noise, while the policy learns patterns shared
across 305 batters.

![Captain vs dedicated search, every context](reports/figures/captain_vs_search.png)

**Overs as they unfold.** Whole overs (spells in Tests) from real match states with two batters
at the crease. Strike changes, wickets fall, and the wicket's value moves with them. Every way of
setting the field plays the same overs with the same random numbers. Cost per ball against
switching to the dedicated field for whoever is on strike:

| | Conventional field | All-batters field | Captain |
|---|---:|---:|---:|
| T20, batters it trained on | +0.157 | +0.005 | **−0.017** ± 0.001 |
| T20, batters it never saw | +0.158 | +0.009 | **−0.013** ± 0.001 |
| ODI, batters it trained on | +0.168 | +0.005 | **−0.007** ± 0.001 |
| ODI, batters it never saw | +0.163 | +0.002 | **−0.006** ± 0.001 |
| Test, batters it trained on | +0.196 | +0.008 | +0.001 ± 0.002 |
| Test, batters it never saw | +0.199 | 0.000 | **−0.006** ± 0.002 |

Over a T20 innings that is about 2 runs saved against switching between dedicated fields, and 2.7
against one good field for everyone. Textbook fields cost around 0.16 runs per ball more than any
optimised field, about 19 runs in a T20 innings.

**Real deliveries from 2025–26.** The 20 batters from the field-setting study have 72 contexts
with at least 150 balls in the held-out seasons. On those deliveries the captain's field beats
the dedicated search in 49 (24 of 29 in T20, 14 of 20 in ODIs, 11 of 23 in Tests) and the
all-batters field in 62. The dedicated search beats the all-batters field in only 36.

![Fields for three batters](reports/figures/captain_fields.png)

Below, a T20 death over plays out with the field reset before every ball. The field changes when
the strike does.

![One over, ball by ball](reports/figures/captain_over.png)

The captain's field for Suryakumar Yadav against spin, at every training checkpoint:

![The field during training](reports/figures/captain_training.gif)

Full results are in [`reports/captain_static.csv`](reports/captain_static.csv),
[`reports/captain_overs.csv`](reports/captain_overs.csv) and
[`reports/captain_holdout.csv`](reports/captain_holdout.csv), with the fields in
[`reports/captain_fields.json`](reports/captain_fields.json).

### Known limitations

- **It inherits the simulator**, including the way the simulator undervalues Test catchers. The
  captain can only be as good as the simulated deliveries it learns from.
- **It doesn't play for the strike.** Each ball is rewarded for the cost it saves on that ball, so
  a field that gives away a single to get the weaker batter on strike earns nothing for it.
- **Moving fielders is free.** It can reshuffle all nine between balls, which a real captain
  would not do. It also plans only within an over or spell, with no thought for the next bowler.
- **The batter doesn't adapt here.** Each batter plays to their shot model whatever field is
  set. The next section gives the batter a say.

## The batter fights back

Everything so far assumes the batter ignores the field. A real batter sees the gaps and goes for
them, and a field that only works against a batter who doesn't look is worth less than it seems.
Here the batter gets to look, and the captain trains against batters who do.

**What the batter can do.** Before each ball it picks an intent: how much to attack (−1 to 1)
and a direction to work the ball towards. The intent reweights the 256 simulated deliveries the
ball is judged on. Attacking moves weight to balls hit to the deep, in the air and over the rope.
The direction moves weight to ground shots heading that way. The share of balls the batter
connects with stays the same, and no intent can conjure a shot the batter doesn't play: it can
only make the ones they have more or less common.

**What the batter sees**: the same match state and 29-number profile as the captain, plus the
field as 24 counts of fielders inside and outside the ring in each 30-degree sector.

**What it costs, from the data.** A batter who changes their game gets out more. In the tracking
data a false shot (not in control) brings a dismissal 13.6% of the time, against 0.24% for a
controlled one, and shots into a batter's rare sectors are false 35% of the time, against 5% in
their usual ones. To put a number on it, every batter's game is compared across phases of the
innings (same batter, same format, same type of bowling, at least 300 balls). The dismissals
beyond what each phase's mix of shots explains on its own are fitted to two things: how much
more the batter attacked, and how far their scoring pattern moved (KL divergence over 27 cells).

| | Extra dismissals per ball, per unit of attack | per nat of pattern shift | Largest shift seen (99th pct) | Cases |
|---|---:|---:|---:|---:|
| T20 | 0.112 ± 0.011 | 0.015 ± 0.014 | 0.154 nats | 676 |
| ODI | 0.043 ± 0.006 | 0.022 ± 0.008 | 0.203 nats | 746 |
| Test | 0 (fit −0.004 ± 0.008) | 0.003 ± 0.011 | 0.049 nats | 1,057 |

Those slopes become an extra chance of getting out on every ball the batter forces, and no
intent can move a batter's game further than the largest shift seen in the data for that format.
In a typical T20 state, a batter attacking flat out pays about one extra wicket for every 24
extra runs.

**The game.** Captain and batter play a zero-sum game: every run the batter gains over their
natural game is a run the captain loses. It is written as a
[PettingZoo](https://pettingzoo.farama.org/) environment, with batched versions of each side for
training.

**Self-play.** It starts from the captain in the previous section. Each round the batter carries
on training (PPO, 2 million balls) against every captain so far, each over played against one of
them. Then the captain carries on training (3 million balls) against every batter so far,
including the natural one. Playing a mix of past opponents rather than only the latest one is
fictitious play: it stops the two sides chasing each other round in circles. Four rounds took
about three hours.

**Exploitability.** To measure how far a field can be exploited, a fresh batter is trained from
scratch (1.5 million balls) against that field-setter alone. What it gains over its natural game
is measured on deliveries neither side trained on. This is done for the captain after every
round and for the two fixed fields from the CMA-ES searches, with the same budget for each.

### Results

Runs per ball the best-response batter gains (± about 0.005, 95% interval; lower is harder to
exploit):

| | All-batters field | Searched field | Captain at the start | Captain after 4 rounds |
|---|---:|---:|---:|---:|
| T20, batters it trained on | 0.329 | 0.327 | 0.328 | **0.308** |
| T20, batters it never saw | 0.327 | 0.313 | 0.323 | **0.301** |
| ODI, batters it trained on | 0.140 | 0.144 | 0.146 | **0.122** |
| ODI, batters it never saw | 0.135 | 0.129 | 0.138 | **0.116** |
| Test, batters it trained on | 0.109 | **0.088** | 0.103 | 0.096 |
| Test, batters it never saw | 0.100 | **0.078** | 0.095 | 0.091 |

Exploitability falls in every round in T20s and ODIs, and in all but the last round in Tests:
by 6% in T20s, 17% in ODIs and 7% in Tests overall. Each round's new batter also gets less out
of the newest captain than the batter before it did (in T20s: 0.328, 0.323, 0.317, 0.313 runs a
ball). In limited-overs cricket the final captain is harder to exploit than either fixed field.
In Tests the dedicated search for each batter is still the hardest to exploit.

The price is small. Against batters playing their natural game, the final captain gives away
0.003 runs a ball more than at the start in T20s and 0.004 more in ODIs, and 0.002 less in Tests.
Against a batter who knows the field, it gives away 0.018 runs a ball less in T20s (about 2 runs
an innings) and 0.020 less in ODIs.

![Exploitability over the rounds](reports/figures/selfplay_exploitability.png)

**Most of what the batter gains can't be fielded away.** Each best-response batter was also set
against every other field. The batters hardly specialise: one trained to beat one field gains
almost as much against any other, and the differences between fields are larger than the
differences between batters. Mostly they learn to attack harder. For a T20 batter in a typical
state, sixes go from 5.7% to 9.5% of deliveries and catching chances from 3.7% to 6.1%, and no
placement of nine fielders takes that away. What self-play wins back is the part tied to the
field, about 0.02 runs a ball.

![Every field against every batter](reports/figures/selfplay_matrix.png)

Below, for three batters: the fields before and after self-play, and where the batter who knows
each field sends its extra shots. Against spin, Suryakumar Yadav's best response nearly doubles
his sixes and works the ball fine, behind the wicket.

![Fields before and after self-play](reports/figures/selfplay_fields.png)

Full results are in [`reports/selfplay_exploitability.csv`](reports/selfplay_exploitability.csv),
[`reports/selfplay_rounds.csv`](reports/selfplay_rounds.csv) and
[`reports/selfplay_matrix.csv`](reports/selfplay_matrix.csv), with the fields in
[`reports/selfplay_fields.json`](reports/selfplay_fields.json) and the fitted risk in
[`reports/forcing_risk.csv`](reports/forcing_risk.csv).

### Known limitations

- **The batter only plays shots they already have.** Reweighting the simulated deliveries can
  make a shot more or less common but can't invent one, so a batter who never ramps can't learn
  to.
- **The risk is linear and capped.** It is fitted from batters' games moving between phases of
  the innings, which is not the same as a batter deliberately going after a gap. In Tests the
  extra risk from attacking comes out at zero, so a Test batter's intent is held back only by
  the cap and by the shots it makes more likely.
- **The batter sees the field coarsely**: fielder counts by sector, not exact positions.
- **It inherits the simulator**, like everything above. Four rounds is also a short run:
  exploitability was still falling at the end in T20s and ODIs.

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

Then download the data, build the table, train the shot model, calibrate the simulator and
draw the figures:

```bash
uv run python scripts/download_data.py
uv run python scripts/build_dataset.py
uv run python scripts/train_shot_model.py
uv run python scripts/calibrate.py
uv run python scripts/optimize_fields.py
uv run python scripts/build_captain_data.py
uv run python scripts/train_captain.py
uv run python scripts/evaluate_captain.py
uv run python scripts/train_selfplay.py
uv run python scripts/evaluate_selfplay.py
uv run python scripts/make_figures.py
```

Training takes about 5 minutes on an RTX 4060 (30 epochs of ~9 s each, plus evaluation) and logs
to a local MLflow database in `mlruns/` (`uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`
to browse runs). The checkpoint is written to `data/models/shot_model.pt`.
Calibrating the simulator takes about 8 minutes and optimising all 76 field scenarios about 16.
Building the captain's delivery banks and running a search for each of its 1,024 batter contexts
takes about an hour and a half. Training the captain takes about two hours, and judging it a few
minutes. Four rounds of self-play take about three hours, and training a best-response batter
against every field takes another hour. `scripts/build_captain_data.py --risk-only` refits just
the batter's forcing risk.

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
configs/formats/        phases and fielding restrictions per format
configs/sim.yaml        fitted simulator parameters
scripts/                download, build, training and figure scripts
src/virtual_captain/
  data/                 loaders, match state and phase features, wagon-wheel coordinates
  field/                geometry, named positions, conventional fields, fielding rules
  models/               bowler and venue context, feature encoding, shot model, baselines
  sim/                  fielding physics, simulator and its calibration
  optim/                wicket values, analyst fields, CMA-ES search, sensitivity
  rl/                   the captain's data, the over-by-over environment and its evaluation,
                        the adaptive batter, the two-player game and self-play
  viz/                  field drawing, wagon wheels and scoring-zone roses
notebooks/              exploration
reports/                coverage, model, simulator, optimiser, captain and self-play
                        results, figures
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

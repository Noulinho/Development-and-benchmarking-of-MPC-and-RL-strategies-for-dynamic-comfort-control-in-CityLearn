# Development and benchmarking of MPC and RL strategies for dynamic comfort control in CityLearn

Thesis benchmark for IEA EBC Annex 96 Common Exercise 1: heat-pump control in
25 Vermont homes using CityLearn. The study compares Safe-PI, SAC, MPC, and
GNU-RL under limited commissioning data and simulated occupant feedback.

**Start with the results:** the notebooks below contain the final thesis tables
and saved figures. They can be viewed directly on GitHub, and their cells run
from the included exports without training a controller.

[![Published results](https://github.com/Noulinho/annex96_common_exercise_1/actions/workflows/published-results.yml/badge.svg)](https://github.com/Noulinho/annex96_common_exercise_1/actions/workflows/published-results.yml)

| Notebook | What you will find |
|---|---|
| [Controller comparison](analysis/01_no_occupant_results.ipynb) | Tables VII–VIII: February/March portfolio KPIs, representative buildings, and model-adaptation diagnostics. |
| [Occupant feedback](analysis/02_occupant_results.ipynb) | Table IX: four occupant types, dynamic references, thermostat overrides, and comfort profiles. |
| [Load shifting](analysis/03_load_shifting_results.ipynb) | Daily demand profiles, reference-bias windows, and deployment KPI trajectories. |
| [Occupancy validation](analysis/09_bspline_occupancy_validation.ipynb) | B-spline forecasts, January-only holdout evaluation, and persistence comparison. Requires the simulation environment to rerun. |

![Final no-occupant daily load profiles](analysis/figures/ce1_thesis_clean_load_figures/figure_A_no_occupant_feb_mar_combined.png)

## Main findings

January supplies commissioning data. Controllers deploy in February, then use
February data for SAC continuation and GNU-RL adaptation before March.

| Controller | February discomfort | March discomfort | Model / training |
|---|---:|---:|---|
| Safe-PI/RBC | 41.71% | 34.18% | PI with safety limits |
| MPC | **1.10%** | **1.17%** | Fitted 2R2C model with Kalman correction |
| GNU-RL | 3.94% | 3.75% | Differentiable 1R1C MPC with online adaptation |
| SAC | 27.09% | 5.70% | January learning, then February continuation |

These are mean building-level no-occupant results from thesis Table VII.
MPC preserves comfort most reliably among the tested controllers. SAC improves
substantially with continuation data. Lower electricity consumption must be
read alongside comfort: an underheating controller can appear energy-efficient.

With simulated occupants, the B-spline dynamic reference reduces thermostat
override pressure, with performance depending on occupant type and building.
The thesis reports approximately 78% override reduction for MPC and 72% for
GNU-RL. The [occupant notebook](analysis/02_occupant_results.ipynb) includes the
paired building-level counts and explains the auxiliary comfort indicators.

## Technical contributions

The thesis work builds on the bundled CityLearn framework and differentiable
MPC/GNU-RL code. The project-specific models, controller integration, occupant
logic, and benchmark analysis are organized below; the bundled frameworks
retain their upstream attribution.

| Project work | Code |
|---|---|
| Thermal model identification and validation | [`src/models/`](src/models/) |
| MPC integration and state estimation | [`src/controllers/mpc.py`](src/controllers/mpc.py), [`scripts/03_run_mpc.py`](scripts/03_run_mpc.py) |
| SAC training and continuation pipeline | [`src/sac_merlin_pipeline.py`](src/sac_merlin_pipeline.py) |
| GNU-RL integration and adaptation | [`src/rlmpc_pipeline.py`](src/rlmpc_pipeline.py) |
| Synthetic occupant feedback and comfort references | [`src/occupants/`](src/occupants/) |
| Reward formulation and benchmark evaluation | [`src/rewards.py`](src/rewards.py), [`analysis/`](analysis/) |

## Reproducibility

| Capability | Status |
|---|---|
| Verify published CSV/PNG integrity | SHA-256 manifest checked automatically |
| Recompute Tables VII–IX from saved building exports | Six tables checked against the published CSVs |
| Rebuild the three reading notebooks | Executed with a locked analysis environment |
| Inspect controller source and experiment configurations | Included in this repository |
| Reproduce full training from a fresh environment | Not independently validated; generated training artifacts remain local |

The analysis workflow is tested on Linux with Python 3.11.14. It runs without
the simulation dependencies. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/verify_results.py
python -m unittest discover -s tests -p test_published_results.py -v
python scripts/build_results_notebooks.py --execute
```

Verification checks all 37 published artifacts, expected experiment identifiers,
25-building coverage, numerical validity, summary calculations, and paired
override comparisons. GitHub Actions runs these checks and executes the three
reading notebooks on pushes and pull requests. Figures are preserved PNG exports;
the notebooks embed them without regenerating simulation trajectories.

## Read the notebooks

To open and rerun the three final-results notebooks:

```bash
python -m pip install -r requirements-analysis.txt
jupyter lab analysis/01_no_occupant_results.ipynb
```

To rebuild all three notebooks into a separate directory:

```bash
python scripts/build_results_notebooks.py --execute --output-dir /tmp/ce1-notebooks
```

Full controller training requires the local CityLearn implementation, simulation
dependencies, and commissioning/model artifacts. See the
[reproduction guide](docs/reproduction.md) for the stages and limitations;
[the scripts guide](scripts/README.md) identifies each entry point.

## Repository map

| Directory | Purpose |
|---|---|
| [`analysis/`](analysis/README.md) | Final reading notebooks, published tables, figures, and provenance manifest. |
| [`scripts/`](scripts/README.md) | Benchmark collection, fitting, training, deployment, and result presentation. |
| [`src/`](src/) | Reusable models, controllers, occupant logic, rewards, and evaluation. |
| [`configs/`](configs/) | Building, controller, and scenario configuration. |
| [`experiments/`](experiments/README.md) | Research history: reward ablations, trial runs, scratch notebooks, and original publication exporters. |
| [`notebooks/`](notebooks/README.md) | Original CE1 onboarding tutorial. |
| [`citylearn/`](citylearn/) | Project's local CityLearn implementation. |
| [`diff_mpc/`](diff_mpc/) / [`Gnu-RL/`](Gnu-RL/) | Differentiable MPC implementation and upstream GNU-RL reference. |
| `results/` | Local generated trajectories, agents, fitted models, and replay buffers; excluded from Git. |

Historical notebooks retain their saved outputs under `experiments/`. Earlier
sections called “final” represent intermediate research milestones; the three
reading notebooks above identify the final thesis result set explicitly.

The occupant models are controlled simulation components. The work does not
establish field performance with real thermostat users. The original license
and upstream attribution remain in [LICENSE](LICENSE) and the bundled projects.

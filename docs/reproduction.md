# Reproducing the analysis

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Validate the dataset

```bash
PYTHONPATH=. python -m analysis.validate_data
```

This drops two corrupted rows, writes coverage and regime summaries to `data/derived/`, and records the 3,969 missing symmetric-model runs in `validation_report.txt`.

## Regenerate summary tables

```bash
PYTHONPATH=. python -m analysis.compute_summaries
```

## Regenerate figures

```bash
PYTHONPATH=. python -m analysis.generate_figures
```

Outputs land in `paper/figures/` as PDF and PNG.

## Rebuild the manuscript

```bash
PYTHONPATH=. python -m analysis.build_manuscript
cd paper
bash build.sh
```

## Re-run the simulation (optional)

The full grid is large (~373k configurations). To resume from checkpoint:

```bash
cd code
python simulation.py
```

Results append to `data/full_results_v2.csv` and update `data/checkpoint_v2.json`.

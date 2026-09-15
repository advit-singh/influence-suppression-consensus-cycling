# Influence Suppression and Consensus Cycling

Independent research on how asymmetric status-based peer penalization ("tall poppy syndrome") interacts with cognitive anchoring in distributed consensus networks.

## Summary

When agents suppress high-performing peers based on positive deviation from the group mean, consensus no longer follows classical ergodicity. Partial stubbornness to initial beliefs creates a switched affine system with four stability regimes:

| Regime | Description | Share of all runs |
|--------|-------------|-------------------|
| **1** | Biased convergence | 23.5% |
| **1B** | Fixed dissensus | 74.0% |
| **2** | Slow mixing | 2.0% |
| **3** | Non-convergent cycling | 0.5% |

In the proposed directional model, limit cycles are rare (0.02%) and emerge only under maximal penalty with non-zero stubbornness. Kinship clusters of size at least **N/5** eliminate cycling entirely. Symmetric suppression increases cycling to 0.52%; the anti-expert inverse-variance baseline fails in 54.5% of runs.

**372,960** validated simulation runs across six models, four topologies, and a pruned penalty/stubbornness/kinship grid.

## Repository layout

```
code/           Simulation engine (Python + original Colab notebook)
data/           Full results CSV, checkpoint, execution log, derived summaries
analysis/       Validation, table generation, figure scripts, LaTeX builder
paper/          LaTeX manuscript, bibliography, compiled figures
docs/           Reproduction instructions
```

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m analysis.validate_data
PYTHONPATH=. python -m analysis.compute_summaries
PYTHONPATH=. python -m analysis.generate_figures
PYTHONPATH=. python -m analysis.build_manuscript
cd paper && bash build.sh
```

See [docs/reproduction.md](docs/reproduction.md) for details.

## Data note

The full pruned parameter grid contains 372,960 configurations across six models. All reported statistics use the complete validated dataset in `data/full_results_v2.csv`.

## Citation

Advit Singh. *Influence Suppression and Consensus Cycling in Status-Penalized Networks.* Independent research, 2026. Code and data: https://github.com/advit-singh/influence-suppression-consensus-cycling

## License

MIT License. See [LICENSE](LICENSE).

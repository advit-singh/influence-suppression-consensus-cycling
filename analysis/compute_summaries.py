"""Compute manuscript summary tables from cleaned simulation output."""

from __future__ import annotations

import pandas as pd

from analysis.data_utils import load_results, save_derived


def regime_rates(frame: pd.DataFrame, model: str | None = None) -> pd.Series:
    subset = frame if model is None else frame[frame["model"] == model]
    return subset["regime"].value_counts(normalize=True).mul(100).round(2)


def cross_model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model"):
        rows.append(
            {
                "model": model,
                "validated_runs": len(group),
                "regime_1_pct": round((group["regime"] == "Regime 1 (Biased Convergence)").mean() * 100, 2),
                "regime_1b_pct": round((group["regime"] == "Regime 1B (Fixed Dissensus)").mean() * 100, 2),
                "regime_2_pct": round((group["regime"] == "Regime 2 (Slow Mixing)").mean() * 100, 2),
                "regime_3_pct": round((group["regime"] == "Regime 3 (Non-convergent Cycling)").mean() * 100, 2),
                "mean_consensus_error": round(group["consensus_error"].mean(), 4),
                "std_consensus_error": round(group["consensus_error"].std(), 4),
                "mean_temporal_variance": group["Var_temp"].mean(),
                "mean_delta_product": round(group["delta_product"].mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def phase_transition_matrix(frame: pd.DataFrame, model: str = "proposed") -> pd.DataFrame:
    subset = frame[frame["model"] == model]
    matrix = subset.groupby(["gamma_stub", "P"])["regime"].apply(
        lambda values: (values == "Regime 3 (Non-convergent Cycling)").mean()
    )
    return matrix.unstack(fill_value=0.0)


def kinship_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[(frame["model"] == "proposed") & (frame["gamma_stub"] == 0.5)]
    matrix = subset.groupby(["K_fraction", "P"])["regime"].apply(
        lambda values: (values == "Regime 3 (Non-convergent Cycling)").mean()
    )
    return matrix.unstack(fill_value=0.0)


def topology_vulnerability(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[(frame["model"] == "proposed") & (frame["P"] >= 0.7)]
    return subset.groupby("topology")["regime"].apply(
        lambda values: (values == "Regime 3 (Non-convergent Cycling)").mean() * 100
    ).round(2).reset_index(name="regime_3_pct")


def main() -> None:
    frame = load_results()

    save_derived(cross_model_summary(frame), "cross_model_summary.csv")
    save_derived(phase_transition_matrix(frame), "phase_transition_r3.csv")
    save_derived(kinship_matrix(frame), "kinship_dampening_matrix.csv")
    save_derived(topology_vulnerability(frame), "topology_vulnerability.csv")

    p0 = frame[(frame["model"] == "proposed") & (frame["P"] == 0.0)]
    p0_summary = p0.groupby(["gamma_stub", "topology"])["regime"].value_counts().rename("count")
    save_derived(p0_summary.reset_index(), "p0_sanity_check.csv")

    print("Derived summaries written to data/derived/")


if __name__ == "__main__":
    main()

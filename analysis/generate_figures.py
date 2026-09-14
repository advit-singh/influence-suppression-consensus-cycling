"""Generate publication figures for the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.data_utils import load_results, repo_root

FIGURE_DIR = repo_root() / "paper" / "figures"
REGIME_ORDER = [
    "Regime 1 (Biased Convergence)",
    "Regime 1B (Fixed Dissensus)",
    "Regime 2 (Slow Mixing)",
    "Regime 3 (Non-convergent Cycling)",
]
REGIME_COLORS = {
    "Regime 1 (Biased Convergence)": "#2ca02c",
    "Regime 1B (Fixed Dissensus)": "#1f77b4",
    "Regime 2 (Slow Mixing)": "#ff7f0e",
    "Regime 3 (Non-convergent Cycling)": "#d62728",
}


def _style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def figure_01_penalty_profiles() -> None:
    d = np.linspace(-1.0, 3.0, 400)
    linear = np.where(d <= 0, 1.0, np.maximum(0.0, 1.0 - d))
    step = np.where(d <= 0.5, 1.0, 0.0)
    tanh = np.where(d <= 0, 1.0, 1.0 - np.tanh(d))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(d, linear, label="Linear", linewidth=2.0)
    ax.plot(d, step, label="Step", linewidth=2.0)
    ax.plot(d, tanh, label="Tanh", linewidth=2.0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Standardized status deviation $D$")
    ax.set_ylabel("Suppression factor $f(D)$ at $P=1$")
    ax.legend(frameon=True)
    ax.set_title("Penalty function shapes")
    save_figure(fig, "fig01_penalty_profiles")


def figure_02_minimal_phase_portrait() -> None:
    # Three-agent schematic from Section 5: x1 high outlier, x2/x3 below mean.
    steps = 80
    gamma = 0.2
    P = 1.0
    x = np.array([0.8, -0.2, -0.4])
    x0 = x.copy()
    traj = [x.copy()]

    for _ in range(steps):
        mean = x.mean()
        std = max(x.std(), 1e-8)
        f = np.ones(3)
        for i in range(3):
            D = (x[i] - mean) / std
            if D > 0:
                f[i] = max(0.0, 1.0 - P * D)
        W = np.array(
            [
                [0.0, f[1], f[2]],
                [f[0], 0.0, f[2]],
                [f[0], f[1], 0.0],
            ]
        )
        for i in range(3):
            row_sum = W[i].sum()
            W[i] = W[i] / row_sum if row_sum > 0 else np.eye(3)[i]
        peer = W @ x
        x = gamma * x0 + (1.0 - gamma) * peer
        traj.append(x.copy())

    traj = np.array(traj)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(traj[:, 0], traj[:, 1], color="#d62728", linewidth=1.5, label="Trajectory")
    ax.scatter(traj[0, 0], traj[0, 1], color="black", s=30, zorder=3)
    ax.scatter(traj[-1, 0], traj[-1, 1], color="#1f77b4", s=30, zorder=3)
    ax.set_xlabel("$x_1(t)$")
    ax.set_ylabel("$x_2(t)$")
    ax.set_title("Minimal three-agent phase projection")
    ax.legend(frameon=True)
    save_figure(fig, "fig02_phase_portrait")


def figure_03_phase_heatmaps(frame: pd.DataFrame) -> None:
    subset = frame[(frame["model"] == "proposed") & (frame["K_fraction"] == "N/10")]
    r3 = subset.groupby(["gamma_stub", "P"])["regime"].apply(
        lambda values: (values == "Regime 3 (Non-convergent Cycling)").mean() * 100
    ).unstack(fill_value=0.0)
    r1b = subset.groupby(["gamma_stub", "P"])["regime"].apply(
        lambda values: (values == "Regime 1B (Fixed Dissensus)").mean() * 100
    ).unstack(fill_value=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    sns.heatmap(r3, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "%"}, ax=axes[0])
    axes[0].set_title("Regime 3 cycling rate")
    axes[0].set_xlabel("Penalty magnitude $P$")
    axes[0].set_ylabel("Stubbornness $\\gamma_{stub}$")

    sns.heatmap(r1b, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "%"}, ax=axes[1])
    axes[1].set_title("Regime 1B fixed dissensus rate")
    axes[1].set_xlabel("Penalty magnitude $P$")
    fig.suptitle("Proposed model phase structure ($K=N/10$)", y=1.02)
    save_figure(fig, "fig03_phase_heatmaps")


def figure_04_kinship_scaling(frame: pd.DataFrame) -> None:
    subset = frame[(frame["model"] == "proposed") & (frame["gamma_stub"] == 0.5)]
    curves = []
    for k_frac in ["N/10", "N/5", "N/2"]:
        part = subset[subset["K_fraction"] == k_frac]
        series = part.groupby("P")["regime"].apply(
            lambda values: (values == "Regime 3 (Non-convergent Cycling)").mean() * 100
        )
        curves.append(series)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = ["o", "s", "^"]
    for series, label, marker in zip(curves, ["N/10", "N/5", "N/2"], markers):
        ax.plot(series.index, series.values, marker=marker, linewidth=2.0, label=f"$K={label}$")
    ax.set_xlabel("Penalty magnitude $P$")
    ax.set_ylabel("Regime 3 cycling probability (%)")
    ax.set_title("Kinship immunity scaling at $\\gamma_{stub}=0.5$")
    ax.legend(frameon=True)
    save_figure(fig, "fig04_kinship_scaling")


def figure_05_ablation(frame: pd.DataFrame) -> None:
    models = ["proposed", "symmetric", "fixed_suppression"]
    labels = ["Proposed", "Symmetric", "Fixed suppression"]
    subset = frame[frame["model"].isin(models)].copy()

    counts = subset.groupby(["model", "regime"]).size().reset_index(name="count")
    totals = counts.groupby("model")["count"].transform("sum")
    counts["pct"] = 100 * counts["count"] / totals
    counts["model"] = pd.Categorical(counts["model"], categories=models, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bottom = np.zeros(len(models))
    for regime in REGIME_ORDER:
        values = []
        for model in models:
            row = counts[(counts["model"] == model) & (counts["regime"] == regime)]
            values.append(float(row["pct"]) if not row.empty else 0.0)
        axes[0].bar(labels, values, bottom=bottom, label=regime.replace("Regime ", ""), color=REGIME_COLORS[regime])
        bottom += np.array(values)
    axes[0].set_ylabel("Share of runs (%)")
    axes[0].set_title("Regime distribution")
    axes[0].legend(fontsize=8, frameon=True)

    sns.boxplot(data=subset, x="model", y="consensus_error", order=models, ax=axes[1])
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("$E_{cons}$")
    axes[1].set_title("Final consensus error")
    save_figure(fig, "fig05_ablation")


def figure_06_baselines(frame: pd.DataFrame) -> None:
    models = ["proposed", "anti-expert", "inverse-reputation", "stubborn-mix"]
    labels = ["Proposed", "Anti-expert", "Inverse reputation", "Stubborn mix"]
    subset = frame[frame["model"].isin(models)].copy()
    subset["model"] = pd.Categorical(subset["model"], categories=models, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=subset, x="model", y="Var_temp", order=models, ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].axhline(1e-3, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("$\\mathrm{Var}_{temp}$")
    axes[0].set_title("Terminal temporal variance")

    sns.boxplot(data=subset, x="model", y="consensus_error", order=models, ax=axes[1])
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("$E_{cons}$")
    axes[1].set_title("Final consensus error")
    save_figure(fig, "fig06_baselines")


def main() -> None:
    _style()
    frame = load_results()
    figure_01_penalty_profiles()
    figure_02_minimal_phase_portrait()
    figure_03_phase_heatmaps(frame)
    figure_04_kinship_scaling(frame)
    figure_05_ablation(frame)
    figure_06_baselines(frame)
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

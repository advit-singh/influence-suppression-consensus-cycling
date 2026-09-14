"""Validate the raw simulation CSV against expected grid coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.data_utils import load_results, repo_root, save_derived
from code.config import CHECKPOINT_JSON, OUTPUT_CSV


def expected_counts() -> dict[str, int]:
    return {
        "proposed": 122_880,
        "symmetric": 122_880,
        "fixed_suppression": 122_880,
        "anti-expert": 1_920,
        "inverse-reputation": 1_920,
        "stubborn-mix": 480,
    }


def main() -> None:
    frame = load_results()
    expected = expected_counts()

    coverage_rows = []
    for model, target in expected.items():
        actual = int((frame["model"] == model).sum())
        coverage_rows.append(
            {
                "model": model,
                "expected_runs": target,
                "actual_runs": actual,
                "coverage_pct": round(100 * actual / target, 2),
                "missing_runs": target - actual,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    save_derived(coverage, "coverage_report.csv")

    regime = (
        frame["regime"]
        .value_counts(dropna=False)
        .rename_axis("regime")
        .reset_index(name="count")
    )
    regime["pct"] = (regime["count"] / len(frame) * 100).round(2)
    save_derived(regime, "regime_breakdown.csv")

    report_lines = [
        f"Validated rows: {len(frame):,}",
        f"Source CSV: {OUTPUT_CSV}",
        "",
        "Coverage by model:",
        coverage.to_string(index=False),
        "",
        "Global regime distribution:",
        regime.to_string(index=False),
    ]

    if CHECKPOINT_JSON.exists():
        with open(CHECKPOINT_JSON, encoding="utf-8") as handle:
            checkpoint_size = len(json.load(handle))
        report_lines.extend(
            [
                "",
                f"Checkpoint entries: {checkpoint_size:,}",
                f"Checkpoint minus clean CSV rows: {checkpoint_size - len(frame):,}",
            ]
        )

    report_path = repo_root() / "data" / "derived" / "validation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

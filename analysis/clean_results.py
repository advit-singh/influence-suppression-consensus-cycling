"""Clean corrupted rows and deduplicate simulation output."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from config import OUTPUT_CSV  # noqa: E402
from simulation import get_param_hash  # noqa: E402

VALID_MODELS = {
    "proposed",
    "symmetric",
    "fixed_suppression",
    "anti-expert",
    "inverse-reputation",
    "stubborn-mix",
}
SHAPE_MAP = {"Linear": 0, "Step": 1, "Tanh": 2, "None": 0}


def row_hash(row) -> str:
    shape = SHAPE_MAP.get(str(row.penalty_shape).strip(), 0)
    return get_param_hash(
        row.model,
        row.topology,
        int(row.N),
        float(row.P),
        float(row.gamma_stub),
        shape,
        str(row.K_fraction).strip(),
        int(row.seed),
    )


def main() -> None:
    frame = pd.read_csv(OUTPUT_CSV)
    before = len(frame)

    frame["penalty_shape"] = frame["penalty_shape"].fillna("None").astype(str).str.strip()
    frame["K_fraction"] = frame["K_fraction"].fillna("None").astype(str).str.strip()
    frame = frame.dropna(subset=["model", "topology", "N", "P", "gamma_stub", "seed"])
    frame = frame[frame["model"].isin(VALID_MODELS)]
    frame = frame[frame["topology"].isin(["ER", "BA", "WS", "k-regular"])]

    frame["_hash"] = [row_hash(row) for row in frame.itertuples(index=False)]
    frame = frame.drop_duplicates(subset="_hash", keep="last").drop(columns="_hash")
    frame = frame.sort_values(["model", "topology", "N", "P", "gamma_stub", "seed"]).reset_index(drop=True)

    backup = OUTPUT_CSV.with_suffix(".backup.csv")
    cleaned = OUTPUT_CSV.with_suffix(".clean.csv")
    shutil.copy2(OUTPUT_CSV, backup)
    frame.to_csv(cleaned, index=False)
    shutil.copy2(cleaned, OUTPUT_CSV)

    print(f"Cleaned CSV: {before:,} -> {len(frame):,} rows")
    print(frame["model"].value_counts())


if __name__ == "__main__":
    main()

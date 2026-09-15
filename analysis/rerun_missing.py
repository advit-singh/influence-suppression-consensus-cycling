"""Re-run parameter configurations missing from the results CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from config import CHECKPOINT_JSON, OUTPUT_CSV  # noqa: E402
from simulation import (  # noqa: E402
    append_results_to_csv,
    build_pruned_grid,
    get_param_hash,
    load_checkpoint,
    log,
    run_single_simulation,
    save_checkpoint_atomic,
)

SHAPE_MAP = {"Linear": 0, "Step": 1, "Tanh": 2}


def existing_hashes() -> set[str]:
    frame = pd.read_csv(OUTPUT_CSV)
    frame["penalty_shape"] = frame["penalty_shape"].fillna("None").astype(str).str.strip()
    frame["K_fraction"] = frame["K_fraction"].fillna("None").astype(str).str.strip()
    frame = frame.dropna(subset=["model", "topology", "N", "P", "gamma_stub", "seed"])
    frame = frame[frame["model"].isin({"proposed", "symmetric", "fixed_suppression", "anti-expert", "inverse-reputation", "stubborn-mix"})]

    hashes = set()
    for row in frame.itertuples(index=False):
        shape = SHAPE_MAP.get(str(row.penalty_shape).strip(), 0)
        hashes.add(
            get_param_hash(
                row.model,
                row.topology,
                int(row.N),
                float(row.P),
                float(row.gamma_stub),
                shape,
                str(row.K_fraction).strip(),
                int(row.seed),
            )
        )
    return hashes


def missing_runs(model: str = "symmetric") -> list[tuple]:
    present = existing_hashes()
    missing = []
    for params in build_pruned_grid():
        if params[0] != model:
            continue
        if get_param_hash(*params) not in present:
            missing.append(params)
    return missing


def main() -> None:
    missing = missing_runs("symmetric")
    log(f"Missing symmetric configurations: {len(missing):,}")
    if not missing:
        print("No missing symmetric runs.")
        return

    log("Warming up numba kernels")
    run_single_simulation("symmetric", "ER", 20, 0.1, 0.1, 0, "N/10", 42, warm_up=True)

    batch_size = 500
    completed = load_checkpoint()
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        log(f"Running batch {start // batch_size + 1} ({len(batch)} configs)")
        results = Parallel(n_jobs=-1)(delayed(run_single_simulation)(*params) for params in batch)
        append_results_to_csv(results)
        for params in batch:
            completed.add(get_param_hash(*params))

    try:
        save_checkpoint_atomic(completed)
    except OSError as exc:
        log(f"Warning: could not update checkpoint ({exc})")

    log(f"Finished backfill. Checkpoint size: {len(completed):,}")


if __name__ == "__main__":
    main()

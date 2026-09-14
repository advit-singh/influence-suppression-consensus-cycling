"""Shared helpers for loading and cleaning simulation output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from code.config import DATA_DIR, DERIVED_DIR, OUTPUT_CSV

BAD_MODELS = {"9135", "s)"}
VALID_TOPOLOGIES = {"ER", "BA", "WS", "k-regular"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_results(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or OUTPUT_CSV
    frame = pd.read_csv(path)
    frame.columns = [column.strip().replace(r"\_", "_") for column in frame.columns]
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[~frame["model"].isin(BAD_MODELS)].copy()
    frame = frame[frame["topology"].isin(VALID_TOPOLOGIES)].copy()
    return frame


def ensure_derived_dir() -> Path:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    return DERIVED_DIR


def save_derived(frame: pd.DataFrame, name: str) -> Path:
    ensure_derived_dir()
    path = DERIVED_DIR / name
    frame.to_csv(path, index=False)
    return path

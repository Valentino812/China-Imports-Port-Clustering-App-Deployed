"""Data preparation and analysis loading for the deployed application.

Two responsibilities live here:

* :func:`run_fixed_analysis` runs the complete preprocessing + modelling
  pipeline and is used **only** by the offline artifact builder.
* :func:`get_precomputed_analysis` loads the precomputed artifact directory
  with ``st.cache_data``. The Streamlit pages never train a model at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.artifacts import (
    DEFAULT_ARTIFACT_DIR,
    load_analysis_artifacts,
)
from src.config import FIXED_K
from src.modeling import run_modeling
from src.preprocessing import (
    CONVERSION_REQUIRED,
    GEOLOCATION_REQUIRED,
    IMPORTS_REQUIRED,
    PROVINCES_REQUIRED,
    align_columns,
    read_table,
    run_preprocessing,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset_input"

DATASET_FILES: dict[str, tuple[str, list[str]]] = {
    "imports": ("port_imports.csv", IMPORTS_REQUIRED),
    "provinces": ("provinces_data.csv", PROVINCES_REQUIRED),
    "conversion": ("HS2022_to_SITC_Conversion.xlsx", CONVERSION_REQUIRED),
    "geolocation": ("port_geolocation.csv", GEOLOCATION_REQUIRED),
}


def _read_bundled(kind: str) -> pd.DataFrame:
    filename, required = DATASET_FILES[kind]
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Berkas data bawaan tidak ditemukan: {path}. "
            "Pastikan folder dataset_input tersedia di dalam aplikasi."
        )
    return align_columns(read_table(path.read_bytes(), path.name), required)


def load_bundled_datasets() -> dict[str, pd.DataFrame]:
    """Read and align the four bundled input datasets."""
    return {kind: _read_bundled(kind) for kind in DATASET_FILES}


def run_fixed_analysis(
    n_clusters: int = FIXED_K,
    progress_callback=None,
) -> dict[str, Any]:
    """Run preprocessing and K-Prototypes modelling on the bundled datasets.

    This is an offline build step. The Streamlit runtime uses
    :func:`get_precomputed_analysis` instead.
    """
    frames = load_bundled_datasets()

    pre_result = run_preprocessing(
        frames["imports"],
        frames["provinces"],
        frames["conversion"],
        frames["geolocation"],
    )
    model_result = run_modeling(
        pre_result["df_final"],
        n_clusters=int(n_clusters),
        progress_callback=progress_callback,
    )

    return {
        "preprocessing": pre_result,
        "modeling": model_result,
        "datasets": {
            "imports": DATASET_FILES["imports"][0],
            "provinces": DATASET_FILES["provinces"][0],
            "conversion": DATASET_FILES["conversion"][0],
            "geolocation": DATASET_FILES["geolocation"][0],
        },
        "n_clusters": int(n_clusters),
    }


def load_precomputed_analysis(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Load the precomputed dashboard result without running any modelling."""
    return load_analysis_artifacts(Path(artifact_dir))


@st.cache_data(show_spinner=False)
def get_precomputed_analysis(
    artifact_dir: str = str(DEFAULT_ARTIFACT_DIR),
) -> dict[str, Any]:
    """Return the cached precomputed analysis result (K = 5).

    ``st.cache_data`` only avoids re-reading the artifact files on every
    Streamlit rerun; the analysis itself was already computed offline.
    """
    return load_precomputed_analysis(artifact_dir)

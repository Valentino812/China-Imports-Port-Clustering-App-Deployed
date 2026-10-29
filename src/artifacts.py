"""Read and write precomputed analysis artifacts for the deployed app.

The dashboard only needs tabular results, metrics, and SHAP values. Fitted
estimator objects (K-Prototypes and Random Forest) are intentionally **not**
serialized: keeping artifacts in Parquet/CSV/JSON makes them portable,
inspectable, and independent of library pickle compatibility.

Artifacts are generated once by ``scripts/build_analysis_artifacts.py`` and
loaded read-only by the Streamlit pages.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import FIXED_K

ARTIFACT_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset_input"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / f"k{FIXED_K}"

INPUT_FILES = (
    "port_imports.csv",
    "provinces_data.csv",
    "HS2022_to_SITC_Conversion.xlsx",
    "port_geolocation.csv",
)

BUILD_COMMAND = ".venv/bin/python scripts/build_analysis_artifacts.py --force"


class ArtifactError(RuntimeError):
    """Base class for artifact loading/building failures."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when the precomputed artifact folder is missing."""


class ArtifactValidationError(ArtifactError):
    """Raised when an artifact is incomplete, stale, or incompatible."""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_input_hashes() -> dict[str, str]:
    """Hash the bundled input files that produced the artifact."""
    hashes: dict[str, str] = {}
    for filename in INPUT_FILES:
        path = DATA_DIR / filename
        if path.exists():
            hashes[filename] = sha256_file(path)
    return hashes


def _library_versions() -> dict[str, str]:
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "kmodes",
        "shap",
        "pyarrow",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas values into standard JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ArtifactValidationError(f"Berkas artifact tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=True)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ArtifactValidationError(f"Berkas artifact tidak ditemukan: {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_analysis_artifacts(
    result: dict[str, Any],
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    silhouette_samples: pd.DataFrame | None = None,
) -> Path:
    """Write all dashboard-facing analysis outputs into ``artifact_dir``.

    The write is staged in a temporary sibling directory and moved into place
    only after every file and the manifest have been written successfully.
    """
    artifact_dir = Path(artifact_dir)
    pre = result["preprocessing"]
    modeling = result["modeling"]

    tmp_dir = artifact_dir.parent / f".{artifact_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    (tmp_dir / "preprocessing").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "modeling" / "surrogate").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "modeling" / "shap").mkdir(parents=True, exist_ok=True)

    # Preprocessing results -------------------------------------------------
    preprocessing_frames = {
        "df_imports.parquet": pre["df_imports"],
        "mapping.parquet": pre["mapping"],
        "df_sitc.parquet": pre["df_sitc"],
        "df_aggregated.parquet": pre["df_aggregated"],
        "df_finished.parquet": pre["df_finished"],
        "df_final.parquet": pre["df_final"],
    }
    for filename, frame in preprocessing_frames.items():
        _write_parquet(frame, tmp_dir / "preprocessing" / filename)
    _write_json(tmp_dir / "preprocessing" / "report.json", pre["report"])

    # Core modelling results -------------------------------------------------
    clustered = modeling["clustered"]
    metadata = {
        "n_clusters": int(modeling["config"]["n_clusters"]),
        "n_observations": int(len(clustered)),
        "n_features": int(len(modeling["feature_cols"])),
        "gamma": float(modeling["gamma"]),
        "silhouette": modeling["silhouette"],
        "silhouette_n": int(modeling["silhouette_n"]),
        "numerical_cols": list(modeling["numerical_cols"]),
        "categorical_cols": list(modeling["categorical_cols"]),
        "feature_cols": list(modeling["feature_cols"]),
        "categorical_index": [int(value) for value in modeling["categorical_index"]],
        "config": dict(modeling["config"]),
        "datasets": dict(result.get("datasets", {})),
    }
    _write_json(tmp_dir / "modeling" / "metadata.json", metadata)

    _write_parquet(clustered, tmp_dir / "preprocessing" / "clustered.parquet")
    _write_parquet(modeling["X_model"], tmp_dir / "modeling" / "X_model.parquet")
    modeling["elbow"].to_csv(
        tmp_dir / "modeling" / "elbow.csv", index=False, encoding="utf-8"
    )

    samples = silhouette_samples
    if samples is None:
        samples = modeling.get("silhouette_samples")
    if samples is not None:
        samples.to_csv(
            tmp_dir / "modeling" / "silhouette_samples.csv",
            index=False,
            encoding="utf-8",
        )

    # Surrogate model results -----------------------------------------------
    surrogate = modeling["surrogate"]
    surrogate["metrics"].to_csv(
        tmp_dir / "modeling" / "surrogate" / "metrics.csv",
        index=False,
        encoding="utf-8",
    )
    _write_json(
        tmp_dir / "modeling" / "surrogate" / "classification_report.json",
        surrogate["classification_report"],
    )
    pd.DataFrame(surrogate["confusion_matrix"]).to_csv(
        tmp_dir / "modeling" / "surrogate" / "confusion_matrix.csv",
        index=False,
        header=False,
    )
    _write_json(
        tmp_dir / "modeling" / "surrogate" / "class_labels.json",
        [int(value) for value in surrogate["class_labels"]],
    )

    # SHAP results -----------------------------------------------------------
    shap = modeling.get("shap", {"available": False, "reason": "SHAP tidak tersedia."})
    shap_available = bool(shap.get("available"))
    shap_status = {
        "available": shap_available,
        "reason": shap.get("reason"),
        "classes": [int(value) for value in shap.get("classes", [])],
        "n_explained": int(shap.get("n_explained", 0)) if shap_available else 0,
        "n_features": int(shap.get("n_features", 0)) if shap_available else 0,
    }
    _write_json(tmp_dir / "modeling" / "shap" / "status.json", shap_status)

    if shap_available:
        shap["global_importance"].to_csv(
            tmp_dir / "modeling" / "shap" / "global_importance.csv",
            index=False,
            encoding="utf-8",
        )
        shap["importance_per_cluster"].to_csv(
            tmp_dir / "modeling" / "shap" / "importance_per_cluster.csv",
            index=True,
            index_label="Klaster",
            encoding="utf-8",
        )
        _write_parquet(
            shap["X_eval"], tmp_dir / "modeling" / "shap" / "X_eval.parquet"
        )
        for cluster_id, frame in shap["grouped_by_cluster"].items():
            _write_parquet(
                frame,
                tmp_dir
                / "modeling"
                / "shap"
                / f"grouped_cluster_{int(cluster_id)}.parquet",
            )

    # Manifest ---------------------------------------------------------------
    report = pre["report"]
    cluster_sizes = (
        clustered["Cluster"].value_counts().sort_index().to_dict()
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "app": "China-Imports-Port-Clustering-App-Deployed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "k": int(modeling["config"]["n_clusters"]),
        "python": platform.python_version(),
        "libraries": _library_versions(),
        "input_sha256": current_input_hashes(),
        "summary": {
            "rows_imports": int(report["rows_imports"]),
            "rows_final": int(report["rows_final"]),
            "n_ports": int(clustered["port"].nunique()),
            "n_provinces": int(clustered["Provinsi"].nunique()),
            "cluster_sizes": {
                str(int(key)): int(value) for key, value in cluster_sizes.items()
            },
            "silhouette": modeling["silhouette"],
            "surrogate_fidelity": float(
                surrogate["metrics"].iloc[0]["Nilai"]
            ),
        },
        "files": sorted(
            str(path.relative_to(tmp_dir)).replace("\\", "/")
            for path in tmp_dir.rglob("*")
            if path.is_file()
        ),
    }
    _write_json(tmp_dir / "manifest.json", manifest)

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    tmp_dir.replace(artifact_dir)
    return artifact_dir


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

REQUIRED_FILES = (
    "manifest.json",
    "preprocessing/report.json",
    "preprocessing/df_imports.parquet",
    "preprocessing/mapping.parquet",
    "preprocessing/df_sitc.parquet",
    "preprocessing/df_aggregated.parquet",
    "preprocessing/df_finished.parquet",
    "preprocessing/df_final.parquet",
    "preprocessing/clustered.parquet",
    "modeling/metadata.json",
    "modeling/X_model.parquet",
    "modeling/elbow.csv",
    "modeling/silhouette_samples.csv",
    "modeling/surrogate/metrics.csv",
    "modeling/surrogate/classification_report.json",
    "modeling/surrogate/confusion_matrix.csv",
    "modeling/surrogate/class_labels.json",
    "modeling/shap/status.json",
)


def _validate_manifest(manifest: dict[str, Any], artifact_dir: Path) -> None:
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            "Versi artifact tidak cocok dengan kode aplikasi "
            f"(artifact: {manifest.get('schema_version')}, "
            f"dibutuhkan: {ARTIFACT_SCHEMA_VERSION}). "
            f"Bangun ulang artifact dengan: {BUILD_COMMAND}"
        )
    if int(manifest.get("k", -1)) != FIXED_K:
        raise ArtifactValidationError(
            f"Artifact dibangun dengan K={manifest.get('k')}, sedangkan aplikasi "
            f"menggunakan K={FIXED_K}. Bangun ulang artifact dengan: {BUILD_COMMAND}"
        )

    missing = [
        relative for relative in REQUIRED_FILES
        if not (artifact_dir / relative).exists()
    ]
    if missing:
        raise ArtifactValidationError(
            "Artifact tidak lengkap. Berkas berikut tidak ditemukan: "
            + ", ".join(missing)
            + f". Bangun ulang artifact dengan: {BUILD_COMMAND}"
        )


def _validate_inputs(manifest: dict[str, Any]) -> None:
    expected = manifest.get("input_sha256", {})
    current = current_input_hashes()
    mismatched = [
        filename
        for filename, digest in expected.items()
        if filename in current and current[filename] != digest
    ]
    if mismatched:
        raise ArtifactValidationError(
            "Artifact sudah tidak sesuai dengan berkas pada dataset_input: "
            + ", ".join(mismatched)
            + f". Bangun ulang artifact dengan: {BUILD_COMMAND}"
        )


def load_analysis_artifacts(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    """Load the precomputed result into the structure used by the dashboard."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactNotFoundError(
            f"Artifact analisis tidak ditemukan di {artifact_dir}. "
            f"Bangun artifact terlebih dahulu dengan: {BUILD_COMMAND}"
        )

    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, artifact_dir)
    if verify_inputs:
        _validate_inputs(manifest)

    pre = {
        "df_imports": _read_parquet(artifact_dir / "preprocessing" / "df_imports.parquet"),
        "mapping": _read_parquet(artifact_dir / "preprocessing" / "mapping.parquet"),
        "df_sitc": _read_parquet(artifact_dir / "preprocessing" / "df_sitc.parquet"),
        "df_aggregated": _read_parquet(
            artifact_dir / "preprocessing" / "df_aggregated.parquet"
        ),
        "df_finished": _read_parquet(
            artifact_dir / "preprocessing" / "df_finished.parquet"
        ),
        "df_final": _read_parquet(artifact_dir / "preprocessing" / "df_final.parquet"),
        "report": _read_json(artifact_dir / "preprocessing" / "report.json"),
    }

    metadata = _read_json(artifact_dir / "modeling" / "metadata.json")
    clustered = _read_parquet(artifact_dir / "preprocessing" / "clustered.parquet")
    X_model = _read_parquet(artifact_dir / "modeling" / "X_model.parquet")
    elbow = pd.read_csv(artifact_dir / "modeling" / "elbow.csv")
    silhouette_samples = pd.read_csv(
        artifact_dir / "modeling" / "silhouette_samples.csv"
    )

    surrogate = {
        "metrics": pd.read_csv(
            artifact_dir / "modeling" / "surrogate" / "metrics.csv"
        ),
        "classification_report": _read_json(
            artifact_dir / "modeling" / "surrogate" / "classification_report.json"
        ),
        "confusion_matrix": pd.read_csv(
            artifact_dir / "modeling" / "surrogate" / "confusion_matrix.csv",
            header=None,
        ).to_numpy(),
        "class_labels": [
            int(value)
            for value in _read_json(
                artifact_dir / "modeling" / "surrogate" / "class_labels.json"
            )
        ],
    }

    shap_status = _read_json(artifact_dir / "modeling" / "shap" / "status.json")
    shap: dict[str, Any] = {
        "available": bool(shap_status.get("available")),
        "reason": shap_status.get("reason"),
    }
    if shap["available"]:
        classes = [int(value) for value in shap_status.get("classes", [])]
        shap.update(
            {
                "classes": classes,
                "n_explained": int(shap_status.get("n_explained", 0)),
                "n_features": int(shap_status.get("n_features", 0)),
                "global_importance": pd.read_csv(
                    artifact_dir / "modeling" / "shap" / "global_importance.csv"
                ),
                "importance_per_cluster": pd.read_csv(
                    artifact_dir / "modeling" / "shap" / "importance_per_cluster.csv",
                    index_col=0,
                ),
                "X_eval": _read_parquet(
                    artifact_dir / "modeling" / "shap" / "X_eval.parquet"
                ),
                "grouped_by_cluster": {
                    cluster_id: _read_parquet(
                        artifact_dir
                        / "modeling"
                        / "shap"
                        / f"grouped_cluster_{cluster_id}.parquet"
                    )
                    for cluster_id in classes
                },
            }
        )

    silhouette = metadata.get("silhouette")
    if silhouette is None:
        silhouette = float("nan")

    modeling = {
        "clustered": clustered,
        "labels": clustered["Cluster"].to_numpy(dtype=int),
        "gamma": float(metadata["gamma"]),
        "X_model": X_model,
        "numerical_cols": list(metadata["numerical_cols"]),
        "categorical_cols": list(metadata["categorical_cols"]),
        "feature_cols": list(metadata["feature_cols"]),
        "categorical_index": list(metadata["categorical_index"]),
        "elbow": elbow,
        "silhouette": float(silhouette),
        "silhouette_n": int(metadata["silhouette_n"]),
        "silhouette_samples": silhouette_samples,
        "surrogate": surrogate,
        "shap": shap,
        "config": dict(metadata["config"]),
    }

    return {
        "preprocessing": pre,
        "modeling": modeling,
        "datasets": dict(metadata.get("datasets", {})),
        "n_clusters": int(metadata["n_clusters"]),
        "artifact": {
            "path": str(artifact_dir),
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": manifest.get("created_at"),
        },
    }

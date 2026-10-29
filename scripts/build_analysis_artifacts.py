#!/usr/bin/env python3
"""Build the precomputed analysis artifacts used by the Streamlit dashboard.

Run once (or after changing ``dataset_input``):

    .venv/bin/python scripts/build_analysis_artifacts.py --force

The script runs the complete preprocessing + K-Prototypes pipeline, computes
per-observation silhouette values, writes the Parquet/CSV/JSON artifact
directory, then reloads it to verify the result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import run_fixed_analysis  # noqa: E402
from src.artifacts import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    load_analysis_artifacts,
    write_analysis_artifacts,
)
from src.config import FIXED_K  # noqa: E402
from src.modeling import SILHOUETTE_SAMPLE_SIZE  # noqa: E402
from src.visualization import kprototypes_silhouette_samples  # noqa: E402


def _progress(fraction: float, message: str) -> None:
    print(f"[{fraction * 100:5.1f}%] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bangun artifact analisis precomputed untuk dashboard."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Folder artifact tujuan (default: {DEFAULT_ARTIFACT_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Timpa artifact yang sudah ada.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and not args.force:
        print(
            f"Artifact sudah ada di {output}. Gunakan --force untuk menimpanya.",
            file=sys.stderr,
        )
        return 1

    started = time.time()
    print(f"Membangun artifact K={FIXED_K} dari dataset_input ...", flush=True)
    result = run_fixed_analysis(FIXED_K, progress_callback=_progress)
    modeling = result["modeling"]

    print("Menghitung silhouette per observasi ...", flush=True)
    silhouette_samples = kprototypes_silhouette_samples(
        modeling["X_model"],
        modeling["labels"],
        float(modeling["gamma"]),
        list(modeling["numerical_cols"]),
        list(modeling["categorical_cols"]),
        sample_size=SILHOUETTE_SAMPLE_SIZE,
        random_state=42,
    )

    print(f"Menulis artifact ke {output} ...", flush=True)
    artifact_dir = write_analysis_artifacts(
        result, output, silhouette_samples=silhouette_samples
    )

    print("Memverifikasi artifact ...", flush=True)
    loaded = load_analysis_artifacts(artifact_dir)
    clustered = loaded["modeling"]["clustered"]
    report = loaded["preprocessing"]["report"]
    elapsed = time.time() - started
    print(
        "\nSelesai.\n"
        f"- Folder artifact : {artifact_dir}\n"
        f"- Waktu build     : {elapsed:.1f} detik\n"
        f"- Observasi final : {report['rows_final']:,}\n"
        f"- Pelabuhan unik  : {clustered['port'].nunique():,}\n"
        f"- Provinsi        : {clustered['Provinsi'].nunique():,}\n"
        f"- Klaster (K)     : {loaded['n_clusters']}\n"
        f"- Silhouette      : {loaded['modeling']['silhouette']:.4f}\n"
        f"- Surrogate       : "
        f"{loaded['modeling']['surrogate']['metrics'].iloc[0]['Nilai']:.4f}\n"
        f"- Silhouette rows : {len(loaded['modeling']['silhouette_samples']):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

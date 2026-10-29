"""Modelling, evaluation, surrogate explanation and SHAP utilities.

This module is a Python translation of ``k-prototypes.ipynb``:

* K-Prototypes with ``init="Huang"``, ``n_init=5``, ``random_state=42`` and
  ``n_jobs=-1``.
* Elbow curve for ``K`` in 2..8 using ``n_init=3`` (as in the notebook).
* Silhouette computed with the mixed K-Prototypes dissimilarity
  (squared Euclidean distance on standardised numeric features plus ``gamma``
  times the categorical mismatch count).
* A Random Forest surrogate with fidelity metrics, and SHAP explanations of the
  surrogate predictions.

The UI starts from ``DEFAULT_K = 4``. The evaluated optimum is ``OPTIMAL_K = 5``
and is selected explicitly through the automatic-K option; the two values are
intentionally different.

The numeric features are standardised exactly once, here, so the frames coming
from :mod:`src.preprocessing` remain in their original units.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DEFAULT_K,
    K_RANGE,
)
from src.preprocessing import CATEGORICAL_COLS, FEATURE_COLS, NUMERICAL_COLS

SILHOUETTE_SAMPLE_SIZE = 2500


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_model_matrix(
    df_raw: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Standardise numeric features once and build the object array for kmodes."""
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    numerical_cols = numerical_cols or NUMERICAL_COLS
    feature_cols = categorical_cols + numerical_cols

    missing = [c for c in feature_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan pada data: {missing}")

    X_model = df_raw[feature_cols].copy()
    X_model[numerical_cols] = X_model[numerical_cols].apply(pd.to_numeric, errors="coerce")
    if X_model[feature_cols].isna().any().any():
        raise ValueError(
            "Masih ada nilai kosong atau gagal dikonversi pada fitur. "
            "Periksa kembali tahap preprocessing."
        )

    scaler = StandardScaler()
    X_model[numerical_cols] = scaler.fit_transform(X_model[numerical_cols])
    X_model[categorical_cols] = X_model[categorical_cols].astype(str)

    categorical_index = [X_model.columns.get_loc(col) for col in categorical_cols]
    data_array = X_model.to_numpy(dtype=object)

    return {
        "X_model": X_model,
        "scaler": scaler,
        "categorical_index": categorical_index,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "feature_cols": feature_cols,
        "data_array": data_array,
    }


# ---------------------------------------------------------------------------
# K-Prototypes
# ---------------------------------------------------------------------------

def compute_elbow(
    data_array: np.ndarray,
    categorical_index: list[int],
    k_range: range | list[int] = K_RANGE,
    random_state: int = 42,
    progress_callback=None,
) -> pd.DataFrame:
    """Fit K-Prototypes for each candidate K and record the cost (Elbow)."""
    rows: list[dict[str, Any]] = []
    candidates = list(k_range)
    for i, k in enumerate(candidates):
        model = KPrototypes(
            n_clusters=k,
            init="Huang",
            n_init=3,
            verbose=0,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit_predict(data_array, categorical=categorical_index)
        rows.append({"K": int(k), "Cost": float(model.cost_)})
        if progress_callback is not None:
            progress_callback((i + 1) / max(1, len(candidates)), f"Elbow K={k}")
    return pd.DataFrame(rows).sort_values("K").reset_index(drop=True)


def fit_final_kprototypes(
    data_array: np.ndarray,
    categorical_index: list[int],
    n_clusters: int = DEFAULT_K,
    random_state: int = 42,
) -> tuple[KPrototypes, np.ndarray]:
    """Train the final K-Prototypes model with the notebook configuration."""
    model = KPrototypes(
        n_clusters=n_clusters,
        init="Huang",
        n_init=5,
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(data_array, categorical=categorical_index)
    return model, labels.astype(int)


def kprototypes_silhouette(
    X_model: pd.DataFrame,
    labels: np.ndarray,
    gamma: float,
    numerical_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    random_state: int = 42,
) -> tuple[float, int]:
    """Silhouette with the mixed K-Prototypes dissimilarity.

    Returns the score and the number of observations it was computed on. When
    the dataset is larger than ``sample_size`` a deterministic sample is drawn
    (documented by the returned count).
    """
    numerical_cols = numerical_cols or NUMERICAL_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    numeric_positions = [X_model.columns.get_loc(c) for c in numerical_cols]
    categorical_positions = [X_model.columns.get_loc(c) for c in categorical_cols]

    arr = X_model.to_numpy(dtype=object)

    def dissimilarity(a, b) -> float:
        a_num = np.asarray([float(a[p]) for p in numeric_positions], dtype=float)
        b_num = np.asarray([float(b[p]) for p in numeric_positions], dtype=float)
        numeric_part = float(np.sum((a_num - b_num) ** 2))
        categorical_part = sum(a[p] != b[p] for p in categorical_positions)
        return numeric_part + gamma * float(categorical_part)

    n = len(arr)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= n:
        return float("nan"), n

    use_sample = n > sample_size
    score = silhouette_score(
        arr,
        labels,
        metric=dissimilarity,
        sample_size=sample_size if use_sample else None,
        random_state=random_state if use_sample else None,
    )
    return float(score), (sample_size if use_sample else n)


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

def train_surrogate(
    df_raw: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the Random Forest surrogate and compute fidelity metrics."""
    X_raw = df_raw[FEATURE_COLS].copy()
    X_raw["sitc"] = X_raw["sitc"].astype(str)
    X_surrogate = pd.get_dummies(
        X_raw, columns=["sitc"], prefix=["SITC"], prefix_sep="_", dtype=float
    )
    y_surrogate = pd.Series(labels, index=df_raw.index).astype(int)

    class_counts = y_surrogate.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Setidaknya ada klaster dengan kurang dari dua baris sehingga stratified "
            "train-test split tidak dapat dilakukan dengan aman."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_surrogate,
        y_surrogate,
        test_size=test_size,
        random_state=random_state,
        stratify=y_surrogate,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = pd.DataFrame(
        {
            "Metrik": [
                "Accuracy / fidelity",
                "Balanced accuracy",
                "Macro F1",
                "Cohen's kappa",
            ],
            "Nilai": [
                float(accuracy_score(y_test, y_pred)),
                float(balanced_accuracy_score(y_test, y_pred)),
                float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                float(cohen_kappa_score(y_test, y_pred)),
            ],
        }
    )
    report = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )
    class_labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    return {
        "model": model,
        "X_surrogate": X_surrogate,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "classification_report": report,
        "class_labels": class_labels,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# SHAP explanation of the surrogate
# ---------------------------------------------------------------------------

def _get_shap_for_class(shap_output, class_index, n_samples, n_features, n_classes):
    if isinstance(shap_output, list):
        return np.asarray(shap_output[class_index])

    values = np.asarray(shap_output)
    if values.ndim == 2:
        if n_classes == 2 and values.shape == (n_samples, n_features):
            return values
        if values.shape == (n_samples, n_features):
            return values
        raise ValueError(f"Format SHAP 2D yang tidak diharapkan: {values.shape}")
    if values.ndim == 3:
        if values.shape == (n_samples, n_features, n_classes):
            return values[:, :, class_index]
        if values.shape == (n_classes, n_samples, n_features):
            return values[class_index, :, :]
    raise ValueError(f"Format SHAP tidak dikenali: bentuk {values.shape}")


def compute_shap(
    surrogate: dict[str, Any],
    feature_cols: list[str] | None = None,
    background_size: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute and group SHAP values. Returns ``available=False`` on failure."""
    feature_cols = feature_cols or FEATURE_COLS
    try:
        import shap
    except Exception as exc:  # pragma: no cover - import guard
        return {"available": False, "reason": f"Paket SHAP tidak tersedia: {exc}"}

    try:
        model = surrogate["model"]
        X_train = surrogate["X_train"]
        X_test = surrogate["X_test"]
        classes = list(model.classes_)

        background = X_train.sample(
            n=min(background_size, len(X_train)), random_state=random_state
        )
        # Keep SHAP values on the predicted-probability scale so the dashboard's
        # positive/negative interpretation matches the model output.
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        shap_raw = explainer.shap_values(X_test, check_additivity=False)

        shap_by_class = {
            int(cluster_id): _get_shap_for_class(
                shap_raw,
                class_index=i,
                n_samples=len(X_test),
                n_features=X_test.shape[1],
                n_classes=len(classes),
            )
            for i, cluster_id in enumerate(classes)
        }

        sitc_dummy_cols = [c for c in X_test.columns if c.startswith("SITC_")]
        if not sitc_dummy_cols:
            return {
                "available": False,
                "reason": "Kolom one-hot SITC tidak ditemukan pada fitur surrogate.",
            }
        sitc_positions = [X_test.columns.get_loc(c) for c in sitc_dummy_cols]

        def group_to_original(encoded: np.ndarray) -> pd.DataFrame:
            grouped = pd.DataFrame(index=X_test.index)
            grouped["sitc"] = encoded[:, sitc_positions].sum(axis=1)
            for feature in NUMERICAL_COLS:
                if feature in X_test.columns:
                    grouped[feature] = encoded[:, X_test.columns.get_loc(feature)]
            return grouped[feature_cols]

        grouped_by_cluster = {
            cluster_id: group_to_original(values)
            for cluster_id, values in shap_by_class.items()
        }

        importance_per_cluster = pd.DataFrame(
            {
                f"Cluster {cluster_id}": contributions.abs().mean(axis=0)
                for cluster_id, contributions in grouped_by_cluster.items()
            }
        ).T
        importance_per_cluster = importance_per_cluster.reindex(columns=feature_cols)

        global_importance = (
            importance_per_cluster.mean(axis=0)
            .sort_values(ascending=False)
            .rename("Mean absolute SHAP")
            .reset_index()
            .rename(columns={"index": "Fitur"})
        )

        return {
            "available": True,
            "classes": classes,
            "shap_by_class": shap_by_class,
            "grouped_by_cluster": grouped_by_cluster,
            "importance_per_cluster": importance_per_cluster,
            "global_importance": global_importance,
            "X_eval": X_test,
            "n_explained": int(len(X_test)),
            "n_features": int(X_test.shape[1]),
        }
    except Exception as exc:
        return {"available": False, "reason": f"Perhitungan SHAP gagal: {exc}"}


# ---------------------------------------------------------------------------
# Full modelling pipeline
# ---------------------------------------------------------------------------

def run_modeling(
    df_final: pd.DataFrame,
    n_clusters: int = DEFAULT_K,
    progress_callback=None,
) -> dict[str, Any]:
    """Prepare features, fit K-Prototypes, evaluate, explain and profile."""
    if n_clusters < 2:
        raise ValueError("Jumlah klaster minimal 2.")

    prepared = prepare_model_matrix(df_final)
    data_array = prepared["data_array"]
    categorical_index = prepared["categorical_index"]

    def _report(fraction: float, text: str) -> None:
        if progress_callback is not None:
            progress_callback(fraction, text)

    _report(0.05, "Menyiapkan matriks fitur")
    elbow = compute_elbow(data_array, categorical_index, progress_callback=progress_callback)

    _report(0.45, f"Melatih K-Prototypes (K={n_clusters})")
    model, labels = fit_final_kprototypes(
        data_array, categorical_index, n_clusters=n_clusters
    )
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("K-Prototypes hanya membentuk satu klaster. Periksa variasi data.")

    _report(0.60, "Menghitung Silhouette campuran")
    gamma = float(getattr(model, "gamma", 1.0))
    silhouette, silhouette_n = kprototypes_silhouette(
        prepared["X_model"], labels, gamma
    )

    _report(0.70, "Melatih surrogate Random Forest")
    surrogate = train_surrogate(df_final, labels)

    _report(0.85, "Menghitung nilai SHAP")
    shap_result = compute_shap(surrogate)

    clustered = df_final.copy()
    clustered["Cluster"] = labels

    _report(1.0, "Menyelesaikan analisis")
    return {
        "clustered": clustered,
        "labels": labels,
        "model": model,
        "gamma": gamma,
        "X_model": prepared["X_model"],
        "scaler": prepared["scaler"],
        "categorical_index": categorical_index,
        "numerical_cols": prepared["numerical_cols"],
        "categorical_cols": prepared["categorical_cols"],
        "feature_cols": prepared["feature_cols"],
        "elbow": elbow,
        "silhouette": silhouette,
        "silhouette_n": silhouette_n,
        "surrogate": surrogate,
        "shap": shap_result,
        "config": {
            "init": "Huang",
            "n_init": 5,
            "random_state": 42,
            "n_jobs": -1,
            "n_clusters": int(n_clusters),
        },
    }

"""Plotly / Matplotlib figure builders for the dashboard views.

Keeping every figure constructor here keeps ``pages/3_Dashboard.py`` focused on
layout and state handling. Data is never fabricated here: figures reflect the
actual analysis output.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import pairwise_distances, silhouette_samples

# Palet kualitatif kontras tinggi dan colorblind-safe (Okabe-Ito). Dipakai untuk
# kategori diskret agar setiap kelompok mendapat warna yang jelas berbeda.
QUALITATIVE_SEQUENCE = [
    "#0072B2",  # biru
    "#E69F00",  # kuning/oranye
    "#009E73",  # hijau
    "#D55E00",  # merah bata
    "#CC79A7",  # magenta
    "#56B4E9",  # biru muda
    "#F0E442",  # kuning
]


def _cluster_sort_key(label: str) -> tuple[int, str]:
    """Sort cluster labels numerically when they contain a number."""
    text = str(label)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (0, int(digits)) if digits else (1, text)


def _cluster_label(value: object) -> str:
    text = str(value)
    return text if text.startswith("Cluster ") else f"Cluster {text}"


def cluster_color_map(cluster_values) -> dict[str, str]:
    """Map every cluster to one consistent qualitative colour.

    All cluster-based figures share this mapping so the same cluster always
    receives the same colour across the dashboard.
    """
    labels = sorted({_cluster_label(value) for value in cluster_values}, key=_cluster_sort_key)
    return {
        label: QUALITATIVE_SEQUENCE[i % len(QUALITATIVE_SEQUENCE)]
        for i, label in enumerate(labels)
    }


def cluster_numeric_stats(
    data: pd.DataFrame,
    numerical_cols: list[str],
) -> pd.DataFrame:
    """Descriptive per-cluster statistics shared by tables and figures."""
    columns = [
        "Klaster",
        "Fitur",
        "Jumlah",
        "Mean",
        "Std",
        "Min",
        "Q1",
        "Median",
        "Q3",
        "Max",
        "IQR",
        "Jumlah Pencilan",
    ]
    if data is None or data.empty or "Cluster" not in data.columns:
        return pd.DataFrame(columns=columns)
    missing = [col for col in numerical_cols if col not in data.columns]
    if missing:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for cluster, group in data.groupby("Cluster", sort=True):
        label = _cluster_label(cluster)
        for feature in numerical_cols:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            count = int(len(values))
            if count == 0:
                rows.append(
                    {
                        "Klaster": label,
                        "Fitur": feature,
                        "Jumlah": 0,
                        "Mean": float("nan"),
                        "Std": float("nan"),
                        "Min": float("nan"),
                        "Q1": float("nan"),
                        "Median": float("nan"),
                        "Q3": float("nan"),
                        "Max": float("nan"),
                        "IQR": float("nan"),
                        "Jumlah Pencilan": 0,
                    }
                )
                continue

            q1 = float(values.quantile(0.25))
            median = float(values.median())
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = values[(values < lower) | (values > upper)]
            std = float(values.std(ddof=1)) if count > 1 else float("nan")

            rows.append(
                {
                    "Klaster": label,
                    "Fitur": feature,
                    "Jumlah": count,
                    "Mean": float(values.mean()),
                    "Std": std,
                    "Min": float(values.min()),
                    "Q1": q1,
                    "Median": median,
                    "Q3": q3,
                    "Max": float(values.max()),
                    "IQR": float(iqr),
                    "Jumlah Pencilan": int(len(outliers)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6B6B6B"),
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, sans-serif", color="#1A1C1C"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    return fig


def cluster_size_bar(cluster_structure: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    if cluster_structure.empty:
        return _empty_figure("Tidak ada data klaster untuk ditampilkan.")
    df = cluster_structure.copy()
    df["Label"] = df["Cluster"].apply(_cluster_label)
    colors = cluster_color_map(df["Label"])
    fig = px.bar(
        df,
        x="Label",
        y=value_col,
        text=value_col,
        color="Label",
        color_discrete_map=colors,
        category_orders={"Label": list(colors)},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="Klaster", yaxis_title=value_col)
    return _style(fig, title=title)


def cluster_pie(cluster_structure: pd.DataFrame) -> go.Figure:
    if cluster_structure.empty:
        return _empty_figure("Tidak ada data klaster untuk ditampilkan.")
    df = cluster_structure.copy()
    df["Label"] = df["Cluster"].apply(_cluster_label)
    colors = cluster_color_map(df["Label"])
    fig = px.pie(
        df,
        names="Label",
        values="jumlah_kombinasi_Port_SITC",
        hole=0.45,
        color="Label",
        color_discrete_map=colors,
        category_orders={"Label": list(colors)},
    )
    fig.update_traces(textinfo="percent+label")
    return _style(fig, title="Proporsi Klaster")


def _format_number(value: object, number_format: str) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return format(float(value), number_format)
    except (TypeError, ValueError):
        return str(value)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _format_bar_label(value: object, number_format: str) -> str:
    """Format a bar label without scientific notation or trailing zeros."""
    text = _format_number(value, number_format)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def numerical_mean_bars(
    data: pd.DataFrame,
    numerical_cols: list[str],
    number_formats: dict[str, str] | None = None,
    feature_labels: dict[str, str] | None = None,
) -> go.Figure:
    if not numerical_cols:
        return _empty_figure("Tidak ada fitur numerik.")
    formats = number_formats or {}
    labels = feature_labels or {}
    means = data.groupby("Cluster")[numerical_cols].mean().reset_index()
    long_df = means.melt(id_vars="Cluster", var_name="Fitur", value_name="Rata-rata")
    long_df["Label"] = long_df["Cluster"].apply(_cluster_label)
    long_df["Teks"] = [
        _format_bar_label(value, formats.get(feature, ",.2f"))
        for value, feature in zip(long_df["Rata-rata"], long_df["Fitur"])
    ]
    long_df["Fitur"] = long_df["Fitur"].map(lambda feature: labels.get(feature, feature))
    colors = cluster_color_map(long_df["Label"])
    fig = px.bar(
        long_df,
        x="Label",
        y="Rata-rata",
        color="Label",
        facet_col="Fitur",
        facet_col_wrap=2,
        color_discrete_map=colors,
        category_orders={"Label": list(colors)},
        text="Teks",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    return _style(fig, height=420, title="Rata-rata Fitur Numerik per Klaster")


def _minimal_feature_stats(values: np.ndarray) -> dict[str, object]:
    """Compute fallback statistics when the shared stats table has no row."""
    series = pd.Series(values)
    count = int(len(series))
    q1 = float(series.quantile(0.25))
    median = float(series.median())
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {
        "count": count,
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if count > 1 else float("nan"),
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "outliers": int(((series < lower) | (series > upper)).sum()),
    }


def _stats_from_row(stats_row: pd.Series) -> dict[str, object]:
    return {
        "count": int(stats_row["Jumlah"]),
        "mean": stats_row["Mean"],
        "std": stats_row["Std"],
        "q1": stats_row["Q1"],
        "median": stats_row["Median"],
        "q3": stats_row["Q3"],
        "iqr": stats_row["IQR"],
        "outliers": int(stats_row["Jumlah Pencilan"]),
    }


def _distribution_customdata(
    stats_values: dict[str, object],
    number_format: str,
    count: int,
) -> list[list[object]]:
    row = [
        stats_values["count"],
        _format_number(stats_values["mean"], number_format),
        _format_number(stats_values["std"], number_format),
        _format_number(stats_values["q1"], number_format),
        _format_number(stats_values["median"], number_format),
        _format_number(stats_values["q3"], number_format),
        _format_number(stats_values["iqr"], number_format),
        stats_values["outliers"],
    ]
    return [row.copy() for _ in range(count)]


def _box_trace(
    label: str,
    values: np.ndarray,
    color: str,
    customdata: list[list[object]],
    number_format: str,
) -> go.Box:
    return go.Box(
        x=[label] * len(values),
        y=values,
        name=label,
        marker_color=color,
        line_color=color,
        boxpoints="outliers",
        jitter=0.3,
        pointpos=0,
        whiskerwidth=0.5,
        quartilemethod="linear",
        showlegend=False,
        customdata=customdata,
        hovertemplate=(
            f"<b>{label}</b><br>"
            "Q1: %{customdata[3]}<br>"
            "Median: %{customdata[4]}<br>"
            "Q3: %{customdata[5]}<br>"
            "IQR: %{customdata[6]}<br>"
            "Rata-rata: %{customdata[1]}<br>"
            "Standar deviasi: %{customdata[2]}<br>"
            "Jumlah pencilan: %{customdata[7]}<br>"
            "Jumlah observasi: %{customdata[0]}<br>"
            f"Nilai: %{{y:{number_format}}}<extra></extra>"
        ),
    )


def _violin_trace(
    label: str,
    values: np.ndarray,
    color: str,
    customdata: list[list[object]],
    number_format: str,
) -> go.Violin:
    return go.Violin(
        x=[label] * len(values),
        y=values,
        name=label,
        line=dict(color=color, width=1.1),
        fillcolor=_hex_to_rgba(color, 0.28),
        box=dict(visible=False),
        meanline=dict(
            visible=True,
            color=color,
            width=1.2,
        ),
        points="outliers",
        jitter=0.25,
        pointpos=0,
        spanmode="soft",
        width=0.8,
        quartilemethod="linear",
        showlegend=False,
        customdata=customdata,
        hovertemplate=(
            f"<b>{label}</b><br>"
            f"Nilai: %{{y:{number_format}}}<br>"
            "Mean: %{customdata[1]}<br>"
            "Standar deviasi: %{customdata[2]}<br>"
            "Jumlah observasi: %{customdata[0]}<extra></extra>"
        ),
    )


def _distribution_figure(
    data: pd.DataFrame | None,
    stats: pd.DataFrame | None,
    numerical_cols: list[str],
    feature_labels: dict[str, str],
    *,
    trace_builder: Callable[..., go.Box | go.Violin],
    title: str | None,
    log_usd: bool,
    number_formats: dict[str, str] | None,
    kind: str,
) -> go.Figure:
    """Build one shared panel for box or violin distribution traces."""
    if data is None or data.empty or "Cluster" not in data.columns:
        return _empty_figure("Data fitur numerik tidak tersedia.")
    if not numerical_cols:
        return _empty_figure(f"Tidak ada fitur numerik untuk {kind}.")

    missing = [col for col in numerical_cols if col not in data.columns]
    if missing:
        return _empty_figure(
            "Kolom fitur numerik tidak lengkap: " + ", ".join(missing)
        )

    number_formats = number_formats or {}
    cluster_labels = sorted(
        {_cluster_label(value) for value in data["Cluster"].dropna()},
        key=_cluster_sort_key,
    )
    colors = cluster_color_map(cluster_labels)
    rows = max(1, (len(numerical_cols) + 1) // 2)
    cols = 1 if len(numerical_cols) == 1 else 2
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[feature_labels.get(col, col) for col in numerical_cols],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    total_excluded = 0
    added_traces = 0

    for index, feature in enumerate(numerical_cols):
        row = index // 2 + 1
        col = index % 2 + 1
        number_format = number_formats.get(feature, ",.2f")
        df = data[["Cluster", feature]].copy()
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df = df.dropna(subset=[feature])
        use_log = bool(log_usd and feature == "usd_value")
        if use_log:
            total_excluded += int((df[feature] <= 0).sum())
            df = df[df[feature] > 0].copy()

        df["Label"] = df["Cluster"].apply(_cluster_label)

        for label in cluster_labels:
            values = df.loc[df["Label"] == label, feature].to_numpy(dtype=float)
            if len(values) == 0:
                continue

            row_df = (
                stats[(stats["Klaster"] == label) & (stats["Fitur"] == feature)]
                if stats is not None and not stats.empty
                else pd.DataFrame()
            )
            stats_values = (
                _minimal_feature_stats(values)
                if row_df.empty
                else _stats_from_row(row_df.iloc[0])
            )
            customdata = _distribution_customdata(
                stats_values, number_format, len(values)
            )

            fig.add_trace(
                trace_builder(
                    label=label,
                    values=values,
                    color=colors[label],
                    customdata=customdata,
                    number_format=number_format,
                ),
                row=row,
                col=col,
            )
            added_traces += 1

        fig.update_xaxes(
            title_text="Klaster",
            type="category",
            categoryorder="array",
            categoryarray=cluster_labels,
            row=row,
            col=col,
        )
        feature_label = feature_labels.get(feature, feature)
        fig.update_yaxes(
            title_text=f"{feature_label} (skala log)" if use_log else feature_label,
            type="log" if use_log else "linear",
            tickformat=",.0f" if use_log else number_format,
            row=row,
            col=col,
        )

    if added_traces == 0:
        return _empty_figure(f"Data fitur numerik tidak tersedia untuk {kind}.")

    fig.update_annotations(font=dict(size=13))
    fig = _style(fig, height=760, title=title)
    fig.layout.meta = {
        "log_scale": bool(log_usd),
        "excluded_nonpositive": total_excluded,
    }
    return fig


def cluster_boxplots(
    data: pd.DataFrame,
    stats: pd.DataFrame,
    numerical_cols: list[str],
    feature_labels: dict[str, str],
    title: str | None = None,
    log_usd: bool = False,
    number_formats: dict[str, str] | None = None,
) -> go.Figure:
    """Combined boxplot panel for all numeric features."""
    return _distribution_figure(
        data,
        stats,
        numerical_cols,
        feature_labels,
        trace_builder=_box_trace,
        title=title,
        log_usd=log_usd,
        number_formats=number_formats,
        kind="boxplot",
    )


def cluster_violinplots(
    data: pd.DataFrame,
    stats: pd.DataFrame,
    numerical_cols: list[str],
    feature_labels: dict[str, str],
    title: str | None = None,
    log_usd: bool = False,
    number_formats: dict[str, str] | None = None,
) -> go.Figure:
    """Combined violin panel showing distribution density and mean/SD per cluster."""
    return _distribution_figure(
        data,
        stats,
        numerical_cols,
        feature_labels,
        trace_builder=_violin_trace,
        title=title,
        log_usd=log_usd,
        number_formats=number_formats,
        kind="violin plot",
    )


def sitc_stacked_bar(sitc_distribution: pd.DataFrame) -> go.Figure:
    if sitc_distribution.empty:
        return _empty_figure("Tidak ada distribusi SITC untuk ditampilkan.")
    df = sitc_distribution.copy()
    df.index = [f"Cluster {c}" for c in df.index]
    df = df.reset_index().rename(columns={"index": "Klaster"})
    long_df = df.melt(id_vars="Klaster", var_name="SITC", value_name="Persentase")
    long_df["SITC"] = long_df["SITC"].astype(str).apply(lambda s: f"SITC {s}")
    fig = px.bar(
        long_df,
        x="Klaster",
        y="Persentase",
        color="SITC",
        color_discrete_sequence=px.colors.qualitative.Set3,
        text="SITC",
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=10)
    fig.update_layout(barmode="stack", yaxis_title="Proporsi dalam klaster (%)", yaxis_range=[0, 100])
    return _style(fig, height=400, title="Komposisi Kelompok SITC per Klaster")


def cluster_z_heatmap(
    z_profile: pd.DataFrame,
    feature_labels: dict[str, str] | None = None,
) -> go.Figure:
    if z_profile.empty:
        return _empty_figure("Tidak ada profil terstandardisasi untuk ditampilkan.")
    display_profile = z_profile
    if feature_labels:
        display_profile = z_profile.rename(columns=feature_labels)
    fig = px.imshow(
        display_profile,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-2,
        zmax=2,
        text_auto=".2f",
        labels=dict(x="Fitur numerik", y="Klaster", color="Z-score"),
    )
    return _style(fig, height=380, title="Profil Rata-rata Fitur Numerik (Z-score)")


def elbow_line(elbow_df: pd.DataFrame, selected_k: int) -> go.Figure:
    if elbow_df.empty:
        return _empty_figure("Elbow belum tersedia.")
    fig = px.line(
        elbow_df,
        x="K",
        y="Cost",
        markers=True,
        labels={"Cost": "Cost K-Prototypes", "K": "Jumlah klaster (K)"},
    )
    fig.add_vline(x=selected_k, line_dash="dash", line_color="#555555")
    fig.add_annotation(x=selected_k, y=elbow_df["Cost"].max(), text=f"K aktif = {selected_k}")
    fig.update_xaxes(
        tickmode="linear",
        tick0=elbow_df["K"].min(),
        dtick=1,
        range=[elbow_df["K"].min() - 0.2, elbow_df["K"].max() + 0.2],
    )
    return _style(fig, height=380, title="Elbow Method untuk K-Prototypes")


def global_shap_bar(
    global_importance: pd.DataFrame,
    feature_labels: dict[str, str] | None = None,
) -> go.Figure:
    if global_importance is None or global_importance.empty:
        return _empty_figure("Global SHAP importance tidak tersedia.")
    df = global_importance.copy()
    if feature_labels:
        df["Fitur"] = df["Fitur"].map(
            lambda feature: feature_labels.get(feature, feature)
        )
    df = df.sort_values("Mean absolute SHAP")
    fig = px.bar(
        df,
        x="Mean absolute SHAP",
        y="Fitur",
        orientation="h",
        color_discrete_sequence=["#4C6EF5"],
        text="Mean absolute SHAP",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    return _style(fig, height=360, title="Global Feature Importance (SHAP)")


def shap_cluster_heatmap(
    importance_per_cluster: pd.DataFrame,
    feature_labels: dict[str, str] | None = None,
) -> go.Figure:
    if importance_per_cluster is None or importance_per_cluster.empty:
        return _empty_figure("Kontribusi SHAP per kelas klaster tidak tersedia.")
    display_importance = importance_per_cluster
    if feature_labels:
        display_importance = importance_per_cluster.rename(columns=feature_labels)
    fig = px.imshow(
        display_importance,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        text_auto=".4f",
        labels=dict(x="Fitur clustering", y="Kelas target surrogate", color="Mean |SHAP|"),
    )
    return _style(fig, height=380, title="Kontribusi Fitur per Kelas Klaster (SHAP)")


def confusion_heatmap(cm: np.ndarray, labels: list) -> go.Figure:
    if cm is None or len(cm) == 0:
        return _empty_figure("Confusion matrix belum tersedia.")
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        x=[f"Pred {c}" for c in labels],
        y=[f"Aktual {c}" for c in labels],
        labels=dict(x="Prediksi surrogate", y="Label asli K-Prototypes", color="Jumlah"),
    )
    return _style(fig, height=420, title="Confusion Matrix Surrogate")


def shap_beeswarm_figure(shap_values, features: pd.DataFrame, title: str):
    """Build a class-specific SHAP beeswarm for all observations in ``X_eval``.

    The caller passes class-specific SHAP values for the complete evaluation
    set. No filter by actual or predicted cluster membership is applied: each
    point represents one observation from the full test set, the horizontal
    position is that observation's SHAP value for the selected class, and the
    colour encodes the feature value.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(9, 5.2))
    shap.summary_plot(
        shap_values,
        features,
        plot_type="dot",
        max_display=min(15, features.shape[1]),
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def geo_scatter(
    data: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    hover_name: str | None,
    title: str,
    color_col: str = "Cluster",
    legend_title: str = "Klaster",
    hover_labels: dict[str, str] | None = None,
) -> go.Figure:
    """Scatter map whose markers are coloured by a discrete category.

    ``color_col`` is converted to text before plotting. That forces Plotly to use
    a qualitative palette; passing a numeric column (for example the raw
    ``Cluster`` labels) would otherwise produce a continuous colour scale.
    """
    df = data.copy()
    group_col = "__kelompok_warna__"
    df[group_col] = df[color_col].astype(str)
    groups = sorted(df[group_col].unique(), key=_cluster_sort_key)
    color_map = cluster_color_map(groups)

    hover_data: dict[str, object] = {}
    for col in ("Provinsi", "Klaster", "sitc"):
        if col in df.columns and col != color_col:
            hover_data[col] = True
    if "usd_value" in df.columns:
        hover_data["usd_value"] = ":,.2f"

    labels = {group_col: legend_title}
    if hover_labels:
        labels.update(hover_labels)

    fig = px.scatter_geo(
        df,
        lat=lat_col,
        lon=lon_col,
        color=group_col,
        hover_name=hover_name,
        hover_data=hover_data,
        category_orders={group_col: groups},
        color_discrete_map=color_map,
        labels=labels,
        scope="asia",
        projection="natural earth",
        title=title,
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.9, line=dict(width=1, color="white"))
    )
    fig.update_geos(fitbounds="locations", visible=True, showcountries=True, countrycolor="#D0D0D0")
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=50, b=0), legend_title_text=legend_title)
    return fig


def _stratified_sample(
    df: pd.DataFrame, max_points: int, random_state: int
) -> pd.DataFrame:
    """Deterministically sample rows while preserving cluster proportions."""
    counts = df["Cluster"].value_counts()
    max_points = max(int(max_points), int(counts.size))
    if len(df) <= max_points:
        return df

    total = int(counts.sum())
    allocation = {
        cluster: max(1, int(round(count / total * max_points)))
        for cluster, count in counts.items()
    }
    allocation = {
        cluster: min(value, int(counts[cluster]))
        for cluster, value in allocation.items()
    }
    while sum(allocation.values()) > max_points:
        candidates = [cluster for cluster, value in allocation.items() if value > 1]
        if not candidates:
            break
        largest = max(candidates, key=lambda cluster: allocation[cluster])
        allocation[largest] -= 1
    while sum(allocation.values()) < max_points:
        candidates = [
            cluster
            for cluster in allocation
            if allocation[cluster] < int(counts[cluster])
        ]
        if not candidates:
            break
        largest = max(candidates, key=lambda cluster: counts[cluster])
        allocation[largest] += 1

    rng = np.random.RandomState(random_state)
    parts = []
    for cluster, n_rows in allocation.items():
        group = df[df["Cluster"] == cluster]
        if n_rows >= len(group):
            parts.append(group)
            continue
        chosen = np.sort(rng.permutation(len(group))[:n_rows])
        parts.append(group.iloc[chosen])
    return pd.concat(parts).sort_index()


def cluster_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    log_scale: bool = False,
    max_points: int = 5000,
    random_state: int = 42,
) -> go.Figure:
    """Scatter plot of two raw features coloured by the actual cluster label.

    Axis values always come from the original units. ``log_scale`` only changes
    the axis rendering; source values and cluster labels are untouched.
    """
    required = ["port", "Provinsi", "sitc", "Cluster", x_col, y_col]
    missing = [col for col in required if col not in data.columns]
    if missing:
        return _empty_figure("Kolom berikut tidak tersedia: " + ", ".join(missing))

    df = data.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col, "Cluster"])
    if df.empty:
        return _empty_figure("Tidak ada observasi dengan nilai yang valid untuk ditampilkan.")

    n_total = int(len(df))
    excluded_nonpositive = 0
    if log_scale:
        positive = (df[x_col] > 0) & (df[y_col] > 0)
        excluded_nonpositive = int((~positive).sum())
        df = df.loc[positive].copy()
        if df.empty:
            return _empty_figure(
                "Tidak ada observasi dengan nilai positif pada kedua sumbu sehingga "
                "skala logaritmik tidak dapat ditampilkan."
            )

    sampled = len(df) > max_points
    if sampled:
        df = _stratified_sample(df, max_points, random_state)

    df["__klaster__"] = df["Cluster"].apply(_cluster_label)
    colors = cluster_color_map(df["__klaster__"])
    hover_data = {
        "Provinsi": True,
        "sitc": True,
        x_col: ":,.2f",
        y_col: ":,.2f",
    }

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="__klaster__",
        hover_name="port",
        hover_data=hover_data,
        color_discrete_map=colors,
        category_orders={"__klaster__": list(colors)},
        labels={
            x_col: x_label,
            y_col: y_label,
            "sitc": "SITC",
            "__klaster__": "Klaster",
        },
        title=title,
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.65, line=dict(width=0.6, color="white"))
    )

    x_title = f"{x_label} (skala log)" if log_scale else x_label
    y_title = f"{y_label} (skala log)" if log_scale else y_label
    fig.update_xaxes(title_text=x_title, type="log" if log_scale else "linear")
    fig.update_yaxes(title_text=y_title, type="log" if log_scale else "linear")

    fig = _style(fig, height=420, title=title)
    fig.layout.meta = {
        "n_total": n_total,
        "n_plotted": int(len(df)),
        "sampled": bool(sampled),
        "excluded_nonpositive": excluded_nonpositive,
        "log_scale": bool(log_scale),
    }
    return fig


def feature_separation_bar(feature_stats: pd.DataFrame) -> go.Figure:
    if feature_stats is None or feature_stats.empty:
        return _empty_figure("Tidak ada statistik fitur.")
    df = feature_stats.dropna(subset=["Rasio Pemisahan"]).sort_values("Rasio Pemisahan")
    if df.empty:
        return _empty_figure("Rasio pemisahan antar klaster tidak dapat dihitung.")
    fig = px.bar(
        df,
        x="Rasio Pemisahan",
        y="Fitur",
        orientation="h",
        color_discrete_sequence=["#12B886"],
        text="Rasio Pemisahan",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return _style(fig, height=320, title="Rasio Pemisahan Rata-rata Antar Klaster")


def kprototypes_silhouette_samples(
    X_model: pd.DataFrame,
    labels,
    gamma: float,
    numerical_cols: list[str],
    categorical_cols: list[str],
    sample_size: int = 2500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Per-observation silhouette using the mixed K-Prototypes dissimilarity.

    The sampling and distance definition replicate ``kprototypes_silhouette`` in
    :mod:`src.modeling`, so the mean of the returned values matches the score
    reported by the evaluation pipeline.
    """
    empty = pd.DataFrame(columns=["Row", "Cluster", "Silhouette"])
    if X_model is None or labels is None:
        return empty

    labels = np.asarray(labels)
    n_rows = int(len(X_model))
    if n_rows == 0 or len(labels) != n_rows or not np.isfinite(gamma):
        return empty

    missing = [
        col
        for col in list(numerical_cols) + list(categorical_cols)
        if col not in X_model.columns
    ]
    if missing:
        return empty

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= n_rows:
        return empty

    if n_rows > sample_size:
        indices = np.random.RandomState(random_state).permutation(n_rows)[:sample_size]
        row_positions = indices
        X_sample = X_model.iloc[indices].reset_index(drop=True)
        labels_sample = labels[indices]
    else:
        row_positions = np.arange(n_rows)
        X_sample = X_model.reset_index(drop=True)
        labels_sample = labels

    numeric_values = X_sample[numerical_cols].to_numpy(dtype=float)
    distances = pairwise_distances(numeric_values, metric="euclidean", squared=True)

    mismatch = np.zeros((len(X_sample), len(X_sample)), dtype=np.int16)
    for col in categorical_cols:
        values = X_sample[col].astype(str).to_numpy()
        mismatch += values[:, None] != values[None, :]
    distances = distances + float(gamma) * mismatch.astype(float)

    values = silhouette_samples(distances, labels_sample, metric="precomputed")
    return pd.DataFrame(
        {
            "Row": row_positions,
            "Cluster": labels_sample.astype(int),
            "Silhouette": values,
        }
    )


def silhouette_plot(
    silhouette_df: pd.DataFrame,
    title: str = "Silhouette Plot per Klaster",
    sampled: bool = False,
    n_total: int | None = None,
) -> go.Figure:
    """Horizontal per-cluster silhouette bands with the overall mean line."""
    if silhouette_df is None or silhouette_df.empty:
        return _empty_figure("Nilai silhouette per observasi tidak tersedia.")
    if not {"Cluster", "Silhouette"}.issubset(silhouette_df.columns):
        return _empty_figure("Kolom nilai silhouette tidak lengkap.")

    df = silhouette_df.dropna(subset=["Cluster", "Silhouette"]).copy()
    if df.empty:
        return _empty_figure("Nilai silhouette per observasi tidak tersedia.")

    df["Label"] = df["Cluster"].apply(_cluster_label)
    colors = cluster_color_map(df["Label"])
    clusters = sorted(df["Label"].unique(), key=_cluster_sort_key)
    mean_value = float(df["Silhouette"].mean())
    negative_share = float((df["Silhouette"] < 0).mean() * 100)

    fig = go.Figure()
    gap = 10
    cursor = 0
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for cluster in clusters:
        values = np.sort(
            df.loc[df["Label"] == cluster, "Silhouette"].to_numpy(dtype=float)
        )
        if len(values) == 0:
            continue
        y_positions = np.arange(cursor, cursor + len(values))
        fig.add_trace(
            go.Bar(
                x=values,
                y=y_positions,
                orientation="h",
                width=1,
                marker=dict(color=colors[cluster], line=dict(width=0)),
                name=cluster,
                hovertemplate=(
                    f"<b>{cluster}</b><br>Silhouette: %{{x:.4f}}<extra></extra>"
                ),
            )
        )
        tick_positions.append(cursor + (len(values) - 1) / 2)
        tick_labels.append(f"{cluster} (n={len(values):,})")
        cursor += len(values) + gap

    x_min = min(0.0, float(df["Silhouette"].min())) - 0.05
    fig.add_vline(x=0, line_color="#D0D0D0", line_width=1)
    fig.add_vline(x=mean_value, line_dash="dash", line_color="#D62728", line_width=2)

    if sampled and n_total:
        mean_text = (
            f"Rata-rata sampel (n={len(df):,} dari {n_total:,}) = {mean_value:.3f}"
        )
    else:
        mean_text = f"Rata-rata = {mean_value:.3f}"

    fig.add_annotation(
        x=mean_value,
        y=1.0,
        xref="x",
        yref="paper",
        yanchor="bottom",
        showarrow=False,
        text=mean_text,
        font=dict(color="#D62728", size=12),
    )
    fig.update_layout(
        barmode="overlay",
        bargap=0,
        legend_title_text="Klaster",
        xaxis=dict(title="Silhouette Coefficient Values", range=[x_min, 1.0]),
        yaxis=dict(
            title="Klaster",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
            showgrid=False,
        ),
    )
    fig = _style(fig, height=520, title=title)
    fig.layout.meta = {
        "mean": mean_value,
        "negative_share": negative_share,
        "n_plotted": int(len(df)),
        "sampled": bool(sampled),
    }
    return fig

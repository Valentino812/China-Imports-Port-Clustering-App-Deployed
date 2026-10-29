"""Dashboard page with five SPA-like internal views.

The five views (Ringkasan, Profil Klaster, Evaluasi, Peta, Data) are switched via
``st.session_state`` so this file remains a single Streamlit page. The header and
sidebar render once and stay consistent across view changes.

This deployed variant always displays the precomputed analysis of the four
bundled datasets with K = 5. If the result has not been loaded from Beranda yet,
it is read from the artifact folder (and cached) automatically before the
selected view is rendered. No preprocessing or modelling runs at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import get_precomputed_analysis
from src.config import FIXED_K, OPTIMAL_K
from src.visualization import (
    cluster_boxplots,
    cluster_numeric_stats,
    cluster_pie,
    cluster_scatter,
    cluster_size_bar,
    cluster_violinplots,
    cluster_z_heatmap,
    confusion_heatmap,
    elbow_line,
    global_shap_bar,
    numerical_mean_bars,
    shap_beeswarm_figure,
    shap_cluster_heatmap,
    silhouette_plot,
    sitc_stacked_bar,
)

VIEWS = ["Ringkasan", "Profil Klaster", "Evaluasi", "Peta", "Data"]

VIEW_TITLES = {
    "Ringkasan": (
        "Ringkasan Hasil Klasterisasi",
        "Tinjauan menyeluruh atas hasil pengelompokan kombinasi pelabuhan dan kategori komoditas SITC, "
        "mencakup jumlah observasi, komposisi klaster, serta proporsi distribusi tiap kelompok.",
    ),
    "Profil Klaster": (
        "Karakteristik & Profil Klaster",
        "Analisis mendalam karakteristik tiap klaster melalui statistik deskriptif fitur numerik, "
        "modus fitur kategorikal, distribusi kelompok komoditas SITC, sebaran wilayah provinsi, "
        "dan interpretasi berbasis nilai SHAP dari model surrogate Random Forest untuk seluruh "
        "data uji.",
    ),
    "Evaluasi": (
        "Evaluasi Model Utama dan Model Surrogate",
        "Evaluasi kualitas klaster K-Prototypes melalui Elbow dan Silhouette, dipisahkan dari "
        "evaluasi fidelity model surrogate Random Forest yang menjadi landasan analisis SHAP.",
    ),
    "Peta": (
        "Peta Distribusi Spasial",
        "Visualisasi interaktif persebaran geografis klaster pelabuhan di seluruh wilayah Indonesia, "
        "difilter berdasarkan kategori komoditas SITC untuk menghindari tumpang tindih titik lokasi.",
    ),
    "Data": (
        "Eksplorasi & Ekspor Data",
        "Pratinjau dan ekspor dataset pada setiap tahapan pipeline analisis — mulai dari data impor "
        "mentah, hasil konversi HS–SITC, agregasi, penggabungan provinsi, hingga dataset final berlabel klaster.",
    ),
}

# Nama kelompok SITC Rev. 4 berdasarkan digit pertama kode SITC.
SITC_LABELS = {
    "0": "Bahan makanan dan binatang hidup",
    "1": "Minuman dan tembakau",
    "2": "Bahan mentah, tidak dapat dimakan, kecuali bahan bakar",
    "3": "Bahan bakar mineral, pelumas, dan bahan terkait",
    "4": "Minyak, lemak, dan lilin hewani serta nabati",
    "5": "Bahan kimia dan produk terkait",
    "6": "Barang manufaktur berbasis bahan baku",
    "7": "Mesin dan peralatan transportasi",
    "8": "Barang manufaktur jadi",
    "9": "Komoditas dan transaksi tidak diklasifikasikan di tempat lain",
}

FEATURE_SPECS = [
    {
        "feature": "usd_value",
        "label": "Nilai Impor (USD)",
        "number_format": ",.2f",
    },
    {
        "feature": "pdrb",
        "label": "PDRB (Miliar rupiah)",
        "number_format": ",.2f",
    },
    {
        "feature": "china_investment_us_ribu",
        "label": "Nilai Investasi Tiongkok (Ribuan USD)",
        "number_format": ",.2f",
    },
    {
        "feature": "micro_small_industries",
        "label": "Jumlah Industri Mikro Kecil",
        "number_format": ",.0f",
    },
]

# Label tampilan saja. Kunci tetap memakai nama kolom asli agar proses
# preprocessing dan modelling tidak berubah.
FEATURE_VALUE_LABELS = {
    spec["feature"]: spec["label"] for spec in FEATURE_SPECS
}
FEATURE_SHORT_LABELS = {
    "usd_value": "Nilai Impor",
    "sitc": "SITC",
    "pdrb": "PDRB",
    "china_investment_us_ribu": "Nilai Investasi Tiongkok",
    "micro_small_industries": "Jumlah Industri Mikro Kecil",
}
FEATURE_FORMATS = {
    spec["feature"]: spec["number_format"] for spec in FEATURE_SPECS
}


def _sitc_label(value: str) -> str:
    label = SITC_LABELS.get(str(value))
    return f"SITC {value}: {label}" if label else f"SITC {value}"


def _sitc_legend_caption() -> str:
    """Return the SITC code legend used in dashboard captions."""
    return "\n".join(
        f"- {_sitc_label(code)}" for code in sorted(SITC_LABELS, key=int)
    )


DASHBOARD_CSS = r"""
<style>
/* ─── Dashboard: unhide sidebar only while this page is rendered ────── */
body:has(.dashboard-page-marker) [data-testid='stSidebar'] {
    display: block !important;
    background: rgba(255,255,255,0.95) !important;
    border-right: 1px solid rgba(226,229,241,0.8) !important;
    min-width: 265px;
    box-shadow: 2px 0 12px rgba(79,107,237,0.06);
}
body:has(.dashboard-page-marker) [data-testid='stSidebar'] > div:first-child {
    padding-top: 1.25rem;
}

/* Streamlit renders the expand button inside the header only while the
   sidebar is collapsed. Keep that button reachable as a floating control. */
body:has(.dashboard-page-marker) header[data-testid='stHeader'] {
    display: block !important;
    position: absolute !important;
    height: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    pointer-events: none !important;
}
body:has(.dashboard-page-marker) [data-testid='stToolbar'] {
    display: flex !important;
    position: fixed !important;
    left: 1rem;
    bottom: 1rem;
    width: auto !important;
    height: auto !important;
    padding: 0 !important;
    background: transparent !important;
    pointer-events: auto !important;
    z-index: 1000;
}
body:has(.dashboard-page-marker) [data-testid='stToolbar'] > div > div:not(:first-child) {
    display: none !important;
}
body:has(.dashboard-page-marker) [data-testid='stExpandSidebarButton'] {
    background: #FFFFFF !important;
    border: 1px solid rgba(226,229,241,0.9) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    box-shadow: 0 2px 8px rgba(79,107,237,0.12) !important;
}

/* Sidebar text */
[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] p,
[data-testid='stSidebar'] label,
[data-testid='stSidebar'] label span,
[data-testid='stSidebar'] [role='radiogroup'] p,
[data-testid='stSidebar'] [data-testid='stWidgetLabel'] {
    color: #1A1C2E !important;
    font-size: 14px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    line-height: 1.5 !important;
}
[data-testid='stSidebar'] [role='radiogroup'] label {
    padding: 7px 10px;
    border-radius: 8px;
    font-weight: 500 !important;
}

/* Sidebar brand block */
.side-brand {
    padding: 6px 4px 18px 4px;
    border-bottom: 1px solid rgba(226,229,241,0.9);
    margin-bottom: 16px;
}
.side-brand-title {
    font-size: 14px;
    font-weight: 700;
    color: #1A1C2E;
    line-height: 1.5;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Sidebar section label */
.side-section-label {
    font-size: 10.5px;
    letter-spacing: 0.12em;
    font-weight: 700;
    color: #6B7280;
    text-transform: uppercase;
    margin: 16px 0 8px 2px;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Sidebar footer */
.sidebar-footer-caption {
    margin-top: 12px;
    text-align: center;
    font-size: 11.5px;
    color: #9CA3AF;
    font-family: 'Inter', system-ui, sans-serif;
    line-height: 1.5;
}
</style>
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _cluster_structure(clustered: pd.DataFrame) -> pd.DataFrame:
    struct = (
        clustered.groupby("Cluster", sort=True)
        .agg(
            jumlah_kombinasi_Port_SITC=("Cluster", "size"),
            jumlah_pelabuhan_unik=("port", "nunique"),
            jumlah_provinsi=("Provinsi", "nunique"),
        )
    )
    struct["persentase_dataset"] = (
        struct["jumlah_kombinasi_Port_SITC"] / len(clustered) * 100
    ).round(2)
    return struct.reset_index()


def _get_mode_text(series: pd.Series, prefix: str | None = None) -> str:
    modes = series.mode(dropna=True).astype(str).tolist()
    if prefix:
        modes = [f"{prefix} {mode}" for mode in modes]
    return ", ".join(modes) if modes else "Tidak tersedia"


def _goto_home() -> None:
    st.switch_page("pages/1_Beranda.py")


def _ensure_result() -> dict | None:
    """Return the prepared result, loading the precomputed artifact if needed."""
    result = st.session_state.get("analysis_result")
    if result:
        return result

    try:
        with st.spinner(
            f"Memuat hasil analisis precomputed (K = {FIXED_K})..."
        ):
            result = get_precomputed_analysis()
    except Exception as exc:  # noqa: BLE001 - surface any artifact failure
        st.session_state["processing_error"] = str(exc)
        return None

    st.session_state["analysis_result"] = result
    st.session_state["pipeline_summary"] = result["preprocessing"]["report"]
    st.session_state["processing_error"] = None
    return result


def _no_results() -> None:
    st.info(
        "Hasil analisis belum dapat ditampilkan. Kembali ke halaman Beranda lalu tekan "
        f"**Lihat Hasil Analisis** untuk menyiapkan hasil dari empat berkas data bawaan (K = {FIXED_K})."
    )
    if st.session_state.get("processing_error"):
        st.error(
            "Terjadi kendala saat menyiapkan hasil analisis: "
            f"{st.session_state['processing_error']}"
        )
    st.button(
        "← Kembali ke Beranda",
        type="primary",
        on_click=_goto_home,
    )


# ---------------------------------------------------------------------------
# View: Ringkasan
# ---------------------------------------------------------------------------

def _render_summary(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    struct = _cluster_structure(clustered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observasi", f"{len(clustered):,}")
    c2.metric("Jumlah Pelabuhan", f"{clustered['port'].nunique():,}")
    c3.metric("Cakupan Provinsi", f"{clustered['Provinsi'].nunique():,}")
    c4.metric("Jumlah Klaster (K)", str(result["modeling"]["config"]["n_clusters"]))

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            cluster_size_bar(struct, "jumlah_kombinasi_Port_SITC", "Ukuran Klaster"),
            width="stretch",
        )
    with right:
        st.plotly_chart(cluster_pie(struct), width="stretch")

    st.subheader("Tabel Ukuran dan Proporsi Tiap Klaster")
    display = struct.rename(
        columns={
            "Cluster": "Klaster",
            "jumlah_kombinasi_Port_SITC": "Jumlah Observasi",
            "jumlah_pelabuhan_unik": "Pelabuhan Unik",
            "jumlah_provinsi": "Provinsi",
            "persentase_dataset": "Proporsi (%)",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)



# ---------------------------------------------------------------------------
# View: Profil Klaster
# ---------------------------------------------------------------------------
#
# [LOG SCALE FEATURE - DISABLED]
# Kontrol skala log pada tab Profil Klaster (scatter, boxplot, dan violinplot)
# dinonaktifkan sementara. Kode asli dipertahankan sebagai komentar agar dapat
# diaktifkan kembali tanpa menulis ulang implementasinya.
# ---------------------------------------------------------------------------

def _render_cluster_overview(struct: pd.DataFrame) -> None:
    st.markdown("**Distribusi Observasi per Klaster**")
    overview = struct[
        ["Cluster", "jumlah_kombinasi_Port_SITC", "persentase_dataset"]
    ].copy()
    overview["Cluster"] = overview["Cluster"].apply(lambda c: f"Cluster {c}")
    overview = overview.rename(
        columns={
            "Cluster": "Klaster",
            "jumlah_kombinasi_Port_SITC": "Jumlah Observasi",
            "persentase_dataset": "Proporsi (%)",
        }
    )
    st.dataframe(overview, width="stretch", hide_index=True)
    st.caption(
        "Tabel ini merangkum jumlah observasi (kombinasi pelabuhan–SITC) dan proporsinya "
        "terhadap total dataset untuk setiap klaster yang terbentuk."
    )


def _render_numeric_stats_table(stats: pd.DataFrame) -> None:
    st.markdown("**Statistik Deskriptif Fitur Numerik per Klaster**")
    if stats.empty:
        st.warning("Statistik fitur numerik tidak tersedia untuk dataset ini.")
    else:
        display_stats = stats[
            ["Klaster", "Fitur", "Mean", "Median", "Q1", "Q3", "IQR"]
        ].copy()
        display_stats["Fitur"] = display_stats["Fitur"].map(
            lambda feature: FEATURE_VALUE_LABELS.get(feature, feature)
        )
        st.dataframe(
            display_stats.round(3),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Statistik deskriptif (rata-rata, median, kuartil bawah Q1, kuartil atas Q3, dan rentang "
        "antarkuartil IQR) dihitung untuk setiap fitur numerik pada masing-masing klaster, "
        "menggunakan nilai dalam satuan asli tanpa standardisasi."
    )


def _render_categorical_modes(
    clustered: pd.DataFrame, categorical_cols: list[str]
) -> None:
    st.markdown("**Nilai Modus Fitur Kategori SITC per Klaster**")
    mode_rows = []
    for cluster_id, group in clustered.groupby("Cluster", sort=True):
        row = {"Klaster": f"Cluster {cluster_id}"}
        for col in categorical_cols:
            row[f"Modus {col.upper()}"] = _get_mode_text(
                group[col], prefix="SITC" if col == "sitc" else None
            )
        mode_rows.append(row)
    st.dataframe(pd.DataFrame(mode_rows), width="stretch", hide_index=True)
    st.caption(
        "Tabel ini menampilkan nilai yang paling sering muncul (modus) pada setiap fitur kategori SITC (Standard International Trade Classification) "
        "untuk masing-masing klaster. Apabila terdapat lebih dari satu nilai dengan frekuensi tertinggi, "
        "seluruh nilai tersebut ditampilkan."
        "\n\n**Keterangan kode SITC:**\n"
        + _sitc_legend_caption()
    )


def _render_sitc_distribution(clustered: pd.DataFrame) -> None:
    st.markdown("**Distribusi Proporsi Kategori SITC per Klaster**")
    sitc_pct = pd.crosstab(
        clustered["Cluster"], clustered["sitc"], normalize="index"
    ).mul(100)
    sitc_pct = sitc_pct.reindex(sorted(sitc_pct.columns), axis=1).round(2)
    sitc_display = sitc_pct.reset_index()
    sitc_display["Cluster"] = sitc_display["Cluster"].apply(lambda c: f"Cluster {c}")
    sitc_display.columns = ["Klaster"] + [
        f"SITC {col}" for col in sitc_pct.columns
    ]
    st.dataframe(sitc_display, width="stretch", hide_index=True)
    st.caption(
        "Tabel lintas-tabulasi (crosstab) ini menunjukkan proporsi (%) setiap kategori SITC (Standard International Trade Classification) "
        "terhadap total observasi dalam klaster yang bersangkutan."
    )
    st.plotly_chart(sitc_stacked_bar(sitc_pct), width="stretch")
    st.caption(
        "Diagram batang bertumpuk ini memvisualisasikan komposisi kategori SITC (Standard International Trade Classification) di setiap klaster "
        "dalam bentuk proporsi relatif, sehingga perbedaan pola komoditas antara klaster dapat "
        "dibandingkan secara langsung."
    )


def _render_mean_feature_section(
    clustered: pd.DataFrame, numerical_cols: list[str]
) -> None:
    st.markdown("**Perbandingan Rata-Rata Fitur Numerik per Klaster**")
    st.plotly_chart(
        numerical_mean_bars(
            clustered,
            numerical_cols,
            FEATURE_FORMATS,
            feature_labels=FEATURE_VALUE_LABELS,
        ),
        width="stretch",
    )
    st.caption(
        "Diagram batang ini menampilkan nilai rata-rata setiap fitur numerik untuk masing-masing klaster "
        "dalam satuan aslinya. Perbandingan ini memungkinkan identifikasi perbedaan besaran absolut "
        "antarklaster pada setiap fitur."
    )


# [LOG SCALE FEATURE - DISABLED] Helper catatan skala log dinonaktifkan.
# def _with_excluded_note(figure: Any, caption: str) -> str:
#     meta = dict(figure.layout.meta or {})
#     if meta.get("excluded_nonpositive"):
#         caption += (
#             f" Sebanyak {meta['excluded_nonpositive']:,} nilai nol atau negatif tidak dapat "
#             "ditampilkan pada skala log."
#         )
#     return caption


def _render_distribution_section(
    clustered: pd.DataFrame,
    stats: pd.DataFrame,
    numerical_cols: list[str],
    *,
    heading: str,
    # [LOG SCALE FEATURE - DISABLED] checkbox_key: str,
    chart_title: str,
    caption: str,
    figure_builder: Callable[..., Any],
) -> None:
    st.markdown(heading)
    # [LOG SCALE FEATURE - DISABLED]
    # log_usd = st.checkbox(
    #     "Gunakan skala logaritmik untuk Nilai Impor (USD)",
    #     key=checkbox_key,
    #     value=False,
    # )
    figure = figure_builder(
        clustered,
        stats,
        numerical_cols,
        FEATURE_VALUE_LABELS,
        title=chart_title,
        # [LOG SCALE FEATURE - DISABLED] log_usd=log_usd,
        number_formats=FEATURE_FORMATS,
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(caption)
    # [LOG SCALE FEATURE - DISABLED]
    # st.caption(_with_excluded_note(figure, caption))


def _render_zscore_section(
    clustered: pd.DataFrame, numerical_cols: list[str]
) -> None:
    st.markdown("**Heatmap Profil Klaster Berdasarkan Z-Score**")
    cluster_means = clustered.groupby("Cluster", sort=True)[numerical_cols].mean()
    global_means = clustered[numerical_cols].mean()
    global_stds = clustered[numerical_cols].std(ddof=0).replace(0, np.nan)
    z_profile = (
        cluster_means.subtract(global_means, axis="columns")
        .divide(global_stds, axis="columns")
        .fillna(0)
    )
    st.plotly_chart(
        cluster_z_heatmap(z_profile, FEATURE_SHORT_LABELS), width="stretch"
    )
    st.caption(
        "Heatmap ini menampilkan Z-score rata-rata setiap fitur numerik per klaster relatif terhadap "
        "rata-rata dan standar deviasi global dataset. Nilai positif mengindikasikan fitur berada "
        "di atas rata-rata global, sedangkan nilai negatif mengindikasikan di bawah rata-rata. "
        "Pola ini digunakan untuk mengidentifikasi karakteristik pembeda setiap klaster."
    )


def _render_profile_statistics(
    clustered: pd.DataFrame,
    struct: pd.DataFrame,
    stats: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> None:
    st.subheader("Profil Statistik Klaster")
    st.caption(
        "Bagian ini merangkum karakteristik setiap klaster melalui jumlah observasi, "
        "statistik deskriptif fitur numerik, modus fitur kategorikal, distribusi SITC (Standard International Trade Classification), "
        "perbandingan rata-rata fitur, sebaran kuartil, kepadatan nilai, dan heatmap Z-score. "
        "Seluruh statistik dihitung dari baris pelabuhan × SITC, bukan dengan bobot sama "
        "antarprovinsi."
    )

    _render_cluster_overview(struct)
    _render_numeric_stats_table(stats)
    _render_categorical_modes(clustered, categorical_cols)
    _render_sitc_distribution(clustered)
    _render_mean_feature_section(clustered, numerical_cols)
    _render_distribution_section(
        clustered,
        stats,
        numerical_cols,
        heading="**Statistik Kuartil Atribut Numerik per Klaster**",
        # [LOG SCALE FEATURE - DISABLED] checkbox_key="profile_box_log_usd",
        chart_title="Statistik Kuartil Atribut Numerik per Klaster",
        caption=(
            "Boxplot menampilkan median, kuartil pertama (Q1), kuartil ketiga (Q3), IQR, dan pencilan "
            "untuk setiap fitur numerik per klaster. Whisker menggunakan aturan 1,5 × IQR; pencilan "
            "di luar whisker tetap ditampilkan.Boxplot menampilkan median, kuartil pertama (Q1), kuartil "
            "ketiga (Q3), IQR, dan pencilan setiap fitur numerik per klaster. Garis di dalam kotak menunjukkan median"
            ", sedangkan tinggi kotak menggambarkan sebaran 50% data bagian tengah. *Whisker* merupakan garis yang memanjang" 
            "dari kotak hingga nilai data terjauh yang masih berada dalam batas 1,5 × IQR, sementara titik di luarnya "
            "merupakan pencilan. Perbandingan elemen tersebut membantu mengidentifikasi perbedaan kecenderungan nilai"
            "dan penyebaran data antar-klaster."
        ),
        figure_builder=cluster_boxplots,
    )
    _render_distribution_section(
        clustered,
        stats,
        numerical_cols,
        heading="**Pola Kepadatan Nilai Atribut Numerik Per Klaster**",
        # [LOG SCALE FEATURE - DISABLED] checkbox_key="profile_violin_log_usd",
        chart_title="Kepadatan Nilai Atribut Numerik per Klaster",
        caption=(
            "Violin plot menunjukkan kepadatan distribusi nilai setiap atribut numerik per klaster. "
            "Lebar violin mencerminkan kepadatan observasi, sedangkan garis mean menunjukkan rata-rata "
            "klaster. Standar deviasi (SD) mengukur rata-rata jarak nilai observasi dari mean: semakin "
            "besar SD, semakin lebar dan bervariasi sebaran nilai dalam klaster. Nilai mean dan SD "
            "ditampilkan pada tooltip. Pencilan tetap ditampilkan."
        ),
        figure_builder=cluster_violinplots,
    )
    _render_zscore_section(clustered, numerical_cols)


def _render_profile(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    numerical_cols = result["modeling"]["numerical_cols"]
    categorical_cols = result["modeling"]["categorical_cols"]
    struct = _cluster_structure(clustered)
    stats = cluster_numeric_stats(clustered, numerical_cols)

    st.info(
        "Profil klaster terdiri atas profil statistik, sebaran klaster, "
        "dan analisis SHAP dari model surrogate."
    )

    tab_stats, tab_scatter, tab_shap = st.tabs(
        ["Profil Statistik", "Sebaran Klaster", "Analisis SHAP"]
    )

    with tab_stats:
        _render_profile_statistics(
            clustered, struct, stats, numerical_cols, categorical_cols
        )

    with tab_scatter:
        _render_cluster_scatters(result)

    with tab_shap:
        _render_shap_section(result)


def _render_cluster_scatters(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    st.subheader("Sebaran Klaster Nilai Impor terhadap Indikator Ekonomi Provinsi")
    st.caption(
        "Setiap titik merepresentasikan satu kombinasi pelabuhan–SITC, dengan warna menunjukkan "
        "keanggotaan klaster. Sumbu menggunakan nilai asli tanpa transformasi."
        # [LOG SCALE FEATURE - DISABLED] "Aktifkan skala logaritmik apabila distribusi nilai
        # sangat menceng untuk memudahkan pembacaan pola."
    )

    specs = [
        {
            "y": "china_investment_us_ribu",
            "y_label": "Nilai Investasi Tiongkok (Ribuan USD)",
            "title": "Nilai Impor (USD) vs Nilai Investasi Tiongkok (Ribuan USD)",
            # [LOG SCALE FEATURE - DISABLED] "key": "profile_scatter_log_investment",
            "caption": (
                "Scatter plot ini menampilkan distribusi kombinasi pelabuhan–SITC berdasarkan nilai "
                "impor dan realisasi investasi Tiongkok di provinsi terkait. Pola konsentrasi atau "
                "persebaran dapat menggambarkan karakteristik klaster, namun tidak mengindikasikan "
                "hubungan kausalitas antara kedua variabel."
            ),
        },
        {
            "y": "micro_small_industries",
            "y_label": "Jumlah Industri Mikro Kecil",
            "title": "Nilai Impor (USD) vs Jumlah Industri Mikro Kecil",
            # [LOG SCALE FEATURE - DISABLED] "key": "profile_scatter_log_industries",
            "caption": (
                "Scatter plot ini menampilkan distribusi kombinasi pelabuhan–SITC berdasarkan nilai "
                "impor dan jumlah unit usaha industri mikro dan kecil di provinsi terkait. "
                "Interpretasi terbatas pada karakteristik sebaran; jumlah industri mikro dan kecil "
                "bukan proksi langsung kapasitas atau kebutuhan impor."
            ),
        },
        {
            "y": "pdrb",
            "y_label": "PDRB (Miliar rupiah)",
            "title": "Nilai Impor (USD) vs PDRB (Miliar rupiah)",
            # [LOG SCALE FEATURE - DISABLED] "key": "profile_scatter_log_pdrb",
            "caption": (
                "Scatter plot ini menampilkan distribusi kombinasi pelabuhan–SITC berdasarkan nilai "
                "impor dan Produk Domestik Regional Bruto (PDRB) provinsi terkait. PDRB "
                "mencerminkan skala aktivitas ekonomi wilayah dan tidak menunjukkan hubungan "
                "kausalitas langsung dengan nilai impor."
            ),
        },
    ]

    for spec in specs:
        # [LOG SCALE FEATURE - DISABLED]
        # log_scale = st.checkbox(
        #     "Gunakan skala logaritmik",
        #     key=spec["key"],
        #     value=False,
        #     help=(
        #         "Skala logaritmik hanya mengubah tampilan sumbu. Nilai sumber dan label klaster "
        #         "tidak berubah; observasi bernilai 0 tidak dapat ditampilkan."
        #     ),
        # )
        fig = cluster_scatter(
            clustered,
            "usd_value",
            spec["y"],
            "Nilai Impor (USD)",
            spec["y_label"],
            spec["title"],
            # [LOG SCALE FEATURE - DISABLED] log_scale=log_scale,
        )
        st.plotly_chart(fig, width="stretch")

        caption = spec["caption"]
        meta = dict(fig.layout.meta or {})
        # [LOG SCALE FEATURE - DISABLED]
        # if meta.get("excluded_nonpositive"):
        #     caption += (
        #         f" Sebanyak {meta['excluded_nonpositive']:,} observasi bernilai 0 atau negatif "
        #         "tidak dapat ditampilkan pada skala log."
        #     )
        if meta.get("sampled"):
            caption += (
                f" Visualisasi menggunakan sampel deterministik {meta['n_plotted']:,} dari "
                f"{meta['n_total']:,} observasi (random_state=42)."
            )
        st.caption(caption)


def _render_shap_section(result: dict) -> None:
    st.subheader("Analisis SHAP (Interpretasi Model Surrogate)")
    st.caption(
        "SHAP mengukur kontribusi fitur terhadap prediksi model surrogate yang dilatih meniru "
        "label K-Prototypes. Nilai dihitung untuk seluruh data uji tanpa filter label; setiap "
        "tab mewakili satu kelas klaster. Interpretasinya bergantung "
        "pada fidelity surrogate yang dapat diperiksa di tab Evaluasi."
    )
    shap_result = result["modeling"]["shap"]
    if not shap_result.get("available"):
        st.warning(
            "Visualisasi SHAP tidak dapat ditampilkan. Alasan: "
            + str(shap_result.get("reason", "prasyarat data tidak terpenuhi"))
        )
        return

    st.markdown("**Kepentingan Fitur Global (Global Feature Importance)**")
    st.plotly_chart(
        global_shap_bar(
            shap_result["global_importance"], FEATURE_SHORT_LABELS
        ),
        width="stretch",
    )
    st.caption(
        "Diagram batang ini menampilkan rata-rata nilai absolut SHAP setiap fitur di seluruh "
        "observasi data uji dan seluruh kelas klaster, sebagai ukuran kepentingan fitur secara "
        "global terhadap prediksi model surrogate."
    )

    st.markdown("**Kontribusi Fitur per Kelas Klaster (Heatmap)**")
    st.plotly_chart(
        shap_cluster_heatmap(
            shap_result["importance_per_cluster"], FEATURE_SHORT_LABELS
        ),
        width="stretch",
    )
    st.caption(
        "Heatmap ini menampilkan rata-rata nilai absolut SHAP per fitur untuk setiap kelas klaster "
        "pada seluruh data uji. Setiap baris adalah kelas target model surrogate, bukan subset "
        "observasi yang menjadi anggota klaster tersebut, sehingga perbedaan tingkat kepentingan "
        "fitur antar kelas klaster dapat diidentifikasi."
    )

    st.markdown("**Kontribusi Fitur terhadap Setiap Kelas Klaster (Beeswarm)**")
    classes = shap_result["classes"]
    features = result["modeling"]["X_model"].loc[
        shap_result["X_eval"].index, result["modeling"]["feature_cols"]
    ].rename(columns=FEATURE_SHORT_LABELS)
    tabs = st.tabs([f"Cluster {c}" for c in classes])
    for tab, cluster_id in zip(tabs, classes):
        with tab:
            grouped_values = shap_result["grouped_by_cluster"][int(cluster_id)]
            try:
                fig = shap_beeswarm_figure(
                    grouped_values.to_numpy(),
                    features,
                    f"Kontribusi Fitur pada Cluster {cluster_id} (SHAP Beeswarm)",
                )
                st.pyplot(fig, width="stretch")
                st.caption(
                    "Setiap titik merepresentasikan satu observasi dari keseluruhan data uji "
                    f"(n={len(grouped_values):,}) untuk kelas Cluster {cluster_id}. Sumbu horizontal "
                    "menunjukkan nilai SHAP fitur terhadap keluaran probabilitas model surrogate untuk "
                    "kelas tersebut: nilai positif meningkatkan skor kelas, sedangkan nilai negatif "
                    "menurunkannya relatif terhadap nilai dasar model. Warna menunjukkan nilai fitur "
                    "pada data uji (fitur numerik telah distandardisasi untuk model). Titik ditumpuk "
                    "vertikal menurut kepadatan dan fitur diurutkan berdasarkan rata-rata nilai SHAP "
                    "absolut. Nilai ini menjelaskan perilaku model surrogate, bukan hubungan kausal, "
                    "dan tidak menunjukkan bahwa observasi tersebut termasuk anggota kelas klaster."
                )
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as exc:
                st.warning(
                    f"Beeswarm SHAP untuk kontribusi ke Cluster {cluster_id} "
                    f"tidak dapat dibuat: {exc}"
                )



# ---------------------------------------------------------------------------
# View: Evaluasi
# ---------------------------------------------------------------------------

def _render_elbow_section(elbow: pd.DataFrame, k: int) -> None:
    st.subheader("Penentuan Jumlah Klaster Optimal dengan Metode Elbow")
    st.plotly_chart(elbow_line(elbow, k), width="stretch")
    k_note = (
        f"K optimal hasil evaluasi adalah K = {OPTIMAL_K}, sedangkan K yang "
        f"digunakan pada aplikasi ini adalah K = {k}."
    )
    if int(k) == OPTIMAL_K:
        k_note += " Nilai ini sama dengan K optimal."
    st.caption(
        "Grafik nilai cost K-Prototypes terhadap jumlah klaster (K). Nilai K optimal "
        "dipilih pada titik 'siku' (elbow), yaitu saat penurunan cost mulai tidak "
        f"signifikan. {k_note} Jumlah klaster telah ditetapkan pada aplikasi ini "
        "sehingga tidak dapat diubah dari antarmuka."
    )
    st.dataframe(
        elbow.rename(columns={"K": "Jumlah Klaster (K)", "Cost": "Total Cost"}).round(4),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Tabel ini mencatat nilai total cost K-Prototypes untuk setiap nilai K yang diuji."
    )


def _render_silhouette_plot(
    silhouette_df: pd.DataFrame,
    modeling: dict,
    silhouette: float,
    sampled: bool,
) -> None:
    st.markdown("**Silhouette Plot per Klaster**")
    st.plotly_chart(
        silhouette_plot(
            silhouette_df,
            sampled=sampled,
            n_total=len(modeling["clustered"]),
        ),
        width="stretch",
    )
    st.caption(
        "Setiap batang merepresentasikan nilai silhouette satu observasi, "
        "dikelompokkan dan diurutkan per klaster. Garis vertikal menunjukkan "
        "rata-rata nilai silhouette keseluruhan sebagai acuan kualitas pemisahan klaster."
    )

    mean_plot = float(silhouette_df["Silhouette"].mean())
    negative_share = float((silhouette_df["Silhouette"] < 0).mean() * 100)
    if abs(mean_plot - float(silhouette)) > 1e-6:
        st.warning(
            "Rata-rata silhouette dari plot "
            f"({mean_plot:.6f}) berbeda dari skor evaluasi "
            f"({float(silhouette):.6f}). Periksa konsistensi sampel evaluasi."
        )

    sample_note = (
        f"Plot memakai sampel deterministik {len(silhouette_df):,} dari "
        f"{len(modeling['clustered']):,} observasi (random_state=42); rata-rata "
        "yang ditampilkan adalah rata-rata sampel tersebut."
        if sampled
        else f"Plot memakai seluruh {len(silhouette_df):,} observasi; rata-rata "
        "sama dengan skor evaluasi di atas."
    )
    st.caption(
        "Nilai silhouette mendekati 1 menunjukkan observasi relatif cocok dengan "
        "klasternya dan terpisah dari klaster terdekat. Nilai mendekati 0 menunjukkan "
        "observasi berada di sekitar batas antarklaster, sedangkan nilai negatif "
        "menunjukkan observasi mungkin lebih dekat ke klaster lain. "
        f"Rata-rata yang diplot = {mean_plot:.4f}; proporsi observasi bernilai "
        f"negatif = {negative_share:.2f}%. {sample_note} Tidak ada ambang kualitas "
        "tunggal: nilai rata-rata, sebaran per klaster, dan proporsi negatif perlu "
        "dibaca bersama-sama."
    )


def _render_silhouette_table(silhouette_df: pd.DataFrame) -> None:
    summary = (
        silhouette_df.groupby("Cluster")["Silhouette"]
        .agg(["size", "mean", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "Cluster",
        "Jumlah",
        "Rata-rata",
        "Minimum",
        "Maksimum",
    ]
    negative_by_cluster = (
        silhouette_df.groupby("Cluster")["Silhouette"]
        .apply(lambda values: float((values < 0).mean() * 100))
        .reset_index(name="% Negatif")
    )
    summary = summary.merge(negative_by_cluster, on="Cluster")
    summary["Klaster"] = summary["Cluster"].apply(
        lambda value: f"Cluster {int(value)}"
    )
    st.dataframe(
        summary[
            [
                "Klaster",
                "Jumlah",
                "Rata-rata",
                "Minimum",
                "Maksimum",
                "% Negatif",
            ]
        ].round(
            {
                "Rata-rata": 3,
                "Minimum": 3,
                "Maksimum": 3,
                "% Negatif": 2,
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_silhouette_downloads(
    silhouette_df: pd.DataFrame,
    modeling: dict,
    sampled: bool,
) -> None:
    if "Row" not in silhouette_df.columns:
        st.warning(
            "Kolom indeks observasi tidak tersedia sehingga unduhan silhouette "
            "tidak dapat disiapkan. Jalankan ulang analisis."
        )
        return

    metadata = modeling["clustered"][["port", "sitc"]].reset_index(drop=True)
    metadata["Row"] = metadata.index
    silhouette_export = silhouette_df.merge(metadata, on="Row", how="left")

    silhouette_all = (
        silhouette_export[["port", "sitc", "Silhouette"]]
        .rename(columns={"Silhouette": "silhouette"})
        .sort_values(["port", "sitc"])
        .reset_index(drop=True)
    )
    silhouette_negative = (
        silhouette_export.loc[
            silhouette_export["Silhouette"] < 0, ["port", "sitc"]
        ]
        .sort_values(["port", "sitc"])
        .reset_index(drop=True)
    )

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Unduh Nilai Silhouette Seluruh Data (CSV)",
            data=silhouette_all.to_csv(index=False).encode("utf-8-sig"),
            file_name="silhouette_seluruh_data.csv",
            mime="text/csv",
            key="download_silhouette_all",
            width="stretch",
        )
    with download_col2:
        st.download_button(
            "Unduh Daftar Silhouette Negatif (CSV)",
            data=silhouette_negative.to_csv(index=False).encode("utf-8-sig"),
            file_name="silhouette_negatif.csv",
            mime="text/csv",
            key="download_silhouette_negative",
            width="stretch",
            disabled=silhouette_negative.empty,
        )
    if sampled:
        st.caption(
            "Unduhan mengikuti observasi yang diplot pada Silhouette Plot. "
            f"Karena data melebihi 2.500 observasi, berkas berisi sampel "
            f"deterministik {len(silhouette_df):,} observasi "
            "(random_state=42)."
        )


def _render_silhouette_section(modeling: dict, silhouette: float) -> None:
    st.subheader("Evaluasi Kohesi dan Separasi Klaster (Skor Silhouette)")
    st.caption(
        "Skor Silhouette dihitung menggunakan metrik ketidaksamaan campuran K-Prototypes: "
        "jumlah kuadrat selisih fitur numerik terstandardisasi ditambah penalti ketidakcocokan "
        "fitur kategorikal yang dibobot oleh parameter gamma."
    )
    if pd.isna(silhouette):
        st.warning("Skor Silhouette tidak dapat dihitung untuk konfigurasi data ini.")
        return

    st.info(
        f"Rata-Rata Skor Silhouette = {silhouette:.4f}, dihitung berdasarkan {modeling['silhouette_n']:,} sampel observasi "
        + (
            "(penarikan sampel deterministik, random_state=42)."
            if modeling["silhouette_n"] < len(modeling["clustered"])
            else "(seluruh data)."
        )
    )

    silhouette_df = modeling.get("silhouette_samples")
    if silhouette_df is None or silhouette_df.empty:
        st.warning(
            "Nilai silhouette per observasi tidak tersedia pada artifact analisis, "
            "sehingga Silhouette Plot tidak dapat ditampilkan. "
            "Bangun ulang artifact melalui scripts/build_analysis_artifacts.py."
        )
        return

    sampled = len(silhouette_df) < len(modeling["clustered"])
    _render_silhouette_plot(silhouette_df, modeling, silhouette, sampled)
    _render_silhouette_table(silhouette_df)
    _render_silhouette_downloads(silhouette_df, modeling, sampled)
    st.caption(
        "Ringkasan statistik nilai silhouette per klaster, mencakup jumlah observasi, "
        "rata-rata, nilai minimum, maksimum, dan proporsi observasi dengan nilai silhouette "
        "negatif — yang mengindikasikan observasi berpotensi lebih dekat ke klaster lain."
    )


def _render_kprototypes_config(modeling: dict) -> None:
    st.subheader("Konfigurasi Parameter Algoritma K-Prototypes")
    config = modeling["config"]
    st.dataframe(
        pd.DataFrame(
            {
                "Parameter": [
                    "init",
                    "n_init",
                    "random_state",
                    "n_jobs",
                    "gamma",
                ],
                "Nilai": [
                    str(config["init"]),
                    str(config["n_init"]),
                    str(config["random_state"]),
                    str(config["n_jobs"]),
                    f"{modeling['gamma']:.4f}",
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Tabel ini mencatat nilai parameter yang digunakan pada eksekusi algoritma K-Prototypes, "
        "termasuk nilai gamma yang menentukan bobot relatif fitur kategorikal terhadap fitur numerik "
        "dalam metrik ketidaksamaan campuran."
    )


def _render_kprototypes_evaluation(modeling: dict) -> None:
    k = modeling["config"]["n_clusters"]
    silhouette = modeling["silhouette"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Klaster (K)", str(k))
    c2.metric("Rata-Rata Skor Silhouette", _fmt(silhouette))
    c3.metric("Ukuran Sampel Evaluasi", f"{modeling['silhouette_n']:,}")

    _render_elbow_section(modeling["elbow"], k)
    _render_silhouette_section(modeling, silhouette)
    _render_kprototypes_config(modeling)


def _render_surrogate_metrics(surrogate: dict) -> None:
    st.subheader("Metrik Fidelity Model Surrogate (Random Forest)")
    st.dataframe(surrogate["metrics"].round(4), width="stretch", hide_index=True)
    st.caption(
        "Metrik ini mengukur tingkat kesesuaian (fidelity) antara prediksi model surrogate Random Forest "
        "dan label klaster K-Prototypes pada data uji. Nilai akurasi dan F1-score yang tinggi "
        "mengindikasikan bahwa surrogate mereplikasi label klaster dengan baik, sehingga nilai SHAP "
        "yang diturunkan dari surrogate dapat dianggap valid sebagai interpretasi pengelompokan."
    )


def _render_surrogate_report(surrogate: dict) -> None:
    report = surrogate["classification_report"]
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[
        report_df.index.map(
            lambda value: value not in {"accuracy", "macro avg", "weighted avg"}
        )
    ]
    report_df.index = [f"Cluster {i}" for i in report_df.index]
    st.markdown("**Laporan Klasifikasi per Klaster (Classification Report)**")
    st.dataframe(report_df.round(4), width="stretch")
    st.caption(
        "Tabel klasifikasi menampilkan precision, recall, dan F1-score model surrogate untuk "
        "setiap klaster pada data uji. Metrik ini dihitung berdasarkan perbandingan prediksi "
        "surrogate dengan label klaster K-Prototypes."
    )


def _render_surrogate_confusion(surrogate: dict) -> None:
    st.markdown("**Matriks Konfusi (Confusion Matrix)**")
    st.plotly_chart(
        confusion_heatmap(surrogate["confusion_matrix"], surrogate["class_labels"]),
        width="stretch",
    )
    st.caption(
        "Matriks konfusi mengukur seberapa baik surrogate meniru label klaster K-Prototypes pada data uji "
        "— bukan akurasi terhadap data asli. Semakin dominan elemen diagonal, semakin tinggi fidelity "
        "surrogate, dan semakin dapat dipercaya nilai SHAP sebagai cerminan logika pengelompokan K-Prototypes."
    )


def _render_surrogate_evaluation(surrogate: dict) -> None:
    st.info(
        "Model surrogate dilatih untuk mereplikasi label klaster K-Prototypes. Metrik di "
        "bawah mengukur fidelity surrogate terhadap label tersebut, bukan kualitas "
        "pengelompokan. Fidelity yang tinggi mendukung validitas interpretasi SHAP yang "
        "diturunkan dari model surrogate."
    )
    _render_surrogate_metrics(surrogate)
    _render_surrogate_report(surrogate)
    _render_surrogate_confusion(surrogate)


def _render_evaluation(result: dict) -> None:
    modeling = result["modeling"]

    st.info(
        "Halaman ini memisahkan evaluasi model utama klastering K-Prototypes dari evaluasi model "
        "surrogate Random Forest untuk kebutuhan analisis SHAP. Metrik surrogate mengukur seberapa baik model pendekatan "
        "meniru label klaster, bukan kualitas klaster itu sendiri."
    )

    tab_main, tab_surrogate = st.tabs(
        ["Model Utama (K-Prototypes)", "Model Surrogate SHAP (Random Forest)"]
    )

    with tab_main:
        _render_kprototypes_evaluation(modeling)

    with tab_surrogate:
        _render_surrogate_evaluation(modeling["surrogate"])


def _find_coordinate_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    lat = next((lookup[k] for k in ["latitude", "lat", "lintang"] if k in lookup), None)
    lon = next((lookup[k] for k in ["longitude", "lon", "lng", "bujur"] if k in lookup), None)
    return lat, lon


def _render_map(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    lat_col, lon_col = _find_coordinate_columns(clustered)

    if not lat_col or not lon_col:
        st.info(
            "Dataset input saat ini belum memuat kolom koordinat latitude dan longitude pelabuhan. "
            "Visualisasi peta dinonaktifkan untuk menjaga keakuratan informasi geografis."
        )
        st.markdown(
            "**Persyaratan kolom untuk mengaktifkan visualisasi peta:**\n"
            "- `latitude` (atau `lat` / `lintang`)\n"
            "- `longitude` (atau `lon` / `lng` / `bujur`)\n\n"
            "Sertakan data koordinat resmi pelabuhan pada berkas input, kemudian jalankan ulang "
            "proses analisis untuk menampilkan peta interaktif."
        )
        with st.expander("Contoh sintaks kode peta (untuk pengembangan fitur mendatang):"):
            st.code(
                "fig = px.scatter_geo(\n"
                "    clustered, lat='latitude', lon='longitude',\n"
                "    color='Cluster', hover_name='port',\n"
                "    scope='asia', projection='natural earth'\n"
                ")\n"
                "st.plotly_chart(fig, width='stretch')",
                language="python",
            )
        st.dataframe(
            pd.DataFrame({"Daftar Kolom pada Dataset": list(clustered.columns)}),
            width="stretch",
            hide_index=True,
        )
        return

    map_data = clustered.copy()
    map_data[lat_col] = pd.to_numeric(map_data[lat_col], errors="coerce")
    map_data[lon_col] = pd.to_numeric(map_data[lon_col], errors="coerce")
    map_data = map_data.dropna(subset=[lat_col, lon_col])
    map_data = map_data[
        map_data[lat_col].between(-90, 90) & map_data[lon_col].between(-180, 180)
    ]
    if map_data.empty:
        st.warning("Tidak ditemukan data koordinat yang valid pada dataset.")
        return

    if "sitc" not in map_data.columns:
        st.warning("Kolom `sitc` tidak ditemukan sehingga filter kategori SITC tidak tersedia.")
        return

    sitc_options = sorted(
        map_data["sitc"].astype(str).unique(),
        key=lambda value: int(value) if value.isdigit() else 99,
    )
    if st.session_state.get("map_sitc_choice") not in sitc_options:
        st.session_state["map_sitc_choice"] = sitc_options[0]

    selected_sitc = st.selectbox(
        "Pilih Satu Kategori SITC",
        sitc_options,
        key="map_sitc_choice",
        format_func=_sitc_label,
        help=(
            "Peta hanya menampilkan satu kategori SITC agar beberapa kategori pada "
            "pelabuhan yang sama tidak saling menumpuk."
        ),
    )

    subset = map_data[map_data["sitc"].astype(str) == selected_sitc].copy()
    subset = subset.drop_duplicates(subset=["port"]).reset_index(drop=True)
    subset["Klaster"] = subset["Cluster"].apply(
        lambda value: f"Cluster {int(value)}" if pd.notna(value) else "Cluster N/A"
    )

    from src.visualization import geo_scatter

    st.plotly_chart(
        geo_scatter(
            subset,
            lat_col,
            lon_col,
            "port",
            f"Sebaran Klaster Pelabuhan SITC {selected_sitc}",
            color_col="Klaster",
            hover_labels={
                "Provinsi": "Provinsi",
                "sitc": "SITC",
                "usd_value": "Nilai Impor (USD)",
                "Klaster": "Klaster",
            },
        ),
        width="stretch",
    )

    collisions = subset[
        subset.duplicated(subset=[lat_col, lon_col], keep=False)
    ].sort_values([lat_col, lon_col, "port"])
    caption = (
        f"Peta menampilkan {len(subset):,} pelabuhan unik untuk kategori {_sitc_label(selected_sitc)}. "
        "Setiap titik merepresentasikan satu pelabuhan; warna menunjukkan keanggotaan klaster hasil pemodelan. "
        "Filter per kategori SITC diterapkan untuk mencegah penumpukan marker pada lokasi yang sama."
    )
    if not collisions.empty:
        caption += (
            f" Sebanyak {len(collisions):,} pelabuhan berbagi koordinat geografis dengan pelabuhan lain "
            "sehingga marker-nya saling bertumpang tindih."
        )
    st.caption(
        caption
        + "\n\n**Keterangan kode SITC:**\n"
        + _sitc_legend_caption()
    )

    if not collisions.empty:
        with st.expander("Pelabuhan dengan koordinat identik"):
            st.write(
                "Nama pelabuhan berikut berada pada koordinat yang sama sehingga marker-nya "
                "saling menumpuk. Posisi tidak digeser agar tetap akurat secara geografis."
            )
            st.dataframe(
                collisions[["port", "Provinsi", lat_col, lon_col, "Klaster"]],
                width="stretch",
                hide_index=True,
            )



# ---------------------------------------------------------------------------
# View: Data
# ---------------------------------------------------------------------------

def _dataset_catalog(result: dict) -> dict[str, tuple[str, pd.DataFrame]]:

    pre = result["preprocessing"]
    clustered = result["modeling"]["clustered"]
    return {
        "1. Dataset input mentah (port_imports)": (
            "dataset_input_port_imports.csv",
            pre["df_imports"],
        ),
        "2. Hasil konversi HS ke SITC": ("hasil_konversi_sitc.csv", pre["df_sitc"]),
        "3. Hasil agregasi Pelabuhan × SITC": (
            "hasil_agregasi_port_sitc.csv",
            pre["df_aggregated"],
        ),
        "4. Data setelah penggabungan provinsi": (
            "data_gabungan_provinsi.csv",
            pre["df_finished"],
        ),
        "5. Dataset final prapemrosesan": ("dataset_final_preprocessing.csv", pre["df_final"]),
        "6. Dataset dengan label klaster": ("dataset_berlabel_klaster.csv", clustered),
    }


def _render_data(result: dict) -> None:
    catalog = _dataset_catalog(result)

    st.subheader("Ringkasan Statistik Tahapan Prapemrosesan")
    report = result["preprocessing"]["report"]
    conversion = report["conversion"]
    summary = {
        "Baris data impor awal": f"{report['rows_imports']:,}",
        "Kode HS-6 tidak terbaca/valid": f"{conversion['rows_invalid_hs6']:,}",
        "Transaksi tanpa pemetaan SITC": f"{conversion['rows_unmapped']:,}",
        "Nilai transaksi tanpa pemetaan (USD)": f"US$ {conversion['usd_value_unmapped']:,.2f}",
        "Baris setelah agregasi Pelabuhan × SITC": f"{report['rows_aggregated']:,}",
        "Pelabuhan tanpa pemetaan data provinsi": f"{report['province']['rows_unmapped_port']:,}",
        "Baris duplikat yang dibersihkan": f"{report['cleaning']['duplicates_removed']:,}",
        "Baris dataset final siap model": f"{report['rows_final']:,}",
    }
    st.dataframe(
        pd.DataFrame({"Tahapan": list(summary), "Nilai": list(summary.values())}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Tabel ini merangkum jumlah baris data pada setiap tahapan prapemrosesan, "
        "mulai dari data impor mentah hingga dataset final yang siap digunakan sebagai input model."
    )
    with st.expander("Detail penanganan outlier (IQR capping) dan imputasi median:"):
        st.json({"capping": report["capping"], "median_imputed": report["cleaning"]["median_imputed"]})

    st.subheader("Pratinjau dan Ekspor Dataset per Tahapan")
    choice = st.selectbox("Pilih tahapan dataset", list(catalog.keys()), key="data_dataset_choice")
    filename, df = catalog[choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", f"{len(df):,}")
    c2.metric("Jumlah Kolom", f"{len(df.columns):,}")
    c3.metric("Data Kosong (Missing)", f"{int(df.isna().sum().sum()):,}")

    max_rows = st.selectbox(
        "Jumlah baris ditampilkan", [25, 50, 100, 250, 500], index=1, key="data_max_rows"
    )
    st.dataframe(df.head(max_rows), width="stretch", hide_index=True)
    st.caption(
        f"Pratinjau menampilkan {min(max_rows, len(df)):,} baris pertama dari total {len(df):,} baris "
        f"pada tahapan '{choice}'. Gunakan tombol unduh untuk mengekspor seluruh dataset dalam format CSV."
    )

    st.download_button(
        f"Unduh berkas: {choice}",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key="download_dataset_choice",
    )

    st.subheader("Pratinjau Dataset Berlabel Klaster")
    clustered = result["modeling"]["clustered"]
    cluster_choice = st.selectbox(
        "Pilih nomor klaster", sorted(clustered["Cluster"].unique()), key="data_cluster_choice"
    )
    subset = clustered[clustered["Cluster"] == cluster_choice]
    st.dataframe(subset.head(max_rows), width="stretch", hide_index=True)
    st.caption(
        f"Pratinjau menampilkan {min(max_rows, len(subset)):,} baris pertama dari {len(subset):,} "
        f"observasi pada Klaster {cluster_choice}. Gunakan tombol unduh untuk mengekspor seluruh "
        "subset klaster dalam format CSV."
    )
    st.download_button(
        f"Unduh observasi Klaster {cluster_choice} (CSV)",
        data=subset.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"cluster_{cluster_choice}.csv",
        mime="text/csv",
        key="download_cluster_subset",
    )


# ---------------------------------------------------------------------------
# Sidebar + router
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div class='side-brand'><div class='side-brand-title'>DASHBOARD KLASTERISASI<br> KOMBINASI PELABUHAN DAN <br> KELOMPOK KOMODITAS BERDASARKAN AKTIVITAS <br> IMPOR TIONGKOK</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='side-section-label'>TAMPILAN DASHBOARD</div>", unsafe_allow_html=True)
        view = st.radio(
            "Tampilan dashboard",
            options=VIEWS,
            key="dashboard_view",
            label_visibility="collapsed",
        )
        st.markdown("<div class='side-section-label'>KONFIGURASI ANALISIS</div>", unsafe_allow_html=True)
        st.caption(f"K = {FIXED_K} · empat berkas data bawaan")
        if st.button("← Kembali ke Beranda", width="stretch"):
            st.switch_page("pages/1_Beranda.py")
        st.markdown(
            "<div class='sidebar-footer-caption'>Sistem Analitik Klasterisasi Kombinasi Pelabuhan dan Kelompok Komoditas</div>",
            unsafe_allow_html=True,
        )
    return view


def render() -> None:
    st.markdown(
        '<div class="dashboard-page-marker"></div>', unsafe_allow_html=True
    )
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    view = _render_sidebar()

    st.markdown(
        """
        <div class="app-topbar">
          <div class="app-topbar-title">Dashboard Klasterisasi Kombinasi Pelabuhan dan Kelompok Komoditas di Indonesia Berdasarkan Aktivitas Impor Tiongkok</div>
          <div class="app-topbar-subtitle">Analitik · K-Prototypes · Pelabuhan Indonesia</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    title, description = VIEW_TITLES[view]
    st.title(title)
    st.caption(description)

    result = _ensure_result()
    if not result:
        _no_results()
        return

    renderers = {
        "Ringkasan": _render_summary,
        "Profil Klaster": _render_profile,
        "Evaluasi": _render_evaluation,
        "Peta": _render_map,
        "Data": _render_data,
    }
    renderers[view](result)


render()


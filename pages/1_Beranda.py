"""Beranda (homepage) page for the deployed two-page application.

Displays the research title, a short description of the bundled datasets and a
single action button that prepares the fixed analysis (K = 5) and opens the
Dashboard directly. No upload or cluster-count configuration happens here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.analysis import get_fixed_analysis
from src.config import FIXED_K


def _open_dashboard() -> None:
    """Prepare the cached analysis result and switch to the Dashboard."""
    try:
        with st.spinner(
            f"Menyiapkan hasil analisis dari empat berkas data bawaan (K = {FIXED_K}). "
            "Proses pertama dapat memerlukan beberapa menit..."
        ):
            result = get_fixed_analysis(FIXED_K)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
        st.session_state["processing_error"] = str(exc)
        return

    st.session_state["analysis_result"] = result
    st.session_state["pipeline_summary"] = result["preprocessing"]["report"]
    st.session_state["processing_error"] = None
    st.switch_page("pages/2_Dashboard.py")


def render() -> None:
    st.markdown(
        """
        <section class="hero-panel">
          <div class="hero-title">KLASTERISASI KOMBINASI PELABUHAN DAN KELOMPOK KOMODITAS DI INDONESIA BERDASARKAN KARAKTERISTIK IMPOR TIONGKOK DENGAN MENGGUNAKAN METODE K-PROTOTYPES</div>
          <div class="hero-subtitle">Analitik · K-Prototypes · Pelabuhan Indonesia</div>
          <div class="hero-copy">
            Aplikasi analitik pemetaan dan pengelompokan (klasterisasi) kombinasi pelabuhan dan kelompok komoditas di 
            Indonesia berdasarkan karakteristik aktivitas impor dari Tiongkok serta kondisi makroekonomi dari masing-masing 
            wilayah pelabuhan. Sistem yang dikembangkan adalah dashboard analitik berbasis web yang dapat memberikan 
            gambaran terhadap karakteristik, pola, dan perbedaan antarwilayah melalui visualisasi hasil klasterisasi 
            interaktif. Selain itu, sistem juga mendukung proses identifikasi profil setiap kelompok secara objektif dan 
            terukur berdasarkan sejumlah indikator ekonomi dan logistik yang digunakan dalam analisis.
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="section-card">
              <h4>01 · Data Bawaan</h4>
              <p>Empat berkas data wajib — data transaksi impor, data makro provinsi, tabel konversi HS–SITC,
              dan data geolokasi pelabuhan — telah tersedia di dalam aplikasi. Jumlah klaster ditetapkan
              sebesar K = 5.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
              <h4>02 · Proses Klasterisasi</h4>
              <p>Prapemrosesan otomatis, pemodelan K-Prototypes, evaluasi Elbow &amp; Silhouette campuran, hingga
              pelatihan model surrogate Random Forest dan analisis SHAP.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="section-card">
              <h4>03 · Eksplorasi Hasil</h4>
              <p>Telaah wawasan mendalam melalui menu Ringkasan, Profil Klaster, Evaluasi Model, Distribusi Spasial,
              dan Unduhan Data pada dashboard interaktif.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    _spacer_left, center, _spacer_right = st.columns([1, 1.4, 1])
    with center:
        if st.button("Lihat Hasil Analisis", type="primary", width="stretch"):
            _open_dashboard()

    if st.session_state.get("processing_error"):
        st.error(
            "Terjadi kendala saat menyiapkan hasil analisis: "
            f"{st.session_state['processing_error']}"
        )

    st.caption(
        "🔒 Seluruh data yang ada pada aplikasi diproses secara lokal sehingga tidak ada informasi yang "
        "dikirim ke server eksternal. Aplikasi menggunakan empat berkas data bawaan pada folder "
        "dataset_input dengan konfigurasi jumlah klaster K = 5. Tekan tombol Lihat Hasil Analisis untuk "
        "menjalankan prapemrosesan, pemodelan K-Prototypes, evaluasi Silhouette, model surrogate Random "
        "Forest, dan analisis SHAP, lalu langsung membuka dashboard."
    )


render()

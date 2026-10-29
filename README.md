# China Imports Port Clustering App — Deployed

Versi deployment dua halaman dari aplikasi klasterisasi kombinasi pelabuhan dan
kelompok komoditas impor Tiongkok. Aplikasi ini menggunakan empat berkas data
yang telah disertakan, hasil analisis precomputed, dan konfigurasi jumlah
klaster tetap **K = 5**.

## Halaman

1. **Beranda** — menampilkan judul penelitian, ringkasan aplikasi, dan tombol
   **Lihat Hasil Analisis**.
2. **Dashboard** — menampilkan hasil analisis melalui lima tampilan internal:
   Ringkasan, Profil Klaster, Evaluasi, Peta, dan Data.

Tidak ada halaman Input Data. Tombol pada Beranda memuat artifact analisis yang
sudah jadi, lalu langsung berpindah ke Dashboard tanpa menjalankan ulang
preprocessing atau modelling.

## Analisis Precomputed

Hasil analisis dibangun **sekali** ke folder `artifacts/k5/` melalui:

```bash
.venv/bin/python scripts/build_analysis_artifacts.py --force
```

Build script menjalankan preprocessing, K-Prototypes K = 5, Elbow, Silhouette,
Random Forest surrogate, SHAP, dan silhouette per observasi. Setelah selesai,
artifact diverifikasi dengan cara dibaca kembali.

Saat aplikasi Streamlit berjalan, `src/analysis.py` hanya memuat artifact
tersebut dengan `st.cache_data`. Tidak ada pelatihan model pada runtime,
sehingga hasil dapat dibuka dalam hitungan detik dan tetap tersedia setelah
server restart.

Artifact disimpan sebagai Parquet, CSV, dan JSON — bukan pickle model — agar:

- portabel dan mudah diperiksa,
- tidak bergantung pada kompatibilitas pickle scikit-learn/kmodes,
- ukurannya relatif kecil,
- dapat dimuat ulang kapan saja tanpa menjalankan pipeline.

`manifest.json` menyimpan versi schema, K, waktu build, hash keempat berkas
input, versi library, dan ringkasan hasil. Jika `dataset_input` berubah,
aplikasi akan menolak artifact lama dan meminta build ulang.

## Struktur Folder

```text
China-Imports-Port-Clustering-App-Deployed/
├── app.py                          # Entry point dan routing dua halaman
├── pages/
│   ├── 1_Beranda.py                # Halaman beranda
│   └── 2_Dashboard.py              # Dashboard lima tampilan
├── src/
│   ├── analysis.py                 # Build pipeline + loader artifact
│   ├── artifacts.py                # Tulis/baca/validasi artifact
│   ├── config.py                   # Konfigurasi, termasuk FIXED_K = 5
│   ├── modeling.py                 # K-Prototypes, surrogate, SHAP
│   ├── preprocessing.py            # Prapemrosesan HS–SITC dan provinsi
│   └── visualization.py            # Visualisasi Plotly/Matplotlib
├── scripts/
│   └── build_analysis_artifacts.py # Pembangun artifact sekali jalan
├── artifacts/
│   └── k5/                         # Artifact precomputed (dikomit ke Git)
│       ├── manifest.json
│       ├── preprocessing/
│       └── modeling/
├── dataset_input/
│   ├── port_imports.csv
│   ├── provinces_data.csv
│   ├── HS2022_to_SITC_Conversion.xlsx
│   └── port_geolocation.csv
├── .streamlit/config.toml
├── requirements.txt
└── .venv/                          # Virtual environment (sudah terpasang)
```

Folder `artifacts/` **tidak diabaikan** oleh `.gitignore`; artifact memang
disertakan dalam repository agar deployment langsung dapat memuat hasil.

## Menjalankan Aplikasi

Virtual environment sudah dibuat dan seluruh paket pada `requirements.txt`
sudah terpasang.

```bash
cd "China-Imports-Port-Clustering-App-Deployed"
source .venv/bin/activate
streamlit run app.py
```

Atau tanpa mengaktifkan venv:

```bash
cd "China-Imports-Port-Clustering-App-Deployed"
.venv/bin/streamlit run app.py
```

Buka alamat yang ditampilkan Streamlit (default: `http://localhost:8501`).

Jika folder aplikasi dipindahkan ke lokasi lain, virtual environment perlu dibuat
ulang karena path di dalamnya bersifat absolut:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Catatan Analisis

- Prapemrosesan memakai keempat berkas data bawaan: transaksi impor, indikator
  makro provinsi, tabel konversi HS 2022 → SITC Rev. 4, dan geolokasi pelabuhan.
- Model K-Prototypes dilatih dengan `K = 5`, `init="Huang"`, `n_init=5`, dan
  `random_state=42`.
- Build artifact memerlukan sekitar satu menit. Setelah artifact tersedia,
  aplikasi hanya membaca file dan tidak menghitung ulang.
- Seluruh data diproses secara lokal; tidak ada berkas yang dikirim ke server
  eksternal.

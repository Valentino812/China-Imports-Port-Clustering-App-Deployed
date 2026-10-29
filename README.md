# China Imports Port Clustering App — Deployed

Versi deployment dua halaman dari aplikasi klasterisasi kombinasi pelabuhan dan
kelompok komoditas impor Tiongkok. Aplikasi ini menggunakan empat berkas data
yang telah disertakan dan konfigurasi jumlah klaster tetap **K = 5**.

## Halaman

1. **Beranda** — menampilkan judul penelitian, ringkasan aplikasi, dan tombol
   **Lihat Hasil Analisis**.
2. **Dashboard** — menampilkan hasil analisis melalui lima tampilan internal:
   Ringkasan, Profil Klaster, Evaluasi, Peta, dan Data.

Tidak ada halaman Input Data. Tombol pada Beranda langsung menyiapkan hasil
analisis dari data bawaan, lalu berpindah ke Dashboard.

## Struktur Folder

```
China-Imports-Port-Clustering-App-Deployed/
├── app.py                      # Entry point dan routing dua halaman
├── pages/
│   ├── 1_Beranda.py            # Halaman beranda
│   └── 2_Dashboard.py          # Dashboard lima tampilan
├── src/
│   ├── analysis.py             # Pipeline analisis data bawaan + cache
│   ├── config.py               # Konfigurasi, termasuk FIXED_K = 5
│   ├── modeling.py             # K-Prototypes, surrogate, SHAP
│   ├── preprocessing.py        # Prapemrosesan HS–SITC dan provinsi
│   └── visualization.py        # Fungsi visualisasi Plotly/Matplotlib
├── dataset_input/
│   ├── port_imports.csv
│   ├── provinces_data.csv
│   ├── HS2022_to_SITC_Conversion.xlsx
│   └── port_geolocation.csv
├── .streamlit/config.toml
├── requirements.txt
└── .venv/                      # Virtual environment (sudah terpasang)
```

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
- Proses pertama memerlukan sekitar satu menit (Elbow K=2–8, K-Prototypes,
  Silhouette, Random Forest surrogate, dan SHAP). Hasil disimpan dengan
  `st.cache_resource` sehingga navigasi berikutnya langsung ditampilkan.
- Seluruh data diproses secara lokal; tidak ada berkas yang dikirim ke server
  eksternal.

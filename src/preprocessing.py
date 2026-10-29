"""Preprocessing pipeline for China-import port clustering.

The functions here are a Python translation of ``process_datasets.ipynb`` and
``process_provinces.ipynb``. They accept the four mandatory inputs
(``port_imports.csv``, ``provinces_data.csv`` and the HS 2022 -> SITC Rev. 4
conversion table, plus ``port_geolocation.csv``) and produce the raw-scale
dataset that is used for descriptive cluster profiling.

Design notes
------------
* The output of this module keeps features in their **original units**. The
  single standardisation used by K-Prototypes happens later, in
  :mod:`src.modeling`, so descriptive statistics remain interpretable.
* ``usd_value`` is IQR-capped per SITC group with a multiplier of 3, matching
  the notebook. Other numeric features are capped globally with the same
  multiplier. The outdated column name ``small_micro_businesses`` is replaced by
  the actual ``micro_small_industries``.
* If a HS6 code maps to more than one SITC code the pipeline stops instead of
  silently duplicating import values.
* Port -> province information and coordinates come from the mandatory
  ``port_geolocation.csv`` input. The legacy static mapping is kept commented
  out only as historical documentation and is no longer used.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

IMPORTS_REQUIRED = ["port", "hs_code", "usd_value"]
PROVINCES_REQUIRED = [
    "Provinsi",
    "pdrb",
    "china_investment_us_ribu",
    "micro_small_industries",
]
GEOLOCATION_REQUIRED = ["port", "Provinsi", "latitude", "longitude"]
CONVERSION_REQUIRED = ["From HS 2022", "To SITC Rev. 4"]

CATEGORICAL_COLS = ["sitc"]
NUMERICAL_COLS = [
    "usd_value",
    "pdrb",
    "china_investment_us_ribu",
    "micro_small_industries",
]
METADATA_COLS = ["port", "Provinsi", "latitude", "longitude"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS
FINAL_COLS = METADATA_COLS + FEATURE_COLS

IQR_MULTIPLIER = 3
OTHER_NUMERIC_COLS = ["pdrb", "china_investment_us_ribu", "micro_small_industries"]

# DEPRECATED: pemetaan statis di bawah ini tidak lagi dipakai. Informasi provinsi
# pelabuhan kini berasal dari berkas input port_geolocation.csv. Blok dipertahankan
# sebagai dokumentasi historis dan tidak dihapus.
# Port -> province mapping taken verbatim from process_datasets.ipynb.
# PORT_PROV_MAP: dict[str, str] = {
#     'Aceh - Sabang': 'Aceh',
#     'Achmad Yani': 'Maluku Utara',
#     'Achmad Yani (U)': 'Jawa Tengah',
#     'Air Sugihan': 'Sumatera Selatan',
#     'Amamapare': 'Papua Tengah',
#     'Amamapare Ij': 'Papua Tengah',
#     'Ambon': 'Maluku',
#     'Anggrek': 'Gorontalo',
#     'Bahudopi': 'Sulawesi Tengah',
#     'Balantang': 'Sulawesi Selatan',
#     'Balantang/Malili': 'Sulawesi Selatan',
#     'Balikpapan': 'Kalimantan Timur',
#     'Banjarmasin': 'Kalimantan Selatan',
#     'Banyuwangi': 'Jawa Timur',
#     'Batam Island': 'Kepulauan Riau',
#     'Batu Ampar': 'Kepulauan Riau',
#     'Batu Licin': 'Kalimantan Selatan',
#     'Batu Licin (U)': 'Kalimantan Selatan',
#     'Bekasi': 'Jawa Barat',
#     'Belakang Padang': 'Kepulauan Riau',
#     'Belawan': 'Sumatera Utara',
#     'Benete': 'Nusa Tenggara Barat',
#     'Benoa': 'Bali',
#     'Benoa/Loloan': 'Bali',
#     'Berau': 'Kalimantan Timur',
#     'Berau (U)': 'Kalimantan Timur',
#     'Bintuni': 'Papua Barat',
#     'Bintuni Irian Jaya': 'Papua Barat',
#     'Bitung': 'Sulawesi Utara',
#     'Bogor': 'Jawa Barat',
#     'Bontang': 'Kalimantan Timur',
#     'Bonthan Bay Sulawesi': 'Sulawesi Selatan',
#     'Buatan': 'Riau',
#     'Cengkareng': 'Banten',
#     'Cigading': 'Banten',
#     'Cilacap': 'Jawa Tengah',
#     'Cirebon': 'Jawa Barat',
#     'Ciwandan': 'Banten',
#     'Donggala': 'Sulawesi Tengah',
#     'Dumai': 'Riau',
#     'Frans Kasiepo': 'Papua',
#     'Frans Kasiepo (U)': 'Papua',
#     'Futong Terminal': 'Riau',
#     'Gede Bage-Bandung (Ptt)': 'Jawa Barat',
#     'Gedebage Bandung': 'Jawa Barat',
#     'Gorontalo': 'Gorontalo',
#     'Gresik': 'Jawa Timur',
#     'Halim Perdana Kusuma': 'DKI Jakarta',
#     'Halim Perdana Kusuma (U)': 'DKI Jakarta',
#     'Hang Nadim': 'Kepulauan Riau',
#     'Hang Nadim (U)': 'Kepulauan Riau',
#     'Hasanuddin': 'Sulawesi Selatan',
#     'Hasanuddin (U)': 'Sulawesi Selatan',
#     'Husein Sastranegara': 'Jawa Barat',
#     'Husein Sastranegara (U)': 'Jawa Barat',
#     'Jababeka': 'Jawa Barat',
#     'Jakarta / Pasar Ikan': 'DKI Jakarta',
#     'Jakarta Pasar Ikan': 'DKI Jakarta',
#     'Jakarta Pos Pasar Baru': 'DKI Jakarta',
#     'Jambi': 'Jambi',
#     'Jatty Baru': 'Kepulauan Riau',
#     'Jayapura': 'Papua',
#     'Jayapura / Sentani (U)': 'Papua',
#     'Jkt./ Pos Pasar Baru': 'DKI Jakarta',
#     'Juanda (U)-Surabaya': 'Jawa Timur',
#     'Juanda Surabaya': 'Jawa Timur',
#     'Juata Tarakan': 'Kalimantan Utara',
#     'Kabil Panau': 'Kepulauan Riau',
#     'Kabil/Panau': 'Kepulauan Riau',
#     'Kalbut Situbondo': 'Jawa Timur',
#     'Kalianget': 'Jawa Timur',
#     'Kaliorang': 'Kalimantan Timur',
#     'Kariangau': 'Kalimantan Timur',
#     'Kendari': 'Sulawesi Tenggara',
#     'Kendawangan': 'Kalimantan Barat',
#     'Ketapang Barat': 'Kalimantan Barat',
#     'Ketapang K. Barat': 'Kalimantan Barat',
#     'Kijang': 'Kepulauan Riau',
#     'Kolaka': 'Sulawesi Tenggara',
#     'Kolonodale': 'Sulawesi Tengah',
#     'Kotabaru': 'Kalimantan Selatan',
#     'Kuala Enok': 'Riau',
#     'Kuala Namu': 'Sumatera Utara',
#     'Kuala Namu International Airport (U)': 'Sumatera Utara',
#     'Kuala Tanjung': 'Sumatera Utara',
#     'Kuala Tungkal': 'Jambi',
#     'Kumai': 'Kalimantan Tengah',
#     'Lhok Seumawe': 'Aceh',
#     'Lingkas Tarakan': 'Kalimantan Utara',
#     'Lobam': 'Kepulauan Riau',
#     'Luwuk': 'Sulawesi Tengah',
#     'Madiun': 'Jawa Timur',
#     'Madiun /Iswahyudi (U)': 'Jawa Timur',
#     'Makassar': 'Sulawesi Selatan',
#     'Malili': 'Sulawesi Selatan',
#     'Malili Sulawesi': 'Sulawesi Selatan',
#     'Mamuju': 'Sulawesi Barat',
#     'Manado': 'Sulawesi Utara',
#     'Medan': 'Sumatera Utara',
#     'Medan (Ptt)': 'Sumatera Utara',
#     'Mempawah': 'Kalimantan Barat',
#     'Merak': 'Banten',
#     'Merauke': 'Papua Selatan',
#     'Morowali': 'Sulawesi Tengah',
#     'Muara Berau': 'Kalimantan Timur',
#     'Musi River': 'Sumatera Selatan',
#     'Musi River/Boom Baru': 'Sumatera Selatan',
#     'Ngurah Rai': 'Bali',
#     'Ngurah Rai (U)': 'Bali',
#     'Nunukan': 'Kalimantan Utara',
#     'Obi Island': 'Maluku Utara',
#     'Padang': 'Sumatera Barat',
#     'Padang / Tabing (U)': 'Sumatera Barat',
#     'Palembang': 'Sumatera Selatan',
#     'Palembang - Plaju': 'Sumatera Selatan',
#     'Palimanan': 'Jawa Barat',
#     'Paliminan': 'Jawa Barat',
#     'Palmerah/Sultan Thaha (U)': 'Jambi',
#     'Palopo': 'Sulawesi Selatan',
#     'Pangkal Balam': 'Kepulauan Bangka Belitung',
#     'Pangkalan Bun': 'Kalimantan Tengah',
#     'Panipahan': 'Riau',
#     'Panjang': 'Lampung',
#     'Pantoloan': 'Sulawesi Tengah',
#     'Pare-Pare': 'Sulawesi Selatan',
#     'Pasuruan': 'Jawa Timur',
#     'Patimban': 'Jawa Barat',
#     'Pekan Baru': 'Riau',
#     'Perawang': 'Riau',
#     'Pomalaa': 'Sulawesi Tenggara',
#     'Pontianak': 'Kalimantan Barat',
#     'Poso': 'Sulawesi Tengah',
#     'Probolinggo': 'Jawa Timur',
#     'Pulang Pisau': 'Kalimantan Tengah',
#     'Pulau Laut': 'Kalimantan Selatan',
#     'Pulau Sambu': 'Kepulauan Riau',
#     'Purwakarta': 'Jawa Barat',
#     'Rengat': 'Riau',
#     'SM Badaruddin': 'Sumatera Selatan',
#     'Sam Ratulangi': 'Sulawesi Utara',
#     'Sam Ratulangi (U)': 'Sulawesi Utara',
#     'Samarinda': 'Kalimantan Timur',
#     'Samsudin Noor': 'Kalimantan Selatan',
#     'Samsudin Noor (U)': 'Kalimantan Selatan',
#     'Sangkulirang': 'Kalimantan Timur',
#     'Sekupang': 'Kepulauan Riau',
#     'Selat Panjang': 'Riau',
#     'Semarang': 'Jawa Tengah',
#     'Semarang (Ptt)': 'Jawa Tengah',
#     'Sepinggan': 'Kalimantan Timur',
#     'Sepinggan (U)': 'Kalimantan Timur',
#     'Serui': 'Papua',
#     'Siak Sri Indrapura': 'Riau',
#     'Simpang Tiga (U)': 'Riau',
#     'Sinmpang Tiga': 'Riau',
#     'Sm. Badaruddin (U)': 'Sumatera Selatan',
#     'Soekarno Hatta': 'Banten',
#     'Soekarno-Hatta (U)': 'Banten',
#     'Solo': 'Jawa Tengah',
#     'Solo/Jebres/Adi Sumarmo (U)': 'Jawa Tengah',
#     'Sorong': 'Papua Barat Daya',
#     'Sultan Thaha': 'Jambi',
#     'Sungai Guntung': 'Riau',
#     'Sungsang': 'Sumatera Selatan',
#     'Supadio': 'Kalimantan Barat',
#     'Supadio (U)': 'Kalimantan Barat',
#     'Surabaya': 'Jawa Timur',
#     'Surabaya (Ptt)': 'Jawa Timur',
#     'Taliabu': 'Maluku Utara',
#     'Tangerang': 'Banten',
#     'Tanjung Balai Karimun': 'Kepulauan Riau',
#     'Tanjung Batu': 'Kepulauan Riau',
#     'Tanjung Batu Riau': 'Kepulauan Riau',
#     'Tanjung Beringin': 'Sumatera Utara',
#     'Tanjung Buli': 'Maluku Utara',
#     'Tanjung Emas': 'Jawa Tengah',
#     'Tanjung Leneng': 'Banten',
#     'Tanjung Perak': 'Jawa Timur',
#     'Tanjung Pinang': 'Kepulauan Riau',
#     'Tanjung Priok': 'DKI Jakarta',
#     'Tanjung Sangata': 'Kalimantan Timur',
#     'Tanjung Selor': 'Kalimantan Utara',
#     'Tanjung Uban': 'Kepulauan Riau',
#     'Tarakan': 'Kalimantan Utara',
#     'Tarakan (U)': 'Kalimantan Utara',
#     'Tasikmalaya': 'Jawa Barat',
#     'Telok Melano': 'Kalimantan Barat',
#     'Tembilahan': 'Riau',
#     'Tenau': 'Nusa Tenggara Timur',
#     'Ternate': 'Maluku Utara',
#     'Tobelo': 'Maluku Utara',
#     'Tuban': 'Jawa Timur',
#     'Tuban Jv': 'Jawa Timur',
#     'Wahai': 'Maluku',
#     'Weda': 'Maluku Utara',
#     'Yogyakarta': 'DI Yogyakarta',
#     'Yogyakarta International Airport': 'DI Yogyakarta',
# }


# ---------------------------------------------------------------------------
# File reading and validation
# ---------------------------------------------------------------------------

def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read a CSV/XLSX upload and normalise the column names.

    Values are read as text on purpose: HS codes and conversion tables rely on
    leading zeros (for example ``010121``), which numeric inference would drop.
    Numeric columns are coerced explicitly later in the pipeline.
    """
    suffix = filename.lower().rsplit(".", 1)[-1]
    buffer = BytesIO(file_bytes)
    if suffix == "csv":
        try:
            df = pd.read_csv(buffer, dtype=str)
        except UnicodeDecodeError:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding="latin-1", dtype=str)
    elif suffix in {"xlsx", "xlsm"}:
        df = pd.read_excel(buffer, dtype=str)
    elif suffix == "xls":
        raise ValueError("Format .xls lama belum didukung. Simpan file sebagai .xlsx atau .csv.")
    else:
        raise ValueError("Format file belum didukung. Unggah file .csv atau .xlsx.")

    if df.empty or len(df.columns) == 0:
        raise ValueError("Dataset kosong atau tidak memiliki kolom.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalise_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> tuple[bool, str]:
    """Check that ``required`` columns exist (case-insensitive)."""
    if df is None:
        return False, f"{label} belum tersedia."
    lookup = {_normalise_name(c): c for c in df.columns}
    missing = [c for c in required if _normalise_name(c) not in lookup]
    if missing:
        return False, f"Kolom wajib pada {label} tidak ditemukan: {missing}."
    return True, f"{label} memiliki kolom yang dibutuhkan ({len(df):,} baris)."


def align_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Return a copy whose required columns use the canonical names."""
    lookup = {_normalise_name(c): c for c in df.columns}
    out = df.copy()
    for canonical in required:
        actual = lookup[_normalise_name(canonical)]
        if actual != canonical:
            out = out.rename(columns={actual: canonical})
    return out


# ---------------------------------------------------------------------------
# HS -> SITC conversion
# ---------------------------------------------------------------------------

def extract_hs6(hs_series: pd.Series) -> pd.Series:
    """Extract a zero-padded 6-digit HS6 code following the notebook pattern."""
    hs_text = hs_series.astype("string").str.strip()
    hs_bracket = hs_text.str.extract(r"\[(\d+)\]")[0]
    hs_anywhere = hs_text.str.extract(r"(\d{6,})")[0]
    return (
        hs_bracket.fillna(hs_anywhere)
        .str.replace(r"\D", "", regex=True)
        .str[:6]
        .str.zfill(6)
    )


def build_hs_sitc_mapping(conversion_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the HS6 -> SITC mapping and refuse ambiguous mappings."""
    valid, message = validate_columns(conversion_df, CONVERSION_REQUIRED, "tabel konversi")
    if not valid:
        raise ValueError(message)

    df = align_columns(conversion_df, CONVERSION_REQUIRED)
    hs6 = (
        df["From HS 2022"]
        .astype("string")
        .str.extract(r"(\d{6,})")[0]
        .str[:6]
        .str.zfill(6)
    )
    sitc = df["To SITC Rev. 4"].astype("string").str.extract(r"(\d+)")[0]

    mapping = (
        pd.DataFrame({"hs6": hs6, "sitc": sitc})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if mapping.empty:
        raise ValueError("Tabel konversi tidak menghasilkan pemetaan HS6 -> SITC yang valid.")

    multi = mapping.groupby("hs6")["sitc"].nunique()
    multi = multi[multi > 1]
    diagnostics: dict[str, Any] = {
        "unique_hs6": int(mapping["hs6"].nunique()),
        "ambiguous_hs6": int(multi.shape[0]),
    }
    if not multi.empty:
        examples = mapping[mapping["hs6"].isin(multi.index)].sort_values(["hs6", "sitc"])
        raise ValueError(
            "Ditemukan HS6 yang dipetakan ke beberapa kode SITC. Proses dihentikan agar "
            "nilai impor tidak terhitung ganda. Periksa pemetaan berikut:\n"
            + examples.head(20).to_string(index=False)
        )
    return mapping, diagnostics


def convert_hs_to_sitc(
    imports_df: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join port imports with the SITC mapping and report unmapped rows."""
    valid, message = validate_columns(imports_df, IMPORTS_REQUIRED, "port_imports")
    if not valid:
        raise ValueError(message)

    df = align_columns(imports_df, IMPORTS_REQUIRED).copy()
    df["usd_value"] = pd.to_numeric(df["usd_value"], errors="coerce")
    df["hs6"] = extract_hs6(df["hs_code"])

    invalid_hs = int(df["hs6"].isna().sum())
    valid_rows = df.dropna(subset=["hs6", "usd_value"]).copy()

    merged = valid_rows.merge(
        mapping, on="hs6", how="left", validate="many_to_one", indicator=True
    )
    unmapped_mask = merged["_merge"].eq("left_only")
    unmapped_rows = int(unmapped_mask.sum())
    unmapped_value = float(merged.loc[unmapped_mask, "usd_value"].sum())

    df_sitc = (
        merged.loc[~unmapped_mask, ["port", "sitc", "usd_value"]]
        .dropna(subset=["sitc", "usd_value"])
        .reset_index(drop=True)
    )

    report = {
        "rows_invalid_hs6": invalid_hs,
        "rows_unmapped": unmapped_rows,
        "usd_value_unmapped": unmapped_value,
        "rows_after_conversion": int(len(df_sitc)),
    }
    return df_sitc, report


def aggregate_sitc_first_digit(df_sitc: pd.DataFrame) -> pd.DataFrame:
    """Keep the first SITC digit and aggregate ``usd_value`` per port x SITC."""
    df = df_sitc.copy()
    df["sitc"] = df["sitc"].astype("string").str.strip()
    df["usd_value"] = pd.to_numeric(df["usd_value"], errors="coerce")

    valid_mask = df["sitc"].str.fullmatch(r"\d+", na=False)
    df = df.loc[valid_mask].dropna(subset=["port", "usd_value"]).copy()
    df["sitc"] = df["sitc"].str[0]

    grouped = (
        df.groupby(["port", "sitc"], as_index=False)["usd_value"]
        .sum()
        .sort_values(by=["port", "sitc"])
        .reset_index(drop=True)
    )
    if grouped.duplicated(subset=["port", "sitc"]).any():
        raise ValueError("Masih ditemukan duplikat kombinasi Port x SITC.")
    return grouped


# ---------------------------------------------------------------------------
# Province mapping and cleaning
# ---------------------------------------------------------------------------

def clean_province(value: Any) -> Any:
    """Standardise province names exactly as in the notebook."""
    if pd.isna(value):
        return value
    text = str(value).strip().title()
    text = text.replace("Daerah Khusus Ibukota Jakarta", "Dki Jakarta")
    text = text.replace("Daerah Istimewa Yogyakarta", "Di Yogyakarta")
    text = text.replace("Kep. Bangka Belitung", "Kepulauan Bangka Belitung")
    text = text.replace("Kep. Riau", "Kepulauan Riau")
    return text


def prepare_geolocation(
    geolocation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and normalise the ``port_geolocation.csv`` reference table.

    This table is the single source of truth for a port's province and
    coordinates, replacing the legacy static mapping. Failures are raised
    instead of silently dropping rows so no import is profiled without a
    province and a valid location.
    """
    valid, message = validate_columns(
        geolocation_df, GEOLOCATION_REQUIRED, "port_geolocation"
    )
    if not valid:
        raise ValueError(message)

    geo = align_columns(geolocation_df, GEOLOCATION_REQUIRED).copy()
    geo["port"] = geo["port"].astype("string").str.strip()
    geo["Provinsi"] = (
        geo["Provinsi"].astype("string").str.strip().apply(clean_province)
    )
    geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
    geo = geo.replace({"port": {"": pd.NA}, "Provinsi": {"": pd.NA}})

    missing_mask = geo[GEOLOCATION_REQUIRED].isna().any(axis=1)
    if missing_mask.any():
        row_numbers = ", ".join(str(i + 2) for i in geo.index[missing_mask][:20])
        raise ValueError(
            "port_geolocation memiliki nilai kosong atau tidak valid pada kolom wajib "
            f"(baris data: {row_numbers}). Perbaiki berkas lalu unggah ulang."
        )

    duplicates = sorted(geo.loc[geo["port"].duplicated(keep=False), "port"].unique())
    if duplicates:
        raise ValueError(
            "port_geolocation memuat nama pelabuhan duplikat: "
            + ", ".join(map(str, duplicates[:20]))
            + (f" (dan {len(duplicates) - 20} lainnya)" if len(duplicates) > 20 else "")
        )

    out_of_range = geo["latitude"].abs().gt(90) | geo["longitude"].abs().gt(180)
    if out_of_range.any():
        row_numbers = ", ".join(str(i + 2) for i in geo.index[out_of_range][:20])
        raise ValueError(
            "port_geolocation memiliki koordinat di luar rentang yang wajar "
            f"(latitude -90..90, longitude -180..180) pada baris data: {row_numbers}."
        )

    report = {
        "rows_geolocation": int(len(geo)),
        "ports_with_coordinates": int(geo["port"].nunique()),
        "provinces_covered": int(geo["Provinsi"].nunique()),
    }
    return geo[GEOLOCATION_REQUIRED].reset_index(drop=True), report


def merge_provinces(
    grouped_df: pd.DataFrame,
    provinces_df: pd.DataFrame,
    geolocation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach province, coordinates and macro indicators to the aggregates.

    Province and coordinates come from ``port_geolocation.csv``. A port without
    a matching geolocation row aborts the pipeline with a clear message rather
    than producing a row with a missing province.
    """
    valid, message = validate_columns(provinces_df, PROVINCES_REQUIRED, "provinces_data")
    if not valid:
        raise ValueError(message)

    geo, geolocation_report = prepare_geolocation(geolocation_df)

    df = grouped_df.copy()
    df["port"] = df["port"].astype("string").str.strip()

    missing_ports = sorted(set(df["port"].dropna()) - set(geo["port"]))
    if missing_ports:
        listed = ", ".join(map(str, missing_ports[:20]))
        remainder = (
            f" (dan {len(missing_ports) - 20} lainnya)"
            if len(missing_ports) > 20
            else ""
        )
        raise ValueError(
            f"{len(missing_ports)} pelabuhan tidak memiliki baris pada port_geolocation: "
            f"{listed}{remainder}. Lengkapi berkas port_geolocation sebelum analisis."
        )

    merged = df.merge(geo, on="port", how="left", validate="many_to_one")

    prov = align_columns(provinces_df, PROVINCES_REQUIRED).copy()
    prov["Provinsi"] = prov["Provinsi"].apply(clean_province)
    prov = prov.drop_duplicates(subset=["Provinsi"])

    missing_provinces = sorted(set(merged["Provinsi"].dropna()) - set(prov["Provinsi"]))
    if missing_provinces:
        raise ValueError(
            "Provinsi berikut tidak ditemukan pada provinces_data: "
            + ", ".join(map(str, missing_provinces[:20]))
            + ". Lengkapi indikator makro provinsi sebelum analisis."
        )

    merged = merged.merge(
        prov[["Provinsi", "pdrb", "china_investment_us_ribu", "micro_small_industries"]],
        on="Provinsi",
        how="left",
    )
    report = {
        "rows_unmapped_port": 0,
        "unmapped_ports": [],
        "provinces_available": int(prov["Provinsi"].nunique()),
        "provinces_without_macro": [],
        **geolocation_report,
    }
    return merged, report


def clean_and_impute(df_finished: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop duplicates, normalise text, coerce numbers and impute medians."""
    df = df_finished.copy()
    rows_before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(df)

    df["port"] = df["port"].astype(str).str.strip()
    df["Provinsi"] = df["Provinsi"].astype(str).str.strip()
    df["sitc"] = df["sitc"].astype(str)

    missing_before = {c: int(df[c].isna().sum()) for c in NUMERICAL_COLS}
    imputed: dict[str, float] = {}
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in NUMERICAL_COLS:
        if df[col].isna().sum() > 0:
            median_val = float(df[col].median())
            df[col] = df[col].fillna(median_val)
            imputed[col] = median_val

    report = {
        "rows_before_cleaning": rows_before,
        "rows_after_dedup": rows_after_dedup,
        "duplicates_removed": rows_before - rows_after_dedup,
        "missing_numeric_before": missing_before,
        "median_imputed": imputed,
    }
    return df, report


def cap_outliers(series: pd.Series, multiplier: int = IQR_MULTIPLIER) -> pd.Series:
    """Winsorise a series using Q1 - m*IQR and Q3 + m*IQR bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return np.clip(series, q1 - multiplier * iqr, q3 + multiplier * iqr)


def apply_iqr_capping(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """IQR cap every numeric feature and report the affected rows.

    ``usd_value`` is capped per SITC, while the other numeric features are
    capped globally. The report includes one entry per feature plus the number
    of unique rows affected by at least one feature.
    """
    out = df.copy()

    before_usd = out["usd_value"].copy()
    out["usd_value"] = out.groupby("sitc")["usd_value"].transform(cap_outliers)
    masks = {
        "usd_value": (before_usd != out["usd_value"]) & before_usd.notna()
    }

    for col in OTHER_NUMERIC_COLS:
        before = out[col].copy()
        out[col] = cap_outliers(out[col])
        masks[col] = (before != out[col]) & before.notna()

    report: dict[str, Any] = {
        col: int(mask.sum()) for col, mask in masks.items()
    }
    report["rows_any_capped"] = int(
        pd.concat(masks, axis=1).any(axis=1).sum()
    )
    report["total_rows_capped_usd_value"] = report["usd_value"]
    return out, report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_preprocessing(
    imports_df: pd.DataFrame,
    provinces_df: pd.DataFrame,
    conversion_df: pd.DataFrame,
    geolocation_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run the complete preprocessing chain and return frames plus a report."""
    mapping, conversion_diag = build_hs_sitc_mapping(conversion_df)
    df_sitc, conversion_report = convert_hs_to_sitc(imports_df, mapping)
    if df_sitc.empty:
        raise ValueError(
            "Tidak ada baris impor yang berhasil dipetakan ke SITC. Periksa tabel konversi "
            "dan kolom hs_code pada port_imports."
        )

    df_aggregated = aggregate_sitc_first_digit(df_sitc)
    df_finished, province_report = merge_provinces(
        df_aggregated, provinces_df, geolocation_df
    )
    df_cleaned, clean_report = clean_and_impute(df_finished)
    df_final, cap_report = apply_iqr_capping(df_cleaned)

    missing_columns = [c for c in FINAL_COLS if c not in df_final.columns]
    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan pada dataset akhir: "
            + ", ".join(missing_columns)
        )
    df_final = df_final[FINAL_COLS].copy()

    missing_metadata = [
        c for c in ["Provinsi", "latitude", "longitude"] if df_final[c].isna().any()
    ]
    if missing_metadata:
        raise ValueError(
            "Dataset akhir masih memiliki nilai kosong pada kolom: "
            + ", ".join(missing_metadata)
            + ". Periksa port_geolocation dan provinces_data."
        )

    coordinate_coverage = float(
        100.0 * df_final[["latitude", "longitude"]].notna().all(axis=1).mean()
    )

    report: dict[str, Any] = {
        "rows_imports": int(len(imports_df)),
        "conversion": {**conversion_diag, **conversion_report},
        "rows_aggregated": int(len(df_aggregated)),
        "province": province_report,
        "geolocation": {
            "rows_geolocation": province_report["rows_geolocation"],
            "ports_with_coordinates": province_report["ports_with_coordinates"],
            "provinces_covered": province_report["provinces_covered"],
            "coordinate_coverage_pct": round(coordinate_coverage, 2),
        },
        "cleaning": clean_report,
        "capping": cap_report,
        "rows_final": int(len(df_final)),
        "columns_final": list(df_final.columns),
    }

    return {
        "df_imports": align_columns(imports_df, IMPORTS_REQUIRED).copy(),
        "mapping": mapping,
        "df_sitc": df_sitc,
        "df_aggregated": df_aggregated,
        "df_finished": df_finished,
        "df_final": df_final,
        "report": report,
    }

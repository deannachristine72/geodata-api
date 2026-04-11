"""
GeoData Indonesia API — FastAPI + DuckDB
=========================================
Menyediakan endpoint geospasial untuk visualisasi deforestasi Indonesia.

Arsitektur koneksi:
  - DuckDB read-only via read_parquet() — tidak perlu database server
  - threading.local() — satu koneksi DuckDB per thread (thread-safe)
  - asyncio.to_thread() — query DuckDB dijalankan di thread pool agar tidak
    memblok event loop FastAPI

Data files (relatif terhadap DATA_DIR):
  - indonesia_groundsource.parquet  : 370k polygon deforestasi Indonesia
  - kota_boundaries.parquet         : 394 batas kab/kota (GADM)
  - heatmap_kota_all_years.json     : agregasi all-time per kota (pre-computed)
  - heatmap_kota_per_year.json      : agregasi per kota per tahun (pre-computed)
"""

import gzip
import json
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ─── Konfigurasi Path ─────────────────────────────────────────────────────────
# DATA_DIR: default ke ./data (relatif terhadap main.py), bisa di-override via env var
# Contoh lokal  : DATA_DIR=C:\Users\DChristine uvicorn main:app ...
# Contoh Railway: DATA_DIR otomatis ./data (folder di-commit ke repo)
_default_data = Path(__file__).parent / "data"
DATA_DIR = Path(os.environ.get("DATA_DIR", _default_data))

PARQUET_DEFOREST    = DATA_DIR / "indonesia_groundsource.parquet"
PARQUET_KOTA        = DATA_DIR / "kota_boundaries.parquet"
HEATMAP_ALL_YEARS   = DATA_DIR / "heatmap_kota_all_years.json"
HEATMAP_PER_YEAR    = DATA_DIR / "heatmap_kota_per_year.json"

# Batas maksimum polygon per request — lindungi frontend dari overload
DEFAULT_POLYGON_LIMIT = 5_000
MAX_POLYGON_LIMIT     = 10_000

# ─── Thread-local DuckDB Connection ───────────────────────────────────────────
_thread_local = threading.local()

def get_con() -> duckdb.DuckDBPyConnection:
    """
    Kembalikan koneksi DuckDB untuk thread saat ini.
    Buat baru jika belum ada — DuckDB tidak thread-safe,
    sehingga tiap thread butuh koneksinya sendiri.
    """
    if not hasattr(_thread_local, "con"):
        con = duckdb.connect(database=":memory:", read_only=False)
        con.execute("INSTALL spatial; LOAD spatial;")
        _thread_local.con = con
    return _thread_local.con

# ─── State Aplikasi (dimuat saat startup) ─────────────────────────────────────
class AppState:
    heatmap_all_years: list[dict] = []
    heatmap_per_year: dict[str, list[dict]] = {}
    kota_geometries: dict[str, dict] = {}   # hasc_code → geometry dict (sudah di-parse)
    kota_meta: dict[str, dict] = {}          # hasc_code → {id_kota, provinsi, kota_name, kota_type}
    total_deforest_rows: int = 0
    available_years: list[int] = []
    ready: bool = False  # True setelah semua data selesai dimuat
    # Cache heatmap sebagai gzip-compressed bytes, siap dikirim langsung ke client
    # Key: cache_key ("all" atau "2024"), Value: gzip bytes
    _heatmap_cache: dict[str, bytes] = {}
    _heatmap_cache_plain: dict[str, bytes] = {}  # untuk client tanpa Accept-Encoding: gzip

app_state = AppState()


def _build_heatmap_json_bytes(year: Optional[int]) -> bytes:
    """
    Bangun GeoJSON FeatureCollection heatmap dan serialisasi ke bytes.
    Dipanggil sekali saat startup untuk tiap tahun dan semua tahun.
    Geometry sudah pre-parsed (dict), jadi tidak ada json.loads() di sini.
    """
    if year is not None:
        yr_str = str(year)
        if yr_str not in app_state.heatmap_per_year:
            return b'{"type":"FeatureCollection","features":[],"meta":{"year":null,"kota_count":0,"max_records":0}}'
        kota_stats_list = app_state.heatmap_per_year[yr_str]
        kota_stats = {k["hasc_code"]: k for k in kota_stats_list if k.get("hasc_code")}
    else:
        kota_stats = {
            k["hasc_code"]: {
                "kota_name":      k["kota_name"],
                "provinsi":       k["provinsi"],
                "record_count":   k["total_records"],
                "total_area_km2": k["total_area_km2"],
                "hasc_code":      k["hasc_code"],
            }
            for k in app_state.heatmap_all_years
            if k.get("hasc_code")
        }

    if not kota_stats:
        return b'{"type":"FeatureCollection","features":[]}'

    max_count = max(v["record_count"] for v in kota_stats.values()) or 1
    log_max   = math.log1p(max_count)

    features = []
    for hasc_code, stats in kota_stats.items():
        geom_dict = app_state.kota_geometries.get(hasc_code)
        if not geom_dict:
            continue
        record_count = stats["record_count"]
        intensity    = math.log1p(record_count) / log_max if log_max > 0 else 0.0
        features.append({
            "type": "Feature",
            "geometry": geom_dict,
            "properties": {
                "hasc_code":      hasc_code,
                "kota_name":      stats.get("kota_name", ""),
                "provinsi":       stats.get("provinsi", ""),
                "record_count":   record_count,
                "total_area_km2": round(stats.get("total_area_km2", 0), 2),
                "intensity":      round(intensity, 4),
            },
        })

    features.sort(key=lambda f: f["properties"]["intensity"], reverse=True)

    response_dict = {
        "type": "FeatureCollection",
        "features": features,
        "meta": {"year": year, "kota_count": len(features), "max_records": max_count},
    }
    return json.dumps(response_dict, ensure_ascii=False).encode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Muat semua data statis ke memory saat startup."""
    print("[startup] Memuat data statis ke memory...")

    # Muat heatmap pre-computed
    with open(HEATMAP_ALL_YEARS, encoding="utf-8") as f:
        app_state.heatmap_all_years = json.load(f)

    with open(HEATMAP_PER_YEAR, encoding="utf-8") as f:
        app_state.heatmap_per_year = json.load(f)

    # Muat geometri kota boundary ke dict — hasc_code -> geometry dict (pre-parsed)
    # Gunakan DuckDB langsung (tanpa PyArrow) — konsisten dengan stack yang digunakan
    con = get_con()
    kota_rows = con.sql(f"""
        SELECT hasc_code, geometry_geojson, id_kota, provinsi, kota_name, kota_type
        FROM read_parquet('{PARQUET_KOTA.as_posix()}')
        WHERE hasc_code IS NOT NULL
    """).fetchall()
    _hasc_seen: dict[str, str] = {}  # hasc -> kota_name pertama (untuk deteksi duplikat)
    _null_count = 0
    for hasc, geom_json, id_kota, provinsi, kota_name, kota_type in kota_rows:
        if not hasc:
            _null_count += 1
            continue
        if hasc in _hasc_seen:
            print(f"[startup] WARN duplikat hasc={hasc}: '{_hasc_seen[hasc]}' vs '{kota_name} ({provinsi})' — entry ke-2 diabaikan")
            continue
        _hasc_seen[hasc] = f"{kota_name} ({provinsi})"
        app_state.kota_geometries[hasc] = json.loads(geom_json)
        app_state.kota_meta[hasc] = {
            "id_kota":   id_kota,
            "provinsi":  provinsi,
            "kota_name": kota_name,
            "kota_type": kota_type,
        }
    if _null_count:
        print(f"[startup] INFO  {_null_count} kota tanpa hasc_code (n.a.) di-skip")

    # Baca jumlah total baris deforestasi
    con = get_con()
    app_state.total_deforest_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{PARQUET_DEFOREST.as_posix()}')"
    ).fetchone()[0]

    # Baca daftar tahun yang tersedia
    app_state.available_years = sorted(app_state.heatmap_per_year.keys(), key=int)

    print(f"[startup] Deforestasi rows : {app_state.total_deforest_rows:,}")
    print(f"[startup] Kota boundaries  : {len(app_state.kota_geometries)} unik hasc (dari 394 total GADM)")
    print(f"[startup] Tahun tersedia   : {app_state.available_years[0]}-{app_state.available_years[-1]}")

    # Pre-serialize semua heatmap response ke JSON bytes — eliminasi serialisasi per-request
    print("[startup] Pre-serializing + pre-compressing heatmap responses...")
    for cache_key, yr in [("all", None)] + [(str(y), int(y)) for y in app_state.available_years]:
        plain = _build_heatmap_json_bytes(year=yr)
        app_state._heatmap_cache_plain[cache_key] = plain
        app_state._heatmap_cache[cache_key] = gzip.compress(plain, compresslevel=6)
    total = len(app_state._heatmap_cache)
    compressed_kb = sum(len(v) for v in app_state._heatmap_cache.values()) / 1024
    print(f"[startup] Heatmap cache    : {total} entries, {compressed_kb:.0f} KB total compressed")
    app_state.ready = True
    print("[startup] Siap melayani request.")

    yield  # aplikasi berjalan

    print("[shutdown] Menutup koneksi DuckDB...")
    if hasattr(_thread_local, "con"):
        _thread_local.con.close()


# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoData Indonesia API",
    description="API deforestasi Indonesia — polygon rendering dan heatmap kab/kota",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — default ke domain production, override via env var ALLOWED_ORIGINS
_default_origins = ["https://geodata-frontend.vercel.app"]
_origins_env = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _origins_env.split(",") if o.strip()]
    if _origins_env
    else _default_origins
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ─── Helper: Toleransi Simplifikasi Berdasarkan Lebar Bbox ────────────────────
def _simplify_tolerance(min_lon: float, max_lon: float,
                        min_lat: float, max_lat: float) -> float:
    """
    Hitung toleransi ST_Simplify berdasarkan lebar viewport.
    Makin lebar bbox (zoom out) → toleransi makin besar → polygon lebih sederhana.
    Ini mengurangi ukuran payload dan meningkatkan performa rendering frontend.

    Mapping kasar ke zoom level:
      bbox ~50° (zoom 5, seluruh Indonesia)  → tolerance 0.05
      bbox ~10° (zoom 8, satu pulau)         → tolerance 0.01
      bbox  ~1° (zoom 12, satu kota)         → tolerance 0.001
      bbox ~0.1° (zoom 15, kecamatan)        → tolerance 0 (tidak disederhanakan)
    """
    bbox_diagonal = math.sqrt(
        (max_lon - min_lon) ** 2 + (max_lat - min_lat) ** 2
    )
    if bbox_diagonal > 40:
        return 0.05
    if bbox_diagonal > 10:
        return 0.01
    if bbox_diagonal > 2:
        return 0.001
    return 0.0  # tidak disederhanakan untuk zoom tinggi


# ─── Helper: Query Polygon di Thread (blocking, dijalankan via to_thread) ─────
def _query_polygons(
    min_lon: float, max_lon: float,
    min_lat: float, max_lat: float,
    year: Optional[int],
    limit: int,
    tolerance: float,
) -> list[tuple]:
    """
    Jalankan query DuckDB untuk polygon deforestasi dalam bbox.
    Fungsi ini blocking — harus dipanggil via asyncio.to_thread().
    """
    con = get_con()

    year_clause = f"AND year = {year}" if year is not None else ""

    if tolerance > 0:
        geom_expr = f"ST_AsGeoJSON(ST_Simplify(ST_GeomFromWKB(geometry), {tolerance}))"
    else:
        geom_expr = "ST_AsGeoJSON(ST_GeomFromWKB(geometry))"

    sql = f"""
    SELECT
        uuid,
        ROUND(area_km2, 4)  AS area_km2,
        year,
        start_date,
        end_date,
        {geom_expr}         AS geojson
    FROM read_parquet('{PARQUET_DEFOREST.as_posix()}')
    WHERE centroid_lon BETWEEN {min_lon} AND {max_lon}
      AND centroid_lat BETWEEN {min_lat} AND {max_lat}
      {year_clause}
    ORDER BY area_km2 DESC
    LIMIT {limit}
    """
    return con.sql(sql).fetchall()


# ─── Endpoint: Health Check ────────────────────────────────────────────────────
@app.get("/health", summary="Status server dan ringkasan data")
async def health_check():
    if not app_state.ready:
        # Startup masih berjalan — kembalikan 503 agar Railway terus retry
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "Data masih dimuat..."}
        )
    return {
        "status": "ok",
        "total_deforest_rows": app_state.total_deforest_rows,
        "kota_count": len(app_state.kota_geometries),
        "available_years": [int(y) for y in app_state.available_years],
    }


# ─── Endpoint: Daftar Tahun ────────────────────────────────────────────────────
@app.get("/api/years", summary="Daftar tahun yang tersedia beserta jumlah record")
async def get_years():
    """
    Kembalikan daftar tahun yang memiliki data deforestasi di Indonesia,
    beserta jumlah record per tahun (dari heatmap cache).
    """
    year_counts: dict[int, int] = {}
    for yr_str, kota_list in app_state.heatmap_per_year.items():
        year_counts[int(yr_str)] = sum(k["record_count"] for k in kota_list)

    result = [
        {"year": yr, "count": cnt}
        for yr, cnt in sorted(year_counts.items())
    ]
    payload = json.dumps({"years": result}, ensure_ascii=False).encode("utf-8")
    return Response(
        content=gzip.compress(payload, compresslevel=6),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


# ─── Endpoint: Polygon Deforestasi dalam Viewport ─────────────────────────────
@app.get("/api/polygons", summary="Polygon deforestasi dalam bounding box viewport")
async def get_polygons(
    min_lon: float = Query(..., ge=-180, le=180,  description="Longitude minimum (barat)"),
    min_lat: float = Query(..., ge=-90,  le=90,   description="Latitude minimum (selatan)"),
    max_lon: float = Query(..., ge=-180, le=180,  description="Longitude maksimum (timur)"),
    max_lat: float = Query(..., ge=-90,  le=90,   description="Latitude maksimum (utara)"),
    year:    Optional[int] = Query(None, ge=2000, le=2030, description="Filter tahun (opsional)"),
    limit:   int = Query(DEFAULT_POLYGON_LIMIT, ge=1, le=MAX_POLYGON_LIMIT,
                         description=f"Maks polygon dikembalikan (default {DEFAULT_POLYGON_LIMIT})"),
):
    """
    Kembalikan polygon deforestasi Indonesia dalam GeoJSON FeatureCollection.

    - Filter berdasarkan **centroid** polygon yang masuk dalam bbox viewport.
    - Polygon di-simplify otomatis berdasarkan lebar viewport (zoom out = lebih sederhana).
    - Di-urutkan berdasarkan `area_km2` descending (polygon terbesar duluan).
    - Gunakan `limit` untuk mengontrol jumlah hasil (max 10.000).
    """
    if min_lon >= max_lon:
        raise HTTPException(400, "min_lon harus lebih kecil dari max_lon")
    if min_lat >= max_lat:
        raise HTTPException(400, "min_lat harus lebih kecil dari max_lat")

    tolerance = _simplify_tolerance(min_lon, max_lon, min_lat, max_lat)

    # Jalankan query DuckDB di thread pool (tidak memblok event loop)
    import asyncio
    rows = await asyncio.to_thread(
        _query_polygons,
        min_lon, max_lon, min_lat, max_lat,
        year, limit, tolerance,
    )

    # Bangun GeoJSON FeatureCollection
    features = []
    for uuid, area_km2, yr, start_date, end_date, geojson_str in rows:
        if geojson_str is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": json.loads(geojson_str),
            "properties": {
                "uuid":       uuid,
                "area_km2":   area_km2,
                "year":       yr,
                "start_date": start_date,
                "end_date":   end_date,
            },
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "count":     len(features),
            "year":      year,
            "bbox":      [min_lon, min_lat, max_lon, max_lat],
            "tolerance": tolerance,
            "limit":     limit,
        },
    }
    payload = json.dumps(fc, ensure_ascii=False).encode("utf-8")
    return Response(
        content=gzip.compress(payload, compresslevel=6),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


# ─── Endpoint: Heatmap Deforestasi per Kab/Kota ───────────────────────────────
@app.get("/api/heatmap/kota", summary="Choropleth deforestasi per kabupaten/kota")
async def get_heatmap_kota(
    year: Optional[int] = Query(
        None, ge=2000, le=2030,
        description="Filter tahun. Jika tidak diisi, gunakan data semua tahun."
    ),
    accept_encoding: Optional[str] = Query(None, include_in_schema=False),
):
    """
    Kembalikan GeoJSON FeatureCollection dengan polygon tiap kabupaten/kota
    yang diwarnai berdasarkan intensitas deforestasi (choropleth).

    Properties tiap feature:
    - `record_count`   : jumlah polygon deforestasi dalam kota ini
    - `total_area_km2` : total luas deforestasi (km²)
    - `intensity`      : nilai 0.0-1.0 (ternormalisasi log-scale untuk warna)
    - `provinsi`, `kota_name`, `kota_type`, `hasc_code`

    Data dari cache pre-compressed saat startup — tidak ada DuckDB query atau
    serialisasi saat runtime. Response time < 10ms.
    """
    cache_key = str(year) if year is not None else "all"

    if cache_key not in app_state._heatmap_cache:
        raise HTTPException(404, f"Tidak ada data untuk tahun {year}")

    # Kembalikan gzip-compressed bytes langsung — bypass GZipMiddleware
    return Response(
        content=app_state._heatmap_cache[cache_key],
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


# ─── Helper: Query Centroid Points ───────────────────────────────────────────
def _query_centroids(
    min_lon: float, max_lon: float,
    min_lat: float, max_lat: float,
    year: Optional[int],
    limit: int,
) -> list[tuple]:
    """Query centroid points (tanpa geometry) — jauh lebih ringan dari polygon."""
    con = get_con()
    year_clause = f"AND year = {year}" if year is not None else ""
    sql = f"""
    SELECT
        centroid_lon,
        centroid_lat,
        ROUND(area_km2, 4) AS area_km2,
        year,
        uuid
    FROM read_parquet('{PARQUET_DEFOREST.as_posix()}')
    WHERE centroid_lon BETWEEN {min_lon} AND {max_lon}
      AND centroid_lat BETWEEN {min_lat} AND {max_lat}
      {year_clause}
    ORDER BY area_km2 DESC
    LIMIT {limit}
    """
    return con.sql(sql).fetchall()


# ─── Endpoint: Centroid Points dalam Viewport ────────────────────────────────
@app.get("/api/centroids", summary="Titik centroid deforestasi dalam bounding box viewport")
async def get_centroids(
    min_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    year: Optional[int] = Query(None, ge=2000, le=2030),
    limit: int = Query(8000, ge=1, le=15000),
):
    """
    Kembalikan centroid points (lon, lat, area, year, uuid) dalam viewport.
    Jauh lebih ringan dari /api/polygons — tanpa geometry processing.
    """
    if min_lon >= max_lon:
        raise HTTPException(400, "min_lon harus lebih kecil dari max_lon")
    if min_lat >= max_lat:
        raise HTTPException(400, "min_lat harus lebih kecil dari max_lat")

    import asyncio
    rows = await asyncio.to_thread(
        _query_centroids,
        min_lon, max_lon, min_lat, max_lat,
        year, limit,
    )

    # Format array-of-arrays untuk minimisasi payload
    points = [[lon, lat, area, yr, uid] for lon, lat, area, yr, uid in rows]

    result = {
        "points": points,
        "meta": {
            "count": len(points),
            "year": year,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "limit": limit,
        },
    }
    payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
    return Response(
        content=gzip.compress(payload, compresslevel=6),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


# ─── Endpoint: Daftar Kota untuk Search Autocomplete ─────────────────────────
@app.get("/api/search/kota", summary="Daftar kota/kabupaten untuk search autocomplete")
async def search_kota():
    """
    Kembalikan seluruh 386 kota beserta centroid dan bbox untuk frontend search.
    Data sudah di-memory — response time < 5ms.
    """
    kota_list = []
    for hasc_code, meta in app_state.kota_meta.items():
        geom = app_state.kota_geometries.get(hasc_code)
        if not geom:
            continue

        # Hitung bbox dan centroid dari geometry coordinates
        all_lons = []
        all_lats = []
        coords = geom.get("coordinates", [])
        geom_type = geom.get("type", "")

        if geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    for pt in ring:
                        all_lons.append(pt[0])
                        all_lats.append(pt[1])
        elif geom_type == "Polygon":
            for ring in coords:
                for pt in ring:
                    all_lons.append(pt[0])
                    all_lats.append(pt[1])

        if not all_lons:
            continue

        min_lon = min(all_lons)
        max_lon = max(all_lons)
        min_lat = min(all_lats)
        max_lat = max(all_lats)

        kota_list.append({
            "hasc_code": hasc_code,
            "kota_name": meta.get("kota_name", ""),
            "kota_type": meta.get("kota_type", ""),
            "provinsi": meta.get("provinsi", ""),
            "centroid": [round((min_lon + max_lon) / 2, 4), round((min_lat + max_lat) / 2, 4)],
            "bbox": [round(min_lon, 4), round(min_lat, 4), round(max_lon, 4), round(max_lat, 4)],
        })

    kota_list.sort(key=lambda k: (k["provinsi"], k["kota_name"]))

    payload = json.dumps({"kota": kota_list}, ensure_ascii=False).encode("utf-8")
    return Response(
        content=gzip.compress(payload, compresslevel=6),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


# ─── Helper: Query Statistik Agregasi ────────────────────────────────────────
def _query_stats(year: Optional[int]) -> dict:
    """Query statistik agregasi dari parquet."""
    con = get_con()
    year_clause = f"WHERE year = {year}" if year is not None else ""
    sql = f"""
    SELECT
        COUNT(*)                     AS total_events,
        MIN(start_date)              AS date_min,
        MAX(end_date)                AS date_max,
        ROUND(AVG(area_km2), 1)     AS avg_area_km2,
        ROUND(MAX(area_km2), 1)     AS max_area_km2,
        ROUND(SUM(area_km2), 1)     AS total_area_km2
    FROM read_parquet('{PARQUET_DEFOREST.as_posix()}')
    {year_clause}
    """
    row = con.sql(sql).fetchone()
    return {
        "total_events": row[0],
        "date_range": [str(row[1]) if row[1] else None, str(row[2]) if row[2] else None],
        "avg_area_km2": row[3],
        "max_area_km2": row[4],
        "total_area_km2": row[5],
    }


# ─── Endpoint: Statistik Ringkasan Deforestasi ───────────────────────────────
@app.get("/api/stats", summary="Statistik ringkasan deforestasi")
async def get_stats(
    year: Optional[int] = Query(None, ge=2000, le=2030),
):
    """Kembalikan statistik agregasi: total events, date range, avg/max area."""
    import asyncio
    stats = await asyncio.to_thread(_query_stats, year)

    payload = json.dumps(stats, ensure_ascii=False).encode("utf-8")
    return Response(
        content=gzip.compress(payload, compresslevel=6),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


# ─── Endpoint: GeoJSON Boundary Outline Satu Kota ────────────────────────────
@app.get("/api/boundary/kota/{hasc_code}", summary="GeoJSON boundary outline satu kota/kabupaten")
async def get_boundary_kota(hasc_code: str):
    """
    Kembalikan GeoJSON Feature untuk satu kota berdasarkan hasc_code.
    Data dari memory — response time < 5ms.
    """
    geom = app_state.kota_geometries.get(hasc_code)
    meta = app_state.kota_meta.get(hasc_code)

    if not geom or not meta:
        raise HTTPException(404, f"Kota dengan hasc_code '{hasc_code}' tidak ditemukan")

    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "hasc_code": hasc_code,
            "kota_name": meta.get("kota_name", ""),
            "provinsi": meta.get("provinsi", ""),
            "kota_type": meta.get("kota_type", ""),
        },
    }

    payload = json.dumps(feature, ensure_ascii=False).encode("utf-8")
    return Response(
        content=gzip.compress(payload, compresslevel=6),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )
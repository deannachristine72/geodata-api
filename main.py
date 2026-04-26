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

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from lifespan import lifespan
from routes import (
    get_boundary_kota,
    get_centroids,
    get_heatmap_kota,
    get_polygons,
    get_stats,
    get_years,
    health_check,
    search_kota,
)

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoData Indonesia API",
    description="API deforestasi Indonesia — polygon rendering dan heatmap kab/kota",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — default ke domain production, override via env var ALLOWED_ORIGINS
_default_origins = ["https://floodmapindonesia.vercel.app"]
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

# Rate limiting — 60 requests/menit per IP untuk endpoint berat
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── Route Registration ────────────────────────────────────────────────────────
app.get("/health",    summary="Status server dan ringkasan data")(health_check)
app.get("/api/years", summary="Daftar tahun yang tersedia beserta jumlah record")(get_years)

app.get("/api/polygons",  summary="Polygon deforestasi dalam bounding box viewport")(
    limiter.limit("60/minute")(get_polygons)
)
app.get("/api/centroids", summary="Titik centroid deforestasi dalam bounding box viewport")(
    limiter.limit("60/minute")(get_centroids)
)
app.get("/api/stats",     summary="Statistik ringkasan deforestasi")(
    limiter.limit("60/minute")(get_stats)
)

app.get("/api/heatmap/kota",           summary="Choropleth deforestasi per kabupaten/kota")(get_heatmap_kota)
app.get("/api/search/kota",            summary="Daftar kota/kabupaten untuk search autocomplete")(search_kota)
app.get("/api/boundary/kota/{hasc_code}", summary="GeoJSON boundary outline satu kota/kabupaten")(get_boundary_kota)

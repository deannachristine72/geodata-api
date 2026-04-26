import gzip
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import (
    HEATMAP_ALL_YEARS,
    HEATMAP_PER_YEAR,
    PARQUET_DEFOREST,
    PARQUET_KOTA,
    _build_heatmap_json_bytes,
    _thread_local,
    app_state,
    get_con,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Memuat data statis ke memory...")

    with open(HEATMAP_ALL_YEARS, encoding="utf-8") as f:
        app_state.heatmap_all_years = json.load(f)

    with open(HEATMAP_PER_YEAR, encoding="utf-8") as f:
        app_state.heatmap_per_year = json.load(f)

    con = get_con()
    kota_rows = con.sql(f"""
        SELECT hasc_code, geometry_geojson, id_kota, provinsi, kota_name, kota_type
        FROM read_parquet('{PARQUET_KOTA.as_posix()}')
        WHERE hasc_code IS NOT NULL
    """).fetchall()
    _hasc_seen: dict[str, str] = {}
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

    con = get_con()
    app_state.total_deforest_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{PARQUET_DEFOREST.as_posix()}')"
    ).fetchone()[0]

    app_state.available_years = sorted(app_state.heatmap_per_year.keys(), key=int)

    print(f"[startup] Deforestasi rows : {app_state.total_deforest_rows:,}")
    print(f"[startup] Kota boundaries  : {len(app_state.kota_geometries)} unik hasc (dari 394 total GADM)")
    print(f"[startup] Tahun tersedia   : {app_state.available_years[0]}-{app_state.available_years[-1]}")

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

    yield

    print("[shutdown] Menutup koneksi DuckDB...")
    if hasattr(_thread_local, "con"):
        _thread_local.con.close()

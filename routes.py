"""
Semua endpoint FastAPI — health check, polygons, centroids, heatmap, boundary, search, stats.
"""

import asyncio
import gzip
import json
from typing import Optional

from fastapi import HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from config import DEFAULT_POLYGON_LIMIT, MAX_POLYGON_LIMIT, app_state
from queries import (
    _query_centroids,
    _query_polygons,
    _query_stats,
    _simplify_tolerance,
)


async def health_check():
    if not app_state.ready:
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


async def get_years():

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


async def get_polygons(
    request: Request,
    min_lon: float = Query(..., ge=-180, le=180,  description="Longitude minimum (barat)"),
    min_lat: float = Query(..., ge=-90,  le=90,   description="Latitude minimum (selatan)"),
    max_lon: float = Query(..., ge=-180, le=180,  description="Longitude maksimum (timur)"),
    max_lat: float = Query(..., ge=-90,  le=90,   description="Latitude maksimum (utara)"),
    year:    Optional[int] = Query(None, ge=2000, le=2030, description="Filter tahun (opsional)"),
    limit:   int = Query(DEFAULT_POLYGON_LIMIT, ge=1, le=MAX_POLYGON_LIMIT,
                         description=f"Maks polygon dikembalikan (default {DEFAULT_POLYGON_LIMIT})"),
):
    if min_lon >= max_lon:
        raise HTTPException(400, "min_lon harus lebih kecil dari max_lon")
    if min_lat >= max_lat:
        raise HTTPException(400, "min_lat harus lebih kecil dari max_lat")

    tolerance = _simplify_tolerance(min_lon, max_lon, min_lat, max_lat)

    rows = await asyncio.to_thread(
        _query_polygons,
        min_lon, max_lon, min_lat, max_lat,
        year, limit, tolerance,
    )

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


async def get_centroids(
    request: Request,
    min_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    year: Optional[int] = Query(None, ge=2000, le=2030),
    limit: int = Query(8000, ge=1, le=15000),
):

    if min_lon >= max_lon:
        raise HTTPException(400, "min_lon harus lebih kecil dari max_lon")
    if min_lat >= max_lat:
        raise HTTPException(400, "min_lat harus lebih kecil dari max_lat")

    rows = await asyncio.to_thread(
        _query_centroids,
        min_lon, max_lon, min_lat, max_lat,
        year, limit,
    )

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


async def search_kota():

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


async def get_stats(
    request: Request,
    year: Optional[int] = Query(None, ge=2000, le=2030),
):
    """Kembalikan statistik agregasi: total events, date range, avg/max area."""
    stats = await asyncio.to_thread(_query_stats, year)

    payload = json.dumps(stats, ensure_ascii=False).encode("utf-8")
    return Response(
        content=gzip.compress(payload, compresslevel=6),
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


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

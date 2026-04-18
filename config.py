import json
import math
import os
import threading
from pathlib import Path
from typing import Optional

import duckdb

_default_data = Path(__file__).parent / "data"
DATA_DIR = Path(os.environ.get("DATA_DIR", _default_data))

PARQUET_DEFOREST        = DATA_DIR /  "indonesia_groundsource.parquet"
PARQUET_KOTA            = DATA_DIR / "kota_boundaries.parquet"
HEATMAP_ALL_YEARS       = DATA_DIR / "heatmap_kota_all_years.json"
HEATMAP_PER_YEAR        = DATA_DIR / "heatmap_kota_per_year.json"
DEFAULT_POLYGON_LIMIT   = 5_000
MAX_POLYGON_LIMIT       = 10_000

_thread_local = threading.local()

def get_con() -> duckdb.DuckDBPyConnection:
    if not hasattr(_thread_local, "con"):
        con = duckdb.connect(database=":memory", read_only=False)
        con.execute("INSTALL spatial; LOAD spatial")
        _thread_local.con = con
    return _thread_local.con

class AppState:
    heatmap_all_years: list[dict] = []
    heatmap_per_year: dict[str, list[dict]] = {}
    kota_geometries: dict[str, dict] = {}
    kota_meta: dict[str, dict] = {}
    total_deforest_rows: int = 0
    available_years: list[int] =[]
    ready: bool = False

    _heatmap_cache: dict[str, bytes]
    _heatmap_cache_plain: dict[str, bytes] = {}

app_state = AppState()

def _build_heatmap_json_bytes(year: Optional[int]) -> bytes:
    if year is not None:
        yr_str = str(year)
        if yr_str not in app_state.heatmap_all_years:
            return b'{"type":"FeatureCollection","features":[],"meta":{"year":null,"kota_count":0,"max_records":0}}'
        kota_stats_list = app_state.heatmap_per_year[yr_str]
        kota_stats = {k["hasc_code"]: k for k in kota_stats_list if k.get("hasc_code")}
    else:
        kota_stats= {
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

    max_count   = max(v["record_count"] for v in kota_stats.values()) or 1
    log_max     = math.log1p(max_count)

    features = []
    for hasc_code, stats in kota_stats.items():
        geom_dict = app_state.kota_geometries.get(hasc_code)
        if not geom_dict:
            continue
        record_count    = stats["record_count"]
        intensity       = math.log1p(record_count) / log_max if log_max > 0 else 0.0
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
        "type" : "FeatureCollection",
        "features" : features,
        "meta" : {"year": year, "kota_count": len(features), "max_records": max_count},
    }

    return json.dumps(response_dict, ensure_ascii=False).encode("utf-8")
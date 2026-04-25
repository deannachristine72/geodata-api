"""
Helper functions untuk query DuckDB — semua blocking, dipanggil via asyncio.to_thread().
"""

import math
from typing import Optional

from config import PARQUET_DEFOREST, get_con


def _simplify_tolerance(min_lon: float, max_lon: float,
                        min_lat: float, max_lat: float) -> float:

    bbox_diagonal = math.sqrt(
        (max_lon - min_lon) ** 2 + (max_lat - min_lat) ** 2
    )
    if bbox_diagonal > 40:
        return 0.05
    if bbox_diagonal > 10:
        return 0.01
    if bbox_diagonal > 2:
        return 0.001
    return 0.0


def _query_polygons(
    min_lon: float, max_lon: float,
    min_lat: float, max_lat: float,
    year: Optional[int],
    limit: int,
    tolerance: float,
) -> list[tuple]:

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

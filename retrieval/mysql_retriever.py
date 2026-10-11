"""
OPTIONAL MySQL-backed retrieval for the Mumbai engine.

Only used when MUMBAI_DATA_SOURCE=mysql (default is "json" — see config.py
and data/mumbai_places.json). This module exists so that, when the team is
ready, the Mumbai provider dataset can live in the SAME MySQL database your
teammates already run, instead of a JSON file — see sql/schema.sql for the
`mumbai_places` table DDL that matches the columns this module expects.

Design decision (see DESIGN.md Section 14): MySQL holds structured place
metadata; it does NOT need to do vector/semantic search. This module's job
ends at "give me rows that look plausibly relevant" — semantic ranking still
happens in Python over the NormalizedPlace objects built from these rows.

Requires PyMySQL (`pip install pymysql`) — imported lazily so the rest of
the service works with zero MySQL dependency when MUMBAI_DATA_SOURCE=json.
"""
from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger("travelsathi.retrieval.mysql")

# Columns expected in `mumbai_places` (see sql/schema.sql). Kept as an
# explicit list (rather than SELECT *) so a schema change is a one-line diff
# here instead of a silent behaviour change.
MUMBAI_PLACES_COLUMNS = [
    "provider_id", "provider_name", "address", "latitude", "longitude",
    "category", "sub_category", "description", "experience_tags",
    "traveller_tags", "cost_basis", "est_cost_per_person_inr",
    "people_per_experience", "recommended_duration", "opening_hours",
    "weekly_closure", "rating", "review_count", "indoor_outdoor",
    "signature_items", "local_score", "local_confidence",
    "local_recommendation_count", "local_tips", "last_verified", "source_url",
]


def _get_connection():
    import pymysql  # local import: optional dependency

    return pymysql.connect(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def fetch_mumbai_places_raw() -> list[dict]:
    """
    Returns raw rows shaped exactly like data/mumbai_places.json's "places"
    entries, so normalization.normalized_place.from_mumbai_record works
    unchanged regardless of which data source is active.

    JSON-typed columns (category, experience_tags, traveller_tags,
    signature_items, local_tips) are expected to be stored as either a MySQL
    JSON column or a comma-separated VARCHAR/TEXT — from_mumbai_record()
    already handles both a list and a comma-separated string.
    """
    query = f"SELECT {', '.join(MUMBAI_PLACES_COLUMNS)} FROM mumbai_places"
    try:
        conn = _get_connection()
    except ImportError:
        logger.error("PyMySQL not installed. Run `pip install pymysql`, or set MUMBAI_DATA_SOURCE=json.")
        raise
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        logger.info("Fetched %d rows from MySQL mumbai_places table.", len(rows))
        return rows
    finally:
        conn.close()

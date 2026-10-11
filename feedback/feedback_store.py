"""
Feedback storage (Section 22). Two backends selected by config.FEEDBACK_STORE:

  "json"  (default) — append-only JSON-Lines file. Zero setup, human
          readable, good enough for a hackathon's worth of events, and
          trivially loadable into pandas later for the future
          learning-to-rank training set (Section 16).

  "mysql" — appends into the `recommendation_feedback` table (see
          sql/schema.sql), so the event lives in the same database your
          teammates already query, joinable against their `users` table.

Either way, every feedback event captures (query_id, place_id, event, ...)
so a future model can reconstruct: "for this query, with this intent, this
place was clicked/saved/booked/rejected" — the training signal described in
Section 16.
"""
from __future__ import annotations

import json
import logging
import os

from config import settings
from models.schemas import FeedbackEvent

logger = logging.getLogger("travelsathi.feedback")


def _store_json(event: FeedbackEvent) -> None:
    os.makedirs(os.path.dirname(settings.FEEDBACK_LOG_PATH), exist_ok=True)
    with open(settings.FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event.model_dump(mode="json")) + "\n")


def _store_mysql(event: FeedbackEvent) -> None:
    import pymysql  # local import: optional dependency

    conn = pymysql.connect(
        host=settings.MYSQL_HOST, port=settings.MYSQL_PORT, user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD, database=settings.MYSQL_DATABASE, autocommit=True,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO recommendation_feedback
                    (user_id, query_id, place_id, event, value, dwell_time_seconds, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, FROM_UNIXTIME(%s))
                """,
                (
                    event.user_id, event.query_id, event.place_id, event.event.value,
                    event.value, event.dwell_time_seconds, event.timestamp,
                ),
            )
    finally:
        conn.close()


def record_feedback(event: FeedbackEvent) -> None:
    try:
        if settings.FEEDBACK_STORE == "mysql":
            _store_mysql(event)
        else:
            _store_json(event)
        logger.info("Recorded feedback event=%s place_id=%s query_id=%s", event.event, event.place_id, event.query_id)
    except Exception:
        # Feedback is best-effort telemetry — never fail a request over it.
        logger.exception("Failed to record feedback event; continuing.")

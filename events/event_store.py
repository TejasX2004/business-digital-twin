"""
Event Store for Business Digital Twin.
Provides persistence, idempotency checking, and event auditing in PostgreSQL.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from backend.digital_twin.digital_twin import get_connection
from events.event_types import EventStatus, EventType


CREATE_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS business_events (
    event_id VARCHAR(64) PRIMARY KEY,
    event_type VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(32) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);
"""


class EventStore:
    """
    Manages persistence and retrieval of business events.
    """

    def __init__(self):
        self.ensure_table()

    def ensure_table(self) -> None:
        """Create the business_events audit table if it does not exist."""
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_EVENTS_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    def has_event(self, event_id: str, cursor=None) -> bool:
        """Check if an event with event_id has already been processed or recorded."""
        if cursor is not None:
            cursor.execute("SELECT 1 FROM business_events WHERE event_id = %s LIMIT 1;", (event_id,))
            return cursor.fetchone() is not None

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM business_events WHERE event_id = %s LIMIT 1;", (event_id,))
                return cur.fetchone() is not None
        finally:
            conn.close()

    def record_event(
        self,
        event_id: str,
        event_type: str,
        payload: Dict[str, Any],
        status: str,
        error_message: Optional[str] = None,
        cursor=None,
    ) -> None:
        """Record an event into the business_events table."""
        sql = """
        INSERT INTO business_events (
            event_id,
            event_type,
            payload,
            status,
            error_message,
            processed_at
        )
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (event_id) DO UPDATE SET
            status = EXCLUDED.status,
            error_message = EXCLUDED.error_message,
            processed_at = CURRENT_TIMESTAMP;
        """
        payload_json = json.dumps(payload, default=str)
        params = (event_id, event_type, payload_json, status, error_message)

        if cursor is not None:
            cursor.execute(sql, params)
            return

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the most recent events for UI display and auditing."""
        conn = get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        event_id,
                        event_type,
                        created_at,
                        status,
                        error_message,
                        payload
                    FROM business_events
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (limit,)
                )
                rows = cur.fetchall()
                events = []
                for row in rows:
                    events.append({
                        "event_id": row["event_id"],
                        "event_type": row["event_type"],
                        "timestamp": row["created_at"].strftime("%Y-%m-%d %H:%M:%S") if row["created_at"] else "",
                        "status": row["status"],
                        "error_message": row.get("error_message") or "",
                        "payload": row.get("payload") or {},
                    })
                return events
        finally:
            conn.close()

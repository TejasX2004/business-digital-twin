"""
Event Processor for Business Digital Twin.
Coordinates deterministic event processing, transaction boundaries,
idempotency, and audit logging.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Union

from backend.digital_twin.digital_twin import get_connection
from events.event_store import EventStore
from events.event_types import EventProcessingResult, EventStatus, EventType
from events.handlers import (
    handle_expense_recorded,
    handle_inventory_updated,
    handle_payment_received,
    handle_restrict_credit,
    handle_sale_created,
    handle_supplier_payment_updated,
)


class EventProcessor:
    """
    Validates and processes incoming business events against PostgreSQL within transactional boundaries.
    """

    HANDLERS = {
        EventType.SALE_CREATED.value: handle_sale_created,
        EventType.PAYMENT_RECEIVED.value: handle_payment_received,
        EventType.INVENTORY_UPDATED.value: handle_inventory_updated,
        EventType.EXPENSE_RECORDED.value: handle_expense_recorded,
        EventType.SUPPLIER_PAYMENT_UPDATED.value: handle_supplier_payment_updated,
        EventType.RESTRICT_CREDIT.value: handle_restrict_credit,
        EventType.INITIATE_CREDIT_REVIEW.value: handle_restrict_credit,
    }

    def __init__(self, event_store: Optional[EventStore] = None):
        self.store = event_store if event_store is not None else EventStore()

    def process_event(
        self,
        event_type: Union[str, EventType],
        payload: Dict[str, Any],
        event_id: Optional[str] = None,
    ) -> EventProcessingResult:
        """
        Processes a single business event with transaction rollback on error
        and idempotency protection.
        """
        # Normalize event type
        type_str = event_type.value if isinstance(event_type, EventType) else str(event_type).strip()
        if type_str not in self.HANDLERS:
            return EventProcessingResult(
                success=False,
                event_id=event_id or "UNKNOWN",
                event_type=type_str,
                message=f"Unsupported event type: '{type_str}'",
                error=f"Unsupported event type. Must be one of: {list(self.HANDLERS.keys())}",
            )

        # Generate or format event ID
        evt_id = event_id.strip() if event_id and str(event_id).strip() else f"EVT-{uuid.uuid4().hex[:12].upper()}"

        # Idempotency check: prevent duplicate event processing
        if self.store.has_event(evt_id):
            return EventProcessingResult(
                success=False,
                event_id=evt_id,
                event_type=type_str,
                message=f"Duplicate event: Event ID '{evt_id}' has already been processed.",
                error="Duplicate event processing rejected.",
            )

        conn = get_connection()
        conn.autocommit = False

        try:
            with conn.cursor() as cur:
                # 1. Execute business handler with parameterized SQL
                handler = self.HANDLERS[type_str]
                details = handler(cur, payload)

                # 2. Record event in audit table within the same transaction
                self.store.record_event(
                    event_id=evt_id,
                    event_type=type_str,
                    payload=payload,
                    status=EventStatus.SUCCESS.value,
                    cursor=cur,
                )

            # 3. Commit transaction atomically
            conn.commit()

            return EventProcessingResult(
                success=True,
                event_id=evt_id,
                event_type=type_str,
                message=f"Successfully processed {type_str} event.",
                details=details,
            )

        except Exception as e:
            # 4. Rollback transaction on failure so DB state remains pristine
            conn.rollback()

            # Record failed event log in a separate clean transaction
            try:
                self.store.record_event(
                    event_id=evt_id,
                    event_type=type_str,
                    payload=payload,
                    status=EventStatus.FAILED.value,
                    error_message=str(e),
                )
            except Exception:
                pass

            return EventProcessingResult(
                success=False,
                event_id=evt_id,
                event_type=type_str,
                message=f"Failed to process {type_str} event: {e}",
                error=str(e),
            )

        finally:
            conn.close()

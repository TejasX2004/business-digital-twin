"""
Event Types and Data Definitions for Business Digital Twin Event Layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    SALE_CREATED = "SALE_CREATED"
    PAYMENT_RECEIVED = "PAYMENT_RECEIVED"
    INVENTORY_UPDATED = "INVENTORY_UPDATED"
    EXPENSE_RECORDED = "EXPENSE_RECORDED"
    SUPPLIER_PAYMENT_UPDATED = "SUPPLIER_PAYMENT_UPDATED"
    RESTRICT_CREDIT = "RESTRICT_CREDIT"
    INITIATE_CREDIT_REVIEW = "INITIATE_CREDIT_REVIEW"


class EventStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DUPLICATE = "DUPLICATE"


@dataclass
class BusinessEvent:
    event_id: str
    event_type: EventType
    payload: Dict[str, Any]
    status: EventStatus = EventStatus.SUCCESS
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None


@dataclass
class EventProcessingResult:
    success: bool
    event_id: str
    event_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

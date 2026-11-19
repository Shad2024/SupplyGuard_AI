from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    SUPPLIER = "supplier"
    FACTORY = "factory"
    WAREHOUSE = "warehouse"


class EventType(str, Enum):
    ORDER = "order"
    SHIPMENT_UPDATE = "shipment_update"
    DISRUPTION = "disruption"
    SECURITY_EVENT = "security_event"


@dataclass
class ComponentSpec:
    item_code: str
    name: str
    criticality: float
    base_lead_time_ticks: int
    unit_cost: float
    max_throughput: float
    reliability_score: float = 0.95


@dataclass
class Node:
    node_id: str
    name: str
    node_type: NodeType
    tier: int
    capacity_per_tick: float
    trust_score: float = 0.95
    is_compromised: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    edge_id: str
    source_id: str
    target_id: str
    item_code: str
    base_lead_time_ticks: int
    unit_cost: float
    max_throughput: float
    reliability_score: float = 0.95
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InventorySnapshot:
    tick: int
    node_id: str
    item_code: str
    on_hand: float
    incoming_qty: float
    reorder_point: float
    demand_rate: float
    stockout_qty: float = 0.0


@dataclass
class SupplyMessage:
    message_id: str
    tick: int
    sender_id: str
    recipient_id: str
    event_type: EventType
    payload: dict[str, Any]
    signature: str | None = None
    verified: bool = True
    security_alert: str | None = None
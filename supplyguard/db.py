from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from supplyguard.data.generator import GeneratedDataset
from supplyguard.schemas import (
    ComponentSpec,
    Edge,
    EventType,
    InventorySnapshot,
    Node,
    NodeType,
    SupplyMessage,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS components (
    item_code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    criticality REAL NOT NULL,
    base_lead_time_ticks INTEGER NOT NULL,
    unit_cost REAL NOT NULL,
    max_throughput REAL NOT NULL,
    reliability_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    node_type TEXT NOT NULL,
    tier INTEGER NOT NULL,
    capacity_per_tick REAL NOT NULL,
    trust_score REAL NOT NULL,
    is_compromised INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    item_code TEXT NOT NULL,
    base_lead_time_ticks INTEGER NOT NULL,
    unit_cost REAL NOT NULL,
    max_throughput REAL NOT NULL,
    reliability_score REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (source_id) REFERENCES nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES nodes(node_id),
    FOREIGN KEY (item_code) REFERENCES components(item_code)
);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    item_code TEXT NOT NULL,
    on_hand REAL NOT NULL,
    incoming_qty REAL NOT NULL,
    reorder_point REAL NOT NULL,
    demand_rate REAL NOT NULL,
    stockout_qty REAL NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(node_id),
    FOREIGN KEY (item_code) REFERENCES components(item_code)
);

CREATE TABLE IF NOT EXISTS supply_messages (
    message_id TEXT PRIMARY KEY,
    tick INTEGER NOT NULL,
    sender_id TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    signature TEXT,
    verified INTEGER NOT NULL,
    security_alert TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_tick_node ON inventory_snapshots(tick, node_id);
CREATE INDEX IF NOT EXISTS idx_messages_tick ON supply_messages(tick);
"""


class SupplyChainDB:
    def __init__(self, db_path: str | Path = "artifacts/supplyguard_laptop.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)

    def seed_dataset(self, dataset: GeneratedDataset, reset: bool = True) -> None:
        with self.conn:
            if reset:
                self.conn.execute("DELETE FROM supply_messages")
                self.conn.execute("DELETE FROM inventory_snapshots")
                self.conn.execute("DELETE FROM edges")
                self.conn.execute("DELETE FROM nodes")
                self.conn.execute("DELETE FROM components")

            self.conn.executemany(
                """
                INSERT OR REPLACE INTO components
                (item_code, name, criticality, base_lead_time_ticks, unit_cost, max_throughput, reliability_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.item_code, c.name, c.criticality, c.base_lead_time_ticks,
                        c.unit_cost, c.max_throughput, c.reliability_score,
                    )
                    for c in dataset.components
                ],
            )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO nodes
                (node_id, name, node_type, tier, capacity_per_tick, trust_score, is_compromised, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        n.node_id, n.name, n.node_type.value, n.tier,
                        n.capacity_per_tick, n.trust_score, int(n.is_compromised),
                        json.dumps(n.metadata),
                    )
                    for n in dataset.nodes
                ],
            )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO edges
                (edge_id, source_id, target_id, item_code, base_lead_time_ticks, unit_cost, max_throughput, reliability_score, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.edge_id, e.source_id, e.target_id, e.item_code,
                        e.base_lead_time_ticks, e.unit_cost, e.max_throughput,
                        e.reliability_score, json.dumps(e.metadata),
                    )
                    for e in dataset.edges
                ],
            )
            self.conn.executemany(
                """
                INSERT INTO inventory_snapshots
                (tick, node_id, item_code, on_hand, incoming_qty, reorder_point, demand_rate, stockout_qty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.tick, s.node_id, s.item_code, s.on_hand,
                        s.incoming_qty, s.reorder_point, s.demand_rate, s.stockout_qty,
                    )
                    for s in dataset.snapshots
                ],
            )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO supply_messages
                (message_id, tick, sender_id, recipient_id, event_type, payload_json, signature, verified, security_alert)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        m.message_id, m.tick, m.sender_id, m.recipient_id,
                        m.event_type.value, json.dumps(m.payload), m.signature,
                        int(m.verified), m.security_alert,
                    )
                    for m in dataset.messages
                ],
            )

    def fetch_components(self) -> list[ComponentSpec]:
        rows = self.conn.execute("SELECT * FROM components ORDER BY item_code").fetchall()
        return [
            ComponentSpec(
                item_code=r["item_code"], name=r["name"], criticality=r["criticality"],
                base_lead_time_ticks=r["base_lead_time_ticks"], unit_cost=r["unit_cost"],
                max_throughput=r["max_throughput"], reliability_score=r["reliability_score"],
            )
            for r in rows
        ]

    def fetch_nodes(self) -> list[Node]:
        rows = self.conn.execute("SELECT * FROM nodes ORDER BY tier, node_id").fetchall()
        return [
            Node(
                node_id=r["node_id"], name=r["name"], node_type=NodeType(r["node_type"]),
                tier=r["tier"], capacity_per_tick=r["capacity_per_tick"],
                trust_score=r["trust_score"], is_compromised=bool(r["is_compromised"]),
                metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]

    def fetch_edges(self) -> list[Edge]:
        rows = self.conn.execute("SELECT * FROM edges ORDER BY edge_id").fetchall()
        return [
            Edge(
                edge_id=r["edge_id"], source_id=r["source_id"], target_id=r["target_id"],
                item_code=r["item_code"], base_lead_time_ticks=r["base_lead_time_ticks"],
                unit_cost=r["unit_cost"], max_throughput=r["max_throughput"],
                reliability_score=r["reliability_score"], metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]

    def fetch_snapshots(self) -> list[InventorySnapshot]:
        rows = self.conn.execute("SELECT * FROM inventory_snapshots ORDER BY tick, node_id, item_code").fetchall()
        return [
            InventorySnapshot(
                tick=r["tick"], node_id=r["node_id"], item_code=r["item_code"],
                on_hand=r["on_hand"], incoming_qty=r["incoming_qty"],
                reorder_point=r["reorder_point"], demand_rate=r["demand_rate"],
                stockout_qty=r["stockout_qty"],
            )
            for r in rows
        ]

    def fetch_messages(self, only_unverified: bool = False) -> list[SupplyMessage]:
        sql = "SELECT * FROM supply_messages"
        params: tuple[Any, ...] = ()
        if only_unverified:
            sql += " WHERE verified = 0"
        sql += " ORDER BY tick, message_id"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            SupplyMessage(
                message_id=r["message_id"], tick=r["tick"], sender_id=r["sender_id"],
                recipient_id=r["recipient_id"], event_type=EventType(r["event_type"]),
                payload=json.loads(r["payload_json"]), signature=r["signature"],
                verified=bool(r["verified"]), security_alert=r["security_alert"],
            )
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()
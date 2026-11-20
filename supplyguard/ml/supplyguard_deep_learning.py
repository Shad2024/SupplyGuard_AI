#!/usr/bin/env python3
"""
SupplyGuard AI — Standalone Deep Learning Module
================================================

Single-file implementation of the Laptop Manufacturing temporal edge-aware GNN.

What it does:
  1. Reads the Laptop Manufacturing SQLite graph (read-only).
  2. Builds historical node/edge features without future-message leakage.
  3. Trains a GRU + edge-aware message-passing GNN using chronological splits.
  4. Forecasts node risk, stockout probability, stockout quantity, and edge risk.
  5. Saves a checkpoint, metrics, training history, and downstream prediction files.

Dependencies: Python 3.11+, numpy, torch. PyTorch Geometric is optional.

Typical Windows usage from the repository root:
  python supplyguard_deep_learning.py --db backend/supplyguard_laptop.db

If the database is beside this script in backend/supplyguard_laptop.db, --db can be omitted.

Output directory (default): artifacts/ml/
  gnn_checkpoint.pt
  latest_predictions.json
  prediction_index.json
  node_predictions.csv
  edge_predictions.csv
  metrics.json
  scalers.json
  feature_schema.json
  training_history.csv

Prediction contract:
  Node key: (tick, node_id)
    risk_score                 float in [0, 1]
    stockout_prob              float in [0, 1]
    predicted_stockout_qty     float >= 0
  Edge key: (tick, edge_id)
    edge_risk_score            float in [0, 1]

Important:
  - The SQLite database is opened in read-only mode and is never modified.
  - The source dataset has simulation ticks, not calendar dates.
  - This is a research prototype; evaluation on a small synthetic graph is exploratory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



# ==================== CONTRACTS.PY ====================




SCHEMA_VERSION = "1.0.0"
NODE_OUTPUT_FIELDS = (
    "tick",
    "node_id",
    "risk_score",
    "stockout_prob",
    "predicted_stockout_qty",
)
EDGE_OUTPUT_FIELDS = ("tick", "edge_id", "edge_risk_score")


@dataclass(frozen=True)
class NodePrediction:
    tick: int
    node_id: str
    risk_score: float
    stockout_prob: float
    predicted_stockout_qty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.tick < 0 or not self.node_id:
            raise ValueError("Node prediction requires a non-negative tick and node_id.")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1].")
        if not 0.0 <= self.stockout_prob <= 1.0:
            raise ValueError("stockout_prob must be in [0, 1].")
        if not math.isfinite(self.predicted_stockout_qty) or self.predicted_stockout_qty < 0:
            raise ValueError("predicted_stockout_qty must be finite and >= 0.")


@dataclass(frozen=True)
class EdgePrediction:
    tick: int
    edge_id: str
    edge_risk_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.tick < 0 or not self.edge_id:
            raise ValueError("Edge prediction requires a non-negative tick and edge_id.")
        if not 0.0 <= self.edge_risk_score <= 1.0:
            raise ValueError("edge_risk_score must be in [0, 1].")


def validate_prediction_document(document: dict[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported prediction schema: {document.get('schema_version')!r}")
    nodes = document.get("node_predictions")
    edges = document.get("edge_predictions")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Prediction document must contain node_predictions and edge_predictions arrays.")
    for row in nodes:
        NodePrediction(**{key: row[key] for key in NODE_OUTPUT_FIELDS}).validate()
    for row in edges:
        EdgePrediction(**{key: row[key] for key in EDGE_OUTPUT_FIELDS}).validate()


# ==================== DATA.PY ====================





NODE_FEATURE_NAMES = [
    "tick_progress",
    "tier_norm",
    "type_supplier",
    "type_factory",
    "type_warehouse",
    "capacity_log1p",
    "trust_score_prior",
    "inventory_on_hand_log1p",
    "inventory_incoming_log1p",
    "inventory_reorder_point_log1p",
    "inventory_demand_log1p",
    "inventory_stockout_qty_log1p",
    "physics_stockout_baseline_log1p",
    "stockout_item_ratio",
    "mean_on_hand_reorder_ratio",
    "min_on_hand_reorder_ratio",
    "mean_supply_cover_ratio",
    "message_count_lookback_log1p",
    "unverified_count_lookback_log1p",
    "security_alert_count_lookback_log1p",
    "disruption_count_lookback_log1p",
    "verified_ratio_lookback",
    "cumulative_unverified_log1p",
    "cumulative_disruption_log1p",
    "observed_security_flag",
]

EDGE_FEATURE_NAMES = [
    "lead_time_norm",
    "unit_cost_log1p",
    "throughput_log1p",
    "reliability_score",
    "component_criticality",
    "message_count_lookback_log1p",
    "unverified_count_lookback_log1p",
    "security_alert_count_lookback_log1p",
    "delayed_message_count_lookback_log1p",
    "mean_delay_extra_norm",
    "mean_qty_to_throughput",
    "endpoint_disruption_count_lookback_log1p",
]


@dataclass
class GraphData:
    db_path: str
    db_sha256: str
    ticks: list[int]
    node_ids: list[str]
    edge_ids: list[str]
    edge_index: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    node_targets: np.ndarray
    edge_targets: np.ndarray
    stockout_baseline: np.ndarray
    node_feature_names: list[str]
    edge_feature_names: list[str]
    target_available: np.ndarray
    horizon: int
    lookback: int
    source_counts: dict[str, int]


def _table(conn: sqlite3.Connection, preferred: str, fallback: str) -> str:
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if preferred in names:
        return preferred
    if fallback in names:
        return fallback
    raise ValueError(
        f"Database is missing both '{preferred}' and '{fallback}'. "
        "Expected the imported Laptop Manufacturing dataset."
    )


def _read_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    cursor = conn.execute(f'SELECT * FROM "{table}"')
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _safe_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _normalise_id(value: Any) -> str:
    return str(value) if value is not None else ""


def _fit_scaler(values: np.ndarray, fit_ticks: list[int]) -> dict[str, list[float]]:
    fit = values[fit_ticks].reshape(-1, values.shape[-1]).astype(np.float64)
    mean = np.nanmean(fit, axis=0)
    std = np.nanstd(fit, axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def apply_scaler(values: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    scaled = (values.astype(np.float32) - mean) / std
    return np.clip(scaled, -8.0, 8.0).astype(np.float32)


def make_node_sequence(
    scaled_node_features: np.ndarray, tick: int, lookback: int
) -> np.ndarray:
    """Return [node, lookback, feature], left-padding early ticks with tick 0."""
    start = max(0, tick - lookback + 1)
    sequence = scaled_node_features[start : tick + 1]
    if sequence.shape[0] < lookback:
        pad = np.repeat(
            scaled_node_features[[0]],
            lookback - sequence.shape[0],
            axis=0,
        )
        sequence = np.concatenate([pad, sequence], axis=0)
    return np.transpose(sequence, (1, 0, 2)).astype(np.float32)


def load_graph_data(
    db_path: str | Path,
    horizon: int = 3,
    lookback: int = 3,
) -> GraphData:
    """Load the namespaced Laptop Manufacturing SQLite tables without modifying them.

    Security features are derived only from messages observed at or before each tick.
    The static final `is_compromised` and metadata alert totals are intentionally not
    used as historical inputs because the source has no per-tick audit-state history.
    """
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SupplyGuard dataset database not found: {path}")
    if horizon < 1 or lookback < 1:
        raise ValueError("horizon and lookback must both be >= 1")

    uri = path.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        nodes_table = _table(conn, "dataset_nodes", "nodes")
        edges_table = _table(conn, "dataset_edges", "edges")
        components_table = _table(conn, "dataset_components", "components")
        inventory_table = _table(conn, "dataset_inventory_snapshots", "inventory_snapshots")
        messages_table = _table(conn, "dataset_supply_messages", "supply_messages")

        nodes = _read_rows(conn, nodes_table)
        edges = _read_rows(conn, edges_table)
        components = _read_rows(conn, components_table)
        snapshots = _read_rows(conn, inventory_table)
        messages = _read_rows(conn, messages_table)
    finally:
        conn.close()

    nodes.sort(key=lambda row: (int(row.get("tier", 0)), _normalise_id(row.get("node_id"))))
    edges.sort(key=lambda row: _normalise_id(row.get("edge_id")))
    node_ids = [_normalise_id(row["node_id"]) for row in nodes]
    edge_ids = [_normalise_id(row["edge_id"]) for row in edges]
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_index_by_id = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    component_by_code = {
        _normalise_id(row.get("item_code")): row for row in components
    }

    max_tick = max(
        [int(row.get("tick", 0)) for row in snapshots]
        + [int(row.get("tick", 0)) for row in messages]
        + [0]
    )
    ticks = list(range(max_tick + 1))
    tick_count = len(ticks)
    node_count = len(nodes)
    edge_count = len(edges)

    edge_index = np.asarray(
        [
            [node_index[_normalise_id(edge["source_id"])] for edge in edges],
            [node_index[_normalise_id(edge["target_id"])] for edge in edges],
        ],
        dtype=np.int64,
    )

    inventory_by_tick_node: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in snapshots:
        key = (int(row["tick"]), _normalise_id(row["node_id"]))
        inventory_by_tick_node.setdefault(key, []).append(row)

    node_message_counts = np.zeros((tick_count, node_count, 5), dtype=np.float32)
    edge_message_counts = np.zeros((tick_count, edge_count, 7), dtype=np.float32)

    # Messages are assigned to their known sender/recipient and, for disruption
    # events, to the node named in the payload. Spoofed external senders are not
    # silently mapped to a real supplier.
    for message in messages:
        tick = int(message.get("tick", 0))
        if tick < 0 or tick >= tick_count:
            continue
        payload = _safe_json(message.get("payload_json"))
        sender = _normalise_id(message.get("sender_id"))
        recipient = _normalise_id(message.get("recipient_id"))
        event_type = _normalise_id(message.get("event_type")).lower()
        verified = bool(message.get("verified", 0))
        alert = bool(message.get("security_alert"))
        involved = {sender, recipient}
        disrupted_node = _normalise_id(payload.get("disrupted_node"))
        if event_type == "disruption" and disrupted_node:
            involved.add(disrupted_node)

        for node_id in involved:
            idx = node_index.get(node_id)
            if idx is None:
                continue
            node_message_counts[tick, idx, 0] += 1.0
            if not verified:
                node_message_counts[tick, idx, 1] += 1.0
            if alert:
                node_message_counts[tick, idx, 2] += 1.0
            if event_type == "disruption":
                node_message_counts[tick, idx, 3] += 1.0
            if verified:
                node_message_counts[tick, idx, 4] += 1.0

        message_edge_id = _normalise_id(payload.get("edge_id"))
        impacted_edges: set[int] = set()
        if message_edge_id in edge_index_by_id:
            impacted_edges.add(edge_index_by_id[message_edge_id])
        if event_type == "disruption":
            affected = disrupted_node or sender
            if affected in node_index:
                for e_idx, edge in enumerate(edges):
                    if affected in (
                        _normalise_id(edge["source_id"]),
                        _normalise_id(edge["target_id"]),
                    ):
                        impacted_edges.add(e_idx)

        for e_idx in impacted_edges:
            edge_message_counts[tick, e_idx, 0] += 1.0
            if not verified:
                edge_message_counts[tick, e_idx, 1] += 1.0
            if alert:
                edge_message_counts[tick, e_idx, 2] += 1.0
            if event_type == "disruption":
                edge_message_counts[tick, e_idx, 5] += 1.0
            edge = edges[e_idx]
            base_lead = max(1.0, float(edge.get("base_lead_time_ticks", 1.0)))
            eta_tick = payload.get("eta_tick")
            if eta_tick is not None and message_edge_id == _normalise_id(edge["edge_id"]):
                delay_extra = max(
                    0.0,
                    float(eta_tick) - float(tick) - base_lead,
                )
                if delay_extra > 0:
                    edge_message_counts[tick, e_idx, 3] += 1.0
                    edge_message_counts[tick, e_idx, 4] += delay_extra
            qty = payload.get("qty")
            if qty is not None and message_edge_id == _normalise_id(edge["edge_id"]):
                edge_message_counts[tick, e_idx, 6] += max(0.0, float(qty))

    node_features = np.zeros(
        (tick_count, node_count, len(NODE_FEATURE_NAMES)), dtype=np.float32
    )
    stockout_baseline = np.zeros((tick_count, node_count), dtype=np.float32)
    for tick in ticks:
        start = max(0, tick - lookback + 1)
        for n_idx, node in enumerate(nodes):
            node_id = node_ids[n_idx]
            inv_rows = inventory_by_tick_node.get((tick, node_id), [])
            on_hand = [max(0.0, float(row.get("on_hand", 0.0))) for row in inv_rows]
            incoming = [max(0.0, float(row.get("incoming_qty", 0.0))) for row in inv_rows]
            reorder = [max(0.0, float(row.get("reorder_point", 0.0))) for row in inv_rows]
            demand = [max(0.0, float(row.get("demand_rate", 0.0))) for row in inv_rows]
            stockout = [max(0.0, float(row.get("stockout_qty", 0.0))) for row in inv_rows]
            on_hand_reorder = [
                h / max(r, 1.0) for h, r in zip(on_hand, reorder)
            ]
            supply_cover = [
                (h + inc) / max(d, 1.0)
                for h, inc, d in zip(on_hand, incoming, demand)
            ]
            physics_baseline = sum(
                max(0.0, d * horizon - h - inc)
                for h, inc, d in zip(on_hand, incoming, demand)
            )
            stockout_baseline[tick, n_idx] = float(physics_baseline)

            msg_window = node_message_counts[start : tick + 1, n_idx, :].sum(axis=0)
            msg_cumulative = node_message_counts[: tick + 1, n_idx, :].sum(axis=0)
            total_window = float(msg_window[0])
            verified_ratio = (
                float(msg_window[4]) / total_window if total_window > 0 else 1.0
            )
            type_name = _normalise_id(node.get("node_type")).lower()
            values = [
                float(tick) / max(1, tick_count - 1),
                float(node.get("tier", 0)) / 3.0,
                float(type_name == "supplier"),
                float(type_name == "factory"),
                float(type_name == "warehouse"),
                _log1p(float(node.get("capacity_per_tick", 0.0))),
                float(node.get("trust_score", 0.95)),
                _log1p(sum(on_hand)),
                _log1p(sum(incoming)),
                _log1p(sum(reorder)),
                _log1p(sum(demand)),
                _log1p(sum(stockout)),
                _log1p(physics_baseline),
                (sum(1 for value in stockout if value > 0) / max(1, len(stockout))),
                float(np.mean(on_hand_reorder)) if on_hand_reorder else 0.0,
                float(np.min(on_hand_reorder)) if on_hand_reorder else 0.0,
                float(np.mean(supply_cover)) if supply_cover else 0.0,
                _log1p(total_window),
                _log1p(float(msg_window[1])),
                _log1p(float(msg_window[2])),
                _log1p(float(msg_window[3])),
                verified_ratio,
                _log1p(float(msg_cumulative[1])),
                _log1p(float(msg_cumulative[3])),
                float((msg_cumulative[1] + msg_cumulative[2]) > 0),
            ]
            node_features[tick, n_idx, :] = np.asarray(values, dtype=np.float32)

    edge_features = np.zeros(
        (tick_count, edge_count, len(EDGE_FEATURE_NAMES)), dtype=np.float32
    )
    for tick in ticks:
        start = max(0, tick - lookback + 1)
        for e_idx, edge in enumerate(edges):
            item_code = _normalise_id(edge.get("item_code"))
            component = component_by_code.get(item_code, {})
            criticality = _safe_json(edge.get("metadata_json")).get(
                "criticality", component.get("criticality", 0.5)
            )
            msg_window = edge_message_counts[start : tick + 1, e_idx, :].sum(axis=0)
            total = float(msg_window[0])
            mean_delay = (
                float(msg_window[4]) / max(float(msg_window[3]), 1.0)
            )
            throughput = max(1.0, float(edge.get("max_throughput", 1.0)))
            mean_qty_to_throughput = (
                float(msg_window[6]) / max(float(msg_window[0]), 1.0) / throughput
            )
            values = [
                float(edge.get("base_lead_time_ticks", 1.0)) / 10.0,
                _log1p(float(edge.get("unit_cost", 0.0))),
                _log1p(throughput),
                float(edge.get("reliability_score", 0.95)),
                float(criticality),
                _log1p(total),
                _log1p(float(msg_window[1])),
                _log1p(float(msg_window[2])),
                _log1p(float(msg_window[3])),
                mean_delay / 10.0,
                min(2.0, mean_qty_to_throughput),
                _log1p(float(msg_window[5])),
            ]
            edge_features[tick, e_idx, :] = np.asarray(values, dtype=np.float32)

    # Build future-horizon labels. These labels are not used as input features.
    node_targets = np.zeros((tick_count, node_count, 3), dtype=np.float32)
    edge_targets = np.zeros((tick_count, edge_count), dtype=np.float32)
    target_available = np.zeros(tick_count, dtype=bool)

    messages_by_tick: dict[int, list[dict[str, Any]]] = {}
    for message in messages:
        messages_by_tick.setdefault(int(message.get("tick", 0)), []).append(message)

    snapshots_by_tick_node: dict[tuple[int, str], float] = {}
    for row in snapshots:
        key = (int(row["tick"]), _normalise_id(row["node_id"]))
        snapshots_by_tick_node[key] = (
            snapshots_by_tick_node.get(key, 0.0)
            + max(0.0, float(row.get("stockout_qty", 0.0)))
        )

    for tick in ticks:
        if tick + horizon >= tick_count:
            continue
        target_available[tick] = True
        future_ticks = range(tick + 1, tick + horizon + 1)
        future_messages = [
            message
            for future_tick in future_ticks
            for message in messages_by_tick.get(future_tick, [])
        ]

        for n_idx, node_id in enumerate(node_ids):
            future_qty = sum(
                snapshots_by_tick_node.get((future_tick, node_id), 0.0)
                for future_tick in future_ticks
            )
            disruption_flag = False
            cyber_flag = False
            for message in future_messages:
                payload = _safe_json(message.get("payload_json"))
                sender = _normalise_id(message.get("sender_id"))
                recipient = _normalise_id(message.get("recipient_id"))
                event_type = _normalise_id(message.get("event_type")).lower()
                affected = _normalise_id(payload.get("disrupted_node"))
                if event_type == "disruption" and (
                    sender == node_id or affected == node_id
                ):
                    disruption_flag = True
                if (
                    (not bool(message.get("verified", 0)) or bool(message.get("security_alert")))
                    and node_id in {sender, recipient}
                ):
                    cyber_flag = True
            risk_flag = disruption_flag or cyber_flag
            node_targets[tick, n_idx, :] = (
                float(risk_flag),
                float(future_qty > 0.0),
                float(future_qty),
            )

        for e_idx, edge in enumerate(edges):
            edge_id = edge_ids[e_idx]
            source_id = _normalise_id(edge["source_id"])
            target_id = _normalise_id(edge["target_id"])
            base_lead = float(edge.get("base_lead_time_ticks", 1.0))
            edge_risk = False
            for message in future_messages:
                payload = _safe_json(message.get("payload_json"))
                event_type = _normalise_id(message.get("event_type")).lower()
                message_edge_id = _normalise_id(payload.get("edge_id"))
                verified = bool(message.get("verified", 0))
                alert = bool(message.get("security_alert"))
                eta_tick = payload.get("eta_tick")
                delayed = (
                    eta_tick is not None
                    and message_edge_id == edge_id
                    and float(eta_tick) - float(message.get("tick", 0)) > base_lead
                )
                if message_edge_id == edge_id and (
                    (not verified) or alert or delayed
                ):
                    edge_risk = True
                disrupted_node = _normalise_id(payload.get("disrupted_node"))
                sender = _normalise_id(message.get("sender_id"))
                if event_type == "disruption" and (
                    disrupted_node in {source_id, target_id}
                    or sender in {source_id, target_id}
                ):
                    edge_risk = True
            edge_targets[tick, e_idx] = float(edge_risk)

    return GraphData(
        db_path=str(path),
        db_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ticks=ticks,
        node_ids=node_ids,
        edge_ids=edge_ids,
        edge_index=edge_index,
        node_features=node_features,
        edge_features=edge_features,
        node_targets=node_targets,
        edge_targets=edge_targets,
        stockout_baseline=stockout_baseline,
        node_feature_names=list(NODE_FEATURE_NAMES),
        edge_feature_names=list(EDGE_FEATURE_NAMES),
        target_available=target_available,
        horizon=horizon,
        lookback=lookback,
        source_counts={
            "nodes": node_count,
            "edges": edge_count,
            "components": len(components),
            "inventory_snapshots": len(snapshots),
            "supply_messages": len(messages),
            "ticks": tick_count,
        },
    )


# ==================== MODEL.PY ====================



try:
    from torch_geometric.nn import MessagePassing as _MessagePassing
    HAS_PYG = True
except ImportError:
    _MessagePassing = nn.Module
    HAS_PYG = False


class EdgeAwareMessagePassing(_MessagePassing):
    """Directed edge-aware message passing.

    Uses PyTorch Geometric's MessagePassing API when installed. A mathematically
    equivalent index_add_ implementation is provided for environments where
    torch-geometric is not installed (including simpler Windows setups).
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1) -> None:
        if HAS_PYG:
            super().__init__(aggr="mean", node_dim=0)
        else:
            super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.message_net(torch.cat([x_j, edge_attr], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if HAS_PYG:
            aggregated = self.propagate(
                edge_index,
                x=x,
                edge_attr=edge_attr,
                size=(x.size(0), x.size(0)),
            )
        else:
            source, target = edge_index
            messages = self.message_net(torch.cat([x[source], edge_attr], dim=-1))
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, target, messages)
            degree = torch.zeros(
                x.size(0), 1, dtype=x.dtype, device=x.device
            )
            degree.index_add_(
                0,
                target,
                torch.ones(target.shape[0], 1, dtype=x.dtype, device=x.device),
            )
            aggregated = aggregated / degree.clamp_min(1.0)

        update = self.update_net(torch.cat([x, aggregated], dim=-1))
        update = F.dropout(F.gelu(update), p=self.dropout, training=self.training)
        return self.norm(x + update)


class SupplyGuardGNN(nn.Module):
    """GRU + two-hop edge-aware GNN with node and edge prediction heads."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temporal_encoder = nn.GRU(
            input_size=node_feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.node_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.graph_layer_1 = EdgeAwareMessagePassing(hidden_dim, edge_feature_dim, dropout)
        self.graph_layer_2 = EdgeAwareMessagePassing(hidden_dim, edge_feature_dim, dropout)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_sequence: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        stockout_baseline: torch.Tensor | None = None,
        quantity_residual_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # node_sequence: [num_nodes, lookback, node_features]
        _, hidden = self.temporal_encoder(node_sequence)
        node_embedding = self.node_projection(hidden[-1])
        node_embedding = self.graph_layer_1(
            node_embedding, edge_index, edge_features
        )
        node_embedding = self.graph_layer_2(
            node_embedding, edge_index, edge_features
        )

        node_raw = self.node_head(node_embedding)
        source, target = edge_index
        edge_embedding = self.edge_encoder(edge_features)
        edge_input = torch.cat(
            [
                node_embedding[source],
                node_embedding[target],
                edge_embedding,
            ],
            dim=-1,
        )
        edge_logits = self.edge_head(edge_input).squeeze(-1)

        if stockout_baseline is None:
            stockout_qty = F.softplus(node_raw[:, 2])
        else:
            # Physics-informed residual: current demand, on-hand stock and known
            # incoming quantities form a transparent baseline; the GNN learns
            # a bounded log-space correction from graph/security context.
            baseline_log = torch.log1p(stockout_baseline.clamp_min(0.0))
            predicted_log = (
                baseline_log + float(quantity_residual_scale) * node_raw[:, 2]
            ).clamp(min=0.0, max=12.0)
            stockout_qty = torch.expm1(predicted_log)

        return {
            "risk_logits": node_raw[:, 0],
            "stockout_logits": node_raw[:, 1],
            "stockout_qty": stockout_qty,
            "stockout_qty_residual": node_raw[:, 2],
            "edge_logits": edge_logits,
        }


def graph_backend_name() -> str:
    return "torch_geometric.MessagePassing" if HAS_PYG else "native_torch_index_add_fallback"


# ==================== STORE.PY ====================





def load_prediction_index(
    artifacts_path: str | Path = "artifacts/ml",
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    """Load direct lookup indexes keyed by (tick, node_id) and (tick, edge_id).

    `artifacts_path` may be the artifacts directory, latest_predictions.json,
    or prediction_index.json.
    """
    path = Path(artifacts_path).expanduser().resolve()
    if path.is_dir():
        path = path / "latest_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {path}")

    document = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "prediction_index.json":
        nodes_by_tick = document.get("nodes_by_tick", {})
        edges_by_tick = document.get("edges_by_tick", {})
        node_index = {
            (int(tick), str(node_id)): dict(row)
            for tick, records in nodes_by_tick.items()
            for node_id, row in records.items()
        }
        edge_index = {
            (int(tick), str(edge_id)): dict(row)
            for tick, records in edges_by_tick.items()
            for edge_id, row in records.items()
        }
        return node_index, edge_index

    validate_prediction_document(document)
    node_index = {
        (int(row["tick"]), str(row["node_id"])): dict(row)
        for row in document["node_predictions"]
    }
    edge_index = {
        (int(row["tick"]), str(row["edge_id"])): dict(row)
        for row in document["edge_predictions"]
    }
    return node_index, edge_index


def save_prediction_index(
    document: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Save JSON-native nested indexes while retaining the tuple-key Python API."""
    validate_prediction_document(document)
    nodes_by_tick: dict[str, dict[str, dict[str, Any]]] = {}
    edges_by_tick: dict[str, dict[str, dict[str, Any]]] = {}

    for row in document["node_predictions"]:
        tick = str(int(row["tick"]))
        nodes_by_tick.setdefault(tick, {})[str(row["node_id"])] = dict(row)
    for row in document["edge_predictions"]:
        tick = str(int(row["tick"]))
        edges_by_tick.setdefault(tick, {})[str(row["edge_id"])] = dict(row)

    payload = {
        "schema_version": document["schema_version"],
        "generated_at_utc": document.get("generated_at_utc"),
        "forecast_horizon_ticks": document.get("forecast_horizon_ticks"),
        "nodes_by_tick": nodes_by_tick,
        "edges_by_tick": edges_by_tick,
    }
    path = Path(output_dir).resolve() / "prediction_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)
    return path


# ==================== PREDICT.PY ====================






def _torch_load(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


@torch.no_grad()
def predict_from_checkpoint(
    db_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    allow_dataset_change: bool = False,
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path).resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {checkpoint_file}")
    checkpoint = _torch_load(checkpoint_file)
    config = checkpoint["model_config"]
    horizon = int(config["horizon"])
    lookback = int(config["lookback"])

    data = load_graph_data(db_path, horizon=horizon, lookback=lookback)
    expected_hash = checkpoint.get("source_database_sha256")
    if expected_hash and data.db_sha256 != expected_hash and not allow_dataset_change:
        raise ValueError(
            "The SQLite database differs from the one used to train this checkpoint. "
            "Retrain the model or pass --allow-dataset-change deliberately."
        )
    if data.node_ids != checkpoint.get("node_ids"):
        raise ValueError("Node IDs/order differ from the checkpoint; retraining is required.")
    if data.edge_ids != checkpoint.get("edge_ids"):
        raise ValueError("Edge IDs/order differ from the checkpoint; retraining is required.")
    if data.node_feature_names != checkpoint.get("node_feature_names"):
        raise ValueError("Node feature schema differs from the checkpoint.")
    if data.edge_feature_names != checkpoint.get("edge_feature_names"):
        raise ValueError("Edge feature schema differs from the checkpoint.")

    scalers = checkpoint["scalers"]
    scaled_node = apply_scaler(data.node_features, scalers["node"])
    scaled_edge = apply_scaler(data.edge_features, scalers["edge"])

    model = SupplyGuardGNN(
        node_feature_dim=int(config["node_feature_dim"]),
        edge_feature_dim=int(config["edge_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device("cpu")
    model.to(device)
    edge_index = torch.as_tensor(data.edge_index, dtype=torch.long, device=device)

    node_predictions: list[dict[str, Any]] = []
    edge_predictions: list[dict[str, Any]] = []
    for tick in data.ticks:
        node_sequence = make_node_sequence(scaled_node, tick, lookback)
        node_tensor = torch.as_tensor(
            node_sequence, dtype=torch.float32, device=device
        )
        edge_tensor = torch.as_tensor(
            scaled_edge[tick], dtype=torch.float32, device=device
        )
        baseline_tensor = torch.as_tensor(
            data.stockout_baseline[tick], dtype=torch.float32, device=device
        )
        output = model(
            node_tensor, edge_tensor, edge_index, baseline_tensor,
            quantity_residual_scale=float(config.get("quantity_residual_scale", 1.0)),
        )

        risk_values = torch.sigmoid(output["risk_logits"]).cpu().numpy()
        stockout_values = torch.sigmoid(output["stockout_logits"]).cpu().numpy()
        qty_values = output["stockout_qty"].cpu().numpy()
        edge_values = torch.sigmoid(output["edge_logits"]).cpu().numpy()

        for idx, node_id in enumerate(data.node_ids):
            row = NodePrediction(
                tick=int(tick),
                node_id=node_id,
                risk_score=float(np.clip(risk_values[idx], 0.0, 1.0)),
                stockout_prob=float(np.clip(stockout_values[idx], 0.0, 1.0)),
                predicted_stockout_qty=max(0.0, float(qty_values[idx])),
            )
            row.validate()
            node_predictions.append(row.to_dict())

        for idx, edge_id in enumerate(data.edge_ids):
            row = EdgePrediction(
                tick=int(tick),
                edge_id=edge_id,
                edge_risk_score=float(np.clip(edge_values[idx], 0.0, 1.0)),
            )
            row.validate()
            edge_predictions.append(row.to_dict())

    document = {
        "schema_version": SCHEMA_VERSION,
        "model_name": checkpoint.get("model_name", "SupplyGuard Temporal Edge-Aware GNN"),
        "model_backend_at_training": checkpoint.get("model_backend", "unknown"),
        "checkpoint_version": checkpoint.get("checkpoint_version", "1.0.0"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_database_sha256": data.db_sha256,
        "forecast_horizon_ticks": horizon,
        "lookback_ticks": lookback,
        "quantity_residual_scale": float(config.get("quantity_residual_scale", 1.0)),
        "source_counts": data.source_counts,
        "coverage": {
            "ticks": [int(t) for t in data.ticks],
            "node_prediction_records": len(node_predictions),
            "edge_prediction_records": len(edge_predictions),
            "node_ids": data.node_ids,
            "edge_ids": data.edge_ids,
            "ticks_with_training_targets": [
                int(t) for t, available in zip(data.ticks, data.target_available) if available
            ],
        },
        "node_predictions": node_predictions,
        "edge_predictions": edge_predictions,
    }
    validate_prediction_document(document)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "latest_predictions.json", document)
    _write_csv(
        output_path / "node_predictions.csv",
        node_predictions,
        NODE_OUTPUT_FIELDS,
    )
    _write_csv(
        output_path / "edge_predictions.csv",
        edge_predictions,
        EDGE_OUTPUT_FIELDS,
    )
    save_prediction_index(document, output_path)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SupplyGuard node/edge risk predictions.")
    parser.add_argument("--db", default="backend/supplyguard_laptop.db")
    parser.add_argument("--checkpoint", default="artifacts/ml/gnn_checkpoint.pt")
    parser.add_argument("--output", default="artifacts/ml")
    parser.add_argument("--allow-dataset-change", action="store_true")
    args = parser.parse_args()
    document = predict_from_checkpoint(
        db_path=args.db,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        allow_dataset_change=args.allow_dataset_change,
    )
    print(json.dumps({
        "predictions_file": str(Path(args.output).resolve() / "latest_predictions.json"),
        "node_prediction_records": len(document["node_predictions"]),
        "edge_prediction_records": len(document["edge_predictions"]),
        "ticks": document["coverage"]["ticks"],
        "forecast_horizon_ticks": document["forecast_horizon_ticks"],
    }, indent=2))




# ==================== TRAIN.PY ====================






def temporal_split(tick_count: int, horizon: int) -> dict[str, list[int]]:
    """Purged chronological split with disjoint future target windows.

    For the bundled 30-tick dataset and horizon=3:
      train anchors: 0..12 -> target ticks 1..15
      validation:    15..17 -> target ticks 16..20
      test:          20..26 -> target ticks 21..29
    Anchors 13-14 and 18-19 are purged to prevent target-window overlap.
    """
    max_anchor = tick_count - horizon - 1
    if max_anchor < 5:
        raise ValueError(
            f"Need more temporal history: tick_count={tick_count}, horizon={horizon}."
        )
    first_target_cut = int(math.floor(tick_count * 0.57))
    second_target_cut = int(math.floor(tick_count * 0.75))
    available = list(range(max_anchor + 1))

    train = [t for t in available if t + horizon <= first_target_cut]
    validation = [
        t for t in available
        if t + 1 > first_target_cut and t + horizon <= second_target_cut
    ]
    test = [t for t in available if t + 1 > second_target_cut]

    if not train or not validation or not test:
        # A small-dataset fallback still keeps time ordering and a purge gap.
        n = len(available)
        train_end = max(1, int(n * 0.55))
        val_end = max(train_end + 1, int(n * 0.75))
        train = available[:train_end]
        validation = available[min(n, train_end + horizon):val_end]
        test = available[min(n, val_end + horizon):]
    if not train or not validation or not test:
        raise ValueError(
            "Temporal split produced an empty partition. Use more ticks or a shorter horizon."
        )
    return {"train": train, "validation": validation, "test": test}


def _pos_weight(labels: np.ndarray) -> torch.Tensor:
    positive = float(labels.sum())
    negative = float(labels.size - positive)
    if positive <= 0 or negative <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(
        min(5.0, max(0.5, negative / positive)),
        dtype=torch.float32,
    )


def _sample_tensors(
    data: Any,
    tick: int,
    scaled_node: np.ndarray,
    scaled_edge: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_seq = make_node_sequence(scaled_node, tick, data.lookback)
    edge_attr = scaled_edge[tick]
    return (
        torch.as_tensor(node_seq, dtype=torch.float32, device=device),
        torch.as_tensor(edge_attr, dtype=torch.float32, device=device),
        torch.as_tensor(data.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(data.stockout_baseline[tick], dtype=torch.float32, device=device),
    )


def _loss_for_tick(
    model: SupplyGuardGNN,
    data: Any,
    tick: int,
    scaled_node: np.ndarray,
    scaled_edge: np.ndarray,
    device: torch.device,
    risk_pos_weight: torch.Tensor,
    stock_pos_weight: torch.Tensor,
    edge_pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    node_seq, edge_attr, edge_index, stockout_baseline = _sample_tensors(
        data, tick, scaled_node, scaled_edge, device
    )
    output = model(node_seq, edge_attr, edge_index, stockout_baseline)
    target = torch.as_tensor(data.node_targets[tick], dtype=torch.float32, device=device)
    edge_target = torch.as_tensor(
        data.edge_targets[tick], dtype=torch.float32, device=device
    )

    risk_loss = F.binary_cross_entropy_with_logits(
        output["risk_logits"], target[:, 0], pos_weight=risk_pos_weight
    )
    stock_loss = F.binary_cross_entropy_with_logits(
        output["stockout_logits"], target[:, 1], pos_weight=stock_pos_weight
    )
    qty_loss = F.smooth_l1_loss(
        torch.log1p(output["stockout_qty"]),
        torch.log1p(target[:, 2]),
    )
    residual_penalty = 0.02 * output["stockout_qty_residual"].pow(2).mean()
    edge_loss = F.binary_cross_entropy_with_logits(
        output["edge_logits"], edge_target, pos_weight=edge_pos_weight
    )
    total = risk_loss + stock_loss + (0.50 * qty_loss) + edge_loss + residual_penalty
    parts = {
        "risk_loss": float(risk_loss.detach().cpu()),
        "stockout_loss": float(stock_loss.detach().cpu()),
        "qty_loss": float(qty_loss.detach().cpu()),
        "edge_loss": float(edge_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, parts


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int64).reshape(-1)
    scores = scores.astype(np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y = labels.astype(np.int64).reshape(-1)
    p = np.clip(probabilities.astype(np.float64).reshape(-1), 0.0, 1.0)
    predicted = (p >= 0.5).astype(np.int64)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "records": int(y.size),
        "positive_rate": float(y.mean()) if y.size else 0.0,
        "roc_auc": _auc(y, p),
        "brier_score": float(np.mean((p - y) ** 2)) if y.size else None,
        "accuracy_at_0_5": float((predicted == y).mean()) if y.size else None,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(2 * precision * recall / max(1e-12, precision + recall)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


@torch.no_grad()
def _evaluate(
    model: SupplyGuardGNN,
    data: Any,
    anchors: list[int],
    scaled_node: np.ndarray,
    scaled_edge: np.ndarray,
    device: torch.device,
    quantity_residual_scale: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    all_risk_y, all_risk_p = [], []
    all_stock_y, all_stock_p = [], []
    all_qty_y, all_qty_p, all_qty_baseline = [], [], []
    all_edge_y, all_edge_p = [], []
    losses = []
    for tick in anchors:
        node_seq, edge_attr, edge_index, stockout_baseline = _sample_tensors(
            data, tick, scaled_node, scaled_edge, device
        )
        output = model(
            node_seq, edge_attr, edge_index, stockout_baseline,
            quantity_residual_scale=quantity_residual_scale,
        )
        target = data.node_targets[tick]
        edge_target = data.edge_targets[tick]
        risk_y = torch.as_tensor(target[:, 0], dtype=torch.float32, device=device)
        stock_y = torch.as_tensor(target[:, 1], dtype=torch.float32, device=device)
        qty_y = torch.as_tensor(target[:, 2], dtype=torch.float32, device=device)
        edge_y = torch.as_tensor(edge_target, dtype=torch.float32, device=device)

        loss = (
            F.binary_cross_entropy_with_logits(output["risk_logits"], risk_y)
            + F.binary_cross_entropy_with_logits(output["stockout_logits"], stock_y)
            + 0.25 * F.smooth_l1_loss(
                torch.log1p(output["stockout_qty"]), torch.log1p(qty_y)
            )
            + F.binary_cross_entropy_with_logits(output["edge_logits"], edge_y)
        )
        losses.append(float(loss.cpu()))
        all_risk_y.extend(target[:, 0].tolist())
        all_risk_p.extend(torch.sigmoid(output["risk_logits"]).cpu().tolist())
        all_stock_y.extend(target[:, 1].tolist())
        all_stock_p.extend(torch.sigmoid(output["stockout_logits"]).cpu().tolist())
        all_qty_y.extend(target[:, 2].tolist())
        all_qty_p.extend(output["stockout_qty"].cpu().tolist())
        all_qty_baseline.extend(stockout_baseline.cpu().tolist())
        all_edge_y.extend(edge_target.tolist())
        all_edge_p.extend(torch.sigmoid(output["edge_logits"]).cpu().tolist())

    qty_y_np = np.asarray(all_qty_y, dtype=np.float64)
    qty_p_np = np.asarray(all_qty_p, dtype=np.float64)
    qty_baseline_np = np.asarray(all_qty_baseline, dtype=np.float64)
    metrics = {
        "anchors": anchors,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "node_risk": _classification_metrics(
            np.asarray(all_risk_y), np.asarray(all_risk_p)
        ),
        "stockout_probability": _classification_metrics(
            np.asarray(all_stock_y), np.asarray(all_stock_p)
        ),
        "predicted_stockout_qty": {
            "records": int(qty_y_np.size),
            "mae": float(np.mean(np.abs(qty_p_np - qty_y_np))) if qty_y_np.size else None,
            "rmse": float(np.sqrt(np.mean((qty_p_np - qty_y_np) ** 2))) if qty_y_np.size else None,
            "actual_mean": float(qty_y_np.mean()) if qty_y_np.size else None,
            "predicted_mean": float(qty_p_np.mean()) if qty_p_np.size else None,
            "physics_baseline_mae": float(np.mean(np.abs(qty_baseline_np - qty_y_np))) if qty_y_np.size else None,
            "physics_baseline_mean": float(qty_baseline_np.mean()) if qty_baseline_np.size else None,
        },
        "edge_risk": _classification_metrics(
            np.asarray(all_edge_y), np.asarray(all_edge_p)
        ),
    }
    return metrics["mean_loss"] or 0.0, metrics


@torch.no_grad()
def _select_quantity_residual_scale(
    model: SupplyGuardGNN,
    data: Any,
    validation_ticks: list[int],
    scaled_node: np.ndarray,
    scaled_edge: np.ndarray,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Choose residual shrinkage on validation only; never inspect test targets."""
    model.eval()
    candidates = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    actual: list[float] = []
    baseline: list[float] = []
    residual: list[float] = []
    for tick in validation_ticks:
        node_seq, edge_attr, edge_index, stockout_baseline = _sample_tensors(
            data, tick, scaled_node, scaled_edge, device
        )
        output = model(node_seq, edge_attr, edge_index, stockout_baseline)
        actual.extend(data.node_targets[tick, :, 2].tolist())
        baseline.extend(stockout_baseline.cpu().tolist())
        residual.extend(output["stockout_qty_residual"].cpu().tolist())

    y = np.asarray(actual, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    scores: dict[str, float] = {}
    best_scale = 0.0
    best_mae = float("inf")
    for scale in candidates:
        pred_log = np.clip(np.log1p(b) + scale * r, 0.0, 12.0)
        pred = np.expm1(pred_log)
        mae = float(np.mean(np.abs(pred - y)))
        scores[str(scale)] = mae
        if mae < best_mae - 1e-9:
            best_mae = mae
            best_scale = scale
    return best_scale, scores


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def train_model(
    db_path: str | Path,
    output_dir: str | Path,
    horizon: int = 3,
    lookback: int = 3,
    hidden_dim: int = 64,
    epochs: int = 180,
    patience: int = 30,
    learning_rate: float = 0.002,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    data = load_graph_data(db_path, horizon=horizon, lookback=lookback)
    splits = temporal_split(len(data.ticks), horizon)
    train_ticks = splits["train"]

    node_scaler = _fit_scaler(data.node_features, train_ticks)
    edge_scaler = _fit_scaler(data.edge_features, train_ticks)
    scaled_node = apply_scaler(data.node_features, node_scaler)
    scaled_edge = apply_scaler(data.edge_features, edge_scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupplyGuardGNN(
        node_feature_dim=data.node_features.shape[-1],
        edge_feature_dim=data.edge_features.shape[-1],
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_node_targets = data.node_targets[train_ticks]
    train_edge_targets = data.edge_targets[train_ticks]
    risk_weight = _pos_weight(train_node_targets[:, :, 0]).to(device)
    stock_weight = _pos_weight(train_node_targets[:, :, 1]).to(device)
    edge_weight = _pos_weight(train_edge_targets).to(device)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        order = np.random.default_rng(seed + epoch).permutation(train_ticks)
        for tick in order.tolist():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss_for_tick(
                model, data, tick, scaled_node, scaled_edge, device,
                risk_weight, stock_weight, edge_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_loss, _ = _evaluate(
            model, data, splits["validation"], scaled_node, scaled_edge, device
        )
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()

    quantity_residual_scale, quantity_scale_validation_mae = _select_quantity_residual_scale(
        model, data, splits["validation"], scaled_node, scaled_edge, device
    )
    train_metrics = _evaluate(
        model, data, splits["train"], scaled_node, scaled_edge, device,
        quantity_residual_scale=quantity_residual_scale,
    )[1]
    validation_metrics = _evaluate(
        model, data, splits["validation"], scaled_node, scaled_edge, device,
        quantity_residual_scale=quantity_residual_scale,
    )[1]
    test_metrics = _evaluate(
        model, data, splits["test"], scaled_node, scaled_edge, device,
        quantity_residual_scale=quantity_residual_scale,
    )[1]

    scaler_payload = {
        "node": node_scaler,
        "edge": edge_scaler,
        "fit_anchor_ticks": train_ticks,
        "fit_policy": "training-anchor ticks only; no validation/test statistics used",
    }
    checkpoint = {
        "checkpoint_version": "1.0.0",
        "model_name": "SupplyGuard Temporal Edge-Aware GNN",
        "model_backend": graph_backend_name(),
        "model_config": {
            "node_feature_dim": int(data.node_features.shape[-1]),
            "edge_feature_dim": int(data.edge_features.shape[-1]),
            "hidden_dim": int(hidden_dim),
            "lookback": int(lookback),
            "horizon": int(horizon),
            "quantity_mode": "physics_informed_log_residual",
            "quantity_residual_scale": float(quantity_residual_scale),
        },
        "node_feature_names": data.node_feature_names,
        "edge_feature_names": data.edge_feature_names,
        "node_ids": data.node_ids,
        "edge_ids": data.edge_ids,
        "edge_index": data.edge_index.tolist(),
        "scalers": scaler_payload,
        "state_dict": model.state_dict(),
        "source_database_sha256": data.db_sha256,
        "source_counts": data.source_counts,
        "temporal_split": splits,
        "best_epoch": best_epoch,
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_path / "gnn_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    _write_json(output_path / "scalers.json", scaler_payload)
    _write_json(output_path / "feature_schema.json", {
        "node_features": data.node_feature_names,
        "edge_features": data.edge_feature_names,
        "target_order": ["risk_score", "stockout_prob", "predicted_stockout_qty"],
        "edge_target": "edge_risk_score",
        "forecast_horizon_ticks": horizon,
        "lookback_ticks": lookback,
        "security_feature_policy": (
            "Dynamic message audit counts use only messages at or before the anchor tick. "
            "Static final is_compromised and metadata alert totals are excluded from historical inputs."
        ),
    })

    history_path = output_path / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "model_name": "SupplyGuard Temporal Edge-Aware GNN",
        "model_backend": graph_backend_name(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_database_sha256": data.db_sha256,
        "source_counts": data.source_counts,
        "forecast_horizon_ticks": horizon,
        "lookback_ticks": lookback,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_val_loss,
        "quantity_residual_scale_selected_on_validation": quantity_residual_scale,
        "quantity_residual_scale_validation_mae": quantity_scale_validation_mae,
        "device": str(device),
        "temporal_split": splits,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "limitations": [
            "The bundled source contains 30 synthetic simulation ticks, 12 nodes, and 16 directed edges.",
            "The test partition is a small chronological holdout; metrics are exploratory, not production guarantees.",
            "The source has static final compromise flags rather than a per-tick compromise history; those flags are not used as historical input features.",
            "Stockout probability means at least one positive stockout_qty record for the node across the next horizon ticks.",
            "Predicted stockout quantity is the cumulative shortfall across all five item codes over the next horizon ticks.",
            "The quantity head learns a graph-informed log-space residual over a transparent demand/on-hand/incoming physics baseline; its residual scale is selected on validation only.",
        ],
        "training_seconds": round(time.time() - start_time, 3),
    }
    _write_json(output_path / "metrics.json", metrics)

    prediction_document = predict_from_checkpoint(
        db_path=db_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_path,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "predictions": str(output_path / "latest_predictions.json"),
        "metrics": str(output_path / "metrics.json"),
        "node_prediction_records": len(prediction_document["node_predictions"]),
        "edge_prediction_records": len(prediction_document["edge_predictions"]),
        "metrics_summary": {
            "test_node_risk_auc": test_metrics["node_risk"]["roc_auc"],
            "test_stockout_auc": test_metrics["stockout_probability"]["roc_auc"],
            "test_edge_risk_auc": test_metrics["edge_risk"]["roc_auc"],
            "test_stockout_qty_mae": test_metrics["predicted_stockout_qty"]["mae"],
        },
        "model_backend": graph_backend_name(),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "temporal_split": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SupplyGuard temporal GNN.")
    parser.add_argument("--db", default="backend/supplyguard_laptop.db")
    parser.add_argument("--output", default="artifacts/ml")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_model(
        db_path=args.db,
        output_dir=args.output,
        horizon=args.horizon,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))




# ==================== STANDALONE COMMAND-LINE INTERFACE ====================

def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_db_path(user_value: str | None) -> Path:
    if user_value:
        path = Path(user_value).expanduser()
        if not path.is_absolute():
            path = _project_root() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")
        return path

    candidates = [
        _project_root() / "backend" / "supplyguard_laptop.db",
        _project_root() / "artifacts" / "supplyguard_laptop.db",
        _project_root() / "supplyguard_laptop.db",
        Path.cwd() / "backend" / "supplyguard_laptop.db",
        Path.cwd() / "supplyguard_laptop.db",
        _project_root() / "backend" / "data" / "laptop_supply_chain_source.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the Laptop Manufacturing SQLite database. "
        "Pass its path explicitly, for example: "
        "--db backend/supplyguard_laptop.db"
    )


def _resolve_output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or run the standalone SupplyGuard temporal GNN."
    )
    parser.add_argument(
        "--mode", choices=("train", "predict"), default="train",
        help="Train a new model (default) or predict using an existing checkpoint.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to backend/supplyguard_laptop.db. Auto-detected when omitted.",
    )
    parser.add_argument("--output", default="artifacts/ml", help="Output folder.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for --mode predict.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon in ticks.")
    parser.add_argument("--lookback", type=int, default=3, help="Historical lookback in ticks.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-dataset-change", action="store_true",
        help="Allow prediction on a database with a different SHA-256 (IDs/schema must still match).",
    )
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)
    output_path = _resolve_output_path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        result = train_model(
            db_path=db_path,
            output_dir=output_path,
            horizon=args.horizon,
            lookback=args.lookback,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        checkpoint_path = (
            Path(args.checkpoint).expanduser()
            if args.checkpoint else output_path / "gnn_checkpoint.pt"
        )
        if not checkpoint_path.is_absolute():
            checkpoint_path = _project_root() / checkpoint_path
        document = predict_from_checkpoint(
            db_path=db_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_path,
            allow_dataset_change=args.allow_dataset_change,
        )
        print(json.dumps({
            "predictions_file": str(output_path / "latest_predictions.json"),
            "node_prediction_records": len(document["node_predictions"]),
            "edge_prediction_records": len(document["edge_predictions"]),
            "ticks": document["coverage"]["ticks"],
            "forecast_horizon_ticks": document["forecast_horizon_ticks"],
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

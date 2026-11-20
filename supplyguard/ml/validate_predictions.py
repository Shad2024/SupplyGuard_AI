"""Validate SupplyGuard ML prediction files against the shared output contract."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        default="supplyguard/ml/artifacts/ml",
        help="Directory containing latest_predictions.json and prediction CSV files.",
    )
    args = parser.parse_args()
    root = Path(args.artifacts).expanduser().resolve()

    with (root / "latest_predictions.json").open(encoding="utf-8") as handle:
        document = json.load(handle)
    with (root / "node_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        nodes = list(csv.DictReader(handle))
    with (root / "edge_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        edges = list(csv.DictReader(handle))

    node_fields = {
        "tick", "node_id", "risk_score", "stockout_prob", "predicted_stockout_qty"
    }
    edge_fields = {"tick", "edge_id", "edge_risk_score"}
    if not nodes or not edges:
        raise SystemExit("FAIL: node_predictions.csv or edge_predictions.csv is empty")
    if not node_fields.issubset(nodes[0]):
        raise SystemExit(f"FAIL: missing node fields: {sorted(node_fields - set(nodes[0]))}")
    if not edge_fields.issubset(edges[0]):
        raise SystemExit(f"FAIL: missing edge fields: {sorted(edge_fields - set(edges[0]))}")

    for row in nodes:
        risk = float(row["risk_score"])
        probability = float(row["stockout_prob"])
        quantity = float(row["predicted_stockout_qty"])
        if not (math.isfinite(risk) and 0.0 <= risk <= 1.0):
            raise SystemExit("FAIL: invalid node risk_score")
        if not (math.isfinite(probability) and 0.0 <= probability <= 1.0):
            raise SystemExit("FAIL: invalid stockout_prob")
        if not (math.isfinite(quantity) and quantity >= 0.0):
            raise SystemExit("FAIL: invalid predicted_stockout_qty")

    for row in edges:
        risk = float(row["edge_risk_score"])
        if not (math.isfinite(risk) and 0.0 <= risk <= 1.0):
            raise SystemExit("FAIL: invalid edge_risk_score")

    node_keys = [(row["tick"], row["node_id"]) for row in nodes]
    edge_keys = [(row["tick"], row["edge_id"]) for row in edges]
    if len(node_keys) != len(set(node_keys)):
        raise SystemExit("FAIL: duplicate (tick, node_id) keys")
    if len(edge_keys) != len(set(edge_keys)):
        raise SystemExit("FAIL: duplicate (tick, edge_id) keys")

    json_nodes = document.get("node_predictions", [])
    json_edges = document.get("edge_predictions", [])
    if len(nodes) != len(json_nodes) or len(edges) != len(json_edges):
        raise SystemExit("FAIL: JSON/CSV record counts do not match")

    print("========== CONTRACT VALIDATION ==========")
    print(f"Node records: {len(nodes)}")
    print(f"Edge records: {len(edges)}")
    print("Node fields and ranges: PASS")
    print("Edge fields and ranges: PASS")
    print("Unique (tick, node_id): PASS")
    print("Unique (tick, edge_id): PASS")
    print("JSON/CSV record counts: PASS")
    print("========== ALL CHECKS PASSED ==========")


if __name__ == "__main__":
    main()

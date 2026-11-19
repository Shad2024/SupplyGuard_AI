from __future__ import annotations

import random
from dataclasses import dataclass

from supplyguard.schemas import (
    ComponentSpec,
    Edge,
    EventType,
    InventorySnapshot,
    Node,
    NodeType,
    SupplyMessage,
)


@dataclass
class GeneratedDataset:
    domain: str
    components: list[ComponentSpec]
    nodes: list[Node]
    edges: list[Edge]
    snapshots: list[InventorySnapshot]
    messages: list[SupplyMessage]


def get_laptop_domain_specs() -> tuple[list[ComponentSpec], list[tuple[str, str, NodeType, int, float, float, str]]]:
    components = [
        ComponentSpec(
            item_code="PANEL-OLED-15",
            name="15.6-inch OLED Display Assembly",
            criticality=0.95,
            base_lead_time_ticks=4,
            unit_cost=115.0,
            max_throughput=65.0,
            reliability_score=0.91,
        ),
        ComponentSpec(
            item_code="MAINBOARD-SOC",
            name="Mainboard + CPU/GPU SoC Module",
            criticality=0.98,
            base_lead_time_ticks=5,
            unit_cost=190.0,
            max_throughput=55.0,
            reliability_score=0.89,
        ),
        ComponentSpec(
            item_code="BATT-LI-70WH",
            name="70Wh Lithium-Ion Battery Pack",
            criticality=0.85,
            base_lead_time_ticks=3,
            unit_cost=48.0,
            max_throughput=80.0,
            reliability_score=0.93,
        ),
        ComponentSpec(
            item_code="CHASSIS-CNC-ALU",
            name="CNC Machined Aluminum Chassis",
            criticality=0.75,
            base_lead_time_ticks=3,
            unit_cost=62.0,
            max_throughput=85.0,
            reliability_score=0.96,
        ),
        ComponentSpec(
            item_code="LAPTOP-PRO-15",
            name="Assembled ProBook-15 Laptop Unit",
            criticality=1.00,
            base_lead_time_ticks=2,
            unit_cost=680.0,
            max_throughput=110.0,
            reliability_score=0.97,
        ),
    ]

    node_defs = [
        ("T0-GLASS-01", "OLED Substrate & Driver IC Vendor", NodeType.SUPPLIER, 0, 75.0, 0.91, "East Asia"),
        ("T0-SIL-01", "Silicon Wafer & Power IC Foundry", NodeType.SUPPLIER, 0, 65.0, 0.89, "East Asia"),
        ("T0-CELL-01", "High-Density Lithium Cell Plant", NodeType.SUPPLIER, 0, 90.0, 0.92, "Southeast Asia"),
        ("T0-ALLOY-01", "Aerospace Aluminum Ingot Mill", NodeType.SUPPLIER, 0, 95.0, 0.95, "Central Europe"),
        ("T1-DISP-01", "Precision Display Module Integrator", NodeType.SUPPLIER, 1, 70.0, 0.90, "East Asia"),
        ("T1-PCBA-01", "SMT Motherboard Assembly Partner", NodeType.SUPPLIER, 1, 60.0, 0.88, "East Asia"),
        ("T1-BATT-01", "Smart BMS Battery Pack Supplier", NodeType.SUPPLIER, 1, 85.0, 0.93, "North Africa / EU"),
        ("T1-CHSS-01", "CNC Chassis & Thermal Enclosures", NodeType.SUPPLIER, 1, 85.0, 0.94, "Eastern Europe"),
        ("FAC-LAP-01", "Primary Laptop Final Assembly Plant A", NodeType.FACTORY, 2, 110.0, 0.98, "EU Hub"),
        ("FAC-LAP-02", "Secondary Laptop Final Assembly Plant B", NodeType.FACTORY, 2, 95.0, 0.97, "MENA Hub"),
        ("WH-REG-01", "EMEA Regional Distribution Center", NodeType.WAREHOUSE, 3, 140.0, 0.99, "Central EU"),
        ("WH-REG-02", "MENA & Regional Fulfillment Hub", NodeType.WAREHOUSE, 3, 120.0, 0.99, "Cairo / Regional"),
    ]
    return components, node_defs


def generate_laptop_supply_chain(
    seed: int = 42,
    num_ticks: int = 30,
    disruption_prob_per_tick: float = 0.06,
    attack_prob_per_tick: float = 0.05,
) -> GeneratedDataset:
    rng = random.Random(seed)
    components, node_defs = get_laptop_domain_specs()
    comp_by_code = {c.item_code: c for c in components}

    nodes: list[Node] = [
        Node(
            node_id=nid,
            name=name,
            node_type=ntype,
            tier=tier,
            capacity_per_tick=cap,
            trust_score=trust,
            is_compromised=False,
            metadata={"region": region, "domain": "laptop_manufacturing"},
        )
        for nid, name, ntype, tier, cap, trust, region in node_defs
    ]

    edges: list[Edge] = []
    edge_counter = 1

    def add_edge(src: str, dst: str, item_code: str, lt_jitter: int = 0, cost_mult: float = 1.0) -> None:
        nonlocal edge_counter
        spec = comp_by_code[item_code]
        edges.append(
            Edge(
                edge_id=f"E-{edge_counter:04d}",
                source_id=src,
                target_id=dst,
                item_code=item_code,
                base_lead_time_ticks=max(1, spec.base_lead_time_ticks + lt_jitter),
                unit_cost=round(spec.unit_cost * cost_mult, 2),
                max_throughput=round(spec.max_throughput * rng.uniform(0.75, 1.0), 2),
                reliability_score=round(min(0.99, spec.reliability_score + rng.uniform(-0.03, 0.03)), 3),
                metadata={"criticality": spec.criticality},
            )
        )
        edge_counter += 1

    add_edge("T0-GLASS-01", "T1-DISP-01", "PANEL-OLED-15", lt_jitter=-1, cost_mult=0.55)
    add_edge("T0-SIL-01", "T1-PCBA-01", "MAINBOARD-SOC", lt_jitter=-1, cost_mult=0.65)
    add_edge("T0-CELL-01", "T1-BATT-01", "BATT-LI-70WH", lt_jitter=-1, cost_mult=0.60)
    add_edge("T0-ALLOY-01", "T1-CHSS-01", "CHASSIS-CNC-ALU", lt_jitter=-1, cost_mult=0.55)

    for fac in ["FAC-LAP-01", "FAC-LAP-02"]:
        add_edge("T1-DISP-01", fac, "PANEL-OLED-15", lt_jitter=0, cost_mult=1.0)
        add_edge("T1-PCBA-01", fac, "MAINBOARD-SOC", lt_jitter=0, cost_mult=1.0)
        add_edge("T1-BATT-01", fac, "BATT-LI-70WH", lt_jitter=0, cost_mult=1.0)
        add_edge("T1-CHSS-01", fac, "CHASSIS-CNC-ALU", lt_jitter=0, cost_mult=1.0)

    for fac in ["FAC-LAP-01", "FAC-LAP-02"]:
        for wh in ["WH-REG-01", "WH-REG-02"]:
            add_edge(fac, wh, "LAPTOP-PRO-15", lt_jitter=0, cost_mult=1.0)

    all_item_codes = [c.item_code for c in components]
    state: dict[tuple[str, str], float] = {}
    reorder_points: dict[tuple[str, str], float] = {}
    demand_rates: dict[tuple[str, str], float] = {}
    scheduled_arrivals: dict[int, list[tuple[str, str, float]]] = {t: [] for t in range(num_ticks + 12)}

    for node in nodes:
        for item in all_item_codes:
            key = (node.node_id, item)
            spec = comp_by_code[item]
            state[key] = round(rng.uniform(75.0, 175.0), 2)
            reorder_points[key] = round(25.0 + spec.criticality * 35.0, 2)
            if node.node_type == NodeType.WAREHOUSE and item == "LAPTOP-PRO-15":
                demand_rates[key] = round(rng.uniform(14.0, 24.0), 2)
            elif node.node_type == NodeType.FACTORY and item != "LAPTOP-PRO-15":
                demand_rates[key] = round(rng.uniform(10.0, 20.0), 2)
            else:
                demand_rates[key] = round(rng.uniform(5.0, 14.0), 2)

    snapshots: list[InventorySnapshot] = []
    messages: list[SupplyMessage] = []
    msg_counter = 1
    supplier_ids = [n.node_id for n in nodes if n.node_type == NodeType.SUPPLIER]
    downstream_ids = [n.node_id for n in nodes if n.node_type in (NodeType.FACTORY, NodeType.WAREHOUSE)]

    for tick in range(num_ticks):
        active_disrupted_nodes: set[str] = set()

        for node in nodes:
            if rng.random() < disruption_prob_per_tick:
                active_disrupted_nodes.add(node.node_id)
                severity = round(rng.uniform(0.45, 0.95), 2)
                delay_extra = rng.randint(2, 6)
                messages.append(
                    SupplyMessage(
                        message_id=f"MSG-{msg_counter:05d}",
                        tick=tick,
                        sender_id=node.node_id,
                        recipient_id=rng.choice(downstream_ids),
                        event_type=EventType.DISRUPTION,
                        payload={
                            "domain": "laptop_manufacturing",
                            "disrupted_node": node.node_id,
                            "node_name": node.name,
                            "delay_extra_ticks": delay_extra,
                            "severity": severity,
                        },
                        signature=f"sig_valid_{node.node_id}_{tick}",
                        verified=True,
                        security_alert=None,
                    )
                )
                msg_counter += 1

        for edge in edges:
            src_disrupted = edge.source_id in active_disrupted_nodes
            delay_penalty = rng.randint(2, 5) if src_disrupted else 0
            arrival_tick = min(num_ticks + 10, tick + edge.base_lead_time_ticks + delay_penalty)
            qty = round(rng.uniform(8.0, min(28.0, edge.max_throughput * 0.45)), 2)

            if not src_disrupted or rng.random() < 0.30:
                scheduled_arrivals[arrival_tick].append((edge.target_id, edge.item_code, qty))

            is_attack = (edge.source_id in supplier_ids) and (rng.random() < attack_prob_per_tick)
            attack_type = (
                rng.choice(["tampered_payload", "invalid_signature", "spoofed_sender"])
                if is_attack
                else None
            )

            messages.append(
                SupplyMessage(
                    message_id=f"MSG-{msg_counter:05d}",
                    tick=tick,
                    sender_id="SUP-SPOOF-EXT" if attack_type == "spoofed_sender" else edge.source_id,
                    recipient_id=edge.target_id,
                    event_type=EventType.SHIPMENT_UPDATE,
                    payload={
                        "domain": "laptop_manufacturing",
                        "edge_id": edge.edge_id,
                        "item_code": edge.item_code,
                        "qty": round(qty * (3.0 if attack_type == "tampered_payload" else 1.0), 2),
                        "eta_tick": arrival_tick,
                    },
                    signature="sig_invalid_tampered" if attack_type == "invalid_signature" else f"sig_{edge.source_id}_{tick}",
                    verified=not is_attack,
                    security_alert=attack_type,
                )
            )
            msg_counter += 1

        arrivals_this_tick: dict[tuple[str, str], float] = {}
        for dst_id, item_code, qty in scheduled_arrivals.get(tick, []):
            arrivals_this_tick[(dst_id, item_code)] = arrivals_this_tick.get((dst_id, item_code), 0.0) + qty

        incoming_next_tick: dict[tuple[str, str], float] = {}
        for future_t in range(tick + 1, min(num_ticks + 11, tick + 4)):
            for dst_id, item_code, qty in scheduled_arrivals.get(future_t, []):
                incoming_next_tick[(dst_id, item_code)] = incoming_next_tick.get((dst_id, item_code), 0.0) + qty

        for node in nodes:
            for item in all_item_codes:
                key = (node.node_id, item)
                arrived = arrivals_this_tick.get(key, 0.0)
                demand = max(1.0, demand_rates[key] + rng.uniform(-2.5, 4.0))
                if node.node_id in active_disrupted_nodes and node.node_type == NodeType.SUPPLIER:
                    demand *= 0.45

                available = state[key] + arrived
                stockout = max(0.0, demand - available)
                new_on_hand = max(0.0, available - demand)
                state[key] = new_on_hand

                snapshots.append(
                    InventorySnapshot(
                        tick=tick,
                        node_id=node.node_id,
                        item_code=item,
                        on_hand=round(new_on_hand, 2),
                        incoming_qty=round(incoming_next_tick.get(key, 0.0), 2),
                        reorder_point=reorder_points[key],
                        demand_rate=round(demand, 2),
                        stockout_qty=round(stockout, 2),
                    )
                )

    return GeneratedDataset(
        domain="laptop_manufacturing",
        components=components,
        nodes=nodes,
        edges=edges,
        snapshots=snapshots,
        messages=messages,
    )
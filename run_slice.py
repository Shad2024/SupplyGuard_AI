from pathlib import Path

from supplyguard.data.generator import generate_laptop_supply_chain
from supplyguard.db import SupplyChainDB


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "artifacts" / "supplyguard_laptop.db"

    dataset = generate_laptop_supply_chain(seed=42, num_ticks=30)
    db = SupplyChainDB(db_path)
    db.seed_dataset(dataset, reset=True)

    components = db.fetch_components()
    nodes = db.fetch_nodes()
    edges = db.fetch_edges()
    snapshots = db.fetch_snapshots()
    unverified = db.fetch_messages(only_unverified=True)
    stockouts = [s for s in snapshots if s.stockout_qty > 0]

    print(f"Domain: {dataset.domain}")
    print(f"Database created at: {db_path}")
    print(f"Components ({len(components)}): {[c.item_code for c in components]}")
    print(f"Nodes: {len(nodes)} | Edges: {len(edges)}")
    print(f"Inventory snapshots: {len(snapshots)} | Stockout records: {len(stockouts)}")
    print(f"Unverified/attack messages: {len(unverified)}")

    db.close()


if __name__ == "__main__":
    main()
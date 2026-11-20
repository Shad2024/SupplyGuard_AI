# SupplyGuard AI — Deep Learning Module

Standalone deep-learning component for the Laptop Manufacturing supply-chain dataset.

## Scope

This module predicts supply-chain disruption risk over a graph of suppliers, factories, and warehouses. It is designed to provide a stable output contract for the SupplyGuard recommender, SimPy recovery simulator, and dashboard.

The implementation is contained in one main Python file:

- `supplyguard/ml/supplyguard_deep_learning.py`

The database is **not duplicated** in this submission package. The module expects the project database already present in the repository at `artifacts/supplyguard_laptop.db` (or an equivalent SQLite path passed with `--db`).

## Model architecture

**GRU + two-layer edge-aware Graph Neural Network (GNN)**

1. A GRU encodes the recent time window for each node.
2. Node and edge features are projected into a shared hidden representation.
3. Two directed edge-aware message-passing layers propagate information through the supply-chain graph.
4. Prediction heads produce node-level risk and stockout outputs, plus edge-level risk.
5. Stockout quantity is estimated as a graph-informed correction to a transparent demand/on-hand/incoming-inventory baseline.

PyTorch Geometric is optional. If it is not installed, the module uses its native PyTorch `index_add_` message-passing fallback.

## Prediction contract

All records are keyed by simulation tick and entity ID.

### Node predictions — `(tick, node_id)`

| Field | Meaning | Valid range |
|---|---|---|
| `risk_score` | Predicted node disruption/cascade risk | `[0.0, 1.0]` |
| `stockout_prob` | Probability of at least one stockout during the next forecast horizon | `[0.0, 1.0]` |
| `predicted_stockout_qty` | Cumulative predicted shortfall across item codes during the forecast horizon | `>= 0.0` |

### Edge predictions — `(tick, edge_id)`

| Field | Meaning | Valid range |
|---|---|---|
| `edge_risk_score` | Predicted risk associated with the supply-chain connection | `[0.0, 1.0]` |

The default forecast horizon is 3 ticks and the default historical lookback is 3 ticks.

## Dataset inputs

The SQLite loader reads these tables, preferring the `dataset_`-prefixed name and falling back to the unprefixed name:

- `dataset_nodes` / `nodes`
- `dataset_edges` / `edges`
- `dataset_components` / `components`
- `dataset_inventory_snapshots` / `inventory_snapshots`
- `dataset_supply_messages` / `supply_messages`

The database is opened in read-only mode. The model does not modify the source database.

Historical security-message features use only messages observed at or before the prediction tick. Static final compromise flags are not treated as historical per-tick events because the source dataset does not contain a time-resolved compromise history.

## Installation (Windows PowerShell)

Run commands from the **repository root** — the folder containing both `artifacts` and `supplyguard`.

```powershell
python -m pip install -r requirements-ml.txt
```

If PyTorch installation needs an explicit CPU-only wheel, install it with:

```powershell
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Python 3.11 or newer is recommended. A CUDA GPU is not required.

## Train the model

From the repository root:

```powershell
$db = (Resolve-Path ".\artifacts\supplyguard_laptop.db").Path
python -m supplyguard.ml.supplyguard_deep_learning --db $db
```

Default training settings:

- Hidden dimension: 64
- Maximum epochs: 180
- Early-stopping patience: 30 epochs
- Learning rate: 0.002
- Historical lookback: 3 ticks
- Forecast horizon: 3 ticks
- Random seed: 42

The run uses chronological train/validation/test partitions. Scalers are fitted on training ticks only.

To change the horizon or training settings, use the CLI options:

```powershell
python -m supplyguard.ml.supplyguard_deep_learning --help
```

## Run predictions using an existing checkpoint

After training, run:

```powershell
$db = (Resolve-Path ".\artifacts\supplyguard_laptop.db").Path
python -m supplyguard.ml.supplyguard_deep_learning --mode predict --db $db
```

Prediction mode uses `supplyguard/ml/artifacts/ml/gnn_checkpoint.pt` by default and regenerates the prediction files.

## Generated files

By default, files are written to `supplyguard/ml/artifacts/ml/`:

| File | Purpose |
|---|---|
| `gnn_checkpoint.pt` | Trained model checkpoint and model configuration |
| `latest_predictions.json` | Complete node and edge predictions in the shared JSON contract |
| `prediction_index.json` | Nested JSON lookup index grouped by tick and entity ID |
| `node_predictions.csv` | Flat node prediction table |
| `edge_predictions.csv` | Flat edge prediction table |
| `metrics.json` | Train, validation, and test metrics plus dataset fingerprint and limitations |
| `training_history.csv` | Per-epoch training and validation losses |
| `scalers.json` | Feature-scaling parameters fitted on training data |
| `feature_schema.json` | Feature names and prediction-horizon metadata |

## Python integration

The module exposes `load_prediction_index`, which returns two dictionaries keyed by `(tick, entity_id)`:

```python
from supplyguard.ml.supplyguard_deep_learning import load_prediction_index

node_predictions, edge_predictions = load_prediction_index(
    "supplyguard/ml/artifacts/ml"
)

node_result = node_predictions[(22, "NODE_ID")]
edge_result = edge_predictions[(22, "EDGE_ID")]

risk = node_result["risk_score"]
stockout_probability = node_result["stockout_prob"]
shortfall = node_result["predicted_stockout_qty"]
link_risk = edge_result["edge_risk_score"]
```

Replace `NODE_ID` and `EDGE_ID` with IDs that exist in the dataset.

## Reference run on the supplied Laptop Manufacturing dataset

The reference database contains 30 simulation ticks, 12 nodes, 16 directed edges, 5 components, 1,800 inventory snapshots, and 497 supply messages. Its SHA-256 fingerprint is recorded in `metrics.json`.

The reference run completed 34 epochs and selected epoch 4 as the best validation checkpoint. It generated 360 node records and 480 edge records.

| Test metric | Result |
|---|---:|
| Node risk ROC-AUC | `0.8869` |
| Stockout probability ROC-AUC | `null` — not defined for this test split |
| Stockout quantity MAE | `12.3624` units |
| Edge risk ROC-AUC | `0.6724` |

**Important evaluation limitations:** all 60 stockout test labels are positive, so ROC-AUC cannot be computed for stockout probability. The reported 100% threshold accuracy for that split is not evidence of discrimination between positive and negative classes. The stockout quantity MAE improves on the physics baseline only slightly (approximately 0.35%). The dataset is a small synthetic graph, so these metrics are exploratory and are not production-performance guarantees. Evaluate on a larger dataset with both stockout and non-stockout cases before making research or operational claims.

## Output validation

A successful run should produce 360 node records and 480 edge records for the supplied dataset. The included validator checks required fields, numeric ranges, unique `(tick, ID)` keys, and JSON/CSV record-count consistency. Run it from the repository root:

```powershell
python -m supplyguard.ml.validate_predictions
```

The output fields, ranges, and unique `(tick, ID)` keys should be validated before downstream integration.

# Pension-Lens
# PensionLens

GNN-based classification of pension fund structural risk using Form 5500 public filings.

---

## Pipeline Overview

| File | What It Does |
|------|-------------|
| `pensionlens_ingest.py` | Loads Form 5500 + Schedule C CSVs, extracts nodes and edges |
| `pensionlens_fuzzy.py` | Fuzzy name matching on asset managers at 85% threshold, assigns parent group IDs |
| `pensionlens_neo4j.py` | Loads all nodes and edges into Neo4j |
| `pensionlens_gds.py` | First GDS run (deprecated, replaced by gds2) |
| `pensionlens_gds2.py` | Final GDS run: Louvain, Betweenness, PageRank, community concentration |
| `pensionlens_labels.py` | Generates binary labels (0 = risky, 1 = healthy) from structural signals |
| `pensionlens_gnn.py` | HeteroGNN model, layer sweep (1-4), ablations (structural only, homogeneous) |

---

## Node Files

| File | Description |
|------|-------------|
| `nodes_pension_funds.csv` | Pension fund nodes with EIN, plan name, fund type, participants |
| `nodes_asset_managers.csv` | Asset manager nodes with EIN, name, fees, client count |
| `nodes_asset_managers_enriched.csv` | Managers with fuzzy group IDs and same parent flags |

## Edge Files

| File | Description |
|------|-------------|
| `edges_fund_to_manager.csv` | Fund → Manager edges with fee paid as weight |
| `edges_fund_to_fund.csv` | Fund → Fund edges where funds share the same manager |

---

## GDS2 Output Files (inside `gds2_output/` folder)

| File | Description |
|------|-------------|
| `gds_fund_scores.csv` | Per-fund GDS scores: community ID, betweenness, community concentration, same parent fee ratio |
| `gds_manager_scores.csv` | Per-manager GDS scores: PageRank, community ID, parent group info |
| `pensionlens_labeled.csv` | **Main training dataset**: 415 labeled funds with all features and binary labels |
| `pensionlens_labeled_full.csv` | All 4,103 funds including unlabeled ones (label = -1) |
| `pensionlens_results.csv` | Final model results: F1, AUC, Precision, Recall for all 3 model variants |
| `pensionlens_layer_sweep.csv` | Layer sweep results across 1-4 GNN layers |
| `pensionlens_model.pt` | Saved PyTorch model weights (Full HeteroGNN, best checkpoint) |

---

## For the Random Forest Baseline

You only need these files:

1. `pensionlens_labeled.csv` (from `gds2_output/`)

That's it. All features and labels are already computed inside this file.

Feature columns to use as input X:

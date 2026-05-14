# PensionLens

**PensionLens** is a graph-augmented machine learning system for detecting governance risk in U.S. pension funds. It ingests public Form 5500 regulatory filings, builds a heterogeneous knowledge graph in Neo4j, computes graph-theoretic risk signals via Neo4j Graph Data Science (GDS), and trains both a tabular baseline and a Heterogeneous Graph Neural Network (HeteroGNN) to classify pension funds as *risky* or *healthy*.

This repository was developed as the IDL Final Project.

---

## Table of Contents

- [Background](#background)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Pipeline Stages](#pipeline-stages)
  - [1. Data Ingestion](#1-data-ingestion-pensionlens_ingestpy)
  - [2. Fuzzy Name Matching](#2-fuzzy-name-matching-pensionlens_fuzzypy)
  - [3. Neo4j Graph Ingestion](#3-neo4j-graph-ingestion-pensionlens_neo4jpy)
  - [4. GDS Round 1](#4-gds-round-1-pensionlens_gdspy)
  - [5. GDS Round 2](#5-gds-round-2-pensionlens_gds2py)
  - [6. Label Generation](#6-label-generation-pensionlens_labelspy)
  - [7. Tabular Baseline](#7-tabular-baseline-pensionlens_baselinepy)
  - [8. HeteroGNN Model](#8-heterognn-model-pensionlens_gnnpy)
- [Graph Schema](#graph-schema)
- [Feature Engineering](#feature-engineering)
  - [Tabular Features (Baseline)](#tabular-features-baseline)
  - [Graph Features (GNN Only)](#graph-features-gnn-only)
- [Labeling Methodology](#labeling-methodology)
- [Models](#models)
  - [Tabular Baseline](#tabular-baseline)
  - [HeteroGNN](#heterognn)
- [Results](#results)
  - [Baseline Results](#baseline-results)
  - [GNN Results](#gnn-results)
  - [Layer Sweep](#layer-sweep)
- [Output Files](#output-files)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration Reference](#configuration-reference)

---

## Background

U.S. pension funds are required to file Form 5500 annually with the Department of Labor. These filings include Schedule C, which discloses fees paid to third-party service providers (i.e., asset managers). PensionLens exploits this public data to:

1. Map the network of relationships between pension funds and the asset managers they hire.
2. Compute structural risk signals from that network (concentration, circularity, community clustering).
3. Train classifiers that can flag pension funds exhibiting potentially problematic governance patterns.

The key insight is that tabular features alone (fees, participant counts) cannot capture **systemic** risk patterns — such as a fund being deeply embedded in a concentrated manager community, or all fees flowing to a single parent conglomerate. Graph topology is required to surface these signals.

---

## Architecture Overview

```
Form 5500 CSVs (raw)
        │
        ▼
pensionlens_ingest.py          ← builds nodes + edges CSVs
        │
        ▼
pensionlens_fuzzy.py           ← fuzzy name matching → parent group IDs
        │
        ▼
pensionlens_neo4j.py           ← loads graph into Neo4j
        │
        ▼
pensionlens_gds.py             ← GDS Round 1 (PageRank, Louvain, SCC, Betweenness)
        │
        ▼
pensionlens_gds2.py            ← GDS Round 2 (fixed projections, community concentration)
        │
        ▼
pensionlens_labels.py          ← computes risk scores → binary labels
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
pensionlens_baseline.py                        pensionlens_gnn.py
(tabular-only: LR, RF, XGBoost)               (HeteroGNN: fund + manager nodes)
        │                                                  │
        ▼                                                  ▼
baseline_outputs/                              pensionlens_results.csv
  ├─ baseline_results_summary.csv              pensionlens_layer_sweep.csv
  ├─ baseline_cv_scores.csv                    pensionlens_model.pt
  └─ plots/
```

---

## Repository Structure

```
Pension-Lens-gds2_outputs/
│
├── README.md
│
│── Python Scripts (Pipeline)
│   ├── pensionlens_ingest.py           # Stage 1: raw CSV → graph-ready CSVs
│   ├── pensionlens_fuzzy.py            # Stage 2: fuzzy name matching for managers
│   ├── pensionlens_neo4j.py            # Stage 3: load nodes/edges into Neo4j
│   ├── pensionlens_gds.py              # Stage 4: GDS Round 1 algorithms
│   ├── pensionlens_gds2.py             # Stage 5: GDS Round 2 (fixed projections)
│   ├── pensionlens_labels.py           # Stage 6: risk scoring + label generation
│   ├── pensionlens_baseline.py         # Stage 7: tabular baseline classifiers
│   └── pensionlens_gnn.py              # Stage 8: heterogeneous GNN
│
├── Graph Node / Edge CSVs
│   ├── nodes_pension_funds.csv         # ~4,103 pension fund nodes
│   ├── nodes_asset_managers.csv        # ~564 asset manager nodes
│   ├── nodes_asset_managers_enriched.csv  # managers + fuzzy parent group IDs
│   ├── edges_fund_to_manager.csv       # ~1,065 ALLOCATED_TO edges
│   └── edges_fund_to_fund.csv          # ~8,446 SHARES_MANAGER_WITH edges
│
├── GDS Output CSVs
│   ├── gds_fund_scores.csv             # per-fund GDS metrics (betweenness, community, etc.)
│   └── gds_manager_scores.csv          # per-manager GDS metrics (PageRank, community)
│
├── Labeled Datasets
│   ├── pensionlens_labeled.csv         # ~415 labeled funds (training set)
│   └── pensionlens_labeled_full.csv    # all ~4,103 funds (including unlabeled)
│
├── Model Results
│   ├── pensionlens_results.csv         # GNN model comparison summary
│   ├── pensionlens_layer_sweep.csv     # GNN performance across 1–4 layers
│   └── pensionlens_model.pt            # saved PyTorch weights (best HeteroGNN)
│
├── Baseline Results
│   ├── baseline_results_summary.csv    # test-set metrics (LR, RF, XGBoost)
│   ├── baseline_cv_scores.csv          # 5-fold CV scores per model per fold
│   ├── confusion_matrices.png
│   ├── cv_summary.png
│   ├── feature_importance.png
│   ├── loss_curves.png
│   ├── roc_curves.png
│   └── baseline_outputs/
│       ├── baseline_results_summary.csv
│       ├── baseline_cv_scores.csv
│       └── plots/
│           ├── confusion_matrices.png
│           ├── cv_summary.png
│           ├── feature_importance.png
│           ├── loss_curves.png
│           └── roc_curves.png
```

---

## Data Sources

| Source | Description |
|--------|-------------|
| **Form 5500** (`f_5500_2025_latest.csv`) | Annual pension fund filings. Provides plan name, EIN, sponsor, fund type, participant count, state. |
| **Schedule C Part 1 Item 2** (`F_SCH_C_PART1_ITEM2_2025_latest.csv`) | Fee disclosures. Links each pension fund to its service providers, including asset managers and fees paid. |

Both files are sourced from the U.S. Department of Labor's public EFAST2 database and are not included in this repository due to size. They must be downloaded separately and placed at the `BASE_PATH` configured in each script.

---

## Pipeline Stages

### 1. Data Ingestion (`pensionlens_ingest.py`)

Reads raw Form 5500 and Schedule C CSVs and produces four clean CSV files for graph construction.

**Steps:**
- Loads Form 5500 and Schedule C with `latin1` encoding.
- Extracts pension fund node attributes: EIN, plan name, sponsor, fund type, participant count, state, admin.
- Maps numeric `TYPE_PLAN_ENTITY_CD` to readable labels: `Single Employer`, `Multi Employer`, `Multiple Employer`, `DFE`.
- Extracts asset manager nodes by aggregating Schedule C entries: total fees received and number of pension fund clients.
- Builds directed `Fund → Manager` edges by joining Schedule C to Form 5500 via `ACK_ID`. Drops self-loops.
- Builds `Fund ↔ Fund` edges by joining the fund-manager edge table with itself on `manager_ein`, counting how many managers any two funds share.

**Outputs:**

| File | Description |
|------|-------------|
| `nodes_pension_funds.csv` | One row per unique fund EIN |
| `nodes_asset_managers.csv` | One row per unique manager EIN |
| `edges_fund_to_manager.csv` | One row per (fund, manager) relationship |
| `edges_fund_to_fund.csv` | One row per (fund_a, fund_b) pair sharing ≥1 manager |

---

### 2. Fuzzy Name Matching (`pensionlens_fuzzy.py`)

Identifies asset managers that are part of the same corporate family even when named slightly differently across filings (e.g., "BlackRock" vs "Blackrock Advisors LLC").

**Steps:**
- Strips legal suffixes (`LLC`, `INC`, `CORP`, `MANAGEMENT`, etc.) and non-alphanumeric characters from manager names.
- Filters out apparent individual person names (≤3 words with no institutional keyword) to remove individual trustees miscoded as managers.
- Groups managers using `rapidfuzz` `token_sort_ratio` fuzzy matching at an 85% similarity threshold. Short names (≤5 characters) use a stricter 95% threshold to prevent false groupings (e.g., UBS with ADP).
- Assigns a `parent_group_id` to each manager and computes `group_size` and a `same_parent_flag` (1 if the manager shares a parent group with any other manager).

**Output:** `nodes_asset_managers_enriched.csv`

---

### 3. Neo4j Graph Ingestion (`pensionlens_neo4j.py`)

Loads all nodes and edges into a local Neo4j instance using the Bolt driver, in batches of 500 records.

**Graph schema loaded:**

- Node label `:PensionFund` — properties: `ein`, `plan_name`, `sponsor_name`, `fund_type`, `num_participants`, `admin_name`, `state`, `label`
- Node label `:AssetManager` — properties: `ein`, `name`, `total_fees`, `num_clients`
- Relationship `(:PensionFund)-[:ALLOCATED_TO {fee_paid}]->(:AssetManager)`
- Relationship `(:PensionFund)-[:SHARES_MANAGER_WITH {shared_managers}]->(:PensionFund)`

**Verification query** (run in Neo4j Browser):
```cypher
MATCH (n) RETURN labels(n) AS label, count(n) AS count;
```

---

### 4. GDS Round 1 (`pensionlens_gds.py`)

First pass of Neo4j Graph Data Science algorithms. Projects a heterogeneous graph and writes results back to nodes.

**Algorithms run:**

| Algorithm | Target Nodes | Output Property |
|-----------|-------------|-----------------|
| **PageRank** | `AssetManager` | `pagerank_score` |
| **Louvain Community Detection** | `PensionFund`, `AssetManager` | `community_id` |
| **Strongly Connected Components (SCC)** | `PensionFund`, `AssetManager` | `scc_id`, `in_circular_pattern` |
| **Betweenness Centrality** | `AssetManager` | `betweenness_score` |
| **Node Similarity** | `PensionFund` | writes `SIMILAR_PORTFOLIO` edges |

The `in_circular_pattern` flag is set to `1` for any node inside an SCC with more than one member (i.e., part of a directed cycle in the allocation graph).

**Output CSVs:** `gds_fund_scores.csv`, `gds_manager_scores.csv`

---

### 5. GDS Round 2 (`pensionlens_gds2.py`)

Refined second pass that corrects projection issues from Round 1 and adds community concentration metrics. Uses separate graph projections for fund-only and fund+manager analyses.

**Steps:**
1. Projects a **fund-only** graph using `SHARES_MANAGER_WITH` (undirected, weighted) — runs Louvain and Betweenness Centrality on funds.
2. Re-projects with `PensionFund + AssetManager` and `ALLOCATED_TO` (reversed) — runs fee-weighted PageRank on managers.
3. Computes **community concentration** per fund: `1 / (number of distinct manager communities)`. A fund whose managers all belong to the same Louvain community scores 1.0 (maximally concentrated).
4. Computes **same-parent fee ratio** per fund: the fraction of total fees allocated to managers flagged with `same_parent_flag = 1`.

**Output CSVs:** `gds2_outputs/gds_fund_scores.csv`, `gds2_outputs/gds_manager_scores.csv`

---

### 6. Label Generation (`pensionlens_labels.py`)

Constructs the binary supervised training labels using a rule-based risk scoring system applied to each fund's structural and graph features.

**Features computed:**
- `num_managers` — number of distinct managers hired
- `total_fees` — total compensation paid
- `top_manager_concentration` — share of total fees paid to single top manager
- `fee_percentile` — fund's total fee rank among all funds
- `total_shared_managers` — sum of shared manager counts in fund-fund edges
- GDS features from Round 2: `betweenness_score`, `community_concentration`, `same_parent_fee_ratio`, `in_circular_pattern`

**Risk scoring (only applied to funds with manager data):**

| Signal | Condition | Points |
|--------|-----------|--------|
| High single-manager concentration | `top_manager_concentration > 0.70` | +2 |
| Low manager diversification | `num_managers <= 2` | +1 |
| High shared manager overlap | `total_shared_managers > 10` | +1 |
| Top-decile fees | `fee_percentile > 0.90` | +1 |
| High community concentration | `community_concentration >= 0.80` | +1 |
| High same-parent fee ratio | `same_parent_fee_ratio > 0.50` | +2 |

**Label assignment:**
- `risk_score >= 4` → label `0` (Risky)
- `risk_score < 4` → label `1` (Healthy)
- No manager data → label `-1` (Excluded from training)

**Outputs:** `pensionlens_labeled.csv` (~415 labeled funds), `pensionlens_labeled_full.csv` (~4,103 all funds)

---

### 7. Tabular Baseline (`pensionlens_baseline.py`)

A strictly **non-topological** baseline — no graph structure or GDS algorithm outputs are used. This provides a clean ablation: what can be learned from Form 5500 tabular data alone?

**Feature set (11 features after engineering):**

*Original tabular:*
- `num_participants`, `num_managers`, `total_fees`, `top_manager_concentration`, `fee_percentile`, `same_parent_fee_ratio`

*Engineered:*
- `log1p_total_fees`, `log1p_num_participants`, `log1p_fee_per_participant`
- `is_single_manager` (binary flag: `num_managers == 1`)
- `is_concentrated` (binary flag: `top_manager_concentration > 0.70`)

*Categorical encoding:*
- `fund_type` one-hot encoded → `ftype_*` dummy columns (drop-first)

**Explicitly blocked** (topology guard enforced at load time):
- `betweenness_score`, `community_concentration`, `total_shared_managers`, `in_circular_pattern`

**Models trained:**
- Logistic Regression (RobustScaler + L2, `C=0.1`, balanced class weights)
- Random Forest (500 trees, `min_samples_leaf=2`, balanced subsampling)
- XGBoost (300 rounds, early stopping at 30, weighted cross-entropy)

**Evaluation:** 5-fold stratified cross-validation + held-out test set (20%). Outputs confusion matrices, ROC curves, loss curves, feature importance plots, and CV summary box plots.

**CLI usage:**
```bash
python pensionlens_baseline.py --data_dir /path/to/gds2_outputs --output_dir ./baseline_outputs
```

---

### 8. HeteroGNN Model (`pensionlens_gnn.py`)

A **Heterogeneous Graph Neural Network** that jointly embeds pension fund nodes and asset manager nodes using message passing over the `ALLOCATED_TO` bipartite graph.

**Node features:**

| Node Type | Features |
|-----------|----------|
| Fund | `num_managers`, `top_manager_concentration`, `fee_percentile`, `total_shared_managers`, `community_concentration`, `same_parent_fee_ratio`, `betweenness_score`, `in_circular_pattern` |
| Manager | `num_clients`, `total_fees`, `same_parent_flag`, `group_size`, `pagerank_score` |

All features are `StandardScaler` normalized.

**Architecture (`PensionLensGNN`):**
- Input projection layers map fund (8-dim) and manager (5-dim) features to a shared hidden dimension (default: 64).
- `N` stacked `HeteroConv` layers using `SAGEConv` for both `fund→manager` and `manager→fund` message directions.
- Classifier head: `Linear(64 → 32) → ReLU → Dropout(0.3) → Linear(32 → 2)`.
- The graph is made undirected via `ToUndirected()` transform (adds reverse edges).

**Training:**
- Adam optimizer, `lr=0.01`, weight decay `5e-4`.
- Class-weighted cross-entropy loss to handle imbalance.
- 200 epochs for layer sweep, 300 epochs for final model.
- Best validation F1 checkpoint saved.
- Split: 68% train / 12% val / 20% test (stratified).

**Ablation variants:**

| Mode | Description |
|------|-------------|
| `full` | Node features + graph message passing |
| `structural` | Graph only (node features zeroed out) |
| `homogeneous` | Flattened node types (baseline comparison) |

**Layer sweep:** 1–4 layers evaluated; best layer count selected by test F1.

---

## Graph Schema

```
(:PensionFund {ein, plan_name, sponsor_name, fund_type, num_participants,
               admin_name, state, label,
               community_id, betweenness_score, community_concentration,
               same_parent_fee_ratio, in_circular_pattern})

    -[:ALLOCATED_TO {fee_paid}]->

(:AssetManager {ein, name, total_fees, num_clients,
                parent_group_id, group_size, same_parent_flag,
                pagerank_score, betweenness_score, community_id})

(:PensionFund)-[:SHARES_MANAGER_WITH {shared_managers}]->(:PensionFund)
(:PensionFund)-[:SIMILAR_PORTFOLIO {similarity_score}]->(:PensionFund)
```

---

## Feature Engineering

### Tabular Features (Baseline)

| Feature | Source | Description |
|---------|--------|-------------|
| `num_participants` | Form 5500 | Active plan participants |
| `num_managers` | Schedule C | Count of distinct asset managers hired |
| `total_fees` | Schedule C | Total fees paid to all managers |
| `top_manager_concentration` | Schedule C | Fraction of fees paid to the single largest manager (0–1) |
| `fee_percentile` | Schedule C | Fee rank percentile among all funds |
| `same_parent_fee_ratio` | Fuzzy matching | Fraction of fees paid to managers in same corporate family |
| `log1p_total_fees` | Engineered | Log-scale total fees |
| `log1p_num_participants` | Engineered | Log-scale participant count |
| `fee_per_participant` | Engineered | Total fees ÷ participants |
| `is_single_manager` | Engineered | Binary: `num_managers == 1` |
| `is_concentrated` | Engineered | Binary: `top_manager_concentration > 0.70` |
| `ftype_*` | Engineered | One-hot encoded fund type |

### Graph Features (GNN Only)

| Feature | Source | Description |
|---------|--------|-------------|
| `betweenness_score` | GDS (fund-only graph) | Betweenness centrality of fund in shared-manager network |
| `community_concentration` | GDS (Louvain) | 1 / (distinct manager communities). 1.0 = all managers in one community |
| `total_shared_managers` | Edge aggregation | Total shared manager count from fund-fund edges |
| `in_circular_pattern` | GDS (SCC) | Binary: fund is part of a directed cycle in allocation graph |
| `pagerank_score` | GDS (manager) | Manager's PageRank in fee-weighted graph (manager features) |

---

## Labeling Methodology

Labels are generated programmatically from a weighted risk scoring rule applied to structural and graph features. This approach is used because no ground-truth labels exist in the public Form 5500 dataset.

The labeling philosophy is:
- **Risky (0):** Funds showing multiple simultaneous signals of concentrated, opaque, or potentially conflicted fee arrangements.
- **Healthy (1):** Funds with diversified managers, transparent fee distributions, and no evidence of circular or community-concentrated patterns.

The threshold (`risk_score >= 4`) was calibrated to produce a reasonable class balance in the training set. The `risk_score` column is explicitly blocked from all model inputs to prevent leakage.

---

## Models

### Tabular Baseline

Three scikit-learn compatible classifiers, all strictly non-topological:

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | `C=0.1`, `solver=lbfgs`, `class_weight=balanced`, `RobustScaler` |
| Random Forest | 500 trees, `min_samples_leaf=2`, `class_weight=balanced_subsample`, OOB enabled |
| XGBoost | 300 rounds, early stopping 30, `pos_weight` adjusted for imbalance |

### HeteroGNN

Built with PyTorch Geometric:

- `HeteroConv` with `SAGEConv` operators for heterogeneous message passing
- `ToUndirected` transform adds reverse `manager → fund` edges automatically
- Class-weighted `CrossEntropyLoss` for imbalanced training
- Best-checkpoint saving based on validation F1

---

## Results

### Baseline Results

Test-set performance (20% held-out):

| Model | Accuracy | F1 (Macro) | ROC-AUC | Log Loss |
|-------|----------|-----------|---------|----------|
| Logistic Regression | 0.8675 | 0.8407 | 0.9716 | 0.2961 |
| Random Forest | 0.9036 | 0.8683 | 0.9808 | 0.1666 |
| **XGBoost** | **0.9639** | **0.9529** | **0.9885** | **0.1369** |

XGBoost is the strongest tabular baseline, achieving ~96.4% accuracy and ~98.9% ROC-AUC using only non-topological Form 5500 features.

### GNN Results

Test-set performance (20% held-out):

| Model Variant | F1 | AUC | Precision | Recall |
|---------------|-----|-----|-----------|--------|
| Full HeteroGNN (features + graph) | 0.8827 | 0.9555 | 0.8905 | 0.8795 |
| Structural Only (graph, no features) | 0.6388 | 0.5161 | 0.5580 | 0.7470 |
| **Homogeneous GNN** | **0.9050** | **0.9478** | **0.9076** | **0.9036** |

The ablation results show:
- Graph structure alone (structural only) is substantially weaker, confirming node features carry important signal.
- The homogeneous variant (flattened node types) achieves the best F1 in this run, though the full heterogeneous model achieves higher AUC (0.9555 vs. 0.9478), suggesting better probability calibration.

### Layer Sweep

| Layers | F1 | AUC |
|--------|----|-----|
| 1 | 0.9036 | 0.9585 |
| 2 | 0.9050 | 0.9508 |
| 3 | 0.9050 | 0.9570 |
| 4 | 0.9050 | 0.9754 |

F1 plateaus at 2 layers; deeper networks yield higher AUC (4-layer: 0.9754) suggesting larger receptive fields improve ranking quality even when accuracy is similar.

---

## Output Files

| File | Description |
|------|-------------|
| `nodes_pension_funds.csv` | Fund nodes with EIN, plan metadata, fund type |
| `nodes_asset_managers.csv` | Manager nodes with EIN, name, fees, client count |
| `nodes_asset_managers_enriched.csv` | Managers enriched with fuzzy parent group IDs |
| `edges_fund_to_manager.csv` | Fund-to-manager allocation edges with fee amounts |
| `edges_fund_to_fund.csv` | Fund-to-fund co-manager edges with shared manager counts |
| `gds_fund_scores.csv` | Per-fund GDS metrics (community, betweenness, fees, concentration) |
| `gds_manager_scores.csv` | Per-manager GDS metrics (PageRank, community, parent group) |
| `pensionlens_labeled.csv` | Labeled training dataset (~415 funds with manager data) |
| `pensionlens_labeled_full.csv` | All funds including unlabeled (label = -1) |
| `pensionlens_results.csv` | GNN model comparison: Full vs. Structural vs. Homogeneous |
| `pensionlens_layer_sweep.csv` | GNN performance across 1–4 message-passing layers |
| `pensionlens_model.pt` | Saved PyTorch state dict for best HeteroGNN |
| `baseline_results_summary.csv` | Tabular baseline test metrics (LR, RF, XGBoost) |
| `baseline_cv_scores.csv` | Per-fold CV metrics for 5-fold stratified cross-validation |

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- Neo4j Desktop or Neo4j AuraDB (for graph ingestion and GDS)
- Neo4j Graph Data Science plugin installed

### Python Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install rapidfuzz
pip install neo4j
pip install torch torch-geometric
```

For PyTorch Geometric, install the version matching your PyTorch and CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### Neo4j Setup

1. Install Neo4j Desktop.
2. Create a new database and install the **Graph Data Science** plugin.
3. Update `URI`, `USER`, and `PASSWORD` in `pensionlens_neo4j.py`, `pensionlens_gds.py`, and `pensionlens_gds2.py` to match your database credentials.

---

## Running the Pipeline

Run the scripts in order. Update `BASE_PATH` in each script to point to your local data directory.

```bash
# Stage 1: Ingest raw CSVs → graph-ready node/edge CSVs
python pensionlens_ingest.py

# Stage 2: Fuzzy name matching → enriched manager nodes
python pensionlens_fuzzy.py

# Stage 3: Load graph into Neo4j
python pensionlens_neo4j.py

# Stage 4: GDS Round 1 (PageRank, Louvain, SCC, Betweenness)
python pensionlens_gds.py

# Stage 5: GDS Round 2 (refined projections, community concentration)
python pensionlens_gds2.py

# Stage 6: Generate risk labels
python pensionlens_labels.py

# Stage 7: Train tabular baseline (outputs to ./baseline_outputs/)
python pensionlens_baseline.py --data_dir ./gds2_outputs --output_dir ./baseline_outputs

# Stage 8: Train HeteroGNN
python pensionlens_gnn.py
```

> **Note:** Stages 3–5 require a running Neo4j instance. Stages 7–8 are independent of Neo4j and operate on the CSV outputs only.

---

## Configuration Reference

### `pensionlens_baseline.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `TABULAR_FEATURES` | (6 features) | Input features — topology columns are blocked |
| `RANDOM_STATE` | `42` | Global random seed |
| `TEST_SIZE` | `0.20` | Held-out test fraction |
| `VAL_SIZE` | `0.15` | Validation fraction (of training set) for XGB |
| `CV_FOLDS` | `5` | Stratified K-fold splits |
| `XGB_ROUNDS` | `300` | Max XGBoost boosting rounds |
| `EARLY_STOP_RND` | `30` | XGBoost early stopping patience |
| `FEE_CLIP_LOW/HIGH` | `0.01 / 0.99` | Percentile clipping thresholds for `total_fees` |

### `pensionlens_gnn.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden` | `64` | Hidden dimension for all GNN layers |
| `num_layers` | Best from sweep | Number of HeteroConv message-passing layers |
| `epochs` | `300` | Training epochs for final model |
| `lr` | `0.01` | Adam learning rate |
| `weight_decay` | `5e-4` | L2 regularisation |
| `dropout` | `0.3` | Dropout rate in classifier head |

### `pensionlens_labels.py` (Risk Score Thresholds)

| Signal | Threshold | Weight |
|--------|-----------|--------|
| Top manager concentration | > 0.70 | +2 |
| Number of managers | ≤ 2 | +1 |
| Shared manager overlap | > 10 | +1 |
| Fee percentile | > 0.90 | +1 |
| Community concentration | ≥ 0.80 | +1 |
| Same-parent fee ratio | > 0.50 | +2 |
| **Risky threshold** | **≥ 4** | — |

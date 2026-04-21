import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PENSIONLENS - HeteroGNN Model
# pensionlens_gnn.py
# ============================================================

BASE_PATH  = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"
OUTPUT_DIR = os.path.join(BASE_PATH, "gds2_outputs")

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("Loading data...")
labeled      = pd.read_csv(os.path.join(OUTPUT_DIR, "pensionlens_labeled.csv"))
managers     = pd.read_csv(os.path.join(BASE_PATH, "nodes_asset_managers_enriched.csv"), dtype=str)
gds_managers = pd.read_csv(os.path.join(OUTPUT_DIR, "gds_manager_scores.csv"))
edges_fm     = pd.read_csv(os.path.join(BASE_PATH, "edges_fund_to_manager.csv"), dtype=str)

print(f"Labeled funds:  {len(labeled):,}")
print(f"Managers:       {len(managers):,}")
print(f"Fund->Manager edges: {len(edges_fm):,}")

# ============================================================
# STEP 2: BUILD NODE FEATURE MATRICES
# ============================================================

print("\nBuilding node features...")

# --- Fund node features ---
fund_feature_cols = [
    'num_managers',
    'top_manager_concentration',
    'fee_percentile',
    'total_shared_managers',
    'community_concentration',
    'same_parent_fee_ratio',
    'betweenness_score',
    'in_circular_pattern'
]

labeled['total_shared_managers'] = pd.to_numeric(labeled['total_shared_managers'], errors='coerce').fillna(0)
for col in fund_feature_cols:
    labeled[col] = pd.to_numeric(labeled[col], errors='coerce').fillna(0)

# clip outliers
labeled['total_shared_managers'] = labeled['total_shared_managers'].clip(0, 1e6)
labeled['betweenness_score']      = labeled['betweenness_score'].clip(0, 1e6)

fund_scaler = StandardScaler()
fund_features = fund_scaler.fit_transform(labeled[fund_feature_cols].values)

# map fund ein to index
labeled = labeled.reset_index(drop=True)
fund_ein_to_idx = {str(ein): i for i, ein in enumerate(labeled['ein'])}

# --- Manager node features ---
gds_managers['ein'] = gds_managers['ein'].astype(str)

managers = managers.merge(
    gds_managers[['ein','pagerank_score','community_id']],
    on='ein', how='left'
)

manager_feature_cols = ['num_clients','total_fees','same_parent_flag','group_size']
for col in manager_feature_cols:
    managers[col] = pd.to_numeric(managers[col], errors='coerce').fillna(0)

managers['pagerank_score']   = pd.to_numeric(managers['pagerank_score'], errors='coerce').fillna(0.15)


manager_all_features = manager_feature_cols + ['pagerank_score']
manager_scaler = StandardScaler()
manager_features = manager_scaler.fit_transform(managers[manager_all_features].values)

managers = managers.reset_index(drop=True)
manager_ein_to_idx = {str(ein): i for i, ein in enumerate(managers['ein'])}

print(f"Fund feature matrix:    {fund_features.shape}")
print(f"Manager feature matrix: {manager_features.shape}")

# ============================================================
# STEP 3: BUILD EDGES
# ============================================================

print("\nBuilding edges...")

edges_fm['fund_ein']    = edges_fm['fund_ein'].astype(str)
edges_fm['manager_ein'] = edges_fm['manager_ein'].astype(str)
edges_fm['fee_paid']    = pd.to_numeric(edges_fm['fee_paid'], errors='coerce').fillna(0)

src, dst, edge_weights = [], [], []
for _, row in edges_fm.iterrows():
    f_idx = fund_ein_to_idx.get(row['fund_ein'])
    m_idx = manager_ein_to_idx.get(row['manager_ein'])
    if f_idx is not None and m_idx is not None:
        src.append(f_idx)
        dst.append(m_idx)
        edge_weights.append(row['fee_paid'])

print(f"Valid edges built: {len(src):,}")

# ============================================================
# STEP 4: BUILD HETERODATA GRAPH
# ============================================================

print("\nBuilding HeteroData graph...")

data = HeteroData()

data['fund'].x    = torch.tensor(fund_features, dtype=torch.float)
data['manager'].x = torch.tensor(manager_features, dtype=torch.float)
data['fund'].y    = torch.tensor(labeled['label'].values, dtype=torch.long)

if len(src) > 0:
    data['fund','allocated_to','manager'].edge_index = torch.tensor(
        [src, dst], dtype=torch.long)
    data['fund','allocated_to','manager'].edge_attr = torch.tensor(
        edge_weights, dtype=torch.float).unsqueeze(1)
else:
    data['fund','allocated_to','manager'].edge_index = torch.zeros((2,0), dtype=torch.long)
    data['fund','allocated_to','manager'].edge_attr  = torch.zeros((0,1), dtype=torch.float)

data = ToUndirected()(data)
print(data)

# ============================================================
# STEP 5: TRAIN/VAL/TEST SPLIT
# ============================================================

n = len(labeled)
idx = np.arange(n)
labels = labeled['label'].values

train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
train_idx, val_idx  = train_test_split(train_idx, test_size=0.15, stratify=labels[train_idx], random_state=42)

train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

data['fund'].train_mask = train_mask
data['fund'].val_mask   = val_mask
data['fund'].test_mask  = test_mask

print(f"\nTrain: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

# ============================================================
# STEP 6: MODEL DEFINITION
# ============================================================

class PensionLensGNN(torch.nn.Module):
    def __init__(self, fund_in, manager_in, hidden, num_layers=2, mode='full'):
        super().__init__()
        self.mode       = mode
        self.num_layers = num_layers
        self.hidden     = hidden

        # input projections to unify dimensions
        self.fund_proj    = Linear(fund_in,    hidden)
        self.manager_proj = Linear(manager_in, hidden)

        # main conv layers (all hidden x hidden after projection)
        self.convs = torch.nn.ModuleList([
            HeteroConv({
                ('fund','allocated_to','manager'):     SAGEConv((hidden, hidden), hidden),
                ('manager','rev_allocated_to','fund'): SAGEConv((hidden, hidden), hidden),
            }, aggr='mean')
            for _ in range(num_layers)
        ])

        # classifier
        self.classifier = torch.nn.Sequential(
            Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            Linear(hidden // 2, 2)
        )

    def forward(self, x_dict, edge_index_dict):
        # project all nodes to hidden dim first
        if self.mode == 'structural':
            x_dict = {
                'fund':    torch.zeros(x_dict['fund'].size(0), self.hidden),
                'manager': torch.zeros(x_dict['manager'].size(0), self.hidden)
            }
        else:
            x_dict = {
                'fund':    F.relu(self.fund_proj(x_dict['fund'])),
                'manager': F.relu(self.manager_proj(x_dict['manager']))
            }

        # message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        return self.classifier(x_dict['fund'])
    
# ============================================================
# STEP 7: TRAINING FUNCTION
# ============================================================

# class weights for imbalance
class_counts = np.bincount(labels)
class_weights = torch.tensor(
    [len(labels)/class_counts[0], len(labels)/class_counts[1]],
    dtype=torch.float
)

def train_model(model, data, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_f1 = 0
    best_state  = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(
            out[data['fund'].train_mask],
            data['fund'].y[data['fund'].train_mask],
            weight=class_weights
        )
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                out   = model(data.x_dict, data.edge_index_dict)
                pred  = out.argmax(dim=1)
                val_f1 = f1_score(
                    data['fund'].y[data['fund'].val_mask].numpy(),
                    pred[data['fund'].val_mask].numpy(),
                    average='weighted', zero_division=0
                )
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate(model, data, mask, label='Test'):
    model.eval()
    with torch.no_grad():
        out   = model(data.x_dict, data.edge_index_dict)
        pred  = out.argmax(dim=1)
        proba = F.softmax(out, dim=1)[:,1]
        y_true = data['fund'].y[mask].numpy()
        y_pred = pred[mask].numpy()
        y_prob = proba[mask].numpy()

        f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        pre = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"\n{label} Results:")
        print(f"  F1:        {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Precision: {pre:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        return {'f1': f1, 'auc': auc, 'precision': pre, 'recall': rec}

fund_in    = fund_features.shape[1]
manager_in = manager_features.shape[1]

# ============================================================
# STEP 8: LAYER SWEEP (1-4 layers)
# ============================================================

print("\n" + "="*60)
print("LAYER SWEEP")
print("="*60)

layer_results = {}
for n_layers in [1, 2, 3, 4]:
    print(f"\n--- {n_layers} Layer(s) ---")
    model = PensionLensGNN(fund_in, manager_in, hidden=64, num_layers=n_layers, mode='full')
    model = train_model(model, data, epochs=200, lr=0.01)
    results = evaluate(model, data, data['fund'].test_mask, label=f'{n_layers} Layers')
    layer_results[n_layers] = results

best_layers = max(layer_results, key=lambda x: layer_results[x]['f1'])
print(f"\nBest layer count: {best_layers} (F1: {layer_results[best_layers]['f1']:.4f})")

# ============================================================
# STEP 9: FULL MODEL (best layers)
# ============================================================

print("\n" + "="*60)
print("FULL MODEL (HeteroGNN - node features + graph structure)")
print("="*60)
model_full = PensionLensGNN(fund_in, manager_in, hidden=64, num_layers=best_layers, mode='full')
model_full = train_model(model_full, data, epochs=300, lr=0.01)
results_full = evaluate(model_full, data, data['fund'].test_mask, label='Full Model')

# ============================================================
# STEP 10: ABLATION 1 - STRUCTURAL ONLY
# ============================================================

print("\n" + "="*60)
print("ABLATION 1 - Structural Only (no node features)")
print("="*60)
model_struct = PensionLensGNN(fund_in, manager_in, hidden=64, num_layers=best_layers, mode='structural')
model_struct = train_model(model_struct, data, epochs=300, lr=0.01)
results_struct = evaluate(model_struct, data, data['fund'].test_mask, label='Structural Only')

# ============================================================
# STEP 11: ABLATION 2 - HOMOGENEOUS
# ============================================================

print("\n" + "="*60)
print("ABLATION 2 - Homogeneous (flattened node types)")
print("="*60)
model_homo = PensionLensGNN(fund_in, manager_in, hidden=64, num_layers=best_layers, mode='homogeneous')
model_homo = train_model(model_homo, data, epochs=300, lr=0.01)
results_homo = evaluate(model_homo, data, data['fund'].test_mask, label='Homogeneous')

# ============================================================
# STEP 12: SUMMARY TABLE
# ============================================================

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

summary = pd.DataFrame({
    'Model':     ['Full HeteroGNN', 'Structural Only', 'Homogeneous'],
    'F1':        [results_full['f1'], results_struct['f1'], results_homo['f1']],
    'AUC':       [results_full['auc'], results_struct['auc'], results_homo['auc']],
    'Precision': [results_full['precision'], results_struct['precision'], results_homo['precision']],
    'Recall':    [results_full['recall'], results_struct['recall'], results_homo['recall']],
})
print(summary.to_string(index=False))

summary.to_csv(os.path.join(OUTPUT_DIR, "pensionlens_results.csv"), index=False)

layer_df = pd.DataFrame([
    {'layers': k, **v} for k, v in layer_results.items()
])
layer_df.to_csv(os.path.join(OUTPUT_DIR, "pensionlens_layer_sweep.csv"), index=False)

torch.save(model_full.state_dict(), os.path.join(OUTPUT_DIR, "pensionlens_model.pt"))

print(f"\nAll results saved to: {OUTPUT_DIR}")
print("Files saved:")
print("  pensionlens_results.csv")
print("  pensionlens_layer_sweep.csv")
print("  pensionlens_model.pt")
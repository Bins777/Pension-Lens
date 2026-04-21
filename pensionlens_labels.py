import pandas as pd
import numpy as np
import os

# ============================================================
# PENSIONLENS - Label Generation
# pensionlens_labels.py
# ============================================================

BASE_PATH  = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"
OUTPUT_DIR = os.path.join(BASE_PATH, "gds2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD ALL DATA
# ============================================================

print("Loading data...")
funds        = pd.read_csv(os.path.join(BASE_PATH, "nodes_pension_funds.csv"), dtype=str)
edges_fm     = pd.read_csv(os.path.join(BASE_PATH, "edges_fund_to_manager.csv"), dtype=str)
edges_ff     = pd.read_csv(os.path.join(BASE_PATH, "edges_fund_to_fund.csv"), dtype=str)
gds_funds    = pd.read_csv(os.path.join(OUTPUT_DIR, "gds_fund_scores.csv"))
gds_managers = pd.read_csv(os.path.join(OUTPUT_DIR, "gds_manager_scores.csv"))

print(f"Funds:        {len(funds):,}")
print(f"Fund->Manager edges: {len(edges_fm):,}")
print(f"Fund->Fund edges:    {len(edges_ff):,}")
print(f"GDS fund scores:     {len(gds_funds):,}")

# ============================================================
# STEP 2: COMPUTE STRUCTURAL FEATURES PER FUND
# ============================================================

print("\nComputing structural features...")

edges_fm['fee_paid'] = pd.to_numeric(edges_fm['fee_paid'], errors='coerce').fillna(0)
edges_fm['fund_ein'] = edges_fm['fund_ein'].astype(str)
edges_fm['manager_ein'] = edges_fm['manager_ein'].astype(str)

# feature 1: number of unique managers per fund
manager_count = edges_fm.groupby('fund_ein')['manager_ein'].nunique().reset_index()
manager_count.columns = ['ein', 'num_managers']

# feature 2: top manager fee concentration
total_fees = edges_fm.groupby('fund_ein')['fee_paid'].sum().reset_index()
total_fees.columns = ['ein', 'total_fees']

top_manager_fee = edges_fm.groupby(['fund_ein','manager_ein'])['fee_paid'].sum().reset_index()
top_manager_fee = top_manager_fee.sort_values('fee_paid', ascending=False)
top_manager_fee = top_manager_fee.groupby('fund_ein').first().reset_index()
top_manager_fee.columns = ['ein', 'top_manager_ein', 'top_manager_fee']

concentration = total_fees.merge(top_manager_fee, on='ein', how='left')
concentration['top_manager_concentration'] = (
    concentration['top_manager_fee'] / concentration['total_fees']
).clip(0, 1)

# feature 3: shared manager count per fund
edges_ff['fund_a'] = edges_ff['fund_a'].astype(str)
shared_count = edges_ff.groupby('fund_a')['shared_managers'].sum().reset_index()
shared_count.columns = ['ein', 'total_shared_managers']

# feature 4: fee percentile rank within peer funds
concentration['fee_percentile'] = concentration['total_fees'].rank(pct=True)

# ============================================================
# STEP 3: MERGE ALL FEATURES
# ============================================================

print("Merging features...")

funds['ein'] = funds['ein'].astype(str)
gds_funds['ein'] = gds_funds['ein'].astype(str)

df = funds[['ein','plan_name','fund_type','num_participants']].copy()
df = df.merge(manager_count,   on='ein', how='left')
df = df.merge(concentration[['ein','total_fees','top_manager_concentration','fee_percentile']], on='ein', how='left')
df = df.merge(shared_count,    on='ein', how='left')
df = df.merge(gds_funds[['ein','betweenness_score','community_concentration',
                           'same_parent_fee_ratio','in_circular_pattern']], on='ein', how='left')

# fill nulls
df['num_managers']              = df['num_managers'].fillna(0)
df['total_fees']                = df['total_fees'].fillna(0)
df['top_manager_concentration'] = df['top_manager_concentration'].fillna(0)
df['fee_percentile']            = df['fee_percentile'].fillna(0.5)
df['total_shared_managers']     = df['total_shared_managers'].fillna(0)
df['betweenness_score']         = df['betweenness_score'].fillna(0)
df['community_concentration']   = df['community_concentration'].fillna(1.0)
df['same_parent_fee_ratio']     = df['same_parent_fee_ratio'].clip(0, 1).fillna(0)
df['in_circular_pattern']       = df['in_circular_pattern'].fillna(0)

print(f"\nFeature matrix shape: {df.shape}")
print(df.describe())

# ============================================================
# STEP 4: SCORING AND LABELING
# ============================================================

print("\nComputing risk scores and labels...")

# force numeric types
df['total_shared_managers']     = pd.to_numeric(df['total_shared_managers'], errors='coerce').fillna(0)
df['num_managers']              = pd.to_numeric(df['num_managers'], errors='coerce').fillna(0)
df['top_manager_concentration'] = pd.to_numeric(df['top_manager_concentration'], errors='coerce').fillna(0)
df['fee_percentile']            = pd.to_numeric(df['fee_percentile'], errors='coerce').fillna(0.5)
df['community_concentration']   = pd.to_numeric(df['community_concentration'], errors='coerce').fillna(1.0)
df['same_parent_fee_ratio']     = pd.to_numeric(df['same_parent_fee_ratio'], errors='coerce').clip(0,1).fillna(0)
df['betweenness_score']         = pd.to_numeric(df['betweenness_score'], errors='coerce').fillna(0)

df['risk_score'] = 0
df['has_manager_data'] = (df['num_managers'] > 0).astype(int)

# only score funds that have actual manager data
has_data = df['has_manager_data'] == 1

# signal 1: single manager concentration > 70% (raised from 50%)
df.loc[has_data & (df['top_manager_concentration'] > 0.70), 'risk_score'] += 2

# signal 2: 2 or fewer managers (loosened from 1)
df.loc[has_data & (df['num_managers'] <= 2), 'risk_score'] += 1

# signal 3: high shared manager overlap
df.loc[has_data & (df['total_shared_managers'] > 10), 'risk_score'] += 1

# signal 4: fees in top 10%
df.loc[has_data & (df['fee_percentile'] > 0.90), 'risk_score'] += 1

# signal 5: community concentration high (loosened from 1.0 to 0.8)
df.loc[has_data & (df['community_concentration'] >= 0.80), 'risk_score'] += 1

# signal 6: same parent fee ratio > 50%
df.loc[has_data & (df['same_parent_fee_ratio'] > 0.50), 'risk_score'] += 2

# label: risky if score >= 4, healthy if below
df['label'] = -1
df.loc[has_data & (df['risk_score'] >= 4), 'label'] = 0
df.loc[has_data & (df['risk_score'] < 4),  'label'] = 1

# only keep labeled funds for training
df_labeled = df[df['label'] != -1].copy()

# ============================================================
# STEP 5: REPORT
# ============================================================

print(f"\nTotal funds: {len(df):,}")
print(f"Funds with manager data: {df['has_manager_data'].sum():,}")
print(f"Funds excluded (no manager data): {(df['label']==-1).sum():,}")

print(f"\n--- Label Distribution (training set only) ---")
print(df_labeled['label'].value_counts())
print(f"\nHealthy (1): {df_labeled['label'].sum():,} ({df_labeled['label'].mean()*100:.1f}%)")
print(f"Risky   (0): {(df_labeled['label']==0).sum():,} ({(df_labeled['label']==0).mean()*100:.1f}%)")

print(f"\n--- Risk Score Distribution ---")
print(df_labeled['risk_score'].value_counts().sort_index())

print(f"\n--- Feature Stats by Label ---")
feature_cols = ['num_managers','top_manager_concentration',
                'total_shared_managers','community_concentration',
                'same_parent_fee_ratio','betweenness_score']
print(df_labeled.groupby('label')[feature_cols].mean().round(3).to_string())

print(f"\n--- Sample Risky Funds ---")
risky = df_labeled[df_labeled['label']==0][['plan_name','num_managers',
                              'top_manager_concentration',
                              'community_concentration','risk_score']].head(10)
print(risky.to_string())

print(f"\n--- Sample Healthy Funds ---")
healthy = df_labeled[df_labeled['label']==1][['plan_name','num_managers',
                               'top_manager_concentration',
                               'community_concentration','risk_score']].head(10)
print(healthy.to_string())

print(f"\nClass balance ratio: {df_labeled['label'].mean():.2f} (ideal 0.3-0.7)")

# ============================================================
# STEP 6: EXPORT
# ============================================================

df.to_csv(os.path.join(OUTPUT_DIR, "pensionlens_labeled_full.csv"), index=False)
df_labeled.to_csv(os.path.join(OUTPUT_DIR, "pensionlens_labeled.csv"), index=False)
print(f"\nExported pensionlens_labeled.csv -> {len(df_labeled):,} labeled funds")
print(f"Exported pensionlens_labeled_full.csv -> {len(df):,} all funds")
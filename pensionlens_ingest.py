import pandas as pd
import os

BASE_PATH = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"

F5500_CSV = os.path.join(BASE_PATH, "F_5500_2025_Latest", "f_5500_2025_latest.csv")
SCHC_CSV  = os.path.join(BASE_PATH, "F_SCH_C_PART1_ITEM2_2025_Latest", "F_SCH_C_PART1_ITEM2_2025_latest.csv")

# ============================================================
# STEP 1: LOAD RAW DATA
# ============================================================

print("Loading data...")
f5500 = pd.read_csv(F5500_CSV, encoding='latin1', low_memory=False)
schc  = pd.read_csv(SCHC_CSV,  encoding='latin1', low_memory=False)
print(f"Form 5500: {f5500.shape[0]:,} rows")
print(f"Schedule C: {schc.shape[0]:,} rows")

# ============================================================
# STEP 2: CLEAN PENSION FUND NODES (from Form 5500)
# ============================================================

print("\nCleaning pension fund nodes...")

funds = f5500[[
    'ACK_ID',
    'SPONS_DFE_EIN',
    'PLAN_NAME',
    'SPONSOR_DFE_NAME',
    'TYPE_PLAN_ENTITY_CD',
    'TOT_ACTIVE_PARTCP_CNT',
    'ADMIN_NAME',
    'ADMIN_EIN',
    'SPONS_DFE_MAIL_US_STATE'
]].copy()

# Rename for clarity
funds.columns = [
    'ack_id',
    'ein',
    'plan_name',
    'sponsor_name',
    'fund_type_code',
    'num_participants',
    'admin_name',
    'admin_ein',
    'state'
]

# Drop rows with no EIN (can't be a node without an identifier)
funds = funds.dropna(subset=['ein'])

# Clean EIN to integer string
funds['ein'] = funds['ein'].astype(float).astype(int).astype(str)

# Map fund type code to readable label
type_map = {1: 'Single Employer', 2: 'Multi Employer',
            3: 'Multiple Employer', 4: 'DFE'}
funds['fund_type'] = funds['fund_type_code'].map(type_map).fillna('Unknown')

# Add placeholder label (-1 = unlabeled)
funds['label'] = -1

# Drop duplicates by EIN, keep first
funds = funds.drop_duplicates(subset='ein')

print(f"Pension Fund nodes: {len(funds):,}")
print(funds[['ein','plan_name','fund_type','num_participants']].head(5).to_string())

# ============================================================
# STEP 3: CLEAN ASSET MANAGER NODES (from Schedule C)
# ============================================================

print("\nCleaning asset manager nodes...")

managers = schc[[
    'PROVIDER_OTHER_NAME',
    'PROVIDER_OTHER_EIN',
    'PROVIDER_OTHER_DIRECT_COMP_AMT'
]].copy()

managers.columns = ['name', 'ein', 'fee_paid']

# Drop rows with no name or EIN
managers = managers.dropna(subset=['name', 'ein'])

# Clean EIN
managers['ein'] = managers['ein'].astype(float).astype(int).astype(str)

# Clean name
managers['name'] = managers['name'].str.strip().str.upper()

# Aggregate: one node per manager with total fees and client count
manager_nodes = managers.groupby(['ein', 'name']).agg(
    total_fees = ('fee_paid', 'sum'),
    num_clients = ('fee_paid', 'count')
).reset_index()

print(f"Asset Manager nodes: {len(manager_nodes):,}")
print(manager_nodes.head(5).to_string())

# ============================================================
# STEP 4: BUILD EDGES (Pension Fund -> Asset Manager)
# ============================================================

print("\nBuilding edges...")

# Merge Schedule C with Form 5500 via ACK_ID to get fund EIN
schc_merged = schc.merge(
    f5500[['ACK_ID', 'SPONS_DFE_EIN']],
    on='ACK_ID',
    how='left'
)

edges = schc_merged[[
    'SPONS_DFE_EIN',
    'PROVIDER_OTHER_EIN',
    'PROVIDER_OTHER_NAME',
    'PROVIDER_OTHER_DIRECT_COMP_AMT'
]].copy()

edges.columns = ['fund_ein', 'manager_ein', 'manager_name', 'fee_paid']

# Drop rows missing either EIN
edges = edges.dropna(subset=['fund_ein', 'manager_ein'])

# Clean EINs
edges['fund_ein']    = edges['fund_ein'].astype(float).astype(int).astype(str)
edges['manager_ein'] = edges['manager_ein'].astype(float).astype(int).astype(str)

# Drop self loops
edges = edges[edges['fund_ein'] != edges['manager_ein']]

print(f"Edges (Fund -> Manager): {len(edges):,}")
print(edges.head(5).to_string())

# ============================================================
# STEP 5: BUILD FUND-FUND EDGES (Shares Manager With)
# ============================================================

print("\nBuilding fund-fund edges (shared managers)...")

# Find funds that share the same manager
fund_manager = edges[['fund_ein', 'manager_ein']].drop_duplicates()
shared = fund_manager.merge(fund_manager, on='manager_ein')
shared = shared[shared['fund_ein_x'] != shared['fund_ein_y']]
shared = shared.rename(columns={
    'fund_ein_x': 'fund_a',
    'fund_ein_y': 'fund_b'
})

# Count how many managers they share
fund_fund_edges = shared.groupby(['fund_a', 'fund_b']).agg(
    shared_managers=('manager_ein', 'count')
).reset_index()

print(f"Fund-Fund edges: {len(fund_fund_edges):,}")
print(fund_fund_edges.head(5).to_string())

# ============================================================
# STEP 6: EXPORT TO CSV
# ============================================================

print("\nExporting...")

OUT = BASE_PATH

funds.to_csv(os.path.join(OUT, "nodes_pension_funds.csv"), index=False)
manager_nodes.to_csv(os.path.join(OUT, "nodes_asset_managers.csv"), index=False)
edges.to_csv(os.path.join(OUT, "edges_fund_to_manager.csv"), index=False)
fund_fund_edges.to_csv(os.path.join(OUT, "edges_fund_to_fund.csv"), index=False)

print("Done. Files saved:")
print(f"  nodes_pension_funds.csv    -> {len(funds):,} nodes")
print(f"  nodes_asset_managers.csv   -> {len(manager_nodes):,} nodes")
print(f"  edges_fund_to_manager.csv  -> {len(edges):,} edges")
print(f"  edges_fund_to_fund.csv     -> {len(fund_fund_edges):,} edges")
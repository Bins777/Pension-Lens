import pandas as pd
import os
from rapidfuzz import fuzz, process

# ============================================================
# PENSIONLENS - Fuzzy Name Matching
# pensionlens_fuzzy.py
# ============================================================

BASE_PATH = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"

managers = pd.read_csv(os.path.join(BASE_PATH, "nodes_asset_managers.csv"), dtype=str)
print(f"Loaded {len(managers):,} asset managers")

# ============================================================
# STEP 1: CLEAN NAMES
# ============================================================

managers['name_clean'] = (
    managers['name']
    .str.upper()
    .str.strip()
    .str.replace(r'\b(LLC|INC|CORP|LTD|LP|NA|CO|GROUP|ADVISORS|ADVISORY|MANAGEMENT|TRUST|SERVICES|FINANCIAL|INVESTMENTS|CAPITAL)\b', '', regex=True)
    .str.replace(r'[^A-Z0-9 ]', '', regex=True)
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)

print("\nSample cleaned names:")
print(managers[['name', 'name_clean']].head(10).to_string())

# ============================================================
# FIX 1: Remove individual person names (trustees not managers)
# ============================================================

import re

def is_person_name(name):
    # flag names with only 2-3 words and no company suffix
    words = name.strip().split()
    has_company_word = any(w in name.upper() for w in [
        'LLC','INC','CORP','TRUST','FUND','MANAGEMENT',
        'ADVISORS','FINANCIAL','CAPITAL','SERVICES','GROUP',
        'INSURANCE','RETIREMENT','PENSION','INVESTMENT'
    ])
    return len(words) <= 3 and not has_company_word

managers['is_person'] = managers['name'].apply(is_person_name)
managers = managers[managers['is_person'] == False].copy()
print(f"After removing person names: {len(managers):,} managers")

# ============================================================
# FIX 2: Tighten threshold for short names (prevents UBS/ADP over-grouping)
# ============================================================

def get_threshold(name):
    # short names need higher threshold to avoid false groupings
    if len(name.strip()) <= 5:
        return 95
    return 85

# ============================================================
# STEP 2: FUZZY GROUPING AT 85% THRESHOLD
# ============================================================

print("\nRunning fuzzy matching at 85% threshold...")

names = managers['name_clean'].tolist()
eins  = managers['ein'].tolist()

parent_group = {}
group_id     = 0
assigned     = {}

for i, name in enumerate(names):
    if eins[i] in assigned:
        continue
    # find all matches above 85%
    matches = process.extract(name, names, scorer=fuzz.token_sort_ratio, limit=None)
    group_members = [eins[j] for j, (_, score, j) in enumerate(matches) if score >= 85]
    for ein in group_members:
        if ein not in assigned:
            assigned[ein] = group_id
    group_id += 1

managers['parent_group_id'] = managers['ein'].map(assigned)

# ============================================================
# STEP 3: FLAG SAME PARENT GROUP
# ============================================================

group_sizes = managers.groupby('parent_group_id')['ein'].count().reset_index()
group_sizes.columns = ['parent_group_id', 'group_size']
managers = managers.merge(group_sizes, on='parent_group_id', how='left')

# flag managers that share a parent group with others
managers['same_parent_flag'] = (managers['group_size'] > 1).astype(int)

print(f"\nManagers with same parent group: {managers['same_parent_flag'].sum():,}")
print(f"Unique parent groups: {managers['parent_group_id'].nunique():,}")

# show sample groupings
print("\nSample grouped managers:")
sample = managers[managers['same_parent_flag'] == 1][['name', 'name_clean', 'parent_group_id', 'group_size']].head(15)
print(sample.to_string())

# ============================================================
# STEP 4: EXPORT
# ============================================================

managers.to_csv(os.path.join(BASE_PATH, "nodes_asset_managers_enriched.csv"), index=False)
print(f"\nExported nodes_asset_managers_enriched.csv")
from neo4j import GraphDatabase
import pandas as pd
import os

# ============================================================
# PENSIONLENS - Neo4j GDS Algorithms
# pensionlens_gds.py
# ============================================================

BASE_PATH = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"
URI       = "bolt://localhost:7687"
USER      = "neo4j"
PASSWORD  = "Vladivostok1."

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Path to save outputs
BASE_PATH  = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: UPDATE MANAGERS WITH ENRICHED DATA
# ============================================================

def update_managers(tx, batch):
    tx.run("""
        UNWIND $batch AS row
        MATCH (a:AssetManager {ein: row.ein})
        SET a.parent_group_id  = toInteger(row.parent_group_id),
            a.group_size       = toInteger(row.group_size),
            a.same_parent_flag = toInteger(row.same_parent_flag)
    """, batch=batch)

print("Updating managers with enriched data...")
enriched = pd.read_csv(os.path.join(BASE_PATH, "nodes_asset_managers_enriched.csv"), dtype=str)
enriched = enriched.fillna("0")
batch = enriched[['ein','parent_group_id','group_size','same_parent_flag']].to_dict('records')

with driver.session() as session:
    session.execute_write(update_managers, batch)
print(f"Updated {len(batch):,} managers")

# ============================================================
# STEP 2: PROJECT GRAPH FOR GDS
# ============================================================

def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return [dict(r) for r in result]

print("\nProjecting graph for GDS...")

# drop existing projection if exists
run_query("CALL gds.graph.drop('pensionlens', false) YIELD graphName")

# project heterogeneous graph
run_query("""
    CALL gds.graph.project(
        'pensionlens',
        ['PensionFund', 'AssetManager'],
        {
            ALLOCATED_TO: {
                orientation: 'NATURAL',
                properties: ['fee_paid']
            },
            SHARES_MANAGER_WITH: {
                orientation: 'UNDIRECTED',
                properties: ['shared_managers']
            }
        }
    )
""")
print("Graph projected successfully")

# ============================================================
# STEP 3: PAGERANK ON ASSET MANAGERS
# ============================================================

print("\nRunning PageRank on Asset Managers...")

run_query("""
    CALL gds.pageRank.write('pensionlens', {
        nodeLabels: ['AssetManager'],
        relationshipTypes: ['ALLOCATED_TO'],
        writeProperty: 'pagerank_score',
        maxIterations: 20,
        dampingFactor: 0.85
    })
""")
print("PageRank written to AssetManager nodes")

# ============================================================
# STEP 4: LOUVAIN COMMUNITY DETECTION
# ============================================================

print("\nRunning Louvain Community Detection...")

run_query("""
    CALL gds.louvain.write('pensionlens', {
        nodeLabels: ['PensionFund', 'AssetManager'],
        relationshipTypes: ['ALLOCATED_TO', 'SHARES_MANAGER_WITH'],
        writeProperty: 'community_id',
        maxLevels: 10,
        maxIterations: 10
    })
""")
print("Louvain communities written to all nodes")

# ============================================================
# STEP 5: STRONGLY CONNECTED COMPONENTS (Cycle Detection)
# ============================================================

print("\nRunning Strongly Connected Components...")

run_query("""
    CALL gds.scc.write('pensionlens', {
        nodeLabels: ['PensionFund', 'AssetManager'],
        relationshipTypes: ['ALLOCATED_TO'],
        writeProperty: 'scc_id'
    })
""")

# flag funds inside an SCC with more than 1 member (circular pattern)
run_query("""
    MATCH (n)
    WHERE n.scc_id IS NOT NULL
    WITH n.scc_id AS scc, count(*) AS size
    WHERE size > 1
    MATCH (m)
    WHERE m.scc_id = scc
    SET m.in_circular_pattern = 1
""")

run_query("""
    MATCH (n)
    WHERE n.scc_id IS NOT NULL AND n.in_circular_pattern IS NULL
    SET n.in_circular_pattern = 0
""")
print("SCC written, circular pattern flags set")

# ============================================================
# STEP 6: BETWEENNESS CENTRALITY
# ============================================================

print("\nRunning Betweenness Centrality...")

run_query("""
    CALL gds.betweenness.write('pensionlens', {
        nodeLabels: ['AssetManager'],
        relationshipTypes: ['ALLOCATED_TO'],
        writeProperty: 'betweenness_score'
    })
""")
print("Betweenness centrality written to AssetManager nodes")

# ============================================================
# STEP 7: NODE SIMILARITY BETWEEN FUNDS
# ============================================================

print("\nRunning Node Similarity between Pension Funds...")

run_query("""
    CALL gds.nodeSimilarity.write('pensionlens', {
        nodeLabels: ['PensionFund'],
        relationshipTypes: ['ALLOCATED_TO'],
        writeRelationshipType: 'SIMILAR_PORTFOLIO',
        writeProperty: 'similarity_score',
        similarityCutoff: 0.5,
        topK: 5
    })
""")
print("Node similarity edges written as SIMILAR_PORTFOLIO")

# ============================================================
# STEP 8: EXPORT GDS SCORES BACK TO CSV
# ============================================================

print("\nExporting GDS scores...")

# export fund scores
fund_scores = run_query("""
    MATCH (p:PensionFund)
    RETURN p.ein AS ein,
           p.community_id AS community_id,
           p.scc_id AS scc_id,
           p.in_circular_pattern AS in_circular_pattern
""")
df_funds = pd.DataFrame(fund_scores)
df_funds.to_csv(os.path.join(BASE_PATH, "gds_fund_scores.csv"), index=False)
print(f"Fund GDS scores: {len(df_funds):,} rows")

# export manager scores
manager_scores = run_query("""
    MATCH (a:AssetManager)
    RETURN a.ein AS ein,
           a.name AS name,
           a.pagerank_score AS pagerank_score,
           a.betweenness_score AS betweenness_score,
           a.community_id AS community_id,
           a.same_parent_flag AS same_parent_flag,
           a.parent_group_id AS parent_group_id
""")
df_managers = pd.DataFrame(manager_scores)
df_managers.to_csv(os.path.join(BASE_PATH, "gds_manager_scores.csv"), index=False)
print(f"Manager GDS scores: {len(df_managers):,} rows")

print("\nDone. Files saved:")
print("  gds_fund_scores.csv")
print("  gds_manager_scores.csv")

driver.close()
from neo4j import GraphDatabase
import pandas as pd
import os

# ============================================================
# PENSIONLENS - GDS Round 2 (Fixed Projection)
# pensionlens_gds2.py
# ============================================================

BASE_PATH  = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"
OUTPUT_DIR = os.path.join(BASE_PATH, "gds2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "Vladivostok1."

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return [dict(r) for r in result]

# ============================================================
# STEP 1: DROP OLD PROJECTION
# ============================================================

print("Dropping old projection...")
run_query("CALL gds.graph.drop('pensionlens', false) YIELD graphName")

# ============================================================
# STEP 2: PROJECT FUND-ONLY GRAPH (uses SHARES_MANAGER_WITH)
# ============================================================

print("Projecting fund-only graph...")
run_query("""
    CALL gds.graph.project(
        'pensionlens',
        'PensionFund',
        {
            SHARES_MANAGER_WITH: {
                orientation: 'UNDIRECTED',
                properties: ['shared_managers']
            }
        }
    )
""")
print("Graph projected successfully")

# ============================================================
# STEP 3: LOUVAIN COMMUNITY DETECTION ON FUNDS
# ============================================================

print("\nRunning Louvain Community Detection...")
run_query("""
    CALL gds.louvain.write('pensionlens', {
        writeProperty: 'community_id',
        maxLevels: 10,
        maxIterations: 10,
        relationshipWeightProperty: 'shared_managers'
    })
""")
print("Louvain communities written")

# ============================================================
# STEP 4: BETWEENNESS CENTRALITY ON FUNDS
# ============================================================

print("\nRunning Betweenness Centrality on Funds...")
run_query("""
    CALL gds.betweenness.write('pensionlens', {
        writeProperty: 'betweenness_score'
    })
""")
print("Betweenness centrality written")

# ============================================================
# STEP 5: DROP AND REPROJECT FOR MANAGER PAGERANK
# ============================================================

print("\nReprojecting for Manager PageRank...")
run_query("CALL gds.graph.drop('pensionlens', false) YIELD graphName")

run_query("""
    CALL gds.graph.project(
        'pensionlens',
        ['PensionFund', 'AssetManager'],
        {
            ALLOCATED_TO: {
                orientation: 'REVERSE',
                properties: ['fee_paid']
            }
        }
    )
""")

run_query("""
    CALL gds.pageRank.write('pensionlens', {
        nodeLabels: ['AssetManager'],
        writeProperty: 'pagerank_score',
        maxIterations: 20,
        dampingFactor: 0.85,
        relationshipWeightProperty: 'fee_paid'
    })
""")
print("PageRank written to managers")

# ============================================================
# STEP 6: COMPUTE COMMUNITY CONCENTRATION PER FUND
# ============================================================

print("\nComputing community concentration per fund...")

# for each fund get its managers community ids
# if all managers are in same community = concentrated
run_query("""
    MATCH (p:PensionFund)-[:ALLOCATED_TO]->(a:AssetManager)
    WITH p, collect(DISTINCT a.community_id) AS manager_communities,
         count(DISTINCT a) AS total_managers
    SET p.manager_community_count = size(manager_communities),
        p.total_managers = total_managers,
        p.community_concentration = 
            CASE WHEN size(manager_communities) = 0 THEN 1.0
                 ELSE 1.0 / size(manager_communities) 
            END
""")
print("Community concentration written to fund nodes")

# ============================================================
# STEP 7: COMPUTE SAME PARENT CONCENTRATION PER FUND
# ============================================================

print("\nComputing same parent concentration per fund...")
run_query("""
    MATCH (p:PensionFund)-[r:ALLOCATED_TO]->(a:AssetManager)
    WITH p,
         sum(r.fee_paid) AS total_fees,
         sum(CASE WHEN a.same_parent_flag = 1 THEN r.fee_paid ELSE 0 END) AS same_parent_fees
    SET p.total_fees = total_fees,
        p.same_parent_fee_ratio = 
            CASE WHEN total_fees = 0 THEN 0.0
                 ELSE same_parent_fees / total_fees 
            END
""")
print("Same parent concentration written")

# ============================================================
# STEP 8: EXPORT ALL SCORES
# ============================================================

print("\nExporting scores...")

fund_scores = run_query("""
    MATCH (p:PensionFund)
    RETURN p.ein                     AS ein,
           p.community_id            AS community_id,
           p.betweenness_score       AS betweenness_score,
           p.total_managers          AS total_managers,
           p.manager_community_count AS manager_community_count,
           p.community_concentration AS community_concentration,
           p.total_fees              AS total_fees,
           p.same_parent_fee_ratio   AS same_parent_fee_ratio,
           p.in_circular_pattern     AS in_circular_pattern
""")

manager_scores = run_query("""
    MATCH (a:AssetManager)
    RETURN a.ein              AS ein,
           a.name             AS name,
           a.pagerank_score   AS pagerank_score,
           a.num_clients      AS num_clients,
           a.total_fees       AS total_fees,
           a.same_parent_flag AS same_parent_flag,
           a.parent_group_id  AS parent_group_id,
           a.group_size       AS group_size,
           a.community_id     AS community_id
""")

df_funds    = pd.DataFrame(fund_scores)
df_managers = pd.DataFrame(manager_scores)

df_funds.to_csv(os.path.join(OUTPUT_DIR, "gds_fund_scores.csv"), index=False)
df_managers.to_csv(os.path.join(OUTPUT_DIR, "gds_manager_scores.csv"), index=False)

print(f"\nFund scores: {len(df_funds):,} rows")
print(f"Manager scores: {len(df_managers):,} rows")
print(f"\nFund score stats:")
print(df_funds[['betweenness_score','community_concentration','same_parent_fee_ratio']].describe())
print(f"\nTop 10 managers by PageRank:")
print(df_managers.nlargest(10, 'pagerank_score')[['name','pagerank_score','num_clients']].to_string())
print(f"\nFiles saved to: {OUTPUT_DIR}")

driver.close()
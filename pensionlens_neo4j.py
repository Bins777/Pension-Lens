from neo4j import GraphDatabase
import pandas as pd
import os

# ============================================================
# PENSIONLENS - Neo4j Ingestion
# pensionlens_neo4j.py
# ============================================================

BASE_PATH = r"C:\Users\georg\OneDrive\PENSIONLENS IDL FINAL PROJECT"

# UPDATE THESE WITH YOUR NEO4J CREDENTIALS
URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "Vladivostok1."

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# ============================================================
# LOAD CSVs
# ============================================================

funds        = pd.read_csv(os.path.join(BASE_PATH, "nodes_pension_funds.csv"), dtype=str)
managers     = pd.read_csv(os.path.join(BASE_PATH, "nodes_asset_managers.csv"), dtype=str)
edges_fm     = pd.read_csv(os.path.join(BASE_PATH, "edges_fund_to_manager.csv"), dtype=str)
edges_ff     = pd.read_csv(os.path.join(BASE_PATH, "edges_fund_to_fund.csv"), dtype=str)

print(f"Funds: {len(funds):,}")
print(f"Managers: {len(managers):,}")
print(f"Fund->Manager edges: {len(edges_fm):,}")
print(f"Fund->Fund edges: {len(edges_ff):,}")

# ============================================================
# INGEST FUNCTIONS
# ============================================================

def ingest_pension_funds(tx, batch):
    tx.run("""
        UNWIND $batch AS row
        MERGE (p:PensionFund {ein: row.ein})
        SET p.plan_name       = row.plan_name,
            p.sponsor_name    = row.sponsor_name,
            p.fund_type       = row.fund_type,
            p.num_participants = toInteger(row.num_participants),
            p.admin_name      = row.admin_name,
            p.state           = row.state,
            p.label           = toInteger(row.label)
    """, batch=batch)

def ingest_asset_managers(tx, batch):
    tx.run("""
        UNWIND $batch AS row
        MERGE (a:AssetManager {ein: row.ein})
        SET a.name        = row.name,
            a.total_fees  = toFloat(row.total_fees),
            a.num_clients = toInteger(row.num_clients)
    """, batch=batch)

def ingest_fund_manager_edges(tx, batch):
    tx.run("""
        UNWIND $batch AS row
        MATCH (p:PensionFund  {ein: row.fund_ein})
        MATCH (a:AssetManager {ein: row.manager_ein})
        MERGE (p)-[r:ALLOCATED_TO]->(a)
        SET r.fee_paid = toFloat(row.fee_paid)
    """, batch=batch)

def ingest_fund_fund_edges(tx, batch):
    tx.run("""
        UNWIND $batch AS row
        MATCH (p1:PensionFund {ein: row.fund_a})
        MATCH (p2:PensionFund {ein: row.fund_b})
        MERGE (p1)-[r:SHARES_MANAGER_WITH]->(p2)
        SET r.shared_managers = toInteger(row.shared_managers)
    """, batch=batch)

# ============================================================
# BATCH INGEST HELPER
# ============================================================

def ingest_in_batches(df, fn, batch_size=500, label=""):
    total = len(df)
    for i in range(0, total, batch_size):
        batch = df.iloc[i:i+batch_size].fillna("").to_dict('records')
        with driver.session() as session:
            session.execute_write(fn, batch)
        print(f"  {label}: {min(i+batch_size, total):,} / {total:,}")

# ============================================================
# RUN INGESTION
# ============================================================

print("\nIngesting Pension Fund nodes...")
ingest_in_batches(funds, ingest_pension_funds, label="Funds")

print("\nIngesting Asset Manager nodes...")
ingest_in_batches(managers, ingest_asset_managers, label="Managers")

print("\nIngesting Fund -> Manager edges...")
ingest_in_batches(edges_fm, ingest_fund_manager_edges, label="Fund->Manager")

print("\nIngesting Fund -> Fund edges...")
ingest_in_batches(edges_ff, ingest_fund_fund_edges, label="Fund->Fund")

print("\nDone. Run this in Neo4j Browser to verify:")
print("MATCH (n) RETURN labels(n) AS label, count(n) AS count;")

driver.close()
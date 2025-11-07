#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
from neo4j import GraphDatabase
import pandas as pd
import csv
import logging
import os

# Neo4j connection details
URI = "neo4j+s://c8be9c06.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "B3V9DcKgVZ0ThrMrrsO0CfsLokWTmOBGnFM_tiIBeRw"

# Setup logging
logging.basicConfig(filename='./revision_logs/human.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    def get_epitope_properties(self, ids):
        query = """
        MATCH (e:Epitope)
        WHERE e.id IN $ids
        OPTIONAL MATCH (e)-[r1:DERIVED_FROM]->(m:Molecule)
        OPTIONAL MATCH (m)-[r2:ORIGINATES_FROM]->(o:Organism)
        OPTIONAL MATCH (m)-[r3:IS_VARIANT_OF]->(pm:Molecule)
        OPTIONAL MATCH (o)-[r4:BELONGS_TO]->(s:Species)
        RETURN e.id AS id,
               e.host AS host,
               e.is_IFNg_inducing AS is_IFNg_inducing,
               m.iri AS source_molecule_iri,
               m.name AS source_molecule_name,
               pm.iri AS molecule_parent_iri,
               pm.name AS molecule_parent_name,
               o.iri AS source_organism_iri,
               o.name AS source_organism_name,
               s.iri AS species_iri,
               s.name AS species_name,
               r1.start_pos AS start_pos,
               r1.end_pos AS end_pos
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, ids=ids)
                return [record.data() for record in result]
        except Exception as e:
            logging.error(f"Error querying Neo4j for IDs {ids}: {str(e)}")
            return []

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-query", "--query_path", type=str, required=True, help="Path to the query .npy file")
parser.add_argument("-db", "--database_path", type=str, required=True, help="Path to the RAG database .npy file")
parser.add_argument("-out", "--output_path", type=str, required=True, help="Path to save the weighted averages (will append _set.npy)")
parser.add_argument("-index", "--index_path", type=str, required=True, help="Path to the index file")
parser.add_argument("-log", "--log_path", type=str, default="rag_log.csv", help="Path to save the log CSV")
parser.add_argument("-batch", "--batch_size", type=int, default=5, help="Number of query sequences per batch")
args = parser.parse_args()

# Parameters
query_path = args.query_path
database_path = args.database_path
output_path = args.output_path
index_path = args.index_path
log_path = args.log_path
batch_size = args.batch_size

# Fixed parameters
maxseq = 20
num_feature = 1280
top_k = 5

# Define weight sets including ablations
weight_sets = {
    'uniform': [1.0, 0.0, 1.0, 1.0, 1.0],
    # 'set_a': [1.0, 0.0, 0.9, 0.7, 0.8],
    # 'set_b': [1.0, 0.0, 0.6, 0.4, 0.5],
    # 'set_c': [1.0, 0.0, 0.3, 0.1, 0.2],
    # 'set_d': [1.0, 0.0, 0.8, 0.7, 0.9],
    # 'set_e': [1.0, 0.0, 0.5, 0.4, 0.6],
    # 'set_f': [1.0, 0.0, 0.2, 0.1, 0.3],
    # 'host_only': [1.0, 0.0, 0.0, 0.0, 0.0],
    # 'molecule_only': [0.0, 0.0, 1.0, 0.0, 0.0],
    # 'parent_only': [0.0, 0.0, 0.0, 1.0, 0.0],
    # 'organism_only': [0.0, 0.0, 0.0, 0.0, 1.0]
}

rel_keys_order = ['host', 'is_IFNg_inducing', 'source_molecule_iri', 'molecule_parent_iri', 'source_organism_iri']

# Load Query and RAG Database
try:
    query_data = np.load(query_path)  # Shape: (N, 1, 20, 1280)
    rag_database = np.load(database_path)  # Shape: (M, 1, 20, 1280)
    print(f"Loaded RAG database: {rag_database.shape}")
    
except Exception as e:
    logging.error(f"Error loading numpy files: {str(e)}")
    raise

# Load index file
try:
    index_df = pd.read_csv(index_path, sep='\t', header=None, names=['index', 'id'])
    index_to_id = dict(zip(index_df['index'], index_df['id']))
except Exception as e:
    logging.error(f"Error loading index file {index_path}: {str(e)}")
    raise

# Connect to Neo4j
neo4j_conn = Neo4jConnection(URI, USERNAME, PASSWORD)

# Initialize log CSV
log_data = []
log_headers = ['Weight_Set', 'Query_Index', 'Epitope_ID', 'Relationships']

# Function to format relationships for logging
def format_relationships(prop):
    relationships = []
    if prop.get('source_molecule_iri') and prop.get('source_molecule_name'):
        relationships.append(f"{prop['id']} is derived from molecule {prop['source_molecule_name']} (IRI: {prop['source_molecule_iri']})")
    if prop.get('source_organism_iri') and prop.get('source_organism_name'):
        relationships.append(f"{prop['id']} originates from organism {prop['source_organism_name']} (IRI: {prop['source_organism_iri']})")
    if prop.get('molecule_parent_iri') and prop.get('molecule_parent_name'):
        relationships.append(f"{prop['id']} is derived from parent molecule {prop['molecule_parent_name']} (IRI: {prop['molecule_parent_iri']})")
    if prop.get('species_iri') and prop.get('species_name'):
        relationships.append(f"{prop['id']} belongs to species {prop['species_name']} (IRI: {prop['species_iri']})")
    if prop.get('host'):
        relationships.append(f"{prop['id']} tested in host {prop['host']}")
    if prop.get('is_IFNg_inducing') is not None:
        relationships.append(f"{prop['id']} is IFN-gamma inducing: {prop['is_IFNg_inducing']}")
    if prop.get('start_pos') is not None and prop.get('end_pos') is not None:
        relationships.append(f"{prop['id']} spans positions {prop['start_pos']} to {prop['end_pos']}")
    return ', '.join(relationships) if relationships else 'None'

# Function to compute weighted average
def compute_weighted_avg(closest_sequences, top_k_indices, rel_weights, maxseq, num_feature):
    top_k_ids = [index_to_id.get(int(idx), None) for idx in top_k_indices]
    top_k_ids = [id_ for id_ in top_k_ids if id_ is not None]
    
    if not top_k_ids:
        logging.warning("No valid IDs found. Using uniform weights.")
        return np.mean(closest_sequences, axis=0).reshape(1, maxseq, num_feature), top_k_ids, []
    
    properties = neo4j_conn.get_epitope_properties(top_k_ids)
    prop_dict = {p['id']: p for p in properties if p['id'] in top_k_ids}
    
    rel_keys = dict(zip(rel_keys_order, rel_weights))
    
    num_top = len(top_k_indices)
    weights = np.ones(num_top)
    
    for i, id_i in enumerate(top_k_ids):
        if id_i not in prop_dict:
            continue
        for key, weight in rel_keys.items():
            if weight == 0:  # Skip zero-weighted keys
                continue
            if prop_dict[id_i].get(key):
                for j, id_j in enumerate(top_k_ids[i+1:], i+1):
                    if id_j in prop_dict and prop_dict[id_i].get(key) == prop_dict[id_j].get(key):
                        weights[i] += weight
                        weights[j] += weight
    
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(num_top) / num_top
    
    weighted_avg = np.average(closest_sequences, axis=0, weights=weights)
    logging.info(f"Computed weights for IDs {top_k_ids}: {weights}")
    
    relationships = [format_relationships(prop_dict.get(id_, {})) for id_ in top_k_ids]
    
    return weighted_avg.reshape(1, maxseq, num_feature), top_k_ids, relationships

def compute_rag_embedding(query, database, rel_weights, maxseq, num_feature, query_idx, weight_set, top_k=5, seed=42):
    """
    Compute RAG-based embedding
    """

    query = query.reshape(1, maxseq, num_feature)
    database = database.reshape(database.shape[0], maxseq, num_feature)
    
    # Compute Euclidean distances
    distances = np.linalg.norm(database - query, axis=(1, 2))
    
    # Select top-k closest sequences
    top_k_indices = np.argsort(distances)[:top_k]
    closest_sequences = database[top_k_indices]
    
    # Compute weighted average embedding
    weighted_avg, top_k_ids, relationships = compute_weighted_avg(
        closest_sequences, top_k_indices, rel_weights, maxseq, num_feature
    )

    # Fuse query and RAG embedding
    fused_embedding = (query * 0.9) + (weighted_avg * 0.1)
    
    # Logging for inspection
    log_data.append({
        'Weight_Set': weight_set,
        'Query_Index': query_idx,
        'Epitope_ID': ','.join(top_k_ids) if top_k_ids else 'None',
        'Relationships': ','.join(relationships) if relationships else 'None'
    })
    
    return fused_embedding.reshape(1, maxseq, num_feature)

# Process in batches
num_queries = query_data.shape[0]
num_batches = (num_queries + batch_size - 1) // batch_size

print(f"Total queries: {num_queries}")
print(f"Processing in {num_batches} batches of size {batch_size}")
print(f"top_k={top_k}")

for weight_set, rel_weights in weight_sets.items():
    print(f"Processing weight set: {weight_set}")
    
    rag_embeddings = np.zeros_like(query_data)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_queries)
        batch_queries = query_data[start_idx:end_idx]

        for i, query in enumerate(batch_queries):
            try:
                query_idx = start_idx + i
                fuse_embedding = compute_rag_embedding(query, rag_database, rel_weights, maxseq, num_feature, query_idx, weight_set)
                rag_embeddings[query_idx] = fuse_embedding
                logging.info(f"Processed query {query_idx + 1}/{num_queries} in batch {batch_idx + 1} for {weight_set}")
            except Exception as e:
                logging.error(f"Error processing query {query_idx + 1}: {str(e)}")
        
        print(f"Batch {batch_idx + 1}/{num_batches} processed for {weight_set}")

    # Save with unique name
    set_output_path = f"{output_path}_{weight_set}.npy"
    try:
        np.save(set_output_path, rag_embeddings)
        print(f"Weighted averages for {weight_set} saved to {set_output_path}")
    except Exception as e:
        logging.error(f"Error saving embeddings: {str(e)}")
        raise

# Save log CSV
# try:
#     with open(log_path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=log_headers)
#         writer.writeheader()
#         writer.writerows(log_data)
#     print(f"Log CSV saved to {log_path}")
# except Exception as e:
#     logging.error(f"Error saving log CSV: {str(e)}")
#     raise

neo4j_conn.close()
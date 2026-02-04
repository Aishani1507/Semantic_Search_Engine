
# Evaluation and Benchmarking Script
# - Computes standard IR metrics (MRR, Recall@k, NDCG@k)
# - Measures query latency percentiles
# - Compares embedding-only and hybrid retrieval systems

import pandas as pd
import numpy as np
import math
import time

from retrieval.search import semantic_search, hybrid_search_rrf


# Load evaluation data
# test_queries.csv: queries reserved for evaluation
# qrels_filtered.csv: relevance judgments 

test_queries = pd.read_csv("test_queries.csv")
qrels = pd.read_csv("qrels_filtered.csv")


# Metric helpers function

def relevant_pids(qid):
    return set(qrels[qrels["qid"] == qid]["pid"])

def reciprocal_rank(retrieved, relevant):
    for i, pid in enumerate(retrieved, 1):
        if pid in relevant:
            return 1 / i
    return 0

def recall_at_k(retrieved, relevant, k):
    return int(any(pid in relevant for pid in retrieved[:k]))

def precision_at_k(retrieved, relevant, k):
    return sum(pid in relevant for pid in retrieved[:k]) / k

def ndcg_at_k(retrieved, relevant, k):
    dcg = sum(
        1 / math.log2(i + 2)
        for i, pid in enumerate(retrieved[:k])
        if pid in relevant
    )
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0


# EVALUATION
# Evaluate a retrieval function on test queries
# using MRR, Recall@10, and NDCG@10.

def evaluate(search_fn, name):
    mrr, r10, nd10 = [], [], []

    for _, row in test_queries.iterrows():
        qid, query = row["qid"], row["query"]
        rel = relevant_pids(qid)

    # Retrieve top-100 documents for evaluation
        results = search_fn(query, top_k=100)
        retrieved = [r["pid"] for r in results]
    
    # Compute metrics for this query
        mrr.append(reciprocal_rank(retrieved, rel))
        r10.append(recall_at_k(retrieved, rel, 10))
        nd10.append(ndcg_at_k(retrieved, rel, 10))

    # Print averaged metrics
    print(f"\n{name}")
    print("MRR:", np.mean(mrr))
    print("Recall@10:", np.mean(r10))
    print("NDCG@10:", np.mean(nd10))


# Latency benchmarking

def latency(search_fn):
    times = []
    for _, row in test_queries.iterrows():
        start = time.time()
        search_fn(row["query"], top_k=10)
        times.append((time.time() - start) * 1000)

    print("p50:", np.percentile(times, 50))
    print("p95:", np.percentile(times, 95))
    print("p99:", np.percentile(times, 99))


# Run evaluation and benchmarks

if __name__ == "__main__":
    evaluate(semantic_search, "Embedding-only")
    evaluate(hybrid_search_rrf, "Hybrid (RRF)")

    print("\nLatency (Embedding-only)")
    latency(semantic_search)

    print("\nLatency (Hybrid)")
    latency(hybrid_search_rrf)

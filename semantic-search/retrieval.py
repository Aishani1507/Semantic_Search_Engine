
# Semantic and Hybrid Retrieval Implementation
# - Dense retrieval using sentence embeddings + FAISS
# - Lexical retrieval using BM25
# - Hybrid retrieval using Reciprocal Rank Fusion (RRF)

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from pathlib import Path

# Load preprocessed passages
BASE_DIR = Path(r"C:\Users\aisha\OneDrive\Desktop\semantic-search")
passages = pd.read_csv(BASE_DIR / "passages_sampled.csv")

# Load sentence embedding model
# Using a lightweight, high-performance MiniLM model

baseline_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode all documents into dense embeddings
# Embeddings are L2-normalized to enable cosine similarity

doc_embeddings = baseline_model.encode(
    passages["text"].tolist(),
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Build FAISS index for approximate nearest neighbor search
# Using HNSW for fast and scalable retrieval
dim = doc_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200
index.add(np.array(doc_embeddings))


# Build BM25 index for lexical (keyword-based) retrieval
tokenized_docs = [doc.lower().split() for doc in passages["text"]]
bm25 = BM25Okapi(tokenized_docs)

# -------------------- SEARCH FUNCTIONS --------------------
# Perform dense semantic search using embeddings and FAISS.

def semantic_search(query, top_k=10):
    q_emb = baseline_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    return [{
        "pid": passages.iloc[idx]["pid"],
        "score": float(score),
        "text": passages.iloc[idx]["text"][:200]
    } for score, idx in zip(scores[0], indices[0])]

# Perform BM25-based lexical retrieval.
def bm25_search(query, top_k=10):
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(scores)[::-1][:top_k]
    return idx


#Perform hybrid retrieval using Reciprocal Rank Fusion (RRF).
#Combines BM25 and dense retrieval rankings without using raw scores.

def hybrid_search_rrf(query, top_k=10, k=60):
        # BM25 ranking
    bm25_idx = bm25_search(query, top_k)

# Embedding-based ranking
    q_emb = baseline_model.encode([query], normalize_embeddings=True)
    _, emb_idx = index.search(q_emb, top_k)
    emb_idx = emb_idx[0]

 # Reciprocal Rank Fusion
    scores = {}
    for r, i in enumerate(bm25_idx, 1):
        scores[i] = scores.get(i, 0) + 1 / (k + r)
    for r, i in enumerate(emb_idx, 1):
        scores[i] = scores.get(i, 0) + 1 / (k + r)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [{
        "pid": passages.iloc[i]["pid"],
        "score": float(s),
        "text": passages.iloc[i]["text"][:200]
    } for i, s in ranked[:top_k]]



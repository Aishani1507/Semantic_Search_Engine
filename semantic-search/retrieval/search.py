
# Retrieval module:
# - Loads preprocessed passages
# - Builds / loads dense embeddings and FAISS index
# - Builds BM25 index
# - Provides semantic and hybrid (RRF) search functions

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# Determine project root directory (semantic-search/)
BASE_DIR = Path(__file__).resolve().parent.parent
print("Project directory:", BASE_DIR)


# Define paths for required 
PASSAGES_PATH = BASE_DIR / "passages_sampled.csv"
EMB_PATH = BASE_DIR / "doc_embeddings.npy"
FAISS_PATH = BASE_DIR / "faiss.index"


# Load preprocessed and sampled passages
if not PASSAGES_PATH.exists():
    raise FileNotFoundError("passages_sampled.csv not found")

passages = pd.read_csv(PASSAGES_PATH)
print("Loaded passages:", len(passages))


# Load pre-trained sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Load document embeddings if they exist, otherwise compute and save them
if EMB_PATH.exists():
    print("Loading document embeddings...")
    doc_embeddings = np.load(EMB_PATH)
else:
    print("Encoding documents...")
    doc_embeddings = model.encode(
        passages["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    np.save(EMB_PATH, doc_embeddings)
    print("Saved doc_embeddings.npy")

print("Embeddings shape:", doc_embeddings.shape)


# Load FAISS index from disk if available, otherwise build a new one
if FAISS_PATH.exists():
    print("Loading FAISS index...")
    index = faiss.read_index(str(FAISS_PATH))
else:
    print("Building FAISS index...")
    dim = doc_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(doc_embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    print("Saved faiss.index")

print("FAISS index size:", index.ntotal)


# BM25 SETUP
# Tokenize documents and build BM25 lexical inde
tokenized_docs = [doc.lower().split() for doc in passages["text"].tolist()]
bm25 = BM25Okapi(tokenized_docs)


# SEARCH FUNCTIONS 
#Perform dense semantic search using sentence embeddings and FAISS.

def semantic_search(query, top_k):
    # Encode query into the same embedding space
    q_emb = model.encode([query], normalize_embeddings=True)

    # Perform nearest neighbor search
    scores, indices = index.search(q_emb, top_k)

    # Sort results by similarity score (highest first)
    pairs = sorted(
    zip(scores[0], indices[0]),
    key=lambda x: x[0],
    reverse=True
)
    # Format results
    return [{
    "pid": passages.iloc[i]["pid"],
    "score": float(s),
    "text": passages.iloc[i]["text"][:200]
} for s, i in pairs]


#Perform hybrid retrieval using Reciprocal Rank Fusion (RRF)
#to combine BM25 and dense retrieval rankings.

def hybrid_search_rrf(query, top_k, k=60):
    # BM25 lexical ranking
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_rank = np.argsort(bm25_scores)[::-1][:top_k]

    # Dense Embedding ranking
    q_emb = model.encode([query], normalize_embeddings=True)
    _, emb_rank = index.search(q_emb, top_k)
    emb_rank = emb_rank[0]

    # Combine rankings using Reciprocal Rank Fusion
    score_map = {}

    for r, idx in enumerate(bm25_rank, 1):
        score_map[idx] = score_map.get(idx, 0) + 1 / (k + r)

    for r, idx in enumerate(emb_rank, 1):
        score_map[idx] = score_map.get(idx, 0) + 1 / (k + r)

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    # Format final ranked results
    return [{
        "pid": passages.iloc[i]["pid"],
        "score": float(score),
        "text": passages.iloc[i]["text"][:200]
    } for i, score in ranked[:top_k]]

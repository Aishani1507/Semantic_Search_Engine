
# Retrieval module:
# - Loads preprocessed passages
# - Builds / loads dense embeddings and FAISS index
# - Builds BM25 index
# - Provides semantic and hybrid (RRF) search functions

from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# PATH SETUP 
# Determine project root directory (semantic-search/)
BASE_DIR = Path(__file__).resolve().parent.parent

PASSAGES_PATH = BASE_DIR / "passages_sampled.csv"
EMB_PATH = BASE_DIR / "doc_embeddings.npy"
FAISS_PATH = BASE_DIR / "faiss.index"


# LOAD DATA 
if not PASSAGES_PATH.exists():
    raise FileNotFoundError("passages_sampled.csv not found")

passages = pd.read_csv(PASSAGES_PATH)


#  LOAD MODEL
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# LOAD / BUILD EMBEDDINGS 
if EMB_PATH.exists():
    doc_embeddings = np.load(EMB_PATH)
else:
    doc_embeddings = model.encode(
        passages["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    np.save(EMB_PATH, doc_embeddings)


#  FAISS INDEX 
if FAISS_PATH.exists():
    index = faiss.read_index(str(FAISS_PATH))
    
else:
    dim = doc_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(doc_embeddings)
    faiss.write_index(index, str(FAISS_PATH))


#  BM25 SETUP 
tokenized_docs = [doc.lower().split() for doc in passages["text"]]
bm25 = BM25Okapi(tokenized_docs)


#  SEARCH FUNCTIONS
def semantic_search(query, top_k=10):

    """
    Dense semantic search using sentence embeddings and FAISS.
    """

    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    pairs = sorted(
        zip(scores[0], indices[0]),
        key=lambda x: x[0],
        reverse=True
    )

    return [{
        "pid": passages.iloc[i]["pid"],
        "score": float(s),
        "text": passages.iloc[i]["text"][:200]
    } for s, i in pairs]


def hybrid_search_rrf(query, top_k=10, k=60):

    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF)
    combining BM25 and dense retrieval rankings.
    """

    # BM25 ranking
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_rank = np.argsort(bm25_scores)[::-1][:top_k]

    # Dense ranking
    q_emb = model.encode([query], normalize_embeddings=True)
    _, emb_rank = index.search(q_emb, top_k)
    emb_rank = emb_rank[0]

    # RRF fusion
    score_map = {}

    for r, idx in enumerate(bm25_rank, 1):
        score_map[idx] = score_map.get(idx, 0) + 1 / (k + r)

    for r, idx in enumerate(emb_rank, 1):
        score_map[idx] = score_map.get(idx, 0) + 1 / (k + r)

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    return [{
        "pid": passages.iloc[i]["pid"],
        "score": float(score),
        "text": passages.iloc[i]["text"][:200]
    } for i, score in ranked[:top_k]]

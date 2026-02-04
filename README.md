# Semantic_Search_Engine

By: AISHANI KAPOOR
    2025A3RM2259H

This project implements a **semantic search engine** for retrieving relevant documents from a large corpus using **dense embeddings** and **approximate nearest neighbor (ANN)** search.  
An additional **hybrid retrieval system** combining dense embeddings with **BM25 lexical search** is also implemented and evaluated.

## Features
- Dense semantic retrieval using Sentence-BERT
- Approximate nearest neighbor search using FAISS (HNSW index)
- Hybrid retrieval using BM25 + Reciprocal Rank Fusion (RRF)
- Command-line interface (CLI) for querying
- Evaluation using standard IR metrics (MRR, Recall@k, NDCG@k)
- Latency benchmarking (p50, p95, p99)
- Dataset subsampling for scalability analysis

  ## Project Structure
  semantic-search/
│
├── dataset_loader.py # Loads MS MARCO dataset files
├── data_prep.py # Sampling, alignment, train/val/test split
│
├── retrieval/
│ ├── init.py
│ └── search.py # Dense & hybrid retrieval logic
│
├── implementation.py # CLI-based search engine
├── evaluate.py # Evaluation and latency benchmarking
│
├── requirements.txt
├── README.md


## Dataset

This project uses the **MS MARCO Passage Ranking Dataset**.

### Dataset not included
Due to its large size (millions of passages), the dataset is **not included** in this repository.

### Required files
Download the following files from the official MS MARCO website:

https://microsoft.github.io/msmarco/

Required files:
- `collection.tsv`
- `queries.dev.tsv`
- `qrels.dev.tsv`

### Dataset setup
  
1. Download the dataset files
2. Place them in a local directory (e.g. `dataset_IEEE/`)
3. Update the dataset path in `dataset_loader.py`:

```python
BASE_PATH = Path("/path/to/your/msmarco/dataset")
```

To create a virtual environment
```python
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

### Install dependencies
```python
pip install -r requirements.txt
```

## Data Preparation

Run the script to
1. preprocess data
2. Align queries and relevance judgments
3. Create train / validation / test splits

```python
python data_prep.py
```

This will generate:
- passages_sampled.csv
- train_queries.csv
- val_queries.csv
- test_queries.csv
- qrels_filtered.csv

## Running the Search Engine

The system is accessed via a command-line interface.

- Embedding only Retrieval

```python
python implementation.py --query "machine learning"
```

- Hybrid Retrieval(BM25 + Retrieval)
  
```python
python implementation.py --query "machine learning" --mode hybrid
```

Optional parameters:
--top_k : Number of results to return (default: 10)

## Evaluation

Run the evaluation script to compute retrieval quality and efficiency metrics:

```python
python evaluate.py
```

Retrieval Metrics
- Mean Reciprocal Rank (MRR)
  Measures how early the first relevant document appears in the ranked list.
  
- Recall@10
  Measures whether at least one relevant document is retrieved within the top-10 results.
  
- NDCG@10
  Measures ranking quality while accounting for the position of relevant documents.

Metrics are reported for:
- Embedding-only retrieval
- Hybrid retrieval (RRF)
- Efficiency Metrics

Query latency percentiles:
- p50 (median): typical query latency
- p95/ p99 : worst-case latency

### Observations:
- Emedding-only retrieval is significantly faster due to FAISS ANN search.
- Hybrid retrieval incurs higher latency because BM25 scoring is performed over the full document corpus.
- FAISS HNSW indexing provides a good trade-off between retrieval accuracy and speed.

## Summary
This project implements a complete semantic search system with:
- Dense and hybrid retrieval
- Standard IR evaluation metrics
- Latency and scalability analysis
- Clean, modular, and reproducible design



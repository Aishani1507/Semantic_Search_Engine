# Semantic_Search_Engine

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
├── .gitignore

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

To create a virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

## Data Preparation

Run the script to
1. preprocess data
2. Align queries and relevance judgments
3. Create train / validation / test splits

python data_prep.py

This will generate:
- passages_sampled.csv
- train_queries.csv
- val_queries.csv
- test_queries.csv
- qrels_filtered.csv

## Running the Search Engine

The system is accessed via a command-line interface.

- Embedding only Retrieval
python implementation.py --query "machine learning"

- Hybrid Retrieval(BM25 + Retrieval)
python implementation.py --query "machine learning" --mode hybrid

## Evaluation

Run the evaluation script to compute retrieval quality and efficiency metrics:

python evaluate.py

Retrieval Metrics
- Mean Reciprocal Rank (MRR)
- Recall@10
- NDCG@10

Metrics are reported for:
- Embedding-only retrieval
- Hybrid retrieval (RRF)
- Efficiency Metrics

Query latency percentiles:
- p50 (median)
- p95
- p99

## Summary
This project implements a complete semantic search system with:
- Dense and hybrid retrieval
- Standard IR evaluation metrics
- Latency and scalability analysis
- Clean, modular, and reproducible design



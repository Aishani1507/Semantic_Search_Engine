
# dataset_loader.py
"""
Utility functions for loading the MS MARCO dataset components:
- Passages (documents)
- Queries
- Relevance judgments (qrels)

"""

import pandas as pd
from pathlib import Path

# Base path where MS MARCO files are stored
# Update this path according to your device

BASE_PATH = Path(r"C:\Users\aisha\OneDrive\Desktop\dataset_IEEE")

"""
 Args:
        nrows (int, optional): Limits the number of relevance records loaded,
            which is helpful for debugging or subsampling.

    Returns:
        pd.DataFrame: A DataFrame mapping query IDs to passage IDs along with
        their relevance labels.
"""


MSMARCO_DIR = BASE_PATH / "msmarco"
MSMARCO_DIR.mkdir(parents=True, exist_ok=True)

# Load MS MARCO passages
# The collection.tsv file stores passages in the format:
# <passage_id> \t <passage_text>


def load_passages(nrows=None):
    return pd.read_csv(BASE_PATH / "collection.tsv", 
    sep="\t", 
    names=["pid", "text"], 
    dtype={"pid": str}, 
    nrows=nrows,
    encoding="utf-8",
    encoding_errors="replace")

# Load MS MARCO queries
# The queries.dev.tsv file stores passages in the format:
# <query_id> \t <query_text>

def load_queries(nrows=None):
    return pd.read_csv(BASE_PATH / "queries.dev.tsv", 
    sep="\t", 
    names=["qid", "query"], 
    dtype={"qid": str}, 
    nrows=nrows,
    encoding="utf-8",
    encoding_errors="replace")

# Load relevance judgement(qrels)
# The qrels.dev.tsv file stores passages in the format:
# <query_id>\t<unused>\t<passage_id>\t<relevance>

def load_qrels(nrows=None):
    return pd.read_csv(BASE_PATH / "qrels.dev.tsv",
     sep="\t", 
     names=["qid", "unused", "pid", "relevance"], 
     dtype={"qid": str, "pid": str}, 
     nrows=nrows,
     encoding="utf-8",
     encoding_errors="replace")






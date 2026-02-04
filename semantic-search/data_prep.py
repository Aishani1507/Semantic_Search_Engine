from dataset_loader import load_passages, load_queries, load_qrels
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path

passages = load_passages()
queries = load_queries()
qrels = load_qrels()

print("Raw passages:", len(passages))
print("Raw queries:", len(queries))
print("Raw qrels:", len(qrels))


N = 50000   # use 10000 first if Colab RAM is low

passages = passages.sample(n=N, random_state=42).reset_index(drop=True)

print("Sampled passages:", len(passages))

sampled_pids = set(passages["pid"])

qrels = qrels[qrels["pid"].isin(sampled_pids)]

valid_qids = set(qrels["qid"])
queries = queries[queries["qid"].isin(valid_qids)].reset_index(drop=True)

print("Aligned queries:", len(queries))
print("Aligned qrels:", len(qrels))

queries = queries.sample(n=200, random_state=42).reset_index(drop=True)

print("Final queries:", len(queries))

from sklearn.model_selection import train_test_split

train_q, temp_q = train_test_split(
    queries, test_size=0.3, random_state=42
)

val_q, test_q = train_test_split(
    temp_q, test_size=0.5, random_state=42
)

print("Train:", len(train_q))
print("Validation:", len(val_q))
print("Test:", len(test_q))

passages.to_csv("passages_sampled.csv", index=False)
train_q.to_csv("train_queries.csv", index=False)
val_q.to_csv("val_queries.csv", index=False)
test_q.to_csv( "test_queries.csv", index=False)
qrels.to_csv( "qrels_filtered.csv", index=False)


print("data processing complete")


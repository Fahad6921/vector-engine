# Hybrid search, BEIR SciFact

```sh
python bench/beir.py scifact
cargo run --release -p vecengine-storage --example hybrid_eval -- data/scifact
```

5,183 abstracts, 300 test queries. Vectors from `all-MiniLM-L6-v2` (384 dims, cosine), HNSW ef=200. Hybrid fuses the top 100 of each with reciprocal rank fusion (c = 60).

| | nDCG@10 | published BEIR number |
|---|---|---|
| BM25 | 0.6663 | 0.665 (Elasticsearch) |
| vector | 0.6451 | 0.645 (same model) |
| hybrid | **0.6830** | |

# Server, SIFT 200k

```sh
vecengine-server --data /tmp/ve &
python bench/load_server.py sift-128-euclidean --n 200000
oha -z 10s -c 8 -m POST -D data/query.json -T application/json http://127.0.0.1:8080/collections/sift/search
```

Apple M1, 8 GB. Server and load generator on the same machine, sharing 8 cores.

Ingest over HTTP in batches of 2,000: 200,000 vectors in 19.5s (10,275/s, including the HNSW build). Recall@10 over HTTP at ef=100: 0.9975.

Search, k=10, ef=100, no payloads, 10s per row:

| clients | req/s | p50 | p99 |
|---|---|---|---|
| 1 | 6,133 | 0.15 ms | 0.31 ms |
| 4 | 15,298 | 0.22 ms | 0.93 ms |
| 8 | 19,595 | 0.34 ms | 1.04 ms |
| 32 | 25,606 | 1.05 ms | 4.48 ms |

No failed requests. SIGTERM flushed the memtable to a segment and left the WAL empty; on restart all 200,001 points were back.

# vector-engine

A vector search engine written from scratch in Rust: HNSW graph index, quantization, filtered search and hybrid BM25 + vector ranking.

Work in progress. What works today: SIMD distance kernels, exact search, and an in-memory HNSW index evaluated on the standard [ann-benchmarks](https://ann-benchmarks.com) datasets.

## Running it

Needs Rust (stable) and Python 3.10+ for the dataset script.

```sh
# tests
cargo test --release

# download SIFT-1M and GloVe-100 (~1 GB) into data/
python3 -m venv .venv && .venv/bin/pip install -r bench/requirements.txt
.venv/bin/python bench/download.py

# build an HNSW index and sweep the search beam width (ef)
cargo run --release --example eval -- data/sift-128-euclidean

# faster iteration on the first 100k vectors
cargo run --release --example eval -- data/sift-128-euclidean --n 100000 --ef 20,40,80

# exact search, all threads
cargo run --release --example eval -- data/glove-100-angular --index flat

# distance kernel micro-benchmarks
cargo bench --bench distance
```

`eval` options: `--index hnsw|flat`, `--m`, `--efc` (build beam width), `--ef` (comma-separated), `--n`, `--queries`.

## Results so far

Apple M1, 8 GB.

**Distance kernels** (NEON vs plain Rust loop, ns per call):

| dim | L2 SIMD | L2 scalar | speedup |
|---|---|---|---|
| 128 | 11.2 | 63.7 | 5.7x |
| 384 | 24.9 | 250.9 | 10.1x |
| 768 | 51.9 | 621.7 | 12.0x |
| 1536 | 113.3 | 1353.9 | 11.9x |

The scalar loop isn't auto-vectorised because float addition isn't associative, so the compiler can't split the sum across lanes. The SIMD kernels use four independent accumulators to hide FMA latency.

**Exact search**, 10k queries, all 8 threads. One query at a time is memory-bandwidth bound, so batch search scores blocks of 32 queries against cache-sized blocks of data:

| dataset | per-query | tiled | recall@10 |
|---|---|---|---|
| SIFT-1M | 92 QPS | 154 QPS | 0.9993* |
| GloVe-100 | 80 QPS | 241 QPS | 1.0000 |

\* SIFT has many tied distances, so exact search can legitimately return a different id than the ground-truth file.

**HNSW on SIFT-1M**, M=16, efConstruction=200, 2000 queries, single thread, one query at a time. Build: 335s (single-threaded for now).

| ef | recall@10 | QPS |
|---|---|---|
| 10 | 0.6732 | 26,623 |
| 20 | 0.8143 | 17,282 |
| 40 | 0.9117 | 11,083 |
| 80 | 0.9669 | 6,434 |
| 120 | 0.9837 | 4,209 |
| 200 | 0.9937 | 2,779 |
| 400 | 0.9980 | 1,585 |

Compared with exact search at 154 QPS on all 8 threads, HNSW answers ~40x more queries per second on a single thread at 0.97 recall.

## Layout

```
crates/core/src/
  distance.rs   L2 / dot / cosine, scalar + NEON + AVX2
  neighbor.rs   result type and bounded top-k heap
  flat.rs       exact search
  hnsw.rs       HNSW graph index
  dataset.rs    benchmark dataset loading, recall@k
  io.rs         fvecs / ivecs readers
crates/core/examples/eval.rs   build + evaluate on a dataset
bench/download.py              fetch and convert ann-benchmarks datasets
```

## Roadmap

- [x] SIMD distances, exact search
- [x] HNSW
- [ ] Python bindings, benchmarks against hnswlib and FAISS
- [ ] Parallel build, performance pass
- [ ] On-disk segments, WAL, mmap loading, deletes
- [ ] Quantization (SQ8, PQ, binary) with reranking
- [ ] Filtered search
- [ ] BM25 + hybrid search
- [ ] HTTP / gRPC server, Docker image

## License

MIT

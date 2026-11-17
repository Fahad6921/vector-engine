# Filtered search, SIFT 200k

`cargo run --release -p vecengine-storage --example filter_bench -- data/sift-128-euclidean 200000`

Apple M1, one segment, ef=100, 500 queries, 1 thread. Payload `{"bucket": 0..999}`, filter `bucket < x`. Recall@10 is against the exact filtered top 10.

| selected | points | graph recall | graph QPS | scan QPS | auto recall | auto QPS |
|---|---|---|---|---|---|---|
| 0.1% | 180 | 1.0000 | 29 | 98,139 | 1.0000 | 121,246 |
| 0.2% | 385 | 1.0000 | 58 | 80,448 | 1.0000 | 79,143 |
| 0.5% | 1,006 | 1.0000 | 106 | 41,123 | 1.0000 | 40,764 |
| 1% | 2,001 | 1.0000 | 157 | 22,063 | 1.0000 | 22,061 |
| 2% | 4,014 | 1.0000 | 262 | 11,216 | 1.0000 | 10,624 |
| 5% | 10,018 | 1.0000 | 515 | 3,309 | 1.0000 | 1,864 |
| 10% | 20,084 | 1.0000 | 960 | 804 | 1.0000 | 1,010 |
| 20% | 39,966 | 0.9998 | 1,530 | 322 | 0.9998 | 1,582 |
| 50% | 100,092 | 0.9974 | 1,990 | 247 | 0.9974 | 2,022 |
| 100% | 200,000 | 0.9924 | 1,742 | 326 | 0.9924 | 1,716 |

Auto scans exactly when at most 10% of a segment matches, otherwise walks the graph.

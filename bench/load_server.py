"""Load a benchmark dataset into a running vecengine-server, then check recall.

Usage:
    vecengine-server --data /tmp/ve &
    python bench/load_server.py sift-128-euclidean --n 200000

Uses only the standard library and numpy, so it doubles as an example client.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_vecs(path: Path, dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    return raw.reshape(-1, raw[0] + 1)[:, 1:].copy().view(dtype)


def call(base: str, method: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, method=method, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--collection", default="sift")
    p.add_argument("--batch", type=int, default=2000)
    args = p.parse_args()

    d = ROOT / "data" / args.dataset
    train = read_vecs(d / "train.fvecs", np.float32)[: args.n]
    test = read_vecs(d / "test.fvecs", np.float32)[:200]
    metric = {"euclidean": "l2", "angular": "cosine"}[(d / "metric").read_text().strip()]

    call(args.url, "PUT", f"/collections/{args.collection}", {"dim": train.shape[1], "metric": metric})
    t = time.perf_counter()
    for start in range(0, len(train), args.batch):
        points = [
            {"id": i, "vector": train[i].tolist(), "payload": {"bucket": i % 1000}}
            for i in range(start, min(start + args.batch, len(train)))
        ]
        call(args.url, "PUT", f"/collections/{args.collection}/points", {"points": points})
    call(args.url, "POST", f"/collections/{args.collection}/flush")
    took = time.perf_counter() - t
    print(f"loaded {len(train)} vectors in {took:.1f}s ({len(train) / took:,.0f}/s, including the index build)")

    # Recall against exact search over the same subset.
    # ||x||^2 - 2 q.x ranks like ||q - x||^2 without a (queries x n x dim) array.
    dists = (train ** 2).sum(1)[None, :] - 2 * test @ train.T
    truth = np.argsort(dists, axis=1)[:, :10]
    hits = 0
    for q, t_ids in zip(test, truth):
        res = call(args.url, "POST", f"/collections/{args.collection}/search",
                   {"vector": q.tolist(), "k": 10, "ef": 100, "with_payload": False})
        hits += len({h["id"] for h in res["hits"]} & set(t_ids.tolist()))
    print(f"recall@10 over HTTP: {hits / (len(test) * 10):.4f}")

    query = {"vector": test[0].tolist(), "k": 10, "ef": 100, "with_payload": False}
    (ROOT / "data" / "query.json").write_text(json.dumps(query))
    print("wrote data/query.json for load testing")


if __name__ == "__main__":
    main()

"""Recall vs QPS sweep for vecengine, hnswlib and FAISS-HNSW on one dataset.

All libraries get the same M and efConstruction. Queries run one at a time on a
single thread, which is how ann-benchmarks measures QPS. Each ef setting is
timed RUNS times and the best run is kept, to cut scheduler noise.

Usage:
    python bench/run.py sift-128-euclidean
    python bench/run.py glove-100-angular --libs vecengine,hnswlib --queries 2000
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
K = 10
RUNS = 3
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def read_vecs(path: Path, dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(dtype)


def load(name: str):
    d = ROOT / "data" / name
    train = read_vecs(d / "train.fvecs", np.float32)
    test = read_vecs(d / "test.fvecs", np.float32)
    neighbors = read_vecs(d / "neighbors.ivecs", np.int32)
    metric = (d / "metric").read_text().strip()
    return train, test, neighbors, metric


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    hits = sum(len(set(f) & set(t[:K])) for f, t in zip(found, truth))
    return hits / (len(found) * K)


class VecEngine:
    name = "vecengine"

    def __init__(self, dim, metric, m, efc):
        import vecengine

        metric = {"euclidean": "l2", "angular": "cosine"}[metric]
        self.index = vecengine.Index(dim, metric=metric, m=m, ef_construction=efc)

    def build(self, data):
        self.index.add(data)

    def set_ef(self, ef):
        self.ef = ef

    def query(self, q):
        return self.index.search(q, K, ef=self.ef)[0]


class Hnswlib:
    name = "hnswlib"

    def __init__(self, dim, metric, m, efc):
        import hnswlib

        space = {"euclidean": "l2", "angular": "cosine"}[metric]
        self.index = hnswlib.Index(space=space, dim=dim)
        self.m, self.efc = m, efc

    def build(self, data):
        self.index.init_index(max_elements=len(data), M=self.m, ef_construction=self.efc, random_seed=1)
        self.index.add_items(data)
        self.index.set_num_threads(1)

    def set_ef(self, ef):
        self.index.set_ef(max(ef, K))

    def query(self, q):
        return self.index.knn_query(q, k=K)[0][0]


class Faiss:
    name = "faiss-hnsw"

    def __init__(self, dim, metric, m, efc):
        import faiss

        self.faiss = faiss
        self.angular = metric == "angular"
        kind = faiss.METRIC_INNER_PRODUCT if self.angular else faiss.METRIC_L2
        self.index = faiss.IndexHNSWFlat(dim, m, kind)
        self.index.hnsw.efConstruction = efc

    def _prep(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.angular:
            x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        return x

    def build(self, data):
        self.index.add(self._prep(data))
        self.faiss.omp_set_num_threads(1)

    def set_ef(self, ef):
        self.index.hnsw.efSearch = max(ef, K)

    def query(self, q):
        return self.index.search(self._prep(q[None, :]), K)[1][0]


LIBS = {cls.name: cls for cls in [VecEngine, Hnswlib, Faiss]}


def run(lib_cls, train, test, truth, metric, m, efc):
    lib = lib_cls(train.shape[1], metric, m, efc)
    t = time.perf_counter()
    lib.build(train)
    build_s = time.perf_counter() - t
    print(f"{lib.name}: built in {build_s:.1f}s")

    points = []
    for ef in EFS:
        lib.set_ef(ef)
        best = float("inf")
        for _ in range(RUNS):
            found = np.empty((len(test), K), dtype=np.int64)
            t = time.perf_counter()
            for i, q in enumerate(test):
                found[i] = lib.query(q)
            best = min(best, time.perf_counter() - t)
        r = recall(found, truth)
        qps = len(test) / best
        points.append({"ef": ef, "recall": r, "qps": qps})
        print(f"  ef={ef:<4} recall={r:.4f} qps={qps:,.0f}")
    return {"build_seconds": build_s, "points": points}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--libs", default=",".join(LIBS))
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--efc", type=int, default=200)
    args = p.parse_args()

    train, test, neighbors, metric = load(args.dataset)
    test, neighbors = test[: args.queries], neighbors[: args.queries]
    print(f"{args.dataset}: {train.shape}, {len(test)} queries, {metric}")

    out = {
        "dataset": args.dataset,
        "m": args.m,
        "ef_construction": args.efc,
        "queries": len(test),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "results": {},
    }
    for name in args.libs.split(","):
        out["results"][name] = run(LIBS[name], train, test, neighbors, metric, args.m, args.efc)

    dest = ROOT / "bench" / "results" / f"{args.dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

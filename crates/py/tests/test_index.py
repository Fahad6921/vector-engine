import numpy as np
import pytest

import vecengine


def brute_force(data, queries, k, metric):
    if metric == "cosine":
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        dist = -queries @ data.T
    elif metric == "ip":
        dist = -queries @ data.T
    else:
        dist = ((queries[:, None, :] - data[None, :, :]) ** 2).sum(-1)
    return np.argsort(dist, axis=1)[:, :k]


@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_recall_against_brute_force(metric):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2000, 32)).astype(np.float32)
    queries = rng.standard_normal((50, 32)).astype(np.float32)

    index = vecengine.Index(32, metric=metric)
    index.add(data)
    ids, _ = index.search(queries, k=10, ef=128)

    truth = brute_force(data, queries, 10, metric)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
    assert recall > 0.95


def test_shapes():
    index = vecengine.Index(4)
    index.add(np.eye(4, dtype=np.float32))
    assert len(index) == 4

    ids, dists = index.search(np.array([1, 0, 0, 0], dtype=np.float32), k=2)
    assert ids.shape == (2,) and dists.shape == (2,)
    assert ids[0] == 0 and dists[0] == 0.0

    ids, dists = index.search(np.eye(4, dtype=np.float32), k=3)
    assert ids.shape == (4, 3)
    assert list(ids[:, 0]) == [0, 1, 2, 3]


def test_fewer_vectors_than_k_pads():
    index = vecengine.Index(2)
    index.add(np.array([[1, 1]], dtype=np.float32))
    ids, dists = index.search(np.zeros(2, dtype=np.float32), k=3)
    assert list(ids) == [0, -1, -1]
    assert np.isinf(dists[1:]).all()


def test_non_contiguous_input():
    data = np.arange(40, dtype=np.float32).reshape(4, 10)[:, ::2]  # strided view
    index = vecengine.Index(5)
    index.add(data)
    ids, _ = index.search(data[2], k=1)
    assert ids[0] == 2


def test_errors():
    with pytest.raises(ValueError):
        vecengine.Index(8, metric="manhattan")
    index = vecengine.Index(8)
    with pytest.raises(ValueError):
        index.add(np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        index.search(np.zeros(4, dtype=np.float32))


def test_repr():
    index = vecengine.Index(8, metric="cosine", m=8)
    assert repr(index) == 'Index(dim=8, metric="cosine", m=8, ef_construction=200, len=0)'

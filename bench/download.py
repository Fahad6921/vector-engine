"""Download ann-benchmarks datasets and convert them to fvecs/ivecs.

Each dataset ends up in data/<name>/ as:
    train.fvecs      base vectors
    test.fvecs       query vectors
    neighbors.ivecs  ground-truth ids (100 per query)
    metric           "euclidean" or "angular"

Usage:
    python bench/download.py sift-128-euclidean glove-100-angular
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

BASE_URL = "https://ann-benchmarks.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT = ["sift-128-euclidean", "glove-100-angular"]


def write_vecs(path: Path, arr: np.ndarray) -> None:
    """fvecs/ivecs: per row, an int32 dimension followed by the row's values."""
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = arr.view(np.int32)
    out.tofile(path)


def download(url: str, dest: Path) -> None:
    # The server rejects urllib's default User-Agent with a 403.
    req = urllib.request.Request(url, headers={"User-Agent": "vector-engine-bench"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()


def fetch(name: str) -> None:
    out_dir = DATA_DIR / name
    if (out_dir / "neighbors.ivecs").exists():
        print(f"{name}: already present")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = DATA_DIR / f"{name}.hdf5"
    if not h5_path.exists():
        print(f"{name}: downloading")
        tmp = h5_path.with_suffix(".part")
        download(f"{BASE_URL}/{name}.hdf5", tmp)
        tmp.rename(h5_path)
        print()

    print(f"{name}: converting")
    with h5py.File(h5_path, "r") as f:
        write_vecs(out_dir / "train.fvecs", np.asarray(f["train"], dtype=np.float32))
        write_vecs(out_dir / "test.fvecs", np.asarray(f["test"], dtype=np.float32))
        write_vecs(out_dir / "neighbors.ivecs", np.asarray(f["neighbors"], dtype=np.int32))
        (out_dir / "metric").write_text(f.attrs["distance"])

    h5_path.unlink()
    print(f"{name}: done -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=DEFAULT)
    args = parser.parse_args()
    for name in args.datasets:
        fetch(name)


if __name__ == "__main__":
    main()

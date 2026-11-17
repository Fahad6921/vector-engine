"""Download a BEIR dataset and embed it, for the hybrid search evaluation.

Writes data/<name>/:
    corpus.fvecs    one normalised embedding per document
    corpus.tsv      doc index, "title text" (tabs and newlines replaced)
    queries.fvecs   embeddings of the test queries
    queries.tsv     query index, text
    qrels.tsv       query index, doc index, relevance

Usage:
    python bench/beir.py            # scifact
    python bench/beir.py nfcorpus
"""

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean(s: str) -> str:
    return " ".join(s.split())


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = arr.astype(np.float32).view(np.int32)
    out.tofile(path)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    out = ROOT / "data" / name
    if (out / "qrels.tsv").exists():
        print(f"{name}: already present")
        return
    out.mkdir(parents=True, exist_ok=True)

    print(f"{name}: downloading")
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "vector-engine-bench"})
    with urllib.request.urlopen(req) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))

    def lines(member: str):
        with archive.open(f"{name}/{member}") as f:
            yield from io.TextIOWrapper(f, encoding="utf-8")

    corpus = [json.loads(l) for l in lines("corpus.jsonl")]
    doc_index = {d["_id"]: i for i, d in enumerate(corpus)}
    queries = {q["_id"]: q["text"] for q in map(json.loads, lines("queries.jsonl"))}

    qrels = {}
    for i, line in enumerate(lines("qrels/test.tsv")):
        if i == 0:
            continue  # header
        qid, did, score = line.rstrip("\n").split("\t")
        if did in doc_index and int(score) > 0:
            qrels.setdefault(qid, []).append((doc_index[did], int(score)))
    test_qids = sorted(qrels)

    doc_texts = [clean(f"{d.get('title', '')} {d['text']}") for d in corpus]
    query_texts = [clean(queries[q]) for q in test_qids]

    print(f"{name}: embedding {len(doc_texts)} docs and {len(query_texts)} queries with {MODEL}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    docs = model.encode(doc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    qs = model.encode(query_texts, batch_size=64, normalize_embeddings=True)

    write_fvecs(out / "corpus.fvecs", docs)
    write_fvecs(out / "queries.fvecs", qs)
    (out / "corpus.tsv").write_text("".join(f"{i}\t{t}\n" for i, t in enumerate(doc_texts)))
    (out / "queries.tsv").write_text("".join(f"{i}\t{t}\n" for i, t in enumerate(query_texts)))
    (out / "qrels.tsv").write_text(
        "".join(f"{qi}\t{d}\t{s}\n" for qi, q in enumerate(test_qids) for d, s in qrels[q])
    )
    print(f"{name}: done -> {out}")


if __name__ == "__main__":
    main()

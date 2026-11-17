//! nDCG@10 of keyword, vector and hybrid search on a BEIR dataset.
//!
//!     python bench/beir.py scifact
//!     cargo run --release -p vecengine-storage --example hybrid_eval -- data/scifact

use std::collections::HashMap;
use std::path::Path;

use vecengine::io::read_fvecs;
use vecengine::Metric;
use vecengine_storage::{Collection, Config, Hit, Record};

const K: usize = 10;
const EF: usize = 200;

fn read_tsv(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| l.split('\t').map(String::from).collect())
        .collect()
}

/// nDCG@k with linear gains, as BEIR reports it.
fn ndcg(ranked: &[u64], relevant: &HashMap<u64, u32>, k: usize) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| relevant.get(id).copied().unwrap_or(0) as f64 / (i as f64 + 2.0).log2())
        .sum();
    let mut ideal: Vec<u32> = relevant.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &g)| g as f64 / (i as f64 + 2.0).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: hybrid_eval <beir dir>");
    let dir = Path::new(&dir);
    let docs = read_fvecs(dir.join("corpus.fvecs")).unwrap();
    let doc_text = read_tsv(&dir.join("corpus.tsv"));
    let queries = read_fvecs(dir.join("queries.fvecs")).unwrap();
    let query_text = read_tsv(&dir.join("queries.tsv"));
    let mut qrels: HashMap<usize, HashMap<u64, u32>> = HashMap::new();
    for row in read_tsv(&dir.join("qrels.tsv")) {
        qrels
            .entry(row[0].parse().unwrap())
            .or_default()
            .insert(row[1].parse().unwrap(), row[2].parse().unwrap());
    }

    let path = std::env::temp_dir().join(format!("vecengine-hybrid-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    let config = Config {
        sync: false,
        text_field: Some("text".into()),
        ..Config::new(docs.dim, Metric::Cosine)
    };
    let mut c = Collection::open(&path, config).unwrap();
    let records = (0..docs.len())
        .map(|i| Record::Upsert {
            id: i as u64,
            vector: docs.row(i).to_vec(),
            payload: serde_json::json!({ "text": doc_text[i][1] })
                .to_string()
                .into_bytes(),
        })
        .collect();
    c.write(records).unwrap();
    c.flush().unwrap();

    let ids = |hits: Vec<Hit>| hits.into_iter().map(|h| h.id).collect::<Vec<_>>();
    let mut totals = [0.0f64; 3];
    for (q, relevant) in &qrels {
        let text = &query_text[*q][1];
        let vector = queries.row(*q);
        let runs = [
            ids(c.search_text(text, K, None)),
            ids(c.search(vector, K, EF)),
            ids(c.search_hybrid(vector, text, K, EF, None)),
        ];
        for (total, ranked) in totals.iter_mut().zip(&runs) {
            *total += ndcg(ranked, relevant, K);
        }
    }

    let name = dir.file_name().unwrap().to_string_lossy();
    println!("{name}: {} docs, {} queries, nDCG@{K}", docs.len(), qrels.len());
    for (label, total) in ["bm25", "vector (all-MiniLM-L6-v2)", "hybrid (rrf)"]
        .iter()
        .zip(totals)
    {
        println!("  {label:<26} {:.4}", total / qrels.len() as f64);
    }
    let _ = std::fs::remove_dir_all(&path);
}

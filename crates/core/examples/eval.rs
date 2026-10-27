//! Build an index over a benchmark dataset and report build time, QPS and recall@10.
//!
//!     cargo run --release --example eval -- data/sift-128-euclidean
//!     cargo run --release --example eval -- data/glove-100-angular --index flat
//!     cargo run --release --example eval -- data/sift-128-euclidean --n 100000 --m 16 --efc 200 --ef 10,20,40,80
//!
//! `--n` uses the first N training vectors and computes exact ground truth for them.
//! HNSW QPS is single-threaded, one query at a time (the ann-benchmarks convention).

use std::time::Instant;

use vecengine::dataset::{recall_at_k, Dataset};
use vecengine::io::Vectors;
use vecengine::{FlatIndex, Hnsw, HnswParams};

const K: usize = 10;

struct Args {
    dir: String,
    index: String,
    n: Option<usize>,
    queries: usize,
    m: usize,
    efc: usize,
    ef: Vec<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        dir: String::new(),
        index: "hnsw".into(),
        n: None,
        queries: usize::MAX,
        m: 16,
        efc: 200,
        ef: vec![10, 20, 40, 80, 120, 200, 400, 800],
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--index" => a.index = val(),
            "--n" => a.n = Some(val().parse().expect("--n N")),
            "--queries" => a.queries = val().parse().expect("--queries N"),
            "--m" => a.m = val().parse().expect("--m N"),
            "--efc" => a.efc = val().parse().expect("--efc N"),
            "--ef" => a.ef = val().split(',').map(|s| s.parse().expect("--ef a,b,c")).collect(),
            _ => a.dir = arg,
        }
    }
    assert!(!a.dir.is_empty(), "usage: eval <dataset dir> [options]");
    a
}

fn main() {
    let args = parse_args();

    let t = Instant::now();
    let mut ds = Dataset::load(&args.dir).expect("failed to load dataset");
    let dim = ds.train.dim;
    println!(
        "{}: {} train, {} test, dim {dim}, {:?} (loaded in {:.1}s)",
        ds.name,
        ds.train.len(),
        ds.test.len(),
        ds.metric,
        t.elapsed().as_secs_f64()
    );

    let nq = ds.test.len().min(args.queries);
    let queries = ds.test.data[..nq * dim].to_vec();

    if let Some(n) = args.n.filter(|&n| n < ds.train.len()) {
        ds.train.data.truncate(n * dim);
        let t = Instant::now();
        let mut flat = FlatIndex::new(dim, ds.metric);
        flat.add_batch(&ds.train.data);
        let truth = flat.search_batch(&queries, 100);
        ds.neighbors = Vectors {
            dim: 100,
            data: truth.iter().flat_map(|r| r.iter().map(|n| n.id)).collect(),
        };
        println!(
            "using first {n} vectors, ground truth computed in {:.1}s",
            t.elapsed().as_secs_f64()
        );
    }

    match args.index.as_str() {
        "flat" => {
            let mut index = FlatIndex::new(dim, ds.metric);
            index.add_batch(&ds.train.data);
            let t = Instant::now();
            let results = index.search_batch(&queries, K);
            let secs = t.elapsed().as_secs_f64();
            println!(
                "flat: {nq} queries in {secs:.2}s ({:.0} QPS, all threads), recall@{K} = {:.4}",
                nq as f64 / secs,
                recall_at_k(&results, &ds.neighbors, K)
            );
        }
        "hnsw" => {
            let params = HnswParams {
                m: args.m,
                ef_construction: args.efc,
                ..Default::default()
            };
            let t = Instant::now();
            let mut index = Hnsw::new(dim, ds.metric, params);
            index.add_batch(&ds.train.data);
            println!(
                "hnsw M={} efC={}: built {} vectors in {:.1}s, {} levels",
                args.m,
                args.efc,
                index.len(),
                t.elapsed().as_secs_f64(),
                index.max_level() + 1
            );

            println!("{:>6} {:>10} {:>10}", "ef", "recall@10", "QPS");
            for &ef in &args.ef {
                let t = Instant::now();
                let results: Vec<_> = queries
                    .chunks_exact(dim)
                    .map(|q| index.search(q, K, ef))
                    .collect();
                let secs = t.elapsed().as_secs_f64();
                println!(
                    "{ef:>6} {:>10.4} {:>10.0}",
                    recall_at_k(&results, &ds.neighbors, K),
                    nq as f64 / secs
                );
            }
        }
        other => panic!("unknown index {other:?}, expected flat or hnsw"),
    }
}

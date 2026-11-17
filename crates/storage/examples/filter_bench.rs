//! Filtered search: graph walk vs exact scan across filter selectivities.
//!
//!     cargo run --release -p vecengine-storage --example filter_bench -- data/sift-128-euclidean
//!
//! Builds a single-segment collection from the first N vectors of a dataset, gives
//! each point a payload {"bucket": 0..999} (hashed from its id), then for filters
//! matching 0.1%..100% of points measures recall@10 against the exact filtered
//! result and single-threaded QPS for each strategy.

use std::time::Instant;

use vecengine::dataset::Dataset;
use vecengine_storage::{Collection, Config, Filter, FilterStrategy, Record};

const K: usize = 10;
const EF: usize = 100;
const QUERIES: usize = 500;

fn bucket(id: u64) -> u64 {
    let mut z = id.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    (z ^ (z >> 31)) % 1000
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: filter_bench <dataset dir> [n]");
    let n: usize = args.next().map_or(200_000, |s| s.parse().expect("n"));

    let ds = Dataset::load(&dir).expect("failed to load dataset");
    let dim = ds.train.dim;
    let n = n.min(ds.train.len());

    let path = std::env::temp_dir().join(format!("vecengine-filter-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    let config = Config {
        flush_threshold: usize::MAX,
        sync: false,
        ..Config::new(dim, ds.metric)
    };
    let mut c = Collection::open(&path, config).unwrap();
    let t = Instant::now();
    for start in (0..n).step_by(20_000) {
        let records = (start..(start + 20_000).min(n))
            .map(|i| Record::Upsert {
                id: i as u64,
                vector: ds.train.row(i).to_vec(),
                payload: format!(r#"{{"bucket":{}}}"#, bucket(i as u64)).into_bytes(),
            })
            .collect();
        c.write(records).unwrap();
    }
    c.flush().unwrap();
    println!(
        "{}: {n} vectors in one segment, built in {:.1}s; ef={EF}, {QUERIES} queries, 1 thread\n",
        ds.name,
        t.elapsed().as_secs_f64()
    );

    let queries: Vec<&[f32]> = (0..QUERIES).map(|i| ds.test.row(i)).collect();
    let run = |filter: &Filter, strategy| {
        let t = Instant::now();
        let res: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| {
                c.search_filtered(q, K, EF, Some(filter), strategy)
                    .iter()
                    .map(|h| h.id)
                    .collect()
            })
            .collect();
        (res, QUERIES as f64 / t.elapsed().as_secs_f64())
    };

    println!(
        "{:>9} {:>9} | {:>12} {:>8} | {:>8} | {:>12} {:>8}",
        "selected", "points", "graph recall", "QPS", "scan QPS", "auto recall", "QPS"
    );
    for permille in [1u64, 2, 5, 10, 20, 50, 100, 200, 500, 1000] {
        let filter = Filter::parse(&format!(r#"{{"field":"bucket","range":{{"lt":{permille}}}}}"#)).unwrap();
        let matching = (0..n as u64).filter(|&i| bucket(i) < permille).count();

        let (truth, scan_qps) = run(&filter, FilterStrategy::BruteForce);
        let recall = |res: &[Vec<u64>]| {
            let hits: usize = res
                .iter()
                .zip(&truth)
                .map(|(r, t)| r.iter().filter(|id| t.contains(id)).count())
                .sum();
            let possible: usize = truth.iter().map(Vec::len).sum();
            hits as f64 / possible as f64
        };
        let (graph, graph_qps) = run(&filter, FilterStrategy::Graph);
        let (auto, auto_qps) = run(&filter, FilterStrategy::Auto);
        println!(
            "{:>8.1}% {matching:>9} | {:>12.4} {graph_qps:>8.0} | {scan_qps:>8.0} | {:>12.4} {auto_qps:>8.0}",
            permille as f64 / 10.0,
            recall(&graph),
            recall(&auto),
        );
    }
    let _ = std::fs::remove_dir_all(&path);
}

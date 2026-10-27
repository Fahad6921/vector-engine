//! Build an index over a benchmark dataset and report build time, QPS and recall@10.
//!
//!     cargo run --release --example eval -- data/sift-128-euclidean
//!     cargo run --release --example eval -- data/glove-100-angular --queries 1000

use std::time::Instant;

use vecengine::dataset::{recall_at_k, Dataset};
use vecengine::FlatIndex;

const K: usize = 10;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    let mut max_queries = usize::MAX;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--queries" => max_queries = args.next().and_then(|s| s.parse().ok()).expect("--queries N"),
            _ => dir = Some(arg),
        }
    }
    let dir = dir.expect("usage: eval <dataset dir> [--queries N]");

    let t = Instant::now();
    let ds = Dataset::load(&dir).expect("failed to load dataset");
    println!(
        "{}: {} train, {} test, dim {}, {:?} (loaded in {:.1}s)",
        ds.name,
        ds.train.len(),
        ds.test.len(),
        ds.train.dim,
        ds.metric,
        t.elapsed().as_secs_f64()
    );

    let nq = ds.test.len().min(max_queries);
    let queries = &ds.test.data[..nq * ds.test.dim];

    let t = Instant::now();
    let mut index = FlatIndex::new(ds.train.dim, ds.metric);
    index.add_batch(&ds.train.data);
    println!("flat: built in {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let results = index.search_batch(queries, K);
    let secs = t.elapsed().as_secs_f64();
    println!(
        "flat: {nq} queries in {secs:.2}s ({:.0} QPS, all threads), recall@{K} = {:.4}",
        nq as f64 / secs,
        recall_at_k(&results, &ds.neighbors, K)
    );
}

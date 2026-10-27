//! Benchmark datasets in the layout produced by `bench/download.py`, plus recall.

use std::io;
use std::path::Path;

use crate::distance::Metric;
use crate::io::{read_fvecs, read_ivecs, Vectors};
use crate::neighbor::Neighbor;

pub struct Dataset {
    pub name: String,
    pub metric: Metric,
    pub train: Vectors<f32>,
    pub test: Vectors<f32>,
    /// Ground-truth neighbour ids for each test vector, closest first.
    pub neighbors: Vectors<u32>,
}

impl Dataset {
    pub fn load(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref();
        let metric_name = std::fs::read_to_string(dir.join("metric"))?;
        let metric = Metric::from_name(&metric_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown metric {metric_name:?}"),
            )
        })?;
        Ok(Self {
            name: dir
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            metric,
            train: read_fvecs(dir.join("train.fvecs"))?,
            test: read_fvecs(dir.join("test.fvecs"))?,
            neighbors: read_ivecs(dir.join("neighbors.ivecs"))?,
        })
    }
}

/// Mean fraction of the true top-k found in each result list.
pub fn recall_at_k(results: &[Vec<Neighbor>], truth: &Vectors<u32>, k: usize) -> f64 {
    assert!(
        k <= truth.dim,
        "ground truth only has {} neighbours per query",
        truth.dim
    );
    let mut hits = 0usize;
    for (i, res) in results.iter().enumerate() {
        let expected = &truth.row(i)[..k];
        hits += res.iter().take(k).filter(|n| expected.contains(&n.id)).count();
    }
    hits as f64 / (results.len() * k) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_counts_overlap() {
        let truth = Vectors {
            dim: 3,
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let results = vec![
            vec![Neighbor::new(1, 0.0), Neighbor::new(9, 0.0)],
            vec![Neighbor::new(5, 0.0), Neighbor::new(4, 0.0)],
        ];
        assert_eq!(recall_at_k(&results, &truth, 2), 0.75);
    }
}

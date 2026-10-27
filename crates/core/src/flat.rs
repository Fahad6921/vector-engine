//! Exact brute-force search. Used as ground truth and as the fallback for
//! highly selective filters.

use rayon::prelude::*;

use crate::distance::{normalize, Metric};
use crate::neighbor::{Neighbor, TopK};

pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    data: Vec<f32>,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: Metric) -> Self {
        assert!(dim > 0, "dimension must be positive");
        Self {
            dim,
            metric,
            data: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Adds one vector and returns its id.
    pub fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim, "vector has wrong dimension");
        let id = self.len() as u32;
        let start = self.data.len();
        self.data.extend_from_slice(v);
        if self.metric.needs_normalize() {
            normalize(&mut self.data[start..]);
        }
        id
    }

    /// Adds a row-major block of vectors. Ids continue from the current length.
    pub fn add_batch(&mut self, data: &[f32]) {
        assert_eq!(data.len() % self.dim, 0, "batch is not a whole number of vectors");
        self.data.reserve(data.len());
        for v in data.chunks_exact(self.dim) {
            self.add(v);
        }
    }

    pub fn vector(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.data[i..i + self.dim]
    }

    /// The `k` nearest vectors to `query`, closest first.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dim, "query has wrong dimension");
        let mut q = query.to_vec();
        if self.metric.needs_normalize() {
            normalize(&mut q);
        }
        let mut top = TopK::new(k);
        for (id, v) in self.data.chunks_exact(self.dim).enumerate() {
            top.push(Neighbor::new(id as u32, self.metric.distance(&q, v)));
        }
        top.into_sorted_vec()
    }

    /// Searches many queries in parallel. `queries` is row-major.
    ///
    /// Searching one query at a time streams the whole dataset from RAM per query,
    /// which makes it memory-bandwidth bound. Instead each task takes a block of
    /// queries and walks the data in cache-sized blocks, scoring every query in the
    /// block before moving on, so each byte of data is loaded once per query block.
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Vec<Vec<Neighbor>> {
        const QUERY_BLOCK: usize = 32;
        const DATA_BLOCK: usize = 2048;

        assert_eq!(
            queries.len() % self.dim,
            0,
            "queries are not a whole number of vectors"
        );
        queries
            .par_chunks(QUERY_BLOCK * self.dim)
            .flat_map_iter(|block| {
                let qs: Vec<Vec<f32>> = block
                    .chunks_exact(self.dim)
                    .map(|q| {
                        let mut q = q.to_vec();
                        if self.metric.needs_normalize() {
                            normalize(&mut q);
                        }
                        q
                    })
                    .collect();
                let mut tops: Vec<TopK> = qs.iter().map(|_| TopK::new(k)).collect();
                for (b, data) in self.data.chunks(DATA_BLOCK * self.dim).enumerate() {
                    let base = b * DATA_BLOCK;
                    for (q, top) in qs.iter().zip(&mut tops) {
                        for (j, v) in data.chunks_exact(self.dim).enumerate() {
                            top.push(Neighbor::new((base + j) as u32, self.metric.distance(q, v)));
                        }
                    }
                }
                tops.into_iter().map(TopK::into_sorted_vec)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_exact_neighbours() {
        let mut index = FlatIndex::new(2, Metric::L2);
        index.add_batch(&[0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 0.9, 0.1]);
        let res = index.search(&[1.0, 0.0], 2);
        assert_eq!(res[0], Neighbor::new(1, 0.0));
        assert_eq!(res[1].id, 3);
    }

    #[test]
    fn cosine_ignores_magnitude() {
        let mut index = FlatIndex::new(2, Metric::Cosine);
        index.add(&[100.0, 0.0]);
        index.add(&[0.0, 1.0]);
        let res = index.search(&[0.5, 0.0], 1);
        assert_eq!(res[0].id, 0);
        assert!(res[0].distance.abs() < 1e-6);
    }

    #[test]
    fn batch_matches_single() {
        let mut index = FlatIndex::new(3, Metric::InnerProduct);
        let data: Vec<f32> = (0..300).map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0).collect();
        index.add_batch(&data);
        let queries = &data[..30];
        let batch = index.search_batch(queries, 5);
        for (q, res) in queries.as_chunks::<3>().0.iter().zip(&batch) {
            assert_eq!(&index.search(q, 5), res);
        }
    }
}

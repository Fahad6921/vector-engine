//! Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016).
//!
//! Every node lives on layer 0; each layer above holds an exponentially smaller
//! random subset. Search greedily descends the sparse upper layers to find a good
//! entry point, then runs a beam search of width `ef` on layer 0.
//!
//! Neighbour lists are stored in flat fixed-stride arrays (`[count, ids...]`)
//! rather than per-node `Vec`s: no pointer chasing, and the layout can be written
//! to disk and mmapped as-is.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::distance::{normalize, Metric};
use crate::neighbor::Neighbor;

#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Max neighbours per node on upper layers. Layer 0 allows `2 * m`.
    pub m: usize,
    /// Beam width while building. Higher builds a better graph, slower.
    pub ef_construction: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            seed: 0x5eed,
        }
    }
}

pub struct Hnsw {
    dim: usize,
    metric: Metric,
    params: HnswParams,
    /// Max neighbours on layer 0.
    m0: usize,
    /// Level multiplier, 1 / ln(m).
    ml: f64,

    data: Vec<f32>,
    levels: Vec<u8>,
    /// Layer 0 links, stride `m0 + 1`.
    links0: Vec<u32>,
    /// Links for layers 1..=level of each node, stride `m + 1` per layer.
    upper: Vec<Vec<u32>>,

    entry: Option<u32>,
    max_level: usize,
    rng: SplitMix64,
    visited: Mutex<Vec<Visited>>,
}

impl Hnsw {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert!(params.m >= 2, "m must be at least 2");
        Self {
            dim,
            metric,
            params,
            m0: 2 * params.m,
            ml: 1.0 / (params.m as f64).ln(),
            data: Vec::new(),
            levels: Vec::new(),
            links0: Vec::new(),
            upper: Vec::new(),
            entry: None,
            max_level: 0,
            rng: SplitMix64(params.seed),
            visited: Mutex::new(Vec::new()),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn params(&self) -> HnswParams {
        self.params
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn max_level(&self) -> usize {
        self.max_level
    }

    pub fn vector(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.data[i..i + self.dim]
    }

    /// Adds a row-major block of vectors. Ids continue from the current length.
    pub fn add_batch(&mut self, data: &[f32]) {
        assert_eq!(data.len() % self.dim, 0, "batch is not a whole number of vectors");
        for v in data.chunks_exact(self.dim) {
            self.add(v);
        }
    }

    /// Inserts one vector and returns its id.
    pub fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim, "vector has wrong dimension");
        let id = self.len() as u32;
        let level = self.random_level();

        let start = self.data.len();
        self.data.extend_from_slice(v);
        if self.metric.needs_normalize() {
            normalize(&mut self.data[start..]);
        }
        self.levels.push(level as u8);
        self.links0.resize(self.links0.len() + self.m0 + 1, 0);
        self.upper.push(vec![0; level * (self.params.m + 1)]);

        let Some(mut ep) = self.entry else {
            self.entry = Some(id);
            self.max_level = level;
            return id;
        };

        let q = self.vector(id).to_vec();
        let mut ep_dist = self.dist(&q, ep);
        for layer in (level + 1..=self.max_level).rev() {
            (ep, ep_dist) = self.greedy_closest(&q, ep, ep_dist, layer);
        }

        let mut eps = vec![Neighbor::new(ep, ep_dist)];
        for layer in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(&q, &eps, self.params.ef_construction, layer);
            let selected = self.select_neighbors(&candidates, self.params.m);
            let ids: Vec<u32> = selected.iter().map(|n| n.id).collect();
            self.set_links(id, layer, &ids);
            for n in &selected {
                self.connect(n.id, id, n.distance, layer);
            }
            eps = candidates;
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(id);
        }
        id
    }

    /// The `k` approximate nearest neighbours of `query`, closest first.
    /// `ef` is the search beam width; it is raised to `k` if smaller.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dim, "query has wrong dimension");
        let Some(mut ep) = self.entry else {
            return Vec::new();
        };
        let mut q = query.to_vec();
        if self.metric.needs_normalize() {
            normalize(&mut q);
        }

        let mut ep_dist = self.dist(&q, ep);
        for layer in (1..=self.max_level).rev() {
            (ep, ep_dist) = self.greedy_closest(&q, ep, ep_dist, layer);
        }
        let mut res = self.search_layer(&q, &[Neighbor::new(ep, ep_dist)], ef.max(k), 0);
        res.truncate(k);
        res
    }

    /// Searches many queries in parallel. `queries` is row-major.
    pub fn search_batch(&self, queries: &[f32], k: usize, ef: usize) -> Vec<Vec<Neighbor>> {
        queries
            .par_chunks_exact(self.dim)
            .map(|q| self.search(q, k, ef))
            .collect()
    }

    fn random_level(&mut self) -> usize {
        // floor(-ln(U) * ml), U in (0, 1]
        let u = 1.0 - self.rng.next_f64();
        ((-u.ln() * self.ml) as usize).min(u8::MAX as usize)
    }

    #[inline]
    fn dist(&self, q: &[f32], id: u32) -> f32 {
        self.metric.distance(q, self.vector(id))
    }

    #[inline]
    fn links(&self, id: u32, layer: usize) -> &[u32] {
        let list = if layer == 0 {
            let base = id as usize * (self.m0 + 1);
            &self.links0[base..base + self.m0 + 1]
        } else {
            let stride = self.params.m + 1;
            let base = (layer - 1) * stride;
            &self.upper[id as usize][base..base + stride]
        };
        &list[1..1 + list[0] as usize]
    }

    fn set_links(&mut self, id: u32, layer: usize, ids: &[u32]) {
        let list = if layer == 0 {
            let base = id as usize * (self.m0 + 1);
            &mut self.links0[base..base + self.m0 + 1]
        } else {
            let stride = self.params.m + 1;
            let base = (layer - 1) * stride;
            &mut self.upper[id as usize][base..base + stride]
        };
        debug_assert!(ids.len() < list.len());
        list[0] = ids.len() as u32;
        list[1..1 + ids.len()].copy_from_slice(ids);
    }

    /// Adds a link `from -> to`. If `from` is already full, its list is re-pruned
    /// with the selection heuristic, which may drop the new link or an old one.
    fn connect(&mut self, from: u32, to: u32, distance: f32, layer: usize) {
        let cap = if layer == 0 { self.m0 } else { self.params.m };
        let current = self.links(from, layer);
        if current.len() < cap {
            let mut ids = current.to_vec();
            ids.push(to);
            self.set_links(from, layer, &ids);
            return;
        }

        let base = self.vector(from);
        let mut candidates: Vec<Neighbor> = current
            .iter()
            .map(|&id| Neighbor::new(id, self.metric.distance(base, self.vector(id))))
            .collect();
        candidates.push(Neighbor::new(to, distance));
        candidates.sort_unstable();
        let ids: Vec<u32> = self
            .select_neighbors(&candidates, cap)
            .iter()
            .map(|n| n.id)
            .collect();
        self.set_links(from, layer, &ids);
    }

    /// Neighbour selection heuristic (Algorithm 4 in the paper).
    ///
    /// Walk candidates closest first and keep one only if it is closer to the base
    /// node than to every neighbour already kept. This favours links that point in
    /// different directions over a tight cluster of near-duplicates, which keeps the
    /// graph navigable between clusters. `candidates` must be sorted closest first.
    fn select_neighbors(&self, candidates: &[Neighbor], m: usize) -> Vec<Neighbor> {
        let mut kept: Vec<Neighbor> = Vec::with_capacity(m);
        for c in candidates {
            if kept.len() >= m {
                break;
            }
            let v = self.vector(c.id);
            if kept
                .iter()
                .all(|k| self.metric.distance(v, self.vector(k.id)) > c.distance)
            {
                kept.push(*c);
            }
        }
        kept
    }

    /// Greedy walk on one layer: move to the closest neighbour until none is closer.
    fn greedy_closest(&self, q: &[f32], mut ep: u32, mut ep_dist: f32, layer: usize) -> (u32, f32) {
        loop {
            let mut moved = false;
            for &n in self.links(ep, layer) {
                let d = self.dist(q, n);
                if d < ep_dist {
                    ep = n;
                    ep_dist = d;
                    moved = true;
                }
            }
            if !moved {
                return (ep, ep_dist);
            }
        }
    }

    /// Beam search on one layer (Algorithm 2). Returns up to `ef` nodes, closest first.
    fn search_layer(&self, q: &[f32], eps: &[Neighbor], ef: usize, layer: usize) -> Vec<Neighbor> {
        let mut visited = self.visited.lock().unwrap().pop().unwrap_or_default();
        visited.reset(self.len());

        // Min-heap of nodes to expand; max-heap of the best `ef` found so far.
        let mut candidates: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(ef + 1);
        for &ep in eps {
            visited.insert(ep.id);
            candidates.push(Reverse(ep));
            results.push(ep);
            if results.len() > ef {
                results.pop();
            }
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
            if c.distance > worst && results.len() >= ef {
                break;
            }
            for &n in self.links(c.id, layer) {
                if !visited.insert(n) {
                    continue;
                }
                let d = self.dist(q, n);
                let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
                if results.len() < ef || d < worst {
                    let nb = Neighbor::new(n, d);
                    candidates.push(Reverse(nb));
                    results.push(nb);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        self.visited.lock().unwrap().push(visited);
        results.into_sorted_vec()
    }
}

/// Visited set as a generation-stamped array: resetting is O(1) (bump the tag)
/// instead of clearing a HashSet or a bitmap per search.
#[derive(Default)]
struct Visited {
    marks: Vec<u32>,
    tag: u32,
}

impl Visited {
    fn reset(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.tag = self.tag.wrapping_add(1);
        if self.tag == 0 {
            self.marks.fill(0);
            self.tag = 1;
        }
    }

    /// Returns true if `id` was not visited yet.
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        let m = &mut self.marks[id as usize];
        if *m == self.tag {
            false
        } else {
            *m = self.tag;
            true
        }
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatIndex;

    fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64(seed);
        (0..n * dim).map(|_| rng.next_f64() as f32 * 2.0 - 1.0).collect()
    }

    fn recall(metric: Metric) -> f64 {
        let (n, dim, k) = (3000, 16, 10);
        let data = random_data(n, dim, 1);
        let queries = random_data(100, dim, 2);

        let mut flat = FlatIndex::new(dim, metric);
        flat.add_batch(&data);
        let mut hnsw = Hnsw::new(dim, metric, HnswParams::default());
        hnsw.add_batch(&data);

        let mut hits = 0;
        for q in queries.chunks_exact(dim) {
            let truth: Vec<u32> = flat.search(q, k).iter().map(|n| n.id).collect();
            hits += hnsw
                .search(q, k, 64)
                .iter()
                .filter(|n| truth.contains(&n.id))
                .count();
        }
        hits as f64 / (100 * k) as f64
    }

    #[test]
    fn high_recall_l2() {
        let r = recall(Metric::L2);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn high_recall_cosine() {
        let r = recall(Metric::Cosine);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn empty_and_tiny_indexes() {
        let mut index = Hnsw::new(2, Metric::L2, HnswParams::default());
        assert!(index.search(&[0.0, 0.0], 5, 10).is_empty());
        index.add(&[1.0, 1.0]);
        index.add(&[2.0, 2.0]);
        let res = index.search(&[0.0, 0.0], 5, 10);
        assert_eq!(res.iter().map(|n| n.id).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn neighbour_lists_respect_caps() {
        let dim = 8;
        let mut index = Hnsw::new(
            dim,
            Metric::L2,
            HnswParams {
                m: 4,
                ..Default::default()
            },
        );
        index.add_batch(&random_data(2000, dim, 3));
        for id in 0..index.len() as u32 {
            assert!(index.links(id, 0).len() <= 8);
            for layer in 1..=index.levels[id as usize] as usize {
                assert!(index.links(id, layer).len() <= 4);
            }
        }
    }

    #[test]
    fn level_distribution_is_geometric() {
        let mut index = Hnsw::new(4, Metric::L2, HnswParams::default());
        index.add_batch(&random_data(20_000, 4, 4));
        let above0 = index.levels.iter().filter(|&&l| l > 0).count() as f64;
        // P(level > 0) = 1/m = 1/16
        let expected = 20_000.0 / 16.0;
        assert!(
            (above0 - expected).abs() < expected * 0.15,
            "{above0} vs {expected}"
        );
    }
}

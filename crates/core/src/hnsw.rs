//! Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016).
//!
//! Every node lives on layer 0; each layer above holds an exponentially smaller
//! random subset. Search greedily descends the sparse upper layers to find a good
//! entry point, then runs a beam search of width `ef` on layer 0.
//!
//! Neighbour lists are stored in flat fixed-stride arrays (`[count, ids...]`)
//! rather than per-node `Vec`s: no pointer chasing, and the layout can be written
//! to disk and mmapped as-is.
//!
//! Builds run in parallel. Each node has its own lock, taken to read or modify its
//! neighbour list during a build. The lists are atomics so that `search`, which can
//! only run when no build is in progress (`&self` vs `&mut self`), reads them
//! without locking; relaxed atomic loads compile to plain loads.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use parking_lot::Mutex;
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

#[derive(Clone, Copy)]
struct Entry {
    id: u32,
    level: usize,
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
    links0: Vec<AtomicU32>,
    /// Links for layers 1..=level of every node, concatenated. Node `i` owns
    /// `upper[upper_offsets[i]..upper_offsets[i + 1]]`, stride `m + 1` per layer.
    upper: Vec<AtomicU32>,
    upper_offsets: Vec<u64>,
    /// One lock per node, guarding its neighbour lists during builds.
    locks: Vec<Mutex<()>>,

    entry: Option<Entry>,
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
            upper_offsets: vec![0],
            locks: Vec::new(),
            entry: None,
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
        self.entry.map_or(0, |e| e.level)
    }

    pub fn vector(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.data[i..i + self.dim]
    }

    /// Inserts one vector and returns its id.
    pub fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim, "vector has wrong dimension");
        let id = self.len() as u32;
        self.add_batch(v);
        id
    }

    /// Adds a row-major block of vectors, inserting them in parallel.
    /// Ids continue from the current length.
    pub fn add_batch(&mut self, data: &[f32]) {
        assert_eq!(data.len() % self.dim, 0, "batch is not a whole number of vectors");
        let start = self.len();
        let n = data.len() / self.dim;
        if n == 0 {
            return;
        }

        // Allocate everything up front so the parallel phase only needs &self.
        // Levels are drawn sequentially, so they stay deterministic for a seed.
        self.data.reserve(data.len());
        for v in data.chunks_exact(self.dim) {
            let s = self.data.len();
            self.data.extend_from_slice(v);
            if self.metric.needs_normalize() {
                normalize(&mut self.data[s..]);
            }
            let level = self.random_level();
            self.levels.push(level as u8);
            let end = self.upper.len() + level * (self.params.m + 1);
            self.upper.resize_with(end, || AtomicU32::new(0));
            self.upper_offsets.push(end as u64);
            self.locks.push(Mutex::new(()));
        }
        self.links0
            .resize_with((start + n) * (self.m0 + 1), || AtomicU32::new(0));

        let mut first = start;
        let entry = *self.entry.get_or_insert_with(|| {
            first += 1;
            Entry {
                id: start as u32,
                level: self.levels[start] as usize,
            }
        });

        let entry = Mutex::new(entry);
        let this = &*self;
        (first..start + n)
            .into_par_iter()
            .for_each(|id| this.insert(id as u32, &entry));
        self.entry = Some(entry.into_inner());
    }

    /// Links an already-allocated node into the graph (Algorithm 1).
    fn insert(&self, id: u32, entry: &Mutex<Entry>) {
        let level = self.levels[id as usize] as usize;

        // A node that will become the new top level holds the entry lock for its
        // whole insert, so no one else can see a half-linked entry point.
        let mut guard = Some(entry.lock());
        let top = **guard.as_ref().unwrap();
        if level <= top.level {
            guard = None;
        }

        let q = self.vector(id);
        let mut buf = Vec::with_capacity(self.m0);
        let (mut ep, mut ep_dist) = (top.id, self.dist(q, top.id));
        for layer in (level + 1..=top.level).rev() {
            (ep, ep_dist) = self.greedy_closest(q, ep, ep_dist, layer, true, &mut buf);
        }

        let mut eps = vec![Neighbor::new(ep, ep_dist)];
        for layer in (0..=level.min(top.level)).rev() {
            let mut candidates = self.search_layer(q, &eps, self.params.ef_construction, layer, true);
            candidates.retain(|c| c.id != id);
            let selected = self.select_neighbors(&candidates, self.params.m);
            {
                let _lock = self.locks[id as usize].lock();
                let ids: Vec<u32> = selected.iter().map(|n| n.id).collect();
                self.write_links(id, layer, &ids);
            }
            for n in &selected {
                self.connect(n.id, id, n.distance, layer);
            }
            eps = candidates;
        }

        if let Some(mut g) = guard {
            *g = Entry { id, level };
        }
    }

    /// The `k` approximate nearest neighbours of `query`, closest first.
    /// `ef` is the search beam width; it is raised to `k` if smaller.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbor> {
        self.search_filtered(query, k, ef, |_| true)
    }

    /// Like `search`, but only ids for which `allowed(id)` is true are returned.
    /// Used for deletes (tombstones) and metadata filters.
    pub fn search_filtered<F: Fn(u32) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowed: F,
    ) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dim, "query has wrong dimension");
        let Some(top) = self.entry else {
            return Vec::new();
        };
        let mut q = query.to_vec();
        if self.metric.needs_normalize() {
            normalize(&mut q);
        }

        let mut buf = Vec::with_capacity(self.m0);
        let (mut ep, mut ep_dist) = (top.id, self.dist(&q, top.id));
        for layer in (1..=top.level).rev() {
            (ep, ep_dist) = self.greedy_closest(&q, ep, ep_dist, layer, false, &mut buf);
        }
        let eps = [Neighbor::new(ep, ep_dist)];
        let mut res = self.search_layer_filtered(&q, &eps, ef.max(k), 0, false, allowed);
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

    /// The raw `[count, ids...]` slot for a node on a layer.
    #[inline]
    fn list(&self, id: u32, layer: usize) -> &[AtomicU32] {
        if layer == 0 {
            let base = id as usize * (self.m0 + 1);
            &self.links0[base..base + self.m0 + 1]
        } else {
            let stride = self.params.m + 1;
            let base = self.upper_offsets[id as usize] as usize + (layer - 1) * stride;
            &self.upper[base..base + stride]
        }
    }

    /// Copies a node's neighbours into `buf`, taking the node's lock if `lock`.
    #[inline]
    fn read_links(&self, id: u32, layer: usize, lock: bool, buf: &mut Vec<u32>) {
        let list = self.list(id, layer);
        let _lock = lock.then(|| self.locks[id as usize].lock());
        let n = list[0].load(Relaxed) as usize;
        buf.clear();
        buf.extend(list[1..1 + n].iter().map(|a| a.load(Relaxed)));
    }

    /// Overwrites a node's neighbours. Caller must hold the node's lock.
    fn write_links(&self, id: u32, layer: usize, ids: &[u32]) {
        let list = self.list(id, layer);
        debug_assert!(ids.len() < list.len());
        for (slot, &n) in list[1..].iter().zip(ids) {
            slot.store(n, Relaxed);
        }
        list[0].store(ids.len() as u32, Relaxed);
    }

    /// Adds a link `from -> to`. If `from` is already full, its list is re-pruned
    /// with the selection heuristic, which may drop the new link or an old one.
    fn connect(&self, from: u32, to: u32, distance: f32, layer: usize) {
        let cap = if layer == 0 { self.m0 } else { self.params.m };
        let _lock = self.locks[from as usize].lock();
        let list = self.list(from, layer);
        let n = list[0].load(Relaxed) as usize;
        if n < cap {
            list[1 + n].store(to, Relaxed);
            list[0].store(n as u32 + 1, Relaxed);
            return;
        }

        let base = self.vector(from);
        let mut candidates: Vec<Neighbor> = list[1..1 + n]
            .iter()
            .map(|a| {
                let id = a.load(Relaxed);
                Neighbor::new(id, self.metric.distance(base, self.vector(id)))
            })
            .collect();
        candidates.push(Neighbor::new(to, distance));
        candidates.sort_unstable();
        let ids: Vec<u32> = self
            .select_neighbors(&candidates, cap)
            .iter()
            .map(|n| n.id)
            .collect();
        self.write_links(from, layer, &ids);
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
    fn greedy_closest(
        &self,
        q: &[f32],
        mut ep: u32,
        mut ep_dist: f32,
        layer: usize,
        lock: bool,
        buf: &mut Vec<u32>,
    ) -> (u32, f32) {
        loop {
            let mut moved = false;
            self.read_links(ep, layer, lock, buf);
            for &n in buf.iter() {
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
    fn search_layer(
        &self,
        q: &[f32],
        eps: &[Neighbor],
        ef: usize,
        layer: usize,
        lock: bool,
    ) -> Vec<Neighbor> {
        self.search_layer_filtered(q, eps, ef, layer, lock, |_| true)
    }

    /// Beam search where only nodes passing `allowed` can enter the results.
    ///
    /// Rejected nodes are still expanded: they are part of the graph's paths, and
    /// skipping them would disconnect regions of allowed nodes. With very few
    /// allowed nodes the results rarely fill, so the search ends up walking most of
    /// the reachable graph; callers should fall back to brute force in that case.
    fn search_layer_filtered<F: Fn(u32) -> bool>(
        &self,
        q: &[f32],
        eps: &[Neighbor],
        ef: usize,
        layer: usize,
        lock: bool,
        allowed: F,
    ) -> Vec<Neighbor> {
        let mut visited = self.visited.lock().pop().unwrap_or_default();
        visited.reset(self.len());
        let mut buf = Vec::with_capacity(self.m0);

        // Min-heap of nodes to expand; max-heap of the best `ef` allowed nodes so far.
        let mut candidates: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(ef + 1);
        for &ep in eps {
            visited.insert(ep.id);
            candidates.push(Reverse(ep));
            if allowed(ep.id) {
                results.push(ep);
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
            if c.distance > worst && results.len() >= ef {
                break;
            }
            self.read_links(c.id, layer, lock, &mut buf);
            for &n in &buf {
                if !visited.insert(n) {
                    continue;
                }
                let d = self.dist(q, n);
                let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
                if results.len() < ef || d < worst {
                    let nb = Neighbor::new(n, d);
                    candidates.push(Reverse(nb));
                    if allowed(n) {
                        results.push(nb);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        self.visited.lock().push(visited);
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

    fn recall(index: &Hnsw, flat: &FlatIndex, queries: &[f32], k: usize) -> f64 {
        let dim = index.dim();
        let mut hits = 0;
        for q in queries.chunks_exact(dim) {
            let truth: Vec<u32> = flat.search(q, k).iter().map(|n| n.id).collect();
            hits += index
                .search(q, k, 64)
                .iter()
                .filter(|n| truth.contains(&n.id))
                .count();
        }
        hits as f64 / (queries.len() / dim * k) as f64
    }

    fn build_and_measure(metric: Metric) -> f64 {
        let (n, dim) = (3000, 16);
        let data = random_data(n, dim, 1);
        let mut flat = FlatIndex::new(dim, metric);
        flat.add_batch(&data);
        let mut hnsw = Hnsw::new(dim, metric, HnswParams::default());
        hnsw.add_batch(&data);
        recall(&hnsw, &flat, &random_data(100, dim, 2), 10)
    }

    #[test]
    fn high_recall_l2() {
        let r = build_and_measure(Metric::L2);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn high_recall_cosine() {
        let r = build_and_measure(Metric::Cosine);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn incremental_batches_and_single_adds() {
        let dim = 16;
        let data = random_data(3000, dim, 5);
        let mut flat = FlatIndex::new(dim, Metric::L2);
        flat.add_batch(&data);

        let mut hnsw = Hnsw::new(dim, Metric::L2, HnswParams::default());
        let (a, rest) = data.split_at(1000 * dim);
        let (b, c) = rest.split_at(1000 * dim);
        hnsw.add_batch(a);
        for v in b.chunks_exact(dim) {
            hnsw.add(v);
        }
        hnsw.add_batch(c);
        assert_eq!(hnsw.len(), 3000);

        let r = recall(&hnsw, &flat, &random_data(100, dim, 6), 10);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn filtered_search_only_returns_allowed_ids() {
        let (n, dim, k) = (3000, 16, 10);
        let data = random_data(n, dim, 7);
        let mut hnsw = Hnsw::new(dim, Metric::L2, HnswParams::default());
        hnsw.add_batch(&data);

        let mut hits = 0;
        let queries = random_data(50, dim, 8);
        for q in queries.chunks_exact(dim) {
            let res = hnsw.search_filtered(q, k, 64, |id| id % 2 == 0);
            assert_eq!(res.len(), k);
            assert!(res.iter().all(|n| n.id % 2 == 0));

            let mut truth: Vec<Neighbor> = (0..n as u32)
                .step_by(2)
                .map(|id| Neighbor::new(id, Metric::L2.distance(q, hnsw.vector(id))))
                .collect();
            truth.sort_unstable();
            let truth: Vec<u32> = truth[..k].iter().map(|n| n.id).collect();
            hits += res.iter().filter(|n| truth.contains(&n.id)).count();
        }
        let recall = hits as f64 / (50 * k) as f64;
        assert!(recall > 0.95, "recall {recall}");
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
    fn neighbour_lists_respect_caps_and_have_no_self_links() {
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
        let mut buf = Vec::new();
        for id in 0..index.len() as u32 {
            for layer in 0..=index.levels[id as usize] as usize {
                index.read_links(id, layer, false, &mut buf);
                assert!(buf.len() <= if layer == 0 { 8 } else { 4 });
                assert!(!buf.contains(&id), "node {id} links to itself");
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

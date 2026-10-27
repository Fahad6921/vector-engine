use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A search result. Ordered by distance, then id, so results are deterministic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
}

impl Neighbor {
    pub fn new(id: u32, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Keeps the `k` closest neighbours seen so far.
///
/// A max-heap of size k: the root is the worst of the current best, so a new
/// candidate only needs one comparison against it. O(n log k) for n pushes.
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Neighbor>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    #[inline]
    pub fn push(&mut self, n: Neighbor) {
        if self.heap.len() < self.k {
            self.heap.push(n);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if n < *worst {
                *worst = n;
            }
        }
    }

    /// Distance a candidate must beat to get in. Infinite until the heap is full.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().map_or(f32::INFINITY, |n| n.distance)
        }
    }

    /// Results sorted closest first.
    pub fn into_sorted_vec(self) -> Vec<Neighbor> {
        self.heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_k_smallest() {
        let mut top = TopK::new(3);
        for (id, d) in [(0, 5.0), (1, 1.0), (2, 9.0), (3, 0.5), (4, 3.0), (5, 7.0)] {
            top.push(Neighbor::new(id, d));
        }
        let ids: Vec<u32> = top.into_sorted_vec().iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![3, 1, 4]);
    }

    #[test]
    fn k_larger_than_input() {
        let mut top = TopK::new(10);
        top.push(Neighbor::new(7, 2.0));
        assert_eq!(top.threshold(), f32::INFINITY);
        assert_eq!(top.into_sorted_vec().len(), 1);
    }

    #[test]
    fn ties_broken_by_id() {
        let mut top = TopK::new(2);
        for id in [9, 3, 5] {
            top.push(Neighbor::new(id, 1.0));
        }
        let ids: Vec<u32> = top.into_sorted_vec().iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![3, 5]);
    }
}

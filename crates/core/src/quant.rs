//! Vector quantization. Each vector gets a compact code; search walks the graph
//! using cheap approximate distances on the codes, then reranks the final
//! candidates with the full-precision vectors.
//!
//! - SQ8: one byte per dimension, scaled between the per-dimension min and max.
//! - PQ: the vector is split into `m` sub-vectors, each replaced by the id of its
//!   nearest of 256 k-means centroids. Distances are `m` table lookups (ADC).
//! - Binary: one bit per dimension, set if above that dimension's mean. Distance
//!   is Hamming (popcount). Very lossy; relies on reranking.

use rayon::prelude::*;

use crate::distance::{dot, l2_sq, Metric};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    Sq8,
    /// Product quantization with `m` sub-vectors; `dim` must be divisible by `m`.
    Pq {
        m: usize,
    },
    Binary,
}

impl QuantKind {
    /// Parses "sq8", "pq16", "binary".
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "sq8" => Some(QuantKind::Sq8),
            "binary" => Some(QuantKind::Binary),
            _ => s.strip_prefix("pq")?.parse().ok().map(|m| QuantKind::Pq { m }),
        }
    }
}

const KSUB: usize = 256;
const MAX_TRAIN: usize = 65_536;
const KMEANS_ITERS: usize = 20;

pub(crate) enum Quantizer {
    Sq8 {
        min: Vec<f32>,
        scale: Vec<f32>,
    },
    Pq {
        m: usize,
        dsub: usize,
        centroids: Vec<f32>,
    },
    Binary {
        mean: Vec<f32>,
    },
}

/// Per-query state prepared once, then used for every distance.
pub(crate) enum QueryLut {
    /// L2: `q - min`; dot: `q * scale` and the constant `q . min`.
    Sq8 {
        q: Vec<f32>,
        bias: f32,
    },
    /// Distance contribution of each centroid in each sub-space, `m x 256`.
    Pq {
        table: Vec<f32>,
    },
    Binary {
        bits: Vec<u64>,
    },
}

impl Quantizer {
    /// Trains on up to 65k of the given row-major vectors (evenly strided).
    pub fn train(kind: QuantKind, data: &[f32], dim: usize) -> Self {
        let n = data.len() / dim;
        assert!(n > 0, "need data to train a quantizer");
        let step = n.div_ceil(MAX_TRAIN);
        let sample: Vec<&[f32]> = data.chunks_exact(dim).step_by(step).collect();

        match kind {
            QuantKind::Sq8 => {
                let mut min = vec![f32::INFINITY; dim];
                let mut max = vec![f32::NEG_INFINITY; dim];
                for v in &sample {
                    for i in 0..dim {
                        min[i] = min[i].min(v[i]);
                        max[i] = max[i].max(v[i]);
                    }
                }
                let scale = min
                    .iter()
                    .zip(&max)
                    .map(|(lo, hi)| if hi > lo { (hi - lo) / 255.0 } else { 1.0 })
                    .collect();
                Quantizer::Sq8 { min, scale }
            }
            QuantKind::Pq { m } => {
                assert!(
                    m > 0 && dim.is_multiple_of(m),
                    "dim {dim} is not divisible by pq m={m}"
                );
                let dsub = dim / m;
                let centroids = (0..m)
                    .into_par_iter()
                    .flat_map_iter(|s| {
                        let points: Vec<f32> = sample
                            .iter()
                            .flat_map(|v| v[s * dsub..(s + 1) * dsub].iter().copied())
                            .collect();
                        kmeans(&points, dsub, KSUB, s as u64)
                    })
                    .collect();
                Quantizer::Pq { m, dsub, centroids }
            }
            QuantKind::Binary => {
                let mut mean = vec![0f32; dim];
                for v in &sample {
                    for i in 0..dim {
                        mean[i] += v[i];
                    }
                }
                mean.iter_mut().for_each(|x| *x /= sample.len() as f32);
                Quantizer::Binary { mean }
            }
        }
    }

    pub fn code_size(&self, dim: usize) -> usize {
        match self {
            Quantizer::Sq8 { .. } => dim,
            Quantizer::Pq { m, .. } => *m,
            Quantizer::Binary { .. } => dim.div_ceil(64) * 8,
        }
    }

    pub fn encode(&self, v: &[f32], out: &mut Vec<u8>) {
        match self {
            Quantizer::Sq8 { min, scale } => {
                out.extend(
                    v.iter()
                        .zip(min.iter().zip(scale))
                        .map(|(x, (lo, s))| ((x - lo) / s).round().clamp(0.0, 255.0) as u8),
                );
            }
            Quantizer::Pq { m, dsub, centroids } => {
                for s in 0..*m {
                    let sub = &v[s * dsub..(s + 1) * dsub];
                    let book = &centroids[s * KSUB * dsub..(s + 1) * KSUB * dsub];
                    out.push(nearest(sub, book, *dsub) as u8);
                }
            }
            Quantizer::Binary { mean } => {
                for w in binarize(v, mean) {
                    out.extend_from_slice(&w.to_le_bytes());
                }
            }
        }
    }

    pub fn prepare(&self, q: &[f32], metric: Metric) -> QueryLut {
        let use_l2 = metric == Metric::L2;
        match self {
            Quantizer::Sq8 { min, scale } => {
                if use_l2 {
                    QueryLut::Sq8 {
                        q: q.iter().zip(min).map(|(x, lo)| x - lo).collect(),
                        bias: 0.0,
                    }
                } else {
                    QueryLut::Sq8 {
                        q: q.iter().zip(scale).map(|(x, s)| x * s).collect(),
                        bias: dot(q, min),
                    }
                }
            }
            Quantizer::Pq { m, dsub, centroids } => {
                let mut table = Vec::with_capacity(m * KSUB);
                for s in 0..*m {
                    let sub = &q[s * dsub..(s + 1) * dsub];
                    for c in centroids[s * KSUB * dsub..(s + 1) * KSUB * dsub].chunks_exact(*dsub) {
                        table.push(if use_l2 { l2_sq(sub, c) } else { -dot(sub, c) });
                    }
                }
                QueryLut::Pq { table }
            }
            Quantizer::Binary { mean } => QueryLut::Binary {
                bits: binarize(q, mean),
            },
        }
    }

    /// Approximate distance; same ordering as `metric.distance` up to quantization error.
    #[inline]
    pub fn distance(&self, lut: &QueryLut, code: &[u8], metric: Metric) -> f32 {
        match (self, lut) {
            (Quantizer::Sq8 { scale, .. }, QueryLut::Sq8 { q, bias }) => {
                if metric == Metric::L2 {
                    sq8_l2(q, scale, code)
                } else {
                    -(bias + sq8_dot(q, code))
                }
            }
            (Quantizer::Pq { .. }, QueryLut::Pq { table }) => code
                .iter()
                .enumerate()
                .map(|(s, &c)| table[s * KSUB + c as usize])
                .sum(),
            (Quantizer::Binary { .. }, QueryLut::Binary { bits }) => code
                .as_chunks::<8>()
                .0
                .iter()
                .zip(bits)
                .map(|(c, b)| (u64::from_le_bytes(*c) ^ b).count_ones())
                .sum::<u32>() as f32,
            _ => unreachable!("query prepared for a different quantizer"),
        }
    }
}

/// Eight independent accumulators so the loop vectorises (one sum would force
/// sequential float adds).
fn sq8_l2(qm: &[f32], scale: &[f32], code: &[u8]) -> f32 {
    let mut acc = [0f32; 8];
    let (q8, qr) = qm.as_chunks::<8>();
    let (s8, sr) = scale.as_chunks::<8>();
    let (c8, cr) = code.as_chunks::<8>();
    for ((q, s), c) in q8.iter().zip(s8).zip(c8) {
        for j in 0..8 {
            let d = q[j] - c[j] as f32 * s[j];
            acc[j] += d * d;
        }
    }
    let tail: f32 = qr
        .iter()
        .zip(sr)
        .zip(cr)
        .map(|((q, s), &c)| (q - c as f32 * s).powi(2))
        .sum();
    acc.iter().sum::<f32>() + tail
}

fn sq8_dot(qs: &[f32], code: &[u8]) -> f32 {
    let mut acc = [0f32; 8];
    let (q8, qr) = qs.as_chunks::<8>();
    let (c8, cr) = code.as_chunks::<8>();
    for (q, c) in q8.iter().zip(c8) {
        for j in 0..8 {
            acc[j] += q[j] * c[j] as f32;
        }
    }
    let tail: f32 = qr.iter().zip(cr).map(|(q, &c)| q * c as f32).sum();
    acc.iter().sum::<f32>() + tail
}

fn binarize(v: &[f32], mean: &[f32]) -> Vec<u64> {
    let mut words = vec![0u64; v.len().div_ceil(64)];
    for (i, (x, m)) in v.iter().zip(mean).enumerate() {
        if x > m {
            words[i / 64] |= 1 << (i % 64);
        }
    }
    words
}

fn nearest(v: &[f32], book: &[f32], d: usize) -> usize {
    book.chunks_exact(d)
        .map(|c| l2_sq(v, c))
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}

/// Lloyd's k-means on row-major `points` of dimension `d`. Returns `k * d`
/// centroids. With fewer than `k` distinct points, centroids repeat.
fn kmeans(points: &[f32], d: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = points.len() / d;
    let pick = |i: u64| {
        let mut z = seed.wrapping_mul(0x9e3779b97f4a7c15) ^ i.wrapping_mul(0xbf58476d1ce4e5b9);
        z ^= z >> 31;
        (z % n as u64) as usize
    };
    let mut centroids: Vec<f32> = (0..k as u64)
        .flat_map(|i| points[pick(i) * d..(pick(i) + 1) * d].iter().copied())
        .collect();

    for iter in 0..KMEANS_ITERS {
        let assign: Vec<usize> = points
            .par_chunks_exact(d)
            .map(|p| nearest(p, &centroids, d))
            .collect();
        let mut sums = vec![0f64; k * d];
        let mut counts = vec![0usize; k];
        for (p, &c) in points.chunks_exact(d).zip(&assign) {
            counts[c] += 1;
            for (s, x) in sums[c * d..(c + 1) * d].iter_mut().zip(p) {
                *s += *x as f64;
            }
        }
        for c in 0..k {
            let dst = &mut centroids[c * d..(c + 1) * d];
            if counts[c] == 0 {
                // Empty cluster: restart it on a random point.
                let p = pick((iter * k + c) as u64 + 1_000_003);
                dst.copy_from_slice(&points[p * d..(p + 1) * d]);
            } else {
                for (x, s) in dst.iter_mut().zip(&sums[c * d..(c + 1) * d]) {
                    *x = (*s / counts[c] as f64) as f32;
                }
            }
        }
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim)
            .map(|i| {
                let mut z = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
                z ^= z >> 29;
                (z >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    #[test]
    fn parses_names() {
        assert_eq!(QuantKind::from_name("sq8"), Some(QuantKind::Sq8));
        assert_eq!(QuantKind::from_name("pq16"), Some(QuantKind::Pq { m: 16 }));
        assert_eq!(QuantKind::from_name("binary"), Some(QuantKind::Binary));
        assert_eq!(QuantKind::from_name("pq"), None);
    }

    #[test]
    fn sq8_distances_are_close_to_exact() {
        let dim = 37; // exercises the non-multiple-of-8 tail
        let d = data(500, dim);
        let q = Quantizer::train(QuantKind::Sq8, &d, dim);
        for metric in [Metric::L2, Metric::InnerProduct] {
            let a = &d[..dim];
            let mut code = Vec::new();
            q.encode(&d[dim..2 * dim], &mut code);
            let lut = q.prepare(a, metric);
            let approx = q.distance(&lut, &code, metric);
            let exact = metric.distance(a, &d[dim..2 * dim]);
            assert!((approx - exact).abs() < 0.05, "{metric:?}: {approx} vs {exact}");
        }
    }

    #[test]
    fn pq_reduces_error_versus_random_codes() {
        let dim = 16;
        let d = data(4000, dim);
        let q = Quantizer::train(QuantKind::Pq { m: 4 }, &d, dim);
        let lut = q.prepare(&d[..dim], Metric::L2);
        let mut err = 0.0;
        for v in d.chunks_exact(dim).take(200) {
            let mut code = Vec::new();
            q.encode(v, &mut code);
            assert_eq!(code.len(), 4);
            err += (q.distance(&lut, &code, Metric::L2) - l2_sq(&d[..dim], v)).abs();
        }
        // Mean squared distance between random points here is ~16/6 = 2.7.
        assert!(err / 200.0 < 0.5, "mean abs error {}", err / 200.0);
    }

    #[test]
    fn binary_hamming_is_zero_for_identical_vectors() {
        let dim = 100;
        let d = data(300, dim);
        let q = Quantizer::train(QuantKind::Binary, &d, dim);
        assert_eq!(q.code_size(dim), 16);
        let mut code = Vec::new();
        q.encode(&d[..dim], &mut code);
        let lut = q.prepare(&d[..dim], Metric::L2);
        assert_eq!(q.distance(&lut, &code, Metric::L2), 0.0);
    }
}

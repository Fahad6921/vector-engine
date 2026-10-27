//! Distance functions. Every metric is expressed as "smaller is closer":
//!
//! - `L2`: squared Euclidean distance (the sqrt doesn't change the ordering).
//! - `InnerProduct`: negated dot product.
//! - `Cosine`: `1 - dot`, with vectors normalised on insert and queries normalised on search.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    L2,
    InnerProduct,
    Cosine,
}

impl Metric {
    /// Parses the names used by ann-benchmarks ("euclidean", "angular") plus our own.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim() {
            "l2" | "euclidean" => Some(Metric::L2),
            "ip" | "dot" | "inner_product" => Some(Metric::InnerProduct),
            "cosine" | "angular" => Some(Metric::Cosine),
            _ => None,
        }
    }

    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2_sq(a, b),
            Metric::InnerProduct => -dot(a, b),
            Metric::Cosine => 1.0 - dot(a, b),
        }
    }

    /// Whether vectors must be normalised before they are stored or searched.
    pub fn needs_normalize(self) -> bool {
        self == Metric::Cosine
    }
}

/// Squared Euclidean distance.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        neon::l2_sq(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::available() {
            // SAFETY: CPU support checked above, lengths checked by the assert.
            unsafe { avx2::l2_sq(a, b) }
        } else {
            scalar::l2_sq(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::l2_sq(a, b)
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::available() {
            // SAFETY: CPU support checked above, lengths checked by the assert.
            unsafe { avx2::dot(a, b) }
        } else {
            scalar::dot(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::dot(a, b)
    }
}

/// Scales `v` to unit length. Zero vectors are left as they are.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

pub mod scalar {
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

/// NEON is part of the aarch64 baseline, so no runtime detection is needed.
/// Four independent accumulators of 4 lanes each hide the FMA latency.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline]
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let end = n - n % 16;
        // SAFETY: every load reads 4 floats at an offset < end <= n from slices of length n.
        unsafe {
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut i = 0;
            while i < end {
                for (j, acc) in acc.iter_mut().enumerate() {
                    let d = vsubq_f32(vld1q_f32(pa.add(i + 4 * j)), vld1q_f32(pb.add(i + 4 * j)));
                    *acc = vfmaq_f32(*acc, d, d);
                }
                i += 16;
            }
            let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
            vaddvq_f32(sum) + super::scalar::l2_sq(&a[end..], &b[end..])
        }
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let end = n - n % 16;
        // SAFETY: as above.
        unsafe {
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut i = 0;
            while i < end {
                for (j, acc) in acc.iter_mut().enumerate() {
                    *acc = vfmaq_f32(*acc, vld1q_f32(pa.add(i + 4 * j)), vld1q_f32(pb.add(i + 4 * j)));
                }
                i += 16;
            }
            let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
            vaddvq_f32(sum) + super::scalar::dot(&a[end..], &b[end..])
        }
    }
}

/// AVX2 + FMA, 4 accumulators of 8 lanes (32 floats per iteration).
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    #[inline]
    pub fn available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// # Safety
    /// CPU must support AVX2 and FMA; `a` and `b` must have equal length.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let end = n - n % 32;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < end {
            for (j, acc) in acc.iter_mut().enumerate() {
                let d = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 8 * j)),
                    _mm256_loadu_ps(pb.add(i + 8 * j)),
                );
                *acc = _mm256_fmadd_ps(d, d, *acc);
            }
            i += 32;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        hsum(sum) + super::scalar::l2_sq(&a[end..], &b[end..])
    }

    /// # Safety
    /// CPU must support AVX2 and FMA; `a` and `b` must have equal length.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let end = n - n % 32;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < end {
            for (j, acc) in acc.iter_mut().enumerate() {
                *acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 8 * j)),
                    _mm256_loadu_ps(pb.add(i + 8 * j)),
                    *acc,
                );
            }
            i += 32;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        hsum(sum) + super::scalar::dot(&a[end..], &b[end..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Summation-order rounding error scales with the magnitude of the terms, not
    /// of the result: a dot product whose terms cancel to ~0 can still be off by
    /// eps * sum(|a_i * b_i|). So the tolerance is relative to that sum.
    fn close(simd: f32, reference: f32, magnitude: f32) -> bool {
        (simd - reference).abs() <= 1e-5 * magnitude.max(1.0)
    }

    fn pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
        (0usize..300).prop_flat_map(|n| {
            (
                prop::collection::vec(-10.0f32..10.0, n),
                prop::collection::vec(-10.0f32..10.0, n),
            )
        })
    }

    proptest! {
        #[test]
        fn l2_matches_scalar((a, b) in pair()) {
            let expected = scalar::l2_sq(&a, &b);
            prop_assert!(close(l2_sq(&a, &b), expected, expected));
        }

        #[test]
        fn dot_matches_scalar((a, b) in pair()) {
            let magnitude: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
            prop_assert!(close(dot(&a, &b), scalar::dot(&a, &b), magnitude));
        }
    }

    #[test]
    fn metric_ordering() {
        let q = [1.0, 0.0];
        let near = [0.9, 0.1];
        let far = [-1.0, 0.0];
        for m in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            assert!(m.distance(&q, &near) < m.distance(&q, &far), "{m:?}");
        }
    }

    #[test]
    fn normalize_gives_unit_length() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-6);

        let mut zero = vec![0.0; 4];
        normalize(&mut zero);
        assert_eq!(zero, vec![0.0; 4]);
    }

    #[test]
    #[should_panic]
    fn length_mismatch_panics() {
        l2_sq(&[1.0, 2.0], &[1.0]);
    }
}

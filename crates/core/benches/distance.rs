use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use vecengine::distance::{dot, l2_sq, scalar};

fn random_vec(n: usize, seed: u64) -> Vec<f32> {
    // xorshift, good enough for benchmark inputs
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    for (name, simd, plain) in [
        (
            "l2_sq",
            l2_sq as fn(&[f32], &[f32]) -> f32,
            scalar::l2_sq as fn(&[f32], &[f32]) -> f32,
        ),
        ("dot", dot, scalar::dot),
    ] {
        let mut group = c.benchmark_group(name);
        for dim in [96, 128, 384, 768, 1536] {
            let a = random_vec(dim, 1);
            let b = random_vec(dim, 2);
            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bench, _| {
                bench.iter(|| simd(black_box(&a), black_box(&b)))
            });
            group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
                bench.iter(|| plain(black_box(&a), black_box(&b)))
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);

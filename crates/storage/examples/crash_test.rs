//! Kill-and-recover test for collections.
//!
//!     cargo run --release -p vecengine-storage --example crash_test -- 1000
//!
//! A child process applies a deterministic stream of operations (upserts, deletes,
//! multi-record batches, flushes, compactions) to a collection and prints each
//! operation's number once it returns. The parent SIGKILLs it at a random moment,
//! reopens the collection and checks that it holds exactly the result of every
//! acknowledged operation, plus possibly the one in flight, and nothing else.
//! Then it starts a new child from where the state left off.
//!
//! This tests crash consistency of the process, not power loss: the WAL fsync is
//! off to keep iterations fast, and the page cache survives a killed process.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use vecengine::Metric;
use vecengine_storage::{Collection, Config, Record};

const DIM: usize = 8;
const IDS: u64 = 3000;

fn config() -> Config {
    Config {
        flush_threshold: 400,
        sync: false,
        ..Config::new(DIM, Metric::L2)
    }
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn vector(seed: u64) -> Vec<f32> {
    (0..DIM as u64)
        .map(|j| (mix(seed * 31 + j) >> 40) as f32 / (1u64 << 24) as f32)
        .collect()
}

enum Op {
    Write(Vec<Record>),
    Flush,
    Compact,
}

/// Operation `i` of the stream. Pure function of `i`, so parent and child agree.
fn op(i: u64) -> Op {
    let h = mix(i);
    let record = |salt: u64| {
        let h = mix(h ^ salt);
        let id = h % IDS;
        if h % 10 < 7 {
            Record::Upsert {
                id,
                vector: vector(i * 8 + salt),
                payload: (i * 8 + salt).to_le_bytes().to_vec(),
            }
        } else {
            Record::Delete { id }
        }
    };
    match h % 1000 {
        0..=1 => Op::Compact,
        2..=11 => Op::Flush,
        12..=111 => Op::Write((0..5).map(record).collect()),
        _ => Op::Write(vec![record(0)]),
    }
}

type State = HashMap<u64, (Vec<f32>, Vec<u8>)>;

fn apply(state: &mut State, op: &Op) {
    if let Op::Write(records) = op {
        for r in records {
            match r {
                Record::Upsert { id, vector, payload } => {
                    state.insert(*id, (vector.clone(), payload.clone()));
                }
                Record::Delete { id } => {
                    state.remove(id);
                }
            }
        }
    }
}

fn matches(c: &Collection, state: &State) -> Result<(), String> {
    if c.len() != state.len() {
        return Err(format!(
            "collection has {} points, expected {}",
            c.len(),
            state.len()
        ));
    }
    for (id, (v, p)) in state {
        match c.get(*id) {
            Some((cv, cp)) if cv == v.as_slice() && cp == p.as_slice() => {}
            Some(_) => return Err(format!("id {id} has the wrong value")),
            None => return Err(format!("id {id} is missing")),
        }
    }
    Ok(())
}

fn child(dir: &Path, start: u64) {
    let mut c = Collection::open(dir, config()).expect("child failed to open");
    // Runs until the parent kills it.
    let mut i = start;
    loop {
        match op(i) {
            Op::Write(records) => c.write(records),
            Op::Flush => c.flush(),
            Op::Compact => c.compact(),
        }
        .expect("operation failed");
        println!("{i}");
        i += 1;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        child(Path::new(&args[2]), args[3].parse().unwrap());
        return;
    }
    let iterations: u64 = args.get(1).map_or(200, |s| s.parse().expect("iterations"));

    let dir: PathBuf = std::env::temp_dir().join(format!("vecengine-crash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let exe = std::env::current_exe().unwrap();

    let mut state = State::new();
    let mut next = 0u64;
    let mut in_flight_applied = 0;
    let started = Instant::now();

    for iter in 0..iterations {
        let mut child = Command::new(&exe)
            .args(["--child", dir.to_str().unwrap(), &next.to_string()])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_micros(2_000 + mix(iter) % 60_000));
        child.kill().unwrap(); // SIGKILL
        child.wait().unwrap();
        let mut out = String::new();
        child.stdout.take().unwrap().read_to_string(&mut out).unwrap();

        // Each complete line is one acknowledged op; Rust's stdout is line-buffered,
        // so a line is written as soon as the op returns.
        let acked = out.matches('\n').count() as u64;
        for i in next..next + acked {
            apply(&mut state, &op(i));
        }
        next += acked;

        let c = Collection::open(&dir, config()).expect("recovery failed to open");
        if matches(&c, &state).is_ok() {
            // The in-flight op didn't land (or didn't change anything).
        } else {
            let mut with_next = state.clone();
            let pending = op(next);
            apply(&mut with_next, &pending);
            if let Err(e) = matches(&c, &with_next) {
                eprintln!(
                    "iteration {iter}: state after recovery matches neither {next} nor {} ops: {e}",
                    next + 1
                );
                std::process::exit(1);
            }
            state = with_next;
            next += 1;
            in_flight_applied += 1;
        }

        // Exact stored vectors should come back as their own nearest neighbour.
        for (id, (v, _)) in state.iter().take(5) {
            let hits = c.search(v, 5, 100);
            if !hits.iter().any(|h| h.id == *id) {
                eprintln!("iteration {iter}: id {id} not found by searching for its own vector");
                std::process::exit(1);
            }
        }
        if iter % 100 == 99 {
            let s = c.stats();
            println!(
                "{:>5} iterations, {next} ops, {} live, {} segments, {} deleted",
                iter + 1,
                s.live,
                s.segments,
                s.deleted
            );
        }
    }

    println!(
        "ok: {iterations} kills, {next} ops, in-flight op had landed in {in_flight_applied} of them, {:.0}s",
        started.elapsed().as_secs_f64()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
